"""The single AI call, plus a deterministic fallback planner.

The Anthropic planner uses JSON structured outputs (output_config.format), so the
model is grammar-constrained to PLAN_JSON_SCHEMA and cannot return prose, markdown
fences or half-valid JSON. Docs:
https://platform.claude.com/docs/en/build-with-claude/structured-outputs

The mock planner is a greedy scheduler. It is used when no API key is set and as
the baseline the eval harness compares the model against.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from studyplan import prompts
from studyplan.schema import PLAN_JSON_SCHEMA, Availability, PlanRequest, StudyPlan

# One variable per backend. A single shared STUDYPLAN_MODEL cannot serve both:
# with two keys in .env the OpenRouter slug wins and is then sent to Anthropic,
# which 404s. STUDYPLAN_MODEL is still honoured for OpenRouter, where it was
# already being used, so existing .env files keep working.
DEFAULT_MODEL = os.environ.get("STUDYPLAN_ANTHROPIC_MODEL", "claude-sonnet-5")
# Cheaper: claude-haiku-4-5. Stronger: claude-opus-5.
#
# Offline fallback for the model picker. Prefer list_anthropic_models(), which
# asks the API what this particular key may call instead of guessing; this list
# is only what to show when that call cannot be made.
ANTHROPIC_PRESETS = [
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-haiku-4-5",
]


@dataclass
class PlanResult:
    plan: StudyPlan
    source: str  # "model" | "mock"
    latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    repaired: bool = False
    raw: str = ""
    notes: list[str] = field(default_factory=list)


class PlannerError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# Shared call orchestration
# --------------------------------------------------------------------------
class _BasePlanner:
    """generate / replan / repair are backend-agnostic.

    Subclasses implement _payload(messages) -> the exact body that goes on the
    wire, and _call(messages) -> (StudyPlan, meta dict).
    """

    #: Exact payload handed to the transport on the most recent _call().
    last_request: dict | None = None

    def _payload(self, messages: list[dict]) -> dict:  # pragma: no cover
        raise NotImplementedError

    def _call(self, messages: list[dict]) -> tuple[StudyPlan, dict]:  # pragma: no cover
        raise NotImplementedError

    # The message builders are shared by the live calls and the previews below,
    # so what a UI shows cannot drift from what goes over the wire.
    @staticmethod
    def _generate_messages(req: PlanRequest) -> list[dict]:
        return [{"role": "user", "content": prompts.generate_prompt(req)}]

    @staticmethod
    def _replan_messages(req: PlanRequest, previous: StudyPlan, today: date) -> list[dict]:
        return [{"role": "user", "content": prompts.replan_prompt(
            req, previous.model_dump_json(indent=None), today.isoformat())}]

    def preview_generate(self, req: PlanRequest) -> dict:
        """The exact payload generate() is about to send."""
        return self._payload(self._generate_messages(req))

    def preview_replan(self, req: PlanRequest, previous: StudyPlan, today: date) -> dict:
        """The exact payload replan() is about to send."""
        return self._payload(self._replan_messages(req, previous, today))

    @staticmethod
    def _result(plan: StudyPlan, meta: dict, repaired: bool = False) -> PlanResult:
        return PlanResult(plan=plan, source="model", latency_s=meta["latency_s"],
                          input_tokens=meta["input_tokens"],
                          output_tokens=meta["output_tokens"], raw=meta["raw"],
                          repaired=repaired)

    def generate(self, req: PlanRequest) -> PlanResult:
        plan, meta = self._call(self._generate_messages(req))
        res = self._result(plan, meta)
        if getattr(self, "used_fallback", False):
            res.notes.append("Endpoint could not enforce the schema. Ran in JSON mode, "
                             "so the schema was a hint rather than a constraint.")
        return res

    def replan(self, req: PlanRequest, previous: StudyPlan, today: date) -> PlanResult:
        plan, meta = self._call(self._replan_messages(req, previous, today))
        return self._result(plan, meta)

    def repair(self, req: PlanRequest, bad: StudyPlan, violations: list[str]) -> PlanResult:
        """One corrective round trip. Cheaper and more honest than silent patching."""
        plan, meta = self._call([
            {"role": "user", "content": prompts.generate_prompt(req)},
            {"role": "assistant", "content": bad.model_dump_json()},
            {"role": "user", "content": prompts.repair_prompt(violations)},
        ])
        return self._result(plan, meta, repaired=True)


def _clean_json(text: str) -> str:
    """Strip markdown fences and any prose around the JSON object.

    Not needed on Anthropic (decoding is grammar-constrained) but necessary on
    OpenRouter, where a provider may treat the schema as a hint rather than a
    constraint.
    """
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```")[1] if "```" in t[3:] else t.lstrip("`")
        if t.lstrip().lower().startswith("json"):
            t = t.lstrip()[4:]
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end == -1:
        raise PlannerError(f"No JSON object in response: {text[:200]}")
    return t[start:end + 1]


# --------------------------------------------------------------------------
# Anthropic planner
# --------------------------------------------------------------------------
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODELS_URL = "https://api.anthropic.com/v1/models"


@dataclass(frozen=True)
class ModelInfo:
    """One selectable model, as reported by the provider."""

    id: str
    display_name: str = ""
    max_input_tokens: int | None = None
    max_tokens: int | None = None
    structured_outputs: bool = True

    @property
    def label(self) -> str:
        return f"{self.display_name or self.id}  ·  {self.id}"


def _capability(capabilities, name: str) -> bool | None:
    """Whether `name` is supported, or None if the field is absent.

    Current SDKs return a typed ModelCapabilities object; older ones return a
    plain dict. Read both rather than pinning a version.
    """
    node = getattr(capabilities, name, None)
    if node is None and isinstance(capabilities, dict):
        node = capabilities.get(name)
    if node is None:
        return None
    supported = getattr(node, "supported", None)
    if supported is None and isinstance(node, dict):
        supported = node.get("supported")
    return supported


def list_anthropic_models(api_key: str | None = None,
                          structured_only: bool = True) -> list[ModelInfo]:
    """The models this ANTHROPIC_API_KEY is actually entitled to call.

    Reads GET /v1/models, so the result reflects the account rather than a
    hardcoded guess that drifts with every release.

    AnthropicPlanner pins the response to PLAN_JSON_SCHEMA through
    output_config, which a model without structured-output support cannot
    honour, so those are filtered out by default.

    Raises PlannerError when the key is missing or the call fails; callers that
    need a list no matter what can fall back to ANTHROPIC_PRESETS.
    """
    import anthropic

    key = (api_key or os.environ.get("ANTHROPIC_API_KEY") or "").strip().strip('"').strip("'")
    if not key:
        raise PlannerError("ANTHROPIC_API_KEY is not set.")

    client = anthropic.Anthropic(api_key=key)
    try:
        # Iterating the pager (rather than .data) follows every page.
        raw = list(client.models.list())
    except anthropic.AuthenticationError as exc:
        raise PlannerError(f"API key rejected by Anthropic: {exc}") from None
    except anthropic.APIError as exc:
        raise PlannerError(f"Could not list Anthropic models: {exc}") from None

    models = []
    for m in raw:
        supported = _capability(getattr(m, "capabilities", None), "structured_outputs")
        # None means this API version does not report the capability at all.
        # Excluding on that would empty the list, so only a hard False filters.
        if structured_only and supported is False:
            continue
        models.append(ModelInfo(
            id=m.id,
            display_name=getattr(m, "display_name", "") or "",
            max_input_tokens=getattr(m, "max_input_tokens", None),
            max_tokens=getattr(m, "max_tokens", None),
            structured_outputs=bool(supported),
        ))
    return models


class AnthropicPlanner(_BasePlanner):
    endpoint = ANTHROPIC_URL

    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL,
                 max_tokens: int = 15000):
        from anthropic import Anthropic  # imported lazily so the mock path stays dependency-free

        key = (api_key or os.environ.get("ANTHROPIC_API_KEY") or "").strip().strip('"').strip("'")
        if not key:
            raise PlannerError("ANTHROPIC_API_KEY is not set.")
        if not key.startswith("sk-ant-"):
            raise PlannerError(
                "That key is not an Anthropic key (expected sk-ant-...). "
                "An sk-or-v1-... key belongs to OpenRouter: set OPENROUTER_API_KEY instead."
            )
        self.client = Anthropic(api_key=key)
        self.model = model
        self.max_tokens = max_tokens

    def _payload(self, messages: list[dict]) -> dict:
        """The JSON body of the request, exactly as the SDK will send it."""
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": prompts.SYSTEM,
            "messages": messages,
            "output_config": {
                "format": {"type": "json_schema", "schema": PLAN_JSON_SCHEMA}
            },
        }

    def _call(self, messages: list[dict]) -> tuple[StudyPlan, dict]:
        import anthropic

        body = self._payload(messages)
        self.last_request = body
        t0 = time.perf_counter()
        try:
            # Streamed, not because the tokens are rendered as they arrive, but
            # because the SDK refuses a non-streaming request whose max_tokens
            # could plausibly outlast a 10 minute HTTP connection. get_final_message()
            # blocks until the whole reply is in, so the caller sees the same
            # object messages.create() would have returned.
            with self.client.messages.stream(
                model=body["model"],
                max_tokens=body["max_tokens"],
                system=body["system"],
                messages=body["messages"],
                # extra_body keeps this working across SDK versions that predate the
                # typed output_config parameter.
                extra_body={"output_config": body["output_config"]},
            ) as stream:
                resp = stream.get_final_message()
        except anthropic.AuthenticationError as exc:
            raise PlannerError(f"API key rejected by Anthropic: {exc}") from None
        except anthropic.APIStatusError as exc:
            raise PlannerError(f"Anthropic API error {exc.status_code}: {exc}") from None
        latency = time.perf_counter() - t0

        if getattr(resp, "stop_reason", None) == "refusal":
            raise PlannerError("Model refused the request.")
        if getattr(resp, "stop_reason", None) == "max_tokens":
            raise PlannerError("Output truncated. Increase max_tokens or shorten the horizon.")

        text = next((b.text for b in resp.content if getattr(b, "type", "") == "text"), None)
        if text is None:
            raise PlannerError("No text block in response.")

        meta = {
            "latency_s": latency,
            "input_tokens": getattr(resp.usage, "input_tokens", 0),
            "output_tokens": getattr(resp.usage, "output_tokens", 0),
            "raw": text,
        }
        return StudyPlan.model_validate_json(text), meta


# --------------------------------------------------------------------------
# OpenRouter planner
# --------------------------------------------------------------------------
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_OPENROUTER_MODEL = (os.environ.get("STUDYPLAN_OPENROUTER_MODEL")
                            or os.environ.get("STUDYPLAN_MODEL")
                            or "~anthropic/claude-haiku-latest")

# Convenience presets for the sidebar. Any OpenRouter slug works; this list is
# only a shortcut. Schema support is per endpoint and changes over time, so
# verify with:
#   https://openrouter.ai/api/v1/models?supported_parameters=structured_outputs
#
# These are the free slugs that reported structured_outputs support at the time
# of writing. A free model NOT on that list still runs, but only via the JSON
# mode fallback below, where the schema is a hint rather than a constraint.
OPENROUTER_PRESETS = [
    "~anthropic/claude-haiku-latest",
    "~deepseek/deepseek-v4-flash-latest",
    "openai/gpt-oss-20b:free",
    "google/gemma-4-26b-a4b-it:free",
    "anthropic/claude-sonnet-4.6",
    "anthropic/claude-sonnet-5",
]

_SCHEMA_UNSUPPORTED_HINTS = (
    "no endpoints found", "no endpoints available", "no allowed providers",
    "response_format", "json_schema", "structured output", "require_parameters",
)

# A 404 can also mean the account blocks every endpoint that serves this model,
# which has nothing to do with the schema. Retrying without the schema cannot
# help, so this is checked first and reported as its own failure.
_DATA_POLICY_HINTS = ("data policy", "guardrail", "privacy")


def _data_policy_blocked(body: str) -> bool:
    return any(h in body.lower() for h in _DATA_POLICY_HINTS)


def _schema_unsupported(body: str) -> bool:
    low = body.lower()
    if _data_policy_blocked(low):
        return False
    return any(h in low for h in _SCHEMA_UNSUPPORTED_HINTS)


class OpenRouterPlanner(_BasePlanner):
    """OpenAI-compatible endpoint with response_format = json_schema.

    Schema enforcement is per endpoint, not per model, so provider.require_parameters
    is set to keep OpenRouter from routing to an endpoint that would ignore the
    schema. Responses are still cleaned and validated, because some providers treat
    the schema as a strong hint rather than a hard constraint.
    """

    endpoint = OPENROUTER_URL

    def __init__(self, api_key: str | None = None, model: str = DEFAULT_OPENROUTER_MODEL,
                 max_tokens: int = 30000, timeout: float = 180.0,
                 allow_fallback: bool = True):
        import httpx  # ships with the anthropic SDK

        key = (api_key or os.environ.get("OPENROUTER_API_KEY") or "").strip().strip('"').strip("'")
        if not key:
            raise PlannerError("OPENROUTER_API_KEY is not set.")
        self.key = key
        self.model = model
        self.max_tokens = max_tokens
        self.allow_fallback = allow_fallback
        self.used_fallback = False
        self._http = httpx.Client(timeout=timeout)

    def _body(self, messages: list[dict], strict: bool) -> dict:
        if strict:
            system = prompts.SYSTEM
            extra = {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "study_plan", "strict": True,
                                    "schema": PLAN_JSON_SCHEMA},
                },
                # Only route to endpoints that actually honour the schema.
                "provider": {"require_parameters": True},
            }
        else:
            # Endpoint cannot enforce a schema: demote it to an instruction and
            # ask for plain JSON mode, which most endpoints do support.
            system = prompts.SYSTEM + "\n\n" + prompts.schema_hint(PLAN_JSON_SCHEMA)
            extra = {"response_format": {"type": "json_object"}}
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "system", "content": system}] + messages,
            **extra,
        }

    def _payload(self, messages: list[dict]) -> dict:
        # A previous call may already have proved the endpoint cannot enforce the
        # schema, in which case the next one starts in fallback mode too.
        return self._body(messages, strict=not self.used_fallback)

    def _post(self, body: dict):
        return self._http.post(
            OPENROUTER_URL,
            headers={"Authorization": f"Bearer {self.key}",
                     "Content-Type": "application/json",
                     "X-Title": "Study-Plan Optimizer"},
            json=body,
        )

    def _call(self, messages: list[dict]) -> tuple[StudyPlan, dict]:
        body = self._body(messages, strict=True)
        self.last_request = body
        t0 = time.perf_counter()
        resp = self._post(body)

        # Account-level block, not a schema problem. Fail fast: dropping the
        # schema would only spend a second request on the same 404.
        if resp.status_code == 404 and _data_policy_blocked(resp.text):
            raise PlannerError(
                f"OpenRouter has no endpoint it is allowed to use for '{self.model}'.\n"
                "Free (':free') models require permitting providers that may train on "
                "your prompts. Enable that at https://openrouter.ai/settings/privacy, "
                "or switch to the paid slug by dropping the ':free' suffix.\n"
                f"OpenRouter said: {resp.text[:200]}"
            )

        # No endpoint supports json_schema for this model. Retry once without it.
        if resp.status_code in (400, 404) and self.allow_fallback and \
                _schema_unsupported(resp.text):
            self.used_fallback = True
            body = self._body(messages, strict=False)
            self.last_request = body
            resp = self._post(body)
        latency = time.perf_counter() - t0

        if resp.status_code == 401:
            raise PlannerError("OpenRouter rejected the key (401). Check OPENROUTER_API_KEY.")
        if resp.status_code == 402:
            raise PlannerError("OpenRouter: insufficient credits for this model.")
        if resp.status_code == 429:
            raise PlannerError("OpenRouter rate limit hit. Free models are throttled per minute "
                               "and per day. Wait, or switch to the paid slug.")
        if resp.status_code != 200:
            raise PlannerError(f"OpenRouter error {resp.status_code}: {resp.text[:300]}")

        data = resp.json()
        if "error" in data:
            raise PlannerError(f"OpenRouter error: {str(data['error'])[:300]}")
        choice = data["choices"][0]
        if choice.get("finish_reason") == "length":
            raise PlannerError("Output truncated. Increase max_tokens or shorten the horizon.")
        text = choice["message"].get("content") or ""
        usage = data.get("usage", {}) or {}

        meta = {
            "latency_s": latency,
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "raw": text,
        }
        return StudyPlan.model_validate_json(_clean_json(text)), meta


# --------------------------------------------------------------------------
# Deterministic planner (fallback + baseline)
# --------------------------------------------------------------------------
def _urgency(mod, day: date) -> float:
    days_left = max((mod.exam_date - day).days, 0)
    proximity = 1.0 / (1 + days_left)
    return mod.difficulty * (6 - mod.confidence) * (0.4 + proximity)


def _fmt(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _pick_chapter(mod, left: dict[str, int], btype: str) -> str | None:
    """Which chapter this block should cover, or None if the module has none.

    Revision goes to the shakiest chapter, since that is where re-reading pays
    off most; everything else goes to whichever chapter still has the largest
    unscheduled budget, which spreads the module over its whole syllabus
    instead of finishing chapter one before starting chapter two.
    """
    if not left:
        return None
    confidence = {c.name: (c.confidence or mod.confidence) for c in mod.chapters}
    if btype == "revision":
        return min(left, key=lambda n: (confidence.get(n, mod.confidence), -left[n], n))
    unfinished = {n: v for n, v in left.items() if v > 0} or left
    return max(unfinished, key=lambda n: (unfinished[n], -confidence.get(n, mod.confidence)))


class MockPlanner:
    """Greedy urgency-weighted scheduler. Never calls the network."""

    # Mirrors the model planners so callers can introspect any backend the same
    # way. No model is contacted, so both are None and there is no payload.
    model = None
    endpoint = None
    last_request = None

    def preview_generate(self, req: PlanRequest) -> None:
        return None

    def preview_replan(self, req: PlanRequest, previous: StudyPlan, today: date) -> None:
        return None

    def generate(self, req: PlanRequest, start: date | None = None,
                 carry: dict[str, float] | None = None) -> PlanResult:
        av: Availability = req.availability
        start = start or req.start_date
        days = [d for d in req.days() if d >= start]
        remaining = {m.name: m.estimated_hours * 60 for m in req.modules}
        if carry:
            for name, extra in carry.items():
                remaining[name] = remaining.get(name, 0) + extra
        # Per-chapter budgets, drawn down alongside the module budget so heavier
        # and shakier chapters end up with more scheduled minutes. Empty for
        # modules without chapters, which then keep the generic topic text.
        chap_left = {m.name: m.chapter_minutes() for m in req.modules}
        blocks: list[dict] = []
        counter = 1

        # reserve the last available day before each exam for revision
        revision_day: dict[str, date] = {}
        for m in req.modules:
            candidates = [d for d in days if d < m.exam_date and av.minutes_for(d) > 0]
            if candidates:
                revision_day[m.name] = candidates[-1]

        for day in days:
            capacity = av.minutes_for(day)
            if capacity <= 0:
                continue
            cursor = int(av.day_start.split(":")[0]) * 60 + int(av.day_start.split(":")[1])
            used = 0
            forced = [m for m in req.modules if revision_day.get(m.name) == day]
            while used + av.min_session_minutes <= capacity:
                pool = [m for m in req.modules
                        if m.exam_date > day and (remaining[m.name] > 0 or m in forced)]
                if not pool:
                    break
                if forced:
                    mod = forced.pop(0)
                    btype = "revision"
                else:
                    mod = max(pool, key=lambda m: _urgency(m, day))
                    days_left = (mod.exam_date - day).days
                    btype = "revision" if days_left <= 3 else (
                        "practice" if counter % 3 == 0 else "learn")
                dur = int(min(av.max_session_minutes, capacity - used,
                              max(av.min_session_minutes, remaining[mod.name] or av.min_session_minutes)))
                dur = max(dur, av.min_session_minutes)
                if used + dur > capacity:
                    break
                chapter = _pick_chapter(mod, chap_left[mod.name], btype)
                blocks.append({
                    "id": f"b{counter:03d}",
                    "date": day.isoformat(),
                    "start_time": _fmt(cursor),
                    "duration_minutes": dur,
                    "module": mod.name,
                    "topic": chapter or f"{mod.name}: {btype} session",
                    "block_type": btype,
                    "priority": min(5, max(1, round(_urgency(mod, day) / 4))),
                    "rationale": (f"{(mod.exam_date - day).days} days to exam, "
                                  f"difficulty {mod.difficulty}, confidence {mod.confidence}."),
                })
                remaining[mod.name] = max(0.0, remaining[mod.name] - dur)
                if chapter is not None:
                    chap_left[mod.name][chapter] = max(0, chap_left[mod.name][chapter] - dur)
                used += dur
                cursor += dur + 15
                counter += 1

        plan = StudyPlan(
            plan_start=start,
            plan_end=req.end_date,
            strategy=("Greedy baseline: time is allocated by difficulty, low confidence and exam "
                      "proximity, with the last free day before each exam reserved for revision."),
            blocks=[b for b in blocks],  # type: ignore[arg-type]
            risks=[f"{n}: {int(v)} min of estimated workload did not fit."
                   for n, v in remaining.items() if v > 0],
            change_log=[],
        )
        return PlanResult(plan=plan, source="mock")

    def replan(self, req: PlanRequest, previous: StudyPlan, today: date) -> PlanResult:
        missed = [b for b in previous.blocks if b.id in set(req.missed_block_ids)]
        carry: dict[str, float] = {}
        for b in missed:
            carry[b.module] = carry.get(b.module, 0) + b.duration_minutes
        res = self.generate(req, start=today, carry=carry)
        res.plan.change_log = [
            f"Rescheduled {len(missed)} missed block(s), {int(sum(carry.values()))} minutes carried forward.",
            f"Plan rebuilt from {today.isoformat()}; earlier blocks left untouched.",
        ]
        return res

    def repair(self, req: PlanRequest, bad: StudyPlan, violations: list[str]) -> PlanResult:
        return self.generate(req)


def is_free_model(model: str | None, backend: str) -> bool:
    """Whether a call on this backend/model can be billed.

    OpenRouter marks its no-cost endpoints with a ':free' suffix on the slug;
    anything else draws on the account's credits. Anthropic has no free tier, so
    every model there is paid. The baseline planner never calls out at all.
    """
    if backend == "mock" or not model:
        return True
    if backend == "openrouter":
        return model.strip().endswith(":free")
    return False


def available_backends() -> list[str]:
    """Every backend this environment can reach, preferred first.

    Both provider keys can be set at once; the UI picks between them, so this
    reports all of them rather than collapsing to one. OpenRouter comes first,
    so it is what the UI preselects and what active_backend() returns when both
    keys are present. The baseline is always last and always available.
    """
    backends = []
    if os.environ.get("OPENROUTER_API_KEY"):
        backends.append("openrouter")
    if os.environ.get("ANTHROPIC_API_KEY"):
        backends.append("anthropic")
    backends.append("mock")
    return backends


def active_backend() -> str:
    """The backend used when the caller expresses no preference."""
    return available_backends()[0]


def default_model_for(backend: str) -> str | None:
    """The model a backend falls back to. None for the baseline, which has none."""
    return {"anthropic": DEFAULT_MODEL,
            "openrouter": DEFAULT_OPENROUTER_MODEL}.get(backend)


def get_planner(force_mock: bool = False, model: str | None = None,
                backend: str | None = None):
    """`backend` is the explicit UI choice; None keeps the env-order default."""
    backend = "mock" if force_mock else (backend or active_backend())
    if backend == "anthropic":
        return AnthropicPlanner(model=model or DEFAULT_MODEL)
    if backend == "openrouter":
        return OpenRouterPlanner(model=model or DEFAULT_OPENROUTER_MODEL)
    return MockPlanner()
