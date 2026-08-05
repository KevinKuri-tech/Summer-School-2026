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

DEFAULT_MODEL = os.environ.get("STUDYPLAN_MODEL", "claude-sonnet-5")
# Cheaper: claude-haiku-4-5. Stronger: claude-opus-5.
# Structured outputs are supported on claude-opus-5, claude-sonnet-5,
# claude-opus-4-8, claude-fable-5 and claude-haiku-4-5. claude-sonnet-4-6 is
# not on that list, so it is not a safe default here.


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
            resp = self.client.messages.create(
                model=body["model"],
                max_tokens=body["max_tokens"],
                system=body["system"],
                messages=body["messages"],
                # extra_body keeps this working across SDK versions that predate the
                # typed output_config parameter.
                extra_body={"output_config": body["output_config"]},
            )
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
DEFAULT_OPENROUTER_MODEL = os.environ.get(
    "STUDYPLAN_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")

# Convenience presets for the sidebar. Any OpenRouter slug works; this list is
# only a shortcut. Schema support is per endpoint and changes over time, so
# verify with:
#   https://openrouter.ai/api/v1/models?supported_parameters=structured_outputs
#
# These are the free slugs that reported structured_outputs support at the time
# of writing. A free model NOT on that list still runs, but only via the JSON
# mode fallback below, where the schema is a hint rather than a constraint.
OPENROUTER_PRESETS = [
    "~deepseek/deepseek-v4-flash-latest",
    "openai/gpt-oss-20b:free",
    "google/gemma-4-26b-a4b-it:free",
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
                 max_tokens: int = 8000, timeout: float = 180.0,
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
                blocks.append({
                    "id": f"b{counter:03d}",
                    "date": day.isoformat(),
                    "start_time": _fmt(cursor),
                    "duration_minutes": dur,
                    "module": mod.name,
                    "topic": f"{mod.name}: {btype} session",
                    "block_type": btype,
                    "priority": min(5, max(1, round(_urgency(mod, day) / 4))),
                    "rationale": (f"{(mod.exam_date - day).days} days to exam, "
                                  f"difficulty {mod.difficulty}, confidence {mod.confidence}."),
                })
                remaining[mod.name] = max(0.0, remaining[mod.name] - dur)
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


def active_backend() -> str:
    """Which backend the app would use right now: anthropic | openrouter | mock."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENROUTER_API_KEY"):
        return "openrouter"
    return "mock"


def get_planner(force_mock: bool = False, model: str | None = None):
    backend = "mock" if force_mock else active_backend()
    if backend == "anthropic":
        return AnthropicPlanner(model=model or DEFAULT_MODEL)
    if backend == "openrouter":
        return OpenRouterPlanner(model=model or DEFAULT_OPENROUTER_MODEL)
    return MockPlanner()
