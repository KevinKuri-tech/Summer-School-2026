"""OpenRouter variant of the planner.

Same contract as AnthropicPlanner, but goes through OpenRouter's OpenAI-compatible
endpoint and uses response_format json_schema instead of Anthropic's output_config.

    uv add openai
    # .env
    OPENROUTER_API_KEY=sk-or-v1-...
    STUDYPLAN_MODEL=anthropic/claude-sonnet-4.6

Not every model on OpenRouter supports strict json_schema. Free tiers frequently
do not. If the model ignores the schema you will see a validation error instead
of a silently malformed plan, which is the intended failure mode.
"""

from __future__ import annotations

import os
import time
from datetime import date

from . import prompts
from .planner import PlanResult, PlannerError
from .schema import PLAN_JSON_SCHEMA, PlanRequest, StudyPlan

BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = os.environ.get("STUDYPLAN_MODEL", "anthropic/claude-sonnet-4.6")


class OpenRouterPlanner:
    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL,
                 max_tokens: int = 8000):
        try:
            from openai import OpenAI
        except ImportError as exc:  # noqa: BLE001
            raise PlannerError("Install the openai package: uv add openai") from exc

        key = (api_key or os.environ.get("OPENROUTER_API_KEY") or "").strip().strip('"').strip("'")
        if not key:
            raise PlannerError("OPENROUTER_API_KEY is not set.")
        self.client = OpenAI(base_url=BASE_URL, api_key=key)
        self.model = model
        self.max_tokens = max_tokens

    def _call(self, messages: list[dict]) -> tuple[StudyPlan, dict]:
        t0 = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "system", "content": prompts.SYSTEM}, *messages],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "study_plan",
                    "strict": True,
                    "schema": PLAN_JSON_SCHEMA,
                },
            },
        )
        latency = time.perf_counter() - t0
        choice = resp.choices[0]
        if choice.finish_reason == "length":
            raise PlannerError("Output truncated. Raise max_tokens or shorten the horizon.")
        text = choice.message.content
        if not text:
            raise PlannerError("Empty response from the model.")

        usage = getattr(resp, "usage", None)
        meta = {
            "latency_s": latency,
            "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
            "raw": text,
        }
        return StudyPlan.model_validate_json(text), meta

    def _result(self, plan: StudyPlan, meta: dict, repaired: bool = False) -> PlanResult:
        return PlanResult(plan=plan, source="model", latency_s=meta["latency_s"],
                          input_tokens=meta["input_tokens"],
                          output_tokens=meta["output_tokens"], raw=meta["raw"],
                          repaired=repaired)

    def generate(self, req: PlanRequest) -> PlanResult:
        plan, meta = self._call([{"role": "user", "content": prompts.generate_prompt(req)}])
        return self._result(plan, meta)

    def replan(self, req: PlanRequest, previous: StudyPlan, today: date) -> PlanResult:
        msg = prompts.replan_prompt(req, previous.model_dump_json(), today.isoformat())
        plan, meta = self._call([{"role": "user", "content": msg}])
        return self._result(plan, meta)

    def repair(self, req: PlanRequest, bad: StudyPlan, violations: list[str]) -> PlanResult:
        plan, meta = self._call([
            {"role": "user", "content": prompts.generate_prompt(req)},
            {"role": "assistant", "content": bad.model_dump_json()},
            {"role": "user", "content": prompts.repair_prompt(violations)},
        ])
        return self._result(plan, meta, repaired=True)
