"""Prompt text. Kept in one file so it can be versioned and evaluated."""

from __future__ import annotations

import json

from .schema import PlanRequest

SYSTEM = """You are a study-planning engine. You do not tutor, explain subject
matter, or teach content. You only allocate time.

Hard rules, in order of precedence:
1. Never schedule a block for a module on or after that module's exam date.
2. Never exceed the available minutes given for a weekday, and never schedule on
   a blackout date.
3. Every session duration must lie between min_session_minutes and
   max_session_minutes.
4. Blocks on the same day must not overlap; leave at least 15 minutes between
   them.
5. Every module must get at least one revision block in the last three
   available days before its exam.
6. Total allocated time per module should track difficulty, low confidence, and
   exam proximity, not equal splits.
7. Leave roughly 10 percent of total capacity as buffer blocks so missed work
   can be absorbed.
8. If a module lists chapters, every topic for that module must be one of its
   chapter names, copied exactly. Cover every chapter at least once if capacity
   allows, and give more time to chapters with a high weight or a low
   confidence. A module with no chapters keeps a short descriptive topic of
   your own wording.

Front-load difficult and low-confidence modules. Put revision close to the exam.
Keep rationales to one short factual sentence. Do not invent modules, topics
you were not given, or facts about the student.
"""

_GENERATE = """Build a study plan for this student.

<input>
{payload}
</input>

Return one plan covering {start} to {end} inclusive. change_log must be an empty
list.
"""

_REPLAN = """The student is mid-plan and has missed work. Rebuild the remaining
schedule.

<input>
{payload}
</input>

<previous_plan>
{previous}
</previous_plan>

<progress>
completed_block_ids: {done}
missed_block_ids: {missed}
today: {today}
</progress>

Rules for the rebuild:
- Do not schedule anything before {today}. Past blocks are frozen and are given
  to you in locked_blocks for context only; do not repeat them in your output.
- Recover the content of missed blocks by folding it into future blocks. Do not
  simply drop it, and do not blindly re-add every missed block as-is.
- If the missed work no longer fits before the exam, cut the lowest-priority
  material, say so in risks, and protect revision blocks.
- change_log must list every meaningful change in plain language, one line each.

Return the plan for {today} to {end} inclusive.
"""


def _payload(req: PlanRequest) -> str:
    data = json.loads(req.model_dump_json())
    data["availability"]["hours_per_weekday"] = {
        ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][int(k)]: v
        for k, v in data["availability"]["hours_per_weekday"].items()
    }
    return json.dumps(data, indent=2, ensure_ascii=False)


def generate_prompt(req: PlanRequest) -> str:
    return _GENERATE.format(
        payload=_payload(req), start=req.start_date.isoformat(), end=req.end_date.isoformat()
    )


def replan_prompt(req: PlanRequest, previous_plan_json: str, today: str) -> str:
    return _REPLAN.format(
        payload=_payload(req),
        previous=previous_plan_json,
        done=", ".join(req.completed_block_ids) or "none",
        missed=", ".join(req.missed_block_ids) or "none",
        today=today,
        end=req.end_date.isoformat(),
    )


def schema_hint(schema: dict) -> str:
    """Used only in fallback mode, when the endpoint cannot enforce a schema.

    The schema then becomes an instruction rather than a constraint, which is
    strictly weaker. The rule engine still catches whatever slips through.
    """
    return (
        "Reply with a single JSON object and nothing else. No markdown, no code "
        "fences, no commentary. It must match this JSON Schema exactly, with every "
        "required field present:\n"
        f"{json.dumps(schema)}"
    )


def repair_prompt(violations: list[str]) -> str:
    joined = "\n".join(f"- {v}" for v in violations)
    return (
        "Your previous plan violated these hard rules:\n"
        f"{joined}\n\n"
        "Return a corrected full plan. Fix only what is listed, keep everything "
        "else stable, and note each fix in change_log."
    )
