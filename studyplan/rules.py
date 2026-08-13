"""Deterministic guardrails.

Structured outputs guarantee the shape of the plan, not its correctness. These
checks are what actually prevent a confabulated schedule from reaching the
student, and they are the metric the eval harness scores. Anything the model
cannot be trusted to enforce is enforced here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .schema import PlanRequest, StudyBlock, StudyPlan, normalise_chapter

SEVERITIES = ("error", "warning")

# Below this length a containment match is more likely to be a coincidence than
# a real reference, so short chapter names have to match exactly.
_MIN_PARTIAL = 4


def _topic_matches(topic_key: str, chapter_key: str) -> bool:
    if not topic_key or not chapter_key:
        return False
    if topic_key == chapter_key:
        return True
    short, long = sorted((topic_key, chapter_key), key=len)
    return len(short) >= _MIN_PARTIAL and short in long


@dataclass
class Violation:
    code: str
    message: str
    severity: str = "error"
    block_id: str | None = None

    def __str__(self) -> str:
        tag = f"[{self.block_id}] " if self.block_id else ""
        return f"{tag}{self.message}"


def validate_plan(req: PlanRequest, plan: StudyPlan, today: date | None = None) -> list[Violation]:
    v: list[Violation] = []
    av = req.availability
    exams = {m.name: m.exam_date for m in req.modules}
    known = set(exams)
    horizon = set(req.days())

    # R1 unknown module names (hallucinated modules)
    for b in plan.blocks:
        if b.module not in known:
            v.append(Violation("unknown_module", f"Module '{b.module}' was never provided.",
                               block_id=b.id))

    # R2 no work on or after the exam
    for b in plan.blocks:
        exam = exams.get(b.module)
        if exam and b.date >= exam:
            v.append(Violation("after_exam",
                               f"{b.module} block on {b.date} is on/after the exam ({exam}).",
                               block_id=b.id))

    # R3 inside the horizon
    for b in plan.blocks:
        if b.date not in horizon:
            v.append(Violation("outside_horizon", f"Block on {b.date} lies outside the plan window.",
                               block_id=b.id))

    # R4 blackout days and daily capacity
    per_day: dict[date, int] = {}
    for b in plan.blocks:
        per_day[b.date] = per_day.get(b.date, 0) + b.duration_minutes
    for day, minutes in sorted(per_day.items()):
        cap = av.minutes_for(day)
        if cap == 0:
            v.append(Violation("blackout", f"{day} is unavailable but has {minutes} min scheduled."))
        elif minutes > cap:
            v.append(Violation("over_capacity",
                               f"{day}: {minutes} min scheduled, only {cap} min available."))

    # R5 session length
    for b in plan.blocks:
        if not (av.min_session_minutes <= b.duration_minutes <= av.max_session_minutes):
            v.append(Violation("bad_duration",
                               f"{b.duration_minutes} min is outside "
                               f"{av.min_session_minutes}-{av.max_session_minutes}.",
                               block_id=b.id))

    # R6 overlaps
    by_day: dict[date, list[StudyBlock]] = {}
    for b in plan.blocks:
        by_day.setdefault(b.date, []).append(b)
    for day, blocks in by_day.items():
        ordered = sorted(blocks, key=lambda x: x.start_minutes())
        for a, b in zip(ordered, ordered[1:]):
            if b.start_minutes() < a.end_minutes():
                v.append(Violation("overlap", f"{day}: {a.id} and {b.id} overlap.", block_id=b.id))

    # R7 coverage
    scheduled = {b.module for b in plan.blocks}
    for m in req.modules:
        if m.name not in scheduled and m.exam_date > req.start_date:
            v.append(Violation("no_coverage", f"{m.name} has no study block at all."))

    # R8 revision close to the exam
    for m in req.modules:
        window = [b for b in plan.blocks
                  if b.module == m.name and b.block_type == "revision"
                  and 0 < (m.exam_date - b.date).days <= 3]
        has_any = any(b.module == m.name for b in plan.blocks)
        if has_any and not window:
            v.append(Violation("no_late_revision",
                               f"{m.name} has no revision block in the 3 days before the exam.",
                               severity="warning"))

    # R9 replanning must not touch the past
    if today:
        for b in plan.blocks:
            if b.date < today:
                v.append(Violation("past_edit", f"Block scheduled in the past ({b.date}).",
                                   block_id=b.id))

    # R10 workload proportionality (soft)
    got = plan.minutes_by_module()
    for m in req.modules:
        want = m.estimated_hours * 60
        have = got.get(m.name, 0)
        if want and have < 0.4 * want and m.exam_date > req.start_date:
            v.append(Violation("under_allocated",
                               f"{m.name}: {have} min scheduled vs {int(want)} min estimated.",
                               severity="warning"))

    # R11/R12 syllabus grounding. Chapters are optional, so a request without
    # any leaves these two checks completely inert and the older rules decide
    # the outcome exactly as they did before chapters existed.
    #
    # Both are warnings on purpose. A topic the student did not list is worth
    # showing, but it is not a reason to delete a block the student may still
    # want: errors() feeds the repair round and autorepair(), and neither
    # should be dropping otherwise-legal study time over a naming mismatch.
    if any(m.chapters for m in req.modules):
        chapter_keys = {m.name: {c.name: normalise_chapter(c.name) for c in m.chapters}
                        for m in req.modules if m.chapters}

        for b in plan.blocks:
            keys = chapter_keys.get(b.module)
            if not keys:
                continue
            topic_key = normalise_chapter(b.topic)
            if not any(_topic_matches(topic_key, k) for k in keys.values()):
                v.append(Violation("unknown_chapter",
                                   f"'{b.topic}' is not a chapter of {b.module}.",
                                   severity="warning", block_id=b.id))

        for m in req.modules:
            if not m.chapters or m.exam_date <= req.start_date:
                continue
            topics = [normalise_chapter(b.topic) for b in plan.blocks if b.module == m.name]
            for name, key in chapter_keys[m.name].items():
                if not any(_topic_matches(t, key) for t in topics):
                    v.append(Violation("chapter_uncovered",
                                       f"{m.name}: chapter '{name}' has no study block.",
                                       severity="warning"))
    return v


def errors(violations: list[Violation]) -> list[Violation]:
    return [x for x in violations if x.severity == "error"]


def autorepair(req: PlanRequest, plan: StudyPlan, today: date | None = None) -> tuple[StudyPlan, list[str]]:
    """Last-resort deterministic fix: drop illegal blocks, lowest priority first.

    Only used if the model still returns an invalid plan after one repair round.
    Every removal is logged so the student sees what was cut.
    """
    av = req.availability
    exams = {m.name: m.exam_date for m in req.modules}
    horizon = set(req.days())
    log: list[str] = []
    kept: list[StudyBlock] = []

    for b in sorted(plan.blocks, key=lambda x: (-x.priority, x.date, x.start_minutes())):
        reason = None
        if b.module not in exams:
            reason = "unknown module"
        elif b.date >= exams[b.module]:
            reason = "on or after the exam"
        elif b.date not in horizon:
            reason = "outside the plan window"
        elif av.minutes_for(b.date) == 0:
            reason = "unavailable day"
        elif today and b.date < today:
            reason = "in the past"
        if reason:
            log.append(f"Removed {b.id} ({b.module}, {b.date}): {reason}.")
            continue
        used = sum(x.duration_minutes for x in kept if x.date == b.date)
        if used + b.duration_minutes > av.minutes_for(b.date):
            log.append(f"Removed {b.id} ({b.module}, {b.date}): day was over capacity.")
            continue
        kept.append(b)

    kept.sort(key=lambda x: (x.date, x.start_minutes()))
    repaired = plan.model_copy(update={"blocks": kept,
                                       "change_log": plan.change_log + log})
    return repaired, log
