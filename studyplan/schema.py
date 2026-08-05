"""Input and output contracts.

The output JSON schema is written by hand (not derived from Pydantic) because
Anthropic structured outputs reject some JSON Schema features and dislike
$ref/$defs indirection. Every property is required and additionalProperties is
false everywhere, which also keeps the compiled grammar small.
"""

from __future__ import annotations

from datetime import date, timedelta

from pydantic import BaseModel, Field, field_validator

BLOCK_TYPES = ("learn", "practice", "revision", "buffer")


# --------------------------------------------------------------------------
# Input
# --------------------------------------------------------------------------
class Module(BaseModel):
    name: str
    exam_date: date
    difficulty: int = Field(ge=1, le=5)
    confidence: int = Field(ge=1, le=5)  # 1 = knows nothing yet, 5 = solid
    estimated_hours: float = Field(gt=0, le=200)


class Availability(BaseModel):
    # 0 = Monday ... 6 = Sunday
    hours_per_weekday: dict[int, float]
    blackout_dates: list[date] = Field(default_factory=list)
    max_session_minutes: int = 90
    min_session_minutes: int = 30
    day_start: str = "09:00"

    def minutes_for(self, day: date) -> int:
        if day in self.blackout_dates:
            return 0
        return int(round(self.hours_per_weekday.get(day.weekday(), 0.0) * 60))


class PlanRequest(BaseModel):
    start_date: date
    horizon_days: int = Field(ge=10, le=14)
    modules: list[Module] = Field(min_length=1, max_length=6)
    availability: Availability
    preferences: str = ""
    # replanning context
    completed_block_ids: list[str] = Field(default_factory=list)
    missed_block_ids: list[str] = Field(default_factory=list)
    locked_blocks: list[dict] = Field(default_factory=list)  # already-past blocks

    @property
    def end_date(self) -> date:
        return self.start_date + timedelta(days=self.horizon_days - 1)

    def days(self) -> list[date]:
        return [self.start_date + timedelta(days=i) for i in range(self.horizon_days)]


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
class StudyBlock(BaseModel):
    id: str
    date: date
    start_time: str  # "HH:MM"
    duration_minutes: int
    module: str
    topic: str
    block_type: str
    priority: int
    rationale: str

    @field_validator("block_type", mode="before")
    @classmethod
    def _normalise_type(cls, v: str) -> str:
        # Structured outputs do not guarantee enum capitalisation.
        v = str(v).strip().lower()
        return v if v in BLOCK_TYPES else "learn"

    @field_validator("priority", mode="before")
    @classmethod
    def _clamp_priority(cls, v) -> int:
        return max(1, min(5, int(v)))

    def end_minutes(self) -> int:
        return self.start_minutes() + self.duration_minutes

    def start_minutes(self) -> int:
        h, m = self.start_time.split(":")[:2]
        return int(h) * 60 + int(m)


class StudyPlan(BaseModel):
    plan_start: date
    plan_end: date
    strategy: str
    blocks: list[StudyBlock]
    risks: list[str]
    change_log: list[str]

    def total_minutes(self) -> int:
        return sum(b.duration_minutes for b in self.blocks)

    def minutes_by_module(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for b in self.blocks:
            out[b.module] = out.get(b.module, 0) + b.duration_minutes
        return out


# --------------------------------------------------------------------------
# JSON schema handed to the model
# --------------------------------------------------------------------------
PLAN_JSON_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "plan_start": {"type": "string", "description": "First day of the plan, YYYY-MM-DD"},
        "plan_end": {"type": "string", "description": "Last day of the plan, YYYY-MM-DD"},
        "strategy": {
            "type": "string",
            "description": "2-4 sentences explaining the overall allocation logic.",
        },
        "blocks": {
            "type": "array",
            "description": "All study blocks, ordered by date then start_time.",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string", "description": "Stable unique id, e.g. b001"},
                    "date": {"type": "string", "description": "YYYY-MM-DD"},
                    "start_time": {"type": "string", "description": "24h HH:MM"},
                    "duration_minutes": {
                        "type": "integer",
                        "description": "Between min_session_minutes and max_session_minutes.",
                    },
                    "module": {"type": "string", "description": "Exact module name from the input."},
                    "topic": {"type": "string", "description": "Concrete focus of this block."},
                    "block_type": {"type": "string", "enum": list(BLOCK_TYPES)},
                    "priority": {"type": "integer", "description": "1 = low, 5 = critical"},
                    "rationale": {
                        "type": "string",
                        "description": "One short sentence: why this block, here, now.",
                    },
                },
                "required": [
                    "id",
                    "date",
                    "start_time",
                    "duration_minutes",
                    "module",
                    "topic",
                    "block_type",
                    "priority",
                    "rationale",
                ],
            },
        },
        "risks": {
            "type": "array",
            "description": "Plan-level risks, e.g. under-covered modules or tight days.",
            "items": {"type": "string"},
        },
        "change_log": {
            "type": "array",
            "description": "What changed vs the previous plan. Empty list on first generation.",
            "items": {"type": "string"},
        },
    },
    "required": ["plan_start", "plan_end", "strategy", "blocks", "risks", "change_log"],
}
