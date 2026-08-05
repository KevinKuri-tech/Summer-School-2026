from datetime import date, timedelta

import pytest

from studyplan import rules
from studyplan.planner import MockPlanner
from studyplan.schema import Availability, Module, PlanRequest, StudyBlock, StudyPlan

START = date(2026, 3, 2)  # a Monday


def make_request(**kw) -> PlanRequest:
    base = dict(
        start_date=START,
        horizon_days=14,
        modules=[
            Module(name="Stats", exam_date=START + timedelta(days=12), difficulty=4,
                   confidence=2, estimated_hours=10),
            Module(name="Micro", exam_date=START + timedelta(days=9), difficulty=3,
                   confidence=3, estimated_hours=8),
        ],
        availability=Availability(hours_per_weekday={i: 2.0 for i in range(7)}),
    )
    base.update(kw)
    return PlanRequest(**base)


def block(**kw) -> StudyBlock:
    base = dict(id="b1", date=START, start_time="09:00", duration_minutes=60, module="Stats",
                topic="t", block_type="learn", priority=3, rationale="r")
    base.update(kw)
    return StudyBlock(**base)


def plan_with(*blocks) -> StudyPlan:
    return StudyPlan(plan_start=START, plan_end=START + timedelta(days=13), strategy="s",
                     blocks=list(blocks), risks=[], change_log=[])


def codes(violations):
    return {v.code for v in violations}


def test_block_type_is_normalised():
    assert block(block_type="Revision").block_type == "revision"
    assert block(block_type="nonsense").block_type == "learn"


def test_flags_block_on_exam_day():
    req = make_request()
    p = plan_with(block(date=START + timedelta(days=12)))
    assert "after_exam" in codes(rules.validate_plan(req, p))


def test_flags_over_capacity():
    req = make_request()
    p = plan_with(block(id="a", duration_minutes=90),
                  block(id="b", start_time="11:00", duration_minutes=90))
    assert "over_capacity" in codes(rules.validate_plan(req, p))


def test_flags_overlap():
    req = make_request(availability=Availability(hours_per_weekday={i: 8.0 for i in range(7)}))
    p = plan_with(block(id="a", start_time="09:00", duration_minutes=60),
                  block(id="b", start_time="09:30", duration_minutes=60))
    assert "overlap" in codes(rules.validate_plan(req, p))


def test_flags_blackout_and_unknown_module():
    req = make_request(availability=Availability(
        hours_per_weekday={i: 2.0 for i in range(7)},
        blackout_dates=[START + timedelta(days=1)]))
    p = plan_with(block(date=START + timedelta(days=1), module="Astrology"))
    found = codes(rules.validate_plan(req, p))
    assert {"blackout", "unknown_module"} <= found


def test_flags_missing_coverage():
    req = make_request()
    p = plan_with(block())
    assert "no_coverage" in codes(rules.validate_plan(req, p))


def test_replan_may_not_touch_the_past():
    req = make_request()
    today = START + timedelta(days=3)
    p = plan_with(block(date=START + timedelta(days=1)))
    assert "past_edit" in codes(rules.validate_plan(req, p, today=today))


def test_autorepair_drops_illegal_blocks_and_logs():
    req = make_request()
    p = plan_with(block(id="ok"), block(id="bad", date=START + timedelta(days=13)))
    fixed, log = rules.autorepair(req, p)
    assert [b.id for b in fixed.blocks] == ["ok"]
    assert len(log) == 1


def test_mock_planner_produces_a_clean_plan():
    req = make_request()
    result = MockPlanner().generate(req)
    hard = rules.errors(rules.validate_plan(req, result.plan))
    assert hard == [], [str(v) for v in hard]
    assert result.plan.blocks


def test_mock_replan_carries_missed_minutes_forward():
    req = make_request()
    first = MockPlanner().generate(req).plan
    missed = [b.id for b in first.blocks[:2]]
    today = START + timedelta(days=2)
    req2 = req.model_copy(update={"missed_block_ids": missed})
    result = MockPlanner().replan(req2, first, today)
    assert all(b.date >= today for b in result.plan.blocks)
    assert result.plan.change_log
