"""Chapters: the time split, the planner, the rules, setup files, and the UI.

The UI half exists because the chapter table's requirements are behavioural, not
visual: values must survive a switch to another module and back, and nothing may
move or collapse under the student while they type. Those are exactly the
failures that a manual click-through stops catching after the third time, so
they are pinned with AppTest here.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest

from studyplan import rules, setup_io
from studyplan.planner import MockPlanner
from studyplan.schema import (Availability, Chapter, Module, PlanRequest, StudyBlock,
                              StudyPlan)

START = date(2026, 3, 2)


def module(**kw) -> Module:
    return Module(**{"name": "Stats", "exam_date": START + timedelta(days=12),
                     "difficulty": 4, "confidence": 2, "estimated_hours": 10.0, **kw})


def request_with(*mods, **kw) -> PlanRequest:
    return PlanRequest(**{
        "start_date": START, "horizon_days": 14, "modules": list(mods),
        "availability": Availability(hours_per_weekday={i: 3.0 for i in range(7)}),
        **kw})


def codes(violations) -> set[str]:
    return {v.code for v in violations}


# --------------------------------------------------------------------------
# the split
# --------------------------------------------------------------------------
def test_no_chapters_gives_empty_split():
    assert module().chapter_minutes() == {}


@pytest.mark.parametrize("hours", [1.0, 7.5, 13.3, 99.0])
def test_split_sums_exactly_to_the_module_budget(hours):
    m = module(estimated_hours=hours,
               chapters=[Chapter(name=f"C{i}", weight=(i % 5) + 1) for i in range(7)])
    assert sum(m.chapter_minutes().values()) == int(round(hours * 60))


def test_heavier_chapter_gets_more_time():
    split = module(chapters=[Chapter(name="small", weight=1),
                             Chapter(name="big", weight=5)]).chapter_minutes()
    assert split["big"] > split["small"]


def test_low_confidence_beats_equal_weight():
    split = module(chapters=[Chapter(name="shaky", weight=3, confidence=1),
                             Chapter(name="solid", weight=3, confidence=5)]).chapter_minutes()
    assert split["shaky"] > split["solid"]


def test_chapter_confidence_falls_back_to_the_module():
    m = module(confidence=3, chapters=[Chapter(name="a", weight=2),
                                       Chapter(name="b", weight=2, confidence=3)])
    split = m.chapter_minutes()
    assert split["a"] == split["b"]


def test_duplicate_chapter_names_keep_the_budget_intact():
    m = module(estimated_hours=7.5, chapters=[Chapter(name="dup", weight=1),
                                              Chapter(name="dup", weight=4)])
    assert sum(m.chapter_minutes().values()) == 450


# --------------------------------------------------------------------------
# the planner
# --------------------------------------------------------------------------
def test_chapterless_planning_is_unchanged():
    plan = MockPlanner().generate(request_with(module())).plan
    assert all(b.topic.startswith("Stats: ") and b.topic.endswith(" session")
               for b in plan.blocks)


def test_blocks_carry_chapter_names_as_topics():
    m = module(chapters=[Chapter(name="Probability"), Chapter(name="Regression")])
    plan = MockPlanner().generate(request_with(m)).plan
    assert {b.topic for b in plan.blocks} <= {"Probability", "Regression"}


def test_every_chapter_is_covered_when_capacity_allows():
    names = ("Alpha", "Beta", "Gamma")
    m = module(chapters=[Chapter(name=n) for n in names])
    plan = MockPlanner().generate(request_with(m)).plan
    assert {b.topic for b in plan.blocks} == set(names)


def test_heavier_chapter_receives_more_scheduled_minutes():
    m = module(estimated_hours=20.0,
               chapters=[Chapter(name="small", weight=1, confidence=5),
                         Chapter(name="big", weight=5, confidence=1)])
    plan = MockPlanner().generate(request_with(m)).plan
    got: dict[str, int] = {}
    for b in plan.blocks:
        got[b.topic] = got.get(b.topic, 0) + b.duration_minutes
    assert got.get("big", 0) > got.get("small", 0)


def test_revision_targets_the_shakiest_chapter():
    m = module(chapters=[Chapter(name="solid", weight=3, confidence=5),
                         Chapter(name="shaky", weight=3, confidence=1)])
    plan = MockPlanner().generate(request_with(m)).plan
    revision = [b.topic for b in plan.blocks if b.block_type == "revision"]
    assert revision and all(t == "shaky" for t in revision)


# --------------------------------------------------------------------------
# the rules
# --------------------------------------------------------------------------
def _plan(*blocks) -> StudyPlan:
    return StudyPlan(plan_start=START, plan_end=START + timedelta(days=13),
                     strategy="s", blocks=list(blocks), risks=[], change_log=[])


def _block(**kw) -> StudyBlock:
    return StudyBlock(**{"id": "b1", "date": START, "start_time": "09:00",
                         "duration_minutes": 60, "module": "Stats", "topic": "Probability",
                         "block_type": "learn", "priority": 3, "rationale": "r", **kw})


def test_off_syllabus_topic_is_flagged_as_a_warning_only():
    req = request_with(module(chapters=[Chapter(name="Probability")]))
    v = rules.validate_plan(req, _plan(_block(topic="Quantum tunnelling")))
    assert "unknown_chapter" in codes(v)
    assert rules.errors(v) == []


def test_decorated_chapter_name_still_matches():
    req = request_with(module(chapters=[Chapter(name="Distributions")]))
    v = rules.validate_plan(req, _plan(_block(topic="Ch. 3 - Distributions")))
    assert "unknown_chapter" not in codes(v)


def test_uncovered_chapter_is_flagged():
    req = request_with(module(chapters=[Chapter(name="Probability"),
                                        Chapter(name="Regression")]))
    v = rules.validate_plan(req, _plan(_block(topic="Probability")))
    assert "chapter_uncovered" in codes(v)
    assert rules.errors(v) == []


def test_chapter_rules_stay_quiet_without_chapters():
    req = request_with(module())
    v = rules.validate_plan(req, _plan(_block(topic="anything at all")))
    assert not ({"unknown_chapter", "chapter_uncovered"} & codes(v))


def test_autorepair_keeps_blocks_with_off_syllabus_topics():
    req = request_with(module(chapters=[Chapter(name="Probability")]))
    repaired, log = rules.autorepair(req, _plan(_block(topic="Something else")))
    assert len(repaired.blocks) == 1 and log == []


# --------------------------------------------------------------------------
# setup files
# --------------------------------------------------------------------------
def _setup_request() -> PlanRequest:
    return request_with(module(chapters=[Chapter(name="Probability", weight=4, confidence=2)]),
                        preferences="mornings")


def test_setup_round_trips_including_chapters():
    back = setup_io.setup_from_json(setup_io.setup_to_json(_setup_request()))
    assert back.modules[0].name == "Stats"
    assert back.modules[0].exam_date == START + timedelta(days=12)
    assert [(c.name, c.weight, c.confidence) for c in back.modules[0].chapters] \
        == [("Probability", 4, 2)]
    assert back.start_date == START and back.horizon_days == 14
    assert back.availability.hours_per_weekday == {i: 3.0 for i in range(7)}
    assert back.preferences == "mornings"


def test_chapterless_setup_round_trips_too():
    back = setup_io.setup_from_json(setup_io.setup_to_json(request_with(module())))
    assert back.modules[0].chapters == []


@pytest.mark.parametrize("text", [
    "not json at all",
    '{"version": 99, "modules": []}',
    '{"version": 1, "modules": []}',
    "[1, 2, 3]",
])
def test_bad_setup_files_are_rejected(text):
    with pytest.raises(ValueError):
        setup_io.setup_from_json(text)


def test_setup_rejects_an_out_of_range_horizon():
    text = setup_io.setup_to_json(request_with(module()))
    assert '"horizon_days": 14' in text
    with pytest.raises(ValueError, match="horizon_days"):
        setup_io.setup_from_json(text.replace('"horizon_days": 14', '"horizon_days": 40'))


# --------------------------------------------------------------------------
# the syllabus parser
# --------------------------------------------------------------------------
def test_parses_a_pasted_numbered_syllabus():
    rows = setup_io.parse_syllabus(
        "1. Descriptive statistics (pp. 1-20)\n"
        "2) Probability .......... 21\n"
        "Ch. 3 - Distributions\n"
        "- Hypothesis testing\n"
        "\n"
        "* Regression   Seiten 120-148\n")
    assert [r["name"] for r in rows] == ["Descriptive statistics", "Probability",
                                         "Distributions", "Hypothesis testing", "Regression"]
    assert all(r["weight"] == 3 for r in rows)


def test_parser_keeps_a_title_that_merely_starts_with_a_digit():
    assert [r["name"] for r in setup_io.parse_syllabus("3D geometry")] == ["3D geometry"]


def test_parser_drops_duplicates_and_noise():
    rows = setup_io.parse_syllabus("Probability\nprobability\n\n   \n-\nRegression")
    assert [r["name"] for r in rows] == ["Probability", "Regression"]


def test_parser_is_bounded():
    many = "\n".join(f"Chapter number {i}" for i in range(200))
    assert len(setup_io.parse_syllabus(many)) == setup_io.MAX_CHAPTERS


def test_empty_paste_yields_nothing():
    assert setup_io.parse_syllabus("") == []


# --------------------------------------------------------------------------
# the Setup tab
# --------------------------------------------------------------------------
def _fresh_app():
    """A new session on the real app, signed in and pinned to the offline planner."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"),
                           default_timeout=90)
    at.session_state["force_mock"] = True
    # Past the login gate, which otherwise stops the script before any of this
    # exists. The gate itself is covered in test_auth.py.
    at.session_state["auth_user"] = "student1"
    return at.run()


@pytest.fixture
def app():
    return _fresh_app()


def _ui_chapters(at, name) -> list[str]:
    return [str(r.get("name")) for r in at.session_state["chapters"][name]]


def test_setup_tab_renders_without_error(app):
    assert not app.exception


def test_chapters_are_not_behind_per_module_expanders(app):
    # One expander in the whole app, the advanced settings. A chapter table that
    # lives in an expander collapses on the rerun each cell edit triggers.
    assert [e.label for e in app.expander] == [
        "⚙ Advanced settings — session length, blackout dates, preferences"]


def test_editing_chapters_never_moves_the_selection(app):
    app.session_state["chapter_seed"]["Statistics"] = [
        {"name": "Probability", "weight": 3, "confidence": 2}]
    app.run()
    assert not app.exception
    assert app.session_state["chapter_focus"] == "Statistics"
    assert app.radio(key="chapter_focus").options == ["Statistics", "Microeconomics", "Databases"]


def _type_rows(at, module_name, *names):
    """Add rows the way the browser does, as a pending delta on the editor."""
    at.session_state[f"chapters::{module_name}"] = {
        "edited_rows": {},
        "added_rows": [{"name": n, "weight": 3, "confidence": None} for n in names],
        "deleted_rows": [],
    }


def test_each_module_keeps_its_own_chapters_across_switches(app):
    # Rows are added through the editor rather than by writing the seed, because
    # the seed is the one thing a module switch is allowed to rebuild: seeding
    # directly would pass even with the switch handling removed entirely.
    _type_rows(app, "Statistics", "Probability")
    app.run()
    assert _ui_chapters(app, "Statistics") == ["Probability"]

    app.radio(key="chapter_focus").set_value("Databases").run()
    _type_rows(app, "Databases", "SQL joins")
    app.run()
    assert _ui_chapters(app, "Databases") == ["SQL joins"]

    # Back again: Statistics must still hold its own rows, not Databases'.
    app.radio(key="chapter_focus").set_value("Statistics").run()
    assert _ui_chapters(app, "Statistics") == ["Probability"]
    assert _ui_chapters(app, "Databases") == ["SQL joins"]

    app.radio(key="chapter_focus").set_value("Databases").run()
    assert _ui_chapters(app, "Databases") == ["SQL joins"]


def test_chapters_survive_a_switch_that_crosses_an_unrelated_module(app):
    _type_rows(app, "Statistics", "Probability", "Regression")
    app.run()
    for name in ("Microeconomics", "Databases", "Statistics"):
        app.radio(key="chapter_focus").set_value(name).run()
    assert _ui_chapters(app, "Statistics") == ["Probability", "Regression"]
    assert _ui_chapters(app, "Microeconomics") == []


def test_switching_away_mid_edit_keeps_the_pending_edit(app):
    # The one-frame hole: a cell edit and a click on another module arrive in the
    # same message, so the editor is never re-rendered to return its value.
    app.session_state["chapter_seed"]["Statistics"] = [
        {"name": "Probability", "weight": 3, "confidence": 2}]
    app.run()
    app.session_state["chapters::Statistics"] = {
        "edited_rows": {0: {"weight": 5}}, "added_rows": [], "deleted_rows": []}
    app.radio(key="chapter_focus").set_value("Databases").run()
    assert app.session_state["chapters"]["Statistics"][0]["weight"] == 5


def test_renaming_a_module_carries_its_chapters_and_the_selection(app):
    app.session_state["chapter_seed"]["Statistics"] = [
        {"name": "Probability", "weight": 4, "confidence": 2}]
    app.run()

    renamed = [dict(m) for m in app.session_state["modules"]]
    renamed[0]["name"] = "Statistik"
    app.session_state["module_seed"] = renamed
    app.session_state["modules"] = renamed
    del app.session_state["module_editor"]
    app.run()

    assert not app.exception
    assert app.session_state["chapter_focus"] == "Statistik"
    assert _ui_chapters(app, "Statistik") == ["Probability"]
    assert "Statistics" not in app.session_state["chapters"]


def test_deleting_a_module_drops_its_chapters(app):
    app.session_state["chapter_seed"]["Databases"] = [{"name": "SQL joins", "weight": 3}]
    app.run()
    kept = [dict(m) for m in app.session_state["modules"] if m["name"] != "Databases"]
    app.session_state["module_seed"] = kept
    app.session_state["modules"] = kept
    del app.session_state["module_editor"]
    app.run()
    assert "Databases" not in app.session_state["chapters"]


def test_duplicate_module_names_are_refused(app):
    dupes = [dict(m) for m in app.session_state["modules"]]
    dupes[1]["name"] = dupes[0]["name"]
    app.session_state["module_seed"] = dupes
    app.session_state["modules"] = dupes
    del app.session_state["module_editor"]
    app.run()

    next(b for b in app.button if b.label == "Generate plan").click().run()
    assert any("unique" in e.value for e in app.error)
    assert app.session_state["plan"] is None


def _save_button(at):
    return next(b for b in at.download_button if b.label == "Save setup (JSON)")


def test_the_save_button_offers_a_file_once_the_setup_is_valid(app):
    # AppTest cannot read a download button's payload, so what is checked here is
    # that saving is actually available; that the request it serialises carries
    # the chapters is covered by test_chapters_reach_the_generated_plan, which
    # goes through the same build_request().
    assert not _save_button(app).disabled


def test_uploading_a_setup_restores_everything_it_holds():
    saved = setup_io.setup_to_json(PlanRequest(
        start_date=START, horizon_days=11,
        modules=[
            module(name="Algebra", chapters=[Chapter(name="Groups", weight=5, confidence=1),
                                             Chapter(name="Rings", weight=2)]),
            module(name="Optics", estimated_hours=6.0),
        ],
        availability=Availability(hours_per_weekday={i: 1.0 for i in range(7)} | {0: 4.5},
                                  blackout_dates=[START + timedelta(days=3)],
                                  min_session_minutes=45, max_session_minutes=120,
                                  day_start="08:15"),
        preferences="evenings only"))

    at = _fresh_app()
    assert at.session_state["chapters"] == {"Statistics": [], "Microeconomics": [], "Databases": []}
    at.file_uploader(key="setup_upload").upload("setup.json", saved.encode("utf-8"))
    at.run()

    assert not at.exception
    assert [m["name"] for m in at.session_state["modules"]] == ["Algebra", "Optics"]
    assert _ui_chapters(at, "Algebra") == ["Groups", "Rings"]
    assert at.session_state["chapters"]["Optics"] == []
    assert at.session_state["chapter_focus"] == "Algebra"
    assert at.session_state["horizon"] == 11
    assert at.session_state["h0"] == 4.5 and at.session_state["h1"] == 1.0
    assert at.session_state["min_len"] == 45 and at.session_state["max_len"] == 120
    assert at.session_state["day_start"] == "08:15"
    assert at.session_state["blackout_raw"] == (START + timedelta(days=3)).isoformat()
    assert at.session_state["preferences"] == "evenings only"


def test_an_uploaded_setup_survives_further_edits():
    # Re-applying the file on every rerun would wipe whatever is typed next.
    saved = setup_io.setup_to_json(
        request_with(module(name="Algebra", chapters=[Chapter(name="Groups")])))
    at = _fresh_app()
    at.file_uploader(key="setup_upload").upload("setup.json", saved.encode("utf-8"))
    at.run()
    at.slider(key="horizon").set_value(12).run()
    assert at.session_state["horizon"] == 12
    assert _ui_chapters(at, "Algebra") == ["Groups"]


def test_a_broken_setup_file_is_reported_not_raised(app):
    app.file_uploader(key="setup_upload").upload("setup.json", b"not json at all")
    app.run()
    assert not app.exception
    assert any("Could not load that setup" in e.value for e in app.error)


def test_pasting_a_syllabus_fills_the_chapter_table(app):
    app.text_area(key="syllabus::Statistics").set_value(
        "1. Descriptive statistics (pp. 1-20)\n2) Probability .......... 21").run()
    next(b for b in app.button if b.key == "add_syllabus::Statistics").click().run()
    assert not app.exception
    assert _ui_chapters(app, "Statistics") == ["Descriptive statistics", "Probability"]
    # The box empties, so a second click cannot silently double the rows.
    assert app.session_state["syllabus::Statistics"] == ""


def test_pasting_does_not_duplicate_existing_chapters(app):
    _type_rows(app, "Statistics", "Probability")
    app.run()
    app.text_area(key="syllabus::Statistics").set_value("Probability\nRegression").run()
    next(b for b in app.button if b.key == "add_syllabus::Statistics").click().run()
    assert _ui_chapters(app, "Statistics") == ["Probability", "Regression"]


def test_chapters_reach_the_generated_plan(app):
    app.session_state["chapter_seed"]["Statistics"] = [
        {"name": "Probability", "weight": 3, "confidence": 2},
        {"name": "Regression", "weight": 3, "confidence": 2}]
    app.run()
    next(b for b in app.button if b.label == "Generate plan").click().run()
    assert not app.exception
    plan = app.session_state["plan"]
    assert plan is not None
    topics = {b.topic for b in plan.blocks if b.module == "Statistics"}
    assert topics <= {"Probability", "Regression"} and topics
