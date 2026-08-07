"""AI Study-Plan Optimizer, Streamlit UI.

Run: uv run streamlit run app.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

# Make the sibling studyplan/ package importable no matter where the app is
# launched from (cwd, AppTest, some Streamlit/Windows setups).
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from studyplan import exporting, planner as planner_mod, rules
from studyplan.planner import (ANTHROPIC_PRESETS, DEFAULT_MODEL, DEFAULT_OPENROUTER_MODEL,
                               OPENROUTER_PRESETS, MockPlanner, PlannerError,
                               available_backends, is_free_model)
from studyplan.schema import Availability, Module, PlanRequest, StudyBlock, StudyPlan

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

APP_NAME = "Nexora Study"
LOGO = Path(__file__).resolve().parent / "media" / "pictures" / "Logo.png"

st.set_page_config(page_title=APP_NAME, page_icon=str(LOGO) if LOGO.exists() else "📅",
                   layout="wide")

# Top-left app chrome (above the sidebar), plus a header in the main area so the
# name sits next to the mark on every tab.
if LOGO.exists():
    st.logo(str(LOGO), size="large")

_logo_col, _title_col = st.columns([1, 4], vertical_alignment="center")
with _logo_col:
    if LOGO.exists():
        # The file carries generous whitespace margins, so the mark renders a
        # good deal smaller than the box it sits in.
        st.image(str(LOGO), width=240)
with _title_col:
    st.title(APP_NAME, anchor=False)


# --------------------------------------------------------------------------
# state
# --------------------------------------------------------------------------
def init_state() -> None:
    ss = st.session_state
    ss.setdefault("plan", None)          # StudyPlan
    ss.setdefault("request", None)       # PlanRequest
    ss.setdefault("status", {})          # block_id -> todo | done | missed
    ss.setdefault("violations", [])
    ss.setdefault("usage", {"calls": 0, "in": 0, "out": 0})
    ss.setdefault("source", "")
    ss.setdefault("accepted", False)
    ss.setdefault("last_call", None)         # what was sent, where, how long it took
    ss.setdefault("last_replan_call", None)  # the same, for the replan action
    ss.setdefault("paid_model_ack", False)   # cost warning confirmed this session
    ss.setdefault("modules", [
        {"name": "Statistics", "exam_date": date.today() + timedelta(days=12),
         "difficulty": 4, "confidence": 2, "estimated_hours": 14.0},
        {"name": "Microeconomics", "exam_date": date.today() + timedelta(days=9),
         "difficulty": 3, "confidence": 3, "estimated_hours": 9.0},
        {"name": "Databases", "exam_date": date.today() + timedelta(days=14),
         "difficulty": 3, "confidence": 4, "estimated_hours": 7.0},
    ])


init_state()


def get_planner():
    return planner_mod.get_planner(
        force_mock=st.session_state.get("force_mock", False),
        model=st.session_state.get("model") or None,
        backend=st.session_state.get("backend"),
    )


def track(result) -> None:
    u = st.session_state.usage
    u["calls"] += 1
    u["in"] += result.input_tokens
    u["out"] += result.output_tokens
    st.session_state.source = result.source


def run_with_guardrails(fn, req: PlanRequest, *args, today: date | None = None,
                        planner=None, **kw):
    """One generation, one model-side repair round, then deterministic repair.

    Callers that already built a planner (to show the request before the call)
    pass it in, so the panel describes the instance that does the work.
    """
    planner = planner or get_planner()
    result = fn(planner, req, *args, **kw)
    track(result)
    viol = rules.validate_plan(req, result.plan, today=today)
    hard = rules.errors(viol)
    if hard:
        try:
            fixed = planner.repair(req, result.plan, [str(x) for x in hard])
            track(fixed)
            viol2 = rules.validate_plan(req, fixed.plan, today=today)
            if len(rules.errors(viol2)) < len(hard):
                result, viol = fixed, viol2
                hard = rules.errors(viol)
        except PlannerError as exc:
            st.warning(f"Repair call failed: {exc}")
    if hard:
        repaired_plan, log = rules.autorepair(req, result.plan, today=today)
        result.plan = repaired_plan
        viol = rules.validate_plan(req, repaired_plan, today=today)
        if log:
            st.warning(f"{len(log)} invalid block(s) were removed automatically. See Changes.")
    return result, viol


# --------------------------------------------------------------------------
# "what am I waiting for" panel
# --------------------------------------------------------------------------
# Streamlit cannot repaint while the planner call blocks the script thread, so a
# server-side counter would sit frozen at 0.0. This ticks inside an iframe
# instead, which the browser keeps painting independently of Python.
_CLOCK_HTML = """
<div id="clock">waiting 0.0 s</div>
<style>
  html, body { margin: 0; background: transparent; }
  #clock {
    font-family: "Source Sans Pro", -apple-system, BlinkMacSystemFont, sans-serif;
    font-size: 0.9rem; font-variant-numeric: tabular-nums; color: #808495;
    text-align: right; white-space: nowrap;
  }
</style>
<script>
  (function () {
    var started = Date.now();
    var el = document.getElementById("clock");
    setInterval(function () {
      el.textContent = "waiting " + ((Date.now() - started) / 1000).toFixed(1) + " s";
    }, 100);
  })();
</script>
"""

BACKEND_LABEL = {
    "AnthropicPlanner": "Anthropic API",
    "OpenRouterPlanner": "OpenRouter",
    "MockPlanner": "Rule-based baseline, no network call",
}


def call_info(planner, payload: dict | None) -> dict:
    """Everything the panel needs, captured before the call starts."""
    return {
        "model": getattr(planner, "model", None),
        "endpoint": getattr(planner, "endpoint", None),
        "backend": BACKEND_LABEL.get(type(planner).__name__, type(planner).__name__),
        "payload": payload,
        "elapsed": None,          # set once the call returns
    }


def _schema_mode(payload: dict) -> str:
    """How hard the schema is enforced for this particular request."""
    if "output_config" in payload:
        return "json_schema, grammar-constrained by the decoder"
    fmt = (payload.get("response_format") or {}).get("type")
    if fmt == "json_schema":
        return "json_schema, enforced by the provider endpoint"
    if fmt == "json_object":
        return "json_object, the schema is only a hint in the prompt"
    return "unconstrained"


def render_call_panel(info: dict):
    """Model, exact request payload and the clock. Returns the clock slot.

    The caller overwrites that slot with the final duration once the call
    returns, which also tears down the still-ticking iframe.
    """
    payload = info.get("payload")
    with st.container(border=True):
        left, right = st.columns([4, 1], vertical_alignment="center")
        with left:
            if info.get("model"):
                st.markdown(f"**Contacting** `{info['model']}` · {info['backend']}")
                if info.get("endpoint"):
                    st.caption(f"POST {info['endpoint']}")
            else:
                st.markdown(f"**No model contacted** · {info['backend']}")
                st.caption("The plan is computed locally by the greedy scheduler.")
            # Absent while the call is still running, so nothing shows then.
            if info.get("ok") is False:
                st.caption("⚠ This request failed. The payload below is what was sent.")
        with right:
            clock = st.empty()
            with clock:
                if info.get("elapsed") is None:
                    # Static literal, never user input: safe for st.iframe, which
                    # runs the script the counter needs.
                    st.iframe(_CLOCK_HTML, height=24)
                else:
                    st.markdown(f"took **{info['elapsed']:.1f} s**")

        if payload:
            raw = json.dumps(payload, indent=2, ensure_ascii=False)
            # Anthropic carries the system prompt in its own field, OpenAI-shaped
            # bodies put it in messages[0].
            msgs = payload.get("messages", [])
            system = payload.get("system") or next(
                (m["content"] for m in msgs if m.get("role") == "system"), "")
            convo = [m for m in msgs if m.get("role") != "system"]

            with st.expander(f"Exact request sent · {len(raw) / 1024:.1f} KB JSON"):
                st.caption(f"max_tokens {payload.get('max_tokens')} · "
                           f"structured output: {_schema_mode(payload)}")
                if info.get("calls", 1) > 1:
                    st.caption(f"{info['calls']} requests were sent for this action "
                               "(the first plan plus a repair round). The payload "
                               "below is the last one.")
                t_user, t_sys, t_raw = st.tabs(
                    ["User message", "System prompt", "Full JSON body"])
                with t_user:
                    for m in convo:
                        st.caption(f"role: {m.get('role')}")
                        st.code(m.get("content", ""), language="text")
                with t_sys:
                    st.code(system, language="text")
                with t_raw:
                    st.caption("Byte-for-byte what goes into the HTTP request body.")
                    st.code(raw, language="json")
    return clock


def call_with_panel(call_slot, status_slot, planner, payload, spinner_text,
                    fn, req: PlanRequest, **kw):
    """Show the request and a running clock, make the call, freeze the clock.

    Shared by Generate and Replan so both report the same way. Returns
    (result, violations, info); result is None if the call failed.
    """
    info = call_info(planner, payload)
    with call_slot:
        clock = render_call_panel(info)

    before = st.session_state.usage["calls"]
    started = time.perf_counter()
    with status_slot, st.spinner(spinner_text):
        try:
            result, viol = run_with_guardrails(fn, req, planner=planner, **kw)
        except PlannerError as exc:
            st.error(str(exc))
            result, viol = None, []
    info["elapsed"] = time.perf_counter() - started

    # The iframe clock ticks on its own, so replace it rather than leaving it
    # running after the answer is back.
    clock.markdown(f"took **{info['elapsed']:.1f} s**")
    # A repair round is a second request, and on OpenRouter the body that went
    # out can differ from the preview (JSON-mode fallback). Keep what was real.
    info["payload"] = getattr(planner, "last_request", None) or info["payload"]
    info["calls"] = st.session_state.usage["calls"] - before
    info["ok"] = result is not None
    return result, viol, info


# --------------------------------------------------------------------------
# sidebar
# --------------------------------------------------------------------------
BACKEND_NAME = {"anthropic": "Anthropic API", "openrouter": "OpenRouter",
                "mock": "Rule-based baseline (no API call)"}


@st.cache_data(ttl=600, show_spinner=False)
def anthropic_model_options() -> list[tuple[str, str]]:
    """(id, label) for every model the key may call. Empty if unreachable.

    Cached because it is a network round trip and Streamlit reruns the whole
    script on every widget interaction.
    """
    try:
        return [(m.id, m.label) for m in planner_mod.list_anthropic_models()]
    except PlannerError:
        return []


with st.sidebar:
    st.subheader("Engine")
    backends = available_backends()
    # Both provider keys can be present, so the backend is a choice rather than
    # something the environment decides for us. A keyed widget takes its value
    # from session state, which outranks any index= we pass, so the preselection
    # has to be seeded there instead. Re-seeded when a stored value is no longer
    # offered, e.g. after a key was removed from .env.
    if st.session_state.get("backend") not in backends:
        st.session_state["backend"] = backends[0]
    backend = st.radio("Backend", backends, format_func=BACKEND_NAME.get,
                       key="backend", horizontal=False)
    label = BACKEND_NAME[backend]

    if backend == "anthropic":
        options = anthropic_model_options()
        if options:
            ids = [i for i, _ in options]
            labels = dict(options)
            default = DEFAULT_MODEL if DEFAULT_MODEL in ids else ids[0]
            st.session_state["model"] = st.selectbox(
                "Model", ids, index=ids.index(default),
                format_func=lambda i: labels[i], key="anthropic_model")
            c1, c2 = st.columns([3, 1], vertical_alignment="center")
            c1.caption(f"{len(ids)} models available to this key.")
            if c2.button("↻", help="Re-query the Anthropic models list"):
                anthropic_model_options.clear()
                st.rerun()
        else:
            st.session_state["model"] = st.selectbox(
                "Model", ANTHROPIC_PRESETS, key="anthropic_model")
            st.caption("Could not reach the models endpoint, so this is the "
                       "built-in list rather than what the key can call.")
        st.caption("Only models that support structured outputs are listed: the "
                   "planner constrains the response to the plan schema.")
    elif backend == "openrouter":
        # The default slug leads, so it is what the picker preselects even when
        # STUDYPLAN_OPENROUTER_MODEL points at something outside the presets.
        presets = list(dict.fromkeys([DEFAULT_OPENROUTER_MODEL] + OPENROUTER_PRESETS))
        if st.session_state.get("openrouter_choice") not in presets + ["custom..."]:
            st.session_state["openrouter_choice"] = DEFAULT_OPENROUTER_MODEL
        choice = st.selectbox("Model", presets + ["custom..."],
                              key="openrouter_choice")
        st.session_state["model"] = st.text_input(
            "Model slug", value=DEFAULT_OPENROUTER_MODEL, key="openrouter_slug") \
            if choice == "custom..." else choice
        st.caption("Any OpenRouter slug works. `:free` suffix means the free tier, "
                   "which is rate limited and may train on your data.")
    else:
        st.session_state["model"] = None
        st.caption("No model is contacted. The plan is computed by the greedy scheduler.")

    st.toggle("Force baseline planner (no API cost)", key="force_mock",
              value=backend == "mock", disabled=backend == "mock")

    # Cost gate. Checked after the toggle, because forcing the baseline means
    # nothing can be billed no matter which model is picked above.
    effective_backend = "mock" if st.session_state.force_mock else backend
    selected_model = st.session_state.get("model") if effective_backend != "mock" else None
    paid_model = not is_free_model(selected_model, effective_backend)
    needs_ack = paid_model and not st.session_state.paid_model_ack

    if needs_ack:
        st.warning(
            f"`{selected_model}` is a paid model. Generating and replanning will "
            f"bill your {label} account, and each action can cost two requests "
            "because a rule violation triggers a repair round."
        )
        if st.button("I understand, use the paid model", width="stretch"):
            st.session_state.paid_model_ack = True
            st.rerun()
        st.caption("Asked once per session. Pick a `:free` slug or force the "
                   "baseline planner to avoid this.")
    elif paid_model:
        st.caption("Paid model confirmed for this session.")

    u = st.session_state.usage
    st.metric("AI calls this session", u["calls"])
    st.caption(f"Tokens in/out: {u['in']} / {u['out']}")
    st.divider()
    st.caption("Plans are AI-generated. Review and edit before you rely on them, "
               "and check every exam date against the official schedule.")


# --------------------------------------------------------------------------
# tabs
# --------------------------------------------------------------------------
tab_setup, tab_plan, tab_progress, tab_export = st.tabs(
    ["1 Setup", "2 Plan", "3 Progress", "4 Export"])

# ---------------------------------------------------------------- setup ---
with tab_setup:
    st.subheader("Modules")
    st.caption("Three to six modules. Confidence 1 means you have not started, 5 means solid.")
    edited = st.data_editor(
        st.session_state.modules,
        num_rows="dynamic",
        width="stretch",
        column_config={
            "name": st.column_config.TextColumn("Module", required=True),
            "exam_date": st.column_config.DateColumn("Exam date", required=True),
            "difficulty": st.column_config.NumberColumn("Difficulty", min_value=1, max_value=5, step=1),
            "confidence": st.column_config.NumberColumn("Confidence", min_value=1, max_value=5, step=1),
            "estimated_hours": st.column_config.NumberColumn("Est. hours", min_value=1.0,
                                                             max_value=100.0, step=0.5),
        },
        key="module_editor",
    )
    st.session_state.modules = edited

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Availability (hours per weekday)")
        hours = {}
        cols = st.columns(7)
        defaults = [2.0, 2.0, 2.0, 2.0, 1.5, 4.0, 3.0]
        for i, col in enumerate(cols):
            with col:
                hours[i] = st.number_input(WEEKDAYS[i], min_value=0.0, max_value=12.0,
                                           value=defaults[i], step=0.5, key=f"h{i}")
    with c2:
        st.subheader("Window and constraints")
        start = st.date_input("Plan starts", value=date.today())
        horizon = st.slider("Horizon (days)", 10, 14, 14)
        cc1, cc2 = st.columns(2)
        min_len = cc1.number_input("Min session (min)", 20, 90, 30, step=5)
        max_len = cc2.number_input("Max session (min)", 30, 240, 90, step=15)
        day_start = st.text_input("Earliest start time", "09:00")
        blackout_raw = st.text_input("Blackout dates (YYYY-MM-DD, comma separated)", "")
        preferences = st.text_area(
            "Preferences and constraints",
            "Prefer mornings. No study after 21:00. Wednesday evenings are blocked by work.",
            height=80)

    blackout = []
    for token in [t.strip() for t in blackout_raw.split(",") if t.strip()]:
        try:
            blackout.append(datetime.strptime(token, "%Y-%m-%d").date())
        except ValueError:
            st.warning(f"Ignored blackout date '{token}'.")

    def build_request() -> PlanRequest | None:
        try:
            mods = [Module(**{**m, "exam_date": (m["exam_date"] if isinstance(m["exam_date"], date)
                                                 else datetime.strptime(str(m["exam_date"]), "%Y-%m-%d").date())})
                    for m in st.session_state.modules if m.get("name")]
            if not 1 <= len(mods) <= 6:
                st.error("Enter between 1 and 6 modules.")
                return None
            return PlanRequest(
                start_date=start,
                horizon_days=horizon,
                modules=mods,
                availability=Availability(
                    hours_per_weekday=hours, blackout_dates=blackout,
                    max_session_minutes=int(max_len), min_session_minutes=int(min_len),
                    day_start=day_start),
                preferences=preferences,
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Invalid input: {exc}")
            return None

    generate = st.button("Generate plan", type="primary", disabled=needs_ack)
    if needs_ack:
        st.caption("Confirm the paid-model warning in the sidebar first.")
    call_slot = st.container()      # request + clock, directly above the spinner
    status_slot = st.container()    # spinner and the outcome message

    if generate:
        req = build_request()
        if req:
            if all(m.exam_date <= req.start_date for m in req.modules):
                st.error("All exams are on or before the start date.")
            else:
                # Built here rather than inside run_with_guardrails so the request
                # can be shown while the call is still in flight.
                planner = get_planner()
                result, viol, info = call_with_panel(
                    call_slot, status_slot, planner, planner.preview_generate(req),
                    "Planning...", lambda p, r: p.generate(r), req)
                st.session_state.last_call = info

                if result:
                    st.session_state.request = req
                    st.session_state.plan = result.plan
                    st.session_state.violations = viol
                    st.session_state.status = {b.id: "todo" for b in result.plan.blocks}
                    st.session_state.accepted = False
                    with status_slot:
                        st.success(f"Plan generated ({result.source}) in "
                                   f"{info['elapsed']:.1f} s. Open the Plan tab to review.")
    elif st.session_state.last_call:
        # Survives the reruns that any widget interaction triggers.
        with call_slot:
            render_call_panel(st.session_state.last_call)

# ----------------------------------------------------------------- plan ---
with tab_plan:
    plan: StudyPlan | None = st.session_state.plan
    req: PlanRequest | None = st.session_state.request
    if not plan or not req:
        st.info("Generate a plan in the Setup tab first.")
    else:
        errs = [v for v in st.session_state.violations if v.severity == "error"]
        warns = [v for v in st.session_state.violations if v.severity == "warning"]
        cols = st.columns(4)
        cols[0].metric("Blocks", len(plan.blocks))
        cols[1].metric("Total hours", f"{plan.total_minutes() / 60:.1f}")
        cols[2].metric("Rule errors", len(errs))
        cols[3].metric("Warnings", len(warns))

        if errs:
            with st.expander(f"{len(errs)} rule error(s)", expanded=True):
                for v in errs:
                    st.error(str(v))
        if warns:
            with st.expander(f"{len(warns)} warning(s)"):
                for v in warns:
                    st.warning(str(v))

        st.write(plan.strategy)
        if plan.change_log:
            with st.expander("Changes", expanded=True):
                for line in plan.change_log:
                    st.write(f"- {line}")
        if plan.risks:
            with st.expander("Risks flagged by the planner"):
                for r in plan.risks:
                    st.write(f"- {r}")

        if not st.session_state.accepted:
            st.warning("Review the plan and accept it before using it. You can edit any block below.")
            if st.button("Accept plan", type="primary"):
                st.session_state.accepted = True
                st.rerun()
        else:
            st.success("Plan accepted.")

        view = st.radio("View", ["Daily", "Weekly", "Edit"], horizontal=True)

        if view == "Daily":
            days = sorted({b.date for b in plan.blocks})
            if days:
                default = date.today() if date.today() in days else days[0]
                day = st.selectbox("Day", days, index=days.index(default),
                                   format_func=lambda d: f"{WEEKDAYS[d.weekday()]} {d}")
                todays = sorted([b for b in plan.blocks if b.date == day],
                                key=lambda b: b.start_minutes())
                for b in todays:
                    status = st.session_state.status.get(b.id, "todo")
                    icon = {"done": "✓", "missed": "✗", "todo": "•"}[status]
                    with st.container(border=True):
                        c1, c2, c3 = st.columns([5, 1, 1])
                        c1.markdown(
                            f"**{icon} {b.start_time} · {b.module}** · {b.duration_minutes} min · "
                            f"{b.block_type} · P{b.priority}  \n{b.topic}  \n"
                            f"<span style='color:#666;font-size:.85em'>{b.rationale}</span>",
                            unsafe_allow_html=True)
                        if c2.button("Done", key=f"d{b.id}"):
                            st.session_state.status[b.id] = "done"
                            st.rerun()
                        if c3.button("Missed", key=f"m{b.id}"):
                            st.session_state.status[b.id] = "missed"
                            st.rerun()

        elif view == "Weekly":
            rows = exporting.plan_to_rows(plan, st.session_state.status)
            st.dataframe(rows, width="stretch", hide_index=True)

            per_day: dict[str, float] = {}
            for r in rows:
                label = f"{r['weekday']} {r['date'][5:]}"
                per_day[label] = per_day.get(label, 0) + r["duration_minutes"] / 60

            st.subheader("Planned load per day")
            st.caption("Total planned study hours per calendar day, regardless of "
                       "completion status. Days without any block are left out entirely.")
            # Named columns rather than a bare dict: they carry into the axis titles
            # *and* the hover tooltip, which x_label/y_label alone would leave as
            # "index"/"value". rows is already sorted by date, and the labels are
            # strings, so sort=False keeps the bars chronological instead of letting
            # Vega sort them alphabetically by weekday name.
            st.bar_chart(
                [{"Date": k, "Study hours": v} for k, v in per_day.items()],
                x="Date", y="Study hours", sort=False, height=200)

        else:  # Edit
            st.caption("Human-in-the-loop: correct anything the model got wrong, then save. "
                       "Saved edits are re-validated against the same rules.")
            editable = [
                {"id": b.id, "date": b.date, "start_time": b.start_time,
                 "duration_minutes": b.duration_minutes, "module": b.module,
                 "block_type": b.block_type, "priority": b.priority, "topic": b.topic}
                for b in sorted(plan.blocks, key=lambda x: (x.date, x.start_minutes()))
            ]
            new_rows = st.data_editor(editable, num_rows="dynamic", width="stretch",
                                      hide_index=True, key="block_editor")
            if st.button("Save edits"):
                try:
                    blocks = []
                    for r in new_rows:
                        d = r["date"] if isinstance(r["date"], date) else \
                            datetime.strptime(str(r["date"])[:10], "%Y-%m-%d").date()
                        original = next((b for b in plan.blocks if b.id == r["id"]), None)
                        blocks.append(StudyBlock(
                            id=str(r["id"]), date=d, start_time=str(r["start_time"]),
                            duration_minutes=int(r["duration_minutes"]), module=str(r["module"]),
                            topic=str(r["topic"]), block_type=str(r["block_type"]),
                            priority=int(r["priority"]),
                            rationale=(original.rationale if original else "Edited by the student.")))
                    st.session_state.plan = plan.model_copy(update={"blocks": blocks})
                    st.session_state.violations = rules.validate_plan(req, st.session_state.plan)
                    st.success("Edits saved and re-validated.")
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Could not save: {exc}")

# ------------------------------------------------------------- progress ---
with tab_progress:
    plan = st.session_state.plan
    req = st.session_state.request
    if not plan or not req:
        st.info("Generate a plan first.")
    else:
        status = st.session_state.status
        done = [b for b in plan.blocks if status.get(b.id) == "done"]
        missed = [b for b in plan.blocks if status.get(b.id) == "missed"]
        total = plan.total_minutes() or 1
        done_min = sum(b.duration_minutes for b in done)

        c = st.columns(4)
        c[0].metric("Completed blocks", f"{len(done)}/{len(plan.blocks)}")
        c[1].metric("Hours done", f"{done_min / 60:.1f}")
        c[2].metric("Missed blocks", len(missed))
        c[3].metric("Completion", f"{100 * done_min / total:.0f}%")
        st.progress(min(1.0, done_min / total))

        st.subheader("Per module")
        by_mod = plan.minutes_by_module()
        for m in req.modules:
            planned = by_mod.get(m.name, 0)
            got = sum(b.duration_minutes for b in done if b.module == m.name)
            days_left = (m.exam_date - date.today()).days
            st.write(f"**{m.name}** · exam in {days_left} d · {got}/{planned} min")
            st.progress(min(1.0, got / planned) if planned else 0.0)

        st.subheader("Replan")
        st.caption("Rebuilds everything from today, folds missed work into the remaining days, "
                   "and protects revision blocks. Past blocks stay frozen.")
        if missed:
            st.write(", ".join(f"{b.id} ({b.module}, {b.date})" for b in missed))
        replan = st.button("Replan from today", type="primary",
                           disabled=not missed or needs_ack)
        if needs_ack:
            st.caption("Confirm the paid-model warning in the sidebar first.")
        replan_call_slot = st.container()    # request + clock, above the spinner
        replan_status_slot = st.container()  # spinner and the outcome message

        if replan:
            today = max(date.today(), req.start_date)
            new_req = req.model_copy(update={
                "missed_block_ids": [b.id for b in missed],
                "completed_block_ids": [b.id for b in done],
                "locked_blocks": [json.loads(b.model_dump_json())
                                  for b in plan.blocks if b.date < today],
            })
            planner = get_planner()
            result, viol, info = call_with_panel(
                replan_call_slot, replan_status_slot, planner,
                planner.preview_replan(new_req, plan, today),
                "Replanning...", lambda p, r: p.replan(r, plan, today),
                new_req, today=today)
            st.session_state.last_replan_call = info

            if result:
                frozen = [b for b in plan.blocks if b.date < today]
                merged = frozen + [b for b in result.plan.blocks if b.date >= today]
                merged.sort(key=lambda b: (b.date, b.start_minutes()))
                st.session_state.plan = result.plan.model_copy(update={"blocks": merged})
                st.session_state.request = new_req
                st.session_state.violations = viol
                for b in merged:
                    st.session_state.status.setdefault(b.id, "todo")
                st.session_state.accepted = False
                with replan_status_slot:
                    st.success(f"Replanned in {info['elapsed']:.1f} s. "
                               "Review the changes in the Plan tab.")
        elif st.session_state.last_replan_call:
            with replan_call_slot:
                render_call_panel(st.session_state.last_replan_call)

# --------------------------------------------------------------- export ---
with tab_export:
    plan = st.session_state.plan
    if not plan:
        st.info("Generate a plan first.")
    else:
        stamp = date.today().isoformat()
        st.download_button("Download CSV", exporting.to_csv(plan, st.session_state.status),
                           file_name=f"study-plan-{stamp}.csv", mime="text/csv")
        html_doc = exporting.to_html(plan, st.session_state.status)
        st.download_button("Download printable HTML (print to PDF)", html_doc,
                           file_name=f"study-plan-{stamp}.html", mime="text/html")
        st.download_button("Download JSON", plan.model_dump_json(indent=2),
                           file_name=f"study-plan-{stamp}.json", mime="application/json")
        st.caption("The HTML file is styled for printing: open it and use the browser's "
                   "print dialog to get a PDF.")
        with st.expander("Preview rows"):
            st.dataframe(exporting.plan_to_rows(plan, st.session_state.status),
                         width="stretch", hide_index=True)
