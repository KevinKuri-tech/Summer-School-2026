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

import pandas as pd
import streamlit as st

import login
from studyplan import exporting, planner as planner_mod, rules, setup_io
from studyplan.planner import (ANTHROPIC_PRESETS, DEFAULT_MODEL, DEFAULT_OPENROUTER_MODEL,
                               OPENROUTER_PRESETS, MockPlanner, PlannerError,
                               available_backends, is_free_model)
from studyplan.schema import (Availability, Chapter, Module, PlanRequest, StudyBlock,
                              StudyPlan)

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

APP_NAME = "Nexora Study"
LOGO = Path(__file__).resolve().parent / "media" / "pictures" / "Logo.png"

st.set_page_config(page_title=APP_NAME, page_icon=str(LOGO) if LOGO.exists() else "📅",
                   layout="wide")

# The base scale lives in .streamlit/config.toml. What follows is only for the
# parts theme settings do not reach. It leans on Streamlit's internal test ids,
# so it is the first thing to check after a Streamlit upgrade: a stale selector
# degrades to the default size rather than breaking anything.
_TYPE_CSS = """
<style>
  /* Primary navigation, so it outweighs body text. */
  .stTabs [data-baseweb="tab"] p { font-size: 1.35rem; font-weight: 600; }
  /* The grid paints its own cells and ignores the theme base size. */
  [data-testid="stDataFrame"], [data-testid="stDataFrame"] * { font-size: 1rem; }
  /* Most of the explanatory copy in this app is captions, and .8rem is too small. */
  [data-testid="stCaptionContainer"] p { font-size: .95rem; }
  [data-testid="stExpander"] summary p { font-size: 1.05rem; font-weight: 600; }
  [data-testid="stMetricValue"] { font-size: 2rem; }
  [data-testid="stMetricLabel"] p { font-size: 1rem; }
  [data-testid="stWidgetLabel"] p { font-size: 1rem; }
</style>
"""
st.markdown(_TYPE_CSS, unsafe_allow_html=True)

# The gate. Everything below this line — the chrome, the sidebar, the tabs, and
# every widget in them — belongs to a signed-in session: when there is none this
# call draws the login screen and stops the script, so there is nothing further
# down the page to reach past it.
USER = login.require_login()

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
    starter_modules = [
        {"name": "Statistics", "exam_date": date.today() + timedelta(days=12),
         "difficulty": 4, "confidence": 2, "estimated_hours": 14.0},
        {"name": "Microeconomics", "exam_date": date.today() + timedelta(days=9),
         "difficulty": 3, "confidence": 3, "estimated_hours": 9.0},
        {"name": "Databases", "exam_date": date.today() + timedelta(days=14),
         "difficulty": 3, "confidence": 4, "estimated_hours": 7.0},
    ]
    # Two separate entries on purpose: `module_seed` is what the editor is fed and
    # has to stay unchanged across reruns (see the data_editor in Step 1), while
    # `modules` carries the edited rows that the rest of the app reads.
    ss.setdefault("module_seed", [dict(m) for m in starter_modules])
    ss.setdefault("modules", [dict(m) for m in starter_modules])

    # Chapters repeat that split once per module, keyed by module name. Only the
    # focused module's editor is ever rendered, so `chapters` is the only place
    # the other modules' rows survive: Streamlit discards widget state for
    # widgets that a rerun did not draw.
    ss.setdefault("chapter_seed", {m["name"]: [] for m in starter_modules})
    ss.setdefault("chapters", {m["name"]: [] for m in starter_modules})
    ss.setdefault("chapter_focus", starter_modules[0]["name"])
    ss.setdefault("_prev_chapter_focus", starter_modules[0]["name"])
    ss.setdefault("_module_names", [m["name"] for m in starter_modules])
    ss.setdefault("_setup_token", None)   # upload already applied

    # Step 2 and the advanced settings are keyed so that loading a setup file can
    # drive them. A keyed widget's session value outranks any value= argument, so
    # the defaults are seeded here and the widgets below pass key= only.
    for i, default in enumerate([2.0, 2.0, 2.0, 2.0, 1.5, 4.0, 3.0]):
        ss.setdefault(f"h{i}", default)
    ss.setdefault("plan_start", date.today())
    ss.setdefault("horizon", 14)
    ss.setdefault("min_len", 30)
    ss.setdefault("max_len", 90)
    ss.setdefault("day_start", "09:00")
    ss.setdefault("blackout_raw", "")
    ss.setdefault("preferences",
                  "Prefer mornings. No study after 21:00. "
                  "Wednesday evenings are blocked by work.")


init_state()


CHAPTER_COLUMNS = ("name", "weight", "confidence")


def chapter_key(module_name: str) -> str:
    return f"chapters::{module_name}"


def chapter_frame(rows: list[dict]) -> pd.DataFrame:
    """A module's chapter rows as a typed frame.

    The module table can be fed plain dicts because it always has rows to infer
    a shape from. A module with no chapters yet has none, and an editor built
    from an empty list of dicts renders with no columns at all — nothing to type
    into, so the table could never be filled. Declaring the columns and their
    dtypes here keeps the empty grid usable, and keeps Confidence nullable so
    "leave it blank to use the module's" stays expressible.
    """
    frame = pd.DataFrame(list(rows), columns=list(CHAPTER_COLUMNS))
    return frame.astype({"name": "string", "weight": "Int64", "confidence": "Int64"})


def frame_to_rows(frame: pd.DataFrame) -> list[dict]:
    """Back to plain rows, with pandas' missing values normalised to None."""
    return [{c: (None if pd.isna(rec.get(c)) else rec.get(c)) for c in CHAPTER_COLUMNS}
            for rec in frame.to_dict("records")]


def apply_editor_delta(rows: list[dict], delta: dict | None) -> list[dict]:
    """Fold a data_editor's pending edits into the rows it was seeded with.

    Only needed when an editor is about to disappear without having returned:
    clicking a different module blurs the focused cell, so the edit and the new
    selection arrive in the same message and the editor is never re-rendered to
    hand its value back. Streamlit still holds the delta under the widget key at
    callback time, so it is applied here instead.
    """
    out = [dict(r) for r in rows]
    if not delta:
        return out
    for idx, changes in (delta.get("edited_rows") or {}).items():
        i = int(idx)
        if 0 <= i < len(out):
            out[i].update(changes)
    for added in (delta.get("added_rows") or []):
        if added:
            out.append({c: added.get(c) for c in CHAPTER_COLUMNS})
    for i in sorted((delta.get("deleted_rows") or []), reverse=True):
        if 0 <= int(i) < len(out):
            out.pop(int(i))
    return out


def switch_chapter_focus() -> None:
    """Hand the chapter table over from one module to the next.

    Runs before the rerun that redraws the table, which is what makes the switch
    safe in both directions: the outgoing module's edits are captured while its
    widget state still exists, and the incoming module is re-seeded with its
    stored rows so the editor mounts on current data.

    Re-seeding changes that editor's widget identity, which is exactly what
    Step 1 warns against for the module table. It is fine here only because it
    happens in a callback: the new identity and the new data reach the browser
    in the same render, so there is no rerun in which the client posts against a
    stale id. Between switches the seed is left alone.
    """
    ss = st.session_state
    prev, new = ss.get("_prev_chapter_focus"), ss.chapter_focus
    delta = ss.get(chapter_key(prev)) if prev else None
    if delta and prev in ss.chapter_seed:
        # Only when there is something pending. `chapters[prev]` already holds
        # what the editor returned on the last completed rerun, so overwriting
        # it unconditionally would throw that away on every switch. The delta is
        # cumulative since the widget mounted, so applying it to the seed
        # reproduces the editor's current value rather than adding to it.
        ss.chapters[prev] = apply_editor_delta(ss.chapter_seed[prev], delta)
    if new is not None:
        ss.chapter_seed[new] = [dict(r) for r in ss.chapters.get(new, [])]
        ss.pop(chapter_key(new), None)
    ss._prev_chapter_focus = new


def reset_chapter_editor(name: str, rows: list[dict]) -> None:
    """Replace a module's chapter rows from outside the editor.

    Both halves are required, and this is the recipe the module table's comment
    below describes: a new seed alone would leave the old edits stored under the
    widget key and Streamlit would re-apply them on top.
    """
    st.session_state.chapter_seed[name] = [dict(r) for r in rows]
    st.session_state.chapters[name] = [dict(r) for r in rows]
    st.session_state.pop(chapter_key(name), None)


def add_syllabus(name: str) -> None:
    """Append a parsed paste to a module's chapters, skipping ones already there.

    A callback rather than a branch after the button, because it has to empty
    the paste box: session state for a widget cannot be assigned once that
    widget has been instantiated, and callbacks run before that happens.
    """
    ss = st.session_state
    existing = list(ss.chapters.get(name, []))
    have = {str(r.get("name") or "").strip().casefold() for r in existing}
    fresh = [r for r in setup_io.parse_syllabus(ss.get(f"syllabus::{name}") or "")
             if r["name"].casefold() not in have]
    if fresh:
        reset_chapter_editor(name, existing + fresh)
    ss[f"syllabus::{name}"] = ""


def sync_chapter_owners() -> list[str]:
    """Keep the chapter store aligned with the module table. Returns the names.

    Modules are joined to their chapters by name, so a rename has to carry the
    chapters (and the selection) across or the student's work would silently
    detach. Renames are detected by position, since that is the only thing that
    survives an edit to the name cell itself.
    """
    ss = st.session_state
    names = [str(m.get("name") or "").strip() for m in ss.modules]
    names = [n for n in names if n]
    old = ss._module_names

    for i, new in enumerate(names):
        if i < len(old) and old[i] != new and old[i] not in names:
            was = old[i]
            ss.chapter_seed[new] = ss.chapter_seed.pop(was, [])
            ss.chapters[new] = ss.chapters.pop(was, [])
            ss.pop(chapter_key(was), None)
            if ss.get("chapter_focus") == was:
                ss.chapter_focus = new
            if ss.get("_prev_chapter_focus") == was:
                ss._prev_chapter_focus = new

    for name in names:
        ss.chapter_seed.setdefault(name, [])
        ss.chapters.setdefault(name, [])
    for gone in [n for n in ss.chapters if n not in names]:
        ss.chapter_seed.pop(gone, None)
        ss.chapters.pop(gone, None)
        ss.pop(chapter_key(gone), None)

    ss._module_names = names
    return names


def chapters_for(name: str) -> list[Chapter]:
    """The stored rows for one module as validated Chapters, blanks dropped."""
    out = []
    for row in st.session_state.chapters.get(name, []):
        title = str(row.get("name") or "").strip()
        if not title:
            continue
        confidence = row.get("confidence")
        out.append(Chapter(
            name=title,
            weight=int(row.get("weight") or 3),
            confidence=int(confidence) if confidence not in (None, "") else None,
        ))
    return out


def apply_setup(req: PlanRequest) -> None:
    """Drive every Setup widget from a loaded file, then let the rerun redraw."""
    ss = st.session_state
    ss.module_seed = [{"name": m.name, "exam_date": m.exam_date, "difficulty": m.difficulty,
                       "confidence": m.confidence, "estimated_hours": m.estimated_hours}
                      for m in req.modules]
    ss.modules = [dict(m) for m in ss.module_seed]
    ss.pop("module_editor", None)

    for name in list(ss.chapters):
        ss.pop(chapter_key(name), None)
    ss.chapter_seed, ss.chapters = {}, {}
    for m in req.modules:
        reset_chapter_editor(m.name, [c.model_dump() for c in m.chapters])
    ss._module_names = [m.name for m in req.modules]
    ss.chapter_focus = ss._prev_chapter_focus = req.modules[0].name

    av = req.availability
    for i in range(7):
        ss[f"h{i}"] = float(av.hours_per_weekday.get(i, 0.0))
    ss.plan_start = req.start_date
    ss.horizon = req.horizon_days
    ss.min_len = int(av.min_session_minutes)
    ss.max_len = int(av.max_session_minutes)
    ss.day_start = av.day_start
    ss.blackout_raw = ", ".join(d.isoformat() for d in av.blackout_dates)
    ss.preferences = req.preferences


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
    font-size: 1rem; font-variant-numeric: tabular-nums; color: #808495;
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
                    st.iframe(_CLOCK_HTML, height=28)
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
    # Who is signed in, and the way out. Top of the sidebar because these
    # accounts are shared: on a lab machine the first question is whose session
    # this is, and signing out has to be findable without reading the page.
    # Tertiary, so the label fits on one line in a sidebar column: the boxed
    # variants reserve enough padding to wrap "Sign out" in two.
    _who, _out = st.columns([1.5, 1], vertical_alignment="center")
    _who.markdown(f"Signed in as **{USER}**")
    if _out.button("Sign out", type="tertiary", width="stretch",
                   help="Ends the session and clears everything typed into it"):
        login.sign_out()
    st.divider()

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
    # Two required steps first, everything with a working default folded away
    # behind the expander below. The expander body still executes on every rerun,
    # so build_request() sees those values whether or not it was ever opened.

    # Nothing is stored server-side, so a refresh costs the student everything
    # they typed. The load half has to run before any Setup widget is drawn,
    # since it writes their session values; the save half needs build_request(),
    # which is defined further down, so it fills a slot reserved here.
    with st.container(border=True):
        io_load, io_save = st.columns(2, vertical_alignment="bottom")
        with io_load:
            upload = st.file_uploader("Load setup (JSON)", type=["json"], key="setup_upload")
            if upload is not None:
                # The uploader hands the same file back on every rerun, so the
                # import is keyed to the file's identity and applied once.
                token = (upload.name, upload.size)
                if token != st.session_state._setup_token:
                    st.session_state._setup_token = token
                    try:
                        apply_setup(setup_io.setup_from_json(upload.getvalue().decode("utf-8")))
                    except ValueError as exc:
                        st.error(f"Could not load that setup: {exc}")
                    except UnicodeDecodeError:
                        st.error("Could not load that setup: the file is not UTF-8 text.")
                    else:
                        st.rerun()
            else:
                st.session_state._setup_token = None
        save_slot = io_save.container()

    with st.container(border=True):
        st.subheader("Step 1 · Your modules")
        st.caption("Three to six modules. Confidence 1 means you have not started, 5 means solid.")
        # The editor is fed `module_seed`, never its own output. With
        # num_rows="dynamic" Streamlit derives the widget identity from the
        # serialized data itself (only "fixed" editors get a schema-based
        # identity), so assigning the result back into the editor's input would
        # change that identity on the following rerun. The browser posts each
        # edit against the id it last received, which would then be one run
        # stale: the server registers it as a new widget with no pending edits,
        # drops the edit, and the grid remounts on the unedited data — the cell
        # snaps back to its old value until you type it a second time.
        # Keeping the input fixed keeps the id stable, and Streamlit re-applies
        # the accumulated edits (added and deleted rows included) on every rerun.
        # To reset the table programmatically, replace `module_seed` and delete
        # the "module_editor" key so the stored edits go with it.
        edited = st.data_editor(
            st.session_state.module_seed,
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

        # ---- chapters ------------------------------------------------------
        # Optional throughout: leave every table empty and the request, the
        # planner and the rule engine behave exactly as they did before this
        # section existed.
        st.divider()
        st.markdown("**Chapters** · optional")
        st.caption("Splitting a module into chapters turns one hours guess into several, "
                   "and gives every study block a real topic instead of a generated one. "
                   "Weight is relative size, not hours: 1 is short, 5 is the monster.")

        module_names = sync_chapter_owners()
        if not module_names:
            st.caption("Name a module above to add chapters to it.")
        else:
            # A keyed radio outranks any index= argument, so the selection is
            # corrected in state before the widget renders (same reason as the
            # backend radio in the sidebar). A radio and not an expander or a
            # nested tab: both of those reset or collapse on the reruns that
            # editing a cell triggers, which is exactly the behaviour a table
            # full of half-typed input must not have.
            if st.session_state.chapter_focus not in module_names:
                st.session_state.chapter_focus = module_names[0]
                st.session_state._prev_chapter_focus = module_names[0]
            focus = st.radio("Chapters for", module_names, horizontal=True,
                             key="chapter_focus", on_change=switch_chapter_focus,
                             label_visibility="collapsed")

            # Fed the frozen seed, never its own output — see the module editor
            # above for what happens otherwise. The seed only changes in
            # switch_chapter_focus() and reset_chapter_editor().
            st.session_state.chapters[focus] = frame_to_rows(st.data_editor(
                chapter_frame(st.session_state.chapter_seed[focus]),
                num_rows="dynamic",
                width="stretch",
                column_config={
                    "name": st.column_config.TextColumn("Chapter", required=True),
                    "weight": st.column_config.NumberColumn(
                        "Weight", min_value=1, max_value=5, step=1,
                        help="Relative size. 1 = short, 5 = the monster."),
                    "confidence": st.column_config.NumberColumn(
                        "Confidence", min_value=1, max_value=5, step=1,
                        help="Leave empty to use the module's own confidence."),
                },
                key=chapter_key(focus),
            ))

            # Same idea as the supply/demand line in Step 2: show the
            # consequence of the numbers immediately, so a wrong weight is
            # obvious before a plan is ever generated.
            row = next((m for m in st.session_state.modules if m.get("name") == focus), None)
            chapters = chapters_for(focus)
            if chapters and row:
                try:
                    split = Module(**{**row, "chapters": chapters}).chapter_minutes()
                except Exception:  # noqa: BLE001 - incomplete module row, caption is optional
                    split = {}
                if split:
                    st.caption(f"{focus} · {sum(split.values()) / 60:.1f} h over "
                               f"{len(split)} chapters · "
                               + " · ".join(f"{n} {m} min" for n, m in split.items()))

            paste_col, add_col = st.columns([4, 1], vertical_alignment="bottom")
            pasted = paste_col.text_area(
                "Paste a syllabus", height=80, key=f"syllabus::{focus}",
                placeholder="1. Descriptive statistics (pp. 1-20)\n2) Probability .......... 21\n"
                            "Ch. 3 - Distributions",
                help="Numbering and page numbers are stripped. Deliberately not inside an "
                     "expander: one that collapses mid-edit is the behaviour this avoids.")
            found = setup_io.parse_syllabus(pasted or "")
            add_col.button(f"Add {len(found)}" if found else "Add", width="stretch",
                           disabled=not found, key=f"add_syllabus::{focus}",
                           on_click=add_syllabus, args=(focus,))

    with st.container(border=True):
        st.subheader("Step 2 · Your week")
        st.caption("Hours you can realistically study on each weekday.")
        # Every widget from here to the end of the advanced settings is keyed and
        # takes no value= argument: a keyed widget's session value wins anyway,
        # and passing both would make loading a setup file a no-op. Defaults are
        # seeded in init_state().
        hours = {}
        cols = st.columns(7)
        for i, col in enumerate(cols):
            with col:
                hours[i] = st.number_input(WEEKDAYS[i], min_value=0.0, max_value=12.0,
                                           step=0.5, key=f"h{i}")
        st.divider()
        wc1, wc2 = st.columns([1, 2], vertical_alignment="center")
        start = wc1.date_input("Plan starts", key="plan_start")
        horizon = wc2.slider("Horizon (days)", 10, 14, key="horizon")

        # Immediate payoff for filling the two steps in, and it surfaces an
        # over-committed plan before a single API call is spent on it.
        weekly = sum(hours.values())
        supply = weekly * horizon / 7
        demand = sum(float(m.get("estimated_hours") or 0)
                     for m in st.session_state.modules if m.get("name"))
        verdict = "" if supply >= demand else "  ·  ⚠ that is less time than the modules need."
        st.caption(f"About {supply:.1f} h available over {horizon} days · "
                   f"your modules need about {demand:.1f} h.{verdict}")

    with st.expander("⚙ Advanced settings — session length, blackout dates, preferences"):
        cc1, cc2 = st.columns(2)
        min_len = cc1.number_input("Min session (min)", 20, 90, step=5, key="min_len")
        max_len = cc2.number_input("Max session (min)", 30, 240, step=15, key="max_len")
        day_start = st.text_input("Earliest start time", key="day_start")
        blackout_raw = st.text_input("Blackout dates (YYYY-MM-DD, comma separated)",
                                     key="blackout_raw")
        preferences = st.text_area("Preferences and constraints", height=110, key="preferences")

    blackout = []
    for token in [t.strip() for t in blackout_raw.split(",") if t.strip()]:
        try:
            blackout.append(datetime.strptime(token, "%Y-%m-%d").date())
        except ValueError:
            # Reported here rather than inside build_request(), which now also
            # runs quietly for the save button: a typo should be visible as soon
            # as it is typed, not only once a plan is requested.
            st.warning(f"Ignored blackout date '{token}'.")

    def build_request(quiet: bool = False) -> PlanRequest | None:
        """Assemble the request, or None if the input is not usable yet.

        `quiet` suppresses the messages, for callers that only need to know
        whether a valid request exists right now: the save button is rendered on
        every rerun and must not narrate the same complaint each time.
        """
        fail = (lambda _msg: None) if quiet else st.error
        try:
            rows = [m for m in st.session_state.modules if m.get("name")]
            names = [str(m["name"]).strip() for m in rows]
            if len(set(names)) != len(names):
                # Chapters hang off the module name, so duplicates would make two
                # modules share one chapter table.
                fail("Module names must be unique.")
                return None
            mods = [Module(**{**m, "exam_date": (m["exam_date"] if isinstance(m["exam_date"], date)
                                                 else datetime.strptime(str(m["exam_date"]), "%Y-%m-%d").date()),
                              "chapters": chapters_for(str(m["name"]).strip())})
                    for m in rows]
            if not 1 <= len(mods) <= 6:
                fail("Enter between 1 and 6 modules.")
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
            fail(f"Invalid input: {exc}")
            return None

    # Fills the slot reserved at the top of the tab, now that the request can be
    # built. Inputs only: the generated plan is exported from the Export tab.
    savable = build_request(quiet=True)
    with save_slot:
        st.download_button(
            "Save setup (JSON)",
            setup_io.setup_to_json(savable) if savable else "",
            file_name=f"study-setup-{date.today().isoformat()}.json",
            mime="application/json", disabled=savable is None, width="stretch")
        st.caption("Modules, chapters, availability and preferences."
                   if savable else "Fill in Steps 1 and 2 to save.")

    generate = st.button("Generate plan", type="primary", disabled=needs_ack,
                         width="stretch")
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
                            f"<span style='color:#666;font-size:.95em'>{b.rationale}</span>",
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
