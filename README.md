# AI Study-Plan Optimizer

Ships as **Nexora Study** in the UI.

Turns exam dates, workload estimates and real availability into a 10 to 14 day study
schedule, and rebuilds that schedule when the student misses sessions.

The AI does planning and explanation only. It never tutors, never answers subject
questions, and never decides alone what reaches the student: every plan is checked
by a deterministic rule engine and is editable by the student before use.

## Quickstart

```bash
uv sync
cp .env.example .env                     # then put your key in it, optional
uv run streamlit run app.py
```

`.env` is read on import of the `studyplan` package, so the app, the tests and
the eval harness all see it. An already-exported variable wins over the file, so
`export ANTHROPIC_API_KEY=sk-ant-...` still works if you prefer that.

Without an API key the app runs the deterministic baseline planner, so the UI is
fully demoable at zero cost.

```bash
uv run pytest                       # rule engine, chapters and Setup-tab UI (57 tests)
uv run python check_backend.py      # is the configured backend actually reachable?
uv run python eval/run_eval.py      # 15 cases, baseline planner, free
uv run python eval/run_eval.py --live --repair    # same cases against the model
```

## Scope

In: manual input for 1 to 6 modules, optional chapters per module (typed or
pasted from a syllabus), saving and reloading the setup as JSON, AI plan
generation in strict JSON, daily and weekly view, per-block edit, "I missed
this" plus replanning, progress screen, CSV / JSON / printable HTML export,
evaluation on 15 cases.

Out, deliberately: authentication, a server-side database, mobile app, calendar
sync, spaced-repetition engine, chatbot.

Chapters were originally out of scope and are now in, for one reason: without
them `topic` is invented. The system prompt forbids inventing topics while
supplying none, so the model had no choice, and the baseline planner wrote
`"Statistics: learn session"`. Chapters make the topic a lookup instead of a
guess, and therefore checkable — see `unknown_chapter` and `chapter_uncovered`
below. Reading a syllabus *file* is still out; the paste box is a parser, not an
ingestion pipeline. Setup files are the same story: a JSON download and upload,
not a database.

## Architecture

```
app.py                 Streamlit UI (setup, plan, progress, export)
check_backend.py       Connectivity check: .env -> key -> model -> schema-valid reply
studyplan/schema.py    Pydantic input models + hand-written output JSON schema
studyplan/prompts.py   System prompt, generation, replanning, repair prompts
studyplan/planner.py   Backends (Anthropic, OpenRouter) + deterministic baseline
studyplan/rules.py     Guardrails: 12 checks + auto-repair
studyplan/exporting.py CSV and print-ready HTML
studyplan/setup_io.py  Setup file save/load + syllabus paste parser
eval/                  Cases and harness
```

One frontend, one thin core, one AI call per action (two if a rule violation
triggers a repair round). No database: state lives in the Streamlit session and
is exported as CSV, JSON or HTML.

Two different JSON files, deliberately kept apart. `setup_io.py` writes the
*input* (a versioned `PlanRequest`) so a refresh is recoverable; `exporting.py`
writes the *output*. Reusing `PlanRequest` as the file format means there is one
contract to keep in sync, and validation comes free — `pydantic.ValidationError`
is a `ValueError`, so a hand-edited file with a 40 day horizon is rejected with
the field name already in the message.

### The chapter table

Chapters are per module, but there is only ever one table on screen, chosen by a
radio. That costs some Streamlit footwork, and it is worth writing down because
all of it is load-bearing.

`st.data_editor` with `num_rows="dynamic"` derives its widget identity from the
serialized input data. Feed it its own output and the identity changes every
rerun; the browser then posts each edit against an id that is already stale, the
server registers a new widget with no pending edits, and the cell snaps back
until you type it a second time. So both editors are fed a frozen seed
(`module_seed`, `chapter_seed[module]`) and never their own return value.

Only the focused module's editor is rendered, and Streamlit discards widget state
for widgets a rerun did not draw. So the rows live in a plain `chapters[module]`
dict, written from the editor's return value on every rerun, and switching module
runs an `on_change` callback that re-seeds the incoming editor from it. Re-seeding
does change that editor's identity, which is exactly what the paragraph above
warns against — it is safe only because a callback runs *before* the render, so
the new id and the new data reach the browser together and there is no rerun in
which the client can post against the old one.

One gap remains: clicking another module while a cell is still focused blurs the
cell, so the edit and the new selection arrive in the same message and the
outgoing editor is never re-rendered to hand its value back. The same callback
therefore folds that editor's still-live delta into `chapters[module]` first —
but only when a delta is actually present, since `chapters[module]` already holds
the last returned value and overwriting it unconditionally would discard it.

None of this is theoretical: `tests/test_chapters.py` drives the real app through
`AppTest` and fails on each of these paths individually if the corresponding
mechanism is removed. The radio is also not an accident — an expander collapses
and a nested tab resets on the reruns that editing a cell triggers.

### The input contract

```jsonc
{
  "start_date": "2026-03-02", "horizon_days": 14,          // 10..14
  "modules": [{                                             // 1..6
    "name": "Statistics", "exam_date": "2026-03-14",
    "difficulty": 4, "confidence": 2,                       // 1..5 each
    "estimated_hours": 14.0,
    "chapters": [{                                          // optional, 0..40
      "name": "Regression",
      "weight": 4,                                          // 1..5, relative size
      "confidence": 2                                       // 1..5, null = the module's
    }]
  }],
  "availability": {
    "hours_per_weekday": {"0": 2.0, "...": 0.0, "6": 3.0},  // 0 = Monday
    "blackout_dates": ["2026-03-07"],
    "min_session_minutes": 30, "max_session_minutes": 90,
    "day_start": "09:00"
  },
  "preferences": "Prefer mornings. No study after 21:00.",  // free text
  // replanning only:
  "completed_block_ids": [], "missed_block_ids": [], "locked_blocks": []
}
```

`preferences` is the only free-text field. It is passed to the model verbatim and
is *not* enforced by any rule, so it is advisory: the model usually honours it,
nothing guarantees it.

`chapters` is optional everywhere. Leave it empty and the planner, the prompt and
the rule engine behave exactly as they did before the field existed. `weight` is
relative size, never absolute time: `estimated_hours` stays the single source of
truth for a module's total, and `Module.chapter_minutes()` distributes it by
`weight * (6 - confidence)` using largest-remainder rounding, so the parts always
sum to exactly the whole. That is why the two fields can never contradict each
other, and why a student who has no idea how long a chapter takes can still rank
them against each other.

### The AI call

The Anthropic backend calls `client.messages.stream(...)` (not `messages.create`)
with `output_config.format = {type: json_schema, schema: PLAN_JSON_SCHEMA}` passed
through `extra_body`, then blocks on `get_final_message()`. Streaming is not for
progressive rendering — the SDK refuses a non-streaming request whose `max_tokens`
(15 000 here) could outlast a 10 minute HTTP connection. `extra_body` keeps the
call working on SDK versions that predate the typed `output_config` parameter.

Anthropic constrains decoding to the schema, so the response is always syntactically
valid JSON in the shape below; it is still parsed and validated with Pydantic, and
still checked by the rule engine
([docs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs)).

```jsonc
{
  "plan_start": "2026-03-02", "plan_end": "2026-03-15",
  "strategy": "...",
  "blocks": [{
    "id": "b001", "date": "2026-03-02", "start_time": "09:00",
    "duration_minutes": 90, "module": "Statistics", "topic": "Hypothesis testing",
    "block_type": "learn|practice|revision|buffer", "priority": 4,
    "rationale": "..."
  }],
  "risks": ["..."], "change_log": ["..."]
}
```

The schema is written by hand rather than derived from Pydantic: every field is
required and `additionalProperties` is false, which keeps the compiled grammar
small and avoids `$ref` indirection. Structured outputs constrain shape, not
casing, so `block_type` is lower-cased on parse (anything unrecognised falls back
to `learn`) and `priority` is clamped to 1..5.

### Backends

Two live backends plus the baseline. Both provider keys may be set at once; the
sidebar picks between them, and OpenRouter is preselected when its key is present.
With no key at all only the baseline is offered.

| Backend | Default model | Override | Schema enforcement |
| --- | --- | --- | --- |
| OpenRouter | `~anthropic/claude-haiku-latest` | `STUDYPLAN_OPENROUTER_MODEL` (or legacy `STUDYPLAN_MODEL`) | `response_format: json_schema` + `strict`, per endpoint |
| Anthropic | `claude-sonnet-5` | `STUDYPLAN_ANTHROPIC_MODEL` | grammar-constrained by the decoder |
| Baseline | — | — | n/a, constructed in Python |

The two model variables are deliberately separate: a single shared override would
send an OpenRouter slug to the Anthropic endpoint, which 404s.

OpenRouter sets `provider.require_parameters: true` so routing is limited to
endpoints that honour the schema. If OpenRouter still answers 400/404 because no
endpoint supports `json_schema`, the planner retries once in plain JSON mode with
the schema demoted to a prompt instruction, strips markdown fences from the reply,
and flags the degradation in the UI. A 404 caused by the account's data policy
(common with `:free` slugs) is detected separately and reported instead of retried.

**Model picker.** On the Anthropic backend the sidebar calls `GET /v1/models` and
lists what that key may actually call, filtering out models that report no
structured-output support; the result is cached for 10 minutes and refreshable
with the ↻ button. If the endpoint is unreachable it falls back to a built-in
preset list. On OpenRouter any slug can be typed in.

Structured outputs are available on `claude-opus-5`, `claude-sonnet-5`,
`claude-opus-4-8`, `claude-fable-5` and `claude-haiku-4-5` (and legacy
`claude-opus-4-5` / `claude-opus-4-1`); `claude-sonnet-4-6` is not, so it cannot
be used here.

**Cost.** A plan call is roughly 1k input tokens (system + request + schema) and
2 to 5k output tokens for a 25 to 35 block plan. On `claude-sonnet-5`
($3/$15 per M tokens) that is about $0.05 per plan; on `claude-haiku-4-5`
($1/$5) about $0.02. A $10 budget covers a POC comfortably. The sidebar shows a
running call and token count, and a one-time confirmation gate appears before any
paid model can be used — a model is treated as free only when it is the baseline
or an OpenRouter `:free` slug.

**Request transparency.** Before each call the UI renders the exact HTTP body it
is about to send (user message, system prompt, full JSON), the endpoint, the
`max_tokens`, how hard the schema is enforced, and a live "waiting n.n s" counter
that ticks inside an iframe because Streamlit cannot repaint while the call blocks
the script thread. After the call the panel is rewritten with what was really sent
(which differs from the preview if a repair round or the JSON-mode fallback ran).

### Guardrails and human-in-the-loop

Schema validity is not correctness. A schema-valid plan can still put a session
after the exam or overbook a Tuesday. `rules.py` runs 12 checks producing 13
violation codes, deterministically, on every plan:

| Code | Severity | Check |
| --- | --- | --- |
| `unknown_module` | error | Module name was never provided (confabulation) |
| `after_exam` | error | Block on or after that module's exam |
| `outside_horizon` | error | Block outside the plan window |
| `blackout` | error | Day is unavailable but has time scheduled |
| `over_capacity` | error | Day exceeds its minute budget |
| `bad_duration` | error | Session outside min/max length |
| `overlap` | error | Two blocks on one day collide |
| `no_coverage` | error | A module got no time at all |
| `past_edit` | error | Replan rewrote a day that already happened |
| `no_late_revision` | warning | No revision in the last 3 days before the exam |
| `under_allocated` | warning | Module got under 40 percent of its estimate |
| `unknown_chapter` | warning | Block topic is not a chapter the student listed |
| `chapter_uncovered` | warning | A listed chapter got no block at all |

Only errors block; warnings are surfaced and left to the student. Three rules in
the system prompt are intentionally *not* enforced by the engine — the 15 minute
gap between blocks, the ~10 percent buffer allowance, and the free-text
preferences — because they are preferences, not correctness.

The two chapter checks are inert unless at least one module has chapters, so a
chapterless request is scored exactly as it was before chapters existed. They are
warnings and never errors on purpose: `errors()` feeds the repair round and
`autorepair()`, and neither should delete otherwise-legal study time over a
naming mismatch. Matching is done on a normalised form, so the model re-decorating
`Distributions` as `Ch. 3 - Distributions` is not reported as off-syllabus.

Failure path, in order:

1. Generate, validate.
2. If there are errors: one corrective round trip to the model with the violation
   list. The repaired plan is only kept if it has strictly fewer errors than the
   original, so a worse retry cannot win.
3. If errors remain: deterministic removal of every still-illegal block, lowest
   priority first, with each removal written into `change_log` and surfaced as a
   warning in the UI.

The plan then lands in the Plan tab marked "not accepted" with an **Accept**
button. Acceptance is an acknowledgement, not a lock: editing, progress tracking,
replanning and export all work before and after it. Every block is editable in the
Edit view, and saved edits are re-validated against the same rules.

This mirrors the NIST Generative AI Profile's information-integrity and
confabulation guidance: a generative system is treated as a proposer, and a human
plus a rule engine decide.

### Replanning

Marking blocks missed and pressing **Replan from today** sends the previous plan,
the missed and completed ids, and the frozen past (`locked_blocks`) to the model.
The model is told to fold missed content into remaining days rather than re-adding
it verbatim, to cut the lowest-priority material when it no longer fits, to protect
revision blocks, and to list every change. The response is validated with
`today` set, which activates the `past_edit` rule, and blocks before today are
merged back from the previous plan untouched.

The baseline planner's replan is simpler: it sums the missed minutes per module,
adds them to that module's remaining workload, and regenerates from today.

### Export

CSV (semicolon-delimited, one row per block, includes status), JSON (the raw
`StudyPlan`), and a self-contained printable HTML file styled for the browser's
print-to-PDF dialog. Every HTML export carries the line *"Generated with AI
assistance and reviewed by the student. Verify dates against your official exam
schedule."*

## Evaluation

15 cases in `eval/cases.json`: happy path, six modules, an exam in two days, zero
weekend availability, four consecutive blackout days, an infeasible workload, two
exams on the same day, 25–45 minute sessions, 60–180 minute sessions, an easy
module with low confidence versus a hard one with high confidence, one exam
tomorrow plus one at the end of the horizon, the minimum 10 day horizon, and three
replanning scenarios (2 missed blocks, 5 missed blocks, missed work two days
before the exam). The rule engine is the ground truth, so scoring needs no
labelling.

Reported per run: schema validity, hard errors, warnings, violation codes, block
count, capacity use, module coverage, whether a repair round ran, latency, tokens,
and any failure string. Results land in `eval/results.csv` (gitignored).

The baseline planner scores 15/15 rule-clean with 4 `under_allocated` warnings on
the deliberately infeasible cases, and is the bar the model has to match before it
wins on quality of the allocation. `--live` uses whichever backend the environment
prefers; there is no per-run backend flag, so unset a key to force the other one.

## Troubleshooting

`uv run python check_backend.py` isolates each layer between `.env` and a
schema-valid model response — file found, key shape, backend selected, key
accepted, model slug exists, structured outputs supported, and one real (tiny)
round trip against a 1-field schema. It prints a masked key fingerprint only, and
exits non-zero on the first failing layer with the specific fix for it (wrong key
prefix, OpenRouter privacy settings for `:free` slugs, insufficient credits, rate
limit, model that ignores the schema).

## Why an LLM rather than a solver

Scheduling blocks into a fixed calendar under hard capacity constraints is a
textbook optimisation problem. A CP-SAT or greedy solver would be faster,
free, deterministic and provably optimal against whatever objective you write —
and the baseline planner in `planner.py` is exactly that, in 90 lines, scoring
15/15 on the eval set. So the honest framing is not "AI instead of maths"; it is
"AI on top of maths, where the maths has nothing to optimise against."

What the model adds:

- **Topic decomposition.** A solver allocates minutes to *Statistics*. The model
  allocates them to *hypothesis testing*, then *confidence intervals*, then
  *revision of both* — and orders them so prerequisites come first. Nothing in the
  input encodes that structure; it comes from the model's prior knowledge of what
  a statistics course contains.
- **Unstructured constraints.** "Wednesday evenings are blocked by work", "prefer
  mornings", "I always crash after 21:00" are consumed as text. Each one would
  otherwise need its own input field, its own UI control and its own constraint
  term.
- **Judgement under an infeasible objective.** When 55 hours of work must fit into
  12 hours of capacity, there is no optimum — only a defensible triage. The model
  cuts the lowest-value material, says which modules it sacrificed and why, and
  writes that into `risks`. A solver returns INFEASIBLE.
- **Replanning as a semantic operation.** "Fold the missed content into the
  remaining days without simply re-adding those blocks" is a content-level
  instruction, not a constraint. The solver's version — carry the minutes forward
  — is measurably cruder.
- **Explanations that are part of the product.** `strategy`, `rationale`, `risks`
  and `change_log` are the difference between a timetable a student follows and
  one they ignore. They are generated in the same call, for free.

What the model costs: latency (seconds instead of milliseconds), money, and
non-determinism — the same input can produce different plans, and a plan can be
wrong in ways a solver structurally cannot. That is precisely why `rules.py`
exists. The design principle is that **the model is only allowed to be creative
about things that cannot be wrong**: topic names, ordering, prose, triage. Every
claim it makes that *can* be checked — dates, capacity, durations, overlaps,
coverage — is checked, and repaired or deleted if it fails.

If your input has no free text, no topics, and no need for explanation, use the
baseline planner. It is already in the repo and it is better at that job.

## Limits

**Input quality.** Workload estimates come from the student, and a wrong estimate
produces a wrong plan. The planner does not know how long anything actually takes.
Exam dates are never verified against an official source, so every export says so.
Difficulty and confidence are self-reported 1–5 integers with no calibration.

**Non-determinism.** Two identical requests can yield different plans. Rule
compliance is enforced; allocation quality is not, and is only measured in
aggregate by the eval harness.

**No server-side persistence.** Everything lives in `st.session_state`. Refreshing
the browser or restarting the server loses the plan, the progress marks and the
token counter. The Setup tab can save and reload its *inputs* as a JSON file,
which is what makes chapters worth typing in at all, but that is a manual
download and upload — the generated plan, the progress marks and the token
counter are not in it. Export before you close the tab.

**Single user, no auth.** No accounts, no multi-tenancy, no server-side storage.
The API key lives in `.env` on the machine running Streamlit.

**Fixed horizon.** 10 to 14 days, 1 to 6 modules, by schema. A semester plan, a
3-day cram or a 10-module load is out of range. (The Setup tab's caption says
"three to six modules"; the actual validation accepts 1 to 6.)

**Thin input validation.** `day_start` is a free-text field expecting `HH:MM` and
is not validated — a malformed value crashes the baseline planner and is silently
passed through to the model. Nothing stops `min_session_minutes` being set above
`max_session_minutes`, which makes every plan unschedulable.

**Unenforced prompt rules.** The 15 minute gap between blocks, the ~10 percent
buffer allowance and everything in `preferences` are requested in the prompt and
checked by nobody. `no_late_revision` counts calendar days before the exam, not
*available* days, so it can warn on a plan that is actually correct.

**No calendar semantics.** No time zones, no recurring commitments, no
term-boundary awareness, no ICS import or export. A "day" is a naive date and a
block is a start time plus a duration.

**Provider variance.** On OpenRouter, schema enforcement is per endpoint and can
silently degrade to JSON mode; the plan is then more likely to need the repair
round. The Anthropic path is grammar-constrained and does not have this failure
mode.

**Cost model.** A rule violation costs a second request. A pathological input can
therefore cost twice what the sidebar's per-action estimate suggests.

## How it was built

- Days 1 to 3: schema, rule engine, baseline planner, tests.
- Days 4 to 6: AI call with structured outputs, repair loop, Setup and Plan tabs.
- Days 7 to 9: replanning, progress screen, export.
- Days 10 to 12: eval harness, prompt iteration against the 15 cases.
- Days 13 to 14: hardening, second backend, README, demo run.
