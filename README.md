# AI Study-Plan Optimizer

Turns exam dates, workload estimates and real availability into a 10 to 14 day study
schedule, and rebuilds that schedule when the student misses sessions.

The AI does planning and explanation only. It never tutors, never answers subject
questions, and never decides alone what reaches the student.

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
uv run pytest                       # rule engine tests
uv run python eval/run_eval.py      # 15 cases, baseline planner, free
uv run python eval/run_eval.py --live --repair    # same cases against the model
```

## Scope

In: manual input for 1 to 6 modules, AI plan generation in strict JSON, daily and
weekly view, "I missed this" plus replanning, progress screen, CSV and printable
HTML export, evaluation on 15 cases.

Out, deliberately: syllabus ingestion, authentication, mobile app, calendar sync,
spaced-repetition engine, chatbot.

## Architecture

```
app.py                 Streamlit UI (setup, plan, progress, export)
studyplan/schema.py    Pydantic input models + hand-written output JSON schema
studyplan/prompts.py   System prompt, generation, replanning, repair prompts
studyplan/planner.py   One AI call (structured outputs) + deterministic baseline
studyplan/rules.py     Guardrails: 10 checks + auto-repair
studyplan/exporting.py CSV and print-ready HTML
eval/                  Cases and harness
```

One frontend, one thin core, one AI call per action. No database: state lives in
the Streamlit session and is exported as CSV or JSON.

### The AI call

`messages.create` with `output_config.format = {type: json_schema, schema: ...}`.
Anthropic constrains decoding to the schema, so the response is always valid JSON
in the shape below and there is no parsing or retry logic
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
small and avoids `$ref` indirection. Structured outputs do not guarantee enum
capitalisation, so `block_type` is normalised on parse.

Two backends are supported. OpenRouter is preferred and preselected when
`OPENROUTER_API_KEY` is set, Anthropic is used when only `ANTHROPIC_API_KEY` is
set, otherwise the baseline planner runs.
OpenRouter uses the OpenAI-compatible `response_format: json_schema` with
`strict: true` and `provider.require_parameters: true`, so routing is limited to
endpoints that honour the schema. Enforcement there is per endpoint rather than
guaranteed, so responses are still fence-stripped and re-validated.

Model: OpenRouter defaults to `~anthropic/claude-haiku-latest`, override with
`STUDYPLAN_OPENROUTER_MODEL`. The Anthropic backend defaults to `claude-sonnet-5`
and has its own `STUDYPLAN_ANTHROPIC_MODEL`, because a shared override would send
an OpenRouter slug to Anthropic.
`claude-haiku-4-5` is cheaper, `claude-opus-5` is stronger. Structured outputs
are supported on `claude-opus-5`, `claude-sonnet-5`, `claude-opus-4-8`,
`claude-fable-5` and `claude-haiku-4-5`; `claude-sonnet-4-6` is not on that
list, so it is not usable as the default here. Each plan call is roughly 1 to 3k input and
2 to 5k output tokens, so a 10 dollar budget covers a POC comfortably. The sidebar
shows a running token count.

### Guardrails and human-in-the-loop

Schema validity is not correctness. A schema-valid plan can still put a session
after the exam or overbook a Tuesday. `rules.py` checks every plan
deterministically:

| Code | Check |
| --- | --- |
| `unknown_module` | Module name was never provided (confabulation) |
| `after_exam` | Block on or after that module's exam |
| `outside_horizon` | Block outside the plan window |
| `blackout` / `over_capacity` | Unavailable day, or day over its budget |
| `bad_duration` | Session outside min/max length |
| `overlap` | Two blocks collide |
| `no_coverage` | A module got no time at all |
| `no_late_revision` (warn) | No revision in the last 3 days before the exam |
| `past_edit` | Replan rewrote a day that already happened |
| `under_allocated` (warn) | Module got under 40 percent of its estimate |

Failure path: one corrective round trip to the model with the violations listed,
then deterministic removal of any still-illegal block, with every removal written
into `change_log`. The plan is marked unaccepted until the student presses
**Accept**, and every block is editable before that. This mirrors the NIST
Generative AI Profile's information-integrity and confabulation guidance: a
generative system is treated as a proposer, and a human plus a rule engine decide.

### Replanning

Marking blocks missed and pressing **Replan from today** sends the previous plan,
the missed and completed ids, and the frozen past to the model. The model folds
missed content into remaining days rather than re-adding it verbatim, cuts the
lowest-priority material when it no longer fits, protects revision blocks, and
lists every change. Blocks before today are merged back untouched.

## Evaluation

15 cases in `eval/cases.json` cover the happy path, six modules, an exam in two
days, zero weekend availability, a four-day blackout streak, an infeasible
workload, two exams on one day, short and long session constraints, the minimum
horizon, and three replanning scenarios. The rule engine is the ground truth, so
scoring needs no labelling.

Reported per run: schema validity, hard errors, warnings, violation codes, block
count, capacity use, module coverage, latency, tokens. Results land in
`eval/results.csv`. The baseline planner scores 15/15 rule-clean and is the bar
the model has to match before it wins on quality of the allocation.

## Two-week plan

- Days 1 to 3: schema, rule engine, baseline planner, tests.
- Days 4 to 6: AI call with structured outputs, repair loop, Setup and Plan tabs.
- Days 7 to 9: replanning, progress screen, export.
- Days 10 to 12: eval harness, prompt iteration against the 15 cases.
- Days 13 to 14: hardening, README, demo run.

## Limits

Workload estimates come from the student, and a wrong estimate produces a wrong
plan. The planner does not know how long anything actually takes. Exam dates are
never verified against an official source, so the app says so on every export.
