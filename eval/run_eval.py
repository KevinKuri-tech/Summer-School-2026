"""Evaluation harness.

Scores every case on: schema validity, hard rule violations, warnings, coverage,
capacity use, latency and tokens. Deterministic rules are the ground truth, so
no human labelling is needed.

    uv run python eval/run_eval.py            # baseline planner, no API cost
    uv run python eval/run_eval.py --live     # calls the model
    uv run python eval/run_eval.py --live --repair --limit 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from studyplan import rules  # noqa: E402
from studyplan.planner import PlannerError, active_backend, get_planner  # noqa: E402
from studyplan.schema import Availability, Module, PlanRequest  # noqa: E402

CASES = Path(__file__).with_name("cases.json")
OUT = Path(__file__).with_name("results.csv")


def build_request(case: dict, start: date) -> PlanRequest:
    return PlanRequest(
        start_date=start,
        horizon_days=case["horizon_days"],
        modules=[Module(name=m["name"], exam_date=start + timedelta(days=m["exam_in_days"]),
                        difficulty=m["difficulty"], confidence=m["confidence"],
                        estimated_hours=m["estimated_hours"]) for m in case["modules"]],
        availability=Availability(
            hours_per_weekday={i: float(h) for i, h in enumerate(case["hours"])},
            blackout_dates=[start + timedelta(days=o) for o in case.get("blackout_offsets", [])],
            min_session_minutes=case.get("min_session", 30),
            max_session_minutes=case.get("max_session", 90),
        ),
        preferences=case.get("note", ""),
    )


def capacity_use(req: PlanRequest, plan) -> float:
    cap = sum(req.availability.minutes_for(d) for d in req.days())
    return round(plan.total_minutes() / cap, 3) if cap else 0.0


def run_case(case: dict, planner, repair: bool) -> dict:
    start = date.today()
    req = build_request(case, start)
    row = {"id": case["id"], "mode": case["mode"], "schema_valid": 1, "errors": 0,
           "warnings": 0, "codes": "", "blocks": 0, "capacity_use": 0.0,
           "modules_covered": 0, "latency_s": 0.0, "tokens_in": 0, "tokens_out": 0,
           "repaired": 0, "failure": ""}
    t0 = time.perf_counter()
    try:
        res = planner.generate(req)
        today = None
        if case["mode"] == "replan":
            first = sorted(res.plan.blocks, key=lambda b: (b.date, b.start_minutes()))
            missed = [b.id for b in first[: case.get("miss_first_n", 2)]]
            today = start + timedelta(days=case.get("replan_day_offset", 2))
            req = req.model_copy(update={"missed_block_ids": missed})
            prev = res.plan
            res2 = planner.replan(req, prev, today)
            row["tokens_in"] += res.input_tokens
            row["tokens_out"] += res.output_tokens
            res = res2
        viol = rules.validate_plan(req, res.plan, today=today)
        hard = rules.errors(viol)
        if hard and repair:
            try:
                fixed = planner.repair(req, res.plan, [str(x) for x in hard])
                v2 = rules.validate_plan(req, fixed.plan, today=today)
                if len(rules.errors(v2)) < len(hard):
                    res, viol, hard = fixed, v2, rules.errors(v2)
                    row["repaired"] = 1
            except PlannerError as exc:
                row["failure"] = f"repair: {exc}"
        row.update({
            "errors": len(hard),
            "warnings": len(viol) - len(hard),
            "codes": "|".join(sorted({v.code for v in viol})),
            "blocks": len(res.plan.blocks),
            "capacity_use": capacity_use(req, res.plan),
            "modules_covered": len({b.module for b in res.plan.blocks} & {m.name for m in req.modules}),
            "tokens_in": row["tokens_in"] + res.input_tokens,
            "tokens_out": row["tokens_out"] + res.output_tokens,
        })
    except Exception as exc:  # noqa: BLE001  - schema failure or API error
        row["schema_valid"] = 0
        row["failure"] = f"{type(exc).__name__}: {exc}"[:180]
    row["latency_s"] = round(time.perf_counter() - t0, 2)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="call the Anthropic API")
    ap.add_argument("--repair", action="store_true", help="allow one corrective round trip")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    cases = json.loads(CASES.read_text())
    if args.limit:
        cases = cases[: args.limit]
    planner = get_planner(force_mock=not args.live, model=args.model)
    backend = active_backend() if args.live else "mock"

    rows = []
    for case in cases:
        row = run_case(case, planner, args.repair)
        rows.append(row)
        flag = "ok  " if row["schema_valid"] and row["errors"] == 0 else "FAIL"
        print(f"{flag} {row['id']:<32} err={row['errors']} warn={row['warnings']} "
              f"blocks={row['blocks']:>3} cap={row['capacity_use']:.2f} "
              f"{row['latency_s']:>5.1f}s {row['codes']} {row['failure']}")

    n = len(rows)
    passed = sum(1 for r in rows if r["schema_valid"] and r["errors"] == 0)
    codes = Counter(c for r in rows for c in r["codes"].split("|") if c)
    print("\n" + "-" * 70)
    print(f"planner: {backend}   cases: {n}")
    print(f"schema valid : {sum(r['schema_valid'] for r in rows)}/{n}")
    print(f"rule clean   : {passed}/{n} ({100 * passed / n:.0f}%)")
    print(f"warnings     : {sum(r['warnings'] for r in rows)}")
    print(f"tokens       : {sum(r['tokens_in'] for r in rows)} in / "
          f"{sum(r['tokens_out'] for r in rows)} out")
    print(f"mean latency : {sum(r['latency_s'] for r in rows) / n:.2f}s")
    if codes:
        print("violations   : " + ", ".join(f"{k}={v}" for k, v in codes.most_common()))

    import csv
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwritten: {OUT}")
    return 0 if passed == n else 1


if __name__ == "__main__":
    raise SystemExit(main())
