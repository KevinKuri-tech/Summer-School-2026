"""Export helpers: CSV and a printable HTML view (browser print -> PDF)."""

from __future__ import annotations

import csv
import html
import io
from datetime import date

from .schema import StudyPlan

COLUMNS = ["id", "date", "weekday", "start_time", "duration_minutes", "module",
           "block_type", "priority", "topic", "rationale", "status"]

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def plan_to_rows(plan: StudyPlan, status: dict[str, str] | None = None) -> list[dict]:
    status = status or {}
    rows = []
    for b in sorted(plan.blocks, key=lambda x: (x.date, x.start_minutes())):
        rows.append({
            "id": b.id,
            "date": b.date.isoformat(),
            "weekday": WEEKDAYS[b.date.weekday()],
            "start_time": b.start_time,
            "duration_minutes": b.duration_minutes,
            "module": b.module,
            "block_type": b.block_type,
            "priority": b.priority,
            "topic": b.topic,
            "rationale": b.rationale,
            "status": status.get(b.id, "todo"),
        })
    return rows


def to_csv(plan: StudyPlan, status: dict[str, str] | None = None) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLUMNS, delimiter=";")
    w.writeheader()
    w.writerows(plan_to_rows(plan, status))
    return buf.getvalue()


def to_html(plan: StudyPlan, status: dict[str, str] | None = None,
            title: str = "Study plan") -> str:
    rows = plan_to_rows(plan, status)
    by_day: dict[str, list[dict]] = {}
    for r in rows:
        by_day.setdefault(f"{r['weekday']} {r['date']}", []).append(r)

    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title>",
        "<style>",
        "body{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;margin:2rem;color:#111}",
        "h1{font-size:1.4rem} h2{font-size:1rem;margin:1.4rem 0 .3rem;border-bottom:1px solid #ddd}",
        ".meta{color:#555;font-size:.85rem} table{border-collapse:collapse;width:100%}",
        "td,th{text-align:left;padding:.3rem .5rem;font-size:.85rem;border-bottom:1px solid #eee}",
        ".p5,.p4{font-weight:600} .done{color:#777;text-decoration:line-through}",
        "@media print{body{margin:1rem} h2{page-break-after:avoid}}",
        "</style></head><body>",
        f"<h1>{html.escape(title)}</h1>",
        f"<p class='meta'>{plan.plan_start} to {plan.plan_end} &middot; "
        f"{len(plan.blocks)} blocks &middot; {plan.total_minutes() // 60}h "
        f"{plan.total_minutes() % 60}min</p>",
        f"<p>{html.escape(plan.strategy)}</p>",
    ]
    for day, items in by_day.items():
        parts.append(f"<h2>{html.escape(day)}</h2><table>")
        for r in items:
            cls = f"p{r['priority']}" + (" done" if r["status"] == "done" else "")
            parts.append(
                f"<tr class='{cls}'><td>{r['start_time']}</td><td>{r['duration_minutes']} min</td>"
                f"<td>{html.escape(r['module'])}</td><td>{html.escape(r['block_type'])}</td>"
                f"<td>{html.escape(r['topic'])}</td><td>{html.escape(r['rationale'])}</td></tr>"
            )
        parts.append("</table>")
    if plan.risks:
        parts.append("<h2>Risks</h2><ul>" +
                     "".join(f"<li>{html.escape(x)}</li>" for x in plan.risks) + "</ul>")
    if plan.change_log:
        parts.append("<h2>Changes</h2><ul>" +
                     "".join(f"<li>{html.escape(x)}</li>" for x in plan.change_log) + "</ul>")
    parts.append("<p class='meta'>Generated with AI assistance and reviewed by the student. "
                 "Verify dates against your official exam schedule.</p>")
    parts.append("</body></html>")
    return "\n".join(parts)
