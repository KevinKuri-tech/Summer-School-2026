"""Saving and reloading the Setup tab, plus syllabus paste parsing.

The setup file is a versioned `PlanRequest` and nothing else. Reusing the input
contract means there is only one thing to keep in sync, and validation comes for
free: `pydantic.ValidationError` subclasses `ValueError`, so a file with a
40-day horizon is rejected with the field name already in the message.

This is deliberately not the plan export. `exporting.py` writes the generated
schedule; this writes what the student typed in.
"""

from __future__ import annotations

import json
import re

from .schema import PlanRequest, strip_chapter_decoration

SETUP_VERSION = 1

# Never let a paste blow up the prompt payload or the chapter editor. Matches
# Module.chapters' own max_length.
MAX_CHAPTERS = 40

# Trailing page references, in the orders they actually appear in syllabi:
# "(pp. 1-20)", "Probability .......... 21", "Regression Seiten 120-148".
_PAGE_REFS = (
    re.compile(r"\s*\(\s*(?:pp?\.?|s\.?|seiten?|pages?)?\s*\d+(?:\s*[-–]\s*\d+)?\s*\)\s*$", re.I),
    re.compile(r"\s*\.{3,}\s*\d+(?:\s*[-–]\s*\d+)?\s*$"),
    re.compile(r"\s*(?:pp?\.?|s\.?|seiten?|pages?)\s*\d+(?:\s*[-–]\s*\d+)?\s*$", re.I),
)


def setup_to_json(req: PlanRequest) -> str:
    """Serialise a validated request as a portable, hand-editable setup file."""
    payload = {"version": SETUP_VERSION, **json.loads(req.model_dump_json())}
    return json.dumps(payload, indent=2, ensure_ascii=False)


def setup_from_json(text: str) -> PlanRequest:
    """Parse a setup file. Raises ValueError on anything it will not accept."""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"That is not a JSON file ({exc}).") from exc
    if not isinstance(data, dict):
        raise ValueError("A setup file must be a JSON object.")

    version = data.pop("version", None)
    if version != SETUP_VERSION:
        raise ValueError(
            f"Unsupported setup file version {version!r}, this app writes version {SETUP_VERSION}."
        )
    # ValidationError is a ValueError, and its message already names the field
    # that failed, so callers get a usable error without any rewrapping here.
    return PlanRequest.model_validate(data)


def parse_syllabus(text: str, limit: int = MAX_CHAPTERS) -> list[dict]:
    """Turn a pasted table of contents into chapter rows.

    Everything the student pastes is noise around the one thing wanted, the
    chapter title: numbering in front, page numbers behind. Both are stripped,
    duplicates are dropped case-insensitively while keeping the syllabus order,
    and every chapter starts at the neutral weight so a paste never pretends to
    know how big anything is.
    """
    rows: list[dict] = []
    seen: set[str] = set()
    for raw in (text or "").splitlines():
        name = strip_chapter_decoration(raw)
        for pattern in _PAGE_REFS:
            name = pattern.sub("", name)
        name = name.strip(" \t.-–—:;,")
        if not any(ch.isalnum() for ch in name):
            continue
        key = name.casefold()
        if key in seen:
            continue
        seen.add(key)
        rows.append({"name": name, "weight": 3, "confidence": None})
        if len(rows) >= limit:
            break
    return rows
