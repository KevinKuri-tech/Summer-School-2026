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

from .schema import SECTION_WORD, PlanRequest, strip_chapter_decoration

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

# Copying a syllabus out of a PDF or a rendered web page routinely strips every
# line break, so the whole document arrives as one multi-thousand-character
# paragraph and the line-per-chapter assumption below collapses to a single row.
# The two rescues here only ever fire on a line already too long to be a real
# chapter title, which keeps a normal paste bit-for-bit untouched.
_RUNON = 120

# "...absolute zero.Topic 1.2: Standard Deviation..." - a section word followed
# by its number is the one dependable seam in a paragraph that lost its newlines.
_SECTION_BREAK = re.compile(rf"(?<=\S)(?={SECTION_WORD}\.?\s*\d)", re.I)

# "...in Haunted HousesCalculating the mean..." - the title ran straight into its
# description, leaving a lowercase letter against a capital with no space. Real
# words do that too ("JavaScript"), so a seam only counts when what follows
# reads like a sentence rather than more title; see _reads_as_prose.
_GLUED = re.compile(r"(?<=[a-z])(?=[A-Z])")


def _reads_as_prose(tail: str) -> bool:
    """Does `tail` look like a description sentence rather than more title?

    Titles stay in title case ("Script Programming and Web Development"), while
    descriptions drop into ordinary sentence case after their first word
    ("Calculating the mean, median, and mode"). Half the next few words being
    lowercase separates the two: lower and "JavaScript" starts getting split,
    higher and a description with several proper nouns stops being recognised.
    """
    words = tail.split()
    if len(words) < 5:
        return False
    following = words[1:7]
    lowercase = sum(1 for w in following if w[:1].islower())
    return lowercase >= len(following) * 0.5


def _drop_glued_description(name: str) -> str:
    """Cut a description that a lost line break welded onto its title."""
    for match in _GLUED.finditer(name):
        head = name[:match.start()]
        # A seam this early is a capitalised word, not the end of a title.
        if len(head) >= 12 and _reads_as_prose(name[match.start():]):
            return head
    return name


def _resegment(text: str) -> list[str]:
    """Split a paste into candidate chapter lines, repairing run-on paragraphs."""
    lines: list[str] = []
    for line in (text or "").splitlines():
        # Short lines are already one chapter each. Only a run-on is worth
        # cutting, so "Introduction to Part 2 of the course" survives intact.
        lines.extend(_SECTION_BREAK.split(line) if len(line) > _RUNON else [line])
    return lines


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
    chapter title: numbering in front, page numbers behind, and a description
    welded on when the paste lost its line breaks. All three are stripped,
    duplicates are dropped case-insensitively while keeping the syllabus order,
    and every chapter starts at the neutral weight so a paste never pretends to
    know how big anything is.
    """
    rows: list[dict] = []
    seen: set[str] = set()
    for raw in _resegment(text):
        name = strip_chapter_decoration(raw)
        name = _drop_glued_description(name)
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
