"""AI Study-Plan Optimizer core package.

Importing this package loads `.env` into the process environment. Every entry
point (app.py, eval/run_eval.py, the tests) reaches the backends through
`studyplan.*`, so this runs before `planner.py` reads ANTHROPIC_API_KEY /
OPENROUTER_API_KEY / STUDYPLAN_MODEL at module import time.
"""

from __future__ import annotations

import os
from pathlib import Path

__version__ = "0.1.0"


def load_dotenv(path: str | os.PathLike[str] | None = None) -> dict[str, str]:
    """Read KEY=value lines from `.env` into os.environ.

    Deliberately minimal, matching the contract in .env.example: one KEY=value
    per line, no quotes, no trailing comments. Blank lines and lines starting
    with # are skipped, as is anything already set in the real environment, so
    an exported variable always wins over the file.

    Returns the variables this call actually set.
    """
    env_path = Path(path) if path else Path(__file__).resolve().parent.parent / ".env"
    if not env_path.is_file():
        return {}

    applied: dict[str, str] = {}
    for raw in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        # Tolerate quotes even though .env.example says not to use them.
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            applied[key] = value
    return applied


load_dotenv()
