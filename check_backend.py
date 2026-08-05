"""Backend connectivity check.

Isolates each layer between .env and a schema-valid model response, so a failure
points at one thing instead of "the app doesn't work".

    uv run python check_backend.py

Exits 0 if the configured backend can produce schema-valid JSON, 1 otherwise.
The API key is never printed; only a masked fingerprint is shown.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import httpx  # noqa: E402

from studyplan import load_dotenv  # noqa: E402  (import also triggers the .env read)
from studyplan.planner import (DEFAULT_MODEL, DEFAULT_OPENROUTER_MODEL,  # noqa: E402
                               active_backend)

OK, BAD, WARN = "[ok]", "[FAIL]", "[warn]"

# A deliberately tiny schema: proves the structured-output path works end to end
# without paying for a full plan. Use eval/run_eval.py --live for the real one.
PING_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
}


def mask(key: str) -> str:
    return f"{key[:8]}...{key[-4:]} ({len(key)} chars)" if len(key) > 14 else "(too short)"


def step(n: int, title: str) -> None:
    print(f"\n{n}. {title}")


def check_env() -> tuple[str, str]:
    """Return (backend, key). Exits on failure."""
    step(1, "Configuration")
    env_file = Path(__file__).resolve().parent / ".env"
    print(f"   {OK if env_file.is_file() else WARN} .env file: "
          f"{env_file if env_file.is_file() else 'not found (using real environment only)'}")

    backend = active_backend()
    if backend == "mock":
        print(f"   {BAD} No API key found.")
        print("       Set OPENROUTER_API_KEY (or ANTHROPIC_API_KEY) in .env.")
        print("       Copy .env.example to .env and fill in your key.")
        sys.exit(1)

    var = "OPENROUTER_API_KEY" if backend == "openrouter" else "ANTHROPIC_API_KEY"
    key = os.environ[var].strip().strip('"').strip("'")
    model = os.environ.get("STUDYPLAN_MODEL") or (
        DEFAULT_OPENROUTER_MODEL if backend == "openrouter" else DEFAULT_MODEL)

    print(f"   {OK} Backend: {backend}")
    print(f"   {OK} {var}: {mask(key)}")
    print(f"   {OK} Model: {model}")

    if backend == "openrouter" and not key.startswith("sk-or-"):
        print(f"   {WARN} Key does not start with 'sk-or-' - is this an OpenRouter key?")
    if backend == "anthropic" and not key.startswith("sk-ant-"):
        print(f"   {WARN} Key does not start with 'sk-ant-' - is this an Anthropic key?")

    os.environ["STUDYPLAN_MODEL"] = model
    return backend, key


def check_openrouter(key: str) -> int:
    model = os.environ["STUDYPLAN_MODEL"]
    http = httpx.Client(timeout=60.0)
    auth = {"Authorization": f"Bearer {key}"}

    # ---- 2. key valid? -----------------------------------------------------
    step(2, "Authentication (does OpenRouter accept the key?)")
    try:
        r = http.get("https://openrouter.ai/api/v1/key", headers=auth)
    except httpx.HTTPError as exc:
        print(f"   {BAD} Could not reach OpenRouter: {exc}")
        print("       Check your internet connection / proxy / firewall.")
        return 1
    if r.status_code == 401:
        print(f"   {BAD} 401 Unauthorized - the key was rejected.")
        print("       Re-copy it from https://openrouter.ai/keys (no quotes, no spaces).")
        return 1
    if r.status_code != 200:
        print(f"   {BAD} Unexpected {r.status_code}: {r.text[:200]}")
        return 1

    data = r.json().get("data", {})
    print(f"   {OK} Key accepted.")
    if data.get("label"):
        print(f"       label: {data['label']}")
    limit, usage = data.get("limit"), data.get("usage")
    print(f"       usage: {usage}   credit limit: {'unlimited' if limit is None else limit}")
    if data.get("is_free_tier"):
        print(f"   {WARN} Free-tier account: expect per-minute and per-day throttling.")

    # ---- 3. model reachable and schema-capable? ----------------------------
    step(3, f"Model availability ({model})")
    r = http.get("https://openrouter.ai/api/v1/models")
    if r.status_code != 200:
        print(f"   {WARN} Could not list models ({r.status_code}); skipping this check.")
    else:
        entry = next((m for m in r.json()["data"] if m["id"] == model), None)
        if entry is None:
            print(f"   {BAD} '{model}' is not a known OpenRouter slug.")
            print("       Browse valid slugs at https://openrouter.ai/models")
            return 1
        supports = "structured_outputs" in (entry.get("supported_parameters") or [])
        print(f"   {OK} Slug exists (context {entry.get('context_length')}).")
        if supports:
            print(f"   {OK} Supports structured outputs - the schema will be enforced.")
        else:
            print(f"   {WARN} Does NOT support structured outputs.")
            print("       The app still runs, but falls back to JSON mode where the")
            print("       schema is a hint, so expect more rule violations.")

    # ---- 4. a real (tiny) completion ---------------------------------------
    step(4, "Live round trip (a real request, minimal cost)")
    body = {
        "model": model,
        "max_tokens": 50,
        "messages": [
            {"role": "system", "content": "You reply only with JSON."},
            {"role": "user", "content": 'Reply with exactly {"ok": true}'},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "ping", "strict": True, "schema": PING_SCHEMA},
        },
        "provider": {"require_parameters": True},
    }
    r = http.post("https://openrouter.ai/api/v1/chat/completions",
                  headers={**auth, "Content-Type": "application/json",
                           "X-Title": "Study-Plan Optimizer check"},
                  json=body)

    if r.status_code == 404 and any(
            h in r.text.lower() for h in ("data policy", "guardrail", "privacy")):
        print(f"   {BAD} 404 - no endpoint your account is allowed to use.")
        print("       Free (':free') models require permitting providers that may")
        print("       train on your prompts. Enable that at:")
        print("           https://openrouter.ai/settings/privacy")
        print("       Or drop the ':free' suffix to use the paid endpoint.")
        return 1
    if r.status_code == 402:
        print(f"   {BAD} 402 - insufficient credits for this model.")
        return 1
    if r.status_code == 429:
        print(f"   {BAD} 429 - rate limited. Free models are throttled; wait and retry.")
        return 1
    if r.status_code != 200:
        print(f"   {BAD} {r.status_code}: {r.text[:300]}")
        return 1

    payload = r.json()
    if "error" in payload:
        print(f"   {BAD} {str(payload['error'])[:300]}")
        return 1

    text = payload["choices"][0]["message"].get("content") or ""
    usage = payload.get("usage", {}) or {}
    print(f"   {OK} HTTP 200 from {payload.get('model', model)}")
    print(f"       tokens: {usage.get('prompt_tokens')} in / "
          f"{usage.get('completion_tokens')} out")
    print(f"       raw reply: {text.strip()[:120]!r}")

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        print(f"   {BAD} Reply was not valid JSON - this model ignores the schema.")
        print("       Pick a slug that supports structured outputs (see step 3).")
        return 1
    if parsed.get("ok") is not True:
        print(f"   {WARN} Valid JSON but unexpected content: {parsed}")
    else:
        print(f"   {OK} Parsed and schema-valid.")
    return 0


def check_anthropic(key: str) -> int:
    model = os.environ["STUDYPLAN_MODEL"]
    step(2, f"Live round trip ({model})")
    try:
        import anthropic
    except ImportError:
        print(f"   {BAD} anthropic package not installed. Run: uv sync")
        return 1

    client = anthropic.Anthropic(api_key=key)
    try:
        resp = client.messages.create(
            model=model, max_tokens=50,
            system="You reply only with JSON.",
            messages=[{"role": "user", "content": 'Reply with exactly {"ok": true}'}],
            extra_body={"output_config": {
                "format": {"type": "json_schema", "schema": PING_SCHEMA}}},
        )
    except anthropic.AuthenticationError:
        print(f"   {BAD} Key rejected by Anthropic.")
        return 1
    except anthropic.NotFoundError:
        print(f"   {BAD} Model '{model}' not found - check STUDYPLAN_MODEL.")
        return 1
    except anthropic.APIStatusError as exc:
        print(f"   {BAD} API error {exc.status_code}: {exc}")
        return 1

    text = next((b.text for b in resp.content if getattr(b, "type", "") == "text"), "")
    print(f"   {OK} HTTP 200 from {resp.model}")
    print(f"       tokens: {resp.usage.input_tokens} in / {resp.usage.output_tokens} out")
    print(f"       raw reply: {text.strip()[:120]!r}")
    try:
        json.loads(text)
    except json.JSONDecodeError:
        print(f"   {BAD} Reply was not valid JSON.")
        return 1
    print(f"   {OK} Parsed and schema-valid.")
    return 0


def main() -> int:
    load_dotenv()
    print("Study-Plan Optimizer - backend check")
    print("=" * 60)
    backend, key = check_env()
    rc = check_openrouter(key) if backend == "openrouter" else check_anthropic(key)

    print("\n" + "=" * 60)
    if rc == 0:
        print("PASS - the backend works. Next:")
        print("   uv run python eval/run_eval.py --live --limit 1   # full schema")
        print("   uv run streamlit run app.py                       # the app")
    else:
        print("FAIL - fix the [FAIL] item above and re-run.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
