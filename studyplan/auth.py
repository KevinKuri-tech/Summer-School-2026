"""Account lookup for the sign-in gate.

Deliberately Streamlit-free, like the rest of this package: the gate in
`login.py` owns the screen and the session, this module owns the one question
that decides everything ("is this pair valid?"), and it can be tested without
booting an app.

Passwords are stored as PBKDF2-HMAC-SHA256 digests rather than plaintext, so
the repository never carries the secrets themselves and a wrong guess cannot be
confirmed by reading this file. That is the floor rather than the ceiling: this
is a demo roster of shared classroom accounts, not an identity provider. There
is no registration, no reset, no per-user data separation on disk, and a digest
plus this file is still enough to mount an offline guess at a weak password.
Anything beyond a classroom demo wants a real IdP.

The roster can be replaced without editing code, which is how a deployment
should set its own accounts:

    STUDYPLAN_USERS=alice:hunter2;bob:pbkdf2_sha256$240000$...$...

Pairs are `name:secret`, separated by `;`. The secret is either a digest minted
by `python -m studyplan.auth <password>` (preferred, nothing readable in the
environment) or a plaintext password (convenient, and no worse than the API
keys already sitting in `.env`).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os

# Cost of one verification. Tuned to land around a tenth of a second on a
# laptop: slow enough to make offline guessing expensive, fast enough that the
# student does not notice it on the way in.
ITERATIONS = 240_000
_SCHEME = "pbkdf2_sha256"

# The built-in roster: three classroom accounts sharing one password.
_BUILTIN_USERS = {
    "student1": "pbkdf2_sha256$240000$0FgkW5ZOaJ/sNngQS/1Ptw$t6wq+70h+FR1YaY/HO/feVHTK7hZ+DkIbioZnBTDf94",
    "student2": "pbkdf2_sha256$240000$IdFpqUvNbqcseVLt186oTQ$nMuPIweDYJdSS6kdhpjVchhNXsxK5eY2G7oMv3RVrlU",
    "student3": "pbkdf2_sha256$240000$XuR2/vmbNFoIK9GHg7zb6g$3a+7H+iUF6wBUL/D1x9af53Wl/7cMmvIW5IvAc8dhq4",
}


def _b64(raw: bytes) -> str:
    """Padless base64, so a digest never contains `=` and stays one token."""
    return base64.b64encode(raw).decode("ascii").rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.b64decode(text + "=" * (-len(text) % 4))


def hash_password(password: str, *, salt: bytes | None = None,
                  iterations: int = ITERATIONS) -> str:
    """A password as `pbkdf2_sha256$iterations$salt$digest`.

    The salt is per-account and random by default, so the same password does
    not produce the same digest twice and the roster above cannot be read as
    "these three are identical".
    """
    salt = os.urandom(16) if salt is None else salt
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"{_SCHEME}${iterations}${_b64(salt)}${_b64(digest)}"


def check_password(password: str, stored: str) -> bool:
    """Does `password` reproduce `stored`?

    `stored` may also be a plaintext password, which is what makes the env
    override usable without minting digests first. Malformed entries are a
    failed check rather than an exception: a typo in the environment must not
    take the whole app down with a traceback on the login screen.
    """
    if not stored.startswith(_SCHEME + "$"):
        return hmac.compare_digest(password, stored)
    try:
        _, iterations, salt, digest = stored.split("$")
        candidate = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), _unb64(salt), int(iterations))
    except (ValueError, TypeError):
        return False
    # compare_digest rather than ==, so the comparison cannot be timed to
    # recover the digest a byte at a time.
    return hmac.compare_digest(candidate, _unb64(digest))


def _roster() -> dict[str, str]:
    """username -> stored secret, from the environment if it says so.

    Read on every call rather than cached at import, so a test can set the
    variable and see it take effect.
    """
    raw = os.getenv("STUDYPLAN_USERS", "").strip()
    if not raw:
        return dict(_BUILTIN_USERS)
    roster: dict[str, str] = {}
    for pair in raw.split(";"):
        name, sep, secret = pair.partition(":")
        if sep and name.strip():
            roster[normalise_username(name)] = secret.strip()
    return roster or dict(_BUILTIN_USERS)


def normalise_username(name: str) -> str:
    """Usernames are case- and whitespace-insensitive; passwords are not.

    A student typing `Student1 ` on a phone keyboard has not got it wrong, and
    the roster is small enough that collisions are not a concern. Passwords
    stay exact, because folding case there would throw away entropy.
    """
    return name.strip().casefold()


def users() -> list[str]:
    """Every known username, for the hint on the login screen."""
    return sorted(_roster())


def verify(username: str, password: str) -> bool:
    """The whole question: may this pair in?

    An unknown username still pays for a full PBKDF2 round against a throwaway
    digest, so "no such user" and "wrong password" take the same time and the
    screen cannot be used to enumerate the roster.
    """
    stored = _roster().get(normalise_username(username))
    if stored is None:
        check_password(password, hash_password("", salt=b"\0" * 16))
        return False
    return check_password(password, stored)


if __name__ == "__main__":  # pragma: no cover - operator convenience
    import sys

    if len(sys.argv) != 2:
        sys.exit("usage: python -m studyplan.auth <password>   # prints a digest")
    print(hash_password(sys.argv[1]))
