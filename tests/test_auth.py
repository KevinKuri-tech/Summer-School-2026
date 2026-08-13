"""The sign-in gate: the credential check, and the fact that it actually gates.

The second half matters more than the first. A login screen that renders
beautifully and still leaves the app reachable underneath it is the failure
mode worth pinning, so these tests assert on what the *app* draws when no one
is signed in, not on what the login module returns.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from studyplan import auth

APP = str(Path(__file__).resolve().parent.parent / "app.py")
PASSWORD = "SummerAI2026"


# --------------------------------------------------------------------------
# hashing
# --------------------------------------------------------------------------
def test_digest_round_trips():
    stored = auth.hash_password("hunter2")
    assert auth.check_password("hunter2", stored)
    assert not auth.check_password("hunter3", stored)


def test_digest_carries_neither_the_password_nor_a_shared_salt():
    a = auth.hash_password("hunter2")
    b = auth.hash_password("hunter2")
    assert "hunter2" not in a
    assert a != b, "each digest needs its own salt"
    assert auth.check_password("hunter2", b)


def test_a_malformed_digest_is_a_failed_check_not_a_crash():
    # A typo in STUDYPLAN_USERS must not take the login screen down with it.
    for junk in ["pbkdf2_sha256$", "pbkdf2_sha256$x$y$z", "pbkdf2_sha256$1$!!$!!"]:
        assert not auth.check_password("hunter2", junk)


def test_plaintext_entries_still_verify():
    # What makes STUDYPLAN_USERS usable without minting a digest first.
    assert auth.check_password("hunter2", "hunter2")
    assert not auth.check_password("hunter2", "hunter3")


# --------------------------------------------------------------------------
# the roster
# --------------------------------------------------------------------------
def test_the_three_accounts_share_the_one_password():
    assert auth.users() == ["student1", "student2", "student3"]
    for user in auth.users():
        assert auth.verify(user, PASSWORD)


@pytest.mark.parametrize("username, password", [
    ("student1", "summerai2026"),      # passwords stay case sensitive
    ("student1", "SummerAI2026 "),     # and are not trimmed
    ("student1", ""),
    ("student4", PASSWORD),            # not on the roster
    ("", PASSWORD),
])
def test_rejections(username, password):
    assert not auth.verify(username, password)


@pytest.mark.parametrize("typed", ["Student1", " student1", "STUDENT1 "])
def test_usernames_survive_case_and_stray_whitespace(typed):
    assert auth.verify(typed, PASSWORD)


def test_the_roster_can_be_replaced_from_the_environment(monkeypatch):
    monkeypatch.setenv("STUDYPLAN_USERS",
                       f"alice:{auth.hash_password('wonder')};bob:plain")
    assert auth.users() == ["alice", "bob"]
    assert auth.verify("alice", "wonder")
    assert auth.verify("bob", "plain")
    assert not auth.verify("student1", PASSWORD), "the built-ins are replaced, not merged"


def test_an_unknown_username_costs_what_a_known_one_costs():
    # Same work either way, so the screen cannot be timed to find out which
    # usernames exist. Generous bounds: this is a ratio check, not a benchmark.
    def took(user):
        start = time.perf_counter()
        auth.verify(user, PASSWORD + "!")
        return time.perf_counter() - start

    known, unknown = took("student1"), took("nobody")
    assert 0.4 < unknown / known < 2.5


# --------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------
def _app(**session):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=90)
    at.session_state["force_mock"] = True
    for key, value in session.items():
        at.session_state[key] = value
    return at.run()


@pytest.fixture
def gate():
    return _app()


def test_signed_out_gets_the_login_screen_and_nothing_else(gate):
    assert not gate.exception
    assert [t.key for t in gate.text_input] == ["nx_username", "nx_password"]
    # The app proper is not merely hidden, it is never drawn: no tabs, no
    # sidebar engine picker, no plan state.
    assert gate.tabs == []
    assert gate.radio == []
    assert "modules" not in gate.session_state


def test_the_right_password_opens_the_app(gate):
    gate.text_input(key="nx_username").set_value("student1")
    gate.text_input(key="nx_password").set_value(PASSWORD)
    gate.button[0].click().run()

    assert not gate.exception
    assert gate.session_state["auth_user"] == "student1"
    assert [t.label for t in gate.tabs] == ["1 Setup", "2 Plan", "3 Progress", "4 Export"]


def test_a_wrong_password_stays_outside_and_says_so(gate):
    gate.text_input(key="nx_username").set_value("student1")
    gate.text_input(key="nx_password").set_value("nope")
    gate.button[0].click().run()

    assert not gate.exception
    assert "auth_user" not in gate.session_state
    assert gate.tabs == []
    assert gate.error, "the screen has to say what happened"


def test_repeated_failures_reach_a_cooldown(gate):
    import login

    for _ in range(login.MAX_ATTEMPTS):
        gate.text_input(key="nx_username").set_value("student1")
        gate.text_input(key="nx_password").set_value("nope")
        gate.button[0].click().run()

    assert "Try again in" in gate.error[0].value
    # And the cooldown outranks the right password while it lasts.
    gate.text_input(key="nx_password").set_value(PASSWORD)
    gate.button[0].click().run()
    assert "auth_user" not in gate.session_state


def test_the_gate_does_not_trust_a_username_off_the_roster():
    # Session state is server side, but the check is still made every rerun so
    # that an account removed from the roster loses its open sessions.
    at = _app(auth_user="ghost")
    assert at.tabs == []
    assert [t.key for t in at.text_input] == ["nx_username", "nx_password"]


def test_signing_out_clears_what_the_session_held():
    at = _app(auth_user="student1")
    assert at.session_state["modules"], "precondition: the app seeded its state"

    at.sidebar.button[0].click().run()

    assert not at.exception
    assert "auth_user" not in at.session_state
    assert "modules" not in at.session_state, "the next student must not inherit a plan"
    assert at.tabs == []
