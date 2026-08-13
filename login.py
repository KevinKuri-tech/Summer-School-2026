"""The sign-in gate and the screen behind it.

`require_login()` is the only entry point the app needs. It returns the signed-in
username, or it draws the login screen and stops the script — nothing after the
call runs, so the gate cannot be walked around by a widget further down the page.

The screen itself is one composition: the intro video fills the viewport on a
loop, the logo opens at hero size over it, and after a beat the logo settles up
into a header while the sign-in card rises from below. Both movements are CSS on
elements that already sit in their final layout position, so only `transform`
and `opacity` animate — nothing reflows, and the card occupies its space from the
first frame even while it is still invisible.

Everything here is scoped by `.stApp:has(#nx-login)`, a marker only this screen
prints. The gate stops the script before the app is drawn, so the two never
coexist and none of this styling can reach the planner UI.
"""

from __future__ import annotations

import base64
import io
import time
from pathlib import Path

import streamlit as st

from studyplan import auth

ROOT = Path(__file__).resolve().parent
LOGO = ROOT / "media" / "pictures" / "Logo.png"
INTRO_VIDEO = ROOT / "media" / "videos" / "intro_video.mp4"

# Where the signed-in username lives. Session state is server-side and per
# browser session, so it is a fact the page cannot forge, unlike a cookie.
SESSION_USER = "auth_user"

# Wrong guesses allowed before the form goes quiet for a while. This is a speed
# bump for someone typing at the screen, not a real defence: session state is
# per session, so a new tab starts a fresh count. Rate limiting that means
# anything belongs in front of the server.
MAX_ATTEMPTS = 5
LOCKOUT_SECONDS = 30

# The roster is a set of shared classroom accounts, so the screen names them.
# Flip this off for anything that is not a demo — it is the only place the
# valid usernames are published.
SHOW_ACCOUNT_HINT = True

_INTRO_DONE = "_nx_intro_played"
_FAILURES = "_nx_failures"
_LOCKED_UNTIL = "_nx_locked_until"


# --------------------------------------------------------------------------
# session
# --------------------------------------------------------------------------
def current_user() -> str | None:
    """The signed-in username, or None.

    Re-checked against the roster rather than trusted on sight, so an account
    removed from `STUDYPLAN_USERS` loses its open sessions at the next rerun.
    """
    user = st.session_state.get(SESSION_USER)
    return user if user in auth.users() else None


def sign_out() -> None:
    """Drop the session, including everything the student typed.

    A shared machine is the normal case for these accounts, so the plan, the
    modules and the setup all go with the user rather than waiting on screen
    for whoever signs in next.
    """
    st.session_state.clear()
    st.rerun()


def require_login() -> str:
    """The signed-in username. Draws the login screen and stops if there is none."""
    user = current_user()
    if user is not None:
        return user
    _render_login()          # ends in st.stop()
    raise AssertionError("unreachable: _render_login stops the script")


# --------------------------------------------------------------------------
# assets
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _logo_light_uri(path: str, mtime: float) -> str:
    """The logo as white artwork on transparency, as a data URI.

    The source file is black-on-white with wide paper margins, which is right
    for the app's light chrome and wrong on top of a video: it would land as a
    white slab. Reading the greyscale as an alpha channel inverts it properly —
    the paper becomes fully transparent, the ink becomes fully opaque, and every
    antialiased edge in between keeps its exact coverage, which a CSS
    `filter: invert()` cannot do without also painting the paper. Cropping to
    the ink then makes the rendered size mean what it says.

    `mtime` is not used in the body; it is part of the cache key, so replacing
    the logo file invalidates this without a restart.
    """
    from PIL import Image, ImageOps

    ink = ImageOps.invert(Image.open(path).convert("L"))
    # The paper is not quite pure white, and 2% of alpha across the whole sheet
    # is a visible veil over dark footage. Flooring it also gives the crop below
    # something to find: bounding the ink, not the noise around it.
    ink = ink.point(lambda v: 0 if v < 12 else v)
    box = ink.getbbox()          # on the ink alone: RGBA's would be the full sheet
    light = Image.new("RGBA", ink.size, (255, 255, 255, 255))
    light.putalpha(ink)
    light = light.crop(box) if box else light

    buf = io.BytesIO()
    light.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _logo_markup(intro: bool) -> str:
    """The marker and the logo, in one element.

    The marker is what every selector on this screen is scoped by, and what
    carries the class that decides whether the opening move plays. It ships
    with the logo because it is an empty div that costs no layout: the block is
    a logo as far as the page is concerned, so the gap above the card stays
    honest.
    """
    marker = f'<div id="nx-login" class="{"nx-intro" if intro else ""}"></div>'
    if not LOGO.exists():
        return marker
    uri = _logo_light_uri(str(LOGO), LOGO.stat().st_mtime)
    return (marker
            + f'<div class="nx-logo"><img src="{uri}" alt="Nexora Study"></div>')


# --------------------------------------------------------------------------
# the screen
# --------------------------------------------------------------------------
_LOGIN_CSS = """
<style>
  /* --- chrome ------------------------------------------------------------ */
  /* No sidebar, no toolbar, no footer: there is exactly one thing to do here. */
  .stApp:has(#nx-login) [data-testid="stHeader"],
  .stApp:has(#nx-login) [data-testid="stToolbar"],
  .stApp:has(#nx-login) [data-testid="stSidebar"],
  .stApp:has(#nx-login) [data-testid="stSidebarCollapsedControl"],
  .stApp:has(#nx-login) [data-testid="stStatusWidget"],
  .stApp:has(#nx-login) [data-testid="stDecoration"] { display: none !important; }

  /* The video is a fixed layer behind everything, so the app's own surfaces
     have to stop painting over it. The near-black underneath is what shows in
     the moment before the first frame decodes, and behind any letterboxing. */
  .stApp:has(#nx-login),
  .stApp:has(#nx-login) [data-testid="stAppViewContainer"],
  .stApp:has(#nx-login) [data-testid="stMain"] { background: #05070d !important; }

  /* One narrow column, vertically centred in the viewport. */
  .stApp:has(#nx-login) [data-testid="stMainBlockContainer"] {
    position: relative; z-index: 2;
    max-width: 470px;
    min-height: 100vh;
    padding: 2.5rem 1rem;
    display: flex; flex-direction: column;
  }
  /* Centred by auto margins rather than justify-content, because the two
     differ exactly where it matters: on a viewport shorter than the
     composition, auto margins collapse to zero and the screen simply scrolls,
     while centring would split the overflow across both ends and cut the top
     off the logo with no way to scroll back up to it. */
  .stApp:has(#nx-login) [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] {
    margin-block: auto;
  }

  /* --- the three layers -------------------------------------------------- */
  /* Video, scrim and composition are siblings in one stacking context — the
     column that holds them — so the whole depth of this screen is these three
     z-indexes and nothing else has to be reasoned about. Getting it wrong is
     not subtle: a scrim that lands on top dims the card and its button along
     with the footage. */
  .stApp:has(#nx-login) [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > * {
    position: relative; z-index: 2;
  }

  /* Darkens the footage enough to carry white type at any point in the loop,
     and pulls the corners down so the centre reads as the focus. Painted by
     the column itself rather than by a div, which keeps it out of the flow
     without having to be excused from it. */
  .stApp:has(#nx-login) [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"]::before {
    content: ""; position: fixed; inset: 0; z-index: 1; pointer-events: none;
    background:
      radial-gradient(120% 85% at 50% 42%, rgba(5,7,13,.22) 0%, rgba(5,7,13,.64) 62%, rgba(5,7,13,.90) 100%),
      linear-gradient(180deg, rgba(5,7,13,.50) 0%, rgba(5,7,13,.20) 35%, rgba(5,7,13,.75) 100%);
  }

  /* --- video ------------------------------------------------------------- */
  /* Taken out of the flow and pinned to the viewport, so the composition above
     is laid out as if the video were not there at all. Spelled out through the
     full chain so it outranks the layer rule above without an !important. */
  .stApp:has(#nx-login) [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:has([data-testid="stVideo"]) {
    position: fixed; inset: 0; z-index: 0;
    width: 100vw; height: 100vh; margin: 0; padding: 0;
    pointer-events: none;
  }
  .stApp:has(#nx-login) [data-testid="stVideo"],
  .stApp:has(#nx-login) [data-testid="stVideo"] video {
    width: 100%; height: 100%; max-height: none;
    object-fit: cover; border-radius: 0;
  }
  /* st.video has no way to ask for a player without a transport bar, and a
     scrubber across the bottom of a login screen is not scenery. */
  .stApp:has(#nx-login) video::-webkit-media-controls,
  .stApp:has(#nx-login) video::-webkit-media-controls-enclosure,
  .stApp:has(#nx-login) video::-webkit-media-controls-panel {
    display: none !important; opacity: 0 !important;
  }

  /* --- logo -------------------------------------------------------------- */
  .stApp:has(#nx-login) .nx-logo { display: flex; justify-content: center; }
  .stApp:has(#nx-login) .nx-logo img {
    width: min(300px, 62vw);
    filter: drop-shadow(0 6px 26px rgba(0,0,0,.65));
    transform-origin: 50% 100%;
  }

  /* The opening move. The logo starts at hero size over the middle of the
     video — the background, in effect — holds there, then settles up into the
     header while the card rises underneath it. Scaling from the bottom edge
     keeps the two arrivals reading as one movement. */
  .stApp:has(#nx-login.nx-intro) .nx-logo img {
    animation: nx-logo-settle 2s cubic-bezier(.16,.84,.44,1) both;
  }
  @keyframes nx-logo-settle {
    0%   { transform: translateY(19vh) scale(1.95); opacity: 0; filter: blur(9px); }
    20%  { transform: translateY(19vh) scale(1.95); opacity: 1; filter: blur(0); }
    52%  { transform: translateY(19vh) scale(1.95); opacity: 1; }
    100% { transform: none; opacity: 1; }
  }

  /* --- card -------------------------------------------------------------- */
  /* st.container(border=True) draws its border on the block *inside* the
     wrapper, so that block is the card: styling the wrapper instead leaves
     Streamlit's own outline showing as a second box within the first. The
     marker span is what tells this container apart from any other. Both are
     internal test ids, so this is the first thing to check after a Streamlit
     upgrade — a stale selector costs the card its surface, not the screen its
     function. */
  .stApp:has(#nx-login) [data-testid="stLayoutWrapper"]:has(.nx-card) { margin-top: 1.6rem; }
  .stApp:has(#nx-login) [data-testid="stLayoutWrapper"]:has(.nx-card) > [data-testid="stVerticalBlock"] {
    padding: 1.7rem 1.7rem 1.3rem;
    border: 1px solid rgba(255,255,255,.13);
    border-radius: 20px;
    background: rgba(9,12,20,.58);
    backdrop-filter: blur(22px) saturate(140%);
    -webkit-backdrop-filter: blur(22px) saturate(140%);
    box-shadow: 0 30px 80px rgba(0,0,0,.55), inset 0 1px 0 rgba(255,255,255,.07);
  }
  .stApp:has(#nx-login.nx-intro) [data-testid="stLayoutWrapper"]:has(.nx-card) {
    animation: nx-card-rise 900ms 1.04s cubic-bezier(.16,.84,.44,1) both;
  }
  @keyframes nx-card-rise {
    from { transform: translateY(42vh); opacity: 0; }
    to   { transform: none; opacity: 1; }
  }

  /* The card is dark whatever the theme is, so its contents are light. */
  .stApp:has(#nx-login) [data-testid="stLayoutWrapper"]:has(.nx-card) * {
    color: #eef2fa;
  }
  .stApp:has(#nx-login) .nx-title {
    margin: 0 0 .15rem; font-size: 1.45rem; font-weight: 650; letter-spacing: -.01em;
  }
  .stApp:has(#nx-login) .nx-sub {
    margin: 0 0 1.1rem; font-size: .95rem; color: rgba(238,242,250,.62) !important;
  }
  .stApp:has(#nx-login) .nx-foot {
    margin: .9rem 0 0; font-size: .85rem; line-height: 1.5;
    color: rgba(238,242,250,.52) !important;
  }
  .stApp:has(#nx-login) .nx-foot code {
    background: rgba(255,255,255,.09); color: rgba(238,242,250,.8) !important;
    padding: .05em .4em; border-radius: 5px; font-size: .95em;
  }

  /* --- fields ------------------------------------------------------------ */
  /* The root element is the box: it carries Streamlit's border and its focus
     ring, and the input inside it is transparent. Restating the fill here
     rather than on the input keeps that focus ring working. */
  .stApp:has(#nx-login) [data-testid="stTextInputRootElement"] {
    background: rgba(255,255,255,.055) !important;
    border-color: rgba(255,255,255,.18) !important;
    border-radius: 11px !important;
  }
  .stApp:has(#nx-login) [data-testid="stTextInputRootElement"]:focus-within {
    background: rgba(255,255,255,.09) !important;
  }
  .stApp:has(#nx-login) [data-testid="stTextInput"] input {
    background: transparent !important;
    color: #fff !important;
    padding-top: .62rem; padding-bottom: .62rem;
  }
  .stApp:has(#nx-login) [data-testid="stTextInput"] input::placeholder {
    color: rgba(255,255,255,.38) !important;
  }
  .stApp:has(#nx-login) [data-testid="stWidgetLabel"] p {
    font-size: .92rem !important; font-weight: 550;
    color: rgba(238,242,250,.78) !important;
  }
  .stApp:has(#nx-login) [data-testid="stForm"] { border: 0; padding: 0; }

  .stApp:has(#nx-login) [data-testid="stFormSubmitButton"] button {
    margin-top: .35rem; padding: .58rem 1rem;
    border-radius: 11px; border: 0; font-weight: 600; font-size: 1.02rem;
    box-shadow: 0 10px 26px rgba(255,75,75,.28);
  }
  .stApp:has(#nx-login) [data-testid="stFormSubmitButton"] button:hover:not(:disabled) {
    transform: translateY(-1px);
  }

  /* Streamlit's alert is a light slab by default, which would punch a hole in
     the glass. Same semantics, restated in the card's palette. */
  .stApp:has(#nx-login) [data-testid="stAlert"] {
    background: rgba(255,75,75,.13) !important;
    border: 1px solid rgba(255,75,75,.34);
    border-radius: 11px;
  }
  .stApp:has(#nx-login) [data-testid="stAlert"] * { color: #ffd9d9 !important; }

  /* --- small screens ----------------------------------------------------- */
  @media (max-height: 680px) {
    .stApp:has(#nx-login) [data-testid="stMainBlockContainer"] { padding: 1.25rem 1rem; }
    .stApp:has(#nx-login) .nx-logo img { width: min(230px, 55vw); }
    @keyframes nx-logo-settle {
      0%   { transform: translateY(13vh) scale(1.6); opacity: 0; filter: blur(9px); }
      20%  { transform: translateY(13vh) scale(1.6); opacity: 1; filter: blur(0); }
      52%  { transform: translateY(13vh) scale(1.6); opacity: 1; }
      100% { transform: none; opacity: 1; }
    }
  }

  /* The choreography is decoration. Anyone who has asked for less motion gets
     the same screen, already arrived. */
  @media (prefers-reduced-motion: reduce) {
    .stApp:has(#nx-login) *, .stApp:has(#nx-login) *::before {
      animation: none !important; transition: none !important;
    }
  }
</style>
"""


def _lock_remaining() -> int:
    """Seconds left on the cooldown, 0 if the form is open."""
    return max(0, int(round(st.session_state.get(_LOCKED_UNTIL, 0.0) - time.monotonic())))


def _attempt(username: str, password: str) -> str | None:
    """Check one submission. Returns an error message, or None on success."""
    locked = _lock_remaining()
    if locked:
        return f"Too many attempts. Try again in {locked} seconds."
    if not username or not password:
        return "Enter your username and password."

    if auth.verify(username, password):
        st.session_state[SESSION_USER] = auth.normalise_username(username)
        st.session_state[_FAILURES] = 0
        st.session_state[_LOCKED_UNTIL] = 0.0
        return None

    failures = st.session_state.get(_FAILURES, 0) + 1
    st.session_state[_FAILURES] = failures
    if failures >= MAX_ATTEMPTS:
        st.session_state[_LOCKED_UNTIL] = time.monotonic() + LOCKOUT_SECONDS
        st.session_state[_FAILURES] = 0
        return f"Too many attempts. Try again in {LOCKOUT_SECONDS} seconds."
    # Which half was wrong is not said on purpose: it would turn the form into
    # a way to find out which usernames exist.
    left = MAX_ATTEMPTS - failures
    return (f"That username and password do not match. "
            f"{left} attempt{'s' if left != 1 else ''} left.")


def _render_login() -> None:
    """Draw the screen, handle one submission, and stop the script."""
    st.markdown(_LOGIN_CSS, unsafe_allow_html=True)

    # First render of a session gets the opening move; a rerun (a wrong password,
    # usually) finds the screen already arrived instead of replaying it.
    intro = not st.session_state.get(_INTRO_DONE, False)

    if INTRO_VIDEO.exists():
        # muted is what makes autoplay legal in every browser, and this footage
        # is scenery rather than something to listen to.
        st.video(str(INTRO_VIDEO), loop=True, autoplay=True, muted=True)

    st.markdown(_logo_markup(intro), unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<span class="nx-card"></span>'
                    '<p class="nx-title">Sign in</p>'
                    '<p class="nx-sub">Your study plan is waiting.</p>',
                    unsafe_allow_html=True)

        # The form stays live during a cooldown rather than going disabled: the
        # remaining seconds are only recomputed on a rerun, and a disabled
        # submit button is exactly the thing that can no longer cause one.
        # Submitting into the lock costs nothing and refreshes the count.
        with st.form("nx_login", clear_on_submit=False, border=False):
            username = st.text_input("Username", key="nx_username",
                                     autocomplete="username",
                                     placeholder="student1")
            password = st.text_input("Password", key="nx_password", type="password",
                                     autocomplete="current-password",
                                     placeholder="••••••••")
            submitted = st.form_submit_button("Sign in", type="primary",
                                              width="stretch")

        if submitted:
            error = _attempt(username, password)
            if error is None:
                # Into the app. The typed password is not cleared by hand:
                # Streamlit drops widget state for widgets a run did not draw,
                # and the next run draws the app instead of this form.
                st.rerun()
            st.error(error, icon=":material/lock:")

        if SHOW_ACCOUNT_HINT:
            names = ", ".join(f"<code>{u}</code>" for u in auth.users())
            st.markdown(f'<p class="nx-foot">Demo accounts: {names} — '
                        'same password for all three.</p>', unsafe_allow_html=True)

    if intro:
        st.session_state[_INTRO_DONE] = True

    st.stop()
