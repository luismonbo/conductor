"""Anonymous visitor identity — one opaque cookie, deliberately unsigned.

Signing prevents forging a *valid-looking* id. A 256-bit random token is
already unforgeable, and an invented id matches zero rows, so signing buys
nothing here. See docs/superpowers/specs/2026-09-19-visitor-scoped-threads-design.md.

The cookie is the only thing protecting a visitor's thread history, so it is
HttpOnly (out of reach of page script) and, by default, Secure: never sent over
plain HTTP, where it could be read in transit. The caller decides `secure` from
Settings.session_cookie_secure, which exists only for local development on a
browser that does not treat http://localhost as a secure context.

Set from middleware, not a FastAPI dependency: fastapi 0.136.3 silently drops a
dependency-set Set-Cookie header when the endpoint returns a Response directly,
which /chat/stream and /resume/{thread_id} both do.
"""
from __future__ import annotations

import secrets

from starlette.requests import Request
from starlette.responses import Response

COOKIE_NAME = "sid"
COOKIE_MAX_AGE = 31_536_000  # one year
_TOKEN_BYTES = 32


def read_or_mint(request: Request) -> tuple[str, bool]:
    """Return (session_id, minted). `minted` is True when the caller must set the cookie."""
    sid = request.cookies.get(COOKIE_NAME)
    if sid:
        return sid, False
    return secrets.token_urlsafe(_TOKEN_BYTES), True


def attach(
    response: Response, sid: str, *, secure: bool, persistent: bool = True
) -> None:
    """Set the cookie. With `persistent` false it has no Max-Age or Expires, so the
    browser drops it when it closes. (Max-Age=0 would make the browser delete it at once.)"""
    response.set_cookie(
        COOKIE_NAME,
        sid,
        httponly=True,
        secure=secure,
        samesite="lax",
        max_age=COOKIE_MAX_AGE if persistent else None,
        path="/",
    )
