"""Anonymous visitor identity — one opaque cookie, deliberately unsigned.

Signing prevents forging a *valid-looking* id. A 256-bit random token is
already unforgeable, and an invented id matches zero rows, so signing buys
nothing here. See docs/superpowers/specs/2026-09-19-visitor-scoped-threads-design.md.

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


def attach(response: Response, sid: str) -> None:
    response.set_cookie(
        COOKIE_NAME,
        sid,
        httponly=True,
        samesite="lax",
        max_age=COOKIE_MAX_AGE,
        path="/",
    )
