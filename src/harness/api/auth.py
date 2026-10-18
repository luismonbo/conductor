"""Shared-API-key authentication — one FastAPI dependency gating every
non-exempt route. See docs/superpowers/specs/2026-08-28-authentication-design.md.

Re-reads get_settings() per call, same pattern as rate_limit.py's
strict_limit()/default_limit(), so HARNESS_API_KEY changes (including the
ones test fixtures set via monkeypatch.setenv after this module is first
imported) take effect immediately.
"""
from __future__ import annotations

import hmac

from fastapi import HTTPException, Request

from harness.config.settings import get_settings


def has_valid_api_key(request: Request, api_key: str) -> bool:
    """True only for ``Authorization: Bearer <api_key>``.

    The one place the shared key is compared (constant time, on bytes, so a
    non-ASCII token cannot raise). The rate limiter's key function uses it too,
    to tell requests that came through the key-injecting proxy from the rest.
    """
    authorization = request.headers.get("Authorization", "")
    scheme, _, token = authorization.partition(" ")
    return scheme == "Bearer" and bool(token) and hmac.compare_digest(token.encode(), api_key.encode())


def require_api_key(request: Request) -> None:
    settings = get_settings()
    if not settings.auth_enabled:
        return

    if not has_valid_api_key(request, settings.api_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )
