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


def require_api_key(request: Request) -> None:
    settings = get_settings()
    if not settings.auth_enabled:
        return

    authorization = request.headers.get("Authorization", "")
    scheme, _, token = authorization.partition(" ")

    if scheme != "Bearer" or not token or not hmac.compare_digest(token.encode(), settings.api_key.encode()):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )
