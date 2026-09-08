"""Integration tests for shared-API-key auth.

docs/superpowers/specs/2026-08-28-authentication-design.md
"""
from __future__ import annotations

import httpx
import pytest

import harness.api.main as _main
from harness.config.settings import get_settings


@pytest.fixture
async def client():
    transport = httpx.ASGITransport(app=_main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# The six routes the auth spec requires to be protected. Placeholder path
# segments (e.g. "does-not-matter") are fine for the two {thread_id} routes:
# the auth dependency must reject the request before the route body ever
# looks up whether the thread exists.
_PROTECTED_ROUTES = [
    pytest.param("GET", "/models", None, id="get-models"),
    pytest.param("GET", "/threads", None, id="get-threads"),
    pytest.param("GET", "/threads/does-not-matter", None, id="get-thread-by-id"),
    pytest.param("POST", "/chat/stream", {"message": "hi"}, id="post-chat-stream"),
    pytest.param(
        "POST", "/resume/does-not-matter", {"decision": {"approved": True}}, id="post-resume"
    ),
    pytest.param("POST", "/cancel/does-not-matter", None, id="post-cancel"),
]


@pytest.mark.parametrize("method, path, json_body", _PROTECTED_ROUTES)
async def test_protected_route_without_key_returns_401(client, method, path, json_body):
    resp = await client.request(method, path, json=json_body)

    assert resp.status_code == 401
    assert resp.headers["www-authenticate"] == "Bearer"
    assert resp.json() == {"detail": "Invalid or missing API key."}


async def test_protected_route_with_wrong_key_returns_401(client):
    resp = await client.get("/models", headers={"Authorization": "Bearer wrong-key"})

    assert resp.status_code == 401


async def test_protected_route_with_non_ascii_token_returns_401(client):
    """A raw non-ASCII header byte must not crash hmac.compare_digest into a 500.

    The value is sent as bytes (not str) to bypass httpx's own client-side
    ascii-encode guard on str header values -- a non-Python caller has no such
    guard, so the server must handle this on its own. Starlette decodes header
    bytes as latin-1 (never raises), so the app sees a str with a non-ASCII
    character by the time it reaches require_api_key.
    """
    resp = await client.get("/models", headers={"Authorization": b"Bearer caf\xe9"})

    assert resp.status_code == 401


async def test_protected_route_with_correct_key_succeeds(client):
    key = get_settings().api_key
    resp = await client.get("/models", headers={"Authorization": f"Bearer {key}"})

    assert resp.status_code == 200


async def test_health_never_requires_key(client):
    resp = await client.get("/health")

    assert resp.status_code == 200


async def test_auth_disabled_bypasses_check(monkeypatch, client):
    monkeypatch.setenv("HARNESS_AUTH_ENABLED", "false")

    resp = await client.get("/models")

    assert resp.status_code == 200
