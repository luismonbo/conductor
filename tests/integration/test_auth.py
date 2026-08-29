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


async def test_protected_route_without_key_returns_401(client):
    resp = await client.get("/models")

    assert resp.status_code == 401
    assert resp.headers["www-authenticate"] == "Bearer"
    assert resp.json() == {"detail": "Invalid or missing API key."}


async def test_protected_route_with_wrong_key_returns_401(client):
    resp = await client.get("/models", headers={"Authorization": "Bearer wrong-key"})

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
