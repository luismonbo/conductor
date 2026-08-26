"""Integration tests for API rate limiting.

Uses the real configured tier values (15/min strict, 60/min default) rather
than stand-in numbers, so these tests prove the actual defaults work -- all
calls are in-memory/fake-backend and fast enough that this isn't a runtime
concern. See docs/superpowers/specs/2026-08-27-rate-limiting-design.md.
"""
from __future__ import annotations

import json

import pytest
from httpx import ASGITransport, AsyncClient

from harness.api.main import app


def _sse_frames(response) -> list[dict]:
    return [
        json.loads(line[len("data: "):])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]


@pytest.mark.asyncio
async def test_strict_tier_limits_chat_stream(monkeypatch):
    """15 chat/stream calls succeed; the 16th is rejected with 429 + Retry-After."""
    monkeypatch.setenv("HARNESS_LLM_BACKEND", "fake")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        for _ in range(15):
            response = await client.post(
                "/chat/stream", json={"message": "hi"}, timeout=10.0,
            )
            assert response.status_code == 200
            assert _sse_frames(response)[-1]["type"] in ("final", "error")

        rejected = await client.post("/chat/stream", json={"message": "hi"})

    assert rejected.status_code == 429
    assert "retry-after" in rejected.headers
    assert "detail" in rejected.json()


@pytest.mark.asyncio
async def test_default_tier_limits_threads():
    """60 /threads calls succeed; the 61st is rejected with 429."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        for _ in range(60):
            response = await client.get("/threads")
            assert response.status_code == 200

        rejected = await client.get("/threads")

    assert rejected.status_code == 429
    assert "detail" in rejected.json()


@pytest.mark.asyncio
async def test_health_never_rate_limited():
    """/health stays exempt even after far more than either tier's threshold."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        for _ in range(70):
            response = await client.get("/health")
            assert response.status_code == 200


@pytest.mark.asyncio
async def test_tiers_are_independent(monkeypatch):
    """Exhausting the strict tier doesn't affect the default tier."""
    monkeypatch.setenv("HARNESS_LLM_BACKEND", "fake")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        for _ in range(15):
            response = await client.post(
                "/chat/stream", json={"message": "hi"}, timeout=10.0,
            )
            assert response.status_code == 200

        rejected = await client.post("/chat/stream", json={"message": "hi"})
        assert rejected.status_code == 429

        still_ok = await client.get("/threads")

    assert still_ok.status_code == 200


@pytest.mark.asyncio
async def test_rate_limit_disabled_via_setting(monkeypatch):
    """HARNESS_RATE_LIMIT_ENABLED=false bypasses limiting entirely."""
    monkeypatch.setenv("HARNESS_LLM_BACKEND", "fake")
    monkeypatch.setenv("HARNESS_RATE_LIMIT_ENABLED", "false")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        for _ in range(20):  # well past the normal 15/minute strict threshold
            response = await client.post(
                "/chat/stream", json={"message": "hi"}, timeout=10.0,
            )
            assert response.status_code == 200
