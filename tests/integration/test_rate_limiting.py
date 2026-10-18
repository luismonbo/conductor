"""Integration tests for API rate limiting.

Uses the real configured tier values (15/min strict, 60/min default) rather
than stand-in numbers, so these tests prove the actual defaults work -- all
calls are in-memory/fake-backend and fast enough that this isn't a runtime
concern. See docs/superpowers/specs/2026-08-27-rate-limiting-design.md.
"""
from __future__ import annotations

import json
import logging
import uuid

import pytest
from httpx2 import ASGITransport, AsyncClient

from harness.api.main import app
from harness.config.settings import get_settings
from harness.observability.logging_setup import configure_logging


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
    async with AsyncClient(
        transport=transport,
        base_url="https://test",
        headers={"Authorization": f"Bearer {get_settings().api_key}"},
    ) as client:
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
    async with AsyncClient(
        transport=transport,
        base_url="https://test",
        headers={"Authorization": f"Bearer {get_settings().api_key}"},
    ) as client:
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
    async with AsyncClient(
        transport=transport,
        base_url="https://test",
        headers={"Authorization": f"Bearer {get_settings().api_key}"},
    ) as client:
        for _ in range(70):
            response = await client.get("/health")
            assert response.status_code == 200


@pytest.mark.asyncio
async def test_tiers_are_independent(monkeypatch):
    """Exhausting the strict tier doesn't affect the default tier."""
    monkeypatch.setenv("HARNESS_LLM_BACKEND", "fake")

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport,
        base_url="https://test",
        headers={"Authorization": f"Bearer {get_settings().api_key}"},
    ) as client:
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
    async with AsyncClient(
        transport=transport,
        base_url="https://test",
        headers={"Authorization": f"Bearer {get_settings().api_key}"},
    ) as client:
        for _ in range(20):  # well past the normal 15/minute strict threshold
            response = await client.post(
                "/chat/stream", json={"message": "hi"}, timeout=10.0,
            )
            assert response.status_code == 200


# --- one bucket per visitor, behind the proxy that injects the key -----------------
# The proxy (the Cloudflare Worker, or Caddy locally) is the only holder of the shared
# key, and it names the visitor in X-Visitor-IP. See
# docs/superpowers/specs/2026-10-18-production-proxy-design.md.


def _keyed_client() -> AsyncClient:
    """A client that carries the valid shared key, as the proxy does."""
    return AsyncClient(
        transport=ASGITransport(app=app),
        base_url="https://test",
        headers={"Authorization": f"Bearer {get_settings().api_key}"},
    )


async def _chat(client: AsyncClient, ip: str | None):
    headers = {"X-Visitor-IP": ip} if ip is not None else {}
    return await client.post("/chat/stream", json={"message": "hi"}, headers=headers, timeout=10.0)


@pytest.mark.asyncio
async def test_visitors_behind_the_proxy_have_separate_buckets(monkeypatch):
    """One visitor using up their 15 messages does not stop another visitor."""
    monkeypatch.setenv("HARNESS_LLM_BACKEND", "fake")

    async with _keyed_client() as client:
        for _ in range(15):
            assert (await _chat(client, "203.0.113.1")).status_code == 200
        rejected = await _chat(client, "203.0.113.1")
        other_visitor = await _chat(client, "203.0.113.2")

    assert rejected.status_code == 429
    assert other_visitor.status_code == 200


@pytest.mark.asyncio
async def test_a_visitor_ip_header_without_the_key_is_a_401_and_costs_no_bucket(monkeypatch):
    """Whoever lacks the key cannot pick a bucket, or use one up for a real visitor."""
    monkeypatch.setenv("HARNESS_LLM_BACKEND", "fake")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="https://test") as anonymous:
        forged = [await _chat(anonymous, "203.0.113.4") for _ in range(16)]
    async with _keyed_client() as client:
        genuine = [await _chat(client, "203.0.113.4") for _ in range(15)]

    assert {r.status_code for r in forged} == {401}
    assert {r.status_code for r in genuine} == {200}


@pytest.mark.asyncio
async def test_keyed_requests_without_the_header_share_the_socket_bucket(monkeypatch):
    """A caller that holds the key but names no visitor (curl, a script) is limited by
    its socket address, as before."""
    monkeypatch.setenv("HARNESS_LLM_BACKEND", "fake")

    async with _keyed_client() as client:
        for _ in range(15):
            assert (await _chat(client, None)).status_code == 200
        rejected = await _chat(client, None)
        named_visitor = await _chat(client, "203.0.113.3")

    assert rejected.status_code == 429
    assert named_visitor.status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "value",
    ["A" * 4096, "1.2.3.4, 5.6.7.8", "fe80::1%eth0", "001.002.003.004"],
    ids=["4kb-junk", "comma-list", "scoped-ipv6", "leading-zeros"],
)
async def test_hostile_visitor_ip_headers_do_not_break_requests(value):
    """A header the API cannot use falls back to the socket bucket: no 500, no skipped limit."""
    async with _keyed_client() as client:
        response = await client.get("/threads", headers={"X-Visitor-IP": value})

    assert response.status_code == 200


# --- a tripped limit leaves no visitor IP in the logs (launch gate 14f) ------------


def _rate_limit_lines(caplog) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == "harness.api.main" and record.getMessage().startswith("rate limit exceeded")
    ]


@pytest.mark.asyncio
async def test_a_429_logs_no_visitor_ip(monkeypatch, caplog):
    """slowapi would log the limiter key, which is the visitor's IP; the handler's own line must not."""
    monkeypatch.setenv("HARNESS_LLM_BACKEND", "fake")
    configure_logging("INFO")  # the app's lifespan does this in production; ASGITransport does not run it
    caplog.set_level(logging.INFO)

    async with _keyed_client() as client:
        for _ in range(15):
            assert (await _chat(client, "203.0.113.77")).status_code == 200
        rejected = await _chat(client, "203.0.113.77")

    assert rejected.status_code == 429
    assert "203.0.113.77" not in caplog.text
    assert _rate_limit_lines(caplog) == ["rate limit exceeded route=/chat/stream limit=15 per 1 minute"]


@pytest.mark.asyncio
async def test_the_429_line_names_the_route_template_not_the_thread_id(monkeypatch, caplog):
    """/resume/<thread id> is logged as its route template, so no thread id reaches the logs."""
    monkeypatch.setenv("HARNESS_LLM_BACKEND", "fake")
    configure_logging("INFO")
    caplog.set_level(logging.INFO)
    thread_id = str(uuid.uuid4())

    async with _keyed_client() as client:
        statuses = [
            (
                await client.post(
                    f"/resume/{thread_id}", json={"decision": {"approved": True}}, timeout=10.0
                )
            ).status_code
            for _ in range(16)
        ]

    assert statuses == [404] * 15 + [429]
    assert _rate_limit_lines(caplog) == ["rate limit exceeded route=/resume/{thread_id} limit=15 per 1 minute"]
