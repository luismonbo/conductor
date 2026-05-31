"""Integration tests for POST /chat/stream and POST /cancel endpoints.

Uses httpx AsyncClient with ASGITransport to exercise the real FastAPI app.
HARNESS_LLM_BACKEND=fake ensures no credentials are needed; the fake backend
returns a canned final-answer response so the stream terminates deterministically.
"""
from __future__ import annotations

import json

import pytest
from httpx import ASGITransport, AsyncClient

from harness.api.main import app


@pytest.mark.asyncio
async def test_chat_stream_starts_with_conversation_id(monkeypatch):
    """First SSE event must contain conversation_id."""
    monkeypatch.setenv("HARNESS_LLM_BACKEND", "fake")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        events = []
        async with client.stream(
            "POST",
            "/chat/stream",
            json={"message": "hello"},
            timeout=10.0,
        ) as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    events.append(json.loads(line[6:]))
                if events and events[-1].get("type") in ("done", "error"):
                    break

    assert len(events) >= 1
    assert "conversation_id" in events[0]


@pytest.mark.asyncio
async def test_chat_stream_ends_with_done_or_error(monkeypatch):
    """Stream must terminate with a done or error event."""
    monkeypatch.setenv("HARNESS_LLM_BACKEND", "fake")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        events = []
        async with client.stream(
            "POST",
            "/chat/stream",
            json={"message": "ping"},
            timeout=10.0,
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    events.append(json.loads(line[6:]))
                if events and events[-1].get("type") in ("done", "error"):
                    break

    terminal = events[-1]
    assert terminal["type"] in ("done", "error")


@pytest.mark.asyncio
async def test_chat_stream_propagates_explicit_conversation_id(monkeypatch):
    """If conversation_id is supplied in the request body, it must appear in the first SSE."""
    monkeypatch.setenv("HARNESS_LLM_BACKEND", "fake")
    cid = "my-test-conv-123"

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        events = []
        async with client.stream(
            "POST",
            "/chat/stream",
            json={"message": "hi", "conversation_id": cid},
            timeout=10.0,
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    events.append(json.loads(line[6:]))
                if events and events[-1].get("type") in ("done", "error"):
                    break

    assert events[0]["conversation_id"] == cid


@pytest.mark.asyncio
async def test_cancel_unknown_conversation_returns_not_found(monkeypatch):
    """Cancelling a conversation that was never started returns status=not_found."""
    monkeypatch.setenv("HARNESS_LLM_BACKEND", "fake")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/cancel/nonexistent-id-xyz")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "not_found"
    assert body["conversation_id"] == "nonexistent-id-xyz"


@pytest.mark.asyncio
async def test_blocking_chat_endpoint_still_works(monkeypatch):
    """POST /chat must remain functional alongside the new streaming endpoint."""
    monkeypatch.setenv("HARNESS_LLM_BACKEND", "fake")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/chat", json={"message": "hello"})

    assert response.status_code == 200
    body = response.json()
    assert "output" in body
    assert "conversation_id" in body
    assert "stopped_reason" in body
