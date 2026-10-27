"""Integration tests for POST /chat/stream, POST /resume, and POST /cancel.

Uses httpx2 AsyncClient with ASGITransport to exercise the real FastAPI app.
HARNESS_LLM_BACKEND=fake ensures no credentials are needed; the fake backend
returns a canned final-answer response so the stream terminates deterministically.
"""
from __future__ import annotations

import json

import pytest
from httpx2 import ASGITransport, AsyncClient

from harness.api.main import app
from harness.config.settings import get_settings
from tests.integration.conftest import sse_frames


@pytest.mark.asyncio
async def test_chat_stream_starts_with_thread_id(monkeypatch):
    """First SSE event must contain thread_id."""
    monkeypatch.setenv("HARNESS_LLM_BACKEND", "fake")

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport,
        base_url="https://test",
        headers={"Authorization": f"Bearer {get_settings().api_key}"},
    ) as client:
        events = []
        async with client.stream(
            "POST", "/chat/stream", json={"message": "hello"}, timeout=10.0,
        ) as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    events.append(json.loads(line[6:]))
                if events and events[-1].get("type") in ("final", "error"):
                    break

    assert len(events) >= 1
    assert "thread_id" in events[0]


@pytest.mark.asyncio
async def test_chat_stream_ends_with_final_or_error(monkeypatch):
    """Stream must terminate with a final or error event."""
    monkeypatch.setenv("HARNESS_LLM_BACKEND", "fake")

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport,
        base_url="https://test",
        headers={"Authorization": f"Bearer {get_settings().api_key}"},
    ) as client:
        events = []
        async with client.stream(
            "POST", "/chat/stream", json={"message": "ping"}, timeout=10.0,
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    events.append(json.loads(line[6:]))
                if events and events[-1].get("type") in ("final", "error"):
                    break

    terminal = events[-1]
    assert terminal["type"] in ("final", "error")


@pytest.mark.asyncio
async def test_continuing_a_thread_echoes_its_server_issued_id(monkeypatch):
    """Thread ids are server-issued: a visitor takes the id from the first SSE
    frame of a new thread, sends it back to continue, and gets it echoed."""
    monkeypatch.setenv("HARNESS_LLM_BACKEND", "fake")

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport,
        base_url="https://test",
        headers={"Authorization": f"Bearer {get_settings().api_key}"},
    ) as client:
        started = await client.post("/chat/stream", json={"message": "hi"}, timeout=10.0)
        thread_id = sse_frames(started)[0]["thread_id"]

        continued = await client.post(
            "/chat/stream",
            json={"message": "and again", "thread_id": thread_id},
            timeout=10.0,
        )

    assert continued.status_code == 200
    assert sse_frames(continued)[0]["thread_id"] == thread_id


@pytest.mark.asyncio
async def test_cancel_unknown_thread_returns_not_found(monkeypatch):
    """Cancelling a thread that was never started returns status=not_found."""
    monkeypatch.setenv("HARNESS_LLM_BACKEND", "fake")

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport,
        base_url="https://test",
        headers={"Authorization": f"Bearer {get_settings().api_key}"},
    ) as client:
        response = await client.post("/cancel/nonexistent-id-xyz")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "not_found"
    assert body["thread_id"] == "nonexistent-id-xyz"


# ---------------------------------------------------------------------------
# Token streaming
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_token_events_appear_in_stream(monkeypatch):
    """token events must appear in the SSE stream before the final event."""
    monkeypatch.setenv("HARNESS_LLM_BACKEND", "fake")

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport,
        base_url="https://test",
        headers={"Authorization": f"Bearer {get_settings().api_key}"},
    ) as client:
        events = []
        async with client.stream(
            "POST", "/chat/stream", json={"message": "hello"}, timeout=10.0,
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    events.append(json.loads(line[6:]))
                if events and events[-1].get("type") in ("final", "error"):
                    break

    token_events = [e for e in events if e.get("type") == "token"]
    assert len(token_events) > 0, (
        f"Expected token events; got types: {[e.get('type') for e in events]}"
    )


# ---------------------------------------------------------------------------
# HITL resume
# ---------------------------------------------------------------------------

def _make_hitl_agent_factory(scripted_responses):
    """Return a build_agent replacement whose agent has an approvable calculator."""
    from harness.adapters.llm.fake import FakeLLMClient
    from harness.adapters.tools.calculator import CalculatorTool
    from harness.core.tools.registry import ToolRegistry
    from harness.orchestration.loop import build_graph

    class ApprovableCalculator(CalculatorTool):
        @property
        def requires_approval(self) -> bool:
            return True

    def factory(settings, checkpointer):
        llm = FakeLLMClient(scripted_responses)
        registry = ToolRegistry()
        registry.register(ApprovableCalculator())
        return build_graph(
            llm, checkpointer, registry=registry, system_prompt="You are a test agent."
        )

    return factory


@pytest.mark.asyncio
async def test_hitl_resume_approved_produces_final(monkeypatch, tmp_path):
    """interrupt → resume(approved=True) must produce a final event."""
    from harness.core.types import LLMResponse, ToolCall

    monkeypatch.setenv("HARNESS_LLM_BACKEND", "fake")
    # sqlite, the default checkpointer, so the interrupt and the resume
    # round-trip through real persistence. Resume under the memory
    # checkpointer is covered in test_thread_ownership.py.
    monkeypatch.setenv("HARNESS_CHECKPOINTER", "sqlite")
    monkeypatch.setenv("HARNESS_CHECKPOINTER_URL", str(tmp_path / "hitl-approve.sqlite"))
    monkeypatch.setattr(
        "harness.orchestration.build.build_agent",
        _make_hitl_agent_factory([
            LLMResponse(
                text="",
                tool_calls=(ToolCall(id="c1", name="calculator", arguments={"expression": "2+2"}),),
            ),
            LLMResponse(text="The answer is 4."),
        ]),
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport,
        base_url="https://test",
        headers={"Authorization": f"Bearer {get_settings().api_key}"},
    ) as client:
        # No thread_id: server-issues one (thread ids are server-issued
        # only — see thread_ownership design), same as the real frontend.
        events1 = []
        async with client.stream(
            "POST", "/chat/stream",
            json={"message": "what is 2+2?"},
            timeout=10.0,
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    events1.append(json.loads(line[6:]))
                if events1 and events1[-1].get("type") in ("final", "error", "interrupt"):
                    break

        assert any(e.get("type") == "interrupt" for e in events1), (
            f"Expected interrupt in first stream; got: {[e.get('type') for e in events1]}"
        )
        thread_id = events1[0]["thread_id"]

        events2 = []
        async with client.stream(
            "POST", f"/resume/{thread_id}",
            json={"decision": {"approved": True}},
            timeout=10.0,
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    events2.append(json.loads(line[6:]))
                if events2 and events2[-1].get("type") in ("final", "error"):
                    break

    types2 = [e.get("type") for e in events2]
    assert "final" in types2, f"Expected final after approval; got: {types2}"
    assert "error" not in types2


@pytest.mark.asyncio
async def test_hitl_resume_rejected_agent_recovers(monkeypatch, tmp_path):
    """interrupt → resume(approved=False) must produce final (agent recovers), not error."""
    from harness.core.types import LLMResponse, ToolCall

    monkeypatch.setenv("HARNESS_LLM_BACKEND", "fake")
    # sqlite, the default checkpointer, so the interrupt and the resume
    # round-trip through real persistence. Resume under the memory
    # checkpointer is covered in test_thread_ownership.py.
    monkeypatch.setenv("HARNESS_CHECKPOINTER", "sqlite")
    monkeypatch.setenv("HARNESS_CHECKPOINTER_URL", str(tmp_path / "hitl-reject.sqlite"))
    monkeypatch.setattr(
        "harness.orchestration.build.build_agent",
        _make_hitl_agent_factory([
            LLMResponse(
                text="",
                tool_calls=(ToolCall(id="c1", name="calculator", arguments={"expression": "1+1"}),),
            ),
            LLMResponse(text="The tool was rejected. I cannot complete this calculation."),
        ]),
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport,
        base_url="https://test",
        headers={"Authorization": f"Bearer {get_settings().api_key}"},
    ) as client:
        # No thread_id: server-issues one (thread ids are server-issued
        # only — see thread_ownership design), same as the real frontend.
        events1 = []
        async with client.stream(
            "POST", "/chat/stream",
            json={"message": "what is 1+1?"},
            timeout=10.0,
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    events1.append(json.loads(line[6:]))
                if events1 and events1[-1].get("type") in ("final", "error", "interrupt"):
                    break

        assert any(e.get("type") == "interrupt" for e in events1), (
            f"Expected interrupt in first stream; got: {[e.get('type') for e in events1]}"
        )
        thread_id = events1[0]["thread_id"]

        events2 = []
        async with client.stream(
            "POST", f"/resume/{thread_id}",
            json={"decision": {"approved": False}},
            timeout=10.0,
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    events2.append(json.loads(line[6:]))
                if events2 and events2[-1].get("type") in ("final", "error"):
                    break

    types2 = [e.get("type") for e in events2]
    assert "final" in types2, f"Expected final after rejection (agent recovers); got: {types2}"
    assert "error" not in types2
