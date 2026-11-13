"""Several memory proposals in one turn, over the API: one review per resume.

The graph-level rules are in tests/unit/test_approval_gate_reviews.py. This checks
what a browser sees: each stream ends on the next proposal, and no tool result
appears until the last one is approved.
"""
from __future__ import annotations

import httpx2

import harness.orchestration.build as build_module
from harness.adapters.llm.fake import FakeLLMClient
from harness.core.types import LLMResponse, ToolCall
from tests.integration.conftest import sse_frames, thread_id_from


def _propose_two_memories(client_with_fake, monkeypatch) -> httpx2.AsyncClient:
    """The agent's first reply proposes two memories; its second says it saved them."""
    client, _ = client_with_fake
    # In memory, so an approved proposal never reaches the developer's own store.
    monkeypatch.setenv("HARNESS_MEMORY_BACKEND", "in_memory")
    two_proposals = LLMResponse(
        text="",
        tool_calls=(
            ToolCall(id="c1", name="remember", arguments={"text": "likes tea"}),
            ToolCall(id="c2", name="remember", arguments={"text": "lives in Rome"}),
        ),
    )
    fake = FakeLLMClient([two_proposals, LLMResponse(text="Saved both.")])
    monkeypatch.setattr(build_module, "build_llm", lambda settings, parser: fake)
    return client


def _frame_types(response: httpx2.Response) -> list[str]:
    """The type of every frame after the first, which carries the thread id."""
    return [frame["type"] for frame in sse_frames(response)[1:]]


def _interrupts(response: httpx2.Response) -> list[dict]:
    return [frame["args"] for frame in sse_frames(response) if frame.get("type") == "interrupt"]


async def test_each_proposal_of_a_turn_is_reviewed_before_any_tool_runs(
    client_with_fake, monkeypatch
):
    client = _propose_two_memories(client_with_fake, monkeypatch)
    approve = {"decision": {"action": "approve"}}

    started = await client.post("/chat/stream", json={"message": "remember two things"})
    thread_id = thread_id_from(started)
    after_first_approval = await client.post(f"/resume/{thread_id}", json=approve)
    after_second_approval = await client.post(f"/resume/{thread_id}", json=approve)

    assert [card["proposed"] for card in _interrupts(started)] == ["likes tea"]
    assert [card["proposed"] for card in _interrupts(after_first_approval)] == ["lives in Rome"]
    assert "tool_result" not in _frame_types(after_first_approval)
    assert _frame_types(after_second_approval).count("tool_result") == 2
    assert _frame_types(after_second_approval)[-1] == "final"
