"""Several memory proposals in one turn, over the API: one review per resume.

The graph-level rules are in tests/unit/test_approval_gate_reviews.py. This checks
what a browser sees: each stream ends on the next proposal, no tool result appears
until the last one is approved, and an answer counts only for the interrupt it names.
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
    # Tool results with their text, so a test can tell a saved proposal from a rejected one.
    monkeypatch.setenv("HARNESS_SHOW_TOOL_DETAILS", "true")
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


def _tool_results(response: httpx2.Response) -> list[str]:
    return [frame["text"] for frame in sse_frames(response) if frame.get("type") == "tool_result"]


def _interrupts(response: httpx2.Response) -> list[dict]:
    return [frame["args"] for frame in sse_frames(response) if frame.get("type") == "interrupt"]


def _answer(response: httpx2.Response, action: str) -> dict:
    """A /resume body answering the card this response ended on, the way the UI does."""
    [card] = _interrupts(response)
    return {"interrupt_id": card.get("interrupt_id"), "decision": {"action": action}}


async def test_each_proposal_of_a_turn_is_reviewed_before_any_tool_runs(
    client_with_fake, monkeypatch
):
    client = _propose_two_memories(client_with_fake, monkeypatch)

    started = await client.post("/chat/stream", json={"message": "remember two things"})
    thread_id = thread_id_from(started)
    after_first_approval = await client.post(
        f"/resume/{thread_id}", json=_answer(started, "approve")
    )
    after_second_approval = await client.post(
        f"/resume/{thread_id}", json=_answer(after_first_approval, "approve")
    )

    assert [card["proposed"] for card in _interrupts(started)] == ["likes tea"]
    assert [card["proposed"] for card in _interrupts(after_first_approval)] == ["lives in Rome"]
    assert "tool_result" not in _frame_types(after_first_approval)
    assert _frame_types(after_second_approval).count("tool_result") == 2
    assert _frame_types(after_second_approval)[-1] == "final"


async def test_an_answer_to_a_card_already_answered_is_refused(client_with_fake, monkeypatch):
    """A second tab, or a retried request, approves the card that was just answered.

    It must not land on the next card, which nobody has seen yet: that one stays
    pending, and the answer it then gets decides the turn.
    """
    client = _propose_two_memories(client_with_fake, monkeypatch)
    started = await client.post("/chat/stream", json={"message": "remember two things"})
    thread_id = thread_id_from(started)
    first_card_approved = _answer(started, "approve")
    after_first_approval = await client.post(f"/resume/{thread_id}", json=first_card_approved)

    again = await client.post(f"/resume/{thread_id}", json=first_card_approved)
    second_card_denied = await client.post(
        f"/resume/{thread_id}", json=_answer(after_first_approval, "deny")
    )

    assert again.status_code == 409
    assert second_card_denied.status_code == 200
    # Both calls of the turn are rejected, the approved one included: nothing is saved.
    assert _tool_results(second_card_denied) == ["Rejected by user.", "Rejected by user."]


async def test_an_answer_that_names_no_interrupt_is_refused(client_with_fake, monkeypatch):
    """Without the id, the server could only apply the answer to whatever card is pending."""
    client = _propose_two_memories(client_with_fake, monkeypatch)
    started = await client.post("/chat/stream", json={"message": "remember two things"})

    unnamed = await client.post(
        f"/resume/{thread_id_from(started)}", json={"decision": {"action": "approve"}}
    )

    assert unnamed.status_code == 422
