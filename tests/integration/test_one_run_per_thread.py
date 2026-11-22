"""One run per thread at a time, over the API.

Two runs of one thread both continue from its last checkpoint: two answers to one
approval would each run its tools, and two messages would each extend the same
history. While a run holds a thread, the API refuses another one with 409.
"""
from __future__ import annotations

import asyncio

import httpx2

import harness.api.main as _main
import harness.orchestration.build as build_module
from harness.adapters.llm.fake import FakeLLMClient
from harness.core.types import LLMResponse, ToolCall
from tests.integration.conftest import sse_frames, thread_id_from


def _frame_types(response: httpx2.Response) -> list[str]:
    return [frame.get("type") for frame in sse_frames(response)]


def _make_state_reads_wait_for_a_second_one(monkeypatch, timeout: float = 0.5) -> None:
    """Make the race certain instead of likely: every read of a thread's state waits
    until a second read has happened too, or until the timeout, when no second
    request gets that far."""
    graph = _main._runtime.graph
    read_state = graph.aget_state
    reads = 0
    second_read = asyncio.Event()

    async def aget_state(*args, **kwargs):
        nonlocal reads
        snapshot = await read_state(*args, **kwargs)
        reads += 1
        if reads >= 2:
            second_read.set()
        try:
            await asyncio.wait_for(second_read.wait(), timeout)
        except TimeoutError:
            pass
        return snapshot

    monkeypatch.setattr(graph, "aget_state", aget_state)


async def test_two_answers_to_one_card_sent_at_once_run_its_tool_once(
    client_with_fake, monkeypatch
):
    """Two tabs approve the same card at the same moment."""
    client, _ = client_with_fake
    # In memory, so an approved proposal never reaches the developer's own store.
    monkeypatch.setenv("HARNESS_MEMORY_BACKEND", "in_memory")
    proposal = LLMResponse(
        text="",
        tool_calls=(ToolCall(id="c1", name="remember", arguments={"text": "likes tea"}),),
    )
    fake = FakeLLMClient([proposal, LLMResponse(text="Saved.")], repeat_last=True)
    monkeypatch.setattr(build_module, "build_llm", lambda settings, parser: fake)
    started = await client.post("/chat/stream", json={"message": "remember I like tea"})
    thread_id = thread_id_from(started)
    [card] = [frame["args"] for frame in sse_frames(started) if frame.get("type") == "interrupt"]
    approve = {"interrupt_id": card.get("interrupt_id"), "decision": {"action": "approve"}}
    _make_state_reads_wait_for_a_second_one(monkeypatch)

    responses = await asyncio.gather(
        client.post(f"/resume/{thread_id}", json=approve),
        client.post(f"/resume/{thread_id}", json=approve),
    )

    assert sorted(r.status_code for r in responses) == [200, 409]
    assert sum(_frame_types(r).count("tool_result") for r in responses) == 1


async def test_a_run_cancelled_before_its_first_step_does_not_hold_its_thread():
    """Its task is done but still listed: a coroutine cancelled before it starts never
    reaches the finally that would have released the thread, which would then refuse
    every later message with 409."""
    with _main._claim_thread("t", "visitor") as start:
        task = start(asyncio.sleep(60))
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    with _main._claim_thread("t", "visitor"):
        pass


class _HeldLLM(FakeLLMClient):
    """Says "hi", but every call first waits until `release` is set."""

    def __init__(self) -> None:
        super().__init__([LLMResponse(text="hi")], repeat_last=True)
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def stream(self, messages, tools=None, model=None):
        self.started.set()
        await self.release.wait()
        async for chunk in super().stream(messages, tools=tools, model=model):
            yield chunk


async def test_a_message_to_a_thread_whose_reply_is_still_running_is_refused(
    client_with_fake, monkeypatch
):
    client, _ = client_with_fake
    llm = _HeldLLM()
    monkeypatch.setattr(build_module, "build_llm", lambda settings, parser: llm)
    # ASGITransport returns nothing until a run ends, so the first chat runs as a
    # task, and the sid cookie is set first: the held response's cookie can't arrive.
    await client.get("/health")
    first = asyncio.create_task(client.post("/chat/stream", json={"message": "hello"}))
    try:
        await asyncio.wait_for(llm.started.wait(), timeout=5)
        [thread_id] = list(_main._running)
        while_running = await asyncio.wait_for(
            client.post("/chat/stream", json={"message": "again", "thread_id": thread_id}),
            timeout=2,
        )
    finally:
        llm.release.set()
        await asyncio.wait_for(first, timeout=5)
    after_it_ended = await client.post(
        "/chat/stream", json={"message": "and now", "thread_id": thread_id}
    )

    assert while_running.status_code == 409
    assert after_it_ended.status_code == 200
