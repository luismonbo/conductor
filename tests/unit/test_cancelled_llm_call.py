"""A call cancelled mid-stream is still billed, so it must still count toward the token budget."""
from __future__ import annotations

import asyncio

import pytest
from langgraph.checkpoint.memory import MemorySaver

from harness.agents.default.prompt import SYSTEM_PROMPT
from harness.core.tools.registry import ToolRegistry
from harness.core.types import LLMResponse, Message, Role
from harness.observability.token_accumulator import TokenAccumulator
from harness.orchestration.loop import build_graph


class _StalledLLM:
    """Streams one token, then never finishes: the provider never reports usage."""

    @property
    def model_id(self) -> str:
        return "stalled"

    async def generate(self, messages, tools=None, model=None):
        raise NotImplementedError

    async def stream(self, messages, tools=None, model=None):
        yield "partial"
        await asyncio.Event().wait()
        yield LLMResponse(text="never")


async def test_a_call_cancelled_mid_stream_records_an_estimate():
    accumulator = TokenAccumulator()
    queue: asyncio.Queue = asyncio.Queue()
    config = {
        "configurable": {"thread_id": "t", "event_queue": queue, "token_accumulator": accumulator}
    }
    graph = build_graph(
        llm=_StalledLLM(), checkpointer=MemorySaver(),
        registry=ToolRegistry(), system_prompt=SYSTEM_PROMPT,
    )
    run = asyncio.create_task(graph.ainvoke(
        {"messages": [Message(role=Role.USER, content="hello")], "iteration": 0, "max_iterations": 3},
        config,
    ))
    first = await asyncio.wait_for(queue.get(), timeout=5)
    assert first.type == "token"  # the provider has processed the prompt

    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run

    assert accumulator.input_tokens >= len(SYSTEM_PROMPT) // 4
    assert accumulator.output_tokens > 0
