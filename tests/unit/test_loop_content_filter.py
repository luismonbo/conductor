"""A streamed answer the provider's filter cut off is a failed call, not a final answer."""
from __future__ import annotations

import asyncio

import pytest
from langgraph.checkpoint.memory import MemorySaver

from harness.adapters.llm.fake import FakeLLMClient
from harness.core.llm.errors import ContentFiltered
from harness.core.tools.registry import ToolRegistry
from harness.core.types import LLMResponse, Message, Role
from harness.observability.token_accumulator import TokenAccumulator
from harness.orchestration.loop import build_graph

_INPUT = {
    "messages": [Message(role=Role.USER, content="hello")],
    "iteration": 0,
    "max_iterations": 3,
}


def _graph(response: LLMResponse):
    return build_graph(
        llm=FakeLLMClient([response]), checkpointer=MemorySaver(),
        registry=ToolRegistry(), system_prompt="You are a test agent.",
    )


def _config(accumulator: TokenAccumulator) -> dict:
    return {"configurable": {
        "thread_id": "t", "event_queue": asyncio.Queue(), "token_accumulator": accumulator,
    }}


async def test_a_filtered_answer_raises_and_is_never_checkpointed():
    accumulator = TokenAccumulator()
    config = _config(accumulator)
    graph = _graph(LLMResponse(
        text="The first part of an answer", finish_reason="content_filter",
        usage={"prompt_tokens": 40, "completion_tokens": 7},
    ))

    with pytest.raises(ContentFiltered) as info:
        await graph.ainvoke(_INPUT, config)

    assert info.value.stage == "completion"
    # The call was billed, so its reported usage still counts toward the budget.
    assert (accumulator.input_tokens, accumulator.output_tokens) == (40, 7)
    snapshot = await graph.aget_state(config)
    assert [m.role for m in snapshot.values["messages"]] == [Role.USER]


async def test_a_filtered_answer_without_usage_records_an_estimate():
    accumulator = TokenAccumulator()
    graph = _graph(LLMResponse(text="The first part of an answer", finish_reason="content_filter"))

    with pytest.raises(ContentFiltered):
        await graph.ainvoke(_INPUT, _config(accumulator))

    assert accumulator.input_tokens > 0
    assert accumulator.output_tokens > 0
