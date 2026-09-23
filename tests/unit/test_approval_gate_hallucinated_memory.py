"""A remember call for an unregistered tool must not interrupt the run."""
import asyncio

import pytest
from langgraph.checkpoint.memory import MemorySaver

from harness.adapters.llm.fake import FakeLLMClient
from harness.agents.default.graph import build_graph
from harness.agents.default.tools import build_registry
from harness.core.types import LLMResponse, Message, Role, ToolCall


def _hallucinating_llm() -> FakeLLMClient:
    return FakeLLMClient([
        LLMResponse(
            text="",
            tool_calls=(ToolCall(id="c1", name="remember", arguments={"text": "x"}),),
        ),
        LLMResponse(text="done"),
    ])


async def _run(memory_tools: bool) -> dict:
    registry = build_registry(memory_tools=memory_tools)
    graph = build_graph(_hallucinating_llm(), MemorySaver(), registry=registry)
    config = {
        "configurable": {
            "thread_id": "t1",
            "event_queue": asyncio.Queue(),
        }
    }
    return await graph.ainvoke(
        {
            "messages": [Message(role=Role.USER, content="hi")],
            "iteration": 0,
            "max_iterations": 4,
        },
        config,
    )


async def test_no_interrupt_when_memory_tools_disabled():
    result = await _run(memory_tools=False)
    assert not result.get("__interrupt__")


async def test_still_interrupts_when_memory_tools_enabled():
    result = await _run(memory_tools=True)
    assert result.get("__interrupt__")
