"""The step limit stops a model that keeps asking for tools, never one that has answered."""
from __future__ import annotations

import asyncio

from langgraph.checkpoint.memory import MemorySaver

from harness.adapters.llm.fake import FakeLLMClient
from harness.adapters.tools.calculator import CalculatorTool
from harness.core.tools.registry import ToolRegistry
from harness.core.types import AgentEvent, LLMResponse, Message, Role, ToolCall
from harness.orchestration.loop import build_graph


async def _run(
    responses: list[LLMResponse], max_iterations: int
) -> tuple[list[AgentEvent], str]:
    """Run one turn; return the events it emitted and the reason it stopped."""
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    graph = build_graph(
        llm=FakeLLMClient(responses), checkpointer=MemorySaver(),
        registry=registry, system_prompt="You are a test agent.",
    )
    queue: asyncio.Queue = asyncio.Queue()
    stopped_reason = ["unknown"]
    config = {"configurable": {
        "thread_id": "t", "event_queue": queue, "stopped_reason_holder": stopped_reason,
    }}
    await graph.ainvoke(
        {
            "messages": [Message(role=Role.USER, content="what is 6*7?")],
            "iteration": 0,
            "max_iterations": max_iterations,
        },
        config,
    )
    events: list[AgentEvent] = []
    while not queue.empty():
        events.append(queue.get_nowait())
    return events, stopped_reason[0]


async def test_an_answer_on_the_last_allowed_model_call_is_final():
    # Two model calls are allowed: the first asks for a tool, the second answers.
    events, stopped_reason = await _run(
        [
            LLMResponse(
                text="",
                tool_calls=(ToolCall(id="c1", name="calculator", arguments={"expression": "6*7"}),),
            ),
            LLMResponse(text="The result is 42."),
        ],
        max_iterations=2,
    )

    assert [e.type for e in events if e.type in ("final", "error")] == ["final"]
    assert events[-1].text == "The result is 42."
    assert stopped_reason == "final_answer"
