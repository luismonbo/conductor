"""Unit tests for the default agent LangGraph graph.

All tests use MemorySaver (in-process, no disk) and FakeLLMClient
(no network). Each test creates a fresh graph instance.
"""
from __future__ import annotations

import asyncio

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from harness.adapters.llm.fake import FakeLLMClient
from harness.adapters.tools.calculator import CalculatorTool
from harness.agents.default.graph import GraphState, build_graph
from harness.core.tools.registry import ToolRegistry
from harness.core.types import AgentEvent, LLMResponse, Message, Role, ToolCall


def _make_graph(responses: list[LLMResponse], registry: ToolRegistry | None = None):
    return build_graph(
        llm=FakeLLMClient(responses),
        checkpointer=MemorySaver(),
        registry=registry,
    )


async def _invoke_with_sentinel(graph, state_or_command, config: dict) -> list[AgentEvent]:
    """Run graph.ainvoke concurrently with draining the event queue. Always puts sentinel."""
    queue: asyncio.Queue = config["configurable"]["event_queue"]

    async def _go():
        try:
            await graph.ainvoke(state_or_command, config)
        finally:
            await queue.put(None)

    task = asyncio.create_task(_go())
    events: list[AgentEvent] = []
    while True:
        item = await queue.get()
        if item is None:
            break
        events.append(item)
    await task
    return events


def _base_state(max_iterations: int = 8) -> GraphState:
    return {
        "messages": [Message(role=Role.USER, content="test")],
        "iteration": 0,
        "max_iterations": max_iterations,
    }


def _registry_with_approvable() -> ToolRegistry:
    """Registry where calculator requires approval."""
    class ApprovableCalculator(CalculatorTool):
        @property
        def requires_approval(self) -> bool:
            return True

    r = ToolRegistry()
    r.register(ApprovableCalculator())
    return r


@pytest.mark.asyncio
async def test_happy_path_no_tools_emits_final():
    graph = _make_graph([LLMResponse(text="The answer is 42.")])
    queue: asyncio.Queue = asyncio.Queue()
    config = {"configurable": {"thread_id": "t1", "event_queue": queue}}
    events = await _invoke_with_sentinel(graph, _base_state(), config)

    types = [e.type for e in events]
    assert "final" in types
    final = next(e for e in events if e.type == "final")
    assert final.text == "The answer is 42."
    assert final.stopped_reason == "final_answer"


@pytest.mark.asyncio
async def test_tool_use_loop_emits_tool_events_then_final():
    graph = _make_graph([
        LLMResponse(
            text="",
            tool_calls=(ToolCall(id="c1", name="calculator", arguments={"expression": "6*7"}),),
        ),
        LLMResponse(text="The result is 42."),
    ])
    queue: asyncio.Queue = asyncio.Queue()
    config = {"configurable": {"thread_id": "t2", "event_queue": queue}}
    events = await _invoke_with_sentinel(graph, _base_state(), config)

    types = [e.type for e in events]
    assert "tool_call" in types
    assert "tool_result" in types
    assert "final" in types

    tool_result = next(e for e in events if e.type == "tool_result")
    assert tool_result.text == "42"
    assert tool_result.is_error is False


@pytest.mark.asyncio
async def test_iteration_limit_emits_error():
    looping = [
        LLMResponse(
            text="",
            tool_calls=(ToolCall(id=f"c{i}", name="calculator", arguments={"expression": "1+1"}),),
        )
        for i in range(10)
    ]
    graph = _make_graph(looping)
    queue: asyncio.Queue = asyncio.Queue()
    config = {"configurable": {"thread_id": "t3", "event_queue": queue}}
    events = await _invoke_with_sentinel(graph, _base_state(max_iterations=3), config)

    types = [e.type for e in events]
    assert "error" in types
    assert "final" not in types


@pytest.mark.asyncio
async def test_approval_gate_passthrough_when_no_approval_needed():
    """CalculatorTool has requires_approval=False — gate is transparent."""
    graph = _make_graph([
        LLMResponse(
            text="",
            tool_calls=(ToolCall(id="c1", name="calculator", arguments={"expression": "2+2"}),),
        ),
        LLMResponse(text="4"),
    ])
    queue: asyncio.Queue = asyncio.Queue()
    config = {"configurable": {"thread_id": "t4", "event_queue": queue}}
    events = await _invoke_with_sentinel(graph, _base_state(), config)

    types = [e.type for e in events]
    assert "interrupt" not in types
    assert "final" in types


@pytest.mark.asyncio
async def test_approval_gate_interrupt_then_resume_approved():
    registry = _registry_with_approvable()
    graph = build_graph(
        llm=FakeLLMClient([
            LLMResponse(
                text="",
                tool_calls=(ToolCall(id="c1", name="calculator", arguments={"expression": "3+3"}),),
            ),
            LLMResponse(text="6"),
        ]),
        checkpointer=MemorySaver(),
        registry=registry,
    )

    # First invocation — expect interrupt event
    queue1: asyncio.Queue = asyncio.Queue()
    config1 = {"configurable": {"thread_id": "t-approve", "event_queue": queue1}}
    first_pass = await _invoke_with_sentinel(graph, _base_state(), config1)

    assert any(e.type == "interrupt" for e in first_pass), f"Expected interrupt, got: {[e.type for e in first_pass]}"

    # Second invocation — resume with approved=True
    queue2: asyncio.Queue = asyncio.Queue()
    config2 = {"configurable": {"thread_id": "t-approve", "event_queue": queue2}}
    second_pass = await _invoke_with_sentinel(
        graph, Command(resume={"approved": True}), config2
    )

    types2 = [e.type for e in second_pass]
    assert "final" in types2, f"Expected final after approval, got: {types2}"


@pytest.mark.asyncio
async def test_approval_gate_interrupt_then_resume_rejected():
    """Rejection feeds back to the agent as an observation; agent recovers with a final answer."""
    registry = _registry_with_approvable()
    graph = build_graph(
        llm=FakeLLMClient([
            LLMResponse(
                text="",
                tool_calls=(ToolCall(id="c1", name="calculator", arguments={"expression": "1+1"}),),
            ),
            LLMResponse(text="The tool was rejected. I cannot complete this calculation."),
        ]),
        checkpointer=MemorySaver(),
        registry=registry,
    )

    # First invocation — expect interrupt
    queue1: asyncio.Queue = asyncio.Queue()
    config1 = {"configurable": {"thread_id": "t-reject", "event_queue": queue1}}
    first_pass = await _invoke_with_sentinel(graph, _base_state(), config1)
    assert any(e.type == "interrupt" for e in first_pass)

    # Resume with approved=False → agent should recover, NOT emit error
    queue2: asyncio.Queue = asyncio.Queue()
    config2 = {"configurable": {"thread_id": "t-reject", "event_queue": queue2}}
    second_pass = await _invoke_with_sentinel(
        graph, Command(resume={"approved": False}), config2
    )

    types2 = [e.type for e in second_pass]
    assert "final" in types2, f"Expected final after rejection (agent recovers), got: {types2}"
    assert "error" not in types2


@pytest.mark.asyncio
async def test_rejection_loop_hits_iteration_limit():
    """If the agent keeps requesting the same rejected tool, max_iterations stops it."""
    registry = _registry_with_approvable()
    # Enough tool-call responses to fill max_iterations; agent never gives a final answer.
    looping_responses = [
        LLMResponse(
            text="",
            tool_calls=(ToolCall(id=f"c{i}", name="calculator", arguments={"expression": "1+1"}),),
        )
        for i in range(10)
    ]
    graph = build_graph(
        llm=FakeLLMClient(looping_responses),
        checkpointer=MemorySaver(),
        registry=registry,
    )

    # First invocation — hits interrupt
    queue1: asyncio.Queue = asyncio.Queue()
    config1 = {"configurable": {"thread_id": "t-reject-loop", "event_queue": queue1}}
    await _invoke_with_sentinel(graph, _base_state(max_iterations=3), config1)

    # Keep rejecting until iteration limit
    thread_id = "t-reject-loop"
    final_events: list[AgentEvent] = []
    for _ in range(5):  # more rejections than max_iterations
        q: asyncio.Queue = asyncio.Queue()
        config = {"configurable": {"thread_id": thread_id, "event_queue": q}}
        events = await _invoke_with_sentinel(graph, Command(resume={"approved": False}), config)
        final_events = events
        if any(e.type in ("error", "final") for e in events):
            break

    types = [e.type for e in final_events]
    assert "error" in types, f"Expected error after hitting iteration limit, got: {types}"
    assert "final" not in types


@pytest.mark.asyncio
async def test_call_model_emits_token_events():
    """Each word of the LLM response should arrive as a separate token event."""
    graph = _make_graph([LLMResponse(text="Hello world")])
    queue: asyncio.Queue = asyncio.Queue()
    config = {"configurable": {"thread_id": "t-tokens", "event_queue": queue}}
    events = await _invoke_with_sentinel(graph, _base_state(), config)

    token_events = [e for e in events if e.type == "token"]
    assert len(token_events) >= 2  # "Hello" and "world" are separate yields
    full_text = "".join(e.text for e in token_events)
    assert "Hello" in full_text
    assert "world" in full_text
    assert "thinking" not in [e.type for e in events]


@pytest.mark.asyncio
async def test_token_accumulator_populated_on_final():
    from harness.observability.token_accumulator import TokenAccumulator

    graph = _make_graph([
        LLMResponse(text="The answer is 42.", usage={"prompt_tokens": 50, "completion_tokens": 25}),
    ])
    queue: asyncio.Queue = asyncio.Queue()
    accumulator = TokenAccumulator()
    stopped_reason_holder = ["unknown"]
    config = {
        "configurable": {
            "thread_id": "t-acc",
            "event_queue": queue,
            "token_accumulator": accumulator,
            "stopped_reason_holder": stopped_reason_holder,
        }
    }
    await _invoke_with_sentinel(graph, _base_state(), config)

    assert accumulator.input_tokens == 50
    assert accumulator.output_tokens == 25
    assert accumulator.iterations == 1
    assert stopped_reason_holder[0] == "final_answer"


@pytest.mark.asyncio
async def test_token_accumulator_no_usage_stays_zero():
    from harness.observability.token_accumulator import TokenAccumulator

    graph = _make_graph([LLMResponse(text="Hello")])  # no usage field
    queue: asyncio.Queue = asyncio.Queue()
    accumulator = TokenAccumulator()
    config = {
        "configurable": {
            "thread_id": "t-no-usage",
            "event_queue": queue,
            "token_accumulator": accumulator,
        }
    }
    await _invoke_with_sentinel(graph, _base_state(), config)

    assert accumulator.input_tokens == 0
    assert accumulator.output_tokens == 0
    assert accumulator.iterations == 1
