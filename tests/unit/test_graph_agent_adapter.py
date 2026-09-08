"""Unit tests for GraphAgentAdapter — proves it drives the real graph
correctly through the Agent protocol."""
from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import MemorySaver

from harness.adapters.llm.fake import FakeLLMClient
from harness.adapters.tools.calculator import CalculatorTool
from harness.agents.default.graph import build_graph
from harness.core.tools.registry import ToolRegistry
from harness.core.types import AgentState, LLMResponse, Message, Role, ToolCall
from harness.observability.tracer import TraceCollector

from evaluation.harness.graph_agent import GraphAgentAdapter


def _calculator_registry() -> ToolRegistry:
    r = ToolRegistry()
    r.register(CalculatorTool())
    return r


def _approvable_calculator_registry() -> ToolRegistry:
    class ApprovableCalculator(CalculatorTool):
        @property
        def requires_approval(self) -> bool:
            return True

    r = ToolRegistry()
    r.register(ApprovableCalculator())
    return r


@pytest.mark.asyncio
async def test_direct_answer_no_tools():
    graph = build_graph(
        llm=FakeLLMClient([LLMResponse(text="42")]),
        checkpointer=MemorySaver(),
        registry=_calculator_registry(),
    )
    adapter = GraphAgentAdapter(graph)

    result = await adapter.run(
        AgentState(messages=[Message(Role.USER, "what is the answer?")])
    )

    assert result.output == "42"
    assert result.stopped_reason == "final_answer"


@pytest.mark.asyncio
async def test_tool_then_answer_reports_correct_trace():
    graph = build_graph(
        llm=FakeLLMClient([
            LLMResponse(
                text="",
                tool_calls=(ToolCall(id="tc_1", name="calculator", arguments={"expression": "2+2"}),),
            ),
            LLMResponse(text="The answer is 4."),
        ]),
        checkpointer=MemorySaver(),
        registry=_calculator_registry(),
    )
    tracer = TraceCollector()
    adapter = GraphAgentAdapter(graph, tracer=tracer)

    result = await adapter.run(
        AgentState(messages=[Message(Role.USER, "what is 2+2?")])
    )

    assert result.output == "The answer is 4."
    assert result.stopped_reason == "final_answer"

    tool_results = [d for _, e, d in tracer.events if e == "tool_result"]
    assert tool_results == [{"name": "calculator", "is_error": False, "content": "4"}]

    llm_responses = [d for _, e, d in tracer.events if e == "llm_response"]
    called_args = {
        tc["name"]: tc["arguments"]
        for data in llm_responses
        for tc in data["tool_calls"]
    }
    assert called_args == {"calculator": {"expression": "2+2"}}

    # result.state must reflect the real post-run transcript, not the
    # pre-run input state (a single user message, iteration 0).
    assert result.state.iteration == 2
    roles = [m.role for m in result.state.messages]
    assert roles == [Role.USER, Role.ASSISTANT, Role.TOOL, Role.ASSISTANT]
    assert result.state.messages[0].content == "what is 2+2?"
    assert result.state.messages[1].tool_calls[0].name == "calculator"
    assert result.state.messages[2].content == "4"
    assert result.state.messages[3].content == "The answer is 4."


@pytest.mark.asyncio
async def test_max_iterations_reports_stopped_reason():
    responses = [
        LLMResponse(
            text="",
            tool_calls=(ToolCall(id=f"tc_{i}", name="calculator", arguments={"expression": "1+1"}),),
        )
        for i in range(5)
    ]
    graph = build_graph(
        llm=FakeLLMClient(responses),
        checkpointer=MemorySaver(),
        registry=_calculator_registry(),
    )
    adapter = GraphAgentAdapter(graph)

    result = await adapter.run(
        AgentState(
            messages=[Message(Role.USER, "loop forever")],
            max_iterations=2,
        )
    )

    assert result.stopped_reason == "max_iterations"


@pytest.mark.asyncio
async def test_requires_approval_tool_raises_instead_of_hanging():
    graph = build_graph(
        llm=FakeLLMClient([
            LLMResponse(
                text="",
                tool_calls=(ToolCall(id="tc_1", name="calculator", arguments={"expression": "1+1"}),),
            ),
        ]),
        checkpointer=MemorySaver(),
        registry=_approvable_calculator_registry(),
    )
    adapter = GraphAgentAdapter(graph)

    with pytest.raises(RuntimeError, match="requires_approval"):
        await adapter.run(AgentState(messages=[Message(Role.USER, "do the thing")]))
