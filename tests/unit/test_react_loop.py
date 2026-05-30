"""ReAct loop unit tests with a scripted fake LLM (no network).

Proves the three control-flow paths that matter: a direct answer, a
tool-then-answer cycle, and the max-iterations failure boundary.
"""
from __future__ import annotations

import pytest

from harness.adapters.llm.fake import FakeLLMClient
from harness.adapters.tools.calculator import CalculatorTool
from harness.core.agents.react import ReActAgent
from harness.core.tools.registry import ToolRegistry
from harness.core.types import AgentState, LLMResponse, Message, Role, ToolCall


def _registry() -> ToolRegistry:
    r = ToolRegistry()
    r.register(CalculatorTool())
    return r


@pytest.mark.asyncio
async def test_direct_answer_no_tools():
    llm = FakeLLMClient([LLMResponse(text="42")])
    agent = ReActAgent(llm, _registry(), "sys")
    result = await agent.run(
        AgentState(messages=[Message(Role.USER, "what is the answer?")])
    )
    assert result.stopped_reason == "final_answer"
    assert result.output == "42"


@pytest.mark.asyncio
async def test_tool_then_answer():
    llm = FakeLLMClient(
        [
            LLMResponse(
                text="",
                tool_calls=(
                    ToolCall(id="c1", name="calculator", arguments={"expression": "6*7"}),
                ),
            ),
            LLMResponse(text="The result is 42."),
        ]
    )
    agent = ReActAgent(llm, _registry(), "sys")
    result = await agent.run(AgentState(messages=[Message(Role.USER, "6*7?")]))
    assert result.stopped_reason == "final_answer"
    assert "42" in result.output
    # transcript should contain a tool-role observation with the computed value
    tool_msgs = [m for m in result.state.messages if m.role is Role.TOOL]
    assert tool_msgs and tool_msgs[0].content == "42"


@pytest.mark.asyncio
async def test_max_iterations_is_failure():
    # Always asks for a tool, never answers -> must hit the ceiling.
    looping = [
        LLMResponse(
            text="",
            tool_calls=(
                ToolCall(id=f"c{i}", name="calculator", arguments={"expression": "1+1"}),
            ),
        )
        for i in range(20)
    ]
    llm = FakeLLMClient(looping)
    agent = ReActAgent(llm, _registry(), "sys")
    result = await agent.run(
        AgentState(messages=[Message(Role.USER, "loop")], max_iterations=3)
    )
    assert result.stopped_reason == "max_iterations"
    assert result.state.iteration == 3
