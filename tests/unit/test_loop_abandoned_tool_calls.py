"""A tool call that never got its result is answered before the model sees it again.

A run can stop between the model's tool calls and their results: a visitor writes
past a pending approval, or the step limit ends the turn. A provider rejects a
transcript in which a tool call has no result, so without an answer the thread
could never continue.
"""
from __future__ import annotations

import asyncio

from langgraph.checkpoint.memory import MemorySaver

from harness.adapters.llm.fake import FakeLLMClient
from harness.adapters.memory.in_memory import InMemoryLongTerm
from harness.adapters.tools.calculator import CalculatorTool
from harness.adapters.tools.remember import RememberTool
from harness.core.tools.registry import ToolRegistry
from harness.core.types import LLMResponse, Message, Role, ToolCall
from harness.orchestration.loop import build_graph

_REMEMBER = LLMResponse(
    text="", tool_calls=(ToolCall(id="c1", name="remember", arguments={"text": "likes tea"}),)
)
_CALCULATE = LLMResponse(
    text="", tool_calls=(ToolCall(id="c1", name="calculator", arguments={"expression": "6*7"}),)
)


def _agent(responses: list[LLMResponse], memory: InMemoryLongTerm | None = None):
    """The scripted model (it records every transcript it is sent) and its graph."""
    llm = FakeLLMClient(responses)
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(RememberTool(memory if memory is not None else InMemoryLongTerm()))
    graph = build_graph(
        llm=llm, checkpointer=MemorySaver(), registry=registry,
        system_prompt="You are a test agent.",
    )
    return llm, graph


def _config() -> dict:
    return {"configurable": {"thread_id": "t", "event_queue": asyncio.Queue()}}


def _turn(text: str, max_iterations: int = 8) -> dict:
    """A new user turn, as POST /chat/stream starts one."""
    return {
        "messages": [Message(role=Role.USER, content=text)],
        "iteration": 0,
        "max_iterations": max_iterations,
    }


def _unanswered_calls(transcript: list[Message]) -> list[str]:
    """Ids of tool calls with no tool message directly after their assistant message."""
    missing: list[str] = []
    for position, message in enumerate(transcript):
        answered: set[str | None] = set()
        for following in transcript[position + 1:]:
            if following.role != Role.TOOL:
                break
            answered.add(following.tool_call_id)
        missing.extend(tc.id for tc in message.tool_calls if tc.id not in answered)
    return missing


async def test_a_call_abandoned_at_a_pending_approval_is_answered_with_no_result():
    llm, graph = _agent([_REMEMBER, LLMResponse(text="Hello again.")])
    config = _config()
    await graph.ainvoke(_turn("remember I like tea"), config)  # pauses on the proposal

    await graph.ainvoke(_turn("never mind, hi"), config)

    sent = llm.calls[-1]
    assert _unanswered_calls(sent) == []
    answer = next(m for m in sent if m.role == Role.TOOL)
    assert (answer.tool_call_id, answer.name) == ("c1", "remember")
    assert answer.content.startswith("No result was recorded")


async def test_a_call_cut_off_by_the_step_limit_is_answered_on_the_next_turn():
    llm, graph = _agent([_CALCULATE, LLMResponse(text="Still here.")])
    config = _config()
    await graph.ainvoke(_turn("what is 6*7?", max_iterations=1), config)
    stored = (await graph.aget_state(config)).values["messages"]
    assert stored[-1].tool_calls  # the limit ended the turn on a call that never ran

    await graph.ainvoke(_turn("are you there?"), config)

    assert _unanswered_calls(llm.calls[-1]) == []


async def test_an_abandoned_call_stays_answered_on_every_later_turn():
    llm, graph = _agent(
        [_REMEMBER, LLMResponse(text="Hello again."), LLMResponse(text="Still here.")]
    )
    config = _config()
    await graph.ainvoke(_turn("remember I like tea"), config)
    await graph.ainvoke(_turn("never mind, hi"), config)

    await graph.ainvoke(_turn("are you there?"), config)

    assert _unanswered_calls(llm.calls[-1]) == []


async def test_the_answer_to_an_abandoned_call_is_not_stored_in_the_thread():
    """The checkpoint only appends: stored, the answer would land after the next
    user message, where a provider rejects it."""
    _, graph = _agent([_REMEMBER, LLMResponse(text="Hello again.")])
    config = _config()
    await graph.ainvoke(_turn("remember I like tea"), config)

    await graph.ainvoke(_turn("never mind, hi"), config)

    stored = (await graph.aget_state(config)).values["messages"]
    assert [m.role for m in stored] == [Role.USER, Role.ASSISTANT, Role.USER, Role.ASSISTANT]


async def test_a_proposal_the_visitor_wrote_past_is_never_written():
    memory = InMemoryLongTerm()
    _, graph = _agent([_REMEMBER, LLMResponse(text="Hello again.")], memory)
    config = _config()
    await graph.ainvoke(_turn("remember I like tea"), config)

    await graph.ainvoke(_turn("never mind, hi"), config)

    assert await memory.search("tea", k=10) == []


async def test_nothing_is_left_to_resume_after_writing_past_a_proposal():
    _, graph = _agent([_REMEMBER, LLMResponse(text="Hello again.")])
    config = _config()
    await graph.ainvoke(_turn("remember I like tea"), config)

    await graph.ainvoke(_turn("never mind, hi"), config)

    assert (await graph.aget_state(config)).interrupts == ()


async def test_a_call_that_ran_is_sent_with_its_own_result_only():
    llm, graph = _agent([_CALCULATE, LLMResponse(text="42")])

    await graph.ainvoke(_turn("what is 6*7?"), _config())

    assert [m.content for m in llm.calls[-1] if m.role == Role.TOOL] == ["42"]
