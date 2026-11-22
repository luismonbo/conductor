"""The context window: the model is sent the last visitor turns; the checkpoint keeps all."""
from __future__ import annotations

import asyncio

from langgraph.checkpoint.memory import MemorySaver

from harness.adapters.llm.fake import FakeLLMClient
from harness.core.tools.registry import ToolRegistry
from harness.core.types import LLMResponse, Message, Role, ToolCall
from harness.orchestration.loop import build_graph

PROMPT = "You are a test agent."


def _user(text: str) -> Message:
    return Message(role=Role.USER, content=text)


def _assistant(text: str, *calls: ToolCall) -> Message:
    return Message(role=Role.ASSISTANT, content=text, tool_calls=calls)


def _thread(turns: int) -> list[Message]:
    """q1, a1, …, q<turns>: the last visitor message is the new one."""
    history: list[Message] = []
    for i in range(1, turns + 1):
        history.append(_user(f"q{i}"))
        if i < turns:
            history.append(_assistant(f"a{i}"))
    return history


async def _run(history: list[Message], context_turns: int):
    llm = FakeLLMClient([LLMResponse(text="ok")])
    graph = build_graph(
        llm, MemorySaver(), registry=ToolRegistry(), system_prompt=PROMPT,
        context_turns=context_turns,
    )
    config = {"configurable": {"thread_id": "t", "event_queue": asyncio.Queue()}}
    await graph.ainvoke({"messages": history, "iteration": 0, "max_iterations": 4}, config)
    return llm, graph, config


async def _sent(history: list[Message], context_turns: int) -> list[Message]:
    llm, _, _ = await _run(history, context_turns)
    return llm.calls[0]


def _visitor_turns(sent: list[Message]) -> list[str]:
    return [m.content for m in sent if m.role == Role.USER]


async def test_a_long_thread_sends_the_prompt_and_the_last_ten_turns():
    sent = await _sent(_thread(12), context_turns=10)
    assert sent[0] == Message(role=Role.SYSTEM, content=PROMPT)
    assert sent[1].content == "q3"
    assert _visitor_turns(sent) == [f"q{i}" for i in range(3, 13)]


async def test_ten_turns_or_fewer_are_sent_unchanged():
    sent = await _sent(_thread(10), context_turns=10)
    assert _visitor_turns(sent) == [f"q{i}" for i in range(1, 11)]
    assert len(sent) == 1 + 19


async def test_zero_sends_the_whole_thread():
    sent = await _sent(_thread(30), context_turns=0)
    assert _visitor_turns(sent) == [f"q{i}" for i in range(1, 31)]


async def test_the_cut_never_leaves_a_tool_message_without_its_call():
    history: list[Message] = []
    for i in range(1, 13):
        history.append(_user(f"q{i}"))
        if i in (3, 4):  # turn 3 is dropped by the cut below, turn 4 is kept
            call = ToolCall(id=f"c{i}", name="calculator", arguments={"expression": "1+1"})
            history.append(_assistant("", call))
            history.append(
                Message(role=Role.TOOL, content="2", tool_call_id=f"c{i}", name="calculator")
            )
        if i < 12:
            history.append(_assistant(f"a{i}"))

    sent = await _sent(history, context_turns=9)

    calls = {c.id for m in sent if m.role == Role.ASSISTANT for c in m.tool_calls}
    results = [m.tool_call_id for m in sent if m.role == Role.TOOL]
    assert calls == {"c4"} and results == ["c4"]
    assert sent[1] == _user("q4")


async def test_an_old_threads_stored_system_message_gives_way_to_the_current_prompt():
    sent = await _sent([Message(role=Role.SYSTEM, content="OLD PROMPT"), *_thread(12)], 10)
    assert [m.content for m in sent if m.role == Role.SYSTEM] == [PROMPT]


async def test_the_checkpoint_keeps_every_message():
    history = _thread(12)
    _, graph, config = await _run(history, context_turns=10)
    stored = (await graph.aget_state(config)).values["messages"]
    # The serializer gives tool_calls back as a list, so compare what each message says.
    assert [(m.role, m.content) for m in stored] == [
        *[(m.role, m.content) for m in history], (Role.ASSISTANT, "ok"),
    ]
