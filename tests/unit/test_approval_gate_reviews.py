"""What the approval gate asks a human, and what one answer may let run.

Every call that needs approval is reviewed before any call of its turn runs. An
answer applies only to the review it was given for, whatever tools a restarted
process registers, and never to a later turn.
"""
from __future__ import annotations

import asyncio

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from harness.adapters.llm.fake import FakeLLMClient
from harness.adapters.memory.in_memory import InMemoryLongTerm
from harness.adapters.tools.calculator import CalculatorTool
from harness.adapters.tools.remember import RememberTool
from harness.core.tools.registry import ToolRegistry
from harness.core.types import LLMResponse, Message, Role, ToolCall
from harness.orchestration.loop import build_graph

_APPROVE = Command(resume={"action": "approve"})
_DENY = Command(resume={"action": "deny"})


class _ApprovableCalculator(CalculatorTool):
    requires_approval = True


def _remember(call_id: str, text: str) -> ToolCall:
    return ToolCall(id=call_id, name="remember", arguments={"text": text})


def _calculate(call_id: str) -> ToolCall:
    return ToolCall(id=call_id, name="calculator", arguments={"expression": "6*7"})


def _calls(*tool_calls: ToolCall) -> LLMResponse:
    return LLMResponse(text="", tool_calls=tool_calls)


def _build(responses: list[LLMResponse], *tools, checkpointer: MemorySaver | None = None):
    """A graph over `tools`. Two graphs on one checkpointer model a restart of the
    process: the thread is the same, the tools registered may not be."""
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return build_graph(
        llm=FakeLLMClient(responses), checkpointer=checkpointer or MemorySaver(),
        registry=registry, system_prompt="You are a test agent.",
    )


def _graph(responses: list[LLMResponse], memory: InMemoryLongTerm, calculator: CalculatorTool):
    return _build(responses, calculator, RememberTool(memory))


def _config() -> dict:
    return {"configurable": {"thread_id": "t", "event_queue": asyncio.Queue()}}


def _turn(text: str) -> dict:
    """A new user turn, as POST /chat/stream starts one."""
    return {
        "messages": [Message(role=Role.USER, content=text)],
        "iteration": 0,
        "max_iterations": 8,
    }


def _tool_results(result: dict) -> dict[str, str]:
    """call id -> what the model was told about that call."""
    return {
        m.tool_call_id: m.content for m in result["messages"] if m.role == Role.TOOL
    }


async def _remembered(memory: InMemoryLongTerm) -> set[str]:
    return {hit.text for hit in await memory.search("anything", k=50)}


def _pending(result: dict) -> list[dict]:
    """The payloads of the interrupts the run is paused on."""
    return [interrupt.value for interrupt in result.get("__interrupt__", ())]


# --- An answer applies only to its own turn ---------------------------------------


async def test_a_denied_proposal_does_not_reject_the_next_tool_call_of_the_run():
    memory = InMemoryLongTerm()
    graph = _graph(
        [_calls(_remember("c1", "likes tea")), _calls(_calculate("c2")), LLMResponse(text="done")],
        memory, CalculatorTool(),
    )
    config = _config()
    await graph.ainvoke(_turn("remember I like tea, then work out 6*7"), config)

    result = await graph.ainvoke(_DENY, config)

    assert _tool_results(result)["c2"] == "42"


async def test_a_denied_proposal_does_not_reject_tool_calls_on_a_later_turn():
    memory = InMemoryLongTerm()
    graph = _graph(
        [
            _calls(_remember("c1", "likes tea")),
            LLMResponse(text="Okay, I won't save that."),
            _calls(_calculate("c2")),
            LLMResponse(text="42"),
        ],
        memory, CalculatorTool(),
    )
    config = _config()
    await graph.ainvoke(_turn("remember I like tea"), config)
    await graph.ainvoke(_DENY, config)

    result = await graph.ainvoke(_turn("what is 6*7?"), config)

    assert _tool_results(result)["c2"] == "42"


# --- Every call that needs approval is reviewed before any call runs ---------------


def _two_proposals(memory: InMemoryLongTerm):
    return _graph(
        [
            _calls(_remember("c1", "likes tea"), _remember("c2", "lives in Rome")),
            LLMResponse(text="Noted."),
        ],
        memory, CalculatorTool(),
    )


async def test_every_remember_call_of_a_turn_is_proposed():
    graph = _two_proposals(InMemoryLongTerm())
    config = _config()

    first = await graph.ainvoke(_turn("remember two things"), config)
    second = await graph.ainvoke(_APPROVE, config)

    assert _pending(first) == [
        {"mode": "memory_proposal", "proposed": "likes tea", "call_id": "c1"}
    ]
    assert _pending(second) == [
        {"mode": "memory_proposal", "proposed": "lives in Rome", "call_id": "c2"}
    ]


async def test_proposals_are_written_only_once_every_one_of_the_turn_is_approved():
    memory = InMemoryLongTerm()
    graph = _two_proposals(memory)
    config = _config()
    await graph.ainvoke(_turn("remember two things"), config)

    await graph.ainvoke(_APPROVE, config)
    after_one_approval = await _remembered(memory)
    await graph.ainvoke(_APPROVE, config)

    assert after_one_approval == set()
    assert await _remembered(memory) == {"likes tea", "lives in Rome"}


async def test_denying_one_proposal_writes_none_of_the_turn():
    memory = InMemoryLongTerm()
    graph = _two_proposals(memory)
    config = _config()
    await graph.ainvoke(_turn("remember two things"), config)
    await graph.ainvoke(_APPROVE, config)

    await graph.ainvoke(_DENY, config)

    assert await _remembered(memory) == set()


async def test_proposals_that_share_a_call_id_are_each_reviewed():
    """A provider may repeat a tool-call id: approving one call must not approve its twin."""
    memory = InMemoryLongTerm()
    graph = _graph(
        [
            _calls(_remember("c1", "likes tea"), _remember("c1", "lives in Rome")),
            LLMResponse(text="Noted."),
        ],
        memory, CalculatorTool(),
    )
    config = _config()
    await graph.ainvoke(_turn("remember two things"), config)

    after_one_approval = await graph.ainvoke(_APPROVE, config)

    assert _pending(after_one_approval) == [
        {"mode": "memory_proposal", "proposed": "lives in Rome", "call_id": "c1"}
    ]
    assert await _remembered(memory) == set()


async def test_a_memory_proposal_does_not_cover_another_call_that_needs_approval():
    memory = InMemoryLongTerm()
    graph = _graph(
        [_calls(_remember("c1", "likes tea"), _calculate("c2")), LLMResponse(text="Done.")],
        memory, _ApprovableCalculator(),
    )
    config = _config()
    await graph.ainvoke(_turn("remember I like tea and work out 6*7"), config)

    after_memory_approval = await graph.ainvoke(_APPROVE, config)

    assert _pending(after_memory_approval) == [
        {
            "mode": "approval",
            "tool_calls": [
                {"name": "calculator", "args": {"expression": "6*7"}, "call_id": "c2"}
            ],
        }
    ]
    assert await _remembered(memory) == set()


async def test_an_approval_request_lists_every_call_of_the_turn():
    graph = _graph(
        [_calls(_calculate("c1"), _calculate("c2")), LLMResponse(text="Done.")],
        InMemoryLongTerm(), _ApprovableCalculator(),
    )

    result = await graph.ainvoke(_turn("work it out twice"), _config())

    assert _pending(result) == [
        {
            "mode": "approval",
            "tool_calls": [
                {"name": "calculator", "args": {"expression": "6*7"}, "call_id": "c1"},
                {"name": "calculator", "args": {"expression": "6*7"}, "call_id": "c2"},
            ],
        }
    ]


async def test_the_calls_run_only_once_every_review_of_the_turn_is_approved():
    memory = InMemoryLongTerm()
    three_reviews = _calls(
        _remember("c1", "likes tea"), _remember("c2", "lives in Rome"), _calculate("c3")
    )
    graph = _graph(
        [three_reviews, LLMResponse(text="Done.")], memory, _ApprovableCalculator()
    )
    config = _config()
    asked = [await graph.ainvoke(_turn("remember two things and work out 6*7"), config)]
    for _ in range(2):
        asked.append(await graph.ainvoke(_APPROVE, config))
    written_before_the_last_approval = await _remembered(memory)

    result = await graph.ainvoke(_APPROVE, config)

    assert [[review["mode"] for review in _pending(r)] for r in asked] == [
        ["memory_proposal"], ["memory_proposal"], ["approval"]
    ]
    assert written_before_the_last_approval == set()
    assert _tool_results(result)["c3"] == "42"
    assert await _remembered(memory) == {"likes tea", "lives in Rome"}


# --- An answer applies to the review it was given for -----------------------------


async def test_a_denial_holds_when_tool_registration_changed_between_two_reviews():
    """A restart between two reviews of one turn must not shift answers onto other reviews.

    The turn has two reviews: a memory proposal, then the calculator call. The
    proposal is approved, the process restarts without the memory tools, and the
    second answer is a denial. Nothing may run.
    """
    checkpointer = MemorySaver()
    before = _build(
        [_calls(_remember("c1", "likes tea"), _calculate("c2"))],
        _ApprovableCalculator(), RememberTool(InMemoryLongTerm()), checkpointer=checkpointer,
    )
    after = _build(
        [LLMResponse(text="Understood.")], _ApprovableCalculator(), checkpointer=checkpointer
    )
    config = _config()
    await before.ainvoke(_turn("remember I like tea and work out 6*7"), config)
    await before.ainvoke(_APPROVE, config)

    result = await after.ainvoke(_DENY, config)

    assert _tool_results(result)["c2"] == "Rejected by user."


async def test_an_approval_covers_only_the_review_it_answered_after_a_restart():
    """Approving a memory proposal must not approve a call a restart regrouped with it.

    The turn has two reviews: a memory proposal, then the calculator call. The
    process restarts without the memory tools while the proposal is pending. Worked
    out again from the tools then registered, both calls would share one review,
    and the answer to the proposal would approve it.
    """
    checkpointer = MemorySaver()
    before = _build(
        [_calls(_remember("c1", "likes tea"), _calculate("c2"))],
        _ApprovableCalculator(), RememberTool(InMemoryLongTerm()), checkpointer=checkpointer,
    )
    after = _build([LLMResponse(text="Done.")], _ApprovableCalculator(), checkpointer=checkpointer)
    config = _config()
    await before.ainvoke(_turn("remember I like tea and work out 6*7"), config)

    result = await after.ainvoke(_APPROVE, config)

    assert _pending(result) == [
        {
            "mode": "approval",
            "tool_calls": [
                {"name": "calculator", "args": {"expression": "6*7"}, "call_id": "c2"}
            ],
        }
    ]
    assert "c2" not in _tool_results(result)


@pytest.mark.parametrize(
    ("answer", "told_the_model"),
    [
        pytest.param({"action": "deny"}, "Rejected by user.", id="deny"),
        pytest.param({"action": "feedback", "feedback": "use 7*6"}, "use 7*6", id="feedback"),
    ],
)
async def test_an_answer_is_honoured_when_the_tool_stopped_needing_approval(
    answer, told_the_model
):
    """A restart that changes what needs approval must not drop a pending answer."""
    checkpointer = MemorySaver()
    before = _build(
        [_calls(_calculate("c1"))], _ApprovableCalculator(), checkpointer=checkpointer
    )
    after = _build([LLMResponse(text="Understood.")], CalculatorTool(), checkpointer=checkpointer)
    config = _config()
    await before.ainvoke(_turn("work out 6*7"), config)

    result = await after.ainvoke(Command(resume=answer), config)

    assert told_the_model in _tool_results(result)["c1"]


async def test_an_approval_from_an_abandoned_turn_does_not_cover_a_later_call():
    """Approvals belong to the turn they were given in, even when a call id repeats."""
    memory = InMemoryLongTerm()
    graph = _graph(
        [
            _calls(_remember("c1", "likes tea"), _remember("c2", "lives in Rome")),
            _calls(_remember("c1", "has a cat")),
        ],
        memory, CalculatorTool(),
    )
    config = _config()
    await graph.ainvoke(_turn("remember two things"), config)
    await graph.ainvoke(_APPROVE, config)  # c1 is approved; c2 is still pending

    result = await graph.ainvoke(_turn("never mind, remember I have a cat"), config)

    assert _pending(result) == [
        {"mode": "memory_proposal", "proposed": "has a cat", "call_id": "c1"}
    ]
    assert await _remembered(memory) == set()


async def test_a_thread_saved_without_stored_reviews_is_still_asked():
    """A thread saved before reviews were stored may be waiting on one."""
    memory = InMemoryLongTerm()
    graph = _graph(
        [_calls(_remember("c1", "likes tea")), LLMResponse(text="Not saved.")],
        memory, CalculatorTool(),
    )
    config = _config()
    await graph.ainvoke(_turn("remember I like tea"), config)
    # The thread as such a version left it: the model's calls, and no reviews.
    await graph.aupdate_state(config, {"reviews": None}, as_node="call_model")

    asked = await graph.ainvoke(None, config)
    await graph.ainvoke(_DENY, config)

    assert _pending(asked) == [
        {"mode": "memory_proposal", "proposed": "likes tea", "call_id": "c1"}
    ]
    assert await _remembered(memory) == set()


# --- An answer the gate cannot read is a denial -------------------------------------


@pytest.mark.parametrize(
    "answer",
    [
        pytest.param({"action": ["approve"]}, id="action-is-a-list"),
        pytest.param({"action": {"approve": True}}, id="action-is-an-object"),
        pytest.param({"approved": "false"}, id="approved-is-the-string-false"),
        pytest.param({"approved": 1}, id="approved-is-a-number"),
        pytest.param({"action": "approve", "approved": False}, id="keys-disagree"),
        pytest.param({"action": "retry", "approved": True}, id="unknown-action"),
        pytest.param({"action": ["approve"], "approved": True}, id="unreadable-action"),
    ],
)
async def test_an_answer_that_is_not_a_clear_approval_is_a_denial(answer):
    memory = InMemoryLongTerm()
    graph = _graph(
        [_calls(_remember("c1", "likes tea")), LLMResponse(text="Not saved.")],
        memory, CalculatorTool(),
    )
    config = _config()
    await graph.ainvoke(_turn("remember I like tea"), config)

    result = await graph.ainvoke(Command(resume=answer), config)

    assert _tool_results(result)["c1"] == "Rejected by user."
    assert await _remembered(memory) == set()


# --- What an answer streams to the browser -------------------------------------------


def _drain(config: dict) -> list:
    """Every event the run streamed since the last drain."""
    queue: asyncio.Queue = config["configurable"]["event_queue"]
    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    return events


def _reply_text(events: list) -> str:
    """What the browser shows as the reply: the text of every text event, joined."""
    return "".join(event.text for event in events if event.type in ("token", "thinking"))


@pytest.mark.parametrize(
    "answer",
    [
        pytest.param({"action": "deny"}, id="deny"),
        pytest.param({"action": "feedback", "feedback": "only the tea"}, id="feedback"),
    ],
)
async def test_an_answer_streams_what_the_model_was_told_and_then_only_its_reply(answer):
    """Each call's outcome arrives as that call's result, and the reply is the model's
    text alone. A notice streamed as text ran into the reply, and the browser showed
    the reply twice: once behind the notice, once on its own."""
    graph = _two_proposals(InMemoryLongTerm())
    config = _config()
    await graph.ainvoke(_turn("remember two things"), config)
    _drain(config)

    result = await graph.ainvoke(Command(resume=answer), config)

    events = _drain(config)
    streamed_results = {e.call_id: e.text for e in events if e.type == "tool_result"}
    assert streamed_results == _tool_results(result)
    assert _reply_text(events) == "Noted."
