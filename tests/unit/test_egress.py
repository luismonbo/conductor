"""security/egress.py: what a browser may receive (SL-3)."""
from __future__ import annotations

import dataclasses

import pytest

from harness.core.llm.errors import ContentFiltered
from harness.core.types import AgentEvent, Message, Role, ToolCall
from harness.security.egress import (
    CONTENT_FILTERED_TEXT,
    UNAVAILABLE_TEXT,
    error_event,
    to_browser,
    transcript_for_browser,
)

_CALL = AgentEvent(
    type="tool_call", name="search_documents",
    args={"query": "azure", "collection": "docs"}, call_id="c1",
)
_RESULT = AgentEvent(
    type="tool_result", name="search_documents", call_id="c1", is_error=False,
    text='[1] (score: 0.81, source: "Azure adapter", section: n/a)\nchunk text',
)


def test_the_two_sentences_are_the_specs():
    assert CONTENT_FILTERED_TEXT == (
        "I can't respond to that message. Your next message will start a new conversation."
    )
    assert UNAVAILABLE_TEXT == "The assistant is unavailable right now. Please try again later."


def test_a_content_filter_hit_gets_its_own_sentence():
    event = error_event(ContentFiltered("prompt", ("jailbreak",)))

    assert event == AgentEvent(
        type="error", text=CONTENT_FILTERED_TEXT, stopped_reason="content_filter"
    )


@pytest.mark.parametrize("exc", [
    ValueError("/app/data/raw/docs/010-azure.md not found"),
    RuntimeError("Error code: 400 - {'error': {'innererror': {'content_filter_result': {}}}}"),
    Exception(),
], ids=repr)
def test_every_other_failure_gets_the_generic_sentence_never_its_text(exc):
    assert error_event(exc) == AgentEvent(
        type="error", text=UNAVAILABLE_TEXT, stopped_reason="error"
    )


@pytest.mark.parametrize("event_type", ["token", "final", "error", "thinking", "interrupt"])
@pytest.mark.parametrize("show_tool_details", [False, True])
def test_conversation_events_pass_unchanged(event_type, show_tool_details):
    event = AgentEvent(type=event_type, text="hello", args={"mode": "approval"}, stopped_reason="x")

    assert to_browser(event, show_tool_details=show_tool_details) is event


def test_tool_events_keep_only_their_names_when_details_are_off():
    assert to_browser(_CALL, show_tool_details=False) == dataclasses.replace(_CALL, args={})
    assert to_browser(_RESULT, show_tool_details=False) == dataclasses.replace(_RESULT, text="")


def test_a_tool_errors_flag_survives_without_its_text():
    failed = dataclasses.replace(_RESULT, is_error=True, text="Tool 'search_documents' failed.")

    assert to_browser(failed, show_tool_details=False) == dataclasses.replace(failed, text="")


def test_tool_events_pass_unchanged_when_details_are_on():
    assert to_browser(_CALL, show_tool_details=True) is _CALL
    assert to_browser(_RESULT, show_tool_details=True) is _RESULT


@pytest.mark.parametrize("show_tool_details", [False, True])
def test_an_unknown_event_type_never_reaches_a_browser(show_tool_details):
    # e.g. #8's sub-agent steps, until egress has a rule for them
    event = AgentEvent(type="subagent_step", text="internal")

    assert to_browser(event, show_tool_details=show_tool_details) is None


_THREAD = [
    Message(role=Role.SYSTEM, content="the system prompt"),
    Message(role=Role.USER, content="what did Luis build on Azure?"),
    Message(
        role=Role.ASSISTANT, content="",
        tool_calls=(ToolCall(id="c1", name="search_documents", arguments={"query": "azure"}),),
    ),
    Message(role=Role.TOOL, content="[1] (source: ...)", tool_call_id="c1", name="search_documents"),
    Message(role=Role.ASSISTANT, content="An embeddings adapter."),
]


def test_a_transcript_without_details_blanks_tool_arguments_and_results():
    assert transcript_for_browser(_THREAD, show_tool_details=False) == [
        {"role": "user", "content": "what did Luis build on Azure?", "tool_calls": [],
         "tool_call_id": "", "name": ""},
        {"role": "assistant", "content": "",
         "tool_calls": [{"name": "search_documents", "args": {}, "call_id": "c1"}],
         "tool_call_id": "", "name": ""},
        {"role": "tool", "content": "", "tool_calls": [], "tool_call_id": "c1",
         "name": "search_documents"},
        {"role": "assistant", "content": "An embeddings adapter.", "tool_calls": [],
         "tool_call_id": "", "name": ""},
    ]


def test_a_transcript_with_details_keeps_them_and_still_drops_the_system_prompt():
    out = transcript_for_browser(_THREAD, show_tool_details=True)

    assert [m["role"] for m in out] == ["user", "assistant", "tool", "assistant"]
    assert out[1]["tool_calls"][0]["args"] == {"query": "azure"}
    assert out[2]["content"] == "[1] (source: ...)"
