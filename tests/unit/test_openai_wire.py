"""Unit tests for the pure OpenAI chat wire-format helpers (no network)."""
from __future__ import annotations

from types import SimpleNamespace

from harness.adapters.llm.openai_wire import (
    build_request_messages,
    message_to_wire,
    parse_completion,
    spec_to_wire,
)
from harness.adapters.llm.parsers import NativeToolCallParser, PromptedToolCallParser
from harness.core.types import Message, Role, ToolCall, ToolSpec

TOOLS = [
    ToolSpec(
        name="calculator",
        description="do math",
        parameters={
            "type": "object",
            "properties": {"expression": {"type": "string"}},
        },
    )
]


def test_message_to_wire_user():
    wire = message_to_wire(Message(Role.USER, "hi"))
    assert wire == {"role": "user", "content": "hi"}


def test_message_to_wire_tool_role():
    msg = Message(Role.TOOL, "42", tool_call_id="c1", name="calculator")
    wire = message_to_wire(msg)
    assert wire == {"role": "tool", "tool_call_id": "c1", "content": "42"}


def test_message_to_wire_assistant_with_tool_calls():
    msg = Message(
        Role.ASSISTANT,
        "",
        tool_calls=(ToolCall(id="c1", name="calculator", arguments={"expression": "6*7"}),),
    )
    wire = message_to_wire(msg)
    assert wire["role"] == "assistant"
    assert wire["tool_calls"][0]["id"] == "c1"
    assert wire["tool_calls"][0]["function"]["name"] == "calculator"
    assert wire["tool_calls"][0]["function"]["arguments"] == '{"expression": "6*7"}'


def test_spec_to_wire():
    wire = spec_to_wire(TOOLS[0])
    assert wire["type"] == "function"
    assert wire["function"]["name"] == "calculator"
    assert wire["function"]["parameters"] == TOOLS[0].parameters


def test_build_request_messages_native_leaves_system_untouched():
    messages = [Message(Role.SYSTEM, "you are helpful"), Message(Role.USER, "hi")]
    wire = build_request_messages(messages, NativeToolCallParser(), TOOLS)
    assert wire[0] == {"role": "system", "content": "you are helpful"}


def test_build_request_messages_prompted_appends_addendum():
    messages = [Message(Role.SYSTEM, "you are helpful"), Message(Role.USER, "hi")]
    wire = build_request_messages(messages, PromptedToolCallParser(), TOOLS)
    assert wire[0]["content"].startswith("you are helpful")
    assert "tool_call" in wire[0]["content"]
    # original Message is not mutated
    assert messages[0].content == "you are helpful"


def test_build_request_messages_no_tools_no_addendum():
    messages = [Message(Role.SYSTEM, "sys"), Message(Role.USER, "hi")]
    wire = build_request_messages(messages, PromptedToolCallParser(), None)
    assert wire[0] == {"role": "system", "content": "sys"}


def _completion(content="", tool_calls=None, usage=None, finish_reason="stop"):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=usage)


def test_parse_completion_plain_text():
    resp = parse_completion(_completion(content="42"))
    assert resp.text == "42"
    assert resp.tool_calls == ()
    assert resp.wants_tools is False


def test_parse_completion_tool_call():
    tc = SimpleNamespace(
        id="c1",
        function=SimpleNamespace(name="calculator", arguments='{"expression": "6*7"}'),
    )
    resp = parse_completion(_completion(content="", tool_calls=[tc]))
    assert resp.wants_tools is True
    assert resp.tool_calls[0].name == "calculator"
    assert resp.tool_calls[0].arguments == {"expression": "6*7"}


def test_parse_completion_malformed_args_become_empty_dict():
    tc = SimpleNamespace(
        id="c1", function=SimpleNamespace(name="calculator", arguments="not json")
    )
    resp = parse_completion(_completion(tool_calls=[tc]))
    assert resp.tool_calls[0].arguments == {}


def test_parse_completion_usage():
    usage = SimpleNamespace(prompt_tokens=3, completion_tokens=5, total_tokens=8)
    resp = parse_completion(_completion(content="hi", usage=usage))
    assert resp.usage == {
        "prompt_tokens": 3,
        "completion_tokens": 5,
        "total_tokens": 8,
    }
