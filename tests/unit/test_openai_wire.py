"""Unit tests for the pure OpenAI chat wire-format helpers (no network)."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import openai
import pytest

from harness.adapters.llm.openai_wire import (
    _OpenAIBaseClient,
    build_request_messages,
    message_to_wire,
    parse_completion,
    spec_to_wire,
)
from harness.adapters.llm.parsers import NativeToolCallParser, PromptedToolCallParser
from harness.core.llm.errors import ContentFiltered
from harness.core.types import LLMResponse, Message, Role, ToolCall, ToolSpec

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


async def test_stream_populates_usage_from_terminal_chunk():
    """stream() must collect usage from the terminal chunk and include it in LLMResponse."""

    async def _fake_stream():
        # text chunk
        yield SimpleNamespace(
            choices=[SimpleNamespace(
                delta=SimpleNamespace(content="hello", tool_calls=None),
                finish_reason=None,
            )],
            usage=None,
        )
        # finish chunk
        yield SimpleNamespace(
            choices=[SimpleNamespace(
                delta=SimpleNamespace(content=None, tool_calls=None),
                finish_reason="stop",
            )],
            usage=None,
        )
        # terminal usage chunk (empty choices, carries usage)
        yield SimpleNamespace(
            choices=[],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

    mock_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=AsyncMock(return_value=_fake_stream()))
        )
    )
    adapter = _OpenAIBaseClient(mock_client, "test-model", NativeToolCallParser(), 0.0)

    results = []
    async for item in adapter.stream([Message(Role.USER, "hi")]):
        results.append(item)

    final = next(r for r in results if isinstance(r, LLMResponse))
    assert final.usage == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }


# The error object Azure returns when its content filter refuses a prompt (HTTP 400).
_AZURE_FILTER_ERROR = {
    "message": "The response was filtered due to the prompt triggering Azure OpenAI's content management policy.",
    "type": None,
    "param": "prompt",
    "code": "content_filter",
    "status": 400,
    "innererror": {
        "code": "ResponsibleAIPolicyViolation",
        "content_filter_result": {
            "hate": {"filtered": False, "severity": "safe"},
            "jailbreak": {"filtered": True, "detected": True},
            "violence": {"filtered": False, "severity": "safe"},
        },
    },
}


def _bad_request(error: dict) -> openai.BadRequestError:
    """The exception openai 2.x raises for a 400, built the way the SDK builds it."""
    response = httpx.Response(
        400,
        json={"error": error},
        request=httpx.Request("POST", "https://example.test/v1/chat/completions"),
    )
    return openai.BadRequestError(f"Error code: 400 - {{'error': {error}}}", response=response, body=error)


def _client_raising(exc: Exception) -> SimpleNamespace:
    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(side_effect=exc)))
    )


async def test_a_filtered_prompt_becomes_content_filtered_from_generate():
    original = _bad_request(_AZURE_FILTER_ERROR)
    adapter = _OpenAIBaseClient(_client_raising(original), "m", NativeToolCallParser(), 0.0)

    with pytest.raises(ContentFiltered) as info:
        await adapter.generate([Message(Role.USER, "hi")])

    assert info.value.stage == "prompt"
    assert info.value.categories == ("jailbreak",)
    assert info.value.__cause__ is original  # the server log keeps the provider's error


async def test_a_filtered_prompt_becomes_content_filtered_from_stream():
    adapter = _OpenAIBaseClient(
        _client_raising(_bad_request(_AZURE_FILTER_ERROR)), "m", NativeToolCallParser(), 0.0
    )

    with pytest.raises(ContentFiltered) as info:
        async for _ in adapter.stream([Message(Role.USER, "hi")]):
            pass

    assert info.value.categories == ("jailbreak",)


async def test_a_filter_error_without_verdicts_is_still_content_filtered():
    bare = {"message": "The response was filtered", "type": None, "param": "prompt",
            "code": "content_filter", "status": 400}
    adapter = _OpenAIBaseClient(_client_raising(_bad_request(bare)), "m", NativeToolCallParser(), 0.0)

    with pytest.raises(ContentFiltered) as info:
        await adapter.generate([Message(Role.USER, "hi")])

    assert info.value.categories == ()


async def test_other_provider_errors_propagate_unchanged():
    other = _bad_request({"message": "context too long", "type": None, "param": "messages",
                          "code": "context_length_exceeded", "status": 400})
    adapter = _OpenAIBaseClient(_client_raising(other), "m", NativeToolCallParser(), 0.0)

    with pytest.raises(openai.BadRequestError) as info:
        await adapter.generate([Message(Role.USER, "hi")])

    assert info.value is other


def test_content_filtered_names_its_stage_and_categories_never_content():
    assert str(ContentFiltered("prompt", ("jailbreak", "violence"))) == (
        "content filter (prompt): jailbreak, violence"
    )
    assert str(ContentFiltered("completion")) == "content filter (completion): unspecified"
