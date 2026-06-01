"""Unit tests for FakeLLMClient, including stream()."""
from __future__ import annotations

import pytest

from harness.adapters.llm.fake import FakeLLMClient
from harness.core.types import LLMResponse, Message, Role, ToolCall


def _user(text: str) -> Message:
    return Message(role=Role.USER, content=text)


@pytest.mark.asyncio
async def test_stream_yields_tokens_then_response():
    client = FakeLLMClient([LLMResponse(text="Hello world")])
    items = []
    async for item in client.stream([_user("hi")]):
        items.append(item)

    tokens = [i for i in items if isinstance(i, str)]
    responses = [i for i in items if isinstance(i, LLMResponse)]

    assert len(responses) == 1
    assert len(tokens) > 0
    assert "".join(tokens).strip() == "Hello world"


@pytest.mark.asyncio
async def test_stream_final_response_carries_tool_calls():
    tc = ToolCall(id="c1", name="calculator", arguments={"expression": "2+2"})
    client = FakeLLMClient([LLMResponse(text="", tool_calls=(tc,))])
    items = []
    async for item in client.stream([_user("2+2?")]):
        items.append(item)

    final = next(i for i in items if isinstance(i, LLMResponse))
    assert final.wants_tools is True
    assert final.tool_calls[0].name == "calculator"


@pytest.mark.asyncio
async def test_stream_empty_queue_returns_fallback():
    client = FakeLLMClient([])
    items = []
    async for item in client.stream([_user("hi")]):
        items.append(item)

    final = next(i for i in items if isinstance(i, LLMResponse))
    assert "no scripted response" in final.text
