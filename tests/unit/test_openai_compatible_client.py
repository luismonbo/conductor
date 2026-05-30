"""Unit test for OpenAICompatibleClient using an injected fake OpenAI client.

No network and no real `openai` client are constructed: we inject a stand-in
exposing `.chat.completions.create`, so this verifies the adapter wires the
request and parses the response correctly.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from harness.adapters.llm.openai_compatible import OpenAICompatibleClient
from harness.adapters.llm.parsers import NativeToolCallParser
from harness.core.types import Message, Role, ToolSpec

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


class _FakeCompletions:
    def __init__(self, completion):
        self._completion = completion
        self.captured_kwargs: dict = {}

    async def create(self, **kwargs):
        self.captured_kwargs = kwargs
        return self._completion


class _FakeOpenAI:
    def __init__(self, completion):
        self.chat = SimpleNamespace(completions=_FakeCompletions(completion))


def _completion(content="", tool_calls=None, finish_reason="stop"):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=None)


def test_model_id_is_the_configured_model():
    client = OpenAICompatibleClient(
        base_url="http://x/v1",
        model="gemma4:a2b",
        parser=NativeToolCallParser(),
        client=_FakeOpenAI(_completion(content="hi")),
    )
    assert client.model_id == "gemma4:a2b"


@pytest.mark.asyncio
async def test_generate_returns_text():
    fake = _FakeOpenAI(_completion(content="42"))
    client = OpenAICompatibleClient(
        base_url="http://x/v1", model="m", parser=NativeToolCallParser(), client=fake
    )
    resp = await client.generate([Message(Role.USER, "what is 6*7?")])
    assert resp.text == "42"
    assert resp.wants_tools is False
    # model name is forwarded on the request
    assert fake.chat.completions.captured_kwargs["model"] == "m"


@pytest.mark.asyncio
async def test_generate_passes_tools_when_present():
    fake = _FakeOpenAI(_completion(content="ok"))
    client = OpenAICompatibleClient(
        base_url="http://x/v1", model="m", parser=NativeToolCallParser(), client=fake
    )
    await client.generate([Message(Role.USER, "hi")], tools=TOOLS)
    kwargs = fake.chat.completions.captured_kwargs
    assert kwargs["tool_choice"] == "auto"
    assert kwargs["tools"][0]["function"]["name"] == "calculator"


@pytest.mark.asyncio
async def test_generate_omits_tools_when_none():
    fake = _FakeOpenAI(_completion(content="ok"))
    client = OpenAICompatibleClient(
        base_url="http://x/v1", model="m", parser=NativeToolCallParser(), client=fake
    )
    await client.generate([Message(Role.USER, "hi")])
    assert "tools" not in fake.chat.completions.captured_kwargs
