"""Contract suite for LLMClient. Every adapter must pass this.

It asserts protocol-level guarantees (not model quality) and runs offline by
injecting an OpenAI-shaped fake transport:
  - the adapter is a runtime LLMClient and model_id reflects config
  - a plain completion yields text with wants_tools == False
  - a completion carrying tool_calls is normalized to core ToolCall(s)

Add new adapters to ADAPTER_FACTORIES and they inherit the whole contract.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Callable

import pytest

from harness.adapters.llm.azure_openai import AzureOpenAIClient
from harness.adapters.llm.openai_compatible import OpenAICompatibleClient
from harness.adapters.llm.parsers import NativeToolCallParser
from harness.core.llm.client import LLMClient
from harness.core.types import LLMResponse, Message, Role, ToolSpec

_CALC = ToolSpec(
    name="calculator",
    description="do math",
    parameters={"type": "object", "properties": {"expression": {"type": "string"}}},
)


def _completion(content="", tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], usage=None)


def _fake_openai(completion):
    async def create(**kwargs):
        return completion

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def _openai_compatible_factory(completion) -> LLMClient:
    return OpenAICompatibleClient(
        base_url="http://x/v1",
        model="contract-model",
        parser=NativeToolCallParser(),
        client=_fake_openai(completion),
    )


def _azure_factory(completion) -> LLMClient:
    return AzureOpenAIClient(
        deployment="contract-model",
        endpoint="http://x",
        api_version="2024-02-01",
        parser=NativeToolCallParser(),
        client=_fake_openai(completion),
    )


# Extension point: add (id, factory) tuples for each new LLM adapter.
ADAPTER_FACTORIES: list[tuple[str, Callable[[object], LLMClient]]] = [
    ("openai_compatible", _openai_compatible_factory),
    ("azure", _azure_factory),
]


@pytest.fixture(params=[f for _, f in ADAPTER_FACTORIES], ids=[i for i, _ in ADAPTER_FACTORIES])
def make_client(request):
    return request.param


def test_adapter_satisfies_protocol(make_client):
    client = make_client(_completion(content="hi"))
    assert isinstance(client, LLMClient)


def test_model_id_reflects_config(make_client):
    client = make_client(_completion(content="hi"))
    assert client.model_id == "contract-model"


@pytest.mark.asyncio
async def test_plain_completion_has_no_tool_calls(make_client):
    client = make_client(_completion(content="42"))
    resp = await client.generate([Message(Role.USER, "q")])
    assert resp.text == "42"
    assert resp.wants_tools is False
    assert resp.tool_calls == ()


@pytest.mark.asyncio
async def test_tool_call_completion_is_normalized(make_client):
    tc = SimpleNamespace(
        id="c1",
        function=SimpleNamespace(name="calculator", arguments='{"expression": "6*7"}'),
    )
    client = make_client(_completion(content="", tool_calls=[tc]))
    resp = await client.generate([Message(Role.USER, "6*7?")], tools=[_CALC])
    assert resp.wants_tools is True
    assert resp.tool_calls[0].name == "calculator"
    assert resp.tool_calls[0].arguments == {"expression": "6*7"}


# --- streaming helpers ---


class _FakeStreamChunk:
    """Minimal duck-type for ChatCompletionChunk."""

    def __init__(self, content=None, tool_calls=None, finish_reason=None):
        self.choices = [
            type("Choice", (), {
                "delta": type("Delta", (), {"content": content, "tool_calls": tool_calls})(),
                "finish_reason": finish_reason,
            })()
        ]


class _AsyncIter:
    def __init__(self, items):
        self._items = list(items)
        self._idx = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._idx >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._idx]
        self._idx += 1
        return item


def _fake_streaming_openai(chunks):
    async def create(**kwargs):
        return _AsyncIter(chunks)

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def _openai_streaming_factory(chunks) -> LLMClient:
    return OpenAICompatibleClient(
        base_url="http://x/v1",
        model="contract-model",
        parser=NativeToolCallParser(),
        client=_fake_streaming_openai(chunks),
    )


def _azure_streaming_factory(chunks) -> LLMClient:
    return AzureOpenAIClient(
        deployment="contract-model",
        endpoint="http://x",
        api_version="2024-02-01",
        parser=NativeToolCallParser(),
        client=_fake_streaming_openai(chunks),
    )


STREAMING_FACTORIES: list[tuple[str, Callable]] = [
    ("openai_compatible", _openai_streaming_factory),
    ("azure", _azure_streaming_factory),
]


@pytest.fixture(
    params=[f for _, f in STREAMING_FACTORIES],
    ids=[i for i, _ in STREAMING_FACTORIES],
)
def make_streaming_client(request):
    return request.param


@pytest.mark.asyncio
async def test_stream_yields_tokens_then_final_response(make_streaming_client):
    chunks = [
        _FakeStreamChunk(content="Hello"),
        _FakeStreamChunk(content=" world"),
        _FakeStreamChunk(finish_reason="stop"),
    ]
    client = make_streaming_client(chunks)
    items = []
    async for item in client.stream([Message(Role.USER, "hi")]):
        items.append(item)

    tokens = [i for i in items if isinstance(i, str)]
    responses = [i for i in items if isinstance(i, LLMResponse)]

    assert tokens == ["Hello", " world"]
    assert len(responses) == 1
    assert responses[0].text == "Hello world"
    assert responses[0].wants_tools is False


@pytest.mark.asyncio
async def test_stream_assembles_tool_call_from_deltas(make_streaming_client):
    tc_delta_1 = SimpleNamespace(
        index=0,
        id="call_abc",
        function=SimpleNamespace(name="calculator", arguments=""),
    )
    tc_delta_2 = SimpleNamespace(
        index=0,
        id=None,
        function=SimpleNamespace(name=None, arguments='{"expression": "6*7"}'),
    )
    chunks = [
        _FakeStreamChunk(tool_calls=[tc_delta_1]),
        _FakeStreamChunk(tool_calls=[tc_delta_2]),
        _FakeStreamChunk(finish_reason="tool_calls"),
    ]
    client = make_streaming_client(chunks)
    items = []
    async for item in client.stream([Message(Role.USER, "6*7?")], tools=[_CALC]):
        items.append(item)

    final = next(i for i in items if isinstance(i, LLMResponse))
    assert final.wants_tools is True
    assert final.tool_calls[0].name == "calculator"
    assert final.tool_calls[0].arguments == {"expression": "6*7"}
    assert final.tool_calls[0].id == "call_abc"
