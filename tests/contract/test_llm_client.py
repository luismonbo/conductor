"""Contract suite for LLMClient. Every adapter must pass this.

It asserts protocol-level guarantees (not model quality) and runs offline by
injecting an OpenAI-shaped fake transport:
  - the adapter is a runtime LLMClient and model_id reflects config
  - a plain completion yields text with wants_tools == False
  - a completion carrying tool_calls is normalized to core ToolCall(s)

Add new adapters to ADAPTER_FACTORIES and they inherit the whole contract. Azure
joins once it grows a client-injection seam like OpenAICompatibleClient's (today
it builds its SDK client eagerly in __init__).
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Callable

import pytest

from harness.adapters.llm.openai_compatible import OpenAICompatibleClient
from harness.adapters.llm.parsers import NativeToolCallParser
from harness.core.llm.client import LLMClient
from harness.core.types import Message, Role, ToolSpec

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


# Extension point: add (id, factory) tuples for each new LLM adapter.
ADAPTER_FACTORIES: list[tuple[str, Callable[[object], LLMClient]]] = [
    ("openai_compatible", _openai_compatible_factory),
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
