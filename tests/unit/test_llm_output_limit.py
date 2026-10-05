"""The output-token limit is sent under each backend's parameter name when > 0, omitted at 0."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from harness.adapters.llm.azure_openai import AzureOpenAIClient
from harness.adapters.llm.openai_compatible import OpenAICompatibleClient
from harness.adapters.llm.parsers import NativeToolCallParser
from harness.config.settings import Settings
from harness.core.types import Message, Role
from harness.orchestration.build import build_llm

_MSGS = [Message(role=Role.USER, content="hi")]


class _Recorder:
    def __init__(self) -> None:
        self.kwargs: list[dict] = []

    async def create(self, **kwargs):
        self.kwargs.append(kwargs)
        if kwargs.get("stream"):
            return _empty_stream()
        msg = SimpleNamespace(content="ok", tool_calls=None)
        choice = SimpleNamespace(message=msg, finish_reason="stop")
        return SimpleNamespace(choices=[choice], usage=None)


async def _empty_stream():
    for chunk in ():
        yield chunk


def _sdk(recorder: _Recorder) -> SimpleNamespace:
    return SimpleNamespace(chat=SimpleNamespace(completions=recorder))


def _compatible(recorder: _Recorder, limit: int):
    return OpenAICompatibleClient(
        base_url="http://x/v1", model="m", parser=NativeToolCallParser(),
        client=_sdk(recorder), max_output_tokens=limit,
    )


def _azure(recorder: _Recorder, limit: int):
    return AzureOpenAIClient(
        deployment="d", endpoint="https://x", api_version="2024-10-21",
        parser=NativeToolCallParser(), client=_sdk(recorder), max_output_tokens=limit,
    )


async def _call_both(client) -> None:
    await client.generate(_MSGS)
    async for _ in client.stream(_MSGS):
        pass


@pytest.mark.parametrize(
    "make,param", [(_compatible, "max_tokens"), (_azure, "max_completion_tokens")]
)
async def test_generate_and_stream_send_the_limit(make, param):
    recorder = _Recorder()
    await _call_both(make(recorder, 1500))
    assert [kwargs[param] for kwargs in recorder.kwargs] == [1500, 1500]


@pytest.mark.parametrize("make", [_compatible, _azure])
async def test_zero_sends_no_limit(make):
    recorder = _Recorder()
    await _call_both(make(recorder, 0))
    for kwargs in recorder.kwargs:
        assert "max_tokens" not in kwargs
        assert "max_completion_tokens" not in kwargs


def test_build_llm_passes_the_setting():
    settings = Settings(
        _env_file=None, api_key="test-key", input_guard_enabled=False,
        llm_backend="openai_compatible", llm_base_url="http://localhost:8080/v1",
        llm_model="m", default_model="", max_output_tokens=777,
    )
    client = build_llm(settings, NativeToolCallParser())
    assert client._max_output_tokens == 777


def test_build_llm_passes_the_setting_to_azure():
    settings = Settings(
        _env_file=None, api_key="test-key", input_guard_enabled=False,
        llm_backend="azure", azure_endpoint="https://example.openai.azure.com",
        azure_api_key="test-azure-key", max_output_tokens=777,
    )
    client = build_llm(settings, NativeToolCallParser())
    assert client._max_output_tokens == 777
    assert client._OUTPUT_LIMIT_PARAM == "max_completion_tokens"
