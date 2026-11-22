"""reasoning_effort: sent on every Azure chat call when set, left out otherwise.

gpt-6-luna takes function tools over Chat Completions only with reasoning_effort
"none" (a 400 on every request otherwise), and the website agent always sends tools.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from harness.adapters.llm.azure_openai import AzureOpenAIClient
from harness.adapters.llm.parsers import NativeToolCallParser
from harness.config.settings import Settings
from harness.config.targets import resolve_target_overrides
from harness.core.types import Message, Role
from harness.orchestration.build import build_llm

_MSGS = [Message(role=Role.USER, content="hi")]
_BASE = dict(_env_file=None, api_key="test-key", input_guard_enabled=False)
_AZURE = dict(
    llm_backend="azure", azure_endpoint="https://example.openai.azure.com",
    azure_api_key="test-azure-key",
)
_TARGETS = Path(__file__).resolve().parents[2] / "config" / "targets"


class _Recorder:
    """Stands in for the SDK's chat.completions and keeps each call's arguments."""

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


async def _sent(reasoning_effort: str | None) -> list[dict]:
    """The arguments one generate() call and one stream() call send."""
    recorder = _Recorder()
    client = AzureOpenAIClient(
        deployment="d", endpoint="https://x", api_version="2024-10-21",
        parser=NativeToolCallParser(),
        client=SimpleNamespace(chat=SimpleNamespace(completions=recorder)),
        reasoning_effort=reasoning_effort,
    )
    await client.generate(_MSGS)
    async for _ in client.stream(_MSGS):
        pass
    return recorder.kwargs


async def test_generate_and_stream_send_reasoning_effort_when_set():
    sent = await _sent("none")

    assert [kwargs["reasoning_effort"] for kwargs in sent] == ["none", "none"]


async def test_no_reasoning_effort_is_sent_by_default():
    for kwargs in await _sent(None):
        assert "reasoning_effort" not in kwargs


def test_build_llm_passes_the_setting_to_azure():
    settings = Settings(**_BASE, **_AZURE, azure_reasoning_effort="none")

    assert build_llm(settings, NativeToolCallParser())._reasoning_effort == "none"


def test_an_empty_setting_sends_nothing(monkeypatch):
    # A real environment variable would beat the default, so clear it.
    monkeypatch.delenv("HARNESS_AZURE_REASONING_EFFORT", raising=False)
    settings = Settings(**_BASE, **_AZURE)

    assert build_llm(settings, NativeToolCallParser())._reasoning_effort is None


def test_the_setting_refuses_a_reasoning_level():
    # The adapter always sends temperature 0 and tools over Chat Completions, which
    # a reasoning level above "none" can't take: refuse it at startup, not on every call.
    with pytest.raises(ValidationError):
        Settings(**_BASE, azure_reasoning_effort="low")


def test_the_public_site_turns_reasoning_off_for_gpt_6_luna():
    # The pair goes together: without "none", gpt-6-luna refuses every chat request.
    overrides = resolve_target_overrides("luismonbo-com", base_dir=_TARGETS)

    assert overrides["azure_deployment"] == "gpt-6-luna"
    assert overrides["azure_reasoning_effort"] == "none"
