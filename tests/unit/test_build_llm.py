"""build_llm must select the OpenAICompatibleClient for the openai_compatible backend."""
from __future__ import annotations

import pytest

from harness.adapters.llm.openai_compatible import OpenAICompatibleClient
from harness.adapters.llm.parsers import NativeToolCallParser
from harness.config.settings import Settings
from harness.orchestration.build import build_llm


def test_build_llm_selects_openai_compatible():
    settings = Settings(
        _env_file=None,
        llm_backend="openai_compatible",
        llm_base_url="http://localhost:8080/v1",
        llm_model="gemma4:a2b",
    )
    client = build_llm(settings, NativeToolCallParser())
    assert isinstance(client, OpenAICompatibleClient)
    assert client.model_id == "gemma4:a2b"


def test_build_llm_unknown_backend_raises():
    settings = Settings(_env_file=None, llm_backend="nope")
    with pytest.raises(ValueError, match="Unknown llm_backend"):
        build_llm(settings, NativeToolCallParser())
