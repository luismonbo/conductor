"""Pin the OpenAI-compatible backend settings and their defaults."""
from __future__ import annotations

from harness.config.settings import Settings

_LLM_ENV_KEYS = [
    "HARNESS_LLM_BACKEND",
    "HARNESS_LLM_BASE_URL",
    "HARNESS_LLM_MODEL",
    "HARNESS_LLM_API_KEY",
]


def test_openai_compatible_defaults(monkeypatch):
    # Isolate from any ambient HARNESS_* env so we assert real defaults.
    for key in _LLM_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)

    settings = Settings(_env_file=None)

    assert settings.llm_backend == "fake"
    assert settings.llm_base_url == "http://localhost:8080/v1"
    assert settings.llm_model == "gemma4:a2b"
    assert settings.llm_api_key == ""


def test_env_overrides_backend(monkeypatch):
    monkeypatch.setenv("HARNESS_LLM_BACKEND", "openai_compatible")
    monkeypatch.setenv("HARNESS_LLM_BASE_URL", "http://example.test/v1")

    settings = Settings(_env_file=None)

    assert settings.llm_backend == "openai_compatible"
    assert settings.llm_base_url == "http://example.test/v1"


def test_default_checkpointer_is_sqlite():
    s = Settings(_env_file=None)
    assert s.checkpointer == "sqlite"


def test_default_checkpointer_url():
    s = Settings(_env_file=None)
    assert s.checkpointer_url == "./harness.sqlite"


def test_default_agent_is_default():
    s = Settings(_env_file=None)
    assert s.agent == "default"
