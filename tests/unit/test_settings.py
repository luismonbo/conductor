"""Pin the OpenAI-compatible backend settings and their defaults."""
from __future__ import annotations

import pytest

from harness.config.settings import Settings, get_settings

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


def test_get_settings_without_target_matches_plain_defaults(monkeypatch, tmp_path):
    # HARNESS_LLM_BACKEND explicitly cleared, not just left unset: this
    # repo's real .env sets it to "azure", and this codebase has known
    # import-time side effects (pymilvus's own load_dotenv(), main.py's
    # explicit one) that load .env straight into real os.environ — confirmed
    # by this test passing alone but failing once collected with the rest of
    # the suite. chdir alone doesn't help; the pollution is already in
    # os.environ by the time this runs, not something a fresh .env lookup
    # would see.
    monkeypatch.delenv("HARNESS_TARGET", raising=False)
    monkeypatch.delenv("HARNESS_LLM_BACKEND", raising=False)
    monkeypatch.chdir(tmp_path)  # no .env here — isolates from the real one

    settings = get_settings()

    assert settings.llm_backend == "fake"


def test_target_file_overrides_a_field_left_at_default(monkeypatch, tmp_path):
    monkeypatch.delenv("HARNESS_RAG_COLLECTION", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config" / "targets").mkdir(parents=True)
    (tmp_path / "config" / "targets" / "acme.yaml").write_text("rag_collection: acme-docs\n")
    monkeypatch.setenv("HARNESS_TARGET", "acme")

    settings = get_settings()

    assert settings.rag_collection == "acme-docs"


def test_real_env_var_still_beats_the_target_file(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config" / "targets").mkdir(parents=True)
    (tmp_path / "config" / "targets" / "acme.yaml").write_text("rag_collection: acme-docs\n")
    monkeypatch.setenv("HARNESS_TARGET", "acme")
    monkeypatch.setenv("HARNESS_RAG_COLLECTION", "explicit-collection")

    settings = get_settings()

    assert settings.rag_collection == "explicit-collection"


def test_unknown_field_in_target_file_raises(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config" / "targets").mkdir(parents=True)
    (tmp_path / "config" / "targets" / "acme.yaml").write_text("not_a_real_field: 1\n")
    monkeypatch.setenv("HARNESS_TARGET", "acme")

    with pytest.raises(ValueError, match="not_a_real_field"):
        get_settings()
