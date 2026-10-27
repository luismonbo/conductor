"""Pin the OpenAI-compatible backend settings and their defaults."""
from __future__ import annotations

import re

import pytest
from pydantic import ValidationError

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
    monkeypatch.setenv("HARNESS_API_KEY", "test-key")

    settings = Settings(_env_file=None)

    assert settings.llm_backend == "fake"
    assert settings.llm_base_url == "http://localhost:8080/v1"
    assert settings.llm_model == "gemma4:a2b"
    assert settings.llm_api_key == ""


def test_env_overrides_backend(monkeypatch):
    monkeypatch.setenv("HARNESS_LLM_BACKEND", "openai_compatible")
    monkeypatch.setenv("HARNESS_LLM_BASE_URL", "http://example.test/v1")
    monkeypatch.setenv("HARNESS_API_KEY", "test-key")

    settings = Settings(_env_file=None)

    assert settings.llm_backend == "openai_compatible"
    assert settings.llm_base_url == "http://example.test/v1"


def test_default_checkpointer_is_sqlite():
    s = Settings(_env_file=None, api_key="test-key")
    assert s.checkpointer == "sqlite"


def test_default_checkpointer_url():
    s = Settings(_env_file=None, api_key="test-key")
    assert s.checkpointer_url == "./harness.sqlite"


def test_default_agent_is_default():
    s = Settings(_env_file=None, api_key="test-key")
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
    monkeypatch.setenv("HARNESS_API_KEY", "test-key")
    monkeypatch.chdir(tmp_path)  # no .env here — isolates from the real one

    settings = get_settings()

    assert settings.llm_backend == "fake"


def test_target_file_overrides_a_field_left_at_default(monkeypatch, tmp_path):
    monkeypatch.delenv("HARNESS_RAG_COLLECTION", raising=False)
    monkeypatch.setenv("HARNESS_API_KEY", "test-key")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config" / "targets").mkdir(parents=True)
    (tmp_path / "config" / "targets" / "acme.yaml").write_text("rag_collection: acme-docs\n")
    monkeypatch.setenv("HARNESS_TARGET", "acme")

    settings = get_settings()

    assert settings.rag_collection == "acme-docs"


def test_real_env_var_still_beats_the_target_file(monkeypatch, tmp_path):
    monkeypatch.setenv("HARNESS_API_KEY", "test-key")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config" / "targets").mkdir(parents=True)
    (tmp_path / "config" / "targets" / "acme.yaml").write_text("rag_collection: acme-docs\n")
    monkeypatch.setenv("HARNESS_TARGET", "acme")
    monkeypatch.setenv("HARNESS_RAG_COLLECTION", "explicit-collection")

    settings = get_settings()

    assert settings.rag_collection == "explicit-collection"


def _personal_target(monkeypatch, tmp_path, body: str) -> None:
    monkeypatch.chdir(tmp_path)  # no .env here — isolates from the real one
    (tmp_path / "config" / "targets").mkdir(parents=True)
    (tmp_path / "config" / "targets" / "personal.yaml").write_text(body)
    monkeypatch.setenv("HARNESS_TARGET", "personal")


def test_a_target_file_can_turn_the_input_guard_off(monkeypatch, tmp_path):
    # Only the target file says false: the suite-wide env default is removed.
    monkeypatch.delenv("HARNESS_INPUT_GUARD_ENABLED", raising=False)
    monkeypatch.delenv("HARNESS_AZURE_ENDPOINT", raising=False)
    monkeypatch.setenv("HARNESS_API_KEY", "test-key")
    _personal_target(monkeypatch, tmp_path, "input_guard_enabled: false\n")

    assert get_settings().input_guard_enabled is False


def test_a_target_file_can_turn_auth_off(monkeypatch, tmp_path):
    monkeypatch.delenv("HARNESS_AUTH_ENABLED", raising=False)
    monkeypatch.delenv("HARNESS_API_KEY", raising=False)
    _personal_target(monkeypatch, tmp_path, "auth_enabled: false\n")

    assert get_settings().auth_enabled is False


def test_an_explicit_env_var_beats_the_target_file_even_at_its_default(monkeypatch, tmp_path):
    # true is also the field's default; an operator who sets it must still win.
    monkeypatch.setenv("HARNESS_INPUT_GUARD_ENABLED", "true")
    monkeypatch.setenv("HARNESS_AZURE_ENDPOINT", "https://foundry.example")
    monkeypatch.setenv("HARNESS_API_KEY", "test-key")
    _personal_target(monkeypatch, tmp_path, "input_guard_enabled: false\n")

    assert get_settings().input_guard_enabled is True


def test_unknown_field_in_target_file_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("HARNESS_API_KEY", "test-key")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config" / "targets").mkdir(parents=True)
    (tmp_path / "config" / "targets" / "acme.yaml").write_text("not_a_real_field: 1\n")
    monkeypatch.setenv("HARNESS_TARGET", "acme")

    with pytest.raises(ValueError, match="not_a_real_field"):
        get_settings()


def test_tool_details_stay_out_of_the_browser_by_default(monkeypatch):
    monkeypatch.delenv("HARNESS_SHOW_TOOL_DETAILS", raising=False)
    monkeypatch.setenv("HARNESS_API_KEY", "test-key")

    assert Settings(_env_file=None).show_tool_details is False


def test_tool_details_can_be_turned_on(monkeypatch):
    monkeypatch.setenv("HARNESS_SHOW_TOOL_DETAILS", "true")
    monkeypatch.setenv("HARNESS_API_KEY", "test-key")

    assert Settings(_env_file=None).show_tool_details is True


_POSTGRES_URL_ERROR = (
    "HARNESS_CHECKPOINTER=postgres needs HARNESS_CHECKPOINTER_URL set to "
    "a postgresql:// connection string"
)


@pytest.mark.parametrize("url", ["", "./harness.sqlite", "postgresql+psycopg://u:p@h/db"])
def test_postgres_needs_a_postgresql_url(url):
    # Without it, a missing secret leaves the SQLite default for psycopg to choke on.
    with pytest.raises(ValidationError, match=re.escape(_POSTGRES_URL_ERROR)):
        Settings(_env_file=None, api_key="test-key", checkpointer="postgres", checkpointer_url=url)


@pytest.mark.parametrize("url", ["postgresql://u:p@h/db", "postgres://u:p@h/db"])
def test_postgres_takes_either_url_scheme(url):
    settings = Settings(
        _env_file=None, api_key="test-key", checkpointer="postgres", checkpointer_url=url
    )
    assert settings.checkpointer_url == url


def test_a_validation_error_never_shows_the_settings_input(monkeypatch):
    # pydantic ends its error text with input_value=, the first and last characters of
    # the settings dict, which holds HARNESS_API_KEY, HARNESS_AZURE_API_KEY and the
    # connection string. An API log must not carry pieces of them.
    monkeypatch.delenv("HARNESS_API_KEY", raising=False)
    monkeypatch.setenv("HARNESS_INPUT_GUARD_ENABLED", "false")  # the guard's check passes
    with pytest.raises(ValidationError) as excinfo:
        Settings(
            _env_file=None, checkpointer="postgres", checkpointer_url="./harness.sqlite",
            api_key="SECRETKEY",
        )

    text = str(excinfo.value)
    assert _POSTGRES_URL_ERROR in text  # the error under test
    assert "SECRETKEY" not in text
    assert "input_value" not in text
