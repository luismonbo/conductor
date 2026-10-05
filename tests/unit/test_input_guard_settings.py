"""Pin the input-guard settings, their defaults, and the fail-closed key check."""
from __future__ import annotations

import pytest

from harness.config.settings import Settings
from harness.config.targets import resolve_target_overrides

# main.py's load_dotenv() and pymilvus can copy the real .env into os.environ
# during a test session, so every test of a default deletes the keys it reads.
_GUARD_ENV = [
    "HARNESS_INPUT_GUARD_ENABLED",
    "HARNESS_TYPESAFE_API_KEY",
    "HARNESS_GUARD_MODEL",
    "HARNESS_GUARD_TIMEOUT_SECONDS",
    "HARNESS_MAX_MESSAGE_CHARS",
    "HARNESS_MAX_OUTPUT_TOKENS",
    "HARNESS_MAX_THREAD_TURNS",
    "HARNESS_DAILY_TOKEN_BUDGET",
    "HARNESS_SYSTEM_PROMPT",
]


@pytest.fixture
def clean_env(monkeypatch):
    for key in _GUARD_ENV:
        monkeypatch.delenv(key, raising=False)


def test_guard_is_on_by_default_and_refuses_to_start_without_a_key(clean_env):
    with pytest.raises(ValueError, match="HARNESS_TYPESAFE_API_KEY"):
        Settings(_env_file=None, api_key="test-key")


def test_guard_on_with_a_key_constructs(clean_env):
    s = Settings(_env_file=None, api_key="test-key", typesafe_api_key="ts-key")
    assert s.input_guard_enabled is True
    assert s.typesafe_api_key == "ts-key"


def test_guard_off_needs_no_key(clean_env):
    s = Settings(_env_file=None, api_key="test-key", input_guard_enabled=False)
    assert s.input_guard_enabled is False


def test_guard_and_limit_defaults(clean_env):
    s = Settings(_env_file=None, api_key="test-key", input_guard_enabled=False)
    assert s.guard_model == "jev-1.13.0"
    assert s.guard_timeout_seconds == 3.0
    assert s.max_message_chars == 2000
    assert s.max_output_tokens == 1500
    assert s.max_thread_turns == 20
    assert s.daily_token_budget == 33000
    assert s.system_prompt == ""


def test_typesafe_key_is_rejected_in_a_target_file(tmp_path):
    (tmp_path / "site.yaml").write_text("typesafe_api_key: leaked\n")
    with pytest.raises(ValueError, match="typesafe_api_key"):
        resolve_target_overrides("site", base_dir=tmp_path)
