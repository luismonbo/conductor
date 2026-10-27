"""Pin the input-guard settings, their defaults, and the fail-closed endpoint check."""
from __future__ import annotations

import pytest

from harness.config.settings import Settings

# main.py's load_dotenv() and pymilvus can copy the real .env into os.environ
# during a test session, so every test of a default deletes the keys it reads.
_GUARD_ENV = [
    "HARNESS_INPUT_GUARD_ENABLED",
    "HARNESS_AZURE_ENDPOINT",
    "HARNESS_GUARD_DEPLOYMENT",
    "HARNESS_GUARD_TIMEOUT_SECONDS",
    "HARNESS_MAX_MESSAGE_CHARS",
    "HARNESS_MAX_OUTPUT_TOKENS",
    "HARNESS_MAX_THREAD_TURNS",
    "HARNESS_DAILY_TOKEN_BUDGET",
]


@pytest.fixture
def clean_env(monkeypatch):
    for key in _GUARD_ENV:
        monkeypatch.delenv(key, raising=False)


def test_guard_is_on_by_default_and_refuses_to_start_without_an_azure_endpoint(clean_env):
    with pytest.raises(ValueError, match="HARNESS_AZURE_ENDPOINT"):
        Settings(_env_file=None, api_key="test-key")


def test_guard_on_with_an_azure_endpoint_constructs(clean_env):
    s = Settings(_env_file=None, api_key="test-key", azure_endpoint="https://foundry.example")
    assert s.input_guard_enabled is True


def test_guard_off_needs_no_endpoint(clean_env):
    s = Settings(_env_file=None, api_key="test-key", input_guard_enabled=False)
    assert s.input_guard_enabled is False


def test_guard_and_limit_defaults(clean_env):
    s = Settings(_env_file=None, api_key="test-key", input_guard_enabled=False)
    assert s.guard_deployment == "Decision-1"
    assert s.guard_timeout_seconds == 3.0
    assert s.max_message_chars == 2000
    assert s.max_output_tokens == 1500
    assert s.max_thread_turns == 20
    assert s.daily_token_budget == 33000


@pytest.mark.parametrize(
    "field", ["max_message_chars", "max_output_tokens", "max_thread_turns", "daily_token_budget"]
)
def test_a_limit_accepts_zero_to_switch_off_but_rejects_negatives(clean_env, field):
    base = dict(_env_file=None, api_key="test-key", input_guard_enabled=False)
    assert getattr(Settings(**base, **{field: 0}), field) == 0
    with pytest.raises(ValueError, match=field):
        Settings(**base, **{field: -1})


def test_the_guard_timeout_must_be_positive(clean_env):
    # At 0 every classifier call would time out at once: the guard silently off.
    with pytest.raises(ValueError, match="guard_timeout_seconds"):
        Settings(_env_file=None, api_key="test-key", input_guard_enabled=False,
                 guard_timeout_seconds=0)
