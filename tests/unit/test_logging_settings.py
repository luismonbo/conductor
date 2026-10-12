"""HARNESS_LOG_LEVEL: the level of the harness's own log lines."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from harness.config.settings import Settings


@pytest.fixture
def clean_env(monkeypatch):
    # main.py's load_dotenv() can copy the real .env into os.environ mid-session.
    monkeypatch.delenv("HARNESS_LOG_LEVEL", raising=False)


def _settings() -> Settings:
    return Settings(_env_file=None, api_key="test-key", input_guard_enabled=False)


def test_harness_info_lines_are_on_by_default(clean_env):
    assert _settings().log_level == "INFO"


def test_log_level_accepts_lowercase(clean_env, monkeypatch):
    monkeypatch.setenv("HARNESS_LOG_LEVEL", "debug")
    assert _settings().log_level == "DEBUG"


def test_an_unknown_log_level_fails_at_startup(clean_env, monkeypatch):
    monkeypatch.setenv("HARNESS_LOG_LEVEL", "verbose")
    with pytest.raises(ValidationError, match="log_level"):
        _settings()
