"""Pin the rate-limiting settings and their defaults."""
from __future__ import annotations

from harness.config.settings import Settings

_ENV_KEYS = [
    "HARNESS_RATE_LIMIT_ENABLED",
    "HARNESS_RATE_LIMIT_STRICT",
    "HARNESS_RATE_LIMIT_DEFAULT",
]


def test_rate_limit_defaults(monkeypatch):
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HARNESS_API_KEY", "test-key")

    settings = Settings(_env_file=None)

    assert settings.rate_limit_enabled is True
    assert settings.rate_limit_strict == "15/minute"
    assert settings.rate_limit_default == "60/minute"


def test_rate_limit_env_overrides(monkeypatch):
    monkeypatch.setenv("HARNESS_RATE_LIMIT_ENABLED", "false")
    monkeypatch.setenv("HARNESS_RATE_LIMIT_STRICT", "5/minute")
    monkeypatch.setenv("HARNESS_RATE_LIMIT_DEFAULT", "30/minute")
    monkeypatch.setenv("HARNESS_API_KEY", "test-key")

    settings = Settings(_env_file=None)

    assert settings.rate_limit_enabled is False
    assert settings.rate_limit_strict == "5/minute"
    assert settings.rate_limit_default == "30/minute"
