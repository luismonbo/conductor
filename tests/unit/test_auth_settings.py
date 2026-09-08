"""Pin the authentication settings, their defaults, and the fail-closed guard."""
from __future__ import annotations

import pytest

from harness.config.settings import Settings

_ENV_KEYS = [
    "HARNESS_AUTH_ENABLED",
    "HARNESS_API_KEY",
]


def test_auth_enabled_by_default(monkeypatch):
    monkeypatch.delenv("HARNESS_AUTH_ENABLED", raising=False)
    monkeypatch.setenv("HARNESS_API_KEY", "test-key")

    settings = Settings(_env_file=None)

    assert settings.auth_enabled is True
    assert settings.api_key == "test-key"


def test_bare_defaults_fail_closed(monkeypatch):
    """The true out-of-the-box defaults (nothing set) must refuse to
    construct — no route is ever unknowingly unguarded."""
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)

    with pytest.raises(ValueError, match="HARNESS_API_KEY"):
        Settings(_env_file=None)


def test_auth_disabled_with_no_key_is_fine(monkeypatch):
    monkeypatch.setenv("HARNESS_AUTH_ENABLED", "false")
    monkeypatch.delenv("HARNESS_API_KEY", raising=False)

    settings = Settings(_env_file=None)

    assert settings.auth_enabled is False
    assert settings.api_key == ""


def test_auth_enabled_with_key_succeeds(monkeypatch):
    monkeypatch.setenv("HARNESS_AUTH_ENABLED", "true")
    monkeypatch.setenv("HARNESS_API_KEY", "test-key")

    settings = Settings(_env_file=None)

    assert settings.auth_enabled is True
    assert settings.api_key == "test-key"
