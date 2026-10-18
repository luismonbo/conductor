"""The session_cookie_persistent setting: on by default, off for the public site."""
from pathlib import Path

from harness.config.settings import Settings
from harness.config.targets import resolve_target_overrides

_BASE = dict(_env_file=None, api_key="test-key", input_guard_enabled=False)
_TARGETS = Path(__file__).resolve().parents[2] / "config" / "targets"


def test_the_cookie_is_persistent_unless_something_says_otherwise(monkeypatch):
    monkeypatch.delenv("HARNESS_SESSION_COOKIE_PERSISTENT", raising=False)

    assert Settings(**_BASE).session_cookie_persistent is True


def test_the_cookie_can_be_made_a_session_cookie():
    assert Settings(**_BASE, session_cookie_persistent=False).session_cookie_persistent is False


def test_the_environment_can_turn_persistence_off(monkeypatch):
    monkeypatch.setenv("HARNESS_SESSION_COOKIE_PERSISTENT", "false")

    assert Settings(**_BASE).session_cookie_persistent is False


def test_the_public_site_target_asks_for_a_session_cookie():
    overrides = resolve_target_overrides("luismonbo-com", base_dir=_TARGETS)

    assert overrides["session_cookie_persistent"] is False
