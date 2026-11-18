"""The retention_days setting: off (0) by default, 30 days on the public site."""
from pathlib import Path

import pytest
from pydantic import ValidationError

from harness.config.settings import Settings
from harness.config.targets import resolve_target_overrides

_BASE = dict(_env_file=None, api_key="test-key", input_guard_enabled=False)
_TARGETS = Path(__file__).resolve().parents[2] / "config" / "targets"


def test_retention_is_off_unless_something_sets_it(monkeypatch):
    monkeypatch.delenv("HARNESS_RETENTION_DAYS", raising=False)

    assert Settings(**_BASE).retention_days == 0


def test_the_environment_sets_retention(monkeypatch):
    monkeypatch.setenv("HARNESS_RETENTION_DAYS", "30")

    assert Settings(**_BASE).retention_days == 30


def test_negative_retention_is_refused():
    with pytest.raises(ValidationError):
        Settings(**_BASE, retention_days=-1)


def test_the_public_site_target_keeps_conversations_30_days():
    overrides = resolve_target_overrides("luismonbo-com", base_dir=_TARGETS)

    assert overrides["retention_days"] == 30
