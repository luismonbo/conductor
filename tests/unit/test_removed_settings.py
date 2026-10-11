"""Prompts left config: a leftover HARNESS_SYSTEM_PROMPT or system_prompt: fails loudly."""
from __future__ import annotations

from pathlib import Path

import pytest

from harness.config.settings import Settings, get_settings
from harness.config.targets import resolve_target_overrides

_REPO_TARGETS = Path(__file__).resolve().parents[2] / "config" / "targets"


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    """A working directory with no .env and no target, and the shared key set."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HARNESS_TARGET", raising=False)
    monkeypatch.delenv("HARNESS_AGENT", raising=False)
    monkeypatch.delenv("HARNESS_SYSTEM_PROMPT", raising=False)
    monkeypatch.setenv("HARNESS_API_KEY", "test-key")
    return tmp_path


def test_harness_system_prompt_in_the_environment_stops_startup(isolated, monkeypatch):
    monkeypatch.setenv("HARNESS_SYSTEM_PROMPT", "You answer questions about X.")
    with pytest.raises(ValueError, match="HARNESS_SYSTEM_PROMPT was removed"):
        get_settings()


def test_harness_system_prompt_in_dotenv_stops_startup(isolated):
    (isolated / ".env").write_text("HARNESS_SYSTEM_PROMPT=You answer questions about X.\n")
    with pytest.raises(ValueError, match="HARNESS_SYSTEM_PROMPT was removed"):
        get_settings()


def test_a_target_file_cannot_set_a_system_prompt(isolated, monkeypatch):
    (isolated / "config" / "targets").mkdir(parents=True)
    (isolated / "config" / "targets" / "site.yaml").write_text("system_prompt: Answer X.\n")
    monkeypatch.setenv("HARNESS_TARGET", "site")
    with pytest.raises(ValueError, match="system_prompt"):
        get_settings()


def test_settings_works_without_the_removed_variables(isolated):
    assert get_settings().agent == "default"


def test_the_example_target_only_sets_known_fields():
    overrides = resolve_target_overrides("example", base_dir=_REPO_TARGETS)
    assert overrides.keys() <= Settings.model_fields.keys()
