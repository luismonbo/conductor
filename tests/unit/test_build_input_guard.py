"""build_input_guard: limits always; the TypeSafe classifier only when enabled, configured tight."""
from __future__ import annotations

import typesafe_sdk

from harness.config.settings import Settings
from harness.orchestration.build import build_input_guard
from harness.security.input_guard import GuardLimits


def _settings(**overrides) -> Settings:
    return Settings(**{"_env_file": None, "api_key": "test-key", **overrides})


def test_guard_off_builds_limits_only():
    guard = build_input_guard(_settings(
        input_guard_enabled=False, max_message_chars=10, max_thread_turns=3,
        daily_token_budget=500,
    ))
    assert guard.classifier_enabled is False
    assert guard.limits == GuardLimits(
        max_message_chars=10, max_thread_turns=3, daily_token_budget=500
    )


async def test_guard_on_configures_the_sdk_client(monkeypatch):
    captured: dict = {}

    class _RecordingClient:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

        async def aclose(self) -> None:
            captured["closed"] = True

    monkeypatch.setattr(typesafe_sdk, "AsyncTypeSafeClient", _RecordingClient)
    guard = build_input_guard(_settings(
        input_guard_enabled=True, typesafe_api_key="ts-key", guard_timeout_seconds=2.5,
    ))
    assert guard.classifier_enabled is True
    assert captured["api_key"] == "ts-key"
    assert captured["model"] == "jev-1.13.0"
    assert captured["timeout"] == 2.5
    assert captured["retry"].max_retries == 1
    assert captured["retry"].timeout == 2.5
    await guard.aclose()
    assert captured["closed"] is True


async def test_guard_on_builds_with_the_real_sdk_client():
    """The recording stub above accepts any kwargs; this pins them against the
    real constructor, so an SDK upgrade that renames one fails here, not at boot."""
    guard = build_input_guard(_settings(input_guard_enabled=True, typesafe_api_key="ts-key"))
    assert guard.classifier_enabled is True
    await guard.aclose()
