"""Unit tests for the dynamic rate-limit value callables.

Settings are read fresh on every call (not frozen at import) so that
HARNESS_RATE_LIMIT_* env var changes -- including the ones test fixtures set
via monkeypatch after this module has already been imported once -- take
effect immediately. See docs/superpowers/specs/2026-08-27-rate-limiting-design.md.
"""
from __future__ import annotations

from harness.api.rate_limit import default_limit, limiter, strict_limit


def test_strict_limit_reflects_current_settings(monkeypatch):
    monkeypatch.setenv("HARNESS_API_KEY", "test-key")
    monkeypatch.setenv("HARNESS_RATE_LIMIT_STRICT", "7/minute")
    assert strict_limit() == "7/minute"

    monkeypatch.setenv("HARNESS_RATE_LIMIT_STRICT", "9/minute")
    assert strict_limit() == "9/minute"


def test_default_limit_reflects_current_settings(monkeypatch):
    monkeypatch.setenv("HARNESS_API_KEY", "test-key")
    monkeypatch.setenv("HARNESS_RATE_LIMIT_DEFAULT", "20/minute")
    assert default_limit() == "20/minute"

    monkeypatch.setenv("HARNESS_RATE_LIMIT_DEFAULT", "40/minute")
    assert default_limit() == "40/minute"


def test_disabled_flag_overrides_strict_limit(monkeypatch):
    monkeypatch.setenv("HARNESS_API_KEY", "test-key")
    monkeypatch.setenv("HARNESS_RATE_LIMIT_STRICT", "7/minute")
    monkeypatch.setenv("HARNESS_RATE_LIMIT_ENABLED", "false")

    assert strict_limit() != "7/minute"


def test_disabled_flag_overrides_default_limit(monkeypatch):
    monkeypatch.setenv("HARNESS_API_KEY", "test-key")
    monkeypatch.setenv("HARNESS_RATE_LIMIT_DEFAULT", "20/minute")
    monkeypatch.setenv("HARNESS_RATE_LIMIT_ENABLED", "false")

    assert default_limit() != "20/minute"


def test_limiter_is_constructed():
    assert limiter is not None
