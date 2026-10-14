"""Suite-wide test defaults."""
from __future__ import annotations

import logging

import pytest

from harness.observability.logging_setup import _PRIVATE_AT_DEBUG

# The loggers whose levels configure_logging (run by the app's startup) sets.
_CONFIGURED_LOGGERS = ("harness", *_PRIVATE_AT_DEBUG)


@pytest.fixture(autouse=True)
def _restore_logger_levels():
    """Logger levels are process-wide: put back the ones the app's startup sets,
    so no test inherits another test's logging setup."""
    saved = {name: logging.getLogger(name).level for name in _CONFIGURED_LOGGERS}
    yield
    for name, level in saved.items():
        logging.getLogger(name).setLevel(level)


@pytest.fixture(autouse=True)
def _input_guard_off_by_default(monkeypatch):
    """The input guard is default-on and refuses to start without a TypeSafe
    key, so every test runs with it off. Guard tests opt back in explicitly
    (delenv this, or pass input_guard_enabled=True with a key)."""
    monkeypatch.setenv("HARNESS_INPUT_GUARD_ENABLED", "false")


@pytest.fixture
def fixture_agents(monkeypatch):
    """Resolve agent names under tests/fixtures/agents instead of src/harness/agents."""
    import harness.orchestration.build as build_module
    from harness.orchestration.agents import load_agent

    monkeypatch.setattr(
        build_module,
        "load_agent",
        lambda name: load_agent(name, package="tests.fixtures.agents"),
    )


_WIKI_ENV = (
    "HARNESS_WIKI_DIR",
    "HARNESS_WIKI_BLOB_ACCOUNT_URL",
    "HARNESS_WIKI_BLOB_CONTAINER",
    "HARNESS_WIKI_BLOB_PREFIX",
)


@pytest.fixture(autouse=True)
def _no_wiki_by_default(monkeypatch):
    """Every test starts with no wiki source, whatever the developer's .env says.
    Tests that need a wiki set HARNESS_WIKI_DIR (or pass wiki_dir) themselves."""
    for key in _WIKI_ENV:
        monkeypatch.setenv(key, "")
