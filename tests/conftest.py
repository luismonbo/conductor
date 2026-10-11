"""Suite-wide test defaults."""
from __future__ import annotations

import pytest


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
