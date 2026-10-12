"""Startup sets the harness's log level before it builds the entry agent."""
from __future__ import annotations

import logging

import pytest

import harness.api.main as _main
from harness.orchestration.agents import AgentLoadError


async def test_startup_sets_the_log_level_before_building_the_agent(monkeypatch):
    monkeypatch.setenv("HARNESS_LOG_LEVEL", "WARNING")
    monkeypatch.setenv("HARNESS_AGENT", "no_such_agent")  # building the agent fails

    with pytest.raises(AgentLoadError):
        async with _main.lifespan(_main.app):
            pass

    assert logging.getLogger("harness").level == logging.WARNING
