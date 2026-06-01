"""Integration test fixtures — clear module-level app state between tests."""
from __future__ import annotations

import pytest

import harness.api.main as _main


@pytest.fixture(autouse=True)
async def clean_app_state(monkeypatch):
    """Reset all module-level state and force memory checkpointer before each test."""
    monkeypatch.setenv("HARNESS_CHECKPOINTER", "memory")
    _main._running.clear()
    _main._short_term._store.clear()
    _main._registry = None  # force rebuild with test env vars
    yield
    for task in list(_main._running.values()):
        if not task.done():
            task.cancel()
    _main._running.clear()
    _main._short_term._store.clear()
    _main._registry = None
