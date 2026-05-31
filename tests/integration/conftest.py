"""Integration test fixtures — clear module-level app state between tests."""
from __future__ import annotations

import pytest

import harness.api.main as _main


@pytest.fixture(autouse=True)
async def clean_app_state():
    """Reset _running and _short_term before and after every integration test."""
    _main._running.clear()
    _main._short_term._store.clear()
    yield
    for task in list(_main._running.values()):
        if not task.done():
            task.cancel()
    _main._running.clear()
    _main._short_term._store.clear()
