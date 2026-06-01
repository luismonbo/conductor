"""Checkpointer factory unit tests."""
from __future__ import annotations

import pytest

from harness.config.settings import Settings
from harness.orchestration.checkpointer import build_checkpointer


@pytest.mark.unit
def test_memory_checkpointer():
    from langgraph.checkpoint.memory import MemorySaver
    s = Settings(checkpointer="memory")
    cp = build_checkpointer(s)
    assert isinstance(cp, MemorySaver)


@pytest.mark.unit
def test_sqlite_checkpointer():
    from langgraph.checkpoint.sqlite import SqliteSaver
    s = Settings(checkpointer="sqlite", checkpointer_url=":memory:")
    cp = build_checkpointer(s)
    assert isinstance(cp, SqliteSaver)


@pytest.mark.unit
def test_postgres_raises_not_implemented():
    s = Settings(checkpointer="postgres")
    with pytest.raises(NotImplementedError):
        build_checkpointer(s)


@pytest.mark.unit
def test_unknown_raises_value_error():
    s = Settings(checkpointer="redis")
    with pytest.raises(ValueError):
        build_checkpointer(s)
