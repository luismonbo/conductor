"""Checkpointer factory unit tests."""
from __future__ import annotations

import pytest

from harness.config.settings import Settings
from harness.orchestration.checkpointer import build_checkpointer


@pytest.mark.asyncio
async def test_memory_checkpointer():
    from langgraph.checkpoint.memory import MemorySaver
    s = Settings(checkpointer="memory", api_key="test-key")
    cp = await build_checkpointer(s)
    assert isinstance(cp, MemorySaver)


@pytest.mark.asyncio
async def test_sqlite_checkpointer():
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    s = Settings(checkpointer="sqlite", checkpointer_url=":memory:", api_key="test-key")
    cp = await build_checkpointer(s)
    assert isinstance(cp, AsyncSqliteSaver)


@pytest.mark.asyncio
async def test_postgres_raises_not_implemented():
    s = Settings(checkpointer="postgres", api_key="test-key")
    with pytest.raises(NotImplementedError):
        await build_checkpointer(s)


@pytest.mark.asyncio
async def test_unknown_raises_value_error():
    s = Settings(checkpointer="redis", api_key="test-key")
    with pytest.raises(ValueError):
        await build_checkpointer(s)
