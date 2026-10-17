"""open_storage: each backend's checkpointer and run store, opened and closed as
one unit. The Postgres cases are in tests/integration/test_storage_postgres.py."""
from __future__ import annotations

import logging

import aiosqlite
import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

import harness.orchestration.storage as storage_module
from harness.config.settings import Settings
from harness.observability.run_store.sqlite import SqliteRunStore
from harness.orchestration.storage import open_storage


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, api_key="test-key", input_guard_enabled=False, **overrides)


async def test_memory_keeps_owners_in_an_in_process_database():
    storage = await open_storage(_settings(checkpointer="memory"))
    try:
        assert isinstance(storage.checkpointer, MemorySaver)
        assert isinstance(storage.run_store, SqliteRunStore)
        await storage.run_store.start_run("r1", "t1", "default", "fake", "sid")
        assert await storage.run_store.owns_thread("sid", "t1") is True
    finally:
        await storage.aclose()
    with pytest.raises(ValueError, match="no active connection"):
        await storage.run_store._conn.execute("SELECT 1")


async def test_sqlite_creates_every_table_before_serving(tmp_path):
    path = tmp_path / "h.sqlite"
    storage = await open_storage(_settings(checkpointer="sqlite", checkpointer_url=str(path)))
    await storage.aclose()
    async with aiosqlite.connect(path) as conn:
        cur = await conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        tables = {row[0] for row in await cur.fetchall()}
    assert {"checkpoints", "writes", "runs", "guard_decisions"} <= tables


async def test_aclose_closes_both_sqlite_connections_and_can_run_twice(tmp_path):
    storage = await open_storage(
        _settings(checkpointer="sqlite", checkpointer_url=str(tmp_path / "h.sqlite"))
    )
    assert isinstance(storage.checkpointer, AsyncSqliteSaver)
    await storage.aclose()
    await storage.aclose()
    for conn in (storage.checkpointer.conn, storage.run_store._conn):
        with pytest.raises(ValueError, match="no active connection"):
            await conn.execute("SELECT 1")


async def test_a_failure_while_opening_closes_what_was_already_open(monkeypatch, tmp_path):
    opened: list[aiosqlite.Connection] = []
    real_connect = aiosqlite.connect

    def recording_connect(*args, **kwargs):
        opened.append(real_connect(*args, **kwargs))
        return opened[-1]

    async def failing_create_table(self):
        raise aiosqlite.OperationalError("database is locked")

    monkeypatch.setattr(aiosqlite, "connect", recording_connect)
    monkeypatch.setattr(SqliteRunStore, "create_table", failing_create_table)

    with pytest.raises(aiosqlite.OperationalError, match="database is locked"):
        await open_storage(
            _settings(checkpointer="sqlite", checkpointer_url=str(tmp_path / "h.sqlite"))
        )

    assert len(opened) == 2  # the checkpointer's connection and the run store's
    for conn in opened:
        with pytest.raises(ValueError, match="no active connection"):
            await conn.execute("SELECT 1")


async def test_whatever_fails_after_both_are_open_still_closes_them(monkeypatch, tmp_path):
    """The log line is the last thing open_storage does: it must run while the exit
    stack still owns every connection, or an error from it leaks them."""
    opened: list[aiosqlite.Connection] = []
    real_connect = aiosqlite.connect

    def recording_connect(*args, **kwargs):
        opened.append(real_connect(*args, **kwargs))
        return opened[-1]

    def failing_where(settings):
        raise RuntimeError("cannot describe the backend")

    monkeypatch.setattr(aiosqlite, "connect", recording_connect)
    monkeypatch.setattr(storage_module, "_where", failing_where)

    try:
        with pytest.raises(RuntimeError, match="cannot describe the backend"):
            await open_storage(
                _settings(checkpointer="sqlite", checkpointer_url=str(tmp_path / "h.sqlite"))
            )
        assert len(opened) == 2  # the checkpointer's connection and the run store's
        for conn in opened:
            with pytest.raises(ValueError, match="no active connection"):
                await conn.execute("SELECT 1")
    finally:
        for conn in opened:  # aiosqlite's worker thread would keep pytest from exiting
            await conn.close()


async def test_an_unknown_backend_opens_nothing():
    with pytest.raises(ValueError, match="Unknown checkpointer backend: 'redis'"):
        await open_storage(_settings(checkpointer="redis"))


async def test_it_logs_the_backend(tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger="harness.orchestration.storage"):
        storage = await open_storage(
            _settings(checkpointer="sqlite", checkpointer_url=str(tmp_path / "h.sqlite"))
        )
    await storage.aclose()
    assert "storage: sqlite" in caplog.text
