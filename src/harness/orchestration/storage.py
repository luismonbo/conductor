"""Storage: the checkpointer and the run store, opened and closed as one unit.

open_storage opens both for HARNESS_CHECKPOINTER and creates every table they
need, so the API never serves before its tables exist, and Storage.aclose()
closes every connection it opened. See
docs/superpowers/specs/2026-10-15-production-storage-neon-design.md, §4.1.

backend    checkpointer         run store
memory     MemorySaver          SqliteRunStore on an in-process :memory: database
sqlite     AsyncSqliteSaver     SqliteRunStore on the same file
postgres   AsyncPostgresSaver   PostgresRunStore, on the same pool
"""
from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from dataclasses import dataclass

from langgraph.checkpoint.base import BaseCheckpointSaver

from harness.config.settings import Settings
from harness.observability.run_store import RunStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Storage:
    """The checkpointer and run store one API process uses, opened together."""

    checkpointer: BaseCheckpointSaver
    run_store: RunStore
    _close: Callable[[], Awaitable[None]]

    async def aclose(self) -> None:
        """Close every connection open_storage opened. A second call does nothing."""
        await self._close()


_Opened = tuple[BaseCheckpointSaver, RunStore]


async def _sqlite_run_store(path: str, stack: AsyncExitStack) -> RunStore:
    import aiosqlite

    from harness.observability.run_store.sqlite import SqliteRunStore

    conn = await aiosqlite.connect(path)
    stack.push_async_callback(conn.close)
    store = SqliteRunStore(conn)
    await store.create_table()
    return store


async def _open_memory(settings: Settings, stack: AsyncExitStack) -> _Opened:
    from langgraph.checkpoint.memory import MemorySaver

    # Thread ownership is checked against the run store in every mode, so memory
    # mode keeps one too: a database that lives exactly as long as its connection.
    return MemorySaver(), await _sqlite_run_store(":memory:", stack)


async def _open_sqlite(settings: Settings, stack: AsyncExitStack) -> _Opened:
    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    conn = await aiosqlite.connect(settings.checkpointer_url)
    stack.push_async_callback(conn.close)
    checkpointer = AsyncSqliteSaver(conn)
    await checkpointer.setup()
    return checkpointer, await _sqlite_run_store(settings.checkpointer_url, stack)


async def _open_postgres(settings: Settings, stack: AsyncExitStack) -> _Opened:
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    from harness.adapters.postgres.pool import open_pool
    from harness.observability.run_store.postgres import PostgresRunStore

    pool = await open_pool(settings.checkpointer_url)
    stack.push_async_callback(pool.close)
    checkpointer = AsyncPostgresSaver(pool)
    # LangGraph's own migrations, recorded in checkpoint_migrations, so a later
    # version upgrades its tables on the next start.
    await checkpointer.setup()
    run_store = PostgresRunStore(pool)
    await run_store.create_table()
    return checkpointer, run_store


_OPENERS: dict[str, Callable[[Settings, AsyncExitStack], Awaitable[_Opened]]] = {
    "memory": _open_memory,
    "sqlite": _open_sqlite,
    "postgres": _open_postgres,
}


def _where(settings: Settings) -> str:
    """The backend, and for Postgres its host: never the connection string."""
    if settings.checkpointer != "postgres":
        return settings.checkpointer
    from harness.adapters.postgres.pool import host_of

    return f"postgres on {host_of(settings.checkpointer_url)}"


async def open_storage(settings: Settings) -> Storage:
    """Open the checkpointer and run store HARNESS_CHECKPOINTER selects, tables included.

    Fails fast: whatever opened before a failure is closed before the error propagates.
    """
    opener = _OPENERS.get(settings.checkpointer)
    if opener is None:
        raise ValueError(
            f"Unknown checkpointer backend: {settings.checkpointer!r}. "
            "Valid values: memory | sqlite | postgres."
        )
    async with AsyncExitStack() as stack:
        checkpointer, run_store = await opener(settings, stack)
        close = stack.pop_all().aclose
    logger.info("storage: %s", _where(settings))
    return Storage(checkpointer=checkpointer, run_store=run_store, _close=close)
