"""The Postgres connection pool the checkpointer and the run store share.

Every setting suits a remote Postgres reached through Azure's outbound NAT: Neon
in production, and any stock Postgres the same way. See
docs/superpowers/specs/2026-10-15-production-storage-neon-design.md, §4.2.
"""
from __future__ import annotations

from typing import Any

import psycopg
from psycopg import AsyncConnection
from psycopg.conninfo import conninfo_to_dict
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool, PoolTimeout

PostgresPool = AsyncConnectionPool[AsyncConnection[DictRow]]

# One replica and a few concurrent chats; Neon's 0.25 CU allows 104 connections.
MIN_SIZE = 1
MAX_SIZE = 5
# Covers a cold start (Neon wakes "within a few hundred milliseconds") with margin.
CONNECT_TIMEOUT_SECONDS = 10
# Startup waits this long for its first connection: an unreachable database stops
# the boot, not the first request.
OPEN_TIMEOUT_SECONDS = 30

CONNECTION_KWARGS: dict[str, Any] = {
    # What LangGraph's AsyncPostgresSaver.from_conn_string uses: setup() runs
    # CREATE INDEX CONCURRENTLY, which can't run inside a transaction.
    "autocommit": True,
    "row_factory": dict_row,
    "prepare_threshold": 0,
    "connect_timeout": CONNECT_TIMEOUT_SECONDS,
    # Azure drops outbound flows idle for 4 minutes. TCP keepalives keep a live
    # connection open; they end at Neon's proxy, so they don't keep Neon awake.
    "keepalives": 1,
    "keepalives_idle": 60,
    "keepalives_interval": 10,
    "keepalives_count": 3,
}


def host_of(dsn: str) -> str:
    """The server's host, for logs and errors: never the rest of the connection string."""
    try:
        return str(conninfo_to_dict(dsn).get("host") or "localhost")
    except psycopg.Error:
        return "an unreadable connection string"


async def open_pool(dsn: str) -> PostgresPool:
    """An open pool with its first connection made, or ConnectionError naming the host.

    Neon closes connections when it suspends, so each checkout checks its
    connection first (one round trip) and replaces a dead one.
    """
    pool: PostgresPool = AsyncConnectionPool(
        dsn,
        connection_class=AsyncConnection[DictRow],
        kwargs=CONNECTION_KWARGS,
        min_size=MIN_SIZE,
        max_size=MAX_SIZE,
        check=AsyncConnectionPool.check_connection,
        name="harness",
        open=False,
    )
    try:
        await pool.open(wait=True, timeout=OPEN_TIMEOUT_SECONDS)
    except PoolTimeout as exc:
        await pool.close()
        raise ConnectionError(
            f"no Postgres connection to {host_of(dsn)} within {OPEN_TIMEOUT_SECONDS} s "
            "(the warnings above say why)"
        ) from exc
    except BaseException:
        await pool.close()
        raise
    return pool
