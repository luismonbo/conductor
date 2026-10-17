"""The Postgres connection pool the checkpointer and the run store share.

Every setting suits a remote Postgres reached through Azure's outbound NAT: Neon
in production, and any stock Postgres the same way. See
docs/superpowers/specs/2026-10-15-production-storage-neon-design.md, §4.2.
"""
from __future__ import annotations

import re
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
# A checkout waits this long for a connection: one full connect timeout plus margin,
# so a checkout that must replace a dead connection still succeeds. psycopg_pool's
# 30 s default would keep a request waiting 30 s per store call during an outage.
CHECKOUT_TIMEOUT_SECONDS = 15
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


# Where a URL connection string keeps its password: the userinfo, which runs to the
# last "@" before any "?" or "#", so an unencoded "@" or "/" stays inside it.
URL_USERINFO = re.compile(r"^(?P<scheme>[A-Za-z][A-Za-z0-9+.-]*://)(?P<userinfo>[^?#]*)@")
# What libpq reads as something other than the password.
_MISPLACED_IN_PASSWORD = re.compile(r"[@/?#]")


def url_password(dsn: str) -> str | None:
    """The password of a URL connection string, as written: None when it has none."""
    match = URL_USERINFO.match(dsn)
    if match is None or ":" not in match.group("userinfo"):
        return None
    return match.group("userinfo").split(":", 1)[1]


def _parses(dsn: str) -> bool:
    try:
        conninfo_to_dict(dsn)
    except psycopg.Error:
        return False
    return True


def check_dsn(dsn: str) -> None:
    """Raise ValueError unless libpq will read `dsn` the way it was written.

    An unencoded "%" in the password makes psycopg's parser quote the whole password
    in its error. An unencoded "@" or "/" is read as part of the host, which libpq then
    looks up in DNS and names in its error. Either way the pool would log a piece of
    the password, so a malformed URL is refused before anything connects or logs.

    The message names nothing taken from `dsn`, and the error carries no chained
    exception: the parser's error is gone before this one is raised, because uvicorn
    prints a startup error's whole chain.
    """
    if _parses(dsn) and not _MISPLACED_IN_PASSWORD.search(url_password(dsn) or ""):
        return
    raise ValueError(
        "the Postgres connection string is malformed: URL-encode special characters "
        "in the password (@ is %40, / is %2F, % is %25)"
    ) from None


def host_of(dsn: str) -> str:
    """The server's host, for logs and errors: never the rest of the connection string."""
    try:
        return str(conninfo_to_dict(dsn).get("host") or "localhost")
    except psycopg.Error:
        return "an unreadable connection string"


async def check_connection(conn: AsyncConnection[DictRow]) -> None:
    """The pool's check of a connection before it is handed out: one round trip.

    psycopg_pool's own check runs `execute("")`. Under `prepare_threshold: 0` psycopg
    prepares that empty query on every call and never reuses it: two round trips, and
    one more statement left on the connection each time. Pool connections are always
    autocommit (CONNECTION_KWARGS), so the non-autocommit branch of psycopg_pool's check
    isn't needed.
    """
    await conn.execute("", prepare=False)


async def open_pool(dsn: str) -> PostgresPool:
    """An open pool with its first connection made, or ConnectionError naming the host.

    A malformed connection string is a ValueError, raised before the pool exists
    (see check_dsn). Neon closes connections when it suspends, so each checkout
    checks its connection first (one round trip) and replaces a dead one.
    """
    check_dsn(dsn)
    pool: PostgresPool = AsyncConnectionPool(
        dsn,
        connection_class=AsyncConnection[DictRow],
        kwargs=CONNECTION_KWARGS,
        min_size=MIN_SIZE,
        max_size=MAX_SIZE,
        timeout=CHECKOUT_TIMEOUT_SECONDS,
        check=check_connection,
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
