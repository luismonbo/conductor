"""The pool under what Neon and Azure do to connections: close them, and cancel queries."""
from __future__ import annotations

import asyncio

import psycopg
import pytest

from harness.adapters.postgres.pool import MAX_SIZE, PostgresPool, open_pool
from harness.observability.run_store.postgres import PostgresRunStore
from tests.conftest import POSTGRES_DSN


async def _backend_pid(pool: PostgresPool) -> int:
    async with pool.connection() as conn:
        row = await (await conn.execute("SELECT pg_backend_pid() AS pid")).fetchone()
    assert row is not None
    return row["pid"]


async def test_a_connection_the_server_closed_is_replaced_on_checkout(postgres_dsn):
    # Review Focus 1: what Neon's suspend leaves behind in the pool.
    pool = await open_pool(postgres_dsn)
    try:
        store = PostgresRunStore(pool)
        await store.create_table()
        await store.start_run("r1", "mine", "default", "fake", "sid-a")
        pid = await _backend_pid(pool)  # min_size 1, used one at a time: its only connection
        async with await psycopg.AsyncConnection.connect(POSTGRES_DSN, autocommit=True) as admin:
            await admin.execute("SELECT pg_terminate_backend(%s, 5000)", (pid,))

        # Failing closed here would 404 the visitor's own thread.
        assert await store.owns_thread("sid-a", "mine") is True
        assert await _backend_pid(pool) != pid
    finally:
        await pool.close()


async def test_a_checkout_prepares_nothing_on_the_server(postgres_dsn):
    # Live check on Neon: under prepare_threshold 0, psycopg_pool's own check prepared the
    # empty query on every call and never reused it: a second round trip per checkout
    # (41.5 ms against 21.0 ms), and one more statement left on the connection each time.
    pool = await open_pool(postgres_dsn)
    try:
        for _ in range(3):  # each checkout runs the pool's check first
            async with pool.connection() as conn:
                # Unprepared itself, so the listing doesn't list its own statement.
                listing = await conn.execute(
                    "SELECT name FROM pg_prepared_statements", prepare=False
                )
                assert await listing.fetchall() == []
    finally:
        await pool.close()


async def _hold_a_query(pool: PostgresPool) -> None:
    async with pool.connection() as conn:
        await conn.execute("SELECT pg_sleep(10)")


async def test_cancelled_queries_give_their_connections_back(postgres_dsn):
    # Review Focus 2: a visitor closing the tab cancels the run mid-query.
    pool = await open_pool(postgres_dsn)
    try:
        for _ in range(MAX_SIZE + 1):
            task = asyncio.create_task(_hold_a_query(pool))
            await asyncio.sleep(0.2)  # let the query reach the server
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        async with pool.connection(timeout=5) as conn:
            row = await (await conn.execute("SELECT 1 AS one")).fetchone()
        assert row == {"one": 1}
    finally:
        await pool.close()
