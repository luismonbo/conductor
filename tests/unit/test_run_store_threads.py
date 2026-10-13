"""list_threads groups runs by thread, newest first, scoped to one session."""
from typing import AsyncGenerator

import aiosqlite
import pytest

from harness.observability.run_store import RunStore

SID_A = "session-a"
SID_B = "session-b"


@pytest.fixture
async def store() -> AsyncGenerator[RunStore, None]:
    conn = await aiosqlite.connect(":memory:")
    s = RunStore(conn)
    await s.create_table()
    yield s
    await conn.close()


async def test_list_threads_groups_and_orders(store):
    await store.start_run("r1", "thread-a", "default", "fake", SID_A)
    await store.start_run("r2", "thread-a", "default", "fake", SID_A)
    await store.start_run("r3", "thread-b", "default", "fake", SID_A)

    threads = await store.list_threads(SID_A)

    assert [t["thread_id"] for t in threads] == ["thread-b", "thread-a"]
    by_id = {t["thread_id"]: t for t in threads}
    assert by_id["thread-a"]["runs"] == 2
    assert by_id["thread-b"]["runs"] == 1
    assert by_id["thread-a"]["last_at"]  # ISO timestamp present


async def test_list_threads_respects_limit(store):
    for i in range(5):
        await store.start_run(f"r{i}", f"t{i}", "default", "fake", SID_A)
    assert len(await store.list_threads(SID_A, limit=3)) == 3


async def test_list_threads_excludes_other_sessions(store):
    await store.start_run("r1", "mine", "default", "fake", SID_A)
    await store.start_run("r2", "theirs", "default", "fake", SID_B)

    assert [t["thread_id"] for t in await store.list_threads(SID_A)] == ["mine"]
    assert [t["thread_id"] for t in await store.list_threads(SID_B)] == ["theirs"]


async def test_owns_thread(store):
    await store.start_run("r1", "mine", "default", "fake", SID_A)

    assert await store.owns_thread(SID_A, "mine") is True
    assert await store.owns_thread(SID_B, "mine") is False
    assert await store.owns_thread(SID_A, "never-existed") is False


async def test_owns_thread_fails_closed_when_the_store_errors():
    conn = await aiosqlite.connect(":memory:")
    store = RunStore(conn)
    await store.create_table()
    await store.start_run("r1", "mine", "default", "fake", SID_A)
    assert await store.owns_thread(SID_A, "mine") is True

    await conn.close()  # every query now raises

    assert await store.owns_thread(SID_A, "mine") is False


async def test_create_table_is_idempotent_and_migrates_legacy_rows():
    """A database created before this change gains the column without raising."""
    conn = await aiosqlite.connect(":memory:")
    await conn.execute(
        "CREATE TABLE runs (run_id TEXT PRIMARY KEY, thread_id TEXT, agent TEXT, "
        "backend TEXT, started_at TEXT, finished_at TEXT, stopped_reason TEXT, "
        "input_tokens INTEGER, output_tokens INTEGER, iterations INTEGER)"
    )
    await conn.execute(
        "INSERT INTO runs (run_id, thread_id, started_at) VALUES ('old', 'legacy', '2026-01-01')"
    )
    await conn.commit()

    store = RunStore(conn)
    await store.create_table()
    await store.create_table()  # second call must not raise

    # Legacy rows have a NULL session_id and therefore belong to nobody.
    assert await store.list_threads(SID_A) == []
    assert await store.owns_thread(SID_A, "legacy") is False
    await conn.close()
