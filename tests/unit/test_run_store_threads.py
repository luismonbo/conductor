"""list_threads groups runs by thread, newest first."""
from typing import AsyncGenerator

import aiosqlite
import pytest

from harness.observability.run_store import RunStore


@pytest.fixture
async def store() -> AsyncGenerator[RunStore, None]:
    conn = await aiosqlite.connect(":memory:")
    s = RunStore(conn)
    await s.create_table()
    yield s
    await conn.close()


async def test_list_threads_groups_and_orders(store):
    await store.start_run("r1", "thread-a", "default", "fake")
    await store.start_run("r2", "thread-a", "default", "fake")
    await store.start_run("r3", "thread-b", "default", "fake")

    threads = await store.list_threads()

    assert [t["thread_id"] for t in threads] == ["thread-b", "thread-a"]
    by_id = {t["thread_id"]: t for t in threads}
    assert by_id["thread-a"]["runs"] == 2
    assert by_id["thread-b"]["runs"] == 1
    assert by_id["thread-a"]["last_at"]  # ISO timestamp present


async def test_list_threads_respects_limit(store):
    for i in range(5):
        await store.start_run(f"r{i}", f"t{i}", "default", "fake")
    assert len(await store.list_threads(limit=3)) == 3
