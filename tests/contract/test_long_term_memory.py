"""Contract suite for LongTermMemory. Any adapter implementing the protocol
must pass this. When you add PgVectorLongTerm, parametrize this against it (with
a docker Postgres) and you inherit the behavioral guarantees for free."""
from __future__ import annotations

import pytest

from harness.adapters.memory.in_memory import InMemoryLongTerm
from harness.adapters.memory.sqlite_store import SqliteLongTermMemory


@pytest.fixture(params=["in_memory", "sqlite"])
def memory(request, tmp_path):
    if request.param == "sqlite":
        return SqliteLongTermMemory(str(tmp_path / "test_memory.sqlite"))
    return InMemoryLongTerm()


@pytest.mark.asyncio
async def test_write_then_search_finds_it(memory):
    await memory.write("the capital of France is Paris")
    hits = await memory.search("France capital", k=3)
    assert hits
    assert "Paris" in hits[0].text


@pytest.mark.asyncio
async def test_search_empty_returns_empty(memory):
    hits = await memory.search("nothing stored yet", k=3)
    assert hits == []


@pytest.mark.asyncio
async def test_update_changes_text(memory):
    mid = await memory.write("old fact about cats")
    await memory.update(mid, "new fact about cats")
    hits = await memory.search("cats fact", k=1)
    assert "new" in hits[0].text
