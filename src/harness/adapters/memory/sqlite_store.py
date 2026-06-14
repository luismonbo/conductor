"""SQLite-backed LongTermMemory — persistent, lexical search, zero-infra.

Same interface as InMemoryLongTerm; activate via HARNESS_MEMORY_BACKEND=sqlite.
PgVectorLongTerm replaces this when semantic recall and Docker Postgres are ready.
"""
from __future__ import annotations

import json
import uuid

import aiosqlite

from harness.core.memory.store import LongTermMemory, MemoryHit

_SCHEMA = """
CREATE TABLE IF NOT EXISTS long_term_memory (
    id         TEXT PRIMARY KEY,
    text       TEXT NOT NULL,
    metadata   TEXT NOT NULL,
    created_at REAL NOT NULL DEFAULT (unixepoch('now'))
)
"""


class SqliteLongTermMemory(LongTermMemory):
    def __init__(self, path: str = "./harness_memory.sqlite") -> None:
        self._path = path

    async def _ensure_schema(self) -> None:
        async with aiosqlite.connect(self._path) as db:
            await db.execute(_SCHEMA)
            await db.commit()

    async def write(self, text: str, metadata: dict[str, str] | None = None) -> str:
        await self._ensure_schema()
        mid = str(uuid.uuid4())
        async with aiosqlite.connect(self._path) as db:
            await db.execute(
                "INSERT INTO long_term_memory (id, text, metadata) VALUES (?, ?, ?)",
                (mid, text, json.dumps(metadata or {})),
            )
            await db.commit()
        return mid

    async def search(self, query: str, k: int = 5) -> list[MemoryHit]:
        await self._ensure_schema()
        async with aiosqlite.connect(self._path) as db:
            async with db.execute("SELECT text, metadata FROM long_term_memory") as cur:
                rows = await cur.fetchall()

        q = set(query.lower().split())
        scored: list[MemoryHit] = []
        unscored: list[MemoryHit] = []
        for text, meta_json in rows:
            words = set(text.lower().split())
            overlap = len(q & words)
            if overlap:
                score = overlap / max(len(q), 1)
                scored.append(MemoryHit(text=text, score=score, metadata=json.loads(meta_json)))
            else:
                unscored.append(MemoryHit(text=text, score=0.0, metadata=json.loads(meta_json)))
        scored.sort(key=lambda h: h.score, reverse=True)
        # Fall back to returning all memories when no keyword matches so the
        # agent is never left empty-handed just because the query phrasing
        # differs from the stored text.
        return (scored or unscored)[:k]

    async def update(self, memory_id: str, text: str) -> None:
        await self._ensure_schema()
        async with aiosqlite.connect(self._path) as db:
            await db.execute(
                "UPDATE long_term_memory SET text = ? WHERE id = ?",
                (text, memory_id),
            )
            await db.commit()
