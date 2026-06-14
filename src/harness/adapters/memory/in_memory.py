"""In-memory memory adapters.

InMemoryShortTerm  - dev/test stand-in for the LangGraph-checkpointer-backed
                     history. Swap to the checkpointer impl in orchestration/.
InMemoryLongTerm   - a trivial substring-scored LongTermMemory so the whole
                     app runs with zero external services. Replace with
                     PgVectorLongTerm for real semantic recall; both satisfy the
                     same contract test.
"""
from __future__ import annotations

import uuid
from collections import defaultdict

from harness.core.memory.store import LongTermMemory, MemoryHit, ShortTermMemory
from harness.core.types import Message


class InMemoryShortTerm(ShortTermMemory):
    def __init__(self) -> None:
        self._store: dict[str, list[Message]] = defaultdict(list)

    async def append(self, conversation_id: str, message: Message) -> None:
        self._store[conversation_id].append(message)

    async def history(self, conversation_id: str) -> list[Message]:
        return list(self._store[conversation_id])

    async def clear(self, conversation_id: str) -> None:
        self._store.pop(conversation_id, None)


class InMemoryLongTerm(LongTermMemory):
    """Naive lexical scoring. Good enough to exercise the interface; NOT
    semantic. Use PgVectorLongTerm in real runs."""

    def __init__(self) -> None:
        self._items: dict[str, tuple[str, dict[str, str]]] = {}

    async def write(self, text: str, metadata: dict[str, str] | None = None) -> str:
        mid = str(uuid.uuid4())
        self._items[mid] = (text, metadata or {})
        return mid

    async def search(self, query: str, k: int = 5) -> list[MemoryHit]:
        q = set(query.lower().split())
        scored: list[MemoryHit] = []
        unscored: list[MemoryHit] = []
        for text, meta in self._items.values():
            words = set(text.lower().split())
            overlap = len(q & words)
            if overlap:
                score = overlap / max(len(q), 1)
                scored.append(MemoryHit(text=text, score=score, metadata=meta))
            else:
                unscored.append(MemoryHit(text=text, score=0.0, metadata=meta))
        scored.sort(key=lambda h: h.score, reverse=True)
        # Fall back to returning all memories when no keyword matches so the
        # agent is never left empty-handed just because the query phrasing
        # differs from the stored text.
        return (scored or unscored)[:k]

    async def update(self, memory_id: str, text: str) -> None:
        if memory_id in self._items:
            _, meta = self._items[memory_id]
            self._items[memory_id] = (text, meta)
