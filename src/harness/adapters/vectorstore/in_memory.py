"""In-memory VectorStore — test double and zero-infra local fallback, same
role as adapters/memory/in_memory.py."""
from __future__ import annotations

from harness.core.rag.document import Chunk, ScoredChunk


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._rows: dict[str, tuple[Chunk, list[float]]] = {}

    async def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        for chunk, embedding in zip(chunks, embeddings):
            self._rows[chunk.chunk_id] = (chunk, embedding)

    async def search(
        self,
        query_embedding: list[float],
        k: int,
        collection: str | None = None,
        filters: dict[str, str] | None = None,
    ) -> list[ScoredChunk]:
        candidates = list(self._rows.values())
        if collection is not None:
            candidates = [row for row in candidates if row[0].collection == collection]
        if filters:
            candidates = [
                row for row in candidates
                if all(row[0].metadata.get(key) == value for key, value in filters.items())
            ]
        scored = [
            ScoredChunk(chunk=chunk, score=_cosine(query_embedding, embedding))
            for chunk, embedding in candidates
        ]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:k]

    async def delete(self, document_id: str) -> None:
        self._rows = {
            chunk_id: row for chunk_id, row in self._rows.items()
            if row[0].document_id != document_id
        }

    async def count(self, collection: str | None = None) -> int:
        if collection is None:
            return len(self._rows)
        return sum(1 for chunk, _ in self._rows.values() if chunk.collection == collection)
