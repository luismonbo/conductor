"""pgvector-backed LongTermMemory — Phase 5 target (skeleton, not yet implemented).

Replaces the retired Qdrant stub. Long-term memory will live in the SAME Postgres
as the LangGraph checkpointer (spec D9): one datastore, one managed service in the
cloud (`PostgresSaver` tables + a pgvector table coexist). Kept behind the
`LongTermMemory` Protocol, so selecting it stays a config change, not a rewrite.

Embeddings come from an injected callable — exactly the seam the Qdrant version
had — so llama-server `/v1/embeddings` can be used locally and Azure OpenAI
embeddings in the cloud without editing this file.

Status: SKELETON. The real implementation (psycopg3 + pgvector, schema bootstrap,
cosine-distance search) and its docker-Postgres contract test land in Phase 5
(see tests/contract/test_long_term_memory.py). Until then `build.py` raises
NotImplementedError for the `pgvector` backend and the default stays in_memory.
Intended dependency: the `pgvector` extra (`uv sync --extra pgvector`).
"""
from __future__ import annotations

from typing import Awaitable, Callable

from harness.core.memory.store import MemoryHit

Embedder = Callable[[str], Awaitable[list[float]]]

_NOT_IMPLEMENTED = (
    "PgVectorLongTerm is a Phase-5 skeleton; the pgvector backend is not yet "
    "implemented. Track it in docs/superpowers and keep HARNESS_MEMORY_BACKEND=in_memory."
)


class PgVectorLongTerm:
    """LongTermMemory over pgvector. Same interface as InMemoryLongTerm and the
    retired QdrantLongTerm, so switching remains a config change. See the module
    docstring for status and the Phase-5 implementation plan."""

    def __init__(
        self,
        embedder: Embedder,
        dsn: str,
        table: str = "long_term_memory",
        vector_size: int = 1536,
    ) -> None:
        self._embed = embedder
        self._dsn = dsn
        self._table = table
        self._vector_size = vector_size

    async def ensure_schema(self) -> None:
        """Phase 5: CREATE EXTENSION IF NOT EXISTS vector; create the memory
        table + an ANN index (e.g. ivfflat/hnsw) on the embedding column."""
        raise NotImplementedError(_NOT_IMPLEMENTED)

    async def write(self, text: str, metadata: dict[str, str] | None = None) -> str:
        """Phase 5: embed `text`, INSERT (id, text, embedding, metadata), return id."""
        raise NotImplementedError(_NOT_IMPLEMENTED)

    async def search(self, query: str, k: int = 5) -> list[MemoryHit]:
        """Phase 5: embed `query`, ORDER BY embedding <=> query LIMIT k, map rows
        to MemoryHit (score = 1 - cosine_distance)."""
        raise NotImplementedError(_NOT_IMPLEMENTED)

    async def update(self, memory_id: str, text: str) -> None:
        """Phase 5: re-embed `text` and UPDATE the row identified by `memory_id`."""
        raise NotImplementedError(_NOT_IMPLEMENTED)
