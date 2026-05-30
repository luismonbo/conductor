"""Qdrant-backed LongTermMemory (real semantic recall).

Same interface as InMemoryLongTerm, so switching is a config change. Embeddings
come from an injected callable so you can use Azure OpenAI embeddings in the
cloud and a local embedding model on the Pi without editing this file.

Lazily imports qdrant-client so core/tests don't require it installed.
"""
from __future__ import annotations

import uuid
from typing import Awaitable, Callable

from harness.core.memory.store import MemoryHit

Embedder = Callable[[str], Awaitable[list[float]]]


class QdrantLongTerm:
    def __init__(
        self,
        embedder: Embedder,
        url: str,
        collection: str = "long_term_memory",
        vector_size: int = 1536,
    ) -> None:
        from qdrant_client import AsyncQdrantClient
        from qdrant_client.models import Distance, VectorParams

        self._client = AsyncQdrantClient(url=url)
        self._collection = collection
        self._embed = embedder
        self._vector_size = vector_size
        self._VectorParams = VectorParams
        self._Distance = Distance

    async def ensure_collection(self) -> None:
        existing = await self._client.get_collections()
        names = {c.name for c in existing.collections}
        if self._collection not in names:
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=self._VectorParams(
                    size=self._vector_size, distance=self._Distance.COSINE
                ),
            )

    async def write(self, text: str, metadata: dict[str, str] | None = None) -> str:
        from qdrant_client.models import PointStruct

        vector = await self._embed(text)
        mid = str(uuid.uuid4())
        payload = {"text": text, **(metadata or {})}
        await self._client.upsert(
            collection_name=self._collection,
            points=[PointStruct(id=mid, vector=vector, payload=payload)],
        )
        return mid

    async def search(self, query: str, k: int = 5) -> list[MemoryHit]:
        vector = await self._embed(query)
        results = await self._client.search(
            collection_name=self._collection, query_vector=vector, limit=k
        )
        hits: list[MemoryHit] = []
        for r in results:
            payload = dict(r.payload or {})
            text = payload.pop("text", "")
            hits.append(MemoryHit(text=text, score=r.score, metadata=payload))
        return hits

    async def update(self, memory_id: str, text: str) -> None:
        from qdrant_client.models import PointStruct

        vector = await self._embed(text)
        await self._client.upsert(
            collection_name=self._collection,
            points=[
                PointStruct(id=memory_id, vector=vector, payload={"text": text})
            ],
        )
