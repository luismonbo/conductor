# tests/unit/test_vector_store_document_stats.py
from __future__ import annotations

import pytest

from harness.adapters.embedding.fake import FakeEmbedder
from harness.adapters.vectorstore.in_memory import InMemoryVectorStore
from harness.core.rag.document import Chunk


def _chunk(chunk_id: str, document_id: str, collection: str = "papers") -> Chunk:
    return Chunk(
        chunk_id=chunk_id, document_id=document_id, collection=collection,
        text="x", section_path=(),
    )


@pytest.mark.asyncio
async def test_document_stats_counts_chunks_per_document():
    store = InMemoryVectorStore()
    embedder = FakeEmbedder(dimension=4)
    [vec] = await embedder.embed(["x"])
    chunks = [_chunk("d1:0", "d1"), _chunk("d1:1", "d1"), _chunk("d2:0", "d2")]
    await store.upsert(chunks, [vec] * 3)

    assert await store.document_stats() == {"d1": 2, "d2": 1}


@pytest.mark.asyncio
async def test_document_stats_filters_by_collection():
    store = InMemoryVectorStore()
    embedder = FakeEmbedder(dimension=4)
    [vec] = await embedder.embed(["x"])
    await store.upsert(
        [_chunk("a:0", "a", "papers"), _chunk("b:0", "b", "recipes")], [vec, vec]
    )

    assert await store.document_stats(collection="papers") == {"a": 1}


@pytest.mark.asyncio
async def test_document_stats_is_empty_for_empty_store():
    assert await InMemoryVectorStore().document_stats() == {}
