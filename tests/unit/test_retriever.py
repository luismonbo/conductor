from __future__ import annotations

import pytest

from harness.adapters.embedding.fake import FakeEmbedder
from harness.adapters.vectorstore.in_memory import InMemoryVectorStore
from harness.core.rag.document import Chunk
from harness.core.rag.serve import Retriever


@pytest.mark.asyncio
async def test_retrieve_embeds_query_and_searches_store():
    store = InMemoryVectorStore()
    embedder = FakeEmbedder(dimension=8)
    chunk = Chunk(
        chunk_id="c1", document_id="d1", collection="papers",
        text="attention is all you need", section_path=(),
    )
    [vec] = await embedder.embed([chunk.text])
    await store.upsert([chunk], [vec])

    retriever = Retriever(embedder=embedder, vector_store=store)
    results = await retriever.retrieve("attention is all you need", k=3)

    assert results
    assert results[0].chunk.chunk_id == "c1"


@pytest.mark.asyncio
async def test_retrieve_passes_collection_through_to_store():
    store = InMemoryVectorStore()
    embedder = FakeEmbedder(dimension=4)
    a = Chunk(chunk_id="a", document_id="a", collection="papers", text="x", section_path=())
    b = Chunk(chunk_id="b", document_id="b", collection="recipes", text="x", section_path=())
    [vec] = await embedder.embed(["x"])
    await store.upsert([a, b], [vec, vec])

    retriever = Retriever(embedder=embedder, vector_store=store)
    results = await retriever.retrieve("x", k=5, collection="papers")

    assert len(results) == 1
    assert results[0].chunk.document_id == "a"
