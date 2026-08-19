"""Unit tests for SearchDocumentsTool."""
from __future__ import annotations

import pytest

from harness.adapters.embedding.fake import FakeEmbedder
from harness.adapters.tools.search_documents import SearchDocumentsTool
from harness.adapters.vectorstore.in_memory import InMemoryVectorStore
from harness.core.rag.document import Chunk
from harness.core.rag.serve import Retriever


async def _seeded_store_and_retriever(chunks: list[Chunk]) -> tuple[InMemoryVectorStore, Retriever]:
    store = InMemoryVectorStore()
    embedder = FakeEmbedder(dimension=4)
    if chunks:
        vectors = await embedder.embed([c.text for c in chunks])
        await store.upsert(chunks, vectors)
    return store, Retriever(embedder=embedder, vector_store=store)


@pytest.mark.asyncio
async def test_formats_hits_with_score_source_and_section():
    chunk = Chunk(
        chunk_id="c1", document_id="d1", collection="papers",
        text="the model uses self-attention", section_path=("Method", "Ablations"),
        source_path="papers/attn.pdf",
    )
    store, retriever = await _seeded_store_and_retriever([chunk])
    tool = SearchDocumentsTool(retriever, store, default_collection="papers")

    result = await tool.run({"query": "self-attention"})

    assert "papers/attn.pdf" in result
    assert "Method > Ablations" in result
    assert "the model uses self-attention" in result
    assert result.startswith("[1] (score:")


@pytest.mark.asyncio
async def test_empty_collection_returns_not_found_message_not_silent_empty():
    store, retriever = await _seeded_store_and_retriever([])
    tool = SearchDocumentsTool(retriever, store, default_collection="papers")

    result = await tool.run({"query": "anything"})

    assert result == "No documents found in collection 'papers'."


@pytest.mark.asyncio
async def test_zero_hits_in_nonempty_collection_reports_no_relevant_match():
    chunk = Chunk(chunk_id="c1", document_id="d1", collection="papers", text="text", section_path=())
    store, retriever = await _seeded_store_and_retriever([chunk])
    tool = SearchDocumentsTool(retriever, store, default_collection="papers")

    # k=0 forces zero hits from a nonempty collection, isolating this branch
    # from the empty-collection one above — InMemoryVectorStore has no
    # similarity threshold, so a real "semantic miss" can't otherwise
    # produce zero hits from a nonempty index.
    result = await tool.run({"query": "text", "k": 0})

    assert result == "No relevant documents found."


@pytest.mark.asyncio
async def test_k_defaults_when_omitted():
    chunks = [
        Chunk(chunk_id=f"c{i}", document_id="d1", collection="papers", text=f"chunk {i}", section_path=())
        for i in range(3)
    ]
    store, retriever = await _seeded_store_and_retriever(chunks)
    tool = SearchDocumentsTool(retriever, store, default_collection="papers", default_k=2)

    result = await tool.run({"query": "chunk"})

    assert result.count("(score:") == 2


@pytest.mark.asyncio
async def test_schema_omits_collection_param_with_single_collection():
    store, retriever = await _seeded_store_and_retriever([])
    tool = SearchDocumentsTool(retriever, store, default_collection="papers", collections=["papers"])

    assert "collection" not in tool.spec.parameters["properties"]


@pytest.mark.asyncio
async def test_schema_includes_collection_enum_with_multiple_collections():
    store, retriever = await _seeded_store_and_retriever([])
    tool = SearchDocumentsTool(
        retriever, store, default_collection="papers", collections=["papers", "manuals"],
    )

    props = tool.spec.parameters["properties"]
    assert props["collection"]["enum"] == ["papers", "manuals"]
    assert props["collection"]["default"] == "papers"
