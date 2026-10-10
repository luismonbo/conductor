"""Unit tests for SearchDocumentsTool."""
from __future__ import annotations

import pytest

from harness.adapters.embedding.fake import FakeEmbedder
from harness.adapters.tools.search_documents import SearchDocumentsTool
from harness.adapters.vectorstore.in_memory import InMemoryVectorStore
from harness.core.rag.document import Chunk
from harness.core.rag.serve import Retriever
from harness.core.tools.registry import ToolRegistry
from harness.core.types import ToolCall, ToolResult


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


class _NoHitsRetriever:
    """Stub Retriever that always reports zero hits, regardless of k.

    SearchDocumentsTool.run() now clamps k to a minimum of 1 (see _MAX_K),
    so k=0 can no longer be used to force a zero-hit response from a real
    Retriever/InMemoryVectorStore pair — and InMemoryVectorStore has no
    similarity threshold, so a real "semantic miss" can't otherwise produce
    zero hits from a nonempty index. A stub retriever isolates this branch
    (empty hits, nonempty collection) directly instead.
    """

    async def retrieve(self, query: str, k: int, collection: str | None = None):
        return []


@pytest.mark.asyncio
async def test_zero_hits_in_nonempty_collection_reports_no_relevant_match():
    chunk = Chunk(chunk_id="c1", document_id="d1", collection="papers", text="text", section_path=())
    store, _ = await _seeded_store_and_retriever([chunk])
    tool = SearchDocumentsTool(_NoHitsRetriever(), store, default_collection="papers")

    result = await tool.run({"query": "text"})

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


_PRIVATE = Chunk(
    chunk_id="p1", document_id="d1", collection="papers",
    text="private research notes", section_path=(),
)
_PUBLIC = Chunk(
    chunk_id="w1", document_id="d2", collection="docs",
    text="public project write-up", section_path=(),
)


async def _dispatch(tool: SearchDocumentsTool, arguments: dict) -> ToolResult:
    """Run the tool the way the agent loop does, so a refusal arrives as the model sees it."""
    registry = ToolRegistry()
    registry.register(tool)
    return await registry.dispatch(ToolCall(id="c1", name="search_documents", arguments=arguments))


@pytest.mark.asyncio
async def test_a_collection_outside_the_agents_list_is_refused():
    # A model can name a collection its schema never offered (with one collection
    # the schema has no collection parameter at all); those documents stay unreachable.
    store, retriever = await _seeded_store_and_retriever([_PRIVATE, _PUBLIC])
    tool = SearchDocumentsTool(retriever, store, default_collection="docs", collections=["docs"])

    result = await _dispatch(tool, {"query": "research notes", "collection": "papers"})

    assert result.is_error
    assert "private research notes" not in result.content
    assert "docs" in result.content  # tells the model what it can search instead


@pytest.mark.asyncio
async def test_a_collection_on_the_agents_list_can_be_searched():
    store, retriever = await _seeded_store_and_retriever([_PRIVATE, _PUBLIC])
    tool = SearchDocumentsTool(
        retriever, store, default_collection="papers", collections=["papers", "docs"],
    )

    result = await _dispatch(tool, {"query": "write-up", "collection": "docs"})

    assert not result.is_error
    assert "public project write-up" in result.content


@pytest.mark.asyncio
async def test_the_agents_default_collection_stays_searchable_when_unlisted():
    # The default agent's list comes from the ingest manifests; the collection its
    # deployment configures as the default must stay searchable if they omit it.
    store, retriever = await _seeded_store_and_retriever([_PRIVATE, _PUBLIC])
    tool = SearchDocumentsTool(retriever, store, default_collection="papers", collections=["docs"])

    result = await _dispatch(tool, {"query": "notes"})

    assert not result.is_error
    assert "private research notes" in result.content
