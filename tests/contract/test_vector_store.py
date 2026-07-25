"""Contract suite for VectorStore. Any adapter implementing the protocol must
pass this. Parametrize a new backend here and it inherits the behavioral
guarantees for free — mirrors tests/contract/test_long_term_memory.py."""
from __future__ import annotations

import dataclasses

import pytest

from harness.adapters.vectorstore.in_memory import InMemoryVectorStore
from harness.core.rag.document import Chunk


def _chunk(chunk_id: str, document_id: str, collection: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        collection=collection,
        text=text,
        section_path=("Intro",),
        order=0,
        source_path="paper.pdf",
        embedding_model="fake",
        created_at="2026-07-25T00:00:00Z",
    )


@pytest.fixture(params=["in_memory"])
def store(request):
    if request.param == "in_memory":
        return InMemoryVectorStore()
    raise ValueError(request.param)


@pytest.mark.asyncio
async def test_upsert_then_search_finds_it(store):
    chunk = _chunk("doc1:0", "doc1", "papers", "transformers use self-attention")
    await store.upsert([chunk], [[1.0, 0.0, 0.0]])
    results = await store.search([1.0, 0.0, 0.0], k=3)
    assert results
    assert results[0].chunk.chunk_id == "doc1:0"


@pytest.mark.asyncio
async def test_search_empty_store_returns_empty(store):
    results = await store.search([1.0, 0.0, 0.0], k=3)
    assert results == []


@pytest.mark.asyncio
async def test_search_respects_k(store):
    chunks = [_chunk(f"doc1:{i}", "doc1", "papers", f"chunk {i}") for i in range(5)]
    embeddings = [[float(i + 1), 0.0, 0.0] for i in range(5)]
    await store.upsert(chunks, embeddings)
    results = await store.search([1.0, 0.0, 0.0], k=2)
    assert len(results) == 2


@pytest.mark.asyncio
async def test_search_filters_by_collection(store):
    a = _chunk("a:0", "a", "papers", "paper content")
    b = _chunk("b:0", "b", "recipes", "recipe content")
    await store.upsert([a, b], [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    results = await store.search([1.0, 0.0, 0.0], k=5, collection="papers")
    assert len(results) == 1
    assert results[0].chunk.document_id == "a"


@pytest.mark.asyncio
async def test_delete_removes_document_chunks(store):
    chunk = _chunk("doc1:0", "doc1", "papers", "text")
    await store.upsert([chunk], [[1.0, 0.0, 0.0]])
    await store.delete("doc1")
    results = await store.search([1.0, 0.0, 0.0], k=3)
    assert results == []


@pytest.mark.asyncio
async def test_upsert_same_chunk_id_replaces_not_duplicates(store):
    chunk = _chunk("doc1:0", "doc1", "papers", "original text")
    await store.upsert([chunk], [[1.0, 0.0, 0.0]])
    revised = dataclasses.replace(chunk, text="revised text")
    await store.upsert([revised], [[1.0, 0.0, 0.0]])
    assert await store.count(collection="papers") == 1


@pytest.mark.asyncio
async def test_count_reflects_upserts(store):
    assert await store.count() == 0
    chunk = _chunk("doc1:0", "doc1", "papers", "text")
    await store.upsert([chunk], [[1.0, 0.0, 0.0]])
    assert await store.count() == 1


@pytest.mark.asyncio
async def test_search_filters_by_metadata(store):
    a = dataclasses.replace(_chunk("a:0", "a", "papers", "table content"), metadata={"kind": "table"})
    b = dataclasses.replace(_chunk("b:0", "b", "papers", "prose content"), metadata={"kind": "prose"})
    await store.upsert([a, b], [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    results = await store.search([1.0, 0.0, 0.0], k=5, filters={"kind": "table"})
    assert len(results) == 1
    assert results[0].chunk.document_id == "a"


@pytest.mark.asyncio
async def test_search_round_trips_every_chunk_field(store):
    """A retrieved Chunk must be indistinguishable from the one that was stored.

    assemble_prompt() renders source_path and section_path into the grounded
    prompt, so a backend that drops them silently strips every answer of its
    attribution. Backends that can't store a field as a column persist the
    payload alongside it (see PgVectorStore/MilvusStore chunk_json).
    """
    original = Chunk(
        chunk_id="doc1:7",
        document_id="doc1",
        collection="papers",
        text="the model uses self-attention",
        section_path=("Method", "Ablations"),
        section_kind="table",
        order=7,
        page_start=3,
        page_end=4,
        source_path="papers/attn.pdf",
        embedding_model="nomic-embed-text-v1.5",
        chunk_version=1,
        created_at="2026-07-25T00:00:00Z",
        metadata={"kind": "table"},
    )
    await store.upsert([original], [[1.0, 0.0, 0.0]])

    [scored] = await store.search([1.0, 0.0, 0.0], k=1)

    assert scored.chunk == original
