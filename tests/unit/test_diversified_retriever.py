from __future__ import annotations

import pytest

from harness.adapters.embedding.fake import FakeEmbedder
from harness.adapters.vectorstore.in_memory import InMemoryVectorStore
from harness.core.rag.document import Chunk, ScoredChunk
from harness.core.rag.serve import DiversifiedRetriever


class _StubRetriever:
    """Returns a fixed ranking, and records the k it was asked for."""

    def __init__(self, ranking: list[tuple[str, str]]) -> None:
        self._ranking = ranking
        self.requested_k: int | None = None

    async def retrieve(self, query, k=5, collection=None):
        self.requested_k = k
        return [
            ScoredChunk(
                chunk=Chunk(
                    chunk_id=chunk_id, document_id=document_id, collection="papers",
                    text="t", section_path=(),
                ),
                # descending score, so position in the list is the rank
                score=1.0 - i / 100,
            )
            for i, (chunk_id, document_id) in enumerate(self._ranking)
        ][:k]


@pytest.mark.asyncio
async def test_quota_promotes_a_second_documents_best_chunks():
    """The measured failure: one document's chunks occupy every top-k slot
    while the other document's best chunk sits just below the cutoff."""
    ranking = [
        ("A:1", "A"), ("A:2", "A"), ("A:3", "A"), ("A:4", "A"), ("A:5", "A"),
        ("B:1", "B"), ("B:2", "B"),
    ]
    base = _StubRetriever(ranking)
    retriever = DiversifiedRetriever(base, per_document_k=2, overfetch=4)

    results = await retriever.retrieve("q", k=4)

    ids = [sc.chunk.chunk_id for sc in results]
    assert ids == ["A:1", "A:2", "B:1", "B:2"], ids


@pytest.mark.asyncio
async def test_overfetches_so_the_quota_has_candidates_to_work_with():
    base = _StubRetriever([("A:1", "A")])
    retriever = DiversifiedRetriever(base, per_document_k=2, overfetch=5)

    await retriever.retrieve("q", k=4)

    assert base.requested_k == 20  # k * overfetch


@pytest.mark.asyncio
async def test_falls_back_to_global_order_to_fill_remaining_slots():
    """With only one document available the quota must not starve the result —
    leftovers backfill in score order."""
    ranking = [("A:1", "A"), ("A:2", "A"), ("A:3", "A"), ("A:4", "A")]
    retriever = DiversifiedRetriever(_StubRetriever(ranking), per_document_k=2, overfetch=4)

    results = await retriever.retrieve("q", k=4)

    assert [sc.chunk.chunk_id for sc in results] == ["A:1", "A:2", "A:3", "A:4"]


@pytest.mark.asyncio
async def test_result_is_ordered_by_score_not_by_document():
    ranking = [("A:1", "A"), ("B:1", "B"), ("A:2", "A"), ("B:2", "B")]
    retriever = DiversifiedRetriever(_StubRetriever(ranking), per_document_k=1, overfetch=4)

    results = await retriever.retrieve("q", k=4)

    scores = [sc.score for sc in results]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.asyncio
async def test_never_returns_more_than_k():
    ranking = [(f"{d}:{i}", d) for d in "ABC" for i in range(5)]
    retriever = DiversifiedRetriever(_StubRetriever(ranking), per_document_k=3, overfetch=4)

    results = await retriever.retrieve("q", k=4)

    assert len(results) == 4


@pytest.mark.asyncio
async def test_composes_with_a_real_retriever_end_to_end():
    store = InMemoryVectorStore()
    embedder = FakeEmbedder(dimension=8)
    chunks = [
        Chunk(chunk_id="a:0", document_id="a", collection="papers",
              text="transformers use self-attention", section_path=()),
        Chunk(chunk_id="b:0", document_id="b", collection="papers",
              text="meta prompting coordinates experts", section_path=()),
    ]
    vectors = await embedder.embed([c.text for c in chunks])
    await store.upsert(chunks, vectors)

    from harness.core.rag.serve import Retriever

    retriever = DiversifiedRetriever(
        Retriever(embedder=embedder, vector_store=store), per_document_k=1, overfetch=4
    )
    results = await retriever.retrieve("self-attention", k=2)

    assert {sc.chunk.document_id for sc in results} == {"a", "b"}
