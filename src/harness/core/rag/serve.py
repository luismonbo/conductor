"""Serving pipeline: Retriever (embed query -> vector search) is a standalone
primitive, RagPipeline composes it with generation. Kept separate on purpose
— Phase 3's future agent-facing tool calls Retriever.retrieve() alone and
hands the agent raw chunks, the same way RecallTool returns raw memory hits
today, so the agent can assess sufficiency and re-retrieve rather than
reasoning over a pre-baked answer. See "Serving pipeline" in
docs/superpowers/specs/2026-07-25-rag-ingestion-retrieval-design.md."""
from __future__ import annotations

from dataclasses import dataclass

from harness.core.llm.client import LLMClient
from harness.core.rag.document import ScoredChunk
from harness.core.rag.ports import Embedder, VectorStore
from harness.core.types import Message, Role
from harness.observability.tracer import TraceCollector

_GROUNDING_INSTRUCTION = (
    "Answer the question using ONLY the context below. If the context does "
    "not contain enough information to answer, say so explicitly rather "
    "than guessing."
)


class Retriever:
    def __init__(self, embedder: Embedder, vector_store: VectorStore) -> None:
        self._embedder = embedder
        self._vector_store = vector_store

    async def retrieve(
        self, query: str, k: int = 5, collection: str | None = None
    ) -> list[ScoredChunk]:
        [query_embedding] = await self._embedder.embed([query])
        return await self._vector_store.search(query_embedding, k=k, collection=collection)


class DiversifiedRetriever:
    """Enforces a per-document quota over another Retriever's ranking.

    Flat top-k collapses onto whichever document dominates a query. Measured on
    the `papers` corpus (111 chunks vs 41), a multi-hop question had its two
    answer chunks ranked #1 and #2 *within their own document* yet 4th and 6th
    globally — the larger document crowded them out of a top-5. A quota reaches
    them at k=4 where a flat search needs k=6, and the gap widens with corpus
    imbalance.

    It over-fetches, takes the best `per_document_k` from each document, then
    backfills any remaining slots in score order so a single-document corpus is
    never starved. Ordering of the final list stays by score — the quota decides
    *membership*, not rank.

    What this deliberately does NOT fix: a query whose single embedding matches
    neither document well (asking about two papers at once lands between them).
    On that case the answer chunk ranked 60th *within its own document*, which
    no quota can reach. That needs query decomposition, not retrieval tuning.
    """

    def __init__(
        self, retriever: "Retriever", per_document_k: int = 2, overfetch: int = 5
    ) -> None:
        self._retriever = retriever
        self._per_document_k = per_document_k
        self._overfetch = overfetch

    async def retrieve(
        self, query: str, k: int = 5, collection: str | None = None
    ) -> list[ScoredChunk]:
        candidates = await self._retriever.retrieve(query, k * self._overfetch, collection)

        taken: list[ScoredChunk] = []
        leftover: list[ScoredChunk] = []
        seen_per_document: dict[str, int] = {}
        for scored in candidates:
            document_id = scored.chunk.document_id
            count = seen_per_document.get(document_id, 0)
            if count < self._per_document_k:
                seen_per_document[document_id] = count + 1
                taken.append(scored)
            else:
                leftover.append(scored)

        selected = taken[:k]
        if len(selected) < k:
            selected = selected + leftover[: k - len(selected)]
        return sorted(selected, key=lambda sc: sc.score, reverse=True)


def assemble_prompt(query: str, retrieved: list[ScoredChunk]) -> list[Message]:
    if not retrieved:
        context = "(no relevant context was found)"
    else:
        context = "\n\n".join(
            f"[{i + 1}] (source: {sc.chunk.source_path}, "
            f"section: {' > '.join(sc.chunk.section_path) or 'n/a'})\n{sc.chunk.text}"
            for i, sc in enumerate(retrieved)
        )
    return [
        Message(Role.SYSTEM, _GROUNDING_INSTRUCTION),
        Message(Role.USER, f"Context:\n{context}\n\nQuestion: {query}"),
    ]


def render_for_trace(messages: list[Message]) -> str:
    return "\n\n".join(f"[{m.role.value}] {m.content}" for m in messages)


@dataclass(frozen=True)
class RagResult:
    answer: str
    retrieved: tuple[ScoredChunk, ...]
    assembled_prompt: str


class RagPipeline:
    def __init__(
        self, retriever: Retriever, llm: LLMClient, tracer: TraceCollector | None = None
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._tracer = tracer

    async def answer(self, query: str, k: int = 5, collection: str | None = None) -> RagResult:
        retrieved = await self._retriever.retrieve(query, k, collection)
        if self._tracer is not None:
            await self._tracer(
                "retrieval_result",
                {
                    "query": query,
                    "k": k,
                    "chunk_ids": [sc.chunk.chunk_id for sc in retrieved],
                    "scores": [sc.score for sc in retrieved],
                },
            )
        messages = assemble_prompt(query, retrieved)
        response = await self._llm.generate(messages)
        return RagResult(
            answer=response.text,
            retrieved=tuple(retrieved),
            assembled_prompt=render_for_trace(messages),
        )
