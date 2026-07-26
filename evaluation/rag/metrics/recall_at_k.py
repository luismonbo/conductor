"""Recall@k — what proportion of the known-relevant items surfaced in top-k.

Fractional rather than a binary hit-rate: `passed` keeps the same semantics
either way (at least one relevant item found), but a fractional `score` leaves
headroom for Phase 2 to demonstrate that reranking/hybrid retrieval surfaced
*more* of the relevant set. A binary score saturates at 1.0 as soon as one
relevant chunk is found, making exactly those improvements invisible.

Caveat this implies for the dataset: fractional scores assume a case's label
set is exhaustive. Partial labels understate a good retriever.
"""
from __future__ import annotations

from evaluation.harness.metric import MetricResult


class RecallAtKMetric:
    name = "recall_at_k"

    async def score(self, case, result, tracer) -> MetricResult:
        expected_chunks = set(case.expected.relevant_chunk_ids)
        expected_docs = set(case.expected.relevant_document_ids)
        if not expected_chunks and not expected_docs:
            return MetricResult(name=self.name, passed=True, score=1.0, reason="skipped")

        # Chunk IDs are the finer signal but are invalidated by re-chunking;
        # document IDs are content-hash stable. Score against whichever the
        # case actually declares, preferring chunks, so the denominator is
        # never mixed across granularities.
        if expected_chunks:
            expected_ids = expected_chunks
            retrieved_ids = {sc.chunk.chunk_id for sc in result.retrieved}
            granularity = "chunk"
        else:
            expected_ids = expected_docs
            retrieved_ids = {sc.chunk.document_id for sc in result.retrieved}
            granularity = "document"

        found = retrieved_ids & expected_ids
        score = len(found) / len(expected_ids)
        return MetricResult(
            name=self.name,
            passed=score > 0,
            score=score,
            reason=(
                f"{len(found)}/{len(expected_ids)} relevant {granularity}(s) in top-k"
                if found
                else f"no relevant {granularity} in top-k"
            ),
        )
