"""Recall@k — what proportion of the known-relevant items surfaced in top-k.

Fractional rather than a binary hit-rate, so a change that surfaces *more* of
the relevant set is visible; a binary score saturates at 1.0 on the first hit.

Rank-insensitive by construction: the same score whether the relevant chunk
lands at position 1 or position 5. That is what nDCG is for.

Caveat this implies for the dataset: fractional scores assume a case's label
set is exhaustive. Partial labels understate a good retriever.
"""

from __future__ import annotations

from evaluation.harness.metric import MetricResult
from evaluation.rag.dataset import RagEvalCase
from harness.core.rag.document import ScoredChunk


class RecallAtKMetric:
    name = "recall_at_k"

    async def score(
        self, case: RagEvalCase, retrieved: list[ScoredChunk]
    ) -> MetricResult:
        expected_chunks = set(case.expected.relevant_chunk_ids)
        expected_docs = set(case.expected.relevant_document_ids)
        if not expected_chunks and not expected_docs:
            return MetricResult(
                name=self.name,
                passed=True,
                score=1.0,
                reason="skipped — case declares no expectations",
                skipped=True,
            )

        # Chunk IDs are the finer signal but are invalidated by re-chunking;
        # document IDs are content-hash stable. Score against whichever the
        # case actually declares, preferring chunks, so the denominator is
        # never mixed across granularities within a case.
        if expected_chunks:
            expected_ids, granularity = expected_chunks, "chunk"
            retrieved_ids = {sc.chunk.chunk_id for sc in retrieved}
        else:
            expected_ids, granularity = expected_docs, "document"
            retrieved_ids = {sc.chunk.document_id for sc in retrieved}

        found = retrieved_ids & expected_ids
        score = len(found) / len(expected_ids)
        return MetricResult(
            name=self.name,
            passed=score
            > 0,  # "retrieved nothing relevant at all" alarm, not "recall passed"
            score=score,
            reason=(
                f"{len(found)}/{len(expected_ids)} relevant {granularity}(s) in top-k"
                if found
                else f"no relevant {granularity} in top-k"
            ),
            granularity=granularity,
        )
