"""Mean reciprocal rank — how high up the first relevant hit landed.

Complements recall_at_k: recall asks how *much* relevant material surfaced,
MRR asks how well it was *ranked*. A retriever that buries the right chunk at
position 5 scores the same recall as one that puts it first, but 0.2 here
versus 1.0.
"""
from __future__ import annotations

from evaluation.harness.metric import MetricResult


class MRRMetric:
    name = "mrr"

    async def score(self, case, result, tracer) -> MetricResult:
        expected_chunks = set(case.expected.relevant_chunk_ids)
        expected_docs = set(case.expected.relevant_document_ids)
        if not expected_chunks and not expected_docs:
            return MetricResult(name=self.name, passed=True, score=1.0, reason="skipped")

        # Mirror RecallAtKMetric's granularity choice exactly: prefer chunk ids
        # when the case declares them, else fall back to documents. ORing the
        # two instead lets a case score MRR 1.00 ("first relevant hit at rank
        # 1") while recall reports 0.00, because any chunk from a labelled
        # document counts as a hit -- the two metrics then describe the same
        # retrieval differently, which is worse than either being strict.
        if expected_chunks:
            expected_ids = expected_chunks
            retrieved_ids = [sc.chunk.chunk_id for sc in result.retrieved]
            granularity = "chunk"
        else:
            expected_ids = expected_docs
            retrieved_ids = [sc.chunk.document_id for sc in result.retrieved]
            granularity = "document"

        for rank, retrieved_id in enumerate(retrieved_ids, start=1):
            if retrieved_id in expected_ids:
                return MetricResult(
                    name=self.name, passed=True, score=1.0 / rank,
                    reason=f"first relevant {granularity} at rank {rank}",
                )
        return MetricResult(
            name=self.name, passed=False, score=0.0,
            reason=f"no relevant {granularity} found",
        )
