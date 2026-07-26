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

        for rank, scored_chunk in enumerate(result.retrieved, start=1):
            if (
                scored_chunk.chunk.chunk_id in expected_chunks
                or scored_chunk.chunk.document_id in expected_docs
            ):
                return MetricResult(
                    name=self.name, passed=True, score=1.0 / rank,
                    reason=f"first relevant hit at rank {rank}",
                )
        return MetricResult(
            name=self.name, passed=False, score=0.0, reason="no relevant chunk/document found"
        )
