"""Mean reciprocal rank — how high up the first relevant hit landed.

Complements recall_at_k: recall asks how *much* relevant material surfaced,
MRR asks how well the *first* hit was ranked. It stops at that first hit, so
it cannot see relevant chunks buried below it — which is precisely what a
reranker moves. That gap is why nDCG exists.
"""

from __future__ import annotations

from evaluation.harness.metric import MetricResult
from evaluation.rag.dataset import RagEvalCase
from harness.core.rag.document import ScoredChunk


class MRRMetric:
    name = "mrr"

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

        # Mirror RecallAtKMetric's granularity choice exactly. ORing the two
        # instead lets a case score MRR 1.00 while recall reports 0.00, because
        # any chunk from a labelled document counts as a hit — the two metrics
        # then describe the same retrieval differently, which is worse than
        # either being strict.
        if expected_chunks:
            expected_ids, granularity = expected_chunks, "chunk"
            retrieved_ids = [sc.chunk.chunk_id for sc in retrieved]
        else:
            expected_ids, granularity = expected_docs, "document"
            retrieved_ids = [sc.chunk.document_id for sc in retrieved]

        for rank, retrieved_id in enumerate(retrieved_ids, start=1):
            if retrieved_id in expected_ids:
                return MetricResult(
                    name=self.name,
                    passed=True,
                    score=1.0 / rank,
                    reason=f"first relevant {granularity} at rank {rank}",
                    granularity=granularity,
                )
        return MetricResult(
            name=self.name,
            passed=False,
            score=0.0,
            reason=f"no relevant {granularity} found",
            granularity=granularity,
        )
