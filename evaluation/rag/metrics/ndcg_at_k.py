"""nDCG@k — normalized discounted cumulative gain over graded relevance.

The rank-aware metric recall_at_k and mrr cannot supply. Recall is a set
intersection and sees no ordering at all; MRR stops at the first relevant hit.
A reranker's job is promoting the 2nd-through-nth relevant chunks, which only
nDCG observes.

Exponential gain (2**grade - 1, so grades 0/1/2/3 give 0/1/3/7) rather than
linear, matching TREC/LETOR convention: it makes landing the *perfect* chunk
at rank 1 dominate, which is the behaviour a reranker is bought for.

Scores are comparable only at the same k and against the same labels.
Unjudged chunks count as grade 0 — the standard pooling assumption, which
understates a retriever that surfaces a genuinely relevant chunk no pooled
config ever ranked. See datasets/RUBRIC.md.
"""

from __future__ import annotations

import math

from evaluation.harness.metric import MetricResult
from evaluation.rag.dataset import RagEvalCase
from harness.core.rag.document import ScoredChunk


def _dcg(grades: list[int]) -> float:
    return sum((2**g - 1) / math.log2(rank + 1) for rank, g in enumerate(grades, start=1))


class NDCGAtKMetric:
    name = "ndcg_at_k"

    def __init__(self, k: int = 5) -> None:
        self._k = k

    async def score(
        self, case: RagEvalCase, retrieved: list[ScoredChunk]
    ) -> MetricResult:
        graded = case.expected.graded_chunks
        # Document-granularity ranking is near-degenerate on a small corpus:
        # "did the right paper appear" saturates and a reranker barely moves
        # it. nDCG scores chunk-graded cases only.
        if not graded or not any(graded.values()):
            return MetricResult(
                name=self.name,
                passed=True,
                score=1.0,
                reason="skipped — case has no graded chunks",
                skipped=True,
            )

        actual = [graded.get(sc.chunk.chunk_id, 0) for sc in retrieved[: self._k]]
        # IDCG comes from the full label set, never from the retrieved grades:
        # deriving it from what was retrieved lets a retriever that missed a
        # relevant chunk be graded against its own failure and score 1.0.
        ideal = sorted(graded.values(), reverse=True)[: self._k]

        dcg, idcg = _dcg(actual), _dcg(ideal)
        score = dcg / idcg if idcg > 0 else 0.0
        return MetricResult(
            name=self.name,
            passed=score > 0,
            score=score,
            reason=f"nDCG@{self._k}={score:.4f} (DCG {dcg:.4f} / IDCG {idcg:.4f})",
            granularity="chunk",
        )
