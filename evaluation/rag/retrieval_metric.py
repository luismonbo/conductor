"""RetrievalMetric — the layer-1 sibling of RagMetric.

Retrieval metrics score a ranked chunk list and never look at a generated
answer. Narrowing the signature from RagMetric's (case, result, tracer) to
(case, retrieved) makes "this metric costs no LLM call" a property a type
checker can verify, rather than a convention a future edit can quietly break.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from evaluation.harness.metric import MetricResult

if TYPE_CHECKING:
    from harness.core.rag.document import ScoredChunk

    from evaluation.rag.dataset import RagEvalCase


class RetrievalMetric(Protocol):
    @property
    def name(self) -> str: ...

    async def score(
        self, case: "RagEvalCase", retrieved: list["ScoredChunk"]
    ) -> MetricResult: ...
