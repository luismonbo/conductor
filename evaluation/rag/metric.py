"""RagMetric protocol — sibling to evaluation/harness/metric.py's Metric, not
a reuse of it: score() is async (LLM-judge metrics await a judge call, which
the synchronous Metric signature can't express) and the case/result types
differ (RagEvalCase/RagResult vs. EvalCase/AgentRunResult, which Protocol
structural typing checks). MetricResult IS reused directly."""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from evaluation.harness.metric import MetricResult

if TYPE_CHECKING:
    from harness.core.rag.serve import RagResult
    from harness.observability.tracer import TraceCollector

    from evaluation.rag.dataset import RagEvalCase


class RagMetric(Protocol):
    @property
    def name(self) -> str: ...

    async def score(
        self, case: "RagEvalCase", result: "RagResult", tracer: "TraceCollector"
    ) -> MetricResult: ...