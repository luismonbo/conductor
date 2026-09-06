"""Metric protocol and result type.

Any object implementing score(case, result, tracer) satisfies Metric.
Deterministic checks return score 0.0 or 1.0; future LLM-judge metrics
return a continuous value in [0, 1]. The runner only cares about this seam.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from harness.observability.tracer import TraceCollector

    from evaluation.harness.dataset import EvalCase
    from evaluation.harness.runner import AgentRunResult


@dataclass(frozen=True)
class MetricResult:
    name: str
    passed: bool
    score: float  # 0.0 or 1.0 for deterministic; continuous for LLM-judge
    reason: str
    # Which denominator the score was computed at ("chunk" | "document").
    # Means for the same metric at different granularities are NOT comparable
    # and must never be pooled — see EvalReport.aggregate().
    granularity: str | None = None
    # True when the case had nothing to score (e.g. a negative case with empty
    # expectations). Skipped results are excluded from aggregate means.
    skipped: bool = False


class Metric(Protocol):
    @property
    def name(self) -> str: ...

    def score(
        self,
        case: "EvalCase",
        result: "AgentRunResult",
        tracer: "TraceCollector",
    ) -> MetricResult: ...
