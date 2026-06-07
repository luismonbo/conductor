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


class Metric(Protocol):
    @property
    def name(self) -> str: ...

    def score(
        self,
        case: "EvalCase",
        result: "AgentRunResult",
        tracer: "TraceCollector",
    ) -> MetricResult: ...