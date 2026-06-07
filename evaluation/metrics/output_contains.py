"""OutputContainsMetric: do all expected substrings appear in the output?

AND-semantics: every listed string must be present (case-insensitive).
Skips cases where output_contains is empty.
"""
from __future__ import annotations

from harness.observability.tracer import TraceCollector

from evaluation.harness.dataset import EvalCase
from evaluation.harness.metric import MetricResult
from evaluation.harness.runner import AgentRunResult


class OutputContainsMetric:
    @property
    def name(self) -> str:
        return "output_contains"

    def score(
        self,
        case: EvalCase,
        result: AgentRunResult,
        tracer: TraceCollector,
    ) -> MetricResult:
        expected = case.expected.output_contains
        if not expected:
            return MetricResult(
                name=self.name,
                passed=True,
                score=1.0,
                reason="skipped — no output_contains expectation",
            )

        output_lower = result.output.lower()
        missing = [s for s in expected if s.lower() not in output_lower]

        if missing:
            return MetricResult(
                name=self.name,
                passed=False,
                score=0.0,
                reason=f"output missing: {missing}",
            )

        return MetricResult(
            name=self.name,
            passed=True,
            score=1.0,
            reason=f"all substrings found: {expected}",
        )
