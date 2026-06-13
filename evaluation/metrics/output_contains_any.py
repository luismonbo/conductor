"""OutputContainsAnyMetric: at least one expected substring appears in the output.

OR-semantics: passes if any listed string is present (case-insensitive).
Skips cases where output_contains_any is empty.
Use for word/digit variants (e.g. "seven" or "7") where the model may
legitimately produce either form.
"""
from __future__ import annotations

from harness.observability.tracer import TraceCollector

from evaluation.harness.dataset import EvalCase
from evaluation.harness.metric import MetricResult
from evaluation.harness.runner import AgentRunResult


class OutputContainsAnyMetric:
    @property
    def name(self) -> str:
        return "output_contains_any"

    def score(
        self,
        case: EvalCase,
        result: AgentRunResult,
        tracer: TraceCollector,
    ) -> MetricResult:
        expected = case.expected.output_contains_any
        if not expected:
            return MetricResult(
                name=self.name,
                passed=True,
                score=1.0,
                reason="skipped — no output_contains_any expectation",
            )

        output_lower = result.output.lower()
        matched = [s for s in expected if s.lower() in output_lower]

        if matched:
            return MetricResult(
                name=self.name,
                passed=True,
                score=1.0,
                reason=f"found: {matched[0]!r}",
            )

        return MetricResult(
            name=self.name,
            passed=False,
            score=0.0,
            reason=f"output missing all of: {expected}",
        )
