"""ToolCallMetric: was the expected tool invoked at all?"""
from __future__ import annotations

from harness.observability.tracer import TraceCollector

from evaluation.harness.dataset import EvalCase
from evaluation.harness.metric import MetricResult
from evaluation.harness.runner import AgentRunResult


class ToolCallMetric:
    @property
    def name(self) -> str:
        return "tool_call"

    def score(
        self,
        case: EvalCase,
        result: AgentRunResult,
        tracer: TraceCollector,
    ) -> MetricResult:
        expected = case.expected.tool_call
        if expected is None:
            return MetricResult(
                name=self.name,
                passed=True,
                score=1.0,
                reason="skipped — no tool_call expectation",
            )

        called = result.tool_names_called
        if not called:
            return MetricResult(
                name=self.name,
                passed=False,
                score=0.0,
                reason=f"expected '{expected.name}', but no tools were called",
            )

        if expected.name in called:
            return MetricResult(
                name=self.name,
                passed=True,
                score=1.0,
                reason=f"'{expected.name}' was called",
            )

        return MetricResult(
            name=self.name,
            passed=False,
            score=0.0,
            reason=f"expected '{expected.name}', got {called}",
        )
