"""NoToolCallMetric: asserts the agent called no tools."""
from __future__ import annotations

from harness.observability.tracer import TraceCollector

from evaluation.harness.dataset import EvalCase
from evaluation.harness.metric import MetricResult
from evaluation.harness.runner import AgentRunResult


class NoToolCallMetric:
    @property
    def name(self) -> str:
        return "no_tool_call"

    def score(
        self,
        case: EvalCase,
        result: AgentRunResult,
        tracer: TraceCollector,
    ) -> MetricResult:
        if not case.expected.no_tool_call:
            return MetricResult(
                name=self.name,
                passed=True,
                score=1.0,
                reason="skipped — no_tool_call not asserted",
            )

        if not result.tool_names_called:
            return MetricResult(
                name=self.name,
                passed=True,
                score=1.0,
                reason="no tools called",
            )

        return MetricResult(
            name=self.name,
            passed=False,
            score=0.0,
            reason=f"expected no tool calls, but got {result.tool_names_called}",
        )
