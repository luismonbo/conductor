"""ArgSchemaMetric: were the expected arg keys present?

Checks key presence only, not values, because LLM phrasing of argument
values varies legitimately. Skips cases where tool_args is not specified.
"""
from __future__ import annotations

from harness.observability.tracer import TraceCollector

from evaluation.harness.dataset import EvalCase
from evaluation.harness.metric import MetricResult
from evaluation.harness.runner import AgentRunResult


class ArgSchemaMetric:
    @property
    def name(self) -> str:
        return "arg_schema"

    def score(
        self,
        case: EvalCase,
        result: AgentRunResult,
        tracer: TraceCollector,
    ) -> MetricResult:
        expected_args = case.expected.tool_args
        if expected_args is None:
            return MetricResult(
                name=self.name,
                passed=True,
                score=1.0,
                reason="skipped — no tool_args expectation",
            )

        tool_name = case.expected.tool_call.name if case.expected.tool_call else None
        if tool_name is None:
            return MetricResult(
                name=self.name,
                passed=True,
                score=1.0,
                reason="skipped — no tool_call expectation to look up args for",
            )

        actual_args = result.tool_args_by_name.get(tool_name)
        if actual_args is None:
            return MetricResult(
                name=self.name,
                passed=False,
                score=0.0,
                reason=f"'{tool_name}' was not called, cannot check args",
            )

        missing = [k for k in expected_args if k not in actual_args]
        if missing:
            return MetricResult(
                name=self.name,
                passed=False,
                score=0.0,
                reason=f"missing expected arg keys: {missing}",
            )

        return MetricResult(
            name=self.name,
            passed=True,
            score=1.0,
            reason=f"all expected keys present: {list(expected_args)}",
        )
