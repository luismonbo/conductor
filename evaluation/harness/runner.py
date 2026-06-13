"""EvalRunner: runs cases through the agent and scores them with metrics."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from harness.core.types import AgentResult, AgentState, Message, Role
from harness.observability.tracer import TraceCollector

from evaluation.harness.dataset import Dataset, EvalCase
from evaluation.harness.metric import Metric
from evaluation.harness.report import CaseReport, EvalReport


@dataclass
class AgentRunResult:
    """Thin wrapper so metrics don't import AgentResult directly."""
    output: str
    tool_names_called: list[str] = field(default_factory=list)
    tool_args_by_name: dict[str, dict] = field(default_factory=dict)


def _extract_run_result(result: AgentResult, tracer: TraceCollector) -> AgentRunResult:
    tool_names: list[str] = []
    tool_args: dict[str, dict] = {}
    for _, event, data in tracer.events:
        if event == "tool_result":
            name = data.get("name", "")
            tool_names.append(name)
        elif event == "llm_response":
            for tc in data.get("tool_calls", []):
                tool_args[tc["name"]] = tc.get("arguments", {})
    return AgentRunResult(
        output=result.output,
        tool_names_called=tool_names,
        tool_args_by_name=tool_args,
    )


class EvalRunner:
    def __init__(self, agent_factory) -> None:
        # agent_factory() returns a fresh agent for each case
        self._factory = agent_factory

    def run(
        self,
        dataset: Dataset,
        metrics: list[Metric],
        dataset_name: str = "dataset",
    ) -> EvalReport:
        return asyncio.run(self.run_async(dataset, metrics, dataset_name))

    async def run_async(
        self,
        dataset: Dataset,
        metrics: list[Metric],
        dataset_name: str = "dataset",
    ) -> EvalReport:
        report = EvalReport(
            run_id=EvalReport.make_run_id(),
            dataset=dataset_name,
        )
        for case in dataset.cases:
            case_report = await self._run_case(case, metrics)
            report.cases.append(case_report)
        return report

    async def _run_case(self, case: EvalCase, metrics: list[Metric]) -> CaseReport:
        tracer = TraceCollector()
        try:
            result = self._factory(tracer, case.memory_seed)
            agent = await result if asyncio.iscoroutine(result) else result
            state = AgentState(messages=[Message(Role.USER, case.input)])
            agent_result: AgentResult = await agent.run(state)
            run_result = _extract_run_result(agent_result, tracer)
        except Exception as exc:
            return CaseReport(
                case_id=case.id,
                input=case.input,
                passed=False,
                error=str(exc),
            )

        metric_results = [m.score(case, run_result, tracer) for m in metrics]
        passed = all(mr.passed for mr in metric_results)
        return CaseReport(
            case_id=case.id,
            input=case.input,
            passed=passed,
            metric_results=metric_results,
        )
