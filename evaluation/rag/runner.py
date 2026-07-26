"""RagRunner: cases -> RagPipeline -> RagMetric scoring -> EvalReport.
Mirrors evaluation/harness/runner.py's EvalRunner shape and reuses
EvalReport/CaseReport directly (both already generic — no changes needed).
The metric loop is awaited, unlike EvalRunner's, because RagMetric.score is
async (LLM-judge metrics need to await a judge call)."""
from __future__ import annotations

import asyncio

from harness.observability.tracer import TraceCollector

from evaluation.harness.report import CaseReport, EvalReport
from evaluation.rag.dataset import RagDataset, RagEvalCase
from evaluation.rag.metric import RagMetric


class RagRunner:
    def __init__(self, pipeline_factory) -> None:
        # pipeline_factory(tracer) returns a fresh RagPipeline, sync or async
        self._factory = pipeline_factory

    def run(
        self, dataset: RagDataset, metrics: list[RagMetric], dataset_name: str = "dataset"
    ) -> EvalReport:
        return asyncio.run(self.run_async(dataset, metrics, dataset_name))

    async def run_async(
        self, dataset: RagDataset, metrics: list[RagMetric], dataset_name: str = "dataset"
    ) -> EvalReport:
        report = EvalReport(run_id=EvalReport.make_run_id(), dataset=dataset_name)
        for case in dataset.cases:
            report.cases.append(await self._run_case(case, metrics))
        return report

    async def _run_case(self, case: RagEvalCase, metrics: list[RagMetric]) -> CaseReport:
        tracer = TraceCollector()
        try:
            pipeline_or_coro = self._factory(tracer)
            pipeline = (
                await pipeline_or_coro if asyncio.iscoroutine(pipeline_or_coro) else pipeline_or_coro
            )
            result = await pipeline.answer(case.query)
        except Exception as exc:
            return CaseReport(case_id=case.id, input=case.query, passed=False, error=str(exc))

        metric_results = [await m.score(case, result, tracer) for m in metrics]
        passed = all(mr.passed for mr in metric_results)
        return CaseReport(
            case_id=case.id, input=case.query, output=result.answer,
            passed=passed, metric_results=metric_results,
        )