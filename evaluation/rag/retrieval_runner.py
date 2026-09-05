"""RetrievalRunner — layer 1: cases -> Retriever -> RetrievalMetric -> EvalReport.

Takes a Retriever, not a RagPipeline, so it cannot invoke an LLM. That makes
retrieval sweeps free, deterministic and seconds long, which is what makes
tuning hybrid weights, rerank on/off, top-k and overfetch tractable at all.

RagPipeline already composes Retriever as a separate collaborator; this uses
that seam rather than adding a mode flag to RagRunner, so "does this run cost
money" stays a structural property rather than a runtime one.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import asdict, dataclass

from evaluation.harness.report import CaseReport, EvalReport
from evaluation.rag.dataset import RagCorpus, RagDataset, RagEvalCase
from evaluation.rag.retrieval_metric import RetrievalMetric
from harness.core.rag.ports import Retriever


class CorpusMismatchError(RuntimeError):
    """The live index no longer matches what the dataset was labelled against."""


@dataclass(frozen=True)
class RetrievalConfig:
    """The experiment manifest: every knob that changes what retrieval returns.

    Recorded in the report because a score you cannot attribute to a
    configuration is as useless as no score. k lives here rather than in the
    runner's private state because recall@k, MRR and nDCG@k are not comparable
    across different k — it was always part of benchmark identity.

    CAUTION: only `k` and `collection` are actually passed to `Retriever.retrieve()`
    by RetrievalRunner. `per_document_k` and `overfetch` take effect only through
    whichever concrete Retriever object was built by `build_retriever(settings, ...)`
    *before* this config was constructed — build_retriever reads them from Settings,
    not from this dataclass. Setting them here without also rebuilding a matching
    retriever records a manifest that does not describe what actually ran. Whoever
    writes an SP2 sweep must keep the two in sync explicitly.

    SP2 adds: dense_weight, sparse_weight, rerank_model, rerank_top_n,
    contextual_retrieval.
    """

    k: int = 5
    collection: str | None = None
    per_document_k: int | None = None
    overfetch: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def verify_corpus(expected: RagCorpus, actual: dict[str, int]) -> None:
    """Fail loudly when a re-ingest has invalidated the dataset's labels.

    SP1 changed the parser, normalizer and document_id scheme, silently killing
    every label in papers_v1.json — nothing noticed until someone went looking.
    A dataset that declares no fingerprint is not checked.
    """
    if not expected.documents:
        return

    problems: list[str] = []
    for document_id, count in expected.documents.items():
        if document_id not in actual:
            problems.append(f"  {document_id}: missing from the live index")
        elif actual[document_id] != count:
            problems.append(
                f"  {document_id}: labelled against {count} chunks, index has {actual[document_id]}"
            )
    if problems:
        raise CorpusMismatchError(
            "The live index does not match what this dataset was labelled against "
            f"(verified {expected.verified_at or 'unknown date'}):\n"
            + "\n".join(problems)
            + "\n\nLabels are stale. Re-verify them before trusting any score."
        )


class RetrievalRunner:
    def __init__(
        self, retriever: Retriever, config: RetrievalConfig | None = None
    ) -> None:
        self._retriever = retriever
        self._config = config or RetrievalConfig()

    def run(
        self,
        dataset: RagDataset,
        metrics: list[RetrievalMetric],
        dataset_name: str = "dataset",
    ) -> EvalReport:
        return asyncio.run(self.run_async(dataset, metrics, dataset_name))

    async def run_async(
        self,
        dataset: RagDataset,
        metrics: list[RetrievalMetric],
        dataset_name: str = "dataset",
    ) -> EvalReport:
        report = EvalReport(
            run_id=EvalReport.make_run_id(),
            dataset=dataset_name,
            config=self._config.to_dict(),
        )
        for case in dataset.cases:
            report.cases.append(await self._run_case(case, metrics))
        return report

    async def _run_case(
        self, case: RagEvalCase, metrics: list[RetrievalMetric]
    ) -> CaseReport:
        started = time.perf_counter()
        try:
            retrieved = await self._retriever.retrieve(
                case.query, self._config.k, self._config.collection
            )
        except Exception as exc:
            return CaseReport(
                case_id=case.id,
                input=case.query,
                passed=False,
                error=str(exc),
                query_type=case.query_type,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        latency_ms = (time.perf_counter() - started) * 1000

        metric_results = [await m.score(case, retrieved) for m in metrics]
        return CaseReport(
            case_id=case.id,
            input=case.query,
            passed=all(mr.passed for mr in metric_results),
            metric_results=metric_results,
            query_type=case.query_type,
            latency_ms=latency_ms,
        )
