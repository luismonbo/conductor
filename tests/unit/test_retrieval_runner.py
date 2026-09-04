from __future__ import annotations

import pytest

from evaluation.harness.metric import MetricResult
from evaluation.rag.dataset import RagCorpus, RagDataset, RagEvalCase, RagExpected
from evaluation.rag.retrieval_runner import (
    CorpusMismatchError,
    RetrievalConfig,
    RetrievalRunner,
    verify_corpus,
)
from harness.core.rag.document import Chunk, ScoredChunk


class FakeRetriever:
    def __init__(self, chunk_ids: list[str]) -> None:
        self._chunk_ids = chunk_ids
        self.calls: list[tuple[str, int, str | None]] = []

    async def retrieve(self, query, k=5, collection=None):
        self.calls.append((query, k, collection))
        return [
            ScoredChunk(
                chunk=Chunk(chunk_id=cid, document_id="d1", collection="papers",
                            text="x", section_path=()),
                score=1.0,
            )
            for cid in self._chunk_ids[:k]
        ]


class CountingMetric:
    name = "counting"

    def __init__(self) -> None:
        self.seen: list[int] = []

    async def score(self, case, retrieved) -> MetricResult:
        self.seen.append(len(retrieved))
        return MetricResult(name=self.name, passed=True, score=1.0,
                            reason="ok", granularity="chunk")


def _dataset() -> RagDataset:
    return RagDataset(
        cases=[
            RagEvalCase(id="c1", query="q1",
                        expected=RagExpected(graded_chunks={"a": 3}), query_type="factual"),
            RagEvalCase(id="c2", query="q2",
                        expected=RagExpected(graded_chunks={"b": 2}), query_type="multi_hop"),
        ],
        corpus=RagCorpus(collection="papers", documents={"d1": 2}),
    )


@pytest.mark.asyncio
async def test_runs_every_case_and_scores_metrics():
    retriever = FakeRetriever(["a", "z"])
    metric = CountingMetric()

    report = await RetrievalRunner(retriever, RetrievalConfig(k=2)).run_async(
        _dataset(), [metric], dataset_name="d.json"
    )

    assert len(report.cases) == 2
    assert metric.seen == [2, 2]
    assert [c[1] for c in retriever.calls] == [2, 2]


@pytest.mark.asyncio
async def test_records_config_and_case_metadata():
    report = await RetrievalRunner(
        FakeRetriever(["a"]), RetrievalConfig(k=3, collection="papers", per_document_k=2)
    ).run_async(_dataset(), [CountingMetric()], dataset_name="d.json")

    assert report.config["k"] == 3
    assert report.config["per_document_k"] == 2
    assert report.cases[0].query_type == "factual"
    assert report.cases[1].query_type == "multi_hop"
    assert report.cases[0].latency_ms is not None
    assert report.cases[0].latency_ms >= 0


@pytest.mark.asyncio
async def test_retriever_failure_is_captured_not_raised():
    class Boom:
        async def retrieve(self, query, k=5, collection=None):
            raise RuntimeError("index offline")

    report = await RetrievalRunner(Boom(), RetrievalConfig()).run_async(
        _dataset(), [CountingMetric()], dataset_name="d.json"
    )

    assert report.cases[0].passed is False
    assert "index offline" in report.cases[0].error


def test_verify_corpus_passes_on_exact_match():
    verify_corpus(RagCorpus(collection="papers", documents={"d1": 43}), {"d1": 43})


def test_verify_corpus_raises_on_chunk_count_drift():
    with pytest.raises(CorpusMismatchError, match="d1"):
        verify_corpus(RagCorpus(collection="papers", documents={"d1": 43}), {"d1": 46})


def test_verify_corpus_raises_on_missing_document():
    with pytest.raises(CorpusMismatchError, match="missing"):
        verify_corpus(RagCorpus(collection="papers", documents={"d1": 43}), {"d2": 43})


def test_verify_corpus_is_a_noop_when_dataset_declares_no_fingerprint():
    verify_corpus(RagCorpus(), {"anything": 1})
