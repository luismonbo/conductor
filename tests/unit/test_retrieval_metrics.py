from __future__ import annotations

import pytest

from evaluation.rag.dataset import RagEvalCase, RagExpected
from evaluation.rag.metrics.mrr import MRRMetric
from evaluation.rag.metrics.recall_at_k import RecallAtKMetric
from harness.core.rag.document import Chunk, ScoredChunk


def _sc(chunk_id: str, document_id: str = "d1", score: float = 1.0) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id=document_id,
            collection="papers",
            text="x",
            section_path=(),
        ),
        score=score,
    )


def _case(
    graded: dict[str, int] | None = None, docs: list[str] | None = None
) -> RagEvalCase:
    return RagEvalCase(
        id="c",
        query="q",
        expected=RagExpected(
            graded_chunks=graded or {}, relevant_document_ids=docs or []
        ),
    )


@pytest.mark.asyncio
async def test_recall_uses_grade_two_threshold():
    # a:3 and b:2 are relevant; c:1 is not, so the denominator is 2 not 3
    case = _case({"a": 3, "b": 2, "c": 1})

    mr = await RecallAtKMetric().score(case, [_sc("a"), _sc("c")])

    assert mr.score == pytest.approx(0.5)
    assert mr.granularity == "chunk"
    assert mr.skipped is False


@pytest.mark.asyncio
async def test_recall_falls_back_to_documents_when_no_graded_chunks():
    case = _case(docs=["d1", "d2"])

    mr = await RecallAtKMetric().score(case, [_sc("x", document_id="d1")])

    assert mr.score == pytest.approx(0.5)
    assert mr.granularity == "document"


@pytest.mark.asyncio
async def test_recall_marks_empty_expectations_skipped():
    mr = await RecallAtKMetric().score(_case(), [_sc("a")])

    assert mr.skipped is True
    assert mr.granularity is None


@pytest.mark.asyncio
async def test_mrr_scores_reciprocal_of_first_relevant_rank():
    case = _case({"a": 3})

    mr = await MRRMetric().score(case, [_sc("x"), _sc("y"), _sc("a")])

    assert mr.score == pytest.approx(1 / 3)
    assert mr.granularity == "chunk"


@pytest.mark.asyncio
async def test_mrr_falls_back_to_documents_when_no_graded_chunks():
    # First chunk is from an unlabelled document (d1); the labelled document
    # (d2) only shows up at rank 2, so a correct document-id match must score
    # 1/2 here -- not 1.0, which chunk-id matching or presence-only checks
    # would both produce by accident (neither "x" nor "y" is a labelled id).
    case = _case(docs=["d2"])

    mr = await MRRMetric().score(
        case, [_sc("x", document_id="d1"), _sc("y", document_id="d2")]
    )

    assert mr.score == pytest.approx(0.5)
    assert mr.granularity == "document"


@pytest.mark.asyncio
async def test_mrr_ignores_grade_one_chunks():
    # 'a' is graded 1, below the relevance threshold, so it is not a hit
    case = _case({"a": 1, "b": 3})

    mr = await MRRMetric().score(case, [_sc("a"), _sc("b")])

    assert mr.score == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_mrr_marks_empty_expectations_skipped():
    mr = await MRRMetric().score(_case(), [_sc("a")])

    assert mr.skipped is True
