from __future__ import annotations

import pytest

from evaluation.rag.dataset import RagEvalCase, RagExpected
from evaluation.rag.metrics.ndcg_at_k import NDCGAtKMetric
from harness.core.rag.document import Chunk, ScoredChunk


def _sc(chunk_id: str) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id="d1",
            collection="papers",
            text="x",
            section_path=(),
        ),
        score=1.0,
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
async def test_perfect_ranking_scores_exactly_one():
    case = _case({"a": 3, "b": 2})

    mr = await NDCGAtKMetric(k=3).score(case, [_sc("a"), _sc("b"), _sc("x")])

    assert mr.score == pytest.approx(1.0)
    assert mr.granularity == "chunk"


@pytest.mark.asyncio
async def test_hand_computed_value():
    # labels {a:3, b:2}; retrieved [a, x, b]; exponential gain 2**g - 1
    #   DCG  = 7/log2(2) + 0/log2(3) + 3/log2(4) = 7 + 0 + 1.5      = 8.500000
    #   IDCG = 7/log2(2) + 3/log2(3)             = 7 + 1.892789     = 8.892789
    #   nDCG = 8.500000 / 8.892789                                  = 0.955831
    case = _case({"a": 3, "b": 2})

    mr = await NDCGAtKMetric(k=3).score(case, [_sc("a"), _sc("x"), _sc("b")])

    assert mr.score == pytest.approx(0.955831, abs=1e-6)


@pytest.mark.asyncio
async def test_reversed_ranking_scores_below_perfect():
    case = _case({"a": 3, "b": 2})

    perfect = await NDCGAtKMetric(k=2).score(case, [_sc("a"), _sc("b")])
    reversed_ = await NDCGAtKMetric(k=2).score(case, [_sc("b"), _sc("a")])

    assert reversed_.score < perfect.score


@pytest.mark.asyncio
async def test_idcg_uses_full_label_set_not_retrieved_grades():
    # The regression guard. 'b' is labelled but never retrieved. If IDCG were
    # computed from retrieved grades only, this would score 1.0 — the retriever
    # would be graded against its own failure.
    case = _case({"a": 3, "b": 3})

    mr = await NDCGAtKMetric(k=2).score(case, [_sc("a"), _sc("x")])

    assert mr.score < 1.0
    assert mr.score == pytest.approx(7 / (7 + 7 / 1.584962500721156), abs=1e-6)


@pytest.mark.asyncio
async def test_nothing_relevant_retrieved_scores_zero():
    case = _case({"a": 3})

    mr = await NDCGAtKMetric(k=3).score(case, [_sc("x"), _sc("y")])

    assert mr.score == pytest.approx(0.0)
    assert mr.passed is False


@pytest.mark.asyncio
async def test_truncates_to_k():
    # 'a' sits at rank 3, outside k=2, so it contributes nothing
    case = _case({"a": 3})

    mr = await NDCGAtKMetric(k=2).score(case, [_sc("x"), _sc("y"), _sc("a")])

    assert mr.score == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_skips_cases_without_graded_chunks():
    for case in (_case(), _case(docs=["d1"])):
        mr = await NDCGAtKMetric(k=3).score(case, [_sc("a")])

        assert mr.skipped is True
        assert mr.granularity is None


@pytest.mark.asyncio
async def test_all_grades_zero_is_skipped_not_divide_by_zero():
    mr = await NDCGAtKMetric(k=3).score(_case({"a": 0, "b": 0}), [_sc("a")])

    assert mr.skipped is True
