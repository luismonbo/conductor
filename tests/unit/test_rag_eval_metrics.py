from __future__ import annotations

import pytest

from evaluation.rag.dataset import RagEvalCase, RagExpected
from evaluation.rag.metrics.mrr import MRRMetric
from evaluation.rag.metrics.recall_at_k import RecallAtKMetric
from harness.core.rag.document import Chunk, ScoredChunk
from harness.core.rag.serve import RagResult
from harness.observability.tracer import TraceCollector


def _result(chunk_ids: list[str]) -> RagResult:
    retrieved = tuple(
        ScoredChunk(
            chunk=Chunk(
                chunk_id=cid, document_id=cid.split(":")[0], collection="papers",
                text="t", section_path=(),
            ),
            score=1.0,
        )
        for cid in chunk_ids
    )
    return RagResult(answer="an answer", retrieved=retrieved, assembled_prompt="prompt")


@pytest.mark.asyncio
async def test_recall_at_k_passes_when_expected_chunk_in_results():
    case = RagEvalCase(id="q1", query="q", expected=RagExpected(relevant_chunk_ids=["doc1:2"]))
    result = _result(["doc1:0", "doc1:2"])

    mr = await RecallAtKMetric().score(case, result, TraceCollector())

    assert mr.passed
    assert mr.score == 1.0


@pytest.mark.asyncio
async def test_recall_at_k_fails_when_expected_chunk_absent():
    case = RagEvalCase(id="q1", query="q", expected=RagExpected(relevant_chunk_ids=["doc9:0"]))
    result = _result(["doc1:0", "doc1:2"])

    mr = await RecallAtKMetric().score(case, result, TraceCollector())

    assert not mr.passed
    assert mr.score == 0.0


@pytest.mark.asyncio
async def test_recall_at_k_scores_the_fraction_of_relevant_chunks_found():
    case = RagEvalCase(
        id="q1", query="q",
        expected=RagExpected(relevant_chunk_ids=["doc1:0", "doc1:1", "doc1:2", "doc1:3"]),
    )
    result = _result(["doc1:0", "doc9:9"])

    mr = await RecallAtKMetric().score(case, result, TraceCollector())

    assert mr.passed  # found something, so the case is not a retrieval failure
    assert mr.score == pytest.approx(0.25)


@pytest.mark.asyncio
async def test_recall_at_k_falls_back_to_document_ids_when_no_chunk_ids():
    case = RagEvalCase(
        id="q1", query="q", expected=RagExpected(relevant_document_ids=["doc1", "doc2"])
    )
    result = _result(["doc1:0"])

    mr = await RecallAtKMetric().score(case, result, TraceCollector())

    assert mr.passed
    assert mr.score == pytest.approx(0.5)
    assert "document" in mr.reason


@pytest.mark.asyncio
async def test_recall_at_k_skips_when_no_expectation_set():
    case = RagEvalCase(id="q1", query="q", expected=RagExpected())
    result = _result([])

    mr = await RecallAtKMetric().score(case, result, TraceCollector())

    assert mr.passed
    assert mr.reason == "skipped"


@pytest.mark.asyncio
async def test_mrr_scores_reciprocal_of_first_relevant_rank():
    case = RagEvalCase(id="q1", query="q", expected=RagExpected(relevant_chunk_ids=["doc1:2"]))
    result = _result(["doc1:0", "doc1:1", "doc1:2"])

    mr = await MRRMetric().score(case, result, TraceCollector())

    assert mr.passed
    assert mr.score == pytest.approx(1 / 3)


@pytest.mark.asyncio
async def test_mrr_uses_the_same_granularity_as_recall():
    """A case labelled with chunk ids must be scored on chunks by BOTH metrics.

    Falling back to a document-level match here would report 'first relevant
    hit at rank 1' for a retrieval that surfaced no labelled chunk at all,
    while recall_at_k reported 0.00 for the very same result — two metrics
    telling contradictory stories about one retrieval.
    """
    case = RagEvalCase(
        id="q1", query="q",
        expected=RagExpected(
            relevant_chunk_ids=["doc1:7", "doc2:3"],
            relevant_document_ids=["doc1", "doc2"],
        ),
    )
    # Chunks from the right documents, but none of the labelled chunks.
    result = _result(["doc1:0", "doc2:1"])

    recall = await RecallAtKMetric().score(case, result, TraceCollector())
    mrr = await MRRMetric().score(case, result, TraceCollector())

    assert recall.score == 0.0
    assert mrr.score == 0.0
    assert not mrr.passed


@pytest.mark.asyncio
async def test_mrr_falls_back_to_documents_when_only_documents_are_labelled():
    case = RagEvalCase(id="q1", query="q", expected=RagExpected(relevant_document_ids=["doc2"]))
    result = _result(["doc1:0", "doc2:5"])

    mr = await MRRMetric().score(case, result, TraceCollector())

    assert mr.passed
    assert mr.score == pytest.approx(0.5)
    assert "document" in mr.reason


@pytest.mark.asyncio
async def test_mrr_zero_when_not_found():
    case = RagEvalCase(id="q1", query="q", expected=RagExpected(relevant_chunk_ids=["doc9:0"]))
    result = _result(["doc1:0"])

    mr = await MRRMetric().score(case, result, TraceCollector())

    assert not mr.passed
    assert mr.score == 0.0


from evaluation.rag.metrics.answer_relevancy import AnswerRelevancyMetric  # noqa: E402
from evaluation.rag.metrics.faithfulness import FaithfulnessMetric  # noqa: E402
from harness.adapters.llm.fake import FakeLLMClient  # noqa: E402
from harness.core.types import LLMResponse, ToolCall  # noqa: E402


@pytest.mark.asyncio
async def test_faithfulness_passes_when_judge_says_grounded():
    judge = FakeLLMClient([
        LLMResponse(text="", tool_calls=(
            ToolCall(id="c1", name="score_faithfulness",
                     arguments={"grounded": True, "reasoning": "fully supported"}),
        )),
    ])
    metric = FaithfulnessMetric(judge=judge)
    case = RagEvalCase(id="q1", query="q", expected=RagExpected())
    result = RagResult(
        answer="X causes Y",
        retrieved=(ScoredChunk(
            chunk=Chunk(chunk_id="c", document_id="d", collection="papers",
                        text="X causes Y per the study", section_path=()),
            score=1.0,
        ),),
        assembled_prompt="p",
    )

    mr = await metric.score(case, result, TraceCollector())

    assert mr.passed
    assert mr.score == 1.0
    assert mr.reason == "fully supported"


@pytest.mark.asyncio
async def test_faithfulness_fails_when_judge_says_ungrounded():
    judge = FakeLLMClient([
        LLMResponse(text="", tool_calls=(
            ToolCall(id="c1", name="score_faithfulness",
                     arguments={"grounded": False, "reasoning": "invents a claim not in context"}),
        )),
    ])
    metric = FaithfulnessMetric(judge=judge)
    case = RagEvalCase(id="q1", query="q", expected=RagExpected())
    result = RagResult(answer="X causes Z", retrieved=(), assembled_prompt="p")

    mr = await metric.score(case, result, TraceCollector())

    assert not mr.passed


@pytest.mark.asyncio
async def test_faithfulness_fails_safe_when_judge_gives_no_structured_response():
    judge = FakeLLMClient([LLMResponse(text="I'm not sure.", tool_calls=())])
    metric = FaithfulnessMetric(judge=judge)
    case = RagEvalCase(id="q1", query="q", expected=RagExpected())
    result = RagResult(answer="something", retrieved=(), assembled_prompt="p")

    mr = await metric.score(case, result, TraceCollector())

    assert not mr.passed


@pytest.mark.asyncio
async def test_answer_relevancy_passes_when_judge_says_relevant():
    judge = FakeLLMClient([
        LLMResponse(text="", tool_calls=(
            ToolCall(id="c1", name="score_answer_relevancy",
                     arguments={"relevant": True, "reasoning": "directly answers the question"}),
        )),
    ])
    metric = AnswerRelevancyMetric(judge=judge)
    case = RagEvalCase(id="q1", query="what mechanism is used?", expected=RagExpected())
    result = RagResult(answer="Self-attention.", retrieved=(), assembled_prompt="p")

    mr = await metric.score(case, result, TraceCollector())

    assert mr.passed
    assert mr.score == 1.0


@pytest.mark.asyncio
async def test_answer_relevancy_fails_when_judge_says_off_topic():
    judge = FakeLLMClient([
        LLMResponse(text="", tool_calls=(
            ToolCall(id="c1", name="score_answer_relevancy",
                     arguments={"relevant": False, "reasoning": "answers a different question"}),
        )),
    ])
    metric = AnswerRelevancyMetric(judge=judge)
    case = RagEvalCase(id="q1", query="what mechanism is used?", expected=RagExpected())
    result = RagResult(answer="The paper was published in 2017.", retrieved=(), assembled_prompt="p")

    mr = await metric.score(case, result, TraceCollector())

    assert not mr.passed
