from __future__ import annotations

import pytest

from evaluation.rag.dataset import RagEvalCase, RagExpected
from evaluation.rag.metrics.answer_relevancy import AnswerRelevancyMetric
from evaluation.rag.metrics.faithfulness import FaithfulnessMetric
from harness.adapters.llm.fake import FakeLLMClient
from harness.core.rag.document import Chunk, ScoredChunk
from harness.core.rag.serve import RagResult
from harness.core.types import LLMResponse, ToolCall
from harness.observability.tracer import TraceCollector


@pytest.mark.asyncio
async def test_faithfulness_passes_when_judge_says_grounded():
    judge = FakeLLMClient(
        [
            LLMResponse(
                text="",
                tool_calls=(
                    ToolCall(
                        id="c1",
                        name="score_faithfulness",
                        arguments={"grounded": True, "reasoning": "fully supported"},
                    ),
                ),
            ),
        ]
    )
    metric = FaithfulnessMetric(judge=judge)
    case = RagEvalCase(id="q1", query="q", expected=RagExpected())
    result = RagResult(
        answer="X causes Y",
        retrieved=(
            ScoredChunk(
                chunk=Chunk(
                    chunk_id="c",
                    document_id="d",
                    collection="papers",
                    text="X causes Y per the study",
                    section_path=(),
                ),
                score=1.0,
            ),
        ),
        assembled_prompt="p",
    )

    mr = await metric.score(case, result, TraceCollector())

    assert mr.passed
    assert mr.score == 1.0
    assert mr.reason == "fully supported"


@pytest.mark.asyncio
async def test_faithfulness_fails_when_judge_says_ungrounded():
    judge = FakeLLMClient(
        [
            LLMResponse(
                text="",
                tool_calls=(
                    ToolCall(
                        id="c1",
                        name="score_faithfulness",
                        arguments={
                            "grounded": False,
                            "reasoning": "invents a claim not in context",
                        },
                    ),
                ),
            ),
        ]
    )
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
    judge = FakeLLMClient(
        [
            LLMResponse(
                text="",
                tool_calls=(
                    ToolCall(
                        id="c1",
                        name="score_answer_relevancy",
                        arguments={
                            "relevant": True,
                            "reasoning": "directly answers the question",
                        },
                    ),
                ),
            ),
        ]
    )
    metric = AnswerRelevancyMetric(judge=judge)
    case = RagEvalCase(id="q1", query="what mechanism is used?", expected=RagExpected())
    result = RagResult(answer="Self-attention.", retrieved=(), assembled_prompt="p")

    mr = await metric.score(case, result, TraceCollector())

    assert mr.passed
    assert mr.score == 1.0


@pytest.mark.asyncio
async def test_answer_relevancy_fails_when_judge_says_off_topic():
    judge = FakeLLMClient(
        [
            LLMResponse(
                text="",
                tool_calls=(
                    ToolCall(
                        id="c1",
                        name="score_answer_relevancy",
                        arguments={
                            "relevant": False,
                            "reasoning": "answers a different question",
                        },
                    ),
                ),
            ),
        ]
    )
    metric = AnswerRelevancyMetric(judge=judge)
    case = RagEvalCase(id="q1", query="what mechanism is used?", expected=RagExpected())
    result = RagResult(
        answer="The paper was published in 2017.", retrieved=(), assembled_prompt="p"
    )

    mr = await metric.score(case, result, TraceCollector())

    assert not mr.passed
