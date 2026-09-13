from __future__ import annotations

import pytest

from evaluation.rag.dataset import RagDataset, RagEvalCase, RagExpected
from evaluation.rag.metrics.answer_relevancy import AnswerRelevancyMetric
from evaluation.rag.metrics.faithfulness import FaithfulnessMetric
from evaluation.rag.runner import RagRunner
from harness.adapters.embedding.fake import FakeEmbedder
from harness.adapters.llm.fake import FakeLLMClient
from harness.adapters.vectorstore.in_memory import InMemoryVectorStore
from harness.core.rag.document import Chunk
from harness.core.rag.serve import RagPipeline, Retriever
from harness.core.types import LLMResponse, ToolCall


def test_runner_produces_baseline_report_on_empty_vector_store():
    # Sync on purpose: exercises the blocking run() wrapper, which calls
    # asyncio.run() and so cannot be invoked from inside a running loop.
    store = InMemoryVectorStore()  # nothing ingested
    embedder = FakeEmbedder(dimension=4)
    llm = FakeLLMClient(
        [LLMResponse(text="I don't have enough information to answer that.")]
    )

    def pipeline_factory(tracer):
        return RagPipeline(
            retriever=Retriever(embedder=embedder, vector_store=store),
            llm=llm,
            tracer=tracer,
        )

    case = RagEvalCase(
        id="case1", query="what is the paper about?", expected=RagExpected()
    )
    dataset = RagDataset([case])
    runner = RagRunner(pipeline_factory)

    report = runner.run(dataset, metrics=[], dataset_name="test")

    assert report.total == 1
    assert report.cases[0].error is None
    assert report.cases[0].passed  # vacuously true over zero metrics


@pytest.mark.asyncio
async def test_runner_combines_multiple_llm_judge_metrics():
    store = InMemoryVectorStore()
    embedder = FakeEmbedder(dimension=4)
    chunk = Chunk(
        chunk_id="c1",
        document_id="d1",
        collection="papers",
        text="the model uses self-attention",
        section_path=(),
    )
    [vec] = await embedder.embed([chunk.text])
    await store.upsert([chunk], [vec])
    llm = FakeLLMClient([LLMResponse(text="The model uses self-attention.")])
    judge = FakeLLMClient(
        [
            LLMResponse(
                text="",
                tool_calls=(
                    ToolCall(
                        id="c1",
                        name="score_faithfulness",
                        arguments={"grounded": True, "reasoning": "matches context"},
                    ),
                ),
            ),
            LLMResponse(
                text="",
                tool_calls=(
                    ToolCall(
                        id="c2",
                        name="score_answer_relevancy",
                        arguments={"relevant": True, "reasoning": "answers the question"},
                    ),
                ),
            ),
        ]
    )

    def pipeline_factory(tracer):
        return RagPipeline(
            retriever=Retriever(embedder=embedder, vector_store=store),
            llm=llm,
            tracer=tracer,
        )

    case = RagEvalCase(
        id="case1",
        query="what mechanism is used?",
        expected=RagExpected(graded_chunks={"c1": 3}),
    )
    dataset = RagDataset([case])
    runner = RagRunner(pipeline_factory)
    metrics = [
        FaithfulnessMetric(judge=judge),
        AnswerRelevancyMetric(judge=judge),
    ]

    # await run_async(): this test is async (it seeds the store with await),
    # and run() would call asyncio.run() from inside the running loop.
    report = await runner.run_async(dataset, metrics, dataset_name="test")

    assert report.cases[0].passed
    assert {mr.name for mr in report.cases[0].metric_results} == {
        "faithfulness",
        "answer_relevancy",
    }


@pytest.mark.asyncio
async def test_runner_captures_system_and_eval_tokens_separately():
    store = InMemoryVectorStore()
    embedder = FakeEmbedder(dimension=4)
    chunk = Chunk(
        chunk_id="c1",
        document_id="d1",
        collection="papers",
        text="the model uses self-attention",
        section_path=(),
    )
    [vec] = await embedder.embed([chunk.text])
    await store.upsert([chunk], [vec])
    llm = FakeLLMClient(
        [
            LLMResponse(
                text="The model uses self-attention.",
                usage={"input_tokens": 100, "output_tokens": 10},
            )
        ]
    )
    judge = FakeLLMClient(
        [
            LLMResponse(
                text="",
                tool_calls=(
                    ToolCall(
                        id="c1",
                        name="score_faithfulness",
                        arguments={"grounded": True, "reasoning": "matches context"},
                    ),
                ),
                usage={"input_tokens": 50, "output_tokens": 5},
            ),
            LLMResponse(
                text="",
                tool_calls=(
                    ToolCall(
                        id="c2",
                        name="score_answer_relevancy",
                        arguments={"relevant": True, "reasoning": "answers the question"},
                    ),
                ),
                usage={"input_tokens": 40, "output_tokens": 4},
            ),
        ]
    )

    def pipeline_factory(tracer):
        return RagPipeline(
            retriever=Retriever(embedder=embedder, vector_store=store),
            llm=llm,
            tracer=tracer,
        )

    case = RagEvalCase(
        id="case1",
        query="what mechanism is used?",
        expected=RagExpected(graded_chunks={"c1": 3}),
    )
    dataset = RagDataset([case])
    runner = RagRunner(pipeline_factory)
    metrics = [
        FaithfulnessMetric(judge=judge),
        AnswerRelevancyMetric(judge=judge),
    ]

    report = await runner.run_async(dataset, metrics, dataset_name="test")

    # Production cost (the pipeline's own answer call) stays separate from
    # what it cost to *evaluate* (both judges' calls, summed).
    assert report.cases[0].system_tokens == {"input_tokens": 100, "output_tokens": 10}
    assert report.cases[0].eval_tokens == {"input_tokens": 90, "output_tokens": 9}


def test_runner_records_error_without_crashing_the_whole_run():
    def broken_factory(tracer):
        raise RuntimeError("pipeline construction failed")

    case = RagEvalCase(id="case1", query="q", expected=RagExpected())
    dataset = RagDataset([case])
    runner = RagRunner(broken_factory)

    report = runner.run(dataset, metrics=[], dataset_name="test")

    assert report.total == 1
    assert report.cases[0].passed is False
    assert "pipeline construction failed" in report.cases[0].error
