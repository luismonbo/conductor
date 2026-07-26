from __future__ import annotations

import pytest

from harness.adapters.embedding.fake import FakeEmbedder
from harness.adapters.llm.fake import FakeLLMClient
from harness.adapters.vectorstore.in_memory import InMemoryVectorStore
from harness.core.rag.document import Chunk
from harness.core.rag.serve import RagPipeline, Retriever
from harness.core.types import LLMResponse
from harness.observability.tracer import TraceCollector


@pytest.mark.asyncio
async def test_answer_grounds_prompt_in_retrieved_chunks_and_returns_generated_text():
    store = InMemoryVectorStore()
    embedder = FakeEmbedder(dimension=4)
    chunk = Chunk(
        chunk_id="c1", document_id="d1", collection="papers",
        text="the model uses self-attention", section_path=("Method",),
        source_path="papers/attn.pdf",
    )
    [vec] = await embedder.embed([chunk.text])
    await store.upsert([chunk], [vec])
    retriever = Retriever(embedder=embedder, vector_store=store)
    llm = FakeLLMClient([LLMResponse(text="The model uses self-attention.")])

    pipeline = RagPipeline(retriever=retriever, llm=llm)
    result = await pipeline.answer("what mechanism does the model use?", k=3)

    assert result.answer == "The model uses self-attention."
    assert len(result.retrieved) == 1
    assert "self-attention" in result.assembled_prompt
    assert "Method" in result.assembled_prompt
    assert "ONLY the context" in llm.calls[0][0].content  # grounding instruction reached the model


@pytest.mark.asyncio
async def test_answer_handles_empty_retrieval_without_crashing():
    store = InMemoryVectorStore()  # empty
    embedder = FakeEmbedder(dimension=4)
    retriever = Retriever(embedder=embedder, vector_store=store)
    llm = FakeLLMClient([LLMResponse(text="I don't have enough information to answer that.")])

    pipeline = RagPipeline(retriever=retriever, llm=llm)
    result = await pipeline.answer("what is this paper about?")

    assert result.retrieved == ()
    assert "no relevant context" in result.assembled_prompt


@pytest.mark.asyncio
async def test_answer_records_retrieval_event_on_tracer():
    store = InMemoryVectorStore()
    embedder = FakeEmbedder(dimension=4)
    chunk = Chunk(chunk_id="c1", document_id="d1", collection="papers", text="text", section_path=())
    [vec] = await embedder.embed([chunk.text])
    await store.upsert([chunk], [vec])
    retriever = Retriever(embedder=embedder, vector_store=store)
    llm = FakeLLMClient([LLMResponse(text="answer")])
    tracer = TraceCollector()

    pipeline = RagPipeline(retriever=retriever, llm=llm, tracer=tracer)
    await pipeline.answer("query")

    # tracer.events holds (elapsed, event_name, data) tuples.
    events = [data for _, event, data in tracer.events if event == "retrieval_result"]
    assert len(events) == 1
    assert events[0]["chunk_ids"] == ["c1"]
