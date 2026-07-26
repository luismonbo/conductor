from __future__ import annotations

import pytest

from harness.adapters.embedding.fake import FakeEmbedder
from harness.adapters.vectorstore.in_memory import InMemoryVectorStore
from harness.core.rag.document import DocumentSection, NormalizedDocument, ParsedContent
from harness.core.rag.ingest import IngestionPipeline


class _StubParser:
    async def parse(self, path):
        return ParsedContent(text="# Intro\n\nHello.", format="html", parser="markitdown")


class _StubNormalizer:
    async def normalize(self, parsed, source_path, collection):
        return [
            NormalizedDocument(
                document_id=f"{collection}/stubhash",
                source_path=source_path,
                collection=collection,
                title="Stub Doc",
                format=parsed.format,
                parser=parsed.parser,
                content_hash="stubhash",
                sections=(DocumentSection(title="Intro", level=1, text="Hello.", order=0),),
                ingested_at="2026-07-25T00:00:00Z",
            )
        ]


class _StubChunker:
    def chunk(self, document):
        from harness.core.rag.document import Chunk

        return [
            Chunk(
                chunk_id=f"{document.document_id}:0",
                document_id=document.document_id,
                collection=document.collection,
                text=document.sections[0].text,
                section_path=("Intro",),
            )
        ]


class _FailingParser:
    async def parse(self, path):
        raise RuntimeError("simulated parse failure")


def _events_named(tracer, name: str) -> list[dict]:
    """tracer.events holds (elapsed, event_name, data) tuples — return the data dicts."""
    return [data for _, event, data in tracer.events if event == name]


@pytest.mark.asyncio
async def test_ingest_file_embeds_and_upserts_to_all_stores(tmp_path):
    store_a, store_b = InMemoryVectorStore(), InMemoryVectorStore()
    pipeline = IngestionPipeline(
        parser=_StubParser(), normalizer=_StubNormalizer(), chunker=_StubChunker(),
        embedder=FakeEmbedder(dimension=4), vector_stores=[store_a, store_b],
    )
    source = tmp_path / "note.html"
    source.write_text("<h1>Intro</h1><p>Hello.</p>")

    result = await pipeline.ingest_file(source, collection="papers")

    assert result.error is None
    assert result.chunk_count == 1
    assert result.document_ids == ("papers/stubhash",)
    assert await store_a.count(collection="papers") == 1
    assert await store_b.count(collection="papers") == 1


@pytest.mark.asyncio
async def test_ingest_file_stamps_embedding_model_before_upsert(tmp_path):
    store = InMemoryVectorStore()
    pipeline = IngestionPipeline(
        parser=_StubParser(), normalizer=_StubNormalizer(), chunker=_StubChunker(),
        embedder=FakeEmbedder(dimension=4), vector_stores=[store],
    )
    source = tmp_path / "note.html"
    source.write_text("<h1>Intro</h1><p>Hello.</p>")

    await pipeline.ingest_file(source, collection="papers")

    [scored] = await store.search([0.0, 0.0, 0.0, 0.0], k=1, collection="papers")
    assert scored.chunk.embedding_model != ""


@pytest.mark.asyncio
async def test_ingest_collection_does_not_abort_on_one_bad_file(tmp_path):
    good = tmp_path / "good.html"
    good.write_text("<h1>Intro</h1><p>Hello.</p>")
    bad = tmp_path / "bad.html"
    bad.write_text("<h1>Bad</h1>")

    pipeline = IngestionPipeline(
        parser=_FailingParser(), normalizer=_StubNormalizer(), chunker=_StubChunker(),
        embedder=FakeEmbedder(), vector_stores=[InMemoryVectorStore()],
    )

    results = await pipeline.ingest_collection(tmp_path, collection="papers")

    assert len(results) == 2
    assert all(r.error is not None for r in results)  # _FailingParser fails on both, on purpose
    assert all(r.chunk_count == 0 for r in results)


@pytest.mark.asyncio
async def test_ingest_file_records_trace_events(tmp_path):
    from harness.observability.tracer import TraceCollector

    tracer = TraceCollector()
    pipeline = IngestionPipeline(
        parser=_StubParser(), normalizer=_StubNormalizer(), chunker=_StubChunker(),
        embedder=FakeEmbedder(dimension=4), vector_stores=[InMemoryVectorStore()],
        tracer=tracer,
    )
    source = tmp_path / "note.html"
    source.write_text("<h1>Intro</h1><p>Hello.</p>")

    await pipeline.ingest_file(source, collection="papers")

    events = _events_named(tracer, "ingest_file_complete")
    assert len(events) == 1
    assert events[0]["chunk_count"] == 1
    assert events[0]["source_path"] == str(source)


@pytest.mark.asyncio
async def test_ingest_file_records_trace_event_on_failure(tmp_path):
    from harness.observability.tracer import TraceCollector

    tracer = TraceCollector()
    pipeline = IngestionPipeline(
        parser=_FailingParser(), normalizer=_StubNormalizer(), chunker=_StubChunker(),
        embedder=FakeEmbedder(), vector_stores=[InMemoryVectorStore()], tracer=tracer,
    )
    source = tmp_path / "bad.html"
    source.write_text("<h1>Bad</h1>")

    await pipeline.ingest_file(source, collection="papers")

    events = _events_named(tracer, "ingest_file_failed")
    assert len(events) == 1
    assert "simulated parse failure" in events[0]["error"]
