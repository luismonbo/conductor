from __future__ import annotations

import pytest

from harness.adapters.chunking.structure_aware import StructureAwareChunker
from harness.adapters.embedding.fake import FakeEmbedder
from harness.adapters.llm.fake import FakeLLMClient
from harness.adapters.normalization.llm_normalizer import LlmNormalizer
from harness.adapters.parsing.markitdown_parser import MarkitdownParser
from harness.adapters.vectorstore.in_memory import InMemoryVectorStore
from harness.cli import ingest as ingest_cli
from harness.cli.ingest import run_ingest
from harness.config.settings import Settings
from harness.core.rag.ingest import IngestionPipeline
from harness.core.types import LLMResponse, ToolCall


def _normalizer_response() -> LLMResponse:
    """The normalizer reaches the model via tool-calling, so the fake must
    answer with a tool call — build_llm's plain 'fake' backend returns free
    text, which LlmNormalizer correctly rejects."""
    return LLMResponse(
        text="",
        tool_calls=(
            ToolCall(
                id="call_1",
                name="emit_normalized_document",
                arguments={
                    "title": "A Note",
                    "sections": [
                        {"title": "Intro", "level": 1, "kind": "prose",
                         "text": "Hello there.", "order": 0},
                    ],
                },
            ),
        ),
    )


@pytest.mark.asyncio
async def test_run_ingest_processes_directory_and_writes_manifest(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw" / "papers"
    raw_dir.mkdir(parents=True)
    (raw_dir / "note.html").write_text("<h1>Intro</h1><p>Hello there.</p>")

    index_config_dir = tmp_path / "index_config"
    settings = Settings(embedding_backend="fake", embedding_dimension=4)

    # Inject a fully faked pipeline: no network calls, no model downloads.
    def _fake_pipeline(_settings, _backends, tracer=None):
        return IngestionPipeline(
            parser=MarkitdownParser(),
            normalizer=LlmNormalizer(FakeLLMClient([_normalizer_response()])),
            chunker=StructureAwareChunker(),
            embedder=FakeEmbedder(dimension=4),
            vector_stores=[InMemoryVectorStore()],
            tracer=tracer,
        )

    monkeypatch.setattr(ingest_cli, "build_ingestion_pipeline", _fake_pipeline)

    results = await run_ingest(
        settings=settings,
        collection="papers",
        raw_dir=raw_dir,
        index_config_dir=index_config_dir,
        vector_store_backends=["in_memory"],
    )

    assert len(results) == 1
    assert results[0].error is None
    assert results[0].chunk_count == 1

    manifest_path = index_config_dir / "papers.yaml"
    assert manifest_path.exists()
    content = manifest_path.read_text()
    assert "embedding_model" in content
    assert "chunk_version" in content
