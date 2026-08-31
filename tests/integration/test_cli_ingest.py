from __future__ import annotations

import pytest

from harness.cli.ingest import run_ingest
from harness.config.settings import Settings


@pytest.mark.asyncio
async def test_run_ingest_processes_directory_and_writes_manifest(tmp_path):
    """End-to-end through the real composition root: no LLM in the ingest
    path any more, so build_ingestion_pipeline's RoutingNormalizer (routing to
    the deterministic MarkdownNormalizer for markitdown's markdown-ish HTML
    rendering) needs no stubbing to run fast and offline — only the embedder
    and vector store are pinned to fakes/in-memory."""
    raw_dir = tmp_path / "raw" / "papers"
    raw_dir.mkdir(parents=True)
    (raw_dir / "note.html").write_text("<h1>Intro</h1><p>Hello there.</p>")

    index_config_dir = tmp_path / "index_config"
    settings = Settings(
        _env_file=None, embedding_backend="fake", embedding_dimension=4, api_key="test-key",
    )

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
