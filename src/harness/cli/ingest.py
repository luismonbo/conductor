"""CLI entrypoint for RAG ingestion.

Usage:
    uv run python -m harness.cli.ingest --collection papers
    uv run python -m harness.cli.ingest --collection papers --vector-store pgvector
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import yaml

from harness.adapters.chunking.structure_aware import CHUNK_VERSION
from harness.config.settings import Settings, get_settings
from harness.core.rag.ingest import IngestionPipeline, IngestResult
from harness.orchestration.build import build_ingestion_pipeline

_ALL_BACKENDS = ["pgvector", "milvus"]


async def run_ingest(
    settings: Settings,
    collection: str,
    raw_dir: Path,
    index_config_dir: Path,
    vector_store_backends: list[str],
) -> list[IngestResult]:
    pipeline: IngestionPipeline = build_ingestion_pipeline(settings, vector_store_backends)
    results = await pipeline.ingest_collection(raw_dir, collection)

    index_config_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "collection": collection,
        "embedding_model": settings.embedding_model,
        "embedding_dimension": settings.embedding_dimension,
        "chunk_version": CHUNK_VERSION,
        "vector_stores": vector_store_backends,
        "documents_ingested": sum(1 for r in results if r.error is None),
        "documents_failed": sum(1 for r in results if r.error is not None),
    }
    (index_config_dir / f"{collection}.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG vector stores")
    parser.add_argument("--collection", default="papers")
    parser.add_argument(
        "--vector-store", default="all",
        help="pgvector | milvus | all (comma-separated for a subset, e.g. pgvector,milvus)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    settings = get_settings()
    backends = _ALL_BACKENDS if args.vector_store == "all" else args.vector_store.split(",")

    raw_dir = Path("data/raw") / args.collection
    if not raw_dir.exists():
        print(f"No such collection directory: {raw_dir}", file=sys.stderr)
        return 1

    results = asyncio.run(
        run_ingest(
            settings=settings, collection=args.collection, raw_dir=raw_dir,
            index_config_dir=Path("data/index_config"), vector_store_backends=backends,
        )
    )

    ok = sum(1 for r in results if r.error is None)
    print(f"Ingested {ok}/{len(results)} documents from {raw_dir} into {backends}")
    for r in results:
        if r.error:
            print(f"  FAILED {r.source_path}: {r.error}", file=sys.stderr)
    return 0 if all(r.error is None for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
