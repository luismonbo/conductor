"""CLI entrypoint for asking a question against the RAG index.

Usage:
    uv run python -m harness.cli.rag_query "What method does the paper use?"
    uv run python -m harness.cli.rag_query "..." --vector-store milvus --k 3 --show-context
"""
from __future__ import annotations

import argparse
import asyncio
import sys

from harness.config.settings import Settings, get_settings
from harness.core.rag.serve import RagResult
from harness.orchestration.build import build_rag_pipeline


async def run_query(
    settings: Settings, query: str, collection: str, vector_store_backend: str, k: int = 5
) -> RagResult:
    pipeline = build_rag_pipeline(settings, vector_store_backend)
    return await pipeline.answer(query, k=k, collection=collection)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask a question against the RAG index")
    parser.add_argument("query")
    parser.add_argument("--collection", default="papers")
    parser.add_argument(
        "--vector-store", default="pgvector", choices=["pgvector", "milvus", "in_memory"]
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--show-context", action="store_true", help="Print retrieved chunks before the answer"
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    settings = get_settings()
    result = asyncio.run(
        run_query(settings, args.query, args.collection, args.vector_store, args.k)
    )

    if args.show_context:
        print("--- retrieved context ---")
        for sc in result.retrieved:
            print(f"  ({sc.score:.2f}) {sc.chunk.chunk_id}: {sc.chunk.text[:120]}...")
        print()

    print(result.answer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
