"""Browse or semantically search a pgvector-backed RAG collection directly.

With no --query, lists chunks in a collection (optionally one document) via a
direct read against the table — VectorStore has no "list" method, only
similarity search, so this reads the table directly rather than adding one
for a debug tool. With --query, embeds the text and runs a real
VectorStore.search() — the same code path retrieval actually uses.

Usage:
    uv run python scripts/query_pgvector.py --collection papers
    uv run python scripts/query_pgvector.py --collection papers --document-id papers/50455611c3ed8b48
    uv run python scripts/query_pgvector.py --collection papers --query "How many encoder layers?" --k 5
    uv run python scripts/query_pgvector.py --dsn postgresql://harness:harness@localhost:5432/litellm --collection papers
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import psycopg
from psycopg.rows import dict_row
from psycopg.sql import SQL, Identifier

_root = Path(__file__).parent.parent
sys.path.insert(0, str(_root / "src"))
if str(_root) not in sys.path:
    sys.path.append(str(_root))

from harness.config.settings import Settings, get_settings  # noqa: E402
from harness.orchestration.build import build_embedder, build_vector_store  # noqa: E402

_SNIPPET_LEN = 200


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Browse or semantically search a pgvector RAG collection"
    )
    parser.add_argument(
        "--dsn",
        default=None,
        help="Override HARNESS_PGVECTOR_URL, e.g. to point at a different db",
    )
    parser.add_argument("--table", default=None, help="Override HARNESS_PGVECTOR_TABLE")
    parser.add_argument("--collection", default="papers")
    parser.add_argument(
        "--document-id", default=None, help="Browse mode: filter to one document"
    )
    parser.add_argument(
        "--query", default=None, help="Semantic search mode: the query text"
    )
    parser.add_argument("--k", type=int, default=5, help="Semantic search: top-k results")
    parser.add_argument("--limit", type=int, default=20, help="Browse mode: max rows")
    parser.add_argument(
        "--full-text",
        action="store_true",
        help="Show full chunk text instead of a snippet",
    )
    return parser.parse_args()


def _resolve_settings(args: argparse.Namespace) -> Settings:
    settings = get_settings()
    overrides: dict = {}
    if args.dsn:
        overrides["pgvector_url"] = args.dsn
    if args.table:
        overrides["pgvector_table"] = args.table
    return settings.model_copy(update=overrides) if overrides else settings


def _print_row(
    chunk_id: str,
    document_id: str,
    text: str,
    full_text: bool,
    score: float | None = None,
) -> None:
    shown = text if full_text else text[:_SNIPPET_LEN].replace("\n", " ")
    suffix = "..." if not full_text and len(text) > _SNIPPET_LEN else ""
    prefix = f"[{score:.4f}] " if score is not None else ""
    print(f"{prefix}{chunk_id}  ({document_id})")
    print(f"  {shown}{suffix}")


async def _browse(settings: Settings, args: argparse.Namespace) -> int:
    conn = await psycopg.AsyncConnection.connect(settings.pgvector_url, autocommit=True)
    async with conn:
        where = ["collection = %s"]
        params: list[object] = [args.collection]
        if args.document_id:
            where.append("document_id = %s")
            params.append(args.document_id)
        stmt = SQL(
            "SELECT chunk_id, document_id, text FROM {table} "
            "WHERE " + " AND ".join(where) + " "
            "ORDER BY document_id, (chunk_json->>'order')::int LIMIT %s"
        ).format(table=Identifier(settings.pgvector_table))
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(stmt, [*params, args.limit])
            rows = await cur.fetchall()

    if not rows:
        target = f"collection '{args.collection}'"
        if args.document_id:
            target += f", document '{args.document_id}'"
        print(f"No chunks found in {target}.")
        return 0

    print(f"{len(rows)} chunk(s) (showing up to {args.limit}):\n")
    for row in rows:
        _print_row(row["chunk_id"], row["document_id"], row["text"], args.full_text)
        print()
    return 0


async def _search(settings: Settings, args: argparse.Namespace) -> int:
    embedder = build_embedder(settings)
    store = build_vector_store(settings, "pgvector")
    [embedding] = await embedder.embed([args.query])
    results = await store.search(embedding, k=args.k, collection=args.collection)

    if not results:
        print(f"No results in collection '{args.collection}'.")
        return 0

    print(f"Top {len(results)} result(s) for: {args.query!r}\n")
    for scored in results:
        _print_row(
            scored.chunk.chunk_id,
            scored.chunk.document_id,
            scored.chunk.text,
            args.full_text,
            score=scored.score,
        )
        print()
    return 0


def main() -> int:
    args = _parse_args()
    settings = _resolve_settings(args)
    if args.query:
        return asyncio.run(_search(settings, args))
    return asyncio.run(_browse(settings, args))


if __name__ == "__main__":
    sys.exit(main())
