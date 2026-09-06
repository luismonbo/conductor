"""Delete chunks from one app-level collection in the shared Milvus store.

Milvus stores every app-level collection (papers, docs, ...) as rows in ONE
physical Milvus collection, distinguished by a `collection_name` field —
there is no native per-app-collection drop. This deletes document-by-document
via the VectorStore port's delete(document_id), since document_id already
encodes which app collection it belongs to (e.g. "papers/50455611c3ed8b48").

With no --document-id, clears the ENTIRE collection — every document in it.
Pass --document-id (repeatable) to remove only specific documents instead,
leaving the rest of the collection untouched.

Usage:
    uv run python scripts/clear_milvus_collection.py --collection papers
    uv run python scripts/clear_milvus_collection.py --collection papers --yes
    uv run python scripts/clear_milvus_collection.py --collection papers \\
        --document-id papers/703b134f67118e3e --document-id papers/4b3a7e5412a40915
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

_root = Path(__file__).parent.parent
sys.path.insert(0, str(_root / "src"))
if str(_root) not in sys.path:
    sys.path.append(str(_root))

from harness.config.settings import get_settings  # noqa: E402
from harness.orchestration.build import build_vector_store  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete chunks from one collection in the Milvus store "
        "(whole collection, or specific documents with --document-id)"
    )
    parser.add_argument(
        "--collection",
        required=True,
        help="App-level collection to clear, e.g. papers",
    )
    parser.add_argument(
        "--uri",
        default=None,
        help="Override HARNESS_MILVUS_URI (Milvus Lite file path or server URI)",
    )
    parser.add_argument(
        "--document-id",
        action="append",
        dest="document_ids",
        default=None,
        help="Delete only this document (repeatable). Omit to clear the whole collection.",
    )
    parser.add_argument(
        "--yes", action="store_true", help="Delete without a confirmation prompt"
    )
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> int:
    settings = get_settings()
    if args.uri:
        settings = settings.model_copy(update={"milvus_uri": args.uri})
    store = build_vector_store(settings, "milvus")

    stats = await store.document_stats(collection=args.collection)
    if not stats:
        print(f"Collection '{args.collection}' is already empty (or doesn't exist).")
        return 0

    if args.document_ids:
        missing = [d for d in args.document_ids if d not in stats]
        if missing:
            print(
                f"Not found in collection '{args.collection}': {missing}. "
                f"Present: {sorted(stats)}",
                file=sys.stderr,
            )
            return 1
        targets = {d: stats[d] for d in args.document_ids}
        scope = f"{len(targets)} of {len(stats)} document(s) in"
    else:
        targets = stats
        scope = f"ALL {len(stats)} document(s) in"

    total_chunks = sum(targets.values())
    print(f"{scope} collection '{args.collection}' ({total_chunks} chunk(s)):")
    for document_id, count in sorted(targets.items()):
        print(f"  {document_id}: {count} chunk(s)")

    if not args.yes:
        answer = input(f"\nDelete these {total_chunks} chunk(s)? [y/N] ").strip().lower()
        if answer != "y":
            print("Not deleted.")
            return 1

    for document_id in sorted(targets):
        await store.delete(document_id)
        print(f"  deleted {document_id}")

    remaining = await store.document_stats(collection=args.collection)
    still_there = [d for d in targets if d in remaining]
    if still_there:
        print(
            f"\nWarning: {len(still_there)} document(s) still present after delete — "
            "delete() may not have taken effect for all rows.",
            file=sys.stderr,
        )
        return 1
    print(
        f"\nDeleted. Collection '{args.collection}' now has {len(remaining)} document(s)."
    )
    return 0


def main() -> int:
    return asyncio.run(_run(_parse_args()))


if __name__ == "__main__":
    sys.exit(main())
