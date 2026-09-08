"""Refresh a dataset's corpus fingerprint against the live index.

`verify_corpus()` only checks documents already listed in a dataset's
`corpus.documents` — it has no way to notice a document the fingerprint
doesn't mention yet. After ingesting new content into a collection
(`make ingest COLLECTION=papers`), run this to fold the live counts back
into the dataset before labelling against them, so the next eval run's
fingerprint check actually covers what's new.

Usage:
    uv run python scripts/refresh_corpus_fingerprint.py
    uv run python scripts/refresh_corpus_fingerprint.py --dataset papers_v2.json --collection papers
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import date
from pathlib import Path

_root = Path(__file__).parent.parent
sys.path.insert(0, str(_root / "src"))
if str(_root) not in sys.path:
    sys.path.append(str(_root))

from harness.config.settings import get_settings  # noqa: E402
from harness.orchestration.build import build_vector_store  # noqa: E402

_DATASETS_DIR = _root / "evaluation" / "rag" / "datasets"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh a dataset's corpus fingerprint against the live index"
    )
    parser.add_argument("--dataset", default="papers_v2.json")
    parser.add_argument("--collection", default="papers")
    parser.add_argument(
        "--vector-store",
        default="pgvector",
        choices=["pgvector", "milvus", "in_memory"],
    )
    parser.add_argument(
        "--yes", action="store_true", help="Write without a confirmation prompt"
    )
    return parser.parse_args()


async def _live_stats(collection: str, vector_store: str) -> dict[str, int]:
    settings = get_settings()
    store = build_vector_store(settings, vector_store)
    return await store.document_stats(collection=collection)


def main() -> int:
    args = _parse_args()
    path = _DATASETS_DIR / args.dataset
    if not path.exists():
        print(f"Dataset not found: {path}", file=sys.stderr)
        return 1

    doc = json.loads(path.read_text())
    old = doc.get("corpus", {}).get("documents", {})
    new = asyncio.run(_live_stats(args.collection, args.vector_store))

    added = {k: v for k, v in new.items() if k not in old}
    removed = {k: v for k, v in old.items() if k not in new}
    changed = {k: (old[k], new[k]) for k in old if k in new and old[k] != new[k]}
    unchanged = {k for k in old if k in new and old[k] == new[k]}

    if not (added or removed or changed):
        print(f"No drift — {len(unchanged)} document(s) match the live index exactly.")
        return 0

    print(f"Corpus fingerprint drift for '{args.collection}':")
    for doc_id, count in sorted(added.items()):
        print(f"  + {doc_id}: new, {count} chunk(s)")
    for doc_id, count in sorted(removed.items()):
        print(
            f"  - {doc_id}: gone from the index (was {count} chunk(s)) — labelled "
            "cases referencing it will now fail verify_corpus"
        )
    for doc_id, (old_count, new_count) in sorted(changed.items()):
        print(
            f"  ~ {doc_id}: {old_count} -> {new_count} chunk(s) — re-chunked; "
            "existing chunk-level labels for it may now point at the wrong passage"
        )

    if not args.yes:
        answer = (
            input("\nWrite this fingerprint into the dataset? [y/N] ").strip().lower()
        )
        if answer != "y":
            print("Not written.")
            return 1

    doc["corpus"]["documents"] = new
    doc["corpus"]["verified_at"] = date.today().isoformat()
    path.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"\nWrote refreshed fingerprint -> {path}")
    if added:
        print(
            f"Note: {len(added)} new document(s) have no labelled cases yet — grade "
            "some with scripts/label_assist.py before trusting scores that touch them."
        )
    if removed or changed:
        print(
            "Note: some existing labels may now be stale — re-check any case whose "
            "relevant_document_ids references a changed/removed document."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
