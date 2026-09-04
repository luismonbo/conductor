"""Labelling assistant — retrieve a pool, grade it, append the case.

Prints each chunk in the retrieval pool with rank and score so a human can
grade it against datasets/RUBRIC.md. The human confirms or overrides every
grade; nothing is written without confirmation.

Usage:
    uv run python scripts/label_assist.py \
        --query "How many layers are in the encoder stack?" \
        --case-id attn_encoder_layers \
        --query-type factual --difficulty easy

Overrides during a session are signal that the rubric wording is ambiguous.
Note them and tighten RUBRIC.md — that is the point of a calibration pass.
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
from harness.orchestration.build import build_retriever, build_vector_store  # noqa: E402

_DATASETS_DIR = _root / "evaluation" / "rag" / "datasets"
_VALID_GRADES = {"0", "1", "2", "3"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grade a retrieval pool for one eval case"
    )
    parser.add_argument("--query", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--dataset", default="papers_v2.json")
    parser.add_argument("--depth", type=int, default=15, help="Pool depth (default 15)")
    parser.add_argument("--collection", default="papers")
    parser.add_argument(
        "--vector-store", default="pgvector", choices=["pgvector", "milvus", "in_memory"]
    )
    parser.add_argument("--query-type", default="factual")
    parser.add_argument("--difficulty", default="medium")
    parser.add_argument("--expected-behaviour", default="grounded_answer")
    parser.add_argument("--verified-by", default="luis")
    parser.add_argument("--suites", nargs="*", default=[])
    parser.add_argument("--excluded-from", nargs="*", default=[])
    return parser.parse_args()


def _prompt_grade(rank: int, scored) -> int:
    chunk = scored.chunk
    section = " > ".join(chunk.section_path) or "n/a"
    print(f"\n{'-' * 72}")
    print(f"[rank {rank}]  score={scored.score:.4f}  {chunk.chunk_id}")
    print(f"section: {section}   source: {chunk.source_path}")
    print(f"{'-' * 72}")
    print(chunk.text.strip()[:1200])
    print(f"{'-' * 72}")
    while True:
        answer = (
            input(
                "grade 0-3 (3=complete answer, 2=necessary part, "
                "1=on-topic only, 0=irrelevant) [0]: "
            ).strip()
            or "0"
        )
        if answer in _VALID_GRADES:
            return int(answer)
        print("  Enter 0, 1, 2 or 3.")


async def _pool(args) -> list:
    # Quota off while pooling: the pool should be the retriever's raw top-N so
    # grading is not biased by a per-document cap. build_retriever reads the
    # quota from settings, so it is overridden here rather than passed.
    settings = get_settings().model_copy(update={"rag_per_document_k": 0})
    store = build_vector_store(settings, args.vector_store)
    retriever = build_retriever(settings, store)
    return await retriever.retrieve(args.query, args.depth, args.collection)


def main() -> int:
    args = _parse_args()
    pool = asyncio.run(_pool(args))
    if not pool:
        print("Retrieval returned nothing — is the index populated?", file=sys.stderr)
        return 1

    print(f"\nQuery: {args.query}")
    print(f"Grading a pool of {len(pool)} chunk(s) against RUBRIC.md.")

    graded: dict[str, int] = {}
    for rank, scored in enumerate(pool, start=1):
        graded[scored.chunk.chunk_id] = _prompt_grade(rank, scored)

    relevant_documents = sorted(
        {
            scored.chunk.document_id
            for scored in pool
            if graded.get(scored.chunk.chunk_id, 0) >= 2
        }
    )
    print(
        f"\nGraded {sum(1 for g in graded.values() if g >= 2)} chunk(s) as relevant "
        f"(grade >= 2) across {len(relevant_documents)} document(s)."
    )
    reference = input("Reference answer (optional, one line): ").strip()

    case = {
        "id": args.case_id,
        "query": args.query,
        "expected": {
            # Grade-0 chunks carry no information the pooling assumption does
            # not already supply, so they are dropped to keep the file readable.
            "graded_chunks": {cid: g for cid, g in graded.items() if g > 0},
            "relevant_document_ids": relevant_documents,
            "reference_answer": reference,
        },
        "provenance": {
            "source": "human",
            "verified_by": args.verified_by,
            "verified_at": date.today().isoformat(),
        },
        "query_type": args.query_type,
        "difficulty": args.difficulty,
        "expected_behaviour": args.expected_behaviour,
        "suites": args.suites,
        "excluded_from": args.excluded_from,
    }

    path = _DATASETS_DIR / args.dataset
    doc = (
        json.loads(path.read_text())
        if path.exists()
        else {
            "version": "2.0",
            "corpus": {
                "collection": args.collection,
                "documents": {},
                "verified_at": date.today().isoformat(),
            },
            "cases": [],
        }
    )
    doc["cases"] = [c for c in doc["cases"] if c["id"] != args.case_id] + [case]
    path.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"\nWrote case '{args.case_id}' -> {path}")
    print(
        "Remember to refresh the corpus fingerprint (scripts/label_assist.py --refresh-corpus "
        "is not implemented; see Task 11 Step 2)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
