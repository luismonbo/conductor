"""Layer-1 CLI: retrieval-only evaluation, no LLM in the loop.

Free, deterministic, and seconds per run — which is what makes sweeping
retrieval configurations tractable. Compare a baseline and a variant by
running this twice with different flags; per-case scores land in the report
for a paired comparison.

Usage:
    uv run python evaluation/run_retrieval_eval.py
    uv run python evaluation/run_retrieval_eval.py --k 10 --per-document-k 3
    uv run python evaluation/run_retrieval_eval.py --exclude-gate sp2

Do NOT run two Milvus evals concurrently — Milvus Lite is an embedded
single-process DB and the second run fails with "Open local milvus failed".
Parallel sweeps need pgvector, or sequential execution.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Python adds the script directory (evaluation/) to sys.path[0] automatically,
# which makes evaluation/harness/ shadow src/harness/. Force src to front, and
# ensure the project root is present so evaluation.* imports resolve.
_root = Path(__file__).parent.parent
sys.path.insert(0, str(_root / "src"))
if str(_root) not in sys.path:
    sys.path.append(str(_root))

from harness.config.settings import get_settings  # noqa: E402
from harness.orchestration.build import build_retriever, build_vector_store  # noqa: E402

from evaluation.rag.dataset import RagDataset  # noqa: E402
from evaluation.rag.metrics.mrr import MRRMetric  # noqa: E402
from evaluation.rag.metrics.ndcg_at_k import NDCGAtKMetric  # noqa: E402
from evaluation.rag.metrics.recall_at_k import RecallAtKMetric  # noqa: E402
from evaluation.rag.retrieval_runner import (  # noqa: E402
    CorpusMismatchError,
    RetrievalConfig,
    RetrievalRunner,
    verify_corpus,
)
from evaluation.rag.thresholds import gate_or_fail  # noqa: E402

_EVAL_DIR = Path(__file__).parent
_DATASETS_DIR = _EVAL_DIR / "rag" / "datasets"
_REPORTS_DIR = _EVAL_DIR / "reports"
_THRESHOLDS = _EVAL_DIR / "rag" / "thresholds.yaml"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the retrieval-only (layer 1) eval")
    parser.add_argument("--dataset", default="papers_v2.json")
    parser.add_argument("--suite", default=None, help="Only cases declaring this suite")
    parser.add_argument(
        "--exclude-gate",
        default=None,
        help="Drop cases whose excluded_from contains this gate, e.g. sp2",
    )
    parser.add_argument(
        "--vector-store", default="pgvector", choices=["pgvector", "milvus", "in_memory"]
    )
    parser.add_argument("--collection", default="papers")
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--per-document-k", type=int, default=None)
    parser.add_argument("--overfetch", type=int, default=None)
    parser.add_argument(
        "--skip-corpus-check",
        action="store_true",
        help="Bypass the fingerprint check. Scores may be meaningless.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    dataset_path = (
        Path(args.dataset)
        if Path(args.dataset).is_absolute()
        else _DATASETS_DIR / args.dataset
    )
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}", file=sys.stderr)
        return 1

    # build_retriever reads the quota and overfetch from settings rather than
    # from parameters, so CLI flags land via model_copy — the same override
    # pattern run_rag_eval.py already uses. Building RetrievalConfig from the
    # *resolved* settings then records what actually ran, not what was typed.
    settings = get_settings()
    overrides: dict = {}
    if args.k is not None:
        overrides["rag_k"] = args.k
    if args.per_document_k is not None:
        overrides["rag_per_document_k"] = args.per_document_k
    if args.overfetch is not None:
        overrides["rag_overfetch"] = args.overfetch
    if overrides:
        settings = settings.model_copy(update=overrides)

    config = RetrievalConfig(
        k=settings.rag_k,
        collection=args.collection,
        per_document_k=settings.rag_per_document_k,
        overfetch=settings.rag_overfetch,
    )

    dataset = RagDataset.load(dataset_path)
    dataset = dataset.filter_by_suite(args.suite).exclude_gate(args.exclude_gate)
    if not dataset.cases:
        print("No cases to run (filters matched nothing).", file=sys.stderr)
        return 1

    store = build_vector_store(settings, backend=args.vector_store)
    if not args.skip_corpus_check:
        actual = asyncio.run(
            store.document_stats(collection=dataset.corpus.collection or None)
        )
        try:
            verify_corpus(dataset.corpus, actual)
        except CorpusMismatchError as exc:
            print(f"\n{exc}\n", file=sys.stderr)
            return 1

    retriever = build_retriever(settings, store)
    metrics = [RecallAtKMetric(), MRRMetric(), NDCGAtKMetric(k=config.k)]

    print(
        f"Running {len(dataset.cases)} case(s) from {dataset_path.name} against "
        f"{args.vector_store} — {config.to_dict()}"
    )
    report = RetrievalRunner(retriever, config).run(
        dataset, metrics, dataset_name=dataset_path.name
    )
    report.corpus = {
        "collection": dataset.corpus.collection,
        "documents": dataset.corpus.documents,
    }

    out_path = report.save(_REPORTS_DIR)
    report.print_summary()
    print(f"Report saved -> {out_path}")

    return gate_or_fail(report, _THRESHOLDS)


if __name__ == "__main__":
    sys.exit(main())
