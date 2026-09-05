"""CLI entrypoint for the RAG evaluation harness.

Usage:
    uv run python evaluation/run_rag_eval.py
    uv run python evaluation/run_rag_eval.py --suite smoke
    uv run python evaluation/run_rag_eval.py --vector-store milvus
"""

from __future__ import annotations

import argparse
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
from harness.orchestration.build import build_llm, build_parser, build_rag_pipeline  # noqa: E402

from evaluation.rag.dataset import RagDataset  # noqa: E402
from evaluation.rag.metrics.answer_relevancy import AnswerRelevancyMetric  # noqa: E402
from evaluation.rag.metrics.faithfulness import FaithfulnessMetric  # noqa: E402
from evaluation.rag.runner import RagRunner  # noqa: E402
from evaluation.rag.thresholds import gate_or_fail  # noqa: E402

_EVAL_DIR = Path(__file__).parent
_DATASETS_DIR = _EVAL_DIR / "rag" / "datasets"
_REPORTS_DIR = _EVAL_DIR / "reports"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the RAG eval harness")
    parser.add_argument("--dataset", default="papers_v2.json")
    parser.add_argument("--suite", default=None, help="Only cases declaring this suite")
    parser.add_argument(
        "--vector-store", default="pgvector", choices=["pgvector", "milvus", "in_memory"]
    )
    parser.add_argument("--backend", default=None, help="Override HARNESS_LLM_BACKEND")
    parser.add_argument(
        "--k", type=int, default=None, help="Retrieval depth (default HARNESS_RAG_K)"
    )
    parser.add_argument(
        "--per-document-k",
        type=int,
        default=None,
        help="Max chunks per document; 0 disables the quota (default HARNESS_RAG_PER_DOCUMENT_K)",
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

    settings = get_settings()
    overrides: dict = {}
    if args.backend:
        overrides["llm_backend"] = args.backend
    if args.k is not None:
        overrides["rag_k"] = args.k
    if args.per_document_k is not None:
        overrides["rag_per_document_k"] = args.per_document_k
    if overrides:
        settings = settings.model_copy(update=overrides)

    dataset = RagDataset.load(dataset_path).filter_by_suite(args.suite)
    if not dataset.cases:
        print(
            "No cases to run (dataset is empty or filters matched nothing). "
            "Populate evaluation/rag/datasets/papers_v2.json once real papers are ingested.",
            file=sys.stderr,
        )
        return 1

    # Known limitation: the judge shares the generator's model, so a model can
    # grade its own output. There is no separate judge-model config anywhere in
    # this codebase yet; flagged rather than silently accepted.
    judge_llm = build_llm(settings, build_parser(settings))

    def pipeline_factory(tracer):
        return build_rag_pipeline(
            settings, vector_store_backend=args.vector_store, tracer=tracer
        )

    # Retrieval-only metrics (recall@k, MRR, nDCG) live in run_retrieval_eval.py,
    # which runs with no LLM judge. This layer scores only the generated answer.
    metrics = [
        FaithfulnessMetric(judge=judge_llm),
        AnswerRelevancyMetric(judge=judge_llm),
    ]
    runner = RagRunner(pipeline_factory, k=settings.rag_k)
    quota = settings.rag_per_document_k or "off"
    print(
        f"Running {len(dataset.cases)} case(s) from {dataset_path.name} against "
        f"{args.vector_store} (k={settings.rag_k}, per-document quota={quota}) ..."
    )
    report = runner.run(dataset, metrics, dataset_name=dataset_path.name)

    out_path = report.save(_REPORTS_DIR)
    report.print_summary()
    print(f"Report saved -> {out_path}")

    return gate_or_fail(report, _EVAL_DIR / "rag" / "thresholds_generation.yaml")


if __name__ == "__main__":
    sys.exit(main())
