"""CLI entrypoint for the evaluation harness.

Usage:
    uv run python evaluation/run_eval.py
    uv run python evaluation/run_eval.py --tags calculator
    uv run python evaluation/run_eval.py --dataset datasets/tool_use_v1.json
    uv run python evaluation/run_eval.py --backend openai_compatible
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

from harness.config.settings import get_settings
from harness.orchestration.build import build_agent, build_long_term

from evaluation.harness.dataset import Dataset
from evaluation.harness.runner import EvalRunner
from evaluation.metrics.arg_schema import ArgSchemaMetric
from evaluation.metrics.no_tool_call import NoToolCallMetric
from evaluation.metrics.output_contains import OutputContainsMetric
from evaluation.metrics.output_contains_any import OutputContainsAnyMetric
from evaluation.metrics.tool_call import ToolCallMetric

_EVAL_DIR = Path(__file__).parent
_DATASETS_DIR = _EVAL_DIR / "datasets"
_REPORTS_DIR = _EVAL_DIR / "reports"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the eval harness")
    parser.add_argument(
        "--dataset",
        default="tool_use_v1.json",
        help="Dataset filename (relative to evaluation/datasets/) or absolute path",
    )
    parser.add_argument(
        "--tags",
        nargs="*",
        default=[],
        help="Filter cases by tag (any match). Example: --tags calculator",
    )
    parser.add_argument(
        "--backend",
        default=None,
        help="Override HARNESS_LLM_BACKEND (fake | openai_compatible | azure)",
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
    if args.backend:
        settings = settings.model_copy(update={"llm_backend": args.backend})

    async def agent_factory(tracer, memory_seed=None):
        memory = build_long_term(settings)
        for fact in (memory_seed or []):
            await memory.write(fact)
        return build_agent(settings, tracer=tracer, long_term=memory)

    dataset = Dataset.load(dataset_path).filter_by_tags(args.tags)
    if not dataset.cases:
        print("No cases matched the given filters.", file=sys.stderr)
        return 1

    print(f"Running {len(dataset.cases)} case(s) from {dataset_path.name} …")

    metrics = [ToolCallMetric(), ArgSchemaMetric(), OutputContainsMetric(), OutputContainsAnyMetric(), NoToolCallMetric()]
    runner = EvalRunner(agent_factory)
    report = runner.run(dataset, metrics, dataset_name=dataset_path.name)

    out_path = report.save(_REPORTS_DIR)
    report.print_summary()
    print(f"Report saved → {out_path}")

    return 0 if report.pass_rate == 1.0 else 1


if __name__ == "__main__":
    sys.exit(main())
