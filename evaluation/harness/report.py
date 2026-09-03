"""EvalReport: structured result of one eval run."""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from evaluation.harness.metric import MetricResult


@dataclass
class CaseReport:
    case_id: str
    input: str
    passed: bool
    output: str | None = None
    metric_results: list[MetricResult] = field(default_factory=list)
    error: str | None = None
    query_type: str | None = None
    latency_ms: float | None = None
    # What a production query costs (embedding, generation). Kept separate from
    # eval_tokens because conflating them makes the cost axis useless for the
    # exact trade it exists to price: a reranker that lifts quality while
    # doubling per-query cost.
    system_tokens: dict[str, int] | None = None
    # What it cost to *evaluate* (judge calls). Never a production cost.
    eval_tokens: dict[str, int] | None = None


@dataclass
class EvalReport:
    run_id: str
    dataset: str
    cases: list[CaseReport] = field(default_factory=list)
    # The retrieval configuration that produced this run. A score that cannot
    # be attributed to a configuration is as useless as no score.
    config: dict | None = None
    corpus: dict | None = None

    @property
    def total(self) -> int:
        return len(self.cases)

    @property
    def passed(self) -> int:
        return sum(1 for c in self.cases if c.passed)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    def _rows(self, cases: list[CaseReport]) -> list[dict]:
        """Group scores by (metric, granularity) and summarize.

        Granularity is never pooled: document-granularity recall is
        systematically easier than chunk-granularity, so one mean over both is
        a number that means nothing and would drift as the tier ratio changes.
        Skipped results are excluded — a negative case scoring a nominal 1.0
        would otherwise inflate every mean it touches.
        """
        buckets: dict[tuple[str, str | None], list[MetricResult]] = {}
        for case in cases:
            for mr in case.metric_results:
                if mr.skipped:
                    continue
                buckets.setdefault((mr.name, mr.granularity), []).append(mr)

        rows: list[dict] = []
        for (name, granularity), results in sorted(
            buckets.items(), key=lambda kv: (kv[0][0], kv[0][1] or "")
        ):
            scores = [mr.score for mr in results]
            rows.append(
                {
                    "metric": name,
                    "granularity": granularity,
                    "n": len(scores),
                    "passed": sum(1 for mr in results if mr.passed),
                    "failed": sum(1 for mr in results if not mr.passed),
                    "mean": round(statistics.fmean(scores), 4),
                    "median": round(statistics.median(scores), 4),
                    "min": round(min(scores), 4),
                    "max": round(max(scores), 4),
                }
            )
        return rows

    def aggregate(self) -> list[dict]:
        return self._rows(self.cases)

    def by_query_type(self) -> dict[str, list[dict]]:
        """Aggregates say *that* something regressed; this says *what* broke."""
        grouped: dict[str, list[CaseReport]] = {}
        for case in self.cases:
            grouped.setdefault(case.query_type or "untyped", []).append(case)
        return {qt: self._rows(cases) for qt, cases in sorted(grouped.items())}

    def latency_summary(self) -> dict | None:
        values = [c.latency_ms for c in self.cases if c.latency_ms is not None]
        if not values:
            return None
        ordered = sorted(values)
        p95_index = max(0, int(len(ordered) * 0.95) - 1)
        return {
            "mean_ms": round(statistics.fmean(ordered), 2),
            "median_ms": round(statistics.median(ordered), 2),
            "p95_ms": round(ordered[p95_index], 2),
        }

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "dataset": self.dataset,
            "config": self.config,
            "corpus": self.corpus,
            "summary": {
                "total": self.total,
                "passed": self.passed,
                "failed": self.failed,
                "pass_rate": round(self.pass_rate, 4),
                "aggregate": self.aggregate(),
                "by_query_type": self.by_query_type(),
                "latency": self.latency_summary(),
            },
            "cases": [
                {
                    "case_id": c.case_id,
                    "input": c.input,
                    "output": c.output,
                    "passed": c.passed,
                    "error": c.error,
                    "query_type": c.query_type,
                    "latency_ms": c.latency_ms,
                    "system_tokens": c.system_tokens,
                    "eval_tokens": c.eval_tokens,
                    "metrics": [
                        {
                            "name": mr.name,
                            "passed": mr.passed,
                            "score": mr.score,
                            "reason": mr.reason,
                            "granularity": mr.granularity,
                            "skipped": mr.skipped,
                        }
                        for mr in c.metric_results
                    ],
                }
                for c in self.cases
            ],
        }

    def save(self, reports_dir: Path) -> Path:
        reports_dir.mkdir(parents=True, exist_ok=True)
        out = reports_dir / f"{self.run_id}-{Path(self.dataset).stem}.json"
        out.write_text(json.dumps(self.to_dict(), indent=2))
        return out

    def print_summary(self) -> None:
        print(f"\n{'=' * 72}")
        print(f"Eval run: {self.run_id}")
        print(f"Dataset:  {self.dataset}")
        if self.config:
            print(f"Config:   {self.config}")
        print(
            f"Cases:    {self.total}  ({self.failed} with an errored or zero-hit metric)"
        )
        print()
        header = (
            f"  {'metric':<18}{'gran':<10}{'n':>4}"
            f"{'mean':>9}{'median':>9}{'min':>8}{'max':>8}"
        )
        print(header)
        print(f"  {'-' * (len(header) - 2)}")
        for row in self.aggregate():
            print(
                f"  {row['metric']:<18}{row['granularity'] or '-':<10}{row['n']:>4}"
                f"{row['mean']:>9.4f}{row['median']:>9.4f}"
                f"{row['min']:>8.4f}{row['max']:>8.4f}"
            )
        latency = self.latency_summary()
        if latency:
            print(
                f"\n  latency: mean {latency['mean_ms']}ms  "
                f"median {latency['median_ms']}ms  p95 {latency['p95_ms']}ms"
            )
        print("\n  by query_type:")
        for query_type, rows in self.by_query_type().items():
            for row in rows:
                print(
                    f"    {query_type:<16}{row['metric']:<18}"
                    f"{row['granularity'] or '-':<10}n={row['n']:<4}"
                    f"mean={row['mean']:.4f}"
                )
        print()
        for c in self.cases:
            if c.error:
                print(f"  [error] {c.case_id}: {c.error}")
        print(f"{'=' * 72}\n")

    @staticmethod
    def make_run_id() -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
