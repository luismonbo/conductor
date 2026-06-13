"""EvalReport: structured result of one eval run."""
from __future__ import annotations

import json
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


@dataclass
class EvalReport:
    run_id: str
    dataset: str
    cases: list[CaseReport] = field(default_factory=list)

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

    def _by_metric(self) -> dict[str, dict[str, int]]:
        counts: dict[str, dict[str, int]] = {}
        for case in self.cases:
            for mr in case.metric_results:
                bucket = counts.setdefault(mr.name, {"passed": 0, "failed": 0})
                if mr.passed:
                    bucket["passed"] += 1
                else:
                    bucket["failed"] += 1
        return counts

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "dataset": self.dataset,
            "summary": {
                "total": self.total,
                "passed": self.passed,
                "failed": self.failed,
                "pass_rate": round(self.pass_rate, 4),
                "by_metric": self._by_metric(),
            },
            "cases": [
                {
                    "case_id": c.case_id,
                    "input": c.input,
                    "output": c.output,
                    "passed": c.passed,
                    "error": c.error,
                    "metrics": [
                        {
                            "name": mr.name,
                            "passed": mr.passed,
                            "score": mr.score,
                            "reason": mr.reason,
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
        status = "PASS" if self.pass_rate == 1.0 else "FAIL"
        print(f"\n{'='*60}")
        print(f"Eval run: {self.run_id}  [{status}]")
        print(f"Dataset:  {self.dataset}")
        print(f"Results:  {self.passed}/{self.total} passed ({self.pass_rate:.0%})")
        print()
        for name, counts in self._by_metric().items():
            p, f = counts["passed"], counts["failed"]
            print(f"  {name:<25} {p} passed / {f} failed")
        print()
        for c in self.cases:
            icon = "✓" if c.passed else "✗"
            print(f"  {icon} {c.case_id}")
            if not c.passed:
                for mr in c.metric_results:
                    if not mr.passed:
                        print(f"      [{mr.name}] {mr.reason}")
                if c.error:
                    print(f"      [error] {c.error}")
        print(f"{'='*60}\n")

    @staticmethod
    def make_run_id() -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
