"""Pre-registered success criteria.

Thresholds are written down *before* an experiment runs, so a disappointing
result cannot be rationalized after the fact. They are expectations, not
measurements — which is why they are committed while reports are not.

The gate fails only on a *failed threshold*, never on imperfection. A hard
eval set is never 100%; a gate demanding perfection is red permanently and
therefore ignored, which is worse than no gate because it looks like coverage.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from evaluation.harness.report import EvalReport


@dataclass(frozen=True)
class Gate:
    metric: str
    granularity: str | None
    min_mean: float
    # Required. A threshold with no recorded reasoning gets quietly lowered
    # the first time it is inconvenient.
    rationale: str


def load_gates(path: Path) -> list[Gate]:
    if not path.exists():
        return []
    raw = yaml.safe_load(path.read_text()) or {}
    gates: list[Gate] = []
    for entry in raw.get("gates") or []:
        rationale = (entry.get("rationale") or "").strip()
        if not rationale:
            raise ValueError(
                f"gate on '{entry.get('metric')}' has no rationale. Record why this "
                "threshold is what it is, or it will be lowered without argument."
            )
        gates.append(
            Gate(
                metric=entry["metric"],
                granularity=entry.get("granularity"),
                min_mean=float(entry["min_mean"]),
                rationale=rationale,
            )
        )
    return gates


def evaluate_gates(gates: list[Gate], report) -> list[str]:
    """Return one message per failed gate. Empty list means the run passes."""
    rows = {(row["metric"], row["granularity"]): row for row in report.aggregate()}
    failures: list[str] = []
    for gate in gates:
        row = rows.get((gate.metric, gate.granularity))
        if row is None:
            failures.append(
                f"{gate.metric} ({gate.granularity or 'any'}): not present in this run — "
                "the gate cannot be evaluated, which is a failure, not a pass."
            )
            continue
        if row["mean"] < gate.min_mean:
            failures.append(
                f"{gate.metric} ({gate.granularity or 'any'}): mean {row['mean']} "
                f"below threshold {gate.min_mean} over n={row['n']} — {gate.rationale}"
            )
    return failures


def gate_or_fail(report: EvalReport, thresholds_path: Path) -> int:
    """Evaluate thresholds.yaml gates against report; print failures; return the exit code."""
    failures = evaluate_gates(load_gates(thresholds_path), report)
    if failures:
        print("\nGATE FAILURES:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    return 0
