from __future__ import annotations

import pytest

from evaluation.harness.metric import MetricResult
from evaluation.harness.report import CaseReport, EvalReport
from evaluation.rag.thresholds import Gate, evaluate_gates, gate_or_fail, load_gates


def _report(score: float, metric: str = "ndcg_at_k") -> EvalReport:
    return EvalReport(
        run_id="r",
        dataset="d",
        cases=[
            CaseReport(
                case_id="a",
                input="q",
                passed=True,
                metric_results=[
                    MetricResult(
                        name=metric,
                        passed=True,
                        score=score,
                        reason="r",
                        granularity="chunk",
                    ),
                ],
            ),
        ],
    )


def test_no_gates_means_no_failures():
    assert evaluate_gates([], _report(0.01)) == []


def test_satisfied_gate_produces_no_failure():
    gate = Gate(metric="ndcg_at_k", granularity="chunk", min_mean=0.7, rationale="x")

    assert evaluate_gates([gate], _report(0.8)) == []


def test_failed_gate_names_metric_and_both_numbers():
    gate = Gate(metric="ndcg_at_k", granularity="chunk", min_mean=0.7, rationale="x")

    [failure] = evaluate_gates([gate], _report(0.5))
    assert "ndcg_at_k" in failure
    assert "0.5" in failure
    assert "0.7" in failure


def test_gate_on_an_absent_metric_fails_rather_than_passing_silently():
    gate = Gate(metric="missing_metric", granularity="chunk", min_mean=0.7, rationale="x")

    [failure] = evaluate_gates([gate], _report(0.9))
    assert "not present" in failure


def test_loads_empty_gates_file(tmp_path):
    path = tmp_path / "t.yaml"
    path.write_text("gates: []\n")

    assert load_gates(path) == []


def test_loads_a_populated_gate(tmp_path):
    path = tmp_path / "t.yaml"
    path.write_text(
        "gates:\n"
        "  - metric: ndcg_at_k\n"
        "    granularity: chunk\n"
        "    min_mean: 0.72\n"
        "    rationale: baseline plus smallest resolvable effect\n"
    )

    [gate] = load_gates(path)
    assert gate.metric == "ndcg_at_k"
    assert gate.min_mean == pytest.approx(0.72)


def test_gate_without_rationale_is_rejected(tmp_path):
    path = tmp_path / "t.yaml"
    path.write_text("gates:\n  - metric: m\n    granularity: chunk\n    min_mean: 0.5\n")

    with pytest.raises(ValueError, match="rationale"):
        load_gates(path)


def test_missing_file_means_no_gates(tmp_path):
    assert load_gates(tmp_path / "absent.yaml") == []


def test_gate_or_fail_returns_0_when_no_gates(tmp_path, capsys):
    path = tmp_path / "t.yaml"
    path.write_text("gates: []\n")

    assert gate_or_fail(_report(0.5), path) == 0
    assert "GATE FAILURES" not in capsys.readouterr().err


def test_gate_or_fail_returns_0_when_gate_satisfied(tmp_path, capsys):
    path = tmp_path / "t.yaml"
    path.write_text(
        "gates:\n"
        "  - metric: ndcg_at_k\n"
        "    granularity: chunk\n"
        "    min_mean: 0.5\n"
        "    rationale: x\n"
    )

    assert gate_or_fail(_report(0.8), path) == 0
    assert "GATE FAILURES" not in capsys.readouterr().err


def test_gate_or_fail_returns_1_and_prints_failures_when_gate_fails(tmp_path, capsys):
    path = tmp_path / "t.yaml"
    path.write_text(
        "gates:\n"
        "  - metric: ndcg_at_k\n"
        "    granularity: chunk\n"
        "    min_mean: 0.9\n"
        "    rationale: x\n"
    )

    assert gate_or_fail(_report(0.5), path) == 1
    err = capsys.readouterr().err
    assert "GATE FAILURES" in err
    assert "ndcg_at_k" in err
