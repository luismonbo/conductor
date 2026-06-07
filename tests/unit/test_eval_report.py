"""Unit tests for EvalReport serialization and summary calculations."""
import json
import tempfile
from pathlib import Path

from evaluation.harness.metric import MetricResult
from evaluation.harness.report import CaseReport, EvalReport


def _make_report(passed_flags: list[bool]) -> EvalReport:
    report = EvalReport(run_id="20260607-120000", dataset="tool_use_v1.json")
    for i, passed in enumerate(passed_flags):
        mr = MetricResult(
            name="tool_call",
            passed=passed,
            score=1.0 if passed else 0.0,
            reason="ok" if passed else "wrong tool",
        )
        report.cases.append(
            CaseReport(
                case_id=f"case_{i:03d}",
                input="test",
                passed=passed,
                metric_results=[mr],
            )
        )
    return report


class TestEvalReportSummary:
    def test_total(self):
        r = _make_report([True, True, False])
        assert r.total == 3

    def test_passed_count(self):
        r = _make_report([True, True, False])
        assert r.passed == 2

    def test_failed_count(self):
        r = _make_report([True, True, False])
        assert r.failed == 1

    def test_pass_rate(self):
        r = _make_report([True, True, False])
        assert abs(r.pass_rate - 2 / 3) < 0.001

    def test_pass_rate_empty(self):
        r = EvalReport(run_id="x", dataset="d")
        assert r.pass_rate == 0.0

    def test_by_metric_counts(self):
        r = _make_report([True, False])
        by_metric = r._by_metric()
        assert by_metric["tool_call"]["passed"] == 1
        assert by_metric["tool_call"]["failed"] == 1


class TestEvalReportSerialization:
    def test_to_dict_has_required_keys(self):
        r = _make_report([True])
        d = r.to_dict()
        assert "run_id" in d
        assert "dataset" in d
        assert "summary" in d
        assert "cases" in d

    def test_summary_shape(self):
        r = _make_report([True, False])
        summary = r.to_dict()["summary"]
        assert summary["total"] == 2
        assert summary["passed"] == 1
        assert summary["failed"] == 1
        assert "pass_rate" in summary
        assert "by_metric" in summary

    def test_case_shape(self):
        r = _make_report([True])
        case = r.to_dict()["cases"][0]
        assert "case_id" in case
        assert "passed" in case
        assert "metrics" in case

    def test_save_writes_valid_json(self):
        r = _make_report([True])
        with tempfile.TemporaryDirectory() as tmpdir:
            out = r.save(Path(tmpdir))
            data = json.loads(out.read_text())
        assert data["run_id"] == "20260607-120000"

    def test_save_filename_includes_dataset_stem(self):
        r = _make_report([True])
        with tempfile.TemporaryDirectory() as tmpdir:
            out = r.save(Path(tmpdir))
        assert "tool_use_v1" in out.name
