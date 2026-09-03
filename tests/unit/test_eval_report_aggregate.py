from __future__ import annotations

import pytest

from evaluation.harness.metric import MetricResult
from evaluation.harness.report import CaseReport, EvalReport


def _case(case_id, query_type, *results, latency_ms=1.0):
    return CaseReport(
        case_id=case_id,
        input="q",
        passed=True,
        metric_results=list(results),
        query_type=query_type,
        latency_ms=latency_ms,
    )


def _mr(score, granularity="chunk", skipped=False, name="recall_at_k"):
    return MetricResult(
        name=name,
        passed=score > 0,
        score=score,
        reason="r",
        granularity=granularity,
        skipped=skipped,
    )


def test_aggregate_reports_mean_not_just_counts():
    report = EvalReport(
        run_id="r",
        dataset="d",
        cases=[
            _case("a", "factual", _mr(1.0)),
            _case("b", "factual", _mr(0.0)),
            _case("c", "factual", _mr(0.5)),
        ],
    )

    [row] = report.aggregate()
    assert row["n"] == 3
    assert row["mean"] == pytest.approx(0.5)
    assert row["median"] == pytest.approx(0.5)


def test_aggregate_never_pools_across_granularities():
    report = EvalReport(
        run_id="r",
        dataset="d",
        cases=[
            _case("a", "factual", _mr(1.0, granularity="chunk")),
            _case("b", "factual", _mr(0.0, granularity="document")),
        ],
    )

    rows = {row["granularity"]: row for row in report.aggregate()}
    assert set(rows) == {"chunk", "document"}
    assert rows["chunk"]["mean"] == pytest.approx(1.0)
    assert rows["document"]["mean"] == pytest.approx(0.0)


def test_skipped_results_are_excluded_from_the_mean():
    # Without this, negative cases scoring a nominal 1.0 inflate every mean.
    report = EvalReport(
        run_id="r",
        dataset="d",
        cases=[
            _case("a", "factual", _mr(0.0)),
            _case("neg", "negative", _mr(1.0, granularity=None, skipped=True)),
        ],
    )

    [row] = report.aggregate()
    assert row["n"] == 1
    assert row["mean"] == pytest.approx(0.0)


def test_by_query_type_breaks_metrics_out_per_axis():
    report = EvalReport(
        run_id="r",
        dataset="d",
        cases=[
            _case("a", "factual", _mr(1.0)),
            _case("b", "multi_hop", _mr(0.0)),
        ],
    )

    breakdown = report.by_query_type()
    assert breakdown["factual"][0]["mean"] == pytest.approx(1.0)
    assert breakdown["multi_hop"][0]["mean"] == pytest.approx(0.0)


def test_to_dict_carries_config_latency_and_aggregates():
    report = EvalReport(
        run_id="r",
        dataset="d",
        config={"k": 5},
        cases=[
            _case("a", "factual", _mr(1.0), latency_ms=12.5),
        ],
    )

    out = report.to_dict()
    assert out["config"] == {"k": 5}
    assert out["summary"]["aggregate"][0]["mean"] == pytest.approx(1.0)
    assert out["cases"][0]["latency_ms"] == pytest.approx(12.5)
    assert out["cases"][0]["query_type"] == "factual"


def test_aggregate_is_empty_when_every_result_is_skipped():
    report = EvalReport(
        run_id="r",
        dataset="d",
        cases=[
            _case("neg", "negative", _mr(1.0, granularity=None, skipped=True)),
        ],
    )

    assert report.aggregate() == []
