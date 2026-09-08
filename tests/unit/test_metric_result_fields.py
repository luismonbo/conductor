from __future__ import annotations

from evaluation.harness.metric import MetricResult


def test_defaults_are_backward_compatible():
    mr = MetricResult(name="m", passed=True, score=1.0, reason="ok")

    assert mr.granularity is None
    assert mr.skipped is False


def test_carries_granularity_and_skipped():
    mr = MetricResult(
        name="recall_at_k",
        passed=True,
        score=1.0,
        reason="skipped",
        granularity="chunk",
        skipped=True,
    )

    assert mr.granularity == "chunk"
    assert mr.skipped is True
