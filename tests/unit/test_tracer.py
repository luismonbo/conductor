from __future__ import annotations

import pytest

from harness.observability.tracer import TraceCollector


@pytest.mark.asyncio
async def test_usage_for_sums_matching_events():
    tracer = TraceCollector()
    await tracer("token_usage", {"usage": {"input_tokens": 10, "output_tokens": 5}})
    await tracer("token_usage", {"usage": {"input_tokens": 3, "output_tokens": 1}})

    assert tracer.usage_for("token_usage") == {"input_tokens": 13, "output_tokens": 6}


@pytest.mark.asyncio
async def test_usage_for_ignores_other_event_names():
    tracer = TraceCollector()
    await tracer("token_usage", {"usage": {"input_tokens": 10, "output_tokens": 5}})
    await tracer("judge_token_usage", {"usage": {"input_tokens": 100, "output_tokens": 50}})

    assert tracer.usage_for("token_usage") == {"input_tokens": 10, "output_tokens": 5}
    assert tracer.usage_for("judge_token_usage") == {"input_tokens": 100, "output_tokens": 50}


@pytest.mark.asyncio
async def test_usage_for_returns_none_when_nothing_matches():
    tracer = TraceCollector()
    await tracer("retrieval_result", {"chunk_ids": ["c1"]})

    assert tracer.usage_for("token_usage") is None
