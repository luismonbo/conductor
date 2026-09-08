import logging
import pytest

from harness.observability.logging_tracer import LoggingTracer


@pytest.mark.asyncio
async def test_logs_events_at_info(caplog):
    tracer = LoggingTracer()
    with caplog.at_level(logging.INFO):
        await tracer(
            "ingest_stage", {"stage": "chunk", "source_path": "p.pdf", "count": 12}
        )
    assert any("chunk" in r.message and "p.pdf" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_failure_events_log_at_warning(caplog):
    tracer = LoggingTracer()
    with caplog.at_level(logging.WARNING):
        await tracer("ingest_file_failed", {"source_path": "bad.pdf", "error": "boom"})
    assert any(
        r.levelno >= logging.WARNING and "bad.pdf" in r.message for r in caplog.records
    )
