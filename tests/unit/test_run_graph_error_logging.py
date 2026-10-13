"""_run_graph logs the real exception, while the client only ever sees the
fixed sentence security/egress.py gives it."""
import asyncio
import logging

import pytest

from harness.api.main import _run_graph
from harness.core.llm.errors import ContentFiltered
from harness.security.egress import CONTENT_FILTERED_TEXT, UNAVAILABLE_TEXT


class _RaisingGraph:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def ainvoke(self, invoke_arg, config):
        raise self._exc


class _NullRunStore:
    """_run_graph always records the run's outcome; these tests only look at logging."""

    async def finish_run(self, run_id, accumulator, stopped_reason) -> None:
        pass


async def _run(exc: Exception) -> tuple[list, list[str]]:
    queue: asyncio.Queue = asyncio.Queue()
    stopped_reason = [""]
    await _run_graph(
        graph=_RaisingGraph(exc),
        invoke_arg={},
        config={},
        thread_id="t1",
        run_id="r1",
        event_queue=queue,
        accumulator=None,
        stopped_reason_holder=stopped_reason,
        run_store=_NullRunStore(),
    )
    events = []
    while not queue.empty():
        item = queue.get_nowait()
        if item is not None:
            events.append(item)
    return events, stopped_reason


async def test_run_graph_logs_the_real_exception_with_traceback(caplog):
    class PermissionDeniedError(Exception):  # matched by class NAME, not import
        pass

    with caplog.at_level(logging.ERROR):
        await _run(PermissionDeniedError("403 quota exhausted"))

    error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(error_records) == 1
    assert error_records[0].exc_info is not None
    assert "403 quota exhausted" in str(error_records[0].exc_info[1])
    # thread/run id present so the log line can be correlated back to a run.
    assert "t1" in error_records[0].getMessage()
    assert "r1" in error_records[0].getMessage()


async def test_run_graph_sends_the_client_the_generic_sentence_never_the_exception(caplog):
    class PermissionDeniedError(Exception):
        pass

    with caplog.at_level(logging.ERROR):
        events, stopped_reason = await _run(
            PermissionDeniedError("403 quota exhausted, deployment=prod-gpt5")
        )

    assert stopped_reason == ["error"]
    [error] = [e for e in events if e.type == "error"]
    assert (error.text, error.stopped_reason) == (UNAVAILABLE_TEXT, "error")


async def test_a_content_filter_hit_is_one_warning_line_and_its_own_sentence(caplog):
    with caplog.at_level(logging.WARNING):
        events, stopped_reason = await _run(ContentFiltered("prompt", ("jailbreak",)))

    assert stopped_reason == ["content_filter"]
    [error] = [e for e in events if e.type == "error"]
    assert (error.text, error.stopped_reason) == (CONTENT_FILTERED_TEXT, "content_filter")
    [record] = [r for r in caplog.records if r.name == "harness.api.main"]
    assert record.levelno == logging.WARNING
    assert record.exc_info is None  # expected, so no traceback
    message = record.getMessage()
    assert "jailbreak" in message and "t1" in message and "r1" in message


async def test_a_cancelled_run_still_says_cancelled():
    # Review Focus 4: the restructured except clauses keep cancellation apart.
    queue: asyncio.Queue = asyncio.Queue()
    stopped_reason = [""]

    with pytest.raises(asyncio.CancelledError):
        await _run_graph(
            graph=_RaisingGraph(asyncio.CancelledError()),
            invoke_arg={},
            config={},
            thread_id="t1",
            run_id="r1",
            event_queue=queue,
            accumulator=None,
            stopped_reason_holder=stopped_reason,
            run_store=_NullRunStore(),
        )

    assert stopped_reason == ["cancelled"]
    first = queue.get_nowait()
    assert (first.type, first.text) == ("error", "cancelled")
