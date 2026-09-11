"""_run_graph must log the real exception even though the client only ever
sees the friendlied text from _friendly_error() — see test_friendly_error.py
for the message-mapping cases themselves."""
import asyncio
import logging

from harness.api.main import _run_graph


class _RaisingGraph:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def ainvoke(self, invoke_arg, config):
        raise self._exc


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
        run_store=None,
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


async def test_run_graph_still_emits_friendlied_text_to_the_client(caplog):
    class PermissionDeniedError(Exception):
        pass

    with caplog.at_level(logging.ERROR):
        events, stopped_reason = await _run(
            PermissionDeniedError("403 quota exhausted, deployment=prod-gpt5")
        )

    assert stopped_reason == ["error"]
    error_events = [e for e in events if e.type == "error"]
    assert len(error_events) == 1
    assert error_events[0].text == "The assistant is temporarily unavailable. Please try again later."
