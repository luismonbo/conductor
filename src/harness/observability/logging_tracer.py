"""Renders ingestion trace events to stdlib logging for the offline pipeline —
the operator's window into a container batch run. Same async (event, data)
contract as TraceCollector, so it plugs into IngestionPipeline(tracer=...)."""
from __future__ import annotations

import logging
from typing import Any

_WARN_EVENTS = {"ingest_file_failed"}


class LoggingTracer:
    def __init__(self, logger_name: str = "harness.ingest") -> None:
        self._log = logging.getLogger(logger_name)

    async def __call__(self, event: str, data: dict[str, Any]) -> None:
        level = logging.WARNING if event in _WARN_EVENTS else logging.INFO
        if event == "ingest_stage":
            self._log.log(level, "%s: %s (%s) [%s]", event, data.get("stage"),
                          data.get("count"), data.get("source_path"))
        else:
            self._log.log(level, "%s: %s", event, data)
