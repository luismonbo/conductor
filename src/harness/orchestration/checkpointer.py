"""Checkpointer factory.

Selects and constructs the right LangGraph checkpointer backend based on
HARNESS_CHECKPOINTER. MemorySaver is used in tests; SqliteSaver is the
local default. PostgresSaver is wired in Phase 5.
"""
from __future__ import annotations

from harness.config.settings import Settings


def build_checkpointer(settings: Settings):
    """Return a LangGraph checkpointer instance for the configured backend."""
    backend = settings.checkpointer

    if backend == "memory":
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver()

    if backend == "sqlite":
        import sqlite3
        from langgraph.checkpoint.sqlite import SqliteSaver
        conn = sqlite3.connect(settings.checkpointer_url, check_same_thread=False)
        return SqliteSaver(conn)

    if backend == "postgres":
        raise NotImplementedError(
            "PostgresSaver wiring is deferred to Phase 5. "
            "Set HARNESS_CHECKPOINTER=sqlite or HARNESS_CHECKPOINTER=memory."
        )

    raise ValueError(
        f"Unknown checkpointer backend: {backend!r}. "
        "Valid values: memory | sqlite | postgres."
    )
