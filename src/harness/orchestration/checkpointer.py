"""Checkpointer factory.

Selects and constructs the right LangGraph checkpointer backend based on
HARNESS_CHECKPOINTER. MemorySaver is used in tests; AsyncSqliteSaver is the
local default. PostgresSaver is wired in Phase 5.
"""
from __future__ import annotations

from harness.config.settings import Settings


async def build_checkpointer(settings: Settings):
    """Return a LangGraph checkpointer for the configured backend.

    Async because AsyncSqliteSaver requires an event loop at construction time.
    """
    backend = settings.checkpointer

    if backend == "memory":
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver()

    if backend == "sqlite":
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        conn = await aiosqlite.connect(settings.checkpointer_url)
        return AsyncSqliteSaver(conn)

    if backend == "postgres":
        raise NotImplementedError(
            "PostgresSaver wiring is deferred to Phase 5. "
            "Set HARNESS_CHECKPOINTER=sqlite or HARNESS_CHECKPOINTER=memory."
        )

    raise ValueError(
        f"Unknown checkpointer backend: {backend!r}. "
        "Valid values: memory | sqlite | postgres."
    )
