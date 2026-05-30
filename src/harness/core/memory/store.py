"""Memory protocols.

ShortTermMemory  = per-conversation transcript (backed by the LangGraph
                   checkpointer in the orchestration layer, or a fake in tests).
LongTermMemory   = cross-conversation recall. First impl is a Qdrant vector
                   store; a MarkdownStore impl can follow behind the SAME
                   interface, so 'vector vs markdown' is a config choice, not
                   a rewrite. Both must pass tests/contract/test_long_term_memory.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from harness.core.types import Message


@runtime_checkable
class ShortTermMemory(Protocol):
    async def append(self, conversation_id: str, message: Message) -> None: ...
    async def history(self, conversation_id: str) -> list[Message]: ...
    async def clear(self, conversation_id: str) -> None: ...


@dataclass(frozen=True)
class MemoryHit:
    text: str
    score: float
    metadata: dict[str, str]


@runtime_checkable
class LongTermMemory(Protocol):
    async def write(self, text: str, metadata: dict[str, str] | None = None) -> str:
        """Persist a memory; returns its id."""
        ...

    async def search(self, query: str, k: int = 5) -> list[MemoryHit]: ...

    async def update(self, memory_id: str, text: str) -> None: ...
