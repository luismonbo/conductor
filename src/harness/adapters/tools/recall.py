"""A tool that exposes long-term memory search to the agent.

This is the bridge between the two memory concepts: short-term history is fed
to the model automatically every turn, but long-term recall is something the
agent *chooses* to do via this tool. Wrapping LongTermMemory as a Tool keeps
the agent loop unaware of pgvector vs markdown — it just calls 'recall'.
"""
from __future__ import annotations

from typing import Any

from harness.core.memory.store import LongTermMemory
from harness.core.tools.base import ToolError
from harness.core.types import ToolSpec

_DEFAULT_K = 3


class RecallTool:
    def __init__(self, memory: LongTermMemory) -> None:
        self._memory = memory

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="recall",
            description=(
                "Search long-term memory for personal facts about the user — "
                "their name, location, job title, preferences, and ongoing "
                "projects. Call this tool whenever the user asks about "
                "themselves or their personal context. Never guess."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to recall"},
                    "k": {"type": "integer", "description": "Max results", "default": _DEFAULT_K},
                },
                "required": ["query"],
            },
        )

    async def run(self, arguments: dict[str, Any]) -> str:
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ToolError("query is required: say what to recall")
        hits = await self._memory.search(query, k=_k(arguments.get("k")))
        if not hits:
            return "No relevant memories found."
        return "\n".join(f"- ({h.score:.2f}) {h.text}" for h in hits)


def _k(value: Any) -> int:
    # Missing or null means the default: small models send null for an optional parameter.
    if value is None:
        return _DEFAULT_K
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ToolError("k must be a whole number") from exc
