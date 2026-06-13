"""A tool that exposes long-term memory search to the agent.

This is the bridge between the two memory concepts: short-term history is fed
to the model automatically every turn, but long-term recall is something the
agent *chooses* to do via this tool. Wrapping LongTermMemory as a Tool keeps
the agent loop unaware of pgvector vs markdown — it just calls 'recall'.
"""
from __future__ import annotations

from typing import Any

from harness.core.memory.store import LongTermMemory
from harness.core.types import ToolSpec


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
                    "k": {"type": "integer", "description": "Max results", "default": 3},
                },
                "required": ["query"],
            },
        )

    async def run(self, arguments: dict[str, Any]) -> str:
        hits = await self._memory.search(
            arguments["query"], k=int(arguments.get("k", 3))
        )
        if not hits:
            return "No relevant memories found."
        return "\n".join(f"- ({h.score:.2f}) {h.text}" for h in hits)
