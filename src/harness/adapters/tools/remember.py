"""A tool that lets the agent persist facts to long-term memory.

HITL is handled by the graph's _approval_gate node (requires_approval=True),
which interrupts before execution so the user can approve, deny, or give
feedback. This tool only runs after the user approves.
"""
from __future__ import annotations

from typing import Any

from harness.core.memory.store import LongTermMemory
from harness.core.types import ToolSpec


class RememberTool:
    requires_approval = True

    def __init__(self, memory: LongTermMemory) -> None:
        self._memory = memory

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="remember",
            description=(
                "Save a fact about the user to long-term memory — their name, "
                "location, job title, preferences, or ongoing projects. Call "
                "this whenever the user shares personal information that should "
                "be recalled in future conversations."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The fact to store, written as a clear statement",
                    },
                },
                "required": ["text"],
            },
        )

    async def run(self, arguments: dict[str, Any]) -> str:
        memory_id = await self._memory.write(arguments["text"])
        return f"Remembered (id={memory_id})."
