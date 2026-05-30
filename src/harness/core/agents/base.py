"""Agent protocol. A single ReAct agent and (later) a supervisor both satisfy
this, so the orchestration graph can treat them uniformly as nodes."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from harness.core.types import AgentResult, AgentState


@runtime_checkable
class Agent(Protocol):
    async def run(self, state: AgentState) -> AgentResult: ...
