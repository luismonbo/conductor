"""Agent protocol. The seam that lets the eval harness drive a real agent
implementation (GraphAgentAdapter, wrapping the LangGraph graph that serves
live chat) without depending on its internals."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from harness.core.types import AgentResult, AgentState


@runtime_checkable
class Agent(Protocol):
    async def run(self, state: AgentState) -> AgentResult: ...
