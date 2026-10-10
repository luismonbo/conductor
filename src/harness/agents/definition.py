"""The contract every agent folder implements.

An agent is a folder src/harness/agents/<name>/ whose agent.py defines AGENT,
an AgentDefinition. orchestration.agents.load_agent imports it by name, and
orchestration.build compiles it on the shared loop (orchestration/loop.py).
See docs/superpowers/specs/2026-10-08-agent-architecture-design.md.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from harness.config.settings import Settings
from harness.core.memory.store import LongTermMemory
from harness.core.rag.ports import Retriever, VectorStore
from harness.core.tools.registry import ToolRegistry


@dataclass(frozen=True)
class ToolDeps:
    """Shared infrastructure an agent's tools may use, built once by orchestration.build."""

    settings: Settings
    long_term: LongTermMemory
    retriever: Retriever | None  # None when the RAG backend is unavailable
    vector_store: VectorStore | None
    collections: tuple[str, ...]
    # Why RAG is unavailable: an agent that cannot work without it raises from
    # its tools factory with this chained, so startup shows the real cause.
    rag_error: BaseException | None = None


@dataclass(frozen=True)
class AgentDefinition:
    name: str  # must equal the agent's folder name
    prompt: str
    tools: Callable[[ToolDeps], ToolRegistry]
    # Pin: a LiteLLM profile name, or an Azure deployment on the azure backend.
    model: str | None = None
