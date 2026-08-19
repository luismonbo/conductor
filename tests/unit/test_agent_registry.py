"""Agent registry unit tests."""
from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import MemorySaver

from harness.config.settings import Settings
from harness.orchestration.build import build_agent_registry


@pytest.mark.unit
def test_registry_contains_default_key():
    s = Settings(
        _env_file=None,
        llm_backend="fake",
        checkpointer="memory",
        embedding_backend="fake",
        rag_vector_store_backend="in_memory",
    )
    registry = build_agent_registry(s, MemorySaver())
    assert "default" in registry


@pytest.mark.unit
def test_registry_default_is_compiled_graph():
    """The value must be a compiled LangGraph graph (has ainvoke method)."""
    s = Settings(
        _env_file=None,
        llm_backend="fake",
        checkpointer="memory",
        embedding_backend="fake",
        rag_vector_store_backend="in_memory",
    )
    registry = build_agent_registry(s, MemorySaver())
    graph = registry["default"]
    assert callable(getattr(graph, "ainvoke", None))
