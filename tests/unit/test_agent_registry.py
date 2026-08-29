"""Agent registry unit tests."""
from __future__ import annotations

from unittest import mock

import pytest
from langgraph.checkpoint.memory import MemorySaver

import harness.agents.default.tools as default_tools
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
        api_key="test-key",
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
        api_key="test-key",
    )
    registry = build_agent_registry(s, MemorySaver())
    graph = registry["default"]
    assert callable(getattr(graph, "ainvoke", None))


@pytest.mark.unit
def test_registry_wires_search_documents_when_rag_backend_available():
    s = Settings(
        _env_file=None,
        llm_backend="fake",
        checkpointer="memory",
        embedding_backend="fake",
        rag_vector_store_backend="in_memory",
        api_key="test-key",
    )

    with mock.patch(
        "harness.agents.default.tools.build_registry",
        wraps=default_tools.build_registry,
    ) as spy:
        build_agent_registry(s, MemorySaver())

    _, kwargs = spy.call_args
    assert kwargs.get("retriever") is not None
    assert kwargs.get("vector_store") is not None


@pytest.mark.unit
def test_registry_degrades_gracefully_when_rag_backend_unavailable():
    s = Settings(
        _env_file=None,
        llm_backend="fake",
        checkpointer="memory",
        embedding_backend="fake",
        rag_vector_store_backend="not_a_real_backend",
        api_key="test-key",
    )

    # Should not raise — falls back to a registry with no search_documents tool.
    registry = build_agent_registry(s, MemorySaver())
    assert "default" in registry


@pytest.mark.unit
def test_registry_uses_the_passed_in_long_term_when_given():
    s = Settings(
        _env_file=None,
        llm_backend="fake",
        checkpointer="memory",
        embedding_backend="fake",
        rag_vector_store_backend="in_memory",
        api_key="test-key",
    )
    from harness.adapters.memory.in_memory import InMemoryLongTerm
    memory = InMemoryLongTerm()

    with mock.patch(
        "harness.agents.default.tools.build_registry",
        wraps=default_tools.build_registry,
    ) as spy:
        build_agent_registry(s, MemorySaver(), long_term=memory)

    _, kwargs = spy.call_args
    assert kwargs.get("long_term") is memory
