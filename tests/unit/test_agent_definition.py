"""The agent contract (AgentDefinition, ToolDeps) and the default agent's definition."""
from __future__ import annotations

import dataclasses

import pytest

from harness.adapters.embedding.fake import FakeEmbedder
from harness.adapters.memory.in_memory import InMemoryLongTerm
from harness.adapters.vectorstore.in_memory import InMemoryVectorStore
from harness.agents.default.agent import AGENT
from harness.agents.default.prompt import SYSTEM_PROMPT
from harness.agents.definition import ToolDeps
from harness.config.settings import Settings
from harness.core.rag.serve import Retriever


def _deps(settings_overrides: dict | None = None, **fields) -> ToolDeps:
    settings = Settings(
        _env_file=None,
        api_key="test-key",
        input_guard_enabled=False,
        **(settings_overrides or {}),
    )
    base = dict(
        settings=settings,
        long_term=InMemoryLongTerm(),
        retriever=None,
        vector_store=None,
        collections=(),
    )
    return ToolDeps(**{**base, **fields})


def test_the_default_agent_is_named_after_its_folder_and_uses_its_prompt():
    assert AGENT.name == "default"
    assert AGENT.prompt == SYSTEM_PROMPT
    assert AGENT.model is None


def test_the_default_agents_tools_come_from_the_shared_deps():
    names = AGENT.tools(_deps()).names()
    assert "calculator" in names
    assert "search_documents" not in names  # no retriever in these deps


def test_memory_tools_follow_the_deployment_setting():
    off = AGENT.tools(_deps({"memory_tools_enabled": False})).names()
    on = AGENT.tools(_deps({"memory_tools_enabled": True})).names()
    assert {"remember", "recall"}.isdisjoint(off)
    assert {"remember", "recall"} <= set(on)


def test_search_documents_is_offered_when_rag_is_available():
    store = InMemoryVectorStore()
    retriever = Retriever(embedder=FakeEmbedder(dimension=4), vector_store=store)
    deps = _deps(retriever=retriever, vector_store=store, collections=("papers",))
    assert "search_documents" in AGENT.tools(deps).names()


def test_definitions_are_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        AGENT.prompt = "changed"  # type: ignore[misc]
