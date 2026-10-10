"""build_agent: compile one agent from its folder's definition, on the shared loop."""
from __future__ import annotations

import asyncio

import pytest
from langgraph.checkpoint.memory import MemorySaver

import harness.orchestration.build as build_module
from harness.adapters.llm.fake import FakeLLMClient
from harness.adapters.memory.in_memory import InMemoryLongTerm
from harness.config.settings import Settings
from harness.core.types import LLMResponse, Message, Role
from harness.orchestration.build import build_tool_deps


def _settings(**overrides) -> Settings:
    base = dict(
        _env_file=None,
        api_key="test-key",
        input_guard_enabled=False,
        llm_backend="fake",
        checkpointer="memory",
        embedding_backend="fake",
        memory_backend="in_memory",
        rag_vector_store_backend="in_memory",
        memory_tools_enabled=False,
        agent="default",
    )
    return Settings(**{**base, **overrides})


class _RecordingLLM(FakeLLMClient):
    """Answers at once and records the tools each call offered."""

    def __init__(self) -> None:
        super().__init__([LLMResponse(text="ok")], repeat_last=True)
        self.offered: list[set[str]] = []

    async def stream(self, messages, tools=None, model=None):
        self.offered.append({t.name for t in (tools or [])})
        async for chunk in super().stream(messages, tools=tools, model=model):
            yield chunk


async def _run_once(monkeypatch, settings: Settings, **kwargs) -> _RecordingLLM:
    llm = _RecordingLLM()
    monkeypatch.setattr(build_module, "build_llm", lambda s, parser: llm)
    graph = build_module.build_agent(settings, MemorySaver(), **kwargs)
    await graph.ainvoke(
        {"messages": [Message(role=Role.USER, content="hi")], "iteration": 0, "max_iterations": 2},
        {"configurable": {"thread_id": "t", "event_queue": asyncio.Queue()}},
    )
    return llm


async def test_builds_the_entry_agent_from_its_definition(monkeypatch, fixture_agents):
    llm = await _run_once(monkeypatch, _settings(agent="good"))
    assert llm.calls[0][0] == Message(role=Role.SYSTEM, content="You are the good fixture agent.")
    assert llm.offered[0] == {"calculator"}
    assert llm.requested_models == ["fixture-model"]


async def test_a_name_builds_that_agent_instead_of_the_entry_agent(monkeypatch, fixture_agents):
    llm = await _run_once(monkeypatch, _settings(agent="default"), name="good")
    assert llm.calls[0][0].content == "You are the good fixture agent."


async def test_the_default_agent_gets_search_documents_when_rag_is_available(monkeypatch):
    llm = await _run_once(monkeypatch, _settings())
    assert {"calculator", "search_documents"} <= llm.offered[0]


async def test_an_unavailable_rag_backend_only_drops_search_documents(monkeypatch):
    llm = await _run_once(monkeypatch, _settings(rag_vector_store_backend="not_a_real_backend"))
    assert "search_documents" not in llm.offered[0]
    assert "calculator" in llm.offered[0]


def test_tool_deps_use_a_passed_in_long_term():
    memory = InMemoryLongTerm()
    assert build_tool_deps(_settings(), long_term=memory).long_term is memory


def test_tool_deps_record_why_rag_is_unavailable():
    deps = build_tool_deps(_settings(rag_vector_store_backend="not_a_real_backend"))
    assert deps.retriever is None
    assert isinstance(deps.rag_error, ValueError)


def test_an_agent_that_requires_rag_fails_with_the_backend_error(fixture_agents):
    settings = _settings(agent="needs_rag", rag_vector_store_backend="not_a_real_backend")
    with pytest.raises(RuntimeError, match="requires the RAG backend") as excinfo:
        build_module.build_agent(settings, MemorySaver())
    assert isinstance(excinfo.value.__cause__, ValueError)
    assert "not_a_real_backend" in str(excinfo.value.__cause__)
