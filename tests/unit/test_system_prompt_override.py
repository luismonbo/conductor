"""Settings.system_prompt is live: it overrides the deployment agent's prompt, and only that."""
from __future__ import annotations

import asyncio

from langgraph.checkpoint.memory import MemorySaver

import harness.orchestration.build as build_module
from harness.adapters.llm.fake import FakeLLMClient
from harness.agents.default.graph import build_graph
from harness.agents.default.prompt import SYSTEM_PROMPT
from harness.config.settings import Settings
from harness.core.types import LLMResponse, Message, Role
from harness.orchestration.build import build_agent_registry, resolve_system_prompt


def _settings(**overrides) -> Settings:
    base = dict(
        _env_file=None,
        api_key="test-key",
        input_guard_enabled=False,
        llm_backend="fake",
        checkpointer="memory",
        embedding_backend="fake",
        rag_vector_store_backend="in_memory",
        agent="default",
    )
    return Settings(**{**base, **overrides})


def _input():
    return {"messages": [Message(role=Role.USER, content="hi")], "iteration": 0, "max_iterations": 3}


def _config():
    return {"configurable": {"thread_id": "t", "event_queue": asyncio.Queue()}}


def test_an_empty_override_uses_the_agents_own_prompt():
    assert resolve_system_prompt(_settings(system_prompt=""), "default") == SYSTEM_PROMPT


def test_the_override_applies_to_the_deployment_agent():
    s = _settings(system_prompt="You answer questions about X.")
    assert resolve_system_prompt(s, "default") == "You answer questions about X."


def test_the_override_does_not_leak_into_other_agents():
    s = _settings(system_prompt="You answer questions about X.")
    assert resolve_system_prompt(s, "some_other_agent") == SYSTEM_PROMPT


async def test_the_graph_sends_the_system_prompt_it_was_built_with():
    fake = FakeLLMClient([LLMResponse(text="ok")])
    graph = build_graph(llm=fake, checkpointer=MemorySaver(), system_prompt="Custom prompt.")
    await graph.ainvoke(_input(), _config())
    assert fake.calls[0][0] == Message(role=Role.SYSTEM, content="Custom prompt.")


async def test_the_registry_builds_the_default_agent_with_the_override(monkeypatch):
    fake = FakeLLMClient([LLMResponse(text="ok")])
    monkeypatch.setattr(build_module, "build_llm", lambda settings, parser: fake)
    registry = build_agent_registry(_settings(system_prompt="Scoped prompt."), MemorySaver())
    await registry["default"].ainvoke(_input(), _config())
    assert fake.calls[0][0].content == "Scoped prompt."
