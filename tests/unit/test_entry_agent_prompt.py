"""The entry agent's prompt comes from its folder, never from config."""
from __future__ import annotations

import asyncio

import pytest
from langgraph.checkpoint.memory import MemorySaver

from harness.adapters.llm.fake import FakeLLMClient
from harness.agents.default.prompt import SYSTEM_PROMPT
from harness.config.settings import Settings
from harness.core.tools.registry import ToolRegistry
from harness.core.types import LLMResponse, Message, Role
from harness.orchestration.agents import AgentLoadError
from harness.orchestration.build import resolve_system_prompt
from harness.orchestration.loop import build_graph


def _settings(**overrides) -> Settings:
    base = dict(_env_file=None, api_key="test-key", input_guard_enabled=False, agent="default")
    return Settings(**{**base, **overrides})


def test_the_default_entry_agent_uses_its_own_prompt():
    assert resolve_system_prompt(_settings()) == SYSTEM_PROMPT


def test_the_prompt_follows_harness_agent(fixture_agents):
    prompt = resolve_system_prompt(_settings(agent="good"))
    assert prompt == "You are the good fixture agent."


def test_an_unknown_entry_agent_fails_loudly():
    with pytest.raises(AgentLoadError, match="no_such_agent"):
        resolve_system_prompt(_settings(agent="no_such_agent"))


async def test_the_loop_sends_the_prompt_it_was_built_with():
    fake = FakeLLMClient([LLMResponse(text="ok")])
    graph = build_graph(
        fake, MemorySaver(), registry=ToolRegistry(), system_prompt="Custom prompt."
    )
    await graph.ainvoke(
        {"messages": [Message(role=Role.USER, content="hi")], "iteration": 0, "max_iterations": 3},
        {"configurable": {"thread_id": "t", "event_queue": asyncio.Queue()}},
    )
    assert fake.calls[0][0] == Message(role=Role.SYSTEM, content="Custom prompt.")
