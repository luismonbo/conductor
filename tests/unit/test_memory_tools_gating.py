"""remember/recall are opt-in: a public deployment must not expose shared memory."""
import asyncio

from langgraph.checkpoint.memory import MemorySaver

import harness.orchestration.build as build_module
from harness.adapters.llm.fake import FakeLLMClient
from harness.agents.default.tools import build_registry
from harness.config.settings import Settings
from harness.core.types import LLMResponse, Message, Role


def _tool_names(memory_tools: bool) -> set[str]:
    registry = build_registry(memory_tools=memory_tools)
    return {spec.name for spec in registry.specs()}


def test_memory_tools_absent_by_default():
    names = _tool_names(memory_tools=False)
    assert "remember" not in names
    assert "recall" not in names


def test_calculator_survives_with_memory_disabled():
    assert "calculator" in _tool_names(memory_tools=False)


def test_memory_tools_present_when_enabled():
    names = _tool_names(memory_tools=True)
    assert "remember" in names
    assert "recall" in names


# The tests above call build_registry directly, so they cannot see the setting's
# default or how build_agent_registry passes it through. These go through the
# real wiring and record what the model is actually offered.


class _ToolRecordingLLM(FakeLLMClient):
    def __init__(self) -> None:
        super().__init__([LLMResponse(text="ok")], repeat_last=True)
        self.offered: list[set[str]] = []

    async def stream(self, messages, tools=None, model=None):
        self.offered.append({t.name for t in (tools or [])})
        async for chunk in super().stream(messages, tools=tools, model=model):
            yield chunk


async def _offered_tools(monkeypatch, **overrides) -> set[str]:
    llm = _ToolRecordingLLM()
    monkeypatch.setattr(build_module, "build_llm", lambda settings, parser: llm)
    settings = Settings(
        _env_file=None,
        api_key="x",
        rag_vector_store_backend="in_memory",
        # Pinned so the test never depends on a memory backend a local .env
        # put into os.environ (main.py's load_dotenv copies it there).
        memory_backend="in_memory",
        **overrides,
    )
    graph = build_module.build_agent_registry(settings, MemorySaver())["default"]
    await graph.ainvoke(
        {"messages": [Message(role=Role.USER, content="hi")], "iteration": 0, "max_iterations": 2},
        {"configurable": {"thread_id": "t", "event_queue": asyncio.Queue()}},
    )
    return llm.offered[0]


async def test_model_is_not_offered_memory_tools_by_default(monkeypatch):
    # .env / load_dotenv may have put the opt-in into os.environ; the public
    # default is what is under test, so remove it.
    monkeypatch.delenv("HARNESS_MEMORY_TOOLS_ENABLED", raising=False)
    offered = await _offered_tools(monkeypatch)
    assert {"remember", "recall"}.isdisjoint(offered)
    assert "calculator" in offered


async def test_model_is_offered_memory_tools_when_enabled(monkeypatch):
    offered = await _offered_tools(monkeypatch, memory_tools_enabled=True)
    assert {"remember", "recall"} <= offered
