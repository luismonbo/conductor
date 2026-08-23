"""Integration test for the composition evaluation/run_eval.py's agent_factory
performs: build_long_term() + build_agent_registry() feeding a GraphAgentAdapter,
driven by a scripted FakeLLMClient via the same build_llm monkeypatch seam
tests/integration/conftest.py's client_with_fake fixture uses for api/main.py's
identical call to build_agent_registry.

tests/unit/test_graph_agent_adapter.py only exercises GraphAgentAdapter against
a hand-built graph with a one-tool registry. Nothing previously drove the real
multi-tool registry build_agent_registry() actually produces — the composition
this whole migration exists to wire up.
"""
from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import MemorySaver

import harness.orchestration.build as build_module
from harness.adapters.llm.fake import FakeLLMClient
from harness.config.settings import Settings
from harness.core.types import AgentState, LLMResponse, Message, Role, ToolCall
from harness.observability.tracer import TraceCollector
from harness.orchestration.build import build_agent_registry, build_long_term

from evaluation.harness.graph_agent import GraphAgentAdapter


@pytest.mark.asyncio
async def test_agent_factory_composition_calls_tool_then_answers(monkeypatch):
    """Replicates run_eval.py's agent_factory (lines ~79-84) call for call:
    build_long_term(settings) -> build_agent_registry(settings, checkpointer,
    long_term=memory) -> GraphAgentAdapter(registry["default"], tracer=tracer).
    Only the LLM is faked (via the build_llm monkeypatch seam); the registry,
    tools, and graph are all real.
    """
    settings = Settings(
        _env_file=None,
        llm_backend="fake",
        checkpointer="memory",
        embedding_backend="fake",
        memory_backend="in_memory",
        rag_vector_store_backend="in_memory",
    )
    fake = FakeLLMClient([
        LLMResponse(
            text="",
            tool_calls=(ToolCall(id="tc_1", name="calculator", arguments={"expression": "6*7"}),),
        ),
        LLMResponse(text="The answer is 42."),
    ])
    monkeypatch.setattr(build_module, "build_llm", lambda settings, parser: fake)

    memory = build_long_term(settings)
    registry = build_agent_registry(settings, MemorySaver(), long_term=memory)
    tracer = TraceCollector()
    adapter = GraphAgentAdapter(registry["default"], tracer=tracer)

    result = await adapter.run(
        AgentState(messages=[Message(Role.USER, "what is 6*7?")])
    )

    assert result.output == "The answer is 42."
    assert result.stopped_reason == "final_answer"

    tool_results = [d for _, e, d in tracer.events if e == "tool_result"]
    assert tool_results == [{"name": "calculator", "is_error": False, "content": "42"}]
