"""Composition root.

The ONE place that knows about concrete adapters. It reads config, builds the
selected LLM client, memory store, and tools, registers them, and returns a
ready ReActAgent. Everything else depends only on protocols. Adding a backend
means editing this file and nothing in core/.
"""
from __future__ import annotations

from harness.adapters.llm.parsers import NativeToolCallParser, PromptedToolCallParser
from harness.adapters.memory.in_memory import InMemoryLongTerm
from harness.adapters.tools.calculator import CalculatorTool
from harness.adapters.tools.recall import RecallTool
from harness.config.settings import Settings
from harness.core.agents.react import ReActAgent
from harness.core.llm.client import LLMClient
from harness.core.llm.tool_parsing import ToolCallParser
from harness.core.memory.store import LongTermMemory
from harness.core.tools.registry import ToolRegistry


def build_parser(settings: Settings) -> ToolCallParser:
    if settings.tool_parser == "prompted":
        return PromptedToolCallParser()
    return NativeToolCallParser()


def build_llm(settings: Settings, parser: ToolCallParser) -> LLMClient:
    if settings.llm_backend == "azure":
        from harness.adapters.llm.azure_openai import AzureOpenAIClient

        return AzureOpenAIClient(
            deployment=settings.azure_deployment,
            endpoint=settings.azure_endpoint,
            api_version=settings.azure_api_version,
            parser=parser,
            api_key=settings.azure_api_key or None,
        )
    if settings.llm_backend == "openai_compatible":
        from harness.adapters.llm.openai_compatible import OpenAICompatibleClient

        return OpenAICompatibleClient(
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            parser=parser,
            api_key=settings.llm_api_key,
        )
    if settings.llm_backend == "fake":
        # Scripted in tests; here we return a trivial echo so the app boots.
        from harness.adapters.llm.fake import FakeLLMClient
        from harness.core.types import LLMResponse

        return FakeLLMClient(
            [LLMResponse(text="Fake backend is active. Set HARNESS_LLM_BACKEND=azure.")]
        )
    raise ValueError(f"Unknown llm_backend: {settings.llm_backend}")


def build_long_term(settings: Settings) -> LongTermMemory:
    if settings.memory_backend == "pgvector":
        raise NotImplementedError(
            "Wire PgVectorLongTerm with an embedder here (Phase 5)."
        )
    return InMemoryLongTerm()


def build_agent(
    settings: Settings,
    tracer=None,
    long_term: LongTermMemory | None = None,
) -> ReActAgent:
    parser = build_parser(settings)
    llm = build_llm(settings, parser)
    memory = long_term if long_term is not None else build_long_term(settings)

    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(RecallTool(memory))

    return ReActAgent(
        llm=llm,
        tools=registry,
        system_prompt=settings.system_prompt,
        tracer=tracer,
    )


def build_agent_registry(settings: Settings, checkpointer) -> dict[str, object]:
    """Build and return all compiled agent graphs keyed by name.

    The API routes to the agent named in ChatRequest.agent (default: settings.agent).
    Adding a new agent means adding it here and in agents/<name>/.
    """
    from harness.agents.default.graph import build_graph as build_default_graph

    llm = build_llm(settings, build_parser(settings))
    return {
        "default": build_default_graph(llm, checkpointer),
    }
