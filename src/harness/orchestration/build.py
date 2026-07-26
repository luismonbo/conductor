"""Composition root.

The ONE place that knows about concrete adapters. It reads config, builds the
selected LLM client, memory store, and tools, registers them, and returns a
ready ReActAgent. Everything else depends only on protocols. Adding a backend
means editing this file and nothing in core/.
"""
from __future__ import annotations

from harness.adapters.chunking.structure_aware import StructureAwareChunker
from harness.adapters.embedding.fake import FakeEmbedder
from harness.adapters.llm.parsers import NativeToolCallParser, PromptedToolCallParser
from harness.adapters.memory.in_memory import InMemoryLongTerm
from harness.adapters.normalization.llm_normalizer import LlmNormalizer
from harness.adapters.tools.calculator import CalculatorTool
from harness.adapters.tools.recall import RecallTool
from harness.config.settings import Settings
from harness.core.agents.react import ReActAgent
from harness.core.llm.client import LLMClient
from harness.core.llm.tool_parsing import ToolCallParser
from harness.core.memory.store import LongTermMemory
from harness.core.rag.ports import Embedder, VectorStore
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
    if settings.memory_backend == "sqlite":
        from harness.adapters.memory.sqlite_store import SqliteLongTermMemory

        path = settings.memory_url or "./harness_memory.sqlite"
        return SqliteLongTermMemory(path)
    if settings.memory_backend == "pgvector":
        raise NotImplementedError(
            "Wire PgVectorLongTerm with an embedder here (Phase 5)."
        )
    return InMemoryLongTerm()


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_backend == "openai_compatible":
        from harness.adapters.embedding.openai_compatible import OpenAICompatibleEmbedder

        return OpenAICompatibleEmbedder(
            base_url=settings.embedding_base_url,
            model=settings.embedding_model,
            api_key=settings.embedding_api_key,
        )
    if settings.embedding_backend == "azure":
        raise NotImplementedError(
            "Azure embeddings are not wired yet — use HARNESS_EMBEDDING_BACKEND=openai_compatible "
            "or fake. See the RAG design doc's Explicitly Out of Scope section."
        )
    if settings.embedding_backend == "fake":
        return FakeEmbedder(dimension=settings.embedding_dimension)
    raise ValueError(f"Unknown embedding_backend: {settings.embedding_backend}")


def build_vector_store(settings: Settings, backend: str) -> VectorStore:
    if backend == "pgvector":
        from harness.adapters.vectorstore.pgvector_store import PgVectorStore

        return PgVectorStore(
            dsn=settings.pgvector_url,
            table=settings.pgvector_table,
            vector_size=settings.embedding_dimension,
        )
    if backend == "milvus":
        from harness.adapters.vectorstore.milvus_store import MilvusStore

        return MilvusStore(
            uri=settings.milvus_uri,
            collection=settings.milvus_collection,
            vector_size=settings.embedding_dimension,
        )
    if backend == "in_memory":
        from harness.adapters.vectorstore.in_memory import InMemoryVectorStore

        return InMemoryVectorStore()
    raise ValueError(f"Unknown vector store backend: {backend}")


def build_parser_router():
    # Imported lazily: docling pulls torch + transformers, and this module is
    # imported by the API and most of the test suite, which never parse a document.
    from harness.adapters.parsing.docling_parser import DoclingParser
    from harness.adapters.parsing.markitdown_parser import MarkitdownParser
    from harness.adapters.parsing.router import ParserRouter

    return ParserRouter(docling=DoclingParser(), markitdown=MarkitdownParser())


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
    from harness.agents.default.tools import build_registry

    llm = build_llm(settings, build_parser(settings))
    long_term = build_long_term(settings)
    registry = build_registry(long_term)
    return {
        "default": build_default_graph(llm, checkpointer, registry=registry),
    }
