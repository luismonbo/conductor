"""Composition root.

The ONE place that knows about concrete adapters. It reads config, builds the
selected LLM client, memory store, and tools, registers them, and returns a
ready ReActAgent. Everything else depends only on protocols. Adding a backend
means editing this file and nothing in core/.
"""
from __future__ import annotations

from pathlib import Path

import yaml

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
from harness.core.rag.ingest import IngestionPipeline
from harness.core.rag.ports import Embedder, VectorStore
from harness.core.rag.serve import DiversifiedRetriever, RagPipeline, Retriever
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
            model=settings.default_model or settings.llm_model,
            parser=parser,
            api_key=settings.llm_api_key,
        )
    if settings.llm_backend == "fake":
        # Scripted in tests; here we return a trivial echo so the app boots.
        # repeat_last keeps the served fake answering every message (demo/e2e).
        from harness.adapters.llm.fake import FakeLLMClient
        from harness.core.types import LLMResponse

        return FakeLLMClient(
            [LLMResponse(text="Fake backend is active. Set HARNESS_LLM_BACKEND=azure.")],
            repeat_last=True,
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
        from harness.adapters.embedding.azure_openai import AzureOpenAIEmbedder

        return AzureOpenAIEmbedder(
            deployment=settings.azure_embedding_deployment,
            endpoint=settings.azure_endpoint,
            api_version=settings.azure_api_version,
            api_key=settings.azure_api_key or None,
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


def build_retriever(settings: Settings, vector_store: VectorStore) -> Retriever:
    embedder = build_embedder(settings)
    retriever = Retriever(embedder=embedder, vector_store=vector_store)
    if settings.rag_per_document_k > 0:
        retriever = DiversifiedRetriever(
            retriever,
            per_document_k=settings.rag_per_document_k,
            overfetch=settings.rag_overfetch,
        )
    return retriever


def list_collections(index_config_dir: Path = Path("data/index_config")) -> list[str]:
    """Discover ingested collection names from cli/ingest.py's manifest files.

    No hand-maintained enum: cli/ingest.py already writes/updates
    data/index_config/<collection>.yaml on every ingest run, so this stays
    accurate as of the next process restart with zero extra bookkeeping.
    """
    if not index_config_dir.is_dir():
        return []
    names: list[str] = []
    for manifest_path in sorted(index_config_dir.glob("*.yaml")):
        manifest = yaml.safe_load(manifest_path.read_text())
        if manifest and "collection" in manifest:
            names.append(manifest["collection"])
    return names


def build_parser_router():
    # Imported lazily: docling pulls torch + transformers, and this module is
    # imported by the API and most of the test suite, which never parse a document.
    from harness.adapters.parsing.markitdown_parser import MarkitdownParser
    from harness.adapters.parsing.router import DOCLING_EXTENSIONS, ParserRouter

    markitdown = MarkitdownParser()
    if DOCLING_EXTENSIONS:
        from harness.adapters.parsing.docling_parser import DoclingParser

        docling = DoclingParser()
    else:
        # Skip importing docling/torch entirely while DOCLING_EXTENSIONS is
        # empty — merely having torch in the process is enough to trigger the
        # libomp crash it's disabled for (see router.py), so it's not enough
        # to just avoid calling it. Never routed to either way; markitdown
        # here is an inert placeholder for the unused slot.
        docling = markitdown
    return ParserRouter(docling=docling, markitdown=markitdown)


def build_ingestion_pipeline(
    settings: Settings, vector_store_backends: list[str], tracer=None
) -> IngestionPipeline:
    return IngestionPipeline(
        parser=build_parser_router(),
        normalizer=LlmNormalizer(build_llm(settings, build_parser(settings))),
        chunker=StructureAwareChunker(),
        embedder=build_embedder(settings),
        vector_stores=[build_vector_store(settings, backend) for backend in vector_store_backends],
        tracer=tracer,
    )


def build_rag_pipeline(
    settings: Settings, vector_store_backend: str, tracer=None
) -> RagPipeline:
    vector_store = build_vector_store(settings, vector_store_backend)
    retriever = build_retriever(settings, vector_store)
    llm = build_llm(settings, build_parser(settings))
    return RagPipeline(retriever=retriever, llm=llm, tracer=tracer)


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
