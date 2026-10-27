"""Composition root.

The ONE place that knows about concrete adapters. It reads config, builds the
selected LLM client, memory store, and tools, registers them, and returns a
ready agent graph. Everything else depends only on protocols. Adding a backend
means editing this file and nothing in core/.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import httpx
import yaml

from harness.adapters.chunking.structure_aware import StructureAwareChunker
from harness.adapters.embedding.fake import FakeEmbedder
from harness.adapters.llm.parsers import NativeToolCallParser, PromptedToolCallParser
from harness.adapters.memory.in_memory import InMemoryLongTerm
from harness.adapters.wiki.azure_blob import AzureBlobWikiSource
from harness.adapters.wiki.local import LocalWikiSource
from harness.adapters.wiki.pages import build_wiki
from harness.agents.definition import AgentDefinition, ToolDeps
from harness.config.settings import Settings
from harness.core.llm.client import LLMClient
from harness.core.llm.tool_parsing import ToolCallParser
from harness.core.memory.store import LongTermMemory
from harness.core.rag.ingest import IngestionPipeline
from harness.core.rag.ports import Embedder, Retriever as RetrieverPort, VectorStore
from harness.core.rag.serve import DiversifiedRetriever, RagPipeline, Retriever
from harness.core.wiki import Wiki, WikiSource
from harness.orchestration.agents import load_agent
from harness.orchestration.storage import Storage, open_storage
from harness.security.input_guard import GuardLimits, InputGuard


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
            max_output_tokens=settings.max_output_tokens,
        )
    if settings.llm_backend == "openai_compatible":
        from harness.adapters.llm.openai_compatible import OpenAICompatibleClient

        return OpenAICompatibleClient(
            base_url=settings.llm_base_url,
            model=settings.default_model or settings.llm_model,
            parser=parser,
            api_key=settings.llm_api_key,
            max_output_tokens=settings.max_output_tokens,
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


def build_retriever(settings: Settings, vector_store: VectorStore) -> RetrieverPort:
    embedder = build_embedder(settings)
    plain = Retriever(embedder=embedder, vector_store=vector_store)
    if settings.rag_per_document_k > 0:
        return DiversifiedRetriever(
            plain,
            per_document_k=settings.rag_per_document_k,
            overfetch=settings.rag_overfetch,
        )
    return plain


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
    return sorted(dict.fromkeys(names))


def build_parser_router():
    # Deferred: most callers (the API, most of the test suite) never parse a
    # document, so there's no reason to pay markitdown's (or docling's) import
    # cost for them. DoclingParser() itself imports docling lazily inside
    # _convert, so constructing it here stays import-free too.
    from harness.adapters.parsing.docling_parser import DoclingParser
    from harness.adapters.parsing.markdown_passthrough import MarkdownPassthroughParser
    from harness.adapters.parsing.markitdown_parser import MarkitdownParser
    from harness.adapters.parsing.router import ParserRouter

    return ParserRouter(
        markitdown=MarkitdownParser(),
        markdown=MarkdownPassthroughParser(),
        docling=DoclingParser(),
    )


def build_ingestion_pipeline(
    settings: Settings, vector_store_backends: list[str], tracer=None
) -> IngestionPipeline:
    from harness.adapters.normalization.docling_normalizer import DoclingNormalizer
    from harness.adapters.normalization.markdown_normalizer import MarkdownNormalizer
    from harness.adapters.normalization.routing_normalizer import RoutingNormalizer

    return IngestionPipeline(
        parser=build_parser_router(),
        normalizer=RoutingNormalizer(
            markdown=MarkdownNormalizer(), docling=DoclingNormalizer()
        ),
        chunker=StructureAwareChunker(),
        embedder=build_embedder(settings),
        vector_stores=[
            build_vector_store(settings, backend) for backend in vector_store_backends
        ],
        tracer=tracer,
    )


def build_rag_pipeline(
    settings: Settings, vector_store_backend: str, tracer=None
) -> RagPipeline:
    vector_store = build_vector_store(settings, vector_store_backend)
    retriever = build_retriever(settings, vector_store)
    llm = build_llm(settings, build_parser(settings))
    # RagPipeline's constructor is nominally typed to the concrete Retriever
    # (see harness.core.rag.serve); build_retriever's return type is the
    # protocol (RetrieverPort) because it may also return a DiversifiedRetriever.
    # Both concrete classes satisfy the protocol structurally at runtime — this
    # cast only bridges the nominal/structural typing gap for mypy.
    return RagPipeline(retriever=cast(Retriever, retriever), llm=llm, tracer=tracer)


def resolve_system_prompt(settings: Settings) -> str:
    """The entry agent's prompt: the prompt.py of the agent settings.agent names.

    For callers outside the API (run_guard_eval.py), which take the prompt from
    the same definition the agent runs with. Prompts never come from config.
    """
    return load_agent(settings.agent).prompt


def build_wiki_source(settings: Settings) -> WikiSource | None:
    """The deployment's wiki source, or None when it has none (Settings allows one)."""
    if settings.wiki_blob_account_url:
        return AzureBlobWikiSource(
            settings.wiki_blob_account_url, settings.wiki_blob_container, settings.wiki_blob_prefix
        )
    if settings.wiki_dir:
        return LocalWikiSource(settings.wiki_dir)
    return None


def _load_wiki(settings: Settings) -> tuple[Wiki | None, BaseException | None]:
    """The wiki, or why it failed to load. No source configured is not a failure."""
    source = build_wiki_source(settings)
    if source is None:
        return None, None
    try:
        return build_wiki(source), None
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "Wiki unavailable; an agent that needs it will not start", exc_info=True
        )
        return None, exc


def build_tool_deps(settings: Settings, long_term: LongTermMemory | None = None) -> ToolDeps:
    """The shared infrastructure every agent's tools draw from, built once.

    RAG and the wiki degrade gracefully: with a backend unavailable its value is
    None and its error says why, so an agent that cannot work without it fails
    with the real cause while the others simply go without. `long_term` lets a
    caller (the eval harness) pre-seed memory.
    """
    if long_term is None:
        long_term = build_long_term(settings)
    wiki, wiki_error = _load_wiki(settings)
    try:
        vector_store = build_vector_store(settings, settings.rag_vector_store_backend)
        retriever = build_retriever(settings, vector_store)
        collections = list_collections()
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "RAG backend unavailable; search_documents tool disabled", exc_info=True
        )
        return ToolDeps(
            settings=settings,
            long_term=long_term,
            retriever=None,
            vector_store=None,
            collections=(),
            rag_error=exc,
            wiki=wiki,
            wiki_error=wiki_error,
        )
    return ToolDeps(
        settings=settings,
        long_term=long_term,
        retriever=retriever,
        vector_store=vector_store,
        collections=tuple(collections),
        wiki=wiki,
        wiki_error=wiki_error,
    )


def build_agent(
    settings: Settings,
    checkpointer,
    name: str | None = None,
    long_term: LongTermMemory | None = None,
):
    """Compile one agent on the shared loop: the entry agent unless `name` says otherwise.

    The prompt, tools and model pin all come from the agent's folder
    (agents/<name>/agent.py); nothing about an agent comes from config.
    """
    # Deferred: callers that never run an agent (ingestion, the CLI) should not
    # pay LangGraph's import cost.
    from harness.orchestration.loop import build_graph

    definition = load_agent(name or settings.agent)
    return build_graph(
        build_llm(settings, build_parser(settings)),
        checkpointer,
        registry=definition.tools(build_tool_deps(settings, long_term)),
        system_prompt=definition.prompt,
        model_pin=definition.model,
    )


@dataclass(frozen=True)
class AgentRuntime:
    """Everything the API needs to serve the entry agent, built once at startup.

    One immutable snapshot: the graph, the guard's prompt, the run rows' agent
    name, every agent-related setting and the storage they write to come from
    here, so they cannot drift apart even if .env changes while the process runs.
    """

    settings: Settings
    definition: AgentDefinition
    graph: Any
    storage: Storage

    async def aclose(self) -> None:
        """Close the storage's connections; the runtime serves nothing after this."""
        await self.storage.aclose()


async def build_runtime(settings: Settings) -> AgentRuntime:
    """Load the entry agent, open its storage, compile it on the shared loop."""
    definition = load_agent(settings.agent)
    storage = await open_storage(settings)
    try:
        graph = build_agent(settings, storage.checkpointer)
    except BaseException:
        await storage.aclose()
        raise
    return AgentRuntime(
        settings=settings, definition=definition, graph=graph, storage=storage
    )


def build_input_guard(settings: Settings) -> InputGuard:
    """The input guard for chat requests (SL-2).

    The fixed limits always apply (each off at 0); the Microsoft-Decision-1
    classifier only when input_guard_enabled. It calls the same Foundry resource
    as the LLM, with the same credentials: the resource key when azure_api_key
    is set, else the managed identity. A chat reply cannot wait on the guard:
    InputGuard enforces guard_timeout_seconds as a hard ceiling, and the HTTP
    client gets the same budget.
    """
    classifier = None
    if settings.input_guard_enabled:
        from harness.adapters.guard.azure_decision import AzureDecisionClassifier

        auth: dict[str, Any]
        if settings.azure_api_key:
            auth = {"api_key": settings.azure_api_key}
        else:
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider

            auth = {
                "token_provider": get_bearer_token_provider(
                    DefaultAzureCredential(),
                    "https://cognitiveservices.azure.com/.default",
                )
            }
        classifier = AzureDecisionClassifier(
            httpx.AsyncClient(timeout=settings.guard_timeout_seconds),
            settings.azure_endpoint,
            settings.guard_deployment,
            **auth,
        )
    return InputGuard(
        classifier=classifier,
        limits=GuardLimits(
            max_message_chars=settings.max_message_chars,
            max_thread_turns=settings.max_thread_turns,
            daily_token_budget=settings.daily_token_budget,
        ),
        timeout_seconds=settings.guard_timeout_seconds,
    )
