"""A tool that exposes RAG retrieval to the agent.

Calls Retriever.retrieve() directly and hands the agent raw chunks — never
RagPipeline.answer(), which does its own LLM generation and would mean a
nested, redundant LLM call inside a tool call. Mirrors RecallTool's shape:
read-only, no approval, formatted string result, explicit "nothing found"
fallback.
"""
from __future__ import annotations

from pathlib import PurePath
from typing import Any

from harness.core.rag.document import Chunk
from harness.core.rag.ports import Retriever, VectorStore
from harness.core.tools.base import ToolError
from harness.core.types import ToolSpec

_DESCRIPTION = (
    "Search the ingested document knowledge base (papers, files) for "
    "information relevant to a question. Use this for questions about "
    "document content — distinct from 'recall', which is personal facts "
    "about the user. Never guess at document content without calling "
    "this first."
)

_MAX_K = 20


def _cited_name(chunk: Chunk) -> str:
    """What a citation names: the document's title, else its file name without
    folder or extension. Never the path: the model repeats what it reads."""
    return chunk.title or PurePath(chunk.source_path).stem or "untitled"


class SearchDocumentsTool:
    def __init__(
        self,
        retriever: Retriever,
        vector_store: VectorStore,
        default_collection: str,
        default_k: int = 5,
        collections: list[str] | None = None,
    ) -> None:
        self._retriever = retriever
        self._vector_store = vector_store
        self._default_collection = default_collection
        self._default_k = default_k
        self._collections = collections or []
        # What run() may search: the agent's default plus the collections it
        # lists. A model can name any collection in its arguments, even one the
        # schema never offered, so every call is checked against this.
        self._allowed = tuple(dict.fromkeys([default_collection, *self._collections]))

    @property
    def spec(self) -> ToolSpec:
        properties: dict[str, Any] = {
            "query": {"type": "string", "description": "What to search for"},
            "k": {
                "type": "integer",
                "description": "Max chunks to retrieve",
                "default": self._default_k,
                "minimum": 1,
                "maximum": _MAX_K,
            },
        }
        if len(self._collections) > 1:
            properties["collection"] = {
                "type": "string",
                "enum": self._collections,
                "description": "Which document collection to search",
                "default": self._default_collection,
            }
        return ToolSpec(
            name="search_documents",
            description=_DESCRIPTION,
            parameters={
                "type": "object",
                "properties": properties,
                "required": ["query"],
            },
        )

    async def run(self, arguments: dict[str, Any]) -> str:
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ToolError("query is required: say what to search for")
        k = self._k(arguments.get("k"))
        # Missing, null or empty means the default: every vector store reads a
        # None collection as "search everything", so None must never get through.
        collection = arguments.get("collection") or self._default_collection
        if collection not in self._allowed:
            raise ToolError(
                f"collection {collection!r} is not available; "
                f"this assistant can search: {', '.join(self._allowed)}"
            )

        hits = await self._retriever.retrieve(query, k=k, collection=collection)
        if not hits:
            count = await self._vector_store.count(collection=collection)
            if count == 0:
                return f"No documents found in collection '{collection}'."
            return "No relevant documents found."

        lines = []
        for i, sc in enumerate(hits, start=1):
            section = " > ".join(sc.chunk.section_path) or "n/a"
            lines.append(
                f'[{i}] (score: {sc.score:.2f}, source: "{_cited_name(sc.chunk)}", '
                f"section: {section})\n{sc.chunk.text}"
            )
        return "\n\n".join(lines)

    def _k(self, value: Any) -> int:
        # Missing or null means the default: small models send null for an
        # optional parameter.
        if value is None:
            return self._default_k
        try:
            return max(1, min(int(value), _MAX_K))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ToolError(f"k must be a whole number from 1 to {_MAX_K}") from exc
