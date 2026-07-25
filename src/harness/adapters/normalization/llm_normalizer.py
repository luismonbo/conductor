"""Normalizes parser output into NormalizedDocument via LLM structured output.

Reuses the existing tool-calling path rather than parsing free-text JSON: the
model is given one fake tool whose schema mirrors DocumentSection, and the
structured arguments come back on LLMResponse.tool_calls — the same mechanism
already relied on for tool-call reliability elsewhere in this codebase.
"""
from __future__ import annotations

from datetime import datetime, timezone

from harness.core.llm.client import LLMClient
from harness.core.rag.document import (
    DocumentSection,
    NormalizedDocument,
    ParsedContent,
    hash_bytes,
    make_document_id,
)
from harness.core.types import Message, Role, ToolSpec

_NORMALIZE_TOOL = ToolSpec(
    name="emit_normalized_document",
    description=(
        "Emit the normalized structure of a parsed document: a title and an "
        "ordered list of sections. Split on real structural boundaries "
        "(headings) in the source text; mark a section kind='table' when its "
        "text is primarily a table rather than prose."
    ),
    parameters={
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Document title"},
            "sections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "level": {"type": "integer", "description": "Heading depth, 1 = top-level"},
                        "kind": {"type": "string", "enum": ["prose", "table", "list"]},
                        "text": {"type": "string"},
                        "order": {"type": "integer", "description": "Position within the document"},
                    },
                    "required": ["title", "level", "kind", "text", "order"],
                },
            },
        },
        "required": ["title", "sections"],
    },
)


class NormalizationError(Exception):
    pass


class LlmNormalizer:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def normalize(
        self, parsed: ParsedContent, source_path: str, collection: str
    ) -> list[NormalizedDocument]:
        messages = [
            Message(
                Role.SYSTEM,
                "You normalize parsed documents into structured sections. "
                "Call emit_normalized_document with the result. Do not answer "
                "in free text.",
            ),
            Message(Role.USER, parsed.text),
        ]
        response = await self._llm.generate(messages, tools=[_NORMALIZE_TOOL])
        if not response.tool_calls:
            raise NormalizationError(
                f"model did not call emit_normalized_document for {source_path}"
            )

        args = response.tool_calls[0].arguments
        sections = tuple(
            DocumentSection(
                title=s["title"],
                level=s["level"],
                kind=s["kind"],
                text=s["text"],
                order=s["order"],
            )
            for s in args["sections"]
        )
        content_hash = hash_bytes(parsed.text.encode())
        return [
            NormalizedDocument(
                document_id=make_document_id(collection, content_hash),
                source_path=source_path,
                collection=collection,
                title=args["title"],
                format=parsed.format,
                parser=parsed.parser,
                content_hash=content_hash,
                sections=sections,
                ingested_at=datetime.now(timezone.utc).isoformat(),
            )
        ]
