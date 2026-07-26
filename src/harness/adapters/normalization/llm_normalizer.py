"""Normalizes parser output into NormalizedDocument via LLM structured output.

Reuses the existing tool-calling path rather than parsing free-text JSON: the
model is given one fake tool whose schema mirrors DocumentSection, and the
structured arguments come back on LLMResponse.tool_calls — the same mechanism
already relied on for tool-call reliability elsewhere in this codebase.

Long documents are normalized in windows. A real academic PDF parses to tens
of thousands of characters — the "Attention Is All You Need" fixture is ~49k —
which overruns a local model's context in a single call, and the failure mode
is silent: the model returns free text instead of a tool call and the whole
document is dropped. Windowing on paragraph boundaries keeps each call inside
the context budget while preserving the design constraint that section
structure is *interpreted* by the model, never string-munged out of the text.
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


# ~3k tokens of English prose. Sized to leave room for the tool schema and the
# model's reply inside a conservative 4k-8k local context (Ollama's default
# num_ctx is small); raise it when pointing at a long-context cloud model.
_DEFAULT_WINDOW_CHARS = 12_000

_SYSTEM_PROMPT = (
    "You normalize parsed documents into structured sections. "
    "Call emit_normalized_document with the result. Do not answer "
    "in free text."
)


def _split_into_windows(text: str, max_chars: int) -> list[str]:
    """Split on blank lines, packing paragraphs up to max_chars.

    Paragraph boundaries are the safest split point available pre-normalization:
    they rarely cut a sentence, and both parsers emit markdown-ish text where a
    heading and its body are separated by blank lines.
    """
    if len(text) <= max_chars:
        return [text]

    windows: list[str] = []
    current: list[str] = []
    current_len = 0
    for paragraph in text.split("\n\n"):
        # A single paragraph longer than the budget still has to be broken up,
        # or it would be handed to the model whole and fail the same way.
        pieces = (
            [paragraph[i:i + max_chars] for i in range(0, len(paragraph), max_chars)]
            if len(paragraph) > max_chars
            else [paragraph]
        )
        for piece in pieces:
            if current and current_len + len(piece) > max_chars:
                windows.append("\n\n".join(current))
                current, current_len = [], 0
            current.append(piece)
            current_len += len(piece) + 2  # the "\n\n" that will rejoin them
    if current:
        windows.append("\n\n".join(current))
    return windows


class LlmNormalizer:
    def __init__(self, llm: LLMClient, max_window_chars: int = _DEFAULT_WINDOW_CHARS) -> None:
        self._llm = llm
        self._max_window_chars = max_window_chars

    async def normalize(
        self, parsed: ParsedContent, source_path: str, collection: str
    ) -> list[NormalizedDocument]:
        windows = _split_into_windows(parsed.text, self._max_window_chars)

        title = ""
        sections: list[DocumentSection] = []
        failed = 0
        for window in windows:
            args = await self._normalize_window(window)
            if args is None:
                failed += 1
                continue
            if not title:
                title = args.get("title", "")
            for raw in args.get("sections", []):
                sections.append(
                    DocumentSection(
                        title=raw["title"],
                        level=raw["level"],
                        kind=raw["kind"],
                        text=raw["text"],
                        # Renumber across windows: each window numbers its own
                        # sections from 0, which would collide on merge.
                        order=len(sections),
                    )
                )

        if not sections:
            raise NormalizationError(
                f"model did not call emit_normalized_document for {source_path} "
                f"({failed}/{len(windows)} window(s) failed)"
            )

        # Hash the whole document, not the window — document_id must not change
        # when the window size does, or a re-ingest would duplicate the document.
        content_hash = hash_bytes(parsed.text.encode())
        return [
            NormalizedDocument(
                document_id=make_document_id(collection, content_hash),
                source_path=source_path,
                collection=collection,
                title=title,
                format=parsed.format,
                parser=parsed.parser,
                content_hash=content_hash,
                sections=tuple(sections),
                ingested_at=datetime.now(timezone.utc).isoformat(),
                extra={"windows": str(len(windows)), "windows_failed": str(failed)},
            )
        ]

    async def _normalize_window(self, window: str) -> dict | None:
        """Return the tool-call arguments, or None if the model didn't call it."""
        messages = [
            Message(Role.SYSTEM, _SYSTEM_PROMPT),
            Message(Role.USER, window),
        ]
        response = await self._llm.generate(messages, tools=[_NORMALIZE_TOOL])
        if not response.tool_calls:
            return None
        return response.tool_calls[0].arguments
