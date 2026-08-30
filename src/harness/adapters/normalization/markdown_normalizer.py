"""Deterministic markdown normalizer: split parsed markdown on ATX headings
into ordered DocumentSections. No LLM, no network — replaces LlmNormalizer for
self-structuring input (native .md and MarkItDown office output)."""
from __future__ import annotations

import re
from datetime import datetime, timezone

from harness.core.rag.document import (
    DocumentSection,
    NormalizedDocument,
    ParsedContent,
    hash_bytes,
    make_document_id,
)

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


class MarkdownNormalizer:
    async def normalize(
        self, parsed: ParsedContent, source_path: str, collection: str
    ) -> list[NormalizedDocument]:
        sections: list[DocumentSection] = []
        title = ""
        cur_title, cur_level, cur_lines = "", 0, []

        def flush() -> None:
            text = "\n".join(cur_lines).strip()
            if text or cur_title:
                sections.append(
                    DocumentSection(
                        title=cur_title, level=cur_level, kind="prose",
                        text=text, order=len(sections),
                    )
                )

        for line in parsed.text.splitlines():
            m = _HEADING.match(line)
            if m:
                flush()
                cur_level = len(m.group(1))
                cur_title = m.group(2).strip()
                cur_lines = []
                if not title and cur_level == 1:
                    title = cur_title
            else:
                cur_lines.append(line)
        flush()

        if not sections:  # empty input
            sections = [DocumentSection(title="", level=0, kind="prose", text="", order=0)]

        content_hash = hash_bytes(parsed.text.encode())
        return [
            NormalizedDocument(
                document_id=make_document_id(collection, source_path),
                source_path=source_path,
                collection=collection,
                title=title or (sections[0].title if sections else ""),
                format=parsed.format,
                parser=parsed.parser,
                content_hash=content_hash,
                sections=tuple(sections),
                ingested_at=datetime.now(timezone.utc).isoformat(),
            )
        ]
