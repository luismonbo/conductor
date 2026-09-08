"""Deserializes the JSON section schema emitted by DoclingParser into a
NormalizedDocument. Deterministic and docling-free — all docling-specific
extraction lives in DoclingParser (adapters/parsing/docling_parser.py)."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from harness.core.rag.document import (
    DocumentSection,
    NormalizedDocument,
    ParsedContent,
    hash_bytes,
    make_document_id,
)


class DoclingNormalizer:
    async def normalize(
        self, parsed: ParsedContent, source_path: str, collection: str
    ) -> list[NormalizedDocument]:
        payload = json.loads(parsed.text)
        sections = tuple(
            DocumentSection(
                title=s.get("title", ""),
                level=int(s.get("level", 0)),
                kind=s.get("kind", "prose"),
                text=s.get("text", ""),
                order=i,
            )
            for i, s in enumerate(payload.get("sections", []))
        )
        return [
            NormalizedDocument(
                document_id=make_document_id(collection, source_path),
                source_path=source_path,
                collection=collection,
                title=payload.get("title", ""),
                format=parsed.format,
                parser=parsed.parser,
                content_hash=hash_bytes(parsed.text.encode()),
                sections=sections,
                ingested_at=datetime.now(timezone.utc).isoformat(),
            )
        ]
