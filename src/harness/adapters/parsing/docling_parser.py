"""docling-backed Parser — the layout-aware path for PDFs. Uses only
export_to_markdown(): docling's reading-order reconstruction and table
structure recognition are already reflected in that rendering, so this
adapter doesn't depend on DoclingDocument's finer internal object graph."""
from __future__ import annotations

import asyncio
from pathlib import Path

from docling.document_converter import DocumentConverter

from harness.core.rag.document import ParsedContent

_SUCCESS_STATUSES = {"SUCCESS", "PARTIAL_SUCCESS"}


class DoclingParseError(Exception):
    pass


class DoclingParser:
    def __init__(self) -> None:
        self._converter = DocumentConverter()

    async def parse(self, path: Path) -> ParsedContent:
        try:
            result = await asyncio.to_thread(self._converter.convert, str(path))
        except Exception as exc:
            raise DoclingParseError(f"docling failed to convert {path}: {exc}") from exc

        status_name = getattr(result.status, "name", str(result.status))
        if status_name not in _SUCCESS_STATUSES:
            raise DoclingParseError(f"docling conversion status={status_name} for {path}")

        markdown = result.document.export_to_markdown()

        try:
            page_count: int | None = len(result.document.pages)
        except (AttributeError, TypeError):
            page_count = None

        return ParsedContent(text=markdown, format="pdf", parser="docling", page_count=page_count)
