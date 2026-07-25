"""markitdown-backed Parser — the broad-format, lighter-weight path (docx,
pptx, xlsx, html, and the docling-failure fallback for PDFs). See
docs/superpowers/specs/2026-07-25-rag-ingestion-retrieval-design.md."""
from __future__ import annotations

import asyncio
from pathlib import Path

from markitdown import MarkItDown

from harness.core.rag.document import ParsedContent


class MarkitdownParseError(Exception):
    pass


class MarkitdownParser:
    def __init__(self) -> None:
        self._md = MarkItDown(enable_plugins=False)  # no llm_client -> zero network calls

    async def parse(self, path: Path) -> ParsedContent:
        try:
            result = await asyncio.to_thread(self._md.convert, str(path))
        except Exception as exc:
            raise MarkitdownParseError(f"markitdown failed to convert {path}: {exc}") from exc

        text = getattr(result, "markdown", None) or getattr(result, "text_content", None)
        if not text:
            raise MarkitdownParseError(f"markitdown returned no text for {path}")

        return ParsedContent(
            text=text,
            format=path.suffix.lstrip(".").lower(),
            parser="markitdown",
        )
