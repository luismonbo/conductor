"""markitdown-backed Parser — handles the remaining office/html formats
(docx, pptx, xlsx, html, ...). ParserRouter sends .pdf to DoclingParser and
.md/.markdown to MarkdownPassthroughParser; everything else falls through
to this parser (see router.py)."""

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
            raise MarkitdownParseError(
                f"markitdown failed to convert {path}: {exc}"
            ) from exc

        text = getattr(result, "markdown", None) or getattr(result, "text_content", None)
        if not text:
            raise MarkitdownParseError(f"markitdown returned no text for {path}")

        return ParsedContent(
            text=text,
            format=path.suffix.lstrip(".").lower(),
            parser="markitdown",
        )
