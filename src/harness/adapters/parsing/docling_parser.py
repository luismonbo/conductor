"""docling-backed Parser — the layout-aware path for PDFs. Uses only
export_to_markdown(): docling's reading-order reconstruction and table
structure recognition are already reflected in that rendering, so this
adapter doesn't depend on DoclingDocument's finer internal object graph.

OCR is off: RapidOCR (onnxruntime) and docling's layout model (torch) each
bundle their own libomp, and loading both segfaults on macOS (exit 139, right
after model-weight loading — KMP_DUPLICATE_LIB_OK=TRUE does NOT fix this, it
only suppresses the *safe* abort and lets the *unsafe* crash through instead).
Fine for this project's corpus (born-digital text PDFs get nothing from OCR
anyway); revisit with real thread-isolation if scanned/image PDFs are ever
needed."""
from __future__ import annotations

import asyncio
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from harness.core.rag.document import ParsedContent

_SUCCESS_STATUSES = {"SUCCESS", "PARTIAL_SUCCESS"}


class DoclingParseError(Exception):
    pass


class DoclingParser:
    def __init__(self) -> None:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        self._converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )

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
