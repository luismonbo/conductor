"""Docling-backed PDF parser. Runs ONLY in the Linux ingest container (torch +
onnxruntime crash on macOS). Walks Docling's structured document to preserve
real heading levels + tables, and emits the JSON section schema DoclingNormalizer
consumes. Verified against docling==2.123.1; the container run in Task 8 pins the
exact item API."""
from __future__ import annotations

import json
from pathlib import Path

from harness.core.rag.document import ParsedContent


class DoclingParser:
    async def parse(self, path: Path) -> ParsedContent:
        import asyncio
        text = await asyncio.to_thread(self._convert, path)
        return ParsedContent(text=text, format="pdf", parser="docling")

    def _convert(self, path: Path) -> str:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
        opts = PdfPipelineOptions(do_ocr=False, do_table_structure=True)
        conv = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
        doc = conv.convert(str(path)).document
        # export_to_dict() carries structured items with heading levels + tables.
        # Map to our schema. NOTE: the exact dict keys are verified/adjusted
        # during the Task 8 container run against docling==2.123.1.
        payload = doc.export_to_dict()
        sections = _sections_from_docling_dict(payload)
        return json.dumps({"title": payload.get("name", "") or "", "sections": sections})


def _sections_from_docling_dict(payload: dict) -> list[dict]:
    """Pure transform: docling export_to_dict -> [{title,level,kind,text}].
    Kept as a module function so it is unit-testable against a captured fixture
    without importing docling."""
    sections: list[dict] = []
    for item in payload.get("texts", []):
        label = item.get("label", "")
        text = item.get("text", "")
        if label in ("section_header", "title"):
            sections.append({"title": text, "level": int(item.get("level", 1) or 1),
                             "kind": "prose", "text": ""})
        elif sections:
            sections[-1]["text"] = (sections[-1]["text"] + "\n" + text).strip()
        else:
            sections.append({"title": "", "level": 0, "kind": "prose", "text": text})
    for table in payload.get("tables", []):
        sections.append({"title": table.get("caption", "Table") or "Table",
                         "level": 3, "kind": "table",
                         "text": table.get("text", "") or json.dumps(table.get("data", {}))})
    return sections
