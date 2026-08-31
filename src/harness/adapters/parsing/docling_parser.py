"""Docling-backed PDF parser. Runs ONLY in the Linux ingest container (torch +
onnxruntime crash on macOS). Walks Docling's structured document to preserve
real heading levels + tables, and emits the JSON section schema DoclingNormalizer
consumes. Verified against docling==2.123.1 by capturing a real export_to_dict()
payload (tests/fixtures/docling_attention.json, from data/raw/papers/
attention-is-all-you-need.pdf) in the Task 9 ingest container run and correcting
_sections_from_docling_dict's mapping against it — see task-9-report.md for what
the original best-effort mapping got wrong and why."""

from __future__ import annotations

import json
import re
from pathlib import Path

from harness.core.rag.document import ParsedContent

# Matches a numbered-heading prefix like "3", "3.2", "3.2.1" at the start of a
# section_header's text, one dot-separated group per nesting level.
_NUMBERED_HEADING_RE = re.compile(r"^(\d+(?:\.\d+)*)(?:[.\s]|$)")


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
        conv = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
        )
        doc = conv.convert(str(path)).document
        # export_to_dict() carries structured items with heading levels + tables.
        payload = doc.export_to_dict()
        sections = _sections_from_docling_dict(payload)
        return json.dumps({"title": payload.get("name", "") or "", "sections": sections})


def _heading_level(item: dict) -> int:
    """Docling's own per-item `level` is NOT real hierarchy depth in practice:
    every section_header in the captured fixture carries level=1, including
    "3.2.1 Scaled Dot-Product Attention" right alongside the paper's own
    title. Real depth instead lives in the heading text's own numeric prefix
    ("3.2.1" -> 3 groups -> level 3). Falls back to docling's raw level
    (effectively always 1) for unnumbered headings ("Abstract", "References").
    Known limitation: an unnumbered sub-heading (e.g. a figure-panel label
    docling misclassifies as a section_header) still falls back to level 1
    rather than nesting under its true numbered parent — recovering that
    would need the document's parent/child ref tree, not just the heading's
    own text, and isn't needed for any current consumer."""
    match = _NUMBERED_HEADING_RE.match(item.get("text", "").strip())
    if match:
        return match.group(1).count(".") + 1
    return int(item.get("level", 1) or 1)


def _resolve_ref(payload: dict, ref: dict) -> dict | None:
    """Resolve one of docling's internal {"$ref": "#/texts/12"} pointers
    (used e.g. by a table's `captions` list) to the referenced item."""
    parts = ref.get("$ref", "").removeprefix("#/").split("/")
    if len(parts) != 2 or not parts[1].isdigit():
        return None
    group, index = parts
    items = payload.get(group)
    if not isinstance(items, list) or not (0 <= int(index) < len(items)):
        return None
    return items[int(index)]


def _table_title(payload: dict, table: dict) -> str:
    # A table's caption is NOT an inline string on the table — it's a (often
    # empty) list of $ref pointers into payload["texts"].
    captions = [_resolve_ref(payload, ref) for ref in table.get("captions", [])]
    texts = [c["text"] for c in captions if c and c.get("text")]
    return "; ".join(texts) or "Table"


def _table_text(table: dict) -> str:
    # data.grid is docling's row/col-ordered 2D array of cell dicts — reading
    # cell text off it directly gives a compact, readable table instead of a
    # json.dumps() of the full cell metadata (bbox coordinates, row/col
    # spans, ...), which is what table.get("text", "") falling back to
    # table.get("data", {}) would otherwise produce (there is no "text" key
    # on a table at all; both were always the fallback in practice).
    grid = table.get("data", {}).get("grid", [])
    rows = [" | ".join(cell.get("text", "") for cell in row) for row in grid]
    return "\n".join(rows)


def _sections_from_docling_dict(payload: dict) -> list[dict]:
    """Pure transform: docling export_to_dict -> [{title,level,kind,text}].
    Kept as a module function so it is unit-testable against a captured fixture
    without importing docling."""
    sections: list[dict] = []
    for item in payload.get("texts", []):
        if item.get("content_layer") == "furniture":
            # Repeated running headers/footers (arXiv id, conference banner,
            # page numbers) on every page — not real document content, and
            # gluing them onto whatever section happens to be last would
            # otherwise pollute every section's text with the same boilerplate.
            continue
        label = item.get("label", "")
        text = item.get("text", "")
        if label in ("section_header", "title"):
            sections.append(
                {
                    "title": text,
                    "level": _heading_level(item),
                    "kind": "prose",
                    "text": "",
                }
            )
        elif sections:
            sections[-1]["text"] = (sections[-1]["text"] + "\n" + text).strip()
        else:
            sections.append({"title": "", "level": 0, "kind": "prose", "text": text})
    for table in payload.get("tables", []):
        sections.append(
            {
                "title": _table_title(payload, table),
                "level": 3,
                "kind": "table",
                "text": _table_text(table),
            }
        )
    return sections
