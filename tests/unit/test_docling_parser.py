"""Fixture-based mapping tests for _sections_from_docling_dict, run against a
real captured docling export_to_dict() payload (tests/fixtures/
docling_attention.json — docling==2.123.1 on data/raw/papers/
attention-is-all-you-need.pdf, captured in the Task 9 ingest container; see
task-9-report.md). Docling-free: only exercises the pure dict->dict transform,
so this runs on the host like any other unit test."""
from __future__ import annotations

import json
from pathlib import Path

from harness.adapters.parsing.docling_parser import _sections_from_docling_dict

_FIXTURE = json.loads(Path("tests/fixtures/docling_attention.json").read_text())


def test_sections_from_real_docling_dict_have_hierarchy_and_a_table():
    sections = _sections_from_docling_dict(_FIXTURE)
    assert any(s["level"] == 1 for s in sections)      # a top-level heading
    assert any(s["kind"] == "table" for s in sections)  # TableFormer output preserved
    assert all(set(s) == {"title", "level", "kind", "text"} for s in sections)


def test_numbered_subsection_headings_nest_deeper_than_top_level_sections():
    # Docling's own per-item "level" is uniformly 1 for every section_header
    # in the real fixture; real depth has to come from the heading's own
    # numeric prefix ("3.2.1 Scaled Dot-Product Attention" vs "3 Model
    # Architecture"). Guards against silently regressing back to flat level=1
    # everywhere, which would still pass the assertion above.
    sections = _sections_from_docling_dict(_FIXTURE)
    by_title = {s["title"]: s["level"] for s in sections if s["kind"] == "prose"}
    assert by_title["3 Model Architecture"] == 1
    assert by_title["3.1 Encoder and Decoder Stacks"] == 2
    assert by_title["3.2.1 Scaled Dot-Product Attention"] == 3


def test_table_text_is_readable_cell_content_not_raw_cell_metadata():
    # A real docling table has no "text" key at all — table.get("text", "")
    # always falls back, and the naive fallback (json.dumps of the whole
    # `data` dict) would dump bbox coordinates and row/col span metadata for
    # every cell instead of the cell text itself.
    sections = _sections_from_docling_dict(_FIXTURE)
    tables = [s for s in sections if s["kind"] == "table"]
    assert tables
    for table in tables:
        assert "bbox" not in table["text"]
        assert "coord_origin" not in table["text"]
    # Table 2 in the source PDF has a real caption docling attaches via a
    # $ref into texts[], not an inline string.
    assert any("BLEU scores" in t["title"] for t in tables)


def test_repeated_page_furniture_is_not_glued_into_section_text():
    # Every page repeats the arXiv id (page_header) and conference banner
    # (page_footer) as content_layer="furniture" — these must not leak into
    # real section content.
    sections = _sections_from_docling_dict(_FIXTURE)
    assert not any("arXiv:1706.03762" in s["text"] for s in sections)
