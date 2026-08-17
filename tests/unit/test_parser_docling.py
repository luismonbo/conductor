from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from docling.datamodel.base_models import InputFormat

from harness.adapters.parsing.docling_parser import DoclingParseError, DoclingParser


def test_ocr_disabled_by_default():
    # RapidOCR (onnxruntime) and docling's layout model (torch) both bundle
    # their own libomp; loading both segfaults on this machine (exit 139,
    # right after model-weight loading — see docs/devlog/010). All three
    # corpus PDFs are born-digital text anyway, so OCR buys nothing here.
    parser = DoclingParser()
    pdf_options = parser._converter.format_to_options[InputFormat.PDF]
    assert pdf_options.pipeline_options.do_ocr is False


def _fake_conversion_result(markdown: str, status_name: str = "SUCCESS", page_count: int = 3):
    document = SimpleNamespace(
        export_to_markdown=lambda: markdown,
        pages={i: object() for i in range(page_count)},
    )
    return SimpleNamespace(document=document, status=SimpleNamespace(name=status_name))


@pytest.mark.asyncio
async def test_parses_pdf_via_export_to_markdown(tmp_path):
    fixture = tmp_path / "paper.pdf"
    fixture.write_bytes(b"%PDF-1.4 fake bytes for the mock path")
    parser = DoclingParser()

    with patch.object(
        parser._converter, "convert",
        return_value=_fake_conversion_result("# Introduction\n\nWe study X."),
    ):
        parsed = await parser.parse(fixture)

    assert parsed.text == "# Introduction\n\nWe study X."
    assert parsed.format == "pdf"
    assert parsed.parser == "docling"
    assert parsed.page_count == 3


@pytest.mark.asyncio
async def test_raises_docling_parse_error_on_failure_status(tmp_path):
    fixture = tmp_path / "bad.pdf"
    fixture.write_bytes(b"not a real pdf")
    parser = DoclingParser()

    with patch.object(
        parser._converter, "convert",
        return_value=_fake_conversion_result("", status_name="FAILURE"),
    ):
        with pytest.raises(DoclingParseError):
            await parser.parse(fixture)


@pytest.mark.asyncio
async def test_raises_docling_parse_error_when_convert_throws(tmp_path):
    fixture = tmp_path / "bad.pdf"
    fixture.write_bytes(b"not a real pdf")
    parser = DoclingParser()

    with patch.object(parser._converter, "convert", side_effect=RuntimeError("backend crashed")):
        with pytest.raises(DoclingParseError):
            await parser.parse(fixture)
