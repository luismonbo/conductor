"""Container-gated: docling only installs (and only runs without a libomp
crash) inside the Linux ingest image built from Dockerfile.ingest — never on
the macOS host. Run via:
    docker compose --profile ingest run --rm --entrypoint pytest ingest \
        tests/integration/test_docling_parser.py -q
"""
from __future__ import annotations

import pytest

pytest.importorskip("docling")  # skips cleanly where docling isn't installed (macOS host / CI base)


@pytest.mark.asyncio
async def test_docling_parses_real_pdf(tmp_path):
    from pathlib import Path
    from harness.adapters.parsing.docling_parser import DoclingParser
    pdf = Path("data/raw/papers/attention-is-all-you-need.pdf")
    if not pdf.exists():
        pytest.skip("sample PDF not present")
    parsed = await DoclingParser().parse(pdf)
    import json
    payload = json.loads(parsed.text)
    assert parsed.parser == "docling" and len(payload["sections"]) > 5
