import json
import pytest

from harness.adapters.normalization.docling_normalizer import DoclingNormalizer
from harness.core.rag.document import ParsedContent

_FIXTURE = json.dumps({
    "title": "Attention Is All You Need",
    "sections": [
        {"title": "Attention Is All You Need", "level": 1, "kind": "prose", "text": "abstract..."},
        {"title": "3 Model Architecture", "level": 2, "kind": "prose", "text": "we use..."},
        {"title": "Table 1", "level": 3, "kind": "table", "text": "| a | b |\n|---|---|"},
    ],
})


@pytest.mark.asyncio
async def test_deserializes_docling_json_into_sections():
    parsed = ParsedContent(text=_FIXTURE, format="pdf", parser="docling")
    [doc] = await DoclingNormalizer().normalize(parsed, "data/raw/papers/a.pdf", "papers")
    assert doc.title == "Attention Is All You Need"
    assert [(s.level, s.kind) for s in doc.sections] == [(1, "prose"), (2, "prose"), (3, "table")]
    assert doc.document_id.startswith("papers/")
    assert doc.sections[2].order == 2
