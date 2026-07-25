from __future__ import annotations

import pytest

from harness.adapters.llm.fake import FakeLLMClient
from harness.adapters.normalization.llm_normalizer import LlmNormalizer, NormalizationError
from harness.core.rag.document import ParsedContent
from harness.core.types import LLMResponse, ToolCall


def _scripted_response() -> LLMResponse:
    args = {
        "title": "Attention Is What You Need",
        "sections": [
            {"title": "Introduction", "level": 1, "kind": "prose", "text": "We study...", "order": 0},
            {"title": "Results", "level": 1, "kind": "table", "text": "| a | b |", "order": 1},
        ],
    }
    return LLMResponse(
        text="",
        tool_calls=(ToolCall(id="call_1", name="emit_normalized_document", arguments=args),),
    )


@pytest.mark.asyncio
async def test_normalize_builds_one_document_from_tool_call():
    llm = FakeLLMClient([_scripted_response()])
    normalizer = LlmNormalizer(llm)
    parsed = ParsedContent(text="# Attention...\n\nWe study...", format="pdf", parser="docling")

    docs = await normalizer.normalize(parsed, source_path="papers/attn.pdf", collection="papers")

    assert len(docs) == 1
    doc = docs[0]
    assert doc.title == "Attention Is What You Need"
    assert doc.collection == "papers"
    assert doc.source_path == "papers/attn.pdf"
    assert doc.format == "pdf"
    assert doc.parser == "docling"
    assert len(doc.sections) == 2
    assert doc.sections[1].kind == "table"
    assert doc.document_id.startswith("papers/")


@pytest.mark.asyncio
async def test_normalize_raises_when_model_does_not_call_the_tool():
    llm = FakeLLMClient([LLMResponse(text="I cannot do that.", tool_calls=())])
    normalizer = LlmNormalizer(llm)
    parsed = ParsedContent(text="some text", format="html", parser="markitdown")

    with pytest.raises(NormalizationError):
        await normalizer.normalize(parsed, source_path="papers/x.html", collection="papers")
