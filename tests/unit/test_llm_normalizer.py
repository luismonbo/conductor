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


def _window_response(title: str, section_title: str, text: str) -> LLMResponse:
    return LLMResponse(
        text="",
        tool_calls=(
            ToolCall(
                id="c",
                name="emit_normalized_document",
                arguments={
                    "title": title,
                    "sections": [
                        {"title": section_title, "level": 1, "kind": "prose",
                         "text": text, "order": 0},
                    ],
                },
            ),
        ),
    )


@pytest.mark.asyncio
async def test_oversized_document_is_normalized_in_windows_and_merged():
    """A real paper is far larger than a local model's context window. The
    normalizer must window the input rather than sending it whole."""
    # Three paragraphs, each ~120 chars, with a 200-char window -> 3 calls.
    paragraphs = [f"Paragraph {i}. " + ("filler " * 15) for i in range(3)]
    parsed = ParsedContent(text="\n\n".join(paragraphs), format="pdf", parser="docling")

    llm = FakeLLMClient([
        _window_response("A Paper", "Introduction", "first"),
        _window_response("A Paper", "Method", "second"),
        _window_response("A Paper", "Results", "third"),
    ])
    normalizer = LlmNormalizer(llm, max_window_chars=200)

    [doc] = await normalizer.normalize(parsed, source_path="papers/big.pdf", collection="papers")

    assert len(llm.calls) == 3                       # one model call per window
    assert doc.title == "A Paper"                    # taken from the first window
    assert [s.title for s in doc.sections] == ["Introduction", "Method", "Results"]
    assert [s.order for s in doc.sections] == [0, 1, 2]  # renumbered continuously
    assert doc.extra["windows"] == "3"


@pytest.mark.asyncio
async def test_small_document_still_uses_a_single_call():
    parsed = ParsedContent(text="# Intro\n\nShort.", format="pdf", parser="docling")
    llm = FakeLLMClient([_window_response("Small", "Intro", "Short.")])
    normalizer = LlmNormalizer(llm, max_window_chars=10_000)

    [doc] = await normalizer.normalize(parsed, source_path="papers/s.pdf", collection="papers")

    assert len(llm.calls) == 1
    assert doc.extra["windows"] == "1"


@pytest.mark.asyncio
async def test_partial_window_failure_keeps_the_rest_and_records_the_loss():
    paragraphs = [f"Paragraph {i}. " + ("filler " * 15) for i in range(3)]
    parsed = ParsedContent(text="\n\n".join(paragraphs), format="pdf", parser="docling")

    llm = FakeLLMClient([
        _window_response("A Paper", "Introduction", "first"),
        LLMResponse(text="sorry", tool_calls=()),          # this window fails
        _window_response("A Paper", "Results", "third"),
    ])
    normalizer = LlmNormalizer(llm, max_window_chars=200)

    [doc] = await normalizer.normalize(parsed, source_path="papers/big.pdf", collection="papers")

    assert [s.title for s in doc.sections] == ["Introduction", "Results"]
    assert doc.extra["windows_failed"] == "1"


@pytest.mark.asyncio
async def test_raises_when_every_window_fails():
    paragraphs = [f"Paragraph {i}. " + ("filler " * 15) for i in range(2)]
    parsed = ParsedContent(text="\n\n".join(paragraphs), format="pdf", parser="docling")

    llm = FakeLLMClient([
        LLMResponse(text="no", tool_calls=()),
        LLMResponse(text="no", tool_calls=()),
    ])
    normalizer = LlmNormalizer(llm, max_window_chars=200)

    with pytest.raises(NormalizationError):
        await normalizer.normalize(parsed, source_path="papers/big.pdf", collection="papers")


@pytest.mark.asyncio
async def test_document_id_is_stable_regardless_of_window_size():
    """document_id derives from the whole document's content hash, so changing
    the window size must not silently create a duplicate document."""
    parsed = ParsedContent(
        text="\n\n".join(f"Paragraph {i}. " + ("filler " * 15) for i in range(3)),
        format="pdf", parser="docling",
    )
    responses = [_window_response("P", f"S{i}", "t") for i in range(4)]

    windowed = LlmNormalizer(FakeLLMClient(list(responses)), max_window_chars=200)
    whole = LlmNormalizer(FakeLLMClient(list(responses)), max_window_chars=10_000)

    [a] = await windowed.normalize(parsed, source_path="p.pdf", collection="papers")
    [b] = await whole.normalize(parsed, source_path="p.pdf", collection="papers")

    assert a.document_id == b.document_id
