from __future__ import annotations

from harness.core.rag.document import (
    Chunk,
    DocumentSection,
    NormalizedDocument,
    ParsedContent,
    ScoredChunk,
    hash_bytes,
    make_document_id,
)


def test_hash_bytes_is_deterministic_sha256_hex():
    assert hash_bytes(b"hello") == hash_bytes(b"hello")
    assert hash_bytes(b"hello") != hash_bytes(b"world")
    assert len(hash_bytes(b"hello")) == 64  # sha256 hex digest length


def test_normalized_document_holds_ordered_sections():
    sections = (
        DocumentSection(title="Intro", level=1, text="hello", order=0),
        DocumentSection(title="Method", level=1, text="world", order=1),
    )
    doc = NormalizedDocument(
        document_id="papers/abc123",
        source_path="papers/x.pdf",
        collection="papers",
        title="A Paper",
        format="pdf",
        parser="docling",
        content_hash=hash_bytes(b"x"),
        sections=sections,
        ingested_at="2026-07-25T00:00:00Z",
    )
    assert doc.sections[1].title == "Method"
    assert doc.extra == {}


def test_chunk_defaults_are_prose_and_empty_metadata():
    chunk = Chunk(
        chunk_id="papers/abc123:0",
        document_id="papers/abc123",
        collection="papers",
        text="some chunk text",
        section_path=("Intro",),
    )
    assert chunk.section_kind == "prose"
    assert chunk.chunk_version == 1
    assert chunk.metadata == {}


def test_scored_chunk_pairs_chunk_with_score():
    chunk = Chunk(
        chunk_id="c1", document_id="d1", collection="papers", text="t",
        section_path=(),
    )
    scored = ScoredChunk(chunk=chunk, score=0.87)
    assert scored.score == 0.87
    assert scored.chunk is chunk


def test_parsed_content_defaults():
    parsed = ParsedContent(text="# Title\n\nbody", format="pdf", parser="docling")
    assert parsed.page_count is None
    assert parsed.structure_hints == {}


def test_document_id_is_stable_for_same_source_path():
    a = make_document_id("papers", "data/raw/papers/x.pdf")
    b = make_document_id("papers", "data/raw/papers/x.pdf")
    assert a == b
    assert a.startswith("papers/")


def test_document_id_differs_by_source_path_and_collection():
    assert make_document_id("papers", "a.pdf") != make_document_id("papers", "b.pdf")
    assert make_document_id("papers", "a.pdf") != make_document_id("docs", "a.pdf")


def test_document_id_uses_hash_not_naive_truncation():
    # Under naive source_path[:16] truncation, these would collide
    id1 = make_document_id("papers", "data/raw/papers/x.pdf")
    id2 = make_document_id("papers", "data/raw/papers/y.pdf")
    assert id1 != id2
