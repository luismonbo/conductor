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


def test_make_document_id_scopes_by_collection_and_hash_prefix():
    doc_id = make_document_id("papers", hash_bytes(b"content"))
    assert doc_id.startswith("papers/")
    assert len(doc_id) == len("papers/") + 16


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
