from __future__ import annotations

from harness.adapters.chunking.structure_aware import CHUNK_VERSION, StructureAwareChunker
from harness.core.rag.document import DocumentSection, NormalizedDocument, hash_bytes, make_document_id


def _doc(sections: tuple[DocumentSection, ...]) -> NormalizedDocument:
    content_hash = hash_bytes(b"fixture")
    return NormalizedDocument(
        document_id=make_document_id("papers", content_hash),
        source_path="papers/x.pdf",
        collection="papers",
        title="A Paper",
        format="pdf",
        parser="docling",
        content_hash=content_hash,
        sections=sections,
        ingested_at="2026-07-25T00:00:00Z",
    )


def test_small_section_becomes_one_chunk():
    doc = _doc((DocumentSection(title="Intro", level=1, text="A short introduction.", order=0),))
    chunks = StructureAwareChunker().chunk(doc)
    assert len(chunks) == 1
    assert chunks[0].text == "A short introduction."
    assert chunks[0].section_path == ("Intro",)
    assert chunks[0].chunk_id == f"{doc.document_id}:0"
    assert chunks[0].document_id == doc.document_id
    assert chunks[0].chunk_version == CHUNK_VERSION


def test_oversized_section_splits_with_overlap():
    long_text = " ".join(f"word{i}" for i in range(900))  # well over the 400-word target
    doc = _doc((DocumentSection(title="Method", level=1, text=long_text, order=0),))
    chunker = StructureAwareChunker(target_words=400, overlap_words=48)
    chunks = chunker.chunk(doc)
    assert len(chunks) >= 2
    first_words = chunks[0].text.split()
    second_words = chunks[1].text.split()
    # the tail of chunk 0 and the head of chunk 1 overlap
    assert first_words[-1] in second_words[: len(second_words) // 2 + 1]


def test_table_kind_is_carried_from_section_to_chunk():
    doc = _doc((DocumentSection(title="Results", level=1, kind="table", text="| a | b |", order=0),))
    chunks = StructureAwareChunker().chunk(doc)
    assert chunks[0].section_kind == "table"


def test_multiple_sections_get_sequential_order_and_ids():
    doc = _doc((
        DocumentSection(title="Intro", level=1, text="intro text", order=0),
        DocumentSection(title="Method", level=1, text="method text", order=1),
    ))
    chunks = StructureAwareChunker().chunk(doc)
    assert [c.order for c in chunks] == [0, 1]
    assert [c.chunk_id for c in chunks] == [f"{doc.document_id}:0", f"{doc.document_id}:1"]
