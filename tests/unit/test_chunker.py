from __future__ import annotations

from harness.adapters.chunking.structure_aware import CHUNK_VERSION, StructureAwareChunker
from harness.core.rag.document import DocumentSection, NormalizedDocument, hash_bytes, make_document_id


def _doc(sections: tuple[DocumentSection, ...]) -> NormalizedDocument:
    content_hash = hash_bytes(b"fixture")
    return NormalizedDocument(
        document_id=make_document_id("papers", "papers/x.pdf"),
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


def test_section_with_no_whitespace_is_split_by_char_length():
    # Simulates a pdfminer extraction failure where words run together with no
    # spaces (see docs/devlog/010) — word-count splitting alone can't catch
    # this, since len(text.split()) stays tiny (or 1) no matter how long the
    # text actually is. The char cap is the safety net.
    glued_text = "word" * 3000  # 12000 chars, but a single "word" per .split()
    doc = _doc((DocumentSection(title="Broken", level=1, text=glued_text, order=0),))
    chunker = StructureAwareChunker(max_chars=6000)

    chunks = chunker.chunk(doc)

    assert len(chunks) >= 2
    assert all(len(c.text) <= 6000 for c in chunks)
    assert "".join(c.text for c in chunks) == glued_text


def test_multiple_sections_get_sequential_order_and_ids():
    doc = _doc((
        DocumentSection(title="Intro", level=1, text="intro text", order=0),
        DocumentSection(title="Method", level=1, text="method text", order=1),
    ))
    chunks = StructureAwareChunker().chunk(doc)
    assert [c.order for c in chunks] == [0, 1]
    assert [c.chunk_id for c in chunks] == [f"{doc.document_id}:0", f"{doc.document_id}:1"]


def test_section_path_is_full_ancestor_breadcrumb():
    doc = _doc((
        DocumentSection(title="Attention Is All You Need", level=1, text="a", order=0),
        DocumentSection(title="3 Model Architecture", level=2, text="b", order=1),
        DocumentSection(title="3.2 Attention", level=3, text="c", order=2),
        DocumentSection(title="4 Training", level=2, text="d", order=3),
    ))
    paths = [c.section_path for c in StructureAwareChunker().chunk(doc)]
    assert paths[0] == ("Attention Is All You Need",)
    assert paths[1] == ("Attention Is All You Need", "3 Model Architecture")
    assert paths[2] == ("Attention Is All You Need", "3 Model Architecture", "3.2 Attention")
    assert paths[3] == ("Attention Is All You Need", "4 Training")


def test_untitled_leading_section_has_empty_breadcrumb():
    doc = _doc((DocumentSection(title="", level=0, text="lead", order=0),))
    assert StructureAwareChunker().chunk(doc)[0].section_path == ()
