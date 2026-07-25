"""Per-section chunking that respects document structure, splitting oversized
sections with overlap. See the "Chunking" section of
docs/superpowers/specs/2026-07-25-rag-ingestion-retrieval-design.md."""
from __future__ import annotations

from harness.core.rag.document import Chunk, DocumentSection, NormalizedDocument

CHUNK_VERSION = 1


class StructureAwareChunker:
    def __init__(self, target_words: int = 400, overlap_words: int = 48) -> None:
        self._target_words = target_words
        self._overlap_words = overlap_words

    def chunk(self, document: NormalizedDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        index = 0
        for section in sorted(document.sections, key=lambda s: s.order):
            for text in self._split_section(section):
                chunks.append(
                    Chunk(
                        chunk_id=f"{document.document_id}:{index}",
                        document_id=document.document_id,
                        collection=document.collection,
                        text=text,
                        section_path=(section.title,) if section.title else (),
                        section_kind=section.kind,
                        order=index,
                        page_start=section.page_start,
                        page_end=section.page_end,
                        source_path=document.source_path,
                        chunk_version=CHUNK_VERSION,
                        created_at=document.ingested_at,
                    )
                )
                index += 1
        return chunks

    def _split_section(self, section: DocumentSection) -> list[str]:
        words = section.text.split()
        if len(words) <= self._target_words:
            return [section.text] if section.text else []

        pieces: list[str] = []
        start = 0
        step = self._target_words - self._overlap_words
        while start < len(words):
            end = min(start + self._target_words, len(words))
            pieces.append(" ".join(words[start:end]))
            if end == len(words):
                break
            start += step
        return pieces
