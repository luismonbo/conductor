"""Per-section chunking that respects document structure, splitting oversized
sections with overlap. See the "Chunking" section of
docs/superpowers/specs/2026-07-25-rag-ingestion-retrieval-design.md."""
from __future__ import annotations

from harness.core.rag.document import Chunk, DocumentSection, NormalizedDocument

CHUNK_VERSION = 1


class StructureAwareChunker:
    def __init__(
        self, target_words: int = 400, overlap_words: int = 48, max_chars: int = 6000
    ) -> None:
        self._target_words = target_words
        self._overlap_words = overlap_words
        # Safety net, not the primary size control: word-count splitting
        # assumes normal whitespace. Text with few/no spaces (a real pdfminer
        # extraction failure mode, not just theoretical — see docs/devlog/010)
        # undercounts "words" and can slip an oversized piece past the
        # word-based split. 6000 stays comfortably under Milvus's 8192
        # VARCHAR max_length for the text field (milvus_store.py); pgvector
        # has no equivalent limit, so this is harmless there either way.
        self._max_chars = max_chars

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
            pieces = [section.text] if section.text else []
        else:
            pieces = []
            start = 0
            step = self._target_words - self._overlap_words
            while start < len(words):
                end = min(start + self._target_words, len(words))
                pieces.append(" ".join(words[start:end]))
                if end == len(words):
                    break
                start += step
        return [capped for piece in pieces for capped in self._cap_chars(piece)]

    def _cap_chars(self, text: str) -> list[str]:
        if len(text) <= self._max_chars:
            return [text]
        return [text[i : i + self._max_chars] for i in range(0, len(text), self._max_chars)]
