"""Normalized document and chunk schema for the RAG pipeline.

One consistent internal representation regardless of which parser produced
the source content (currently markitdown only; docling was removed as dead
weight, see harness.adapters.parsing.router) — see
docs/superpowers/specs/2026-07-25-rag-ingestion-retrieval-design.md for the
full rationale. Frozen dataclasses, matching harness.core.types style.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_document_id(collection: str, content_hash: str) -> str:
    return f"{collection}/{content_hash[:16]}"


@dataclass(frozen=True)
class ParsedContent:
    """Common pre-normalization shape both parser adapters render to."""
    text: str                  # markdown-ish text, parser's best structural rendering
    format: str
    parser: str
    page_count: int | None = None
    structure_hints: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DocumentSection:
    title: str                 # "" for untitled leading content
    level: int = 0             # heading depth; 0 = non-hierarchical
    kind: str = "prose"        # "prose" | "table" | "list" — open string, not an enum
    text: str = ""
    order: int = 0             # position in document, for stable ordering
    page_start: int | None = None
    page_end: int | None = None


@dataclass(frozen=True)
class NormalizedDocument:
    document_id: str           # make_document_id(collection, content_hash)
    source_path: str           # relative to data/raw/
    collection: str
    title: str
    format: str                # "pdf" | "docx" | "html" | ...
    parser: str                # "markitdown" (docling removed; may return as another value later)
    content_hash: str          # hash_bytes(raw source bytes)
    sections: tuple[DocumentSection, ...]
    ingested_at: str           # ISO timestamp
    extra: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str              # f"{document_id}:{index}"
    document_id: str
    collection: str
    text: str
    section_path: tuple[str, ...]   # heading breadcrumb, e.g. ("Results", "Ablations")
    section_kind: str = "prose"     # carried from the source DocumentSection
    order: int = 0
    page_start: int | None = None
    page_end: int | None = None
    source_path: str = ""
    embedding_model: str = ""       # filled in after embedding, not at chunk time
    chunk_version: int = 1
    created_at: str = ""
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float
