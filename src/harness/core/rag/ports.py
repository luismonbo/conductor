"""Protocols the ingestion pipeline (and, later, the serving pipeline) depend
on. Concrete implementations live in adapters/; core/ imports nothing outward.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from harness.core.rag.document import Chunk, NormalizedDocument, ParsedContent, ScoredChunk


class Parser(Protocol):
    async def parse(self, path: Path) -> ParsedContent: ...


class Normalizer(Protocol):
    async def normalize(
        self, parsed: ParsedContent, source_path: str, collection: str
    ) -> list[NormalizedDocument]: ...


class Chunker(Protocol):
    def chunk(self, document: NormalizedDocument) -> list[Chunk]: ...


@runtime_checkable
class Embedder(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


@runtime_checkable
class VectorStore(Protocol):
    async def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None: ...

    async def search(
        self,
        query_embedding: list[float],
        k: int,
        collection: str | None = None,
        filters: dict[str, str] | None = None,
    ) -> list[ScoredChunk]: ...

    async def delete(self, document_id: str) -> None: ...

    async def count(self, collection: str | None = None) -> int: ...


@runtime_checkable
class Retriever(Protocol):
    async def retrieve(
        self, query: str, k: int = 5, collection: str | None = None
    ) -> list[ScoredChunk]: ...
