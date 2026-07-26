"""pgvector-backed VectorStore for RAG chunks — a distinct table from the
still-unimplemented PgVectorLongTerm (adapters/memory/pgvector_store.py),
sharing the same Postgres instance and vector extension. Lazily bootstraps
its own schema on first use so callers (and tests) don't need a separate
ensure_schema() call before the store is usable.

Structural Chunk fields (section_path, source_path, page numbers, ...) live in
a `chunk_json` payload column rather than one column each: a retrieved Chunk
must be indistinguishable from the stored one — assemble_prompt() renders
source_path and section_path into every grounded prompt — and this table
bootstraps with CREATE TABLE IF NOT EXISTS, so a per-field column would need a
migration mechanism the project doesn't have the first time Chunk gains a
field. `metadata` stays its own column: it is what search filters query.
"""
from __future__ import annotations

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector_async
from psycopg.rows import dict_row
from psycopg.sql import SQL, Identifier, Literal
from psycopg.types.json import Jsonb

from harness.core.rag.document import Chunk, ScoredChunk

# Chunk fields that have their own column; everything else round-trips via chunk_json.
_COLUMN_FIELDS = {"chunk_id", "document_id", "collection", "text", "metadata"}


def _payload(chunk: Chunk) -> dict:
    return {
        "section_path": list(chunk.section_path),
        "section_kind": chunk.section_kind,
        "order": chunk.order,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "source_path": chunk.source_path,
        "embedding_model": chunk.embedding_model,
        "chunk_version": chunk.chunk_version,
        "created_at": chunk.created_at,
    }


def _chunk_from_row(row: dict) -> Chunk:
    payload = row.get("chunk_json") or {}
    return Chunk(
        chunk_id=row["chunk_id"],
        document_id=row["document_id"],
        collection=row["collection"],
        text=row["text"],
        section_path=tuple(payload.get("section_path", ())),
        section_kind=payload.get("section_kind", "prose"),
        order=payload.get("order", 0),
        page_start=payload.get("page_start"),
        page_end=payload.get("page_end"),
        source_path=payload.get("source_path", ""),
        embedding_model=payload.get("embedding_model", ""),
        chunk_version=payload.get("chunk_version", 1),
        created_at=payload.get("created_at", ""),
        metadata=dict(row["metadata"] or {}),
    )


class PgVectorStore:
    def __init__(self, dsn: str, table: str = "rag_chunks", vector_size: int = 768) -> None:
        self._dsn = dsn
        self._table = table
        self._vector_size = vector_size
        self._ready = False

    async def _connect(self) -> psycopg.AsyncConnection:
        conn = await psycopg.AsyncConnection.connect(self._dsn, autocommit=True)
        if not self._ready:
            await self._ensure_schema(conn)
            self._ready = True
        await register_vector_async(conn)
        return conn

    async def _ensure_schema(self, conn: psycopg.AsyncConnection) -> None:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await conn.execute(
            SQL(
                "CREATE TABLE IF NOT EXISTS {table} ("
                "chunk_id text PRIMARY KEY, "
                "document_id text NOT NULL, "
                "collection text NOT NULL, "
                "text text NOT NULL, "
                "metadata jsonb NOT NULL DEFAULT '{{}}'::jsonb, "
                "chunk_json jsonb NOT NULL DEFAULT '{{}}'::jsonb, "
                "embedding vector({dim}) NOT NULL, "
                "created_at timestamptz NOT NULL DEFAULT now())"
            ).format(table=Identifier(self._table), dim=Literal(self._vector_size))
        )
        await conn.execute(
            SQL(
                "CREATE INDEX IF NOT EXISTS {idx} ON {table} "
                "USING hnsw (embedding vector_cosine_ops)"
            ).format(idx=Identifier(f"{self._table}_hnsw_idx"), table=Identifier(self._table))
        )

    async def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        conn = await self._connect()
        async with conn:
            stmt = SQL(
                "INSERT INTO {table} "
                "(chunk_id, document_id, collection, text, metadata, chunk_json, embedding) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (chunk_id) DO UPDATE SET "
                "document_id = EXCLUDED.document_id, collection = EXCLUDED.collection, "
                "text = EXCLUDED.text, metadata = EXCLUDED.metadata, "
                "chunk_json = EXCLUDED.chunk_json, embedding = EXCLUDED.embedding"
            ).format(table=Identifier(self._table))
            async with conn.cursor() as cur:
                for chunk, embedding in zip(chunks, embeddings):
                    await cur.execute(
                        stmt,
                        (
                            chunk.chunk_id, chunk.document_id, chunk.collection,
                            chunk.text, Jsonb(dict(chunk.metadata)), Jsonb(_payload(chunk)),
                            Vector(embedding),
                        ),
                    )

    async def search(
        self,
        query_embedding: list[float],
        k: int,
        collection: str | None = None,
        filters: dict[str, str] | None = None,
    ) -> list[ScoredChunk]:
        conn = await self._connect()
        async with conn:
            where_parts = []
            params: list[object] = []
            if collection is not None:
                where_parts.append(SQL("collection = %s"))
                params.append(collection)
            for key, value in (filters or {}).items():
                where_parts.append(SQL("metadata ->> {} = %s").format(Literal(key)))
                params.append(value)
            where_sql = SQL("WHERE " + " AND ".join(["{}"] * len(where_parts))).format(*where_parts) \
                if where_parts else SQL("")

            stmt = SQL(
                "SELECT chunk_id, document_id, collection, text, metadata, chunk_json, "
                "embedding <=> %s AS distance "
                "FROM {table} {where} "
                "ORDER BY embedding <=> %s LIMIT %s"
            ).format(table=Identifier(self._table), where=where_sql)

            query_vector = Vector(query_embedding)
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(stmt, [query_vector, *params, query_vector, k])
                rows = await cur.fetchall()

        return [
            ScoredChunk(chunk=_chunk_from_row(row), score=1.0 - row["distance"])
            for row in rows
        ]

    async def delete(self, document_id: str) -> None:
        conn = await self._connect()
        async with conn:
            stmt = SQL("DELETE FROM {table} WHERE document_id = %s").format(
                table=Identifier(self._table)
            )
            await conn.execute(stmt, (document_id,))

    async def count(self, collection: str | None = None) -> int:
        conn = await self._connect()
        async with conn:
            if collection is None:
                stmt = SQL("SELECT count(*) FROM {table}").format(table=Identifier(self._table))
                async with conn.cursor() as cur:
                    await cur.execute(stmt)
                    (n,) = await cur.fetchone()
            else:
                stmt = SQL("SELECT count(*) FROM {table} WHERE collection = %s").format(
                    table=Identifier(self._table)
                )
                async with conn.cursor() as cur:
                    await cur.execute(stmt, (collection,))
                    (n,) = await cur.fetchone()
        return int(n)

    async def drop(self) -> None:
        """Test-only cleanup — not part of the VectorStore protocol."""
        conn = await psycopg.AsyncConnection.connect(self._dsn, autocommit=True)
        async with conn:
            await conn.execute(SQL("DROP TABLE IF EXISTS {table}").format(table=Identifier(self._table)))
