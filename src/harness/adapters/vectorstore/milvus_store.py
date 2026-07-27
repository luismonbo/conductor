"""Milvus-backed VectorStore. `uri` pointed at a local file path uses Milvus
Lite (embedded, no server); an http(s) URI uses a real Milvus/Zilliz server —
same MilvusClient API either way, so upgrading later is a config change.

Structural Chunk fields ride along in a `chunk_json` VARCHAR field for the
same reason as the pgvector adapter: a retrieved Chunk must be
indistinguishable from the stored one. `metadata_json` stays separate — it is
what search filters query.
"""
from __future__ import annotations

import json

from pymilvus import DataType, MilvusClient

from harness.core.rag.document import Chunk, ScoredChunk

_TEXT_FIELD_MAX_LEN = 8192
_ID_FIELD_MAX_LEN = 256
_METADATA_FIELD_MAX_LEN = 4096
_CHUNK_JSON_FIELD_MAX_LEN = 4096


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


def _chunk_from_entity(chunk_id: str, entity: dict) -> Chunk:
    payload = json.loads(entity["chunk_json"]) if entity.get("chunk_json") else {}
    metadata = json.loads(entity["metadata_json"]) if entity.get("metadata_json") else {}
    return Chunk(
        chunk_id=chunk_id,
        document_id=entity["document_id"],
        collection=entity["collection_name"],
        text=entity["text"],
        section_path=tuple(payload.get("section_path", ())),
        section_kind=payload.get("section_kind", "prose"),
        order=payload.get("order", 0),
        page_start=payload.get("page_start"),
        page_end=payload.get("page_end"),
        source_path=payload.get("source_path", ""),
        embedding_model=payload.get("embedding_model", ""),
        chunk_version=payload.get("chunk_version", 1),
        created_at=payload.get("created_at", ""),
        metadata=metadata,
    )


class MilvusStore:
    def __init__(self, uri: str, collection: str = "rag_chunks", vector_size: int = 768) -> None:
        self._collection = collection
        self._vector_size = vector_size
        self._client = MilvusClient(uri=uri)
        self._ready = False

    def _ensure_ready(self) -> None:
        if self._ready:
            return
        if self._collection in self._client.list_collections():
            # Creating a collection loads it, so a writer never reaches here.
            # A reader in a fresh process creates nothing and would search an
            # unloaded collection: "state 'released'; call load() before
            # search/get/query". load_collection is idempotent.
            self._client.load_collection(collection_name=self._collection)
        else:
            schema = self._client.create_schema(auto_id=False, enable_dynamic_field=False)
            schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=_ID_FIELD_MAX_LEN)
            schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self._vector_size)
            schema.add_field("document_id", DataType.VARCHAR, max_length=_ID_FIELD_MAX_LEN)
            schema.add_field("collection_name", DataType.VARCHAR, max_length=_ID_FIELD_MAX_LEN)
            schema.add_field("text", DataType.VARCHAR, max_length=_TEXT_FIELD_MAX_LEN)
            schema.add_field("metadata_json", DataType.VARCHAR, max_length=_METADATA_FIELD_MAX_LEN)
            schema.add_field("chunk_json", DataType.VARCHAR, max_length=_CHUNK_JSON_FIELD_MAX_LEN)

            index_params = self._client.prepare_index_params()
            index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")

            self._client.create_collection(
                collection_name=self._collection, schema=schema, index_params=index_params,
            )
        self._ready = True

    async def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        self._ensure_ready()
        rows = [
            {
                "chunk_id": chunk.chunk_id,
                "vector": embedding,
                "document_id": chunk.document_id,
                "collection_name": chunk.collection,
                "text": chunk.text,
                "metadata_json": json.dumps(dict(chunk.metadata)),
                "chunk_json": json.dumps(_payload(chunk)),
            }
            for chunk, embedding in zip(chunks, embeddings)
        ]
        self._client.upsert(collection_name=self._collection, data=rows)

    async def search(
        self,
        query_embedding: list[float],
        k: int,
        collection: str | None = None,
        filters: dict[str, str] | None = None,
    ) -> list[ScoredChunk]:
        self._ensure_ready()
        filter_parts = []
        if collection is not None:
            filter_parts.append(f'collection_name == "{collection}"')
        # `filters` metadata matching happens post-query below — Milvus scalar
        # filter expressions over a JSON-string field aren't used here to keep
        # this adapter's query syntax simple; revisit if filter volume grows.
        filter_expr = " and ".join(filter_parts) if filter_parts else ""

        raw = self._client.search(
            collection_name=self._collection,
            data=[query_embedding],
            limit=k if not filters else max(k * 4, k),
            filter=filter_expr,
            output_fields=["document_id", "collection_name", "text", "metadata_json", "chunk_json"],
            search_params={"metric_type": "COSINE", "params": {}},
        )

        results: list[ScoredChunk] = []
        for hit in raw[0]:
            entity = hit["entity"]
            # pymilvus 3.x keys a Hit by the primary-key field name, not "id";
            # the .id attribute is the version-stable accessor.
            chunk = _chunk_from_entity(hit.id, entity)
            if filters and not all(chunk.metadata.get(key) == value for key, value in filters.items()):
                continue
            results.append(ScoredChunk(chunk=chunk, score=float(hit["distance"])))
            if len(results) == k:
                break
        return results

    async def delete(self, document_id: str) -> None:
        self._ensure_ready()
        self._client.delete(collection_name=self._collection, filter=f'document_id == "{document_id}"')

    async def count(self, collection: str | None = None) -> int:
        self._ensure_ready()
        if collection is None:
            stats = self._client.get_collection_stats(collection_name=self._collection)
            return int(stats["row_count"])
        rows = self._client.query(
            collection_name=self._collection,
            filter=f'collection_name == "{collection}"',
            output_fields=["chunk_id"],
        )
        return len(rows)
