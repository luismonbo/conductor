"""IngestionPipeline: load -> parse -> normalize -> chunk -> embed -> upsert.
Written against protocols only — orchestration/build.py wires concrete
adapters in. A single bad file must not abort a collection run."""

from __future__ import annotations

import dataclasses
from datetime import datetime, timezone
from pathlib import Path

from harness.core.rag.document import Chunk
from harness.core.rag.ports import Chunker, Embedder, Normalizer, Parser, VectorStore


@dataclasses.dataclass(frozen=True)
class IngestResult:
    source_path: str
    document_ids: tuple[str, ...] = ()
    chunk_count: int = 0
    parser_used: str = ""
    error: str | None = None


class IngestionPipeline:
    def __init__(
        self,
        parser: Parser,
        normalizer: Normalizer,
        chunker: Chunker,
        embedder: Embedder,
        vector_stores: list[VectorStore],
        tracer=None,  # TraceCollector | None — typed loosely to avoid a hard import cycle risk;
        # see harness.observability.tracer.TraceCollector for the real shape
    ) -> None:
        self._parser = parser
        self._normalizer = normalizer
        self._chunker = chunker
        self._embedder = embedder
        self._vector_stores = vector_stores
        self._tracer = tracer

    async def _stage(self, stage: str, source_path: str, count: int) -> None:
        if self._tracer is not None:
            await self._tracer(
                "ingest_stage",
                {"stage": stage, "source_path": source_path, "count": count},
            )

    async def ingest_file(self, path: Path, collection: str) -> IngestResult:
        source_path = str(path)
        try:
            parsed = await self._parser.parse(path)
            await self._stage("parse", source_path, 1)

            documents = await self._normalizer.normalize(parsed, source_path, collection)
            await self._stage("normalize", source_path, len(documents))

            all_chunks: list[Chunk] = []
            for document in documents:
                all_chunks.extend(self._chunker.chunk(document))
            await self._stage("chunk", source_path, len(all_chunks))

            # Delete must happen only after a successful embed, or a transient
            # embed failure (e.g. a transient Azure error) leaves the document
            # deleted-but-not-replaced — absent from the index instead of
            # merely stale. A file edited down to zero chunks has no embed
            # step to gate on, so it still deletes unconditionally: otherwise
            # re-ingesting it to empty would leave its old chunks behind
            # forever (idempotency guarantee).
            document_ids = [d.document_id for d in documents]
            if all_chunks:
                embeddings = await self._embedder.embed([c.text for c in all_chunks])
                await self._stage("embed", source_path, len(embeddings))

                stamped = [
                    dataclasses.replace(
                        chunk,
                        embedding_model=getattr(
                            self._embedder, "model_id", type(self._embedder).__name__
                        ),
                        created_at=datetime.now(timezone.utc).isoformat(),
                    )
                    for chunk in all_chunks
                ]
                for store in self._vector_stores:
                    for document_id in document_ids:
                        await store.delete(document_id)
                    await store.upsert(stamped, embeddings)
                await self._stage("upsert", source_path, len(stamped))
            else:
                # Zero chunks: still delete old chunks so an edit-to-empty
                # leaves no orphans behind.
                for store in self._vector_stores:
                    for document_id in document_ids:
                        await store.delete(document_id)

            result = IngestResult(
                source_path=source_path,
                document_ids=tuple(d.document_id for d in documents),
                chunk_count=len(all_chunks),
                parser_used=parsed.parser,
            )
            if self._tracer is not None:
                await self._tracer(
                    "ingest_file_complete",
                    {
                        "source_path": source_path,
                        "document_ids": list(result.document_ids),
                        "chunk_count": result.chunk_count,
                        "parser_used": result.parser_used,
                    },
                )
            return result
        except Exception as exc:
            if self._tracer is not None:
                await self._tracer(
                    "ingest_file_failed", {"source_path": source_path, "error": str(exc)}
                )
            return IngestResult(source_path=source_path, error=str(exc))

    async def ingest_collection(
        self, raw_dir: Path, collection: str
    ) -> list[IngestResult]:
        results = []
        for path in sorted(raw_dir.iterdir()):
            if path.is_file() and not path.name.startswith("."):
                results.append(await self.ingest_file(path, collection))
        return results
