"""The RAG vector index, on Milvus or pgvector: what's indexed, what a query retrieves.

`docs` and `show` read the store directly: the VectorStore port has no "list"
method, and gets none for a debug tool. `search` runs the agent's own retrieval
(the embedder, store, k and per-document quota of search_documents), so it
shows what the model was given. `delete` is the one command that writes.
Nothing here creates a table or a collection.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path, PurePath
from typing import Any, Protocol

import psycopg
import yaml
from psycopg.conninfo import conninfo_to_dict
from psycopg.rows import dict_row
from psycopg.sql import SQL, Identifier
from pymilvus import MilvusClient
from pymilvus.exceptions import ConnectionConfigException

from harness.adapters.vectorstore.milvus_store import _chunk_from_entity
from harness.adapters.vectorstore.pgvector_store import _chunk_from_row
from harness.config.settings import Settings, get_settings
from harness.core.rag.document import Chunk

_EXAMPLES = """examples:
  uv run scripts/rag.py docs
  uv run scripts/rag.py show papers/5045
  uv run scripts/rag.py search "How many encoder layers?"
  uv run scripts/rag.py search "How many encoder layers?" --store pgvector --raw
  uv run scripts/rag.py delete --collection papers --document-id papers/50455611c3ed8b48
"""

# What ingest records per collection (harness.cli.ingest): the embedding model,
# the dimension and the stores. Relative to the repo root, like
# build.list_collections.
_INDEX_CONFIG_DIR = Path("data/index_config")

_STORES = ("milvus", "pgvector")
_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")  # also keeps quotes out of Milvus filters
_LOCAL_HOSTS = {"", "localhost", "127.0.0.1", "::1"}
_CELL_CHARS = 50


class ScriptError(Exception):
    """A failure reported as one line on stderr, with exit code 1."""


# --- formatting --------------------------------------------------------------


def _cut(text: str, width: int) -> str:
    """One line of at most `width` characters; newlines become spaces."""
    line = " ".join(text.split())
    return line if len(line) <= width else line[: width - 1] + "…"


def _print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str))


def _print_table(
    headers: list[str] | None,
    rows: list[list[str]],
    numeric: frozenset[int] = frozenset(),
    indent: str = "",
) -> None:
    """Aligned columns; `numeric` columns align right."""
    lines = [headers, *rows] if headers else rows
    if not lines:
        return
    widths = [max(len(line[i]) for line in lines) for i in range(len(lines[0]))]
    for line in lines:
        cells = [
            cell.rjust(widths[i]) if i in numeric else cell.ljust(widths[i])
            for i, cell in enumerate(line)
        ]
        print(indent + "  ".join(cells).rstrip())


def _first_line(exc: BaseException) -> str:
    text = str(exc).strip()
    return f"{type(exc).__name__}: {text.splitlines()[0]}" if text else type(exc).__name__


def _collection_name(text: str) -> str:
    if not _NAME.match(text):
        raise argparse.ArgumentTypeError("letters, digits, '.', '_' and '-' only")
    return text


def _resolve(prefix: str, candidates: list[str], what: str) -> str:
    """The one candidate `prefix` names: an exact id, or a unique prefix of one."""
    if prefix in candidates:
        return prefix
    matches = sorted({c for c in candidates if c.startswith(prefix)})
    if not matches:
        raise ScriptError(f"no {what} matching {prefix!r}")
    if len(matches) > 1:
        listed = "\n  ".join(matches[:10])
        raise ScriptError(f"{prefix!r} matches {len(matches)} {what}s:\n  {listed}")
    return matches[0]


def _cited_name(chunk: Chunk) -> str:
    """What a citation names, as search_documents does: the title, else the file name."""
    return chunk.title or PurePath(chunk.source_path).stem or "untitled"


def _section(chunk: Chunk) -> str:
    return " › ".join(chunk.section_path) or "-"


def _pages(chunk: Chunk) -> str:
    if chunk.page_start is None:
        return "-"
    if chunk.page_end is not None and chunk.page_end != chunk.page_start:
        return f"p.{chunk.page_start}–{chunk.page_end}"
    return f"p.{chunk.page_start}"


def _indented(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line for line in text.splitlines()) or prefix


# --- reading -----------------------------------------------------------------


@dataclass(frozen=True)
class Document:
    collection: str
    document_id: str
    title: str
    source_path: str
    chunks: int


class Reader(Protocol):
    def collections(self) -> list[str]: ...

    def documents(self, collection: str | None) -> list[Document]: ...

    def document_ids(self) -> list[str]: ...

    def chunks(self, document_id: str) -> list[Chunk]: ...

    def close(self) -> None: ...


def _group(rows: Iterable[tuple[str, str, dict[str, Any]]]) -> list[Document]:
    """Documents from (collection, document id, chunk payload) rows, one row per chunk."""
    counts: dict[tuple[str, str], int] = {}
    payloads: dict[tuple[str, str], dict[str, Any]] = {}
    for collection, document_id, payload in rows:
        key = (collection, document_id)
        counts[key] = counts.get(key, 0) + 1
        payloads.setdefault(key, payload)
    return [
        Document(
            collection=collection,
            document_id=document_id,
            title=payloads[(collection, document_id)].get("title") or "",
            source_path=payloads[(collection, document_id)].get("source_path") or "",
            chunks=count,
        )
        for (collection, document_id), count in counts.items()
    ]


class MilvusReader:
    _FIELDS = [
        "chunk_id",
        "document_id",
        "collection_name",
        "text",
        "metadata_json",
        "chunk_json",
    ]

    def __init__(self, uri: str, physical_collection: str) -> None:
        # Opening a missing Milvus Lite path would create it.
        if "://" not in uri and not Path(uri).exists():
            raise ScriptError(f"no Milvus Lite database at {uri} (run from the repo root?)")
        try:
            self._client = MilvusClient(uri=uri)
        except ConnectionConfigException as exc:
            if "local milvus" in str(exc.message):
                raise ScriptError(
                    f"Milvus Lite ({uri}) is open in another process; only one can use "
                    "it. Stop `make api` or any eval run, or use --store pgvector."
                ) from exc
            raise ScriptError(f"can't open Milvus at {uri}: {exc.message}") from exc
        self._collection = physical_collection
        if not self._client.has_collection(physical_collection):
            self._client.close()
            raise ScriptError(
                f"nothing ingested in milvus yet ({uri} has no {physical_collection!r})"
            )
        self._client.load_collection(collection_name=physical_collection)

    def _query(self, expr: str, fields: list[str]) -> list[dict[str, Any]]:
        return self._client.query(
            collection_name=self._collection, filter=expr, output_fields=fields
        )

    def collections(self) -> list[str]:
        rows = self._query("chunk_id != ''", ["collection_name"])
        return sorted({row["collection_name"] for row in rows})

    def documents(self, collection: str | None) -> list[Document]:
        expr = "chunk_id != ''" if collection is None else f'collection_name == "{collection}"'
        rows = self._query(expr, ["document_id", "collection_name", "chunk_json"])
        return _group(
            (
                row["collection_name"],
                row["document_id"],
                json.loads(row.get("chunk_json") or "{}"),
            )
            for row in rows
        )

    def document_ids(self) -> list[str]:
        rows = self._query("chunk_id != ''", ["document_id"])
        return sorted({row["document_id"] for row in rows})

    def chunks(self, document_id: str) -> list[Chunk]:
        # document_id comes from the store (document_ids), never straight from input.
        rows = self._query(f'document_id == "{document_id}"', self._FIELDS)
        chunks = [_chunk_from_entity(row["chunk_id"], row) for row in rows]
        return sorted(chunks, key=lambda chunk: chunk.order)

    def close(self) -> None:
        self._client.close()


def _server(dsn: str) -> str:
    """user@host:port/db, never the password."""
    try:
        info = conninfo_to_dict(dsn)
    except psycopg.Error:
        return "the configured server"
    host = info.get("host") or "localhost"
    port = info.get("port") or 5432
    return f"{info.get('user', '?')}@{host}:{port}/{info.get('dbname', '?')}"


def _is_local(dsn: str) -> bool:
    try:
        return (conninfo_to_dict(dsn).get("host") or "") in _LOCAL_HOSTS
    except psycopg.Error:
        return False


class PgVectorReader:
    def __init__(self, dsn: str, table: str) -> None:
        if not dsn:
            raise ScriptError("HARNESS_PGVECTOR_URL is empty: set it, or pass --dsn")
        try:
            self._conn = psycopg.connect(
                dsn, autocommit=True, connect_timeout=10, row_factory=dict_row
            )
        except psycopg.OperationalError as exc:
            hint = (
                "start it with `make infra`"
                if _is_local(dsn)
                else "check the server's firewall rule for your IP and sslmode=require"
            )
            raise ScriptError(
                f"can't reach Postgres at {_server(dsn)}: {_first_line(exc)} ({hint})"
            ) from exc
        self._table = Identifier(table)
        found = self._conn.execute("SELECT to_regclass(%s) AS name", (table,)).fetchone()
        if found is None or found["name"] is None:
            self._conn.close()
            raise ScriptError(
                f"nothing ingested in pgvector yet ({_server(dsn)} has no {table!r} table)"
            )

    def collections(self) -> list[str]:
        query = SQL("SELECT DISTINCT collection FROM {t}").format(t=self._table)
        return sorted(row["collection"] for row in self._conn.execute(query).fetchall())

    def documents(self, collection: str | None) -> list[Document]:
        query = SQL(
            "SELECT collection, document_id, count(*) AS chunks, "
            "min(chunk_json->>'title') AS title, "
            "min(chunk_json->>'source_path') AS source_path "
            "FROM {t} {where} GROUP BY collection, document_id"
        ).format(
            t=self._table,
            where=SQL("WHERE collection = %s") if collection else SQL(""),
        )
        rows = self._conn.execute(query, (collection,) if collection else ()).fetchall()
        return [
            Document(
                collection=row["collection"],
                document_id=row["document_id"],
                title=row["title"] or "",
                source_path=row["source_path"] or "",
                chunks=row["chunks"],
            )
            for row in rows
        ]

    def document_ids(self) -> list[str]:
        query = SQL("SELECT DISTINCT document_id FROM {t}").format(t=self._table)
        return sorted(row["document_id"] for row in self._conn.execute(query).fetchall())

    def chunks(self, document_id: str) -> list[Chunk]:
        query = SQL(
            "SELECT chunk_id, document_id, collection, text, metadata, chunk_json "
            "FROM {t} WHERE document_id = %s"
        ).format(t=self._table)
        rows = self._conn.execute(query, (document_id,)).fetchall()
        return sorted((_chunk_from_row(row) for row in rows), key=lambda chunk: chunk.order)

    def close(self) -> None:
        self._conn.close()


def _open_reader(store: str, settings: Settings) -> Reader:
    if store == "milvus":
        return MilvusReader(settings.milvus_uri, settings.milvus_collection)
    return PgVectorReader(settings.pgvector_url, settings.pgvector_table)


def _manifest(collection: str) -> dict[str, Any] | None:
    path = _INDEX_CONFIG_DIR / f"{collection}.yaml"
    if not path.is_file():
        return None
    loaded = yaml.safe_load(path.read_text())
    return loaded if isinstance(loaded, dict) else None


def _embedding(manifest: dict[str, Any] | None) -> str:
    if manifest is None:
        return "no ingest manifest"
    model = manifest.get("embedding_model", "?")
    return f"{model} ({manifest.get('embedding_dimension', '?')})"


# --- commands ----------------------------------------------------------------


def cmd_docs(args: argparse.Namespace, settings: Settings, store: str) -> None:
    reader = _open_reader(store, settings)
    try:
        documents = reader.documents(args.collection)
    finally:
        reader.close()
    by_collection: dict[str, list[Document]] = {}
    for document in documents:
        by_collection.setdefault(document.collection, []).append(document)
    groups = [
        (
            name,
            sorted(docs, key=lambda d: (d.title or d.source_path or d.document_id).lower()),
            _manifest(name),
        )
        for name, docs in sorted(by_collection.items())
    ]
    if args.json:
        _print_json(
            [
                {
                    "collection": name,
                    "store": store,
                    "embedding_model": (manifest or {}).get("embedding_model"),
                    "embedding_dimension": (manifest or {}).get("embedding_dimension"),
                    "documents": [
                        {key: value for key, value in asdict(d).items() if key != "collection"}
                        for d in docs
                    ],
                }
                for name, docs, manifest in groups
            ]
        )
        return
    if not groups:
        where = f"collection {args.collection!r}" if args.collection else "any collection"
        print(f"No chunks in {where} in {store}.")
        return
    for name, docs, manifest in groups:
        chunks = sum(d.chunks for d in docs)
        print(
            f"{name} · {store} · {len(docs)} documents, {chunks:,} chunks · "
            f"{_embedding(manifest)}"
        )
        _print_table(
            None,
            [
                [
                    d.document_id,
                    _cut(d.title or "-", _CELL_CHARS),
                    _cut(d.source_path or "-", _CELL_CHARS),
                    f"{d.chunks:,} chunks",
                ]
                for d in docs
            ],
            numeric=frozenset({3}),
            indent="  ",
        )
        print()


def _show_chunk(chunk: Chunk, *, as_json: bool) -> None:
    if as_json:
        _print_json(asdict(chunk))
        return
    fields = [
        ("chunk", chunk.chunk_id),
        ("document", chunk.document_id),
        ("collection", chunk.collection),
        ("title", chunk.title or "-"),
        ("source", chunk.source_path or "-"),
        ("section", _section(chunk)),
        ("kind", chunk.section_kind),
        ("pages", _pages(chunk)),
        ("order", str(chunk.order)),
        ("embedding", chunk.embedding_model or "-"),
        ("version", str(chunk.chunk_version)),
        ("created", chunk.created_at or "-"),
        ("metadata", json.dumps(chunk.metadata, ensure_ascii=False)),
    ]
    _print_table(None, [[name, value] for name, value in fields])
    print()
    print(chunk.text)


def _show_document(
    document_id: str, chunks: list[Chunk], *, full: bool, as_json: bool
) -> None:
    if as_json:
        _print_json({"document_id": document_id, "chunks": [asdict(c) for c in chunks]})
        return
    print(f"{document_id} · {_cited_name(chunks[0])} · {len(chunks)} chunks")
    print()
    if full:
        for chunk in chunks:
            print(f"[{chunk.order}] {chunk.chunk_id} · {_section(chunk)} · {_pages(chunk)}")
            print(_indented(chunk.text))
            print()
        return
    _print_table(
        ["order", "chunk", "section", "pages", "text"],
        [
            [str(c.order), c.chunk_id, _cut(_section(c), 40), _pages(c), _cut(c.text, 80)]
            for c in chunks
        ],
        numeric=frozenset({0}),
    )


def cmd_show(args: argparse.Namespace, settings: Settings, store: str) -> None:
    prefix, _, index = args.id.partition(":")
    reader = _open_reader(store, settings)
    try:
        document_id = _resolve(prefix, reader.document_ids(), "document")
        chunks = reader.chunks(document_id)
    finally:
        reader.close()
    if not index:
        _show_document(document_id, chunks, full=args.full, as_json=args.json)
        return
    chunk_id = f"{document_id}:{index}"
    chunk = next((c for c in chunks if c.chunk_id == chunk_id), None)
    if chunk is None:
        raise ScriptError(f"no chunk {chunk_id!r}; {document_id} has {len(chunks)} chunks")
    _show_chunk(chunk, as_json=args.json)


# --- command line ------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--store", choices=_STORES, help="default: HARNESS_RAG_VECTOR_STORE_BACKEND"
    )
    common.add_argument(
        "--dsn",
        help="pgvector connection string (default: HARNESS_PGVECTOR_URL), "
        "e.g. an Azure server with sslmode=require",
    )
    parser = argparse.ArgumentParser(
        prog="rag.py",
        description="The RAG vector index on Milvus or pgvector.",
        epilog=_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    docs = commands.add_parser(
        "docs", parents=[common], help="collections and their documents"
    )
    docs.add_argument("--collection", type=_collection_name, help="only this collection")
    docs.add_argument("--json", action="store_true", help="everything as JSON")

    show = commands.add_parser(
        "show", parents=[common], help="a document's chunks, or one chunk in full"
    )
    show.add_argument(
        "id",
        help="a document id or unique prefix (papers/5045), or a chunk (papers/5045:12)",
    )
    show.add_argument("--full", action="store_true", help="every chunk's whole text")
    show.add_argument("--json", action="store_true", help="everything as JSON")

    return parser


def _load_settings(args: argparse.Namespace) -> Settings:
    try:
        settings = get_settings()
    except ValueError as exc:
        raise ScriptError(f"settings: {exc}") from exc
    return settings.model_copy(update={"pgvector_url": args.dsn}) if args.dsn else settings


def _store(args: argparse.Namespace, settings: Settings) -> str:
    store = args.store or settings.rag_vector_store_backend
    if store not in _STORES:
        raise ScriptError(
            f"HARNESS_RAG_VECTOR_STORE_BACKEND={store} lives inside another process; "
            "pass --store milvus or --store pgvector"
        )
    if args.dsn and store != "pgvector":
        raise ScriptError("--dsn needs --store pgvector")
    return store


Command = Callable[[argparse.Namespace, Settings, str], None]

_COMMANDS: dict[str, Command] = {
    "docs": cmd_docs,
    "show": cmd_show,
}


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    # Milvus Lite and pymilvus log their own tracebacks for a busy or missing
    # file; the one-line message below says the same thing.
    for name in ("milvus_lite", "pymilvus"):
        logging.getLogger(name).setLevel(logging.CRITICAL)
    try:
        settings = _load_settings(args)
        _COMMANDS[args.command](args, settings, _store(args, settings))
    except ScriptError as exc:
        print(f"rag.py: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
