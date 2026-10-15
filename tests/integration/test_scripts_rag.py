"""scripts/rag.py on both vector stores: Milvus Lite always, pgvector when Postgres is up.

The index is seeded through the adapters with a keyword embedder (one dimension
per word), so every ranking is known in advance and no test calls Azure.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import psycopg
import pytest
import yaml

import harness.orchestration.build as build
import scripts.rag as rag
from harness.config.settings import Settings
from harness.core.rag.document import Chunk
from harness.orchestration.build import build_vector_store

_PG_DSN = os.environ.get(
    "HARNESS_TEST_PGVECTOR_DSN", "postgresql://harness:harness@localhost:5432/harness"
)
_VOCABULARY = ("encoder", "decoder", "attention")
DIM = 1 + len(_VOCABULARY)
TRANSFORMERS = "papers/aaaa000000000001"
RECURRENT = "papers/bbbb000000000002"
NOTES = "notes/cccc000000000003"


class KeywordEmbedder:
    """One dimension per vocabulary word plus a constant, so no vector is zero."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [
            [1.0, *(float(text.lower().count(word)) for word in _VOCABULARY)]
            for text in texts
        ]


def _chunk(
    document_id: str,
    order: int,
    text: str,
    title: str,
    section: tuple[str, ...],
    pages: tuple[int, int] | None,
) -> Chunk:
    collection = document_id.split("/")[0]
    return Chunk(
        chunk_id=f"{document_id}:{order}",
        document_id=document_id,
        collection=collection,
        text=text,
        section_path=section,
        order=order,
        page_start=pages[0] if pages else None,
        page_end=pages[1] if pages else None,
        source_path=f"{collection}/{title.lower().replace(' ', '-')}.pdf",
        title=title,
        embedding_model="keyword-test",
        created_at="2026-10-15T00:00:00+00:00",
    )


CHUNKS = [
    _chunk(TRANSFORMERS, 0, "encoder encoder encoder stack", "Transformers",
           ("Model", "Encoder"), (3, 3)),
    _chunk(TRANSFORMERS, 1, "encoder encoder layers", "Transformers",
           ("Model", "Encoder"), (3, 4)),
    _chunk(TRANSFORMERS, 2, "decoder only", "Transformers", ("Model", "Decoder"), (5, 5)),
    _chunk(RECURRENT, 0, "encoder attention", "Recurrent nets", ("Background",), None),
    _chunk(NOTES, 0, "attention attention", "Notes", ("Notes",), None),
]


def _require_postgres() -> None:
    try:
        psycopg.connect(_PG_DSN, connect_timeout=3).close()
    except psycopg.OperationalError as exc:
        pytest.skip(f"Postgres unreachable at {_PG_DSN}: {exc}")


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "embedding_backend": "openai_compatible",
        "embedding_model": "keyword-test",
        "embedding_dimension": DIM,
        "rag_collection": "papers",
        "rag_k": 2,
        "rag_per_document_k": 1,
        "rag_overfetch": 5,
        **overrides,
    }
    return Settings(_env_file=None, auth_enabled=False, input_guard_enabled=False, **values)


def _use(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> None:
    monkeypatch.setattr(rag, "get_settings", lambda: settings)


def _write_manifest(directory: Path, collection: str, model: str, dimension: int) -> None:
    manifest = {
        "collection": collection,
        "embedding_model": model,
        "embedding_dimension": dimension,
    }
    (directory / f"{collection}.yaml").write_text(yaml.safe_dump(manifest))


@pytest.fixture(params=["milvus", "pgvector"])
def index(request, tmp_path, monkeypatch):
    store = request.param
    if store == "milvus":
        settings = _settings(
            rag_vector_store_backend="milvus", milvus_uri=str(tmp_path / "milvus.db")
        )
    else:
        _require_postgres()
        settings = _settings(
            rag_vector_store_backend="pgvector",
            pgvector_url=_PG_DSN,
            pgvector_table=f"test_rag_script_{uuid.uuid4().hex[:8]}",
        )
    seeded = build_vector_store(settings, store)
    vectors = asyncio.run(KeywordEmbedder().embed([chunk.text for chunk in CHUNKS]))
    asyncio.run(seeded.upsert(CHUNKS, vectors))
    manifests = tmp_path / "index_config"
    manifests.mkdir()
    _write_manifest(manifests, "papers", "keyword-test", DIM)
    _use(monkeypatch, settings)
    monkeypatch.setattr(rag, "_INDEX_CONFIG_DIR", manifests)
    monkeypatch.setattr(build, "build_embedder", lambda settings: KeywordEmbedder())
    yield SimpleNamespace(store=store, settings=settings, manifests=manifests)
    if store == "pgvector":
        asyncio.run(seeded.drop())


def run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    code = rag.main(list(argv))
    out, err = capsys.readouterr()
    return code, out, err


# --- docs --------------------------------------------------------------------


def test_docs_lists_documents_by_collection(index, capsys):
    code, out, _ = run(capsys, "docs", "--json")
    assert code == 0
    collections = {group["collection"]: group for group in json.loads(out)}
    assert sorted(collections) == ["notes", "papers"]
    papers = collections["papers"]
    assert papers["store"] == index.store
    assert (papers["embedding_model"], papers["embedding_dimension"]) == ("keyword-test", DIM)
    assert [(d["document_id"], d["title"], d["chunks"]) for d in papers["documents"]] == [
        (RECURRENT, "Recurrent nets", 1),
        (TRANSFORMERS, "Transformers", 3),
    ]
    assert collections["notes"]["embedding_model"] is None  # no manifest


def test_docs_text_for_one_collection(index, capsys):
    code, out, _ = run(capsys, "docs", "--collection", "papers")
    assert code == 0
    assert f"papers · {index.store} · 2 documents, 4 chunks · keyword-test ({DIM})" in out
    assert "papers/transformers.pdf" in out
    assert "notes" not in out


# --- opening a store -----------------------------------------------------------


def test_a_missing_milvus_file_is_not_created(tmp_path, monkeypatch, capsys):
    missing = tmp_path / "absent.db"
    _use(monkeypatch, _settings(rag_vector_store_backend="milvus", milvus_uri=str(missing)))
    code, _, err = run(capsys, "docs")
    assert code == 1
    assert "no Milvus Lite database at" in err
    assert not missing.exists()


_HOLDER = (
    "import sys\n"
    "from pymilvus import MilvusClient\n"
    "client = MilvusClient(uri=sys.argv[1])\n"
    "print('held', flush=True)\n"
    "sys.stdin.readline()\n"
    "client.close()\n"
)


def test_milvus_open_in_another_process(tmp_path, monkeypatch, capsys):
    uri = str(tmp_path / "held.db")
    holder = subprocess.Popen(
        [sys.executable, "-c", _HOLDER, uri],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "held"
        _use(monkeypatch, _settings(rag_vector_store_backend="milvus", milvus_uri=uri))
        code, _, err = run(capsys, "docs")
    finally:
        holder.communicate("\n", timeout=60)
    assert code == 1
    assert "is open in another process" in err
    assert "Traceback" not in err


def test_postgres_unreachable_never_prints_the_password(monkeypatch, capsys):
    _use(monkeypatch, _settings(rag_vector_store_backend="milvus"))
    dsn = "postgresql://harness:hunter2@127.0.0.1:1/harness"
    code, _, err = run(capsys, "docs", "--store", "pgvector", "--dsn", dsn)
    assert code == 1
    assert "can't reach Postgres at harness@127.0.0.1:1/harness" in err
    assert "make infra" in err
    assert "hunter2" not in err


def test_a_missing_pgvector_table_is_not_created(monkeypatch, capsys):
    _require_postgres()
    table = f"test_absent_{uuid.uuid4().hex[:8]}"
    _use(
        monkeypatch,
        _settings(
            rag_vector_store_backend="pgvector", pgvector_url=_PG_DSN, pgvector_table=table
        ),
    )
    code, _, err = run(capsys, "docs")
    assert code == 1
    assert "nothing ingested in pgvector" in err
    with psycopg.connect(_PG_DSN) as conn:
        assert conn.execute("SELECT to_regclass(%s)", (table,)).fetchone() == (None,)


def test_dsn_needs_the_pgvector_store(monkeypatch, capsys):
    _use(monkeypatch, _settings(rag_vector_store_backend="milvus"))
    code, _, err = run(capsys, "docs", "--store", "milvus", "--dsn", "postgresql://x@y/z")
    assert code == 1
    assert "--dsn needs --store pgvector" in err


def test_the_in_memory_store_is_refused(monkeypatch, capsys):
    _use(monkeypatch, _settings(rag_vector_store_backend="in_memory"))
    code, _, err = run(capsys, "docs")
    assert code == 1
    assert "pass --store" in err


def test_a_collection_name_that_would_break_a_filter_is_rejected():
    with pytest.raises(SystemExit) as exit_info:
        rag.main(["docs", "--collection", 'papers" or "1'])
    assert exit_info.value.code == 2
