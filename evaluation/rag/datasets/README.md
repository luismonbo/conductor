# RAG eval datasets

## Schema v2

`papers_v2.json` is the eval dataset — three tiers of cases (chunk-graded, document-level,
negative) against a corpus of three papers. See `evaluation/rag/dataset.py` for the loader and
`RUBRIC.md` for how chunks are graded 0-3. Grading methodology lives in `RUBRIC.md` and is not
duplicated here.

## Corpus fingerprint

The dataset's `corpus.documents` field records `{document_id: chunk_count}` at the time it was
labelled. `RetrievalRunner`/`verify_corpus()` (`evaluation/rag/retrieval_runner.py`) compares
this against the live index at startup and fails loudly on drift. Current fingerprint:

- `papers/50455611c3ed8b48` (Transformer / "Attention Is All You Need") — 43 chunks
- `papers/75c2c05bec38ceb9` (Meta-Prompting) — 46 chunks
- `papers/5e80f8764192bbf0` (globally-beneficial-technology) — 59 chunks

## Chunk ids are not stable across re-ingestion

`chunk_id` is `{document_id}:{index}`. Re-ingesting a document — a parser change, a chunking
change — can renumber every chunk and silently invalidate every chunk-level label. `document_id`
is a hash of `(collection, source_path)`, not of chunk or document content (see
`src/harness/core/rag/document.py`), so it survives re-chunking — only the fingerprint above
catches a chunk-numbering change. After any re-ingest, run `evaluation/run_retrieval_eval.py`
and let the fingerprint check fail loudly rather than trusting stale ids. To manually spot-check
a chunk:

```sql
SELECT chunk_id, chunk_json->>'section_path', left(text, 120)
FROM rag_chunks
WHERE chunk_id = '<document_id>:<index>';
```

A label pointing at the wrong passage does not error on its own — it just quietly reports a
retrieval failure that never happened.

## papers_v1.json (deleted)

Superseded entirely by `papers_v2.json` (schema v2: graded 0-3 relevance replaces a binary label
list, plus `provenance`, `suites`, `excluded_from`, and the corpus fingerprint above).
`papers_v1.json`'s labels were invalidated by the `feat/rag-ingestion-v2` re-chunking and were
never re-verified — see `docs/devlog/019-eval-retrieval-instruments.md` for the full rationale
for rebuilding rather than migrating.
