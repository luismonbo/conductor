# RAG eval datasets

## Labeling convention

`recall_at_k` scores **fractional** recall — `|retrieved ∩ expected| / |expected|` — so the
label set determines what a perfect score means. Two rules keep that number honest:

1. **Label the minimal set of chunks that individually suffice to answer the question.**
   Not every chunk that mentions the topic. If one passage answers it, list one id. Listing
   three "related" chunks means a retriever that surfaces the answer-bearing one and nothing
   else scores 0.33 despite fully answering the question.
2. **List several ids only when the answer genuinely requires combining them** (e.g.
   `meta_definition`, where the abstract and the method section each carry half the
   definition).

`relevant_document_ids` is the coarse fallback. `recall_at_k` prefers `relevant_chunk_ids`
when present and only falls back to documents when the chunk list is empty — the denominator
is never mixed across granularities.

## Chunk ids are not stable across re-ingestion

`chunk_id` is `{document_id}:{index}`, and the index comes from LLM-driven section
normalization, which is **non-deterministic**. Re-ingesting the same PDF can renumber every
chunk and silently invalidate every label here.

`document_id` *is* stable — it hashes the parsed document text.

So: after any re-ingest, chunking change, or embedding-model change, **re-verify the chunk
ids** before trusting a recall number. Query them back rather than assuming:

```sql
SELECT chunk_id, chunk_json->>'section_path', left(text, 120)
FROM rag_chunks
WHERE chunk_id = 'papers/ad22e4e8c9f87f5e:5';
```

A label that no longer points at the intended passage does not error — it just quietly
reports a retrieval failure that never happened.

## papers_v1.json

Nine cases over two documents, all verified against the live index on 2026-07-27:

- `attention-is-all-you-need.pdf` → `papers/ad22e4e8c9f87f5e` (41 chunks)
- `Meta-Prompting.pdf` → `papers/23731d65a10a1398` (111 chunks)

Tags: `smoke` (fast subset), `attention` / `meta-prompting` (per document), `factual`,
`reasoning`, `definition`, `negative`.

The `negative` case (`unanswerable_from_corpus`) has deliberately empty expectations, so
both retrieval metrics skip it. It exists to exercise the grounding instruction: the answer
should decline rather than answer from the model's own knowledge, which is what
`faithfulness` scores.
