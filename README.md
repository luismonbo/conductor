# Conductor Agent Harness

A production-grade AI agent platform with **swappable model backends**, a **LangGraph runtime**, and a **Human-in-the-Loop approval gate**. Swap Azure OpenAI for a local Gemma model by changing an env var — no code changes.

```
POST /chat/stream  →  LangGraph StateGraph  →  approval gate  →  tools  →  SSE events
                                                      ↓
                                              interrupt (requires_approval=True)
                                                      ↓
                                         POST /resume/{thread_id}
```

---

## Architecture in one rule

Dependencies point **inward only**.

```
api / agents ──► orchestration ──► adapters ──► core
                                                  ↑
                                     no imports outward from here
```

`core/` is pure Python — no FastAPI, no LangGraph, no OpenAI SDK. Every swap is a config change because the seam is real.

---

## What's built

| Feature | Detail |
|---------|--------|
| **LangGraph runtime** | Five-node `StateGraph`: `call_model → approval_gate → execute_tools → final/error` |
| **HITL approval gate** | Tools opt in with `requires_approval = True`; graph suspends via `interrupt()`, state persisted to checkpointer |
| **Resume endpoint** | `POST /resume/{thread_id}` — approve or reject a paused run, stream resumes |
| **Swappable LLM** | Azure OpenAI / OpenAI-compatible (llama.cpp, vLLM) / fake (tests) |
| **Swappable checkpointer** | `MemorySaver` (tests) · `SqliteSaver` (local default) · Postgres (Phase 5) |
| **SSE streaming** | Real-time `thinking`, `tool_call`, `tool_result`, `interrupt`, `final`, `error` events |
| **React frontend** | Interrupt state UI: amber dot, "Waiting for approval", Reject button |
| **RAG** | docling/markitdown ingestion → LLM-normalized sections → structure-aware chunks → pgvector + Milvus Lite; retrieval, grounded generation, and a RAG eval harness |
| **Multi-provider models** | LiteLLM proxy fans one adapter out to Ollama/Anthropic/OpenAI/Gemini/DeepSeek by model name; `resolve_model()` precedence: tool pin > agent pin > UI picker > default |
| **Langfuse Cloud tracing** | Trace grouping via LangChain callbacks, per-call `generation`/`tool` observations via `langfuse.openai` + explicit spans (neither the LLM call nor tool dispatch goes through LangChain itself) — a dead/misconfigured Langfuse never blocks chat |
| **Threaded chat UI** | Model picker, thread sidebar with transcript reload (`GET /models`, `GET /threads`) |
| **354 tests** | Unit, integration, contract, graph scenarios — all pass with zero credentials |

---

## Layout

```
src/harness/
├── agents/
│   └── default/          ← prompt, tools, LangGraph graph
├── api/                  ← FastAPI endpoints + SSE
├── orchestration/        ← composition root, checkpointer factory
├── adapters/
│   ├── llm/              ← Azure, OpenAI-compatible, fake
│   ├── tools/            ← calculator, recall
│   ├── memory/           ← in-memory, pgvector stub (long-term conversational memory)
│   ├── parsing/          ← docling, markitdown, router
│   ├── normalization/    ← LLM structured-output normalizer
│   ├── chunking/         ← structure-aware chunker
│   ├── embedding/        ← openai-compatible, fake
│   └── vectorstore/      ← pgvector, Milvus Lite, in-memory (RAG chunk index)
├── core/
│   ├── rag/              ← RAG schema, ports, ingestion + serving pipelines
│   └── ...               ← protocols + types, imports nothing outward
├── cli/                  ← ingest.py, rag_query.py
├── observability/        ← per-step tracer + token cost
└── security/             ← input/content/output guards

frontend/src/
├── hooks/useChatStream   ← reducer + stream consumer + resumeStream()
├── components/           ← StatusBar, ChatInput (interrupt-aware)
└── pages/ChatPage
```

---

## Quick start

Full local stack (LiteLLM proxy + Postgres — Langfuse is cloud-hosted, see
Configuration below), then API and frontend each in their own terminal:

```bash
uv sync
cp .env.example .env   # fill in provider keys for the profiles you'll use
echo "HARNESS_API_KEY=$(openssl rand -hex 32)" >> .env   # required: auth fails closed with no key

make up     # infra: postgres, litellm
make api    # FastAPI, reload
make web    # Vite dev server
```

`make up` validates `litellm_config.yaml` before starting anything, so a typo
in a model profile fails fast instead of surfacing as an opaque proxy 400.

Fallback — raw commands, no proxy/tracing, fake LLM, zero credentials:
```bash
export HARNESS_API_KEY=$(openssl rand -hex 32)
uv run uvicorn harness.api.main:app --reload --app-dir src

# health check (the one route auth never gates)
curl localhost:8000/health

# stream a response (fake LLM, no credentials needed beyond the shared key)
curl -N -X POST localhost:8000/chat/stream \
  -H "Authorization: Bearer $HARNESS_API_KEY" \
  -H 'content-type: application/json' \
  -d '{"message": "what is 12 * 9?"}'
```

Frontend (raw):
```bash
cd frontend && pnpm install && pnpm dev
```

Tests (no network, no credentials):
```bash
uv run pytest -q   # 354 tests
# the pgvector contract cases skip cleanly unless `docker compose up -d postgres` is running
```

---

## Configuration

| Env var | Default | Options |
|---------|---------|---------|
| `HARNESS_LLM_BACKEND` | `fake` | `fake` · `openai_compatible` · `azure` |
| `HARNESS_CHECKPOINTER` | `sqlite` | `memory` · `sqlite` · `postgres` (Phase 5) |
| `HARNESS_AGENT` | `default` | agent name from registry |
| `HARNESS_TOOL_PARSER` | `native` | `native` · `prompted` |
| `HARNESS_MAX_ITERATIONS` | `8` | integer |
| `HARNESS_RATE_LIMIT_ENABLED` | `true` | `true` · `false` |
| `HARNESS_RATE_LIMIT_STRICT` | `15/minute` | slowapi rate string; applies to LLM-invoking endpoints |
| `HARNESS_RATE_LIMIT_DEFAULT` | `60/minute` | slowapi rate string; applies to all other rate-limited endpoints |
| `HARNESS_AUTH_ENABLED` | `true` | `true` · `false` — fails closed: refuses to start if `true` with no `HARNESS_API_KEY` set |
| `HARNESS_API_KEY` | — | shared bearer key gating every route except `/health`; generate with `openssl rand -hex 32` |
| `CORS_ORIGINS` | `http://localhost:5173` | comma-separated allowed browser origins; not `HARNESS_`-prefixed |
| `HARNESS_EMBEDDING_BACKEND` | `fake` | `fake` · `openai_compatible` · `azure` (azure not yet wired) |
| `HARNESS_EMBEDDING_MODEL` | `nomic-embed-text-v1.5` | any model your embedding endpoint serves |
| `HARNESS_EMBEDDING_DIMENSION` | `768` | must match the vector store schema; changing it forces a reindex |
| `HARNESS_PGVECTOR_URL` | — | Postgres DSN for the RAG chunk index |
| `HARNESS_MILVUS_URI` | `./data/milvus_papers.db` | file path = Milvus Lite; `http(s)://` = a real server |
| vector store (CLI `--vector-store`) | `pgvector` | `pgvector` · `milvus` · `in_memory` |
| `HARNESS_RAG_K` | `5` | retrieval depth |
| `HARNESS_RAG_PER_DOCUMENT_K` | `2` | max chunks per document; `0` = plain flat top-k |
| `HARNESS_DEFAULT_MODEL` | — | LiteLLM profile name sent when no pin/override applies; supersedes `HARNESS_LLM_MODEL` when set |
| `LITELLM_MASTER_KEY` | — | proxy auth; must equal `HARNESS_LLM_API_KEY` |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | — | empty = tracing disabled, chat unaffected |
| `LANGFUSE_HOST` | `https://cloud.langfuse.com` | Langfuse Cloud project (EU region; US: `https://us.cloud.langfuse.com`) |

Azure example:
```bash
export HARNESS_LLM_BACKEND=azure
export HARNESS_AZURE_ENDPOINT=https://<resource>.openai.azure.com
export HARNESS_AZURE_DEPLOYMENT=gpt-4o
# omit HARNESS_AZURE_API_KEY to use managed identity
```

Local model / any provider, via the LiteLLM proxy (default; see "Model routing" below):
```bash
export HARNESS_LLM_BACKEND=openai_compatible
export HARNESS_LLM_BASE_URL=http://localhost:4000/v1
export HARNESS_LLM_API_KEY=$LITELLM_MASTER_KEY
export HARNESS_DEFAULT_MODEL=ollama-gemma
```

---

## HITL approval gate

Mark any tool as requiring human approval before execution:

```python
class MyMutatingSomethingTool:
    @property
    def requires_approval(self) -> bool:
        return True
```

The graph pauses, emits an `interrupt` SSE event, and closes the stream. State is persisted to the checkpointer. Resume via:

```bash
curl -X POST localhost:8000/resume/<thread_id> \
  -H "Authorization: Bearer $HARNESS_API_KEY" \
  -H 'content-type: application/json' \
  -d '{"decision": {"approved": true}}'
```

Rejection (`approved: false`) routes to the `error` node.

---

## Model routing

The harness keeps exactly one LLM adapter in the hot path — `OpenAICompatibleClient`
— pointed at a LiteLLM proxy container that fans out to Ollama/Anthropic/OpenAI/Gemini/DeepSeek
by model name. Six profiles ship by default (`litellm_config.yaml`); the `ollama-*` ones
are pinned to specific pulled tags rather than auto-discovered — LiteLLM's wildcard
routing doesn't reliably enumerate Ollama models, so each one you want in the picker
gets its own entry, same as every cloud profile below:

| Profile | Provider |
|---------|----------|
| `ollama-gemma` | Ollama, on the host (`gemma4:e2b`) |
| `ollama-gemma-12b` | Ollama, on the host (`gemma4:12b-mlx`) |
| `claude` | Anthropic |
| `gpt` | OpenAI |
| `gemini-flash` | Gemini |
| `deepseek-flash` | DeepSeek |

Every call site resolves its model by precedence — **tool pin > agent pin > UI
picker > default** — via `resolve_model()`
([model_resolution.py](src/harness/core/llm/model_resolution.py)). Pins beat the
picker on purpose: the model picker steers the conductor, never a component that
was pinned for cost or capability reasons.

### Troubleshooting `make up`

- **An `ollama-*` profile fails or 404s**: `litellm_config.yaml`'s Ollama entries
  point at specific pulled tags (`gemma4:e2b`, `gemma4:12b-mlx`) — run `ollama list`
  and edit those lines to match what you've actually pulled.
- **Reusing a `postgres_data` volume from before this Makefile existed**: the
  `litellm` database only gets created by `create-databases.sql` on a *fresh*
  volume. Run `make init-dbs` once.
- **Langfuse tracing not showing up**: it's Langfuse Cloud now (see Configuration),
  not part of `make up` — `docker compose up -d postgres litellm` never touches it.
  Blank `LANGFUSE_PUBLIC_KEY`/`SECRET_KEY` just no-ops tracing either way; chat
  itself is unaffected.

---

## RAG

Drop documents into `data/raw/<collection>/` (gitignored), then ingest and query.
Retrieval is a standalone primitive — it returns raw chunks, never a pre-baked
answer — so a future agent tool can assess sufficiency and re-retrieve.

```bash
# Postgres + pgvector (Milvus Lite is embedded, no service needed)
docker compose up -d postgres

# ingest into both vector stores, then query one
uv run python -m harness.cli.ingest --collection papers --vector-store all
uv run python -m harness.cli.rag_query "What method does the paper use?" --show-context

# same golden set against each backend, to compare them
uv run python evaluation/run_rag_eval.py --vector-store pgvector
uv run python evaluation/run_rag_eval.py --vector-store milvus
```

Ingestion is `parse → normalize → chunk → embed → upsert`. PDFs route to docling
(layout-aware); everything else to markitdown, which is also the fallback if
docling fails, so one bad document never aborts a run. Parser output is
normalized into a common section schema by the model via tool-calling — long
documents are windowed to stay inside the context budget.

`evaluation/rag/datasets/papers_v1.json` ships empty on purpose: cases need
known-correct passages from real documents, so it gets populated once a corpus
exists. The eval runs clean against an empty index and reports a baseline.

---

## Roadmap

- [x] ReAct agent, tools, short-term memory, tracing, FastAPI
- [x] Azure embeddings stub + pgvector `LongTermMemory` protocol
- [x] React + TypeScript frontend (SSE streaming, cancel)
- [x] **LangGraph `StateGraph` runtime + checkpointer**
- [x] **HITL approval gate (`interrupt` / `resume`)**
- [x] **RAG ingestion** — docling/markitdown parsing, LLM-normalized schema,
      structure-aware chunking, dual pgvector + Milvus Lite vector stores
- [x] **RAG retrieval + grounded generation**, RAG eval harness (recall@k, MRR,
      faithfulness, answer relevancy)
- [x] Token streaming (`LLMClient.stream()`)
- [x] **Multi-provider models** — LiteLLM proxy, `resolve_model()` precedence hierarchy
- [x] **Langfuse Cloud tracing** — LangChain callbacks + `langfuse.openai` generation tracing + explicit tool spans, fire-and-forget
- [x] **Model picker + thread sidebar** — `GET /models`, `GET /threads`, transcript reload
- [ ] Full HITL UI (approve button, editable args)
- [ ] Postgres checkpointer
- [ ] Multi-agent: supervisor + specialized sub-agents
- [ ] Verify gate (post-execution fact-checking)
- [ ] RAG Phase 2 (hybrid retrieval, reranking, enforced idempotent re-ingestion,
      groundedness checking, citations) and Phase 3 (agentic retrieval)
- [ ] Tool-level model pins (`resolve_model` supports `tool_pin`; no consumer yet)
