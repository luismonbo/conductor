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
api ──► orchestration ──► agents ──► adapters ──► core
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
| **Swappable storage** | Checkpointer and run store opened together: `MemorySaver` (tests) · SQLite (local default) · Postgres (production, on Neon) |
| **SSE streaming** | Real-time `thinking`, `tool_call`, `tool_result`, `interrupt`, `final`, `error` events |
| **React frontend** | Interrupt state UI: amber dot, "Waiting for approval", Reject button |
| **RAG** | docling/markitdown ingestion → LLM-normalized sections → structure-aware chunks → pgvector + Milvus Lite; retrieval, grounded generation, and a RAG eval harness |
| **Multi-provider models** | LiteLLM proxy fans one adapter out to Ollama/Anthropic/OpenAI/Gemini/DeepSeek by model name; `resolve_model()` precedence: tool pin > agent pin > UI picker > default |
| **Langfuse Cloud tracing** | Trace grouping via LangChain callbacks, per-call `generation`/`tool` observations via `langfuse.openai` + explicit spans (neither the LLM call nor tool dispatch goes through LangChain itself) — a dead/misconfigured Langfuse never blocks chat |
| **Threaded chat UI** | Model picker, thread sidebar with transcript reload (`GET /models`, `GET /threads`) |
| **970 tests** | Unit, integration, contract, graph scenarios — all pass with zero credentials |

---

## Layout

```
src/harness/
├── agents/
│   ├── definition.py     ← AgentDefinition + ToolDeps: the contract every agent folder implements
│   └── default/          ← one folder per agent: prompt.py, tools.py, agent.py
├── api/                  ← FastAPI endpoints + SSE
├── orchestration/        ← shared agent loop, agent loader, composition root, storage (checkpointer + run store)
├── adapters/
│   ├── llm/              ← Azure, OpenAI-compatible, fake
│   ├── tools/            ← calculator, remember, recall, search_documents, wiki
│   ├── memory/           ← in-memory, pgvector stub (long-term conversational memory)
│   ├── parsing/          ← docling, markitdown, router
│   ├── normalization/    ← LLM structured-output normalizer
│   ├── chunking/         ← structure-aware chunker
│   ├── embedding/        ← openai-compatible, fake
│   ├── vectorstore/      ← pgvector, Milvus Lite, in-memory (RAG chunk index)
│   └── wiki/             ← wiki pages from a local folder or Azure Blob, parsed into memory
├── core/
│   ├── rag/              ← RAG schema, ports, ingestion + serving pipelines
│   └── ...               ← protocols + types, imports nothing outward
├── cli/                  ← ingest.py, rag_query.py
├── observability/        ← per-step tracer + token cost
└── security/             ← input guard, egress filter (what may reach a browser)

frontend/src/
├── hooks/useChatStream   ← reducer + stream consumer + resumeStream()
├── components/           ← StatusBar, ChatInput (interrupt-aware)
└── pages/ChatPage
```

Local tools to inspect conversations, runs and the vector index live in `scripts/` (see
[scripts/README.md](scripts/README.md)); they never ship in an image.

---

## Quick start

Full local stack (LiteLLM proxy + Postgres — Langfuse is cloud-hosted, see
Configuration below), then API and frontend each in their own terminal:

```bash
uv sync
cp .env.example .env   # fill in provider keys for the profiles you'll use
echo "HARNESS_API_KEY=$(openssl rand -hex 32)" >> .env   # required: auth fails closed with no key
# Local dev: the input guard (fails closed without a TypeSafe key) and the
# public website's limits off. A public deployment keeps the defaults.
cat >> .env <<'EOF'
HARNESS_INPUT_GUARD_ENABLED=false
HARNESS_MAX_MESSAGE_CHARS=0
HARNESS_MAX_OUTPUT_TOKENS=0
HARNESS_MAX_THREAD_TURNS=0
HARNESS_DAILY_TOKEN_BUDGET=0
EOF

make up     # infra: postgres, litellm
make api    # FastAPI, reload
make web    # Vite dev server
```

`make up` validates `litellm_config.yaml` before starting anything, so a typo
in a model profile fails fast instead of surfacing as an opaque proxy 400.

Fallback — raw commands, no proxy/tracing, fake LLM, zero credentials:
```bash
export HARNESS_API_KEY=$(openssl rand -hex 32)
export HARNESS_INPUT_GUARD_ENABLED=false   # the input guard fails closed without a TypeSafe key
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
uv run pytest -q   # 970 tests
# the Postgres tests (run store, checkpointer, db.py, pgvector) skip unless `make infra` runs;
# HARNESS_TEST_POSTGRES_DSN points them at another server (CI runs them on Postgres 18)
```

---

## Configuration

| Env var | Default | Options |
|---------|---------|---------|
| `HARNESS_LLM_BACKEND` | `fake` | `fake` · `openai_compatible` · `azure` |
| `HARNESS_CHECKPOINTER` | `sqlite` | `memory` · `sqlite` · `postgres`: where threads and the run store live |
| `HARNESS_CHECKPOINTER_URL` | `./harness.sqlite` | the SQLite file, or for `postgres` a `postgresql://` connection string (a secret) |
| `HARNESS_AGENT` | `default` | entry agent: a folder under `src/harness/agents/` (see [Agents](#agents)) |
| `HARNESS_TOOL_PARSER` | `native` | `native` · `prompted` |
| `HARNESS_MAX_ITERATIONS` | `8` | integer |
| `HARNESS_MEMORY_TOOLS_ENABLED` | `false` | `true` · `false` — gates the shared `remember`/`recall` long-term memory tools; must stay off on a public deployment since that store is not partitioned per visitor |
| `HARNESS_SHOW_TOOL_DETAILS` | `false` | `true` · `false` — whether tool arguments and results reach the browser (the chat stream and `GET /threads/{id}`); off, a browser gets each tool's name only. Keep it off on a public deployment |
| `HARNESS_SESSION_COOKIE_SECURE` | `true` | `true` · `false` — marks the visitor `sid` cookie `Secure`, so browsers only send it over HTTPS; set `false` only for local development on a browser that does not treat `http://localhost` as a secure context |
| `HARNESS_SESSION_COOKIE_PERSISTENT` | `true` | `true` · `false` — how long the visitor `sid` cookie lasts: `true` keeps it for a year, `false` sends it without a `Max-Age`, so the browser drops it when it closes (the public site's target sets `false`, as its privacy page says) |
| `HARNESS_LOG_LEVEL` | `INFO` | `DEBUG` · `INFO` · `WARNING` · `ERROR` — level of the harness's own log lines (stderr); every other library stays at `WARNING`. Never set an SDK's own debug switch in production (`TYPESAFE_LOG_LEVEL=debug`, `OPENAI_LOG=debug`, `LANGFUSE_DEBUG=true`), nor run uvicorn with `--log-level trace`: they log visitor messages, the `sid` cookie or the API key |
| `HARNESS_RATE_LIMIT_ENABLED` | `true` | `true` · `false` |
| `HARNESS_RATE_LIMIT_STRICT` | `15/minute` | slowapi rate string; applies to LLM-invoking endpoints |
| `HARNESS_RATE_LIMIT_DEFAULT` | `60/minute` | slowapi rate string; applies to all other rate-limited endpoints |
| `HARNESS_AUTH_ENABLED` | `true` | `true` · `false` — fails closed: refuses to start if `true` with no `HARNESS_API_KEY` set |
| `HARNESS_API_KEY` | — | shared bearer key gating every route except `/health`; generate with `openssl rand -hex 32` |
| `HARNESS_INPUT_GUARD_ENABLED` | `true` | `true` · `false` — TypeSafe (Jev) screens every chat message for extraction, private-info and off-topic requests before the agent sees it; fails closed: refuses to start if `true` with no `HARNESS_TYPESAFE_API_KEY` |
| `HARNESS_TYPESAFE_API_KEY` | — | TypeSafe API key |
| `HARNESS_GUARD_MODEL` | `jev-1.13.0` | pinned Jev version the guard's thresholds are calibrated against |
| `HARNESS_MAX_MESSAGE_CHARS` | `2000` | integer; `0` = off |
| `HARNESS_MAX_OUTPUT_TOKENS` | `1500` | integer, per LLM call; `0` = off |
| `HARNESS_MAX_THREAD_TURNS` | `20` | integer, runs per thread; `0` = off |
| `HARNESS_DAILY_TOKEN_BUDGET` | `33000` | integer, all visitors per UTC day; `0` = off |
| `CORS_ORIGINS` | `http://localhost:5173` | comma-separated allowed browser origins; not `HARNESS_`-prefixed |
| `HARNESS_EMBEDDING_BACKEND` | `fake` | `fake` · `openai_compatible` · `azure` (azure not yet wired) |
| `HARNESS_EMBEDDING_MODEL` | `nomic-embed-text-v1.5` | any model your embedding endpoint serves |
| `HARNESS_EMBEDDING_DIMENSION` | `768` | must match the vector store schema; changing it forces a reindex |
| `HARNESS_PGVECTOR_URL` | — | Postgres DSN for the RAG chunk index |
| `HARNESS_MILVUS_URI` | `./data/milvus_papers.db` | file path = Milvus Lite; `http(s)://` = a real server |
| vector store (CLI `--vector-store`) | `pgvector` | `pgvector` · `milvus` · `in_memory` |
| `HARNESS_RAG_K` | `5` | retrieval depth |
| `HARNESS_RAG_PER_DOCUMENT_K` | `2` | max chunks per document; `0` = plain flat top-k |
| `HARNESS_WIKI_DIR` | — | a local wiki folder holding `index.md`, e.g. `wiki` (the repo's gitignored `wiki/`; see [Agents](#agents)); set this or the Blob settings, not both |
| `HARNESS_WIKI_BLOB_ACCOUNT_URL` | — | `https://<account>.blob.core.windows.net`; read with `DefaultAzureCredential` (needs Storage Blob Data Reader) |
| `HARNESS_WIKI_BLOB_CONTAINER` | — | the private container holding the wiki's pages; set together with the account URL |
| `HARNESS_WIKI_BLOB_PREFIX` | — | optional folder inside the container, e.g. `wiki/` |
| `HARNESS_DEFAULT_MODEL` | — | LiteLLM profile name sent when no pin/override applies; supersedes `HARNESS_LLM_MODEL` when set |
| `LITELLM_MASTER_KEY` | — | proxy auth; must equal `HARNESS_LLM_API_KEY` |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | — | empty = tracing disabled, chat unaffected |
| `LANGFUSE_HOST` | `https://cloud.langfuse.com` | Langfuse Cloud project (EU region; US: `https://us.cloud.langfuse.com`) |

**Rate limiting and visitors.** The limiter keys on the connection's address as uvicorn sees it. In production a Cloudflare Worker holds `HARNESS_API_KEY`, adds it to every request and sets `X-Visitor-IP` to the visitor's address; the limiter then keys on that address, so each visitor gets their own limit instead of the whole internet sharing the platform ingress's address. The header is honored only on a request that carries the valid `HARNESS_API_KEY`; every other request is limited by its connection address. An IPv6 address counts as its /64, and a missing, malformed or repeated header counts as absent. Any proxy that injects the key must therefore own the header: set it to the real visitor's address, or strip it. Caddy and the dev-server proxies strip it, so behind them the limiter keys on whatever address uvicorn sees (with Caddy in the Compose stack, the caller's, taken from `X-Forwarded-For`). Neither the harness nor slowapi logs the limiter key (slowapi's own 429 line, which names it, is silenced), but uvicorn's access log prints the connection's address: on Container Apps that is the platform's proxy, in the Compose stack it is the caller.

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

## Agents

An agent is a folder under `src/harness/agents/`:

```
src/harness/agents/<name>/
├── prompt.py   ← SYSTEM_PROMPT
├── tools.py    ← build_tools(deps) -> ToolRegistry, from the shared infrastructure (ToolDeps)
└── agent.py    ← AGENT = AgentDefinition(name="<name>", prompt=SYSTEM_PROMPT, tools=build_tools)
```

`HARNESS_AGENT=<name>` picks the entry agent, the one that talks to users. It is loaded by name
(`harness.agents.<name>.agent`), so there is no registry to edit, and only that agent is imported.
Every agent runs the same loop, `orchestration/loop.py`. The API builds the entry agent once at
startup; a name that doesn't resolve, or a backend the agent requires, stops the app before it
serves. Prompts live only in `prompt.py`: never in `.env`, Settings or a target file. An agent's
`search_documents` searches only the collections its `tools.py` lists; any other collection the
model names is refused.

**Wiki.** Instead of searching documents, an agent can read a wiki: `read_wiki_index` returns its
`index.md`, and `read_wiki_page` one page by the name its `[[...]]` links use (the page's path
without `.md`). The wiki loads into memory at startup from `HARNESS_WIKI_DIR`, a local folder (in
this repo, the gitignored `wiki/`: `HARNESS_WIKI_DIR=wiki`), or, in production, from a private
Azure Blob container (`HARNESS_WIKI_BLOB_ACCOUNT_URL`,
`HARNESS_WIKI_BLOB_CONTAINER`, optional `HARNESS_WIKI_BLOB_PREFIX`) read with the app's managed
identity; restart the app after changing it. The website agent (`agents/website/`) answers from its
wiki only and does not start without it.

**Private agents.** To keep an agent out of a public repo, gitignore its folder under
`src/harness/agents/`; the Dockerfile's `COPY src/ src/` still bakes it into the image. That relies
on the editable install `uv sync` performs (a `.pth` file pointing at `src/`): a non-editable wheel
build drops gitignored files.
`TARGET_NAME=<target> make proxy` runs that same image locally behind Caddy, with
`config/targets/<target>.yaml` baked in and only secrets taken from the environment.

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
- [x] **Postgres storage** — checkpointer and run store on one pool; Neon in production
- [ ] Multi-agent: supervisor + specialized sub-agents
- [ ] Verify gate (post-execution fact-checking)
- [ ] RAG Phase 2 (hybrid retrieval, reranking, enforced idempotent re-ingestion,
      groundedness checking, citations) and Phase 3 (agentic retrieval)
- [ ] Tool-level model pins (`resolve_model` supports `tool_pin`; no consumer yet)
