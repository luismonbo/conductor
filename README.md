# Conductor

[![CI](https://github.com/luismonbo/conductor/actions/workflows/ci.yml/badge.svg)](https://github.com/luismonbo/conductor/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

An agent harness: the platform an AI agent runs on, with its model access, tools, memory, human
approvals, evaluations and deployment. Built on LangGraph and FastAPI, it has run the public
assistant on [luismonbo.com](https://luismonbo.com) since October 2026.

## Architecture

```text
Browser
  │ luismonbo.com/api/*
  ▼
Cloudflare Worker   adds the API key and sets X-Visitor-IP; the page holds no secret
  │
  ▼
FastAPI             key check, per-visitor rate limit, input guard
  │                 (Azure Container Apps)
  ▼
LangGraph loop      call_model → approval_gate → execute_tools → call_model …
  │                 · model: Azure OpenAI with a managed identity (LiteLLM locally)
  │                 · state: Postgres on Neon for checkpoints and runs (SQLite locally)
  │                 · knowledge: a wiki loaded from private Blob storage at startup
  ▼
egress filter       deny by default, then SSE events back to the browser
```

Dependencies point inward: `api → orchestration → agents → adapters → core`. `core/` holds the
ports and types (LLM, tools, memory, RAG, guard, wiki) and imports nothing outside the standard
library and the harness itself, so anything behind a port is swapped by configuration.

## Design decisions

- **One wire format for every model.** Both LLM clients speak the OpenAI chat format
  ([openai_wire.py](src/harness/adapters/llm/openai_wire.py)): Azure OpenAI in production, any
  OpenAI-compatible endpoint otherwise. Locally that endpoint is a LiteLLM proxy that routes
  profile names ([litellm_config.yaml](litellm_config.yaml)) to Ollama, Anthropic, OpenAI, Gemini
  or DeepSeek. Changing provider is a config edit, and the agent loop never sees one.
- **Approvals are graph pauses.** A tool with `requires_approval = True` stops the run with
  LangGraph's `interrupt()`. The checkpointer keeps the state, so an approval can wait without
  holding a request open, and `POST /resume/{thread_id}` continues the run with approve, deny, or
  feedback for the model.
- **Agents are folders.** `src/harness/agents/<name>/` holds `prompt.py`, `tools.py` and
  `agent.py`. `HARNESS_AGENT` names the entry agent, loaded by name with no registry to edit, and
  its document search reaches only the collections its `tools.py` lists. A deployment is a
  settings file in `config/targets/`, not a fork.
- **Pinned models win.** Each call site resolves its model as tool pin, then agent pin, then the
  UI's picker, then the default ([model_resolution.py](src/harness/core/llm/model_resolution.py)).
  The picker steers the conversation, never a component pinned for cost or capability.
- **Bad configuration fails at startup, not in front of a visitor.** The app refuses to start
  with auth on and no key, with the guard on and no endpoint, with an entry agent that doesn't
  resolve or lacks what it needs (the website agent without its wiki), or with a credential in a
  target file.

## Production and safety

- **The key never reaches a browser.** The page calls `luismonbo.com/api/*`, where a Cloudflare
  Worker adds the shared API key and sets `X-Visitor-IP`. The API trusts that header only on a
  request that carries the valid key, so any proxy that injects the key must set or strip it.
- **Per-visitor limits and a cost ceiling.** Rate limits key on the visitor's address (an IPv6
  address as its /64): 15 a minute on the routes that call the model, 60 elsewhere, and the
  limiter's key is never logged. Messages are capped in length, each model call in output tokens,
  and all visitors together by a daily token budget derived from the provider's monthly quota.
- **Input guard.** Microsoft-Decision-1 screens every message for prompt extraction, requests for
  private information and off-topic use before the agent sees it, with thresholds calibrated on
  labeled cases. By design a classifier error lets the message through, so the chat's
  availability never depends on the classifier; the fixed limits still apply. Refusals come in the
  visitor's language, from a fixed table no model writes.
- **Egress filter.** Everything a browser receives passes [egress.py](src/harness/security/egress.py),
  deny by default: an exception never becomes browser text, tool arguments and results stay on
  the server unless the deployment allows them, and an unknown event type is dropped.
- **Visitor data.** An anonymous random `sid` cookie (HttpOnly, Secure, a session cookie on the
  public site) scopes threads to their visitor. A daily Container Apps Job deletes conversations
  30 days after their last message, and an alert fires when a day passes without a clean run.
  Long-term memory stays off in public, because its store isn't partitioned per visitor yet.
- **Least privilege.** The app reaches the model and the wiki with its managed identity, and its
  secrets come from the Container App's environment, never from the committed target file.
  Checkpoints deserialize only an explicit list of types.
- **Observable without a vendor.** Every run is recorded with its agent, backend, timing, stop
  reason, tokens and iterations, and every guard decision with its probabilities;
  [scripts/db.py](scripts/db.py) reads both.

## Evaluation and testing

- [run_eval.py](evaluation/run_eval.py) drives the real LangGraph agent over a dataset and scores
  tool choice, arguments and answers.
- [run_retrieval_eval.py](evaluation/run_retrieval_eval.py) measures retrieval alone, with no
  LLM: recall@k, MRR and nDCG@k, deterministic and seconds per run, so retrieval settings can be
  swept. [run_rag_eval.py](evaluation/run_rag_eval.py) adds generation: faithfulness and answer
  relevancy.
- [run_guard_eval.py](evaluation/run_guard_eval.py) calibrates the input guard; it is rerun
  whenever the website agent's prompt changes.
- Datasets and pass thresholds are committed; run reports are not.

About 1,200 tests (unit, integration, contract and graph scenarios) need no network and no
credentials: `uv run pytest -q`, or `make test` with the frontend's. CI runs ruff, mypy and the
suite, with the Postgres tests on `pgvector/pgvector:0.8.6-pg18`, the versions Neon runs; a
missing database fails them instead of skipping them.

## Quick start

Needs Python 3.13 and [uv](https://docs.astral.sh/uv/). Until you configure a model, a fake one
answers with a fixed sentence, so this needs no credentials:

```bash
uv sync
export HARNESS_API_KEY=$(openssl rand -hex 32) HARNESS_INPUT_GUARD_ENABLED=false
uv run uvicorn harness.api.main:app --app-dir src &

curl --retry 10 --retry-connrefused localhost:8000/health   # waits for the server to start
curl -N localhost:8000/chat/stream -H "Authorization: Bearer $HARNESS_API_KEY" \
  -H 'content-type: application/json' -d '{"message": "hello"}'
```

The full stack, with real models through the LiteLLM proxy, Postgres and the chat UI, needs Docker
and pnpm:

```bash
cp .env.example .env   # set HARNESS_API_KEY and HARNESS_INPUT_GUARD_ENABLED=false, then uncomment
                       # the LiteLLM block and add a provider key (or pull the Ollama models)
make up                # Postgres and the LiteLLM proxy
make api               # the API on :8000
make web               # in a second terminal: the chat UI (after `pnpm -C frontend install`)
```

## Layout

```text
src/harness/
├── core/            ports and types: LLM, tools, memory, RAG, guard, wiki
├── adapters/        implementations: llm, tools, memory, guard, wiki, RAG ingestion, vector stores
├── agents/          one folder per agent: default (tools, memory, RAG) and website (the public one)
├── orchestration/   the LangGraph loop, agent loading, the composition root, storage, retention
├── security/        the input guard, the egress filter, the visitor-facing texts
├── observability/   logging, tracing, token counting, the run store
├── api/             FastAPI routes, auth, sessions, rate limits, SSE
├── config/          settings and deployment targets
└── cli/             ingest, rag_query, purge
evaluation/          agent, retrieval, RAG and guard evaluations
frontend/            React and TypeScript chat: streaming, approval cards, model picker, threads
scripts/             local tools for conversations, runs and the vector index (scripts/README.md)
```

## Configuration

Settings are `HARNESS_*` environment variables, documented in [.env.example](.env.example). A
deployment is a target: `HARNESS_TARGET=<name>` layers `config/targets/<name>.yaml` under the
environment ([example.yaml](config/targets/example.yaml)), and a target never holds a secret. The
ones to know first:

| Variable | Default | |
|---|---|---|
| `HARNESS_LLM_BACKEND` | `fake` | `openai_compatible` (LiteLLM, llama.cpp, vLLM) or `azure` |
| `HARNESS_AGENT` | `default` | the entry agent's folder under `src/harness/agents/` |
| `HARNESS_CHECKPOINTER` | `sqlite` | `memory`, `sqlite` or `postgres`: threads and the run store |
| `HARNESS_API_KEY` | none | required while `HARNESS_AUTH_ENABLED` is `true`, the default |
| `HARNESS_INPUT_GUARD_ENABLED` | `true` | needs `HARNESS_AZURE_ENDPOINT`; turn it off for local use |

## Roadmap

- Multi-agent delegation: the entry agent calls specialized sub-agents as tools.
- Retrieval, phase 2: hybrid search, reranking, contextual chunks and citations, then RAG on the
  production Postgres.
- Scaling past one replica: the rate limiter and the one-run-per-thread claim keep their state in
  memory, so both need a shared store first.
- The retrieval eval and the frontend's tests and type-check in CI (they run locally today).
- Jailbreak and abuse checks in the input guard.

## License

[Apache License 2.0](LICENSE). Copyright 2026 Luis Monge.
