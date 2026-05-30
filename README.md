# Agent Harness

A production-oriented harness for building AI agents with **swappable model and
memory backends**. Develop against a reliable model (Azure OpenAI), then retarget
to a local Gemma-4 (E4B/26B in the cloud, E2B on a Raspberry Pi) **by changing
config, not code**.

## The one rule that prevents technical debt

Dependencies point **inward**. `core/` imports nothing outward — no framework, no
SDK, no FastAPI. Everything external lives in `adapters/` behind protocols that
`core/` defines. This single rule is what makes every swap a config change:

```
api / orchestration ──► adapters ──► core  (core depends on nothing)
```

## Layout

| Path | Role |
|------|------|
| `core/` | Pure domain: agent loop, tool/memory/LLM **protocols**, types. No IO. |
| `adapters/` | Concrete impls of core protocols: Azure/Ollama/fake LLMs, pgvector/in-memory stores, tools. |
| `orchestration/` | Composition root (`build.py`); LangGraph graph lands here at the multi-agent step. |
| `observability/` | Per-step tracer + cost tracking. |
| `security/` | Input/content/output guards (skeleton). |
| `prompts/` | Versioned, hot-swappable prompt templates. |
| `evaluation/` | Golden dataset + offline/online eval. |
| `tests/contract/` | One suite per protocol; **every** adapter must pass it. ← anti-debt mechanism |

## Swap matrix

| Concern | Dev (Azure) | Edge (Pi) | Selected by |
|---------|-------------|-----------|-------------|
| LLM | `AzureOpenAIClient` | `OllamaClient` (Gemma-4 E2B) | `HARNESS_LLM_BACKEND` |
| Tool calls | `NativeToolCallParser` | `PromptedToolCallParser` | `HARNESS_TOOL_PARSER` |
| Long-term memory | `PgVectorLongTerm` | `InMemoryLongTerm` / `PgVectorLongTerm` | `HARNESS_MEMORY_BACKEND` |

## Run

```bash
pip install -e ".[dev]"          # add ,azure / ,pgvector / ,local as needed
pytest -q                        # 10 tests, no network required
uvicorn harness.api.main:app --reload --app-dir src
curl localhost:8000/health
curl -X POST localhost:8000/chat -H 'content-type: application/json' \
     -d '{"message":"what is 12*9?"}'
```

Default backend is `fake` (runs with zero credentials). To use Azure:

```bash
export HARNESS_LLM_BACKEND=azure HARNESS_TOOL_PARSER=native
export HARNESS_AZURE_ENDPOINT=https://<resource>.openai.azure.com
export HARNESS_AZURE_DEPLOYMENT=gpt-4o
# api key, or omit for managed identity
```

## Roadmap

1. **(done)** Single ReAct agent, tools, short-term memory, tracing, FastAPI.
2. Azure embeddings + `PgVectorLongTerm` wired into `build_long_term`.
3. Ollama adapter + `PromptedToolCallParser`; A/B the parsers on Gemma-4 via `tests/contract`.
4. React + TypeScript frontend against a frozen `/chat` contract (SSE streaming).
5. Multi-agent: LangGraph supervisor + agent nodes in `orchestration/`, Postgres checkpointer.
