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
| **74 tests** | Unit, integration, graph scenarios — all pass with zero credentials |

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
│   └── memory/           ← in-memory, pgvector stub
├── core/                 ← protocols + types, imports nothing outward
├── observability/        ← per-step tracer + token cost
└── security/             ← input/content/output guards

frontend/src/
├── hooks/useChatStream   ← reducer + stream consumer + resumeStream()
├── components/           ← StatusBar, ChatInput (interrupt-aware)
└── pages/ChatPage
```

---

## Quick start

```bash
uv sync
uv run uvicorn harness.api.main:app --reload --app-dir src

# health check
curl localhost:8000/health

# stream a response (fake LLM, no credentials needed)
curl -N -X POST localhost:8000/chat/stream \
  -H 'content-type: application/json' \
  -d '{"message": "what is 12 * 9?"}'
```

Frontend:
```bash
cd frontend && pnpm install && pnpm dev
```

Tests (no network, no credentials):
```bash
uv run pytest -q   # 74 tests
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

Azure example:
```bash
export HARNESS_LLM_BACKEND=azure
export HARNESS_AZURE_ENDPOINT=https://<resource>.openai.azure.com
export HARNESS_AZURE_DEPLOYMENT=gpt-4o
# omit HARNESS_AZURE_API_KEY to use managed identity
```

Local model (llama.cpp / vLLM / Ollama):
```bash
export HARNESS_LLM_BACKEND=openai_compatible
export HARNESS_LLM_BASE_URL=http://localhost:8080/v1
export HARNESS_LLM_MODEL=gemma-3-4b
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
  -H 'content-type: application/json' \
  -d '{"decision": {"approved": true}}'
```

Rejection (`approved: false`) routes to the `error` node.

---

## Roadmap

- [x] ReAct agent, tools, short-term memory, tracing, FastAPI
- [x] Azure embeddings stub + pgvector `LongTermMemory` protocol
- [x] React + TypeScript frontend (SSE streaming, cancel)
- [x] **LangGraph `StateGraph` runtime + checkpointer**
- [x] **HITL approval gate (`interrupt` / `resume`)**
- [ ] Token streaming (`LLMClient.stream()`)
- [ ] Full HITL UI (approve button, editable args)
- [ ] Postgres checkpointer
- [ ] Multi-agent: supervisor + specialized sub-agents
- [ ] Verify gate (post-execution fact-checking)
