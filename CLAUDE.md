# Project context for coding agents

## Architecture invariant (DO NOT VIOLATE)
Dependencies point inward. `src/harness/core/` must import NOTHING from
`adapters/`, `orchestration/`, `api/`, or any third-party SDK (no openai,
qdrant, fastapi, langgraph). If you need an external capability in core,
define a Protocol in core and implement it in adapters/.

## Adding things
- New tool: one file in `adapters/tools/`, register it in `orchestration/build.py`. Do NOT edit the agent loop.
- New LLM/memory backend: implement the core Protocol in `adapters/`, add a branch in `build.py`, run the matching suite in `tests/contract/`.
- A new adapter is not "done" until it passes its contract test.

## Conventions
- Python 3.11+, async throughout, full type hints (mypy strict).
- Tests: unit tests use FakeLLMClient (no network). Contract tests run per protocol.
- Keep API DTOs (`api/schemas.py`) separate from `core/types.py`.

## Version control
- `.claude/` and `docs/` are NOT committed (they are in `.gitignore`). They are local
  working context — agent settings and design/spec notes — not part of the shipped project.
- Commit meaningful changes as we go (one logical change per commit).
