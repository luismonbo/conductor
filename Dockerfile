# syntax=docker/dockerfile:1

# ---- Builder: resolve dependencies + install the project ------------------
FROM python:3.13-slim-trixie AS builder

# Pinned: a new uv release must not change the build under the lockfile.
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# Dependencies first, in their own layer, so code-only changes below don't
# invalidate the install. --no-dev leaves out the dependency group that holds
# pytest, ruff and mypy. No extras are named: every runtime dependency is in the
# base [project] table, and the optional `docling` extra stays out.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

COPY src/ src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# ---- Purge builder: only what the retention pass imports --------------------
# The purge job's image (stage `purge` below) needs settings, psycopg and LangGraph's
# Postgres checkpointer: pyproject's `purge` dependency group, at the versions uv.lock
# pins for the app. --only-group installs the group alone: neither the project nor its
# dependencies. Those packages require others that a pass never loads: langsmith
# (LangChain's tracing client, which langchain-core requires) and what only it needs,
# zstandard alone 22 MB, plus five small ones. --no-install-package leaves those 16
# out, so the venv holds the 22 packages a pass loads: about 50 MB, against 84 MB with
# the 16 and the app's ~700 MB. tests/unit/test_purge_image_dependencies.py checks the
# image gets every package a pass imports and nothing else. Design:
# docs/superpowers/specs/2026-10-14-retention-purge-design.md §8.3.
FROM python:3.13-slim-trixie AS purge-builder

COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --only-group purge \
        --no-install-package anyio --no-install-package distro \
        --no-install-package h11 --no-install-package httpcore \
        --no-install-package httpx --no-install-package jsonpatch \
        --no-install-package jsonpointer --no-install-package langchain-protocol \
        --no-install-package langsmith --no-install-package orjson \
        --no-install-package requests-toolbelt --no-install-package sniffio \
        --no-install-package tenacity --no-install-package websockets \
        --no-install-package xxhash --no-install-package zstandard

# ---- Purge: the retention job's image ---------------------------------------
# Built with `--target purge`; a plain build still ends at `runtime`, the app's image.
# The job runs the entrypoint daily, and `--dry-run` is an argument to it.
FROM python:3.13-slim-trixie AS purge

# Debian security patches, as in `runtime`; the purge needs no extra packages.
RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/*

RUN groupadd --system harness && useradd --system --gid harness --no-create-home harness

WORKDIR /app
COPY --from=purge-builder /app/.venv /app/.venv
# The project isn't installed in this venv (--only-group), so its source goes on
# PYTHONPATH.
COPY src/ src/
ARG TARGET_NAME
COPY config/targets/${TARGET_NAME}.yaml config/targets/${TARGET_NAME}.yaml
# Bytecode now, as root: the harness user can't write __pycache__ at run time.
RUN python -m compileall -q src

ENV HARNESS_TARGET=${TARGET_NAME} \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app/src \
    LANGGRAPH_STRICT_MSGPACK=true

USER harness

ENTRYPOINT ["python", "-m", "harness.cli.purge"]

# ---- Runtime: no compiler, no uv, no build cache ---------------------------
FROM python:3.13-slim-trixie AS runtime

# upgrade: picks up Debian security patches already published for the base
# image (e.g. openssl) without waiting on a new python:3.13-slim-trixie
# release — see `docker scout cves` findings.
# libgomp1: onnxruntime (pulled in by magika, markitdown's file-type-
# detection dependency) dlopens it at import time; Debian's slim image
# doesn't ship it, and skipping this turns into an ImportError on the first
# request that touches ingestion/parsing rather than a build failure.
# (docling used to be the reason for this too, pulling in torch alongside
# it — removed as dead weight, see harness/adapters/parsing/router.py.)
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --system harness && useradd --system --gid harness --no-create-home harness

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src

# Exactly one target's config gets baked in, chosen at build time — not
# "whatever happens to be sitting in config/targets/ locally" (those files
# are gitignored, not dockerignored, so that could silently include every
# other deployment's config too). Build with, e.g.:
#   docker build --build-arg TARGET_NAME=luismonbo-com .
# No default on purpose: with an empty name the COPY below finds no file and the
# build stops, rather than shipping with no config or the wrong one. `make
# proxy` says so before it starts. There is no config-free target: example.yaml
# is a template that can't boot.
ARG TARGET_NAME
COPY config/targets/${TARGET_NAME}.yaml config/targets/${TARGET_NAME}.yaml

# The image already knows which target it was built for — default
# HARNESS_TARGET to it so the only thing the deployed container needs at
# runtime is secrets. A real env var at deploy time still overrides this if
# ever needed.
ENV HARNESS_TARGET=${TARGET_NAME}
ENV PATH="/app/.venv/bin:$PATH"

# LangGraph rebuilds only its built-in safe types and the ones a graph's state
# declares from a checkpoint, instead of whatever type a checkpoint names, so write
# access to the checkpoint tables is not code execution on the next load
# (CVE-2026-28277). It reads this when it is imported, so it has to be set here and
# cannot come from .env, which the API loads after its imports.
ENV LANGGRAPH_STRICT_MSGPACK=true

# Nothing under /app is writable by this user: state lives in Postgres (the
# deployment targets set checkpointer: postgres), so a SQLite default fails at
# boot instead of writing into the image.
USER harness

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "harness.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "src"]
