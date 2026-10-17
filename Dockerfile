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
# base [project] table, and the optional `dev` and `docling` extras stay out.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

COPY src/ src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

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

# Nothing under /app is writable by this user: state lives in Postgres (the
# deployment targets set checkpointer: postgres), so a SQLite default fails at
# boot instead of writing into the image.
USER harness

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "harness.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "src"]
