# syntax=docker/dockerfile:1

# ---- Builder: resolve dependencies + install the project ------------------
FROM python:3.13-slim-trixie AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_DEV=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# Dependencies first, in their own layer, so code-only changes below don't
# invalidate the install. Extras are named explicitly rather than
# --all-extras: pyproject.toml defines a `dev` group under
# [project.optional-dependencies] (separate from the [dependency-groups] one
# UV_NO_DEV covers) — --all-extras would pull pytest/ruff/mypy into a
# production image.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --extra azure --extra pgvector --extra local

COPY src/ src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --extra azure --extra pgvector --extra local

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
# No default on purpose: a build that forgets to specify a target fails
# here, loudly, rather than shipping with no config or the wrong one. Use
# TARGET_NAME=example for a config-free smoke-test build.
ARG TARGET_NAME
COPY config/targets/${TARGET_NAME}.yaml config/targets/${TARGET_NAME}.yaml

# The image already knows which target it was built for — default
# HARNESS_TARGET to it so the only thing the deployed container needs at
# runtime is secrets. A real env var at deploy time still overrides this if
# ever needed.
ENV HARNESS_TARGET=${TARGET_NAME}
ENV PATH="/app/.venv/bin:$PATH"

USER harness

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "harness.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "src"]
