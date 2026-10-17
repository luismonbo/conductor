.PHONY: up down infra proxy api web dev test init-dbs logs ingest

# Bring up infra (postgres, litellm) and print next steps.
# Langfuse runs against Langfuse Cloud, not local infra — see .env.example.
up:
	uv run python scripts/infra/validate_litellm_config.py litellm_config.yaml
	docker compose up -d postgres litellm
	@echo "infra up — API: make api   frontend: make web   litellm ui: http://localhost:4000/ui"

down:
	docker compose down

infra:
	docker compose up -d postgres

# Bring up the Caddy-fronted stack (api + caddy) — the auth/proxy topology
# from docs/superpowers/specs/2026-08-28-authentication-design.md. Port 8000
# is Caddy; api itself is not published to the host in this mode. The api
# image is the deployment image: TARGET_NAME=<target> make proxy bakes
# config/targets/<target>.yaml into it.
proxy:
	@test -n "$(TARGET_NAME)" || { echo "TARGET_NAME is required: TARGET_NAME=<target> make proxy" >&2; exit 1; }
	docker compose up -d --build postgres api caddy
	@echo "proxied stack up — http://localhost:8000 (Caddy; api is internal-only)"

api:
	uv run uvicorn harness.api.main:app --reload --app-dir src

web:
	pnpm -C frontend dev

# One-time for postgres volumes created before the init SQL existed.
init-dbs:
	docker compose exec postgres psql -U harness -c 'CREATE DATABASE litellm;' || true

test:
	uv run pytest -q && pnpm -C frontend test

logs:
	docker compose logs -f litellm

# Offline RAG ingestion, containerized (docling needs Linux; never run on the
# macOS host). COLLECTION selects data/raw/<name>/ — e.g. papers or docs.
ingest:
	docker compose --profile ingest run --rm ingest --collection $(COLLECTION)
