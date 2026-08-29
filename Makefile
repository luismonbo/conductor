.PHONY: up down infra proxy api web dev test init-dbs logs

# Bring up infra (postgres, litellm) and print next steps.
# Langfuse/ClickHouse/Redis/MinIO removed for now — see docker-compose.yml.
up:
	uv run python scripts/validate_litellm_config.py litellm_config.yaml
	docker compose up -d postgres litellm
	@echo "infra up — API: make api   frontend: make web   litellm ui: http://localhost:4000/ui"

down:
	docker compose down

infra:
	docker compose up -d postgres

# Bring up the Caddy-fronted stack (api + caddy) — the auth/proxy topology
# from docs/superpowers/specs/2026-08-28-authentication-design.md. Port 8000
# is Caddy; api itself is not published to the host in this mode.
proxy:
	docker compose up -d --build postgres api caddy
	@echo "proxied stack up — http://localhost:8000 (Caddy; api is internal-only)"

api:
	uv run uvicorn harness.api.main:app --reload --app-dir src

web:
	pnpm -C frontend dev

# One-time for postgres volumes created before the init SQL existed.
init-dbs:
	docker compose exec postgres psql -U harness -c 'CREATE DATABASE litellm;' || true
	docker compose exec postgres psql -U harness -c 'CREATE DATABASE langfuse;' || true

test:
	uv run pytest -q && pnpm -C frontend test

logs:
	docker compose logs -f litellm
