.PHONY: up down infra api web dev test init-dbs logs

# Bring up infra (postgres, litellm, langfuse) and print next steps.
up:
	uv run python scripts/validate_litellm_config.py litellm_config.yaml
	docker compose up -d postgres litellm langfuse-web langfuse-worker clickhouse redis minio
	@echo "infra up — API: make api   frontend: make web   langfuse: http://localhost:3000   litellm ui: http://localhost:4000/ui"

down:
	docker compose down

infra:
	docker compose up -d postgres

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
	docker compose logs -f litellm langfuse-web
