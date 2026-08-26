"""Deployment-target config resolution.

A target is a named, checked-in-but-gitignored YAML file
(config/targets/<name>.yaml) holding non-secret Settings overrides for one
deployment of this codebase — e.g. the daily driver vs. luismonbo.com.
Selected via HARNESS_TARGET; see config/targets/example.yaml for the shape
and settings.get_settings() for how it's layered under real environment
variables.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_TARGETS_DIR = Path("config/targets")

# Fields that can carry a credential — never allowed in a target file.
# Secrets always come from real environment variables / .env locally, or
# the deployment's own environment in production.
SECRET_FIELDS = frozenset(
    {
        "llm_api_key",
        "azure_api_key",
        "embedding_api_key",
        "memory_url",
        "pgvector_url",
        "checkpointer_url",
    }
)


def resolve_target_overrides(
    target_name: str, base_dir: Path = DEFAULT_TARGETS_DIR
) -> dict[str, Any]:
    """Load one target file's raw Settings overrides.

    Raises FileNotFoundError if the target doesn't exist, ValueError if it
    sets a secret field. Does not validate field names against Settings —
    the caller (get_settings()) does that, to avoid a circular import.
    """
    path = base_dir / f"{target_name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(
            f"HARNESS_TARGET={target_name!r} but {path} does not exist. "
            f"See config/targets/example.yaml to create it."
        )

    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"{path} must contain a mapping of field: value, got {type(raw).__name__}"
        )

    secret_keys = SECRET_FIELDS & raw.keys()
    if secret_keys:
        raise ValueError(
            f"{path} sets secret field(s) {sorted(secret_keys)} — secrets belong in "
            f"real environment variables, never in a target file."
        )

    return raw
