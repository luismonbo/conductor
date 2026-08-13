"""Sanity-check litellm_config.yaml before the stack starts.

Run by `make up`; also unit-tested. Exit 1 on problems so a broken config
fails fast instead of surfacing as opaque proxy 400s.
"""
from __future__ import annotations

import sys

import yaml


def validate_config(path: str) -> list[str]:
    problems: list[str] = []
    try:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    except Exception as exc:
        return [f"cannot parse {path}: {exc}"]

    model_list = data.get("model_list") or []
    if not model_list:
        problems.append("model_list is empty")

    seen: set[str] = set()
    for entry in model_list:
        name = (entry or {}).get("model_name") or ""
        params = (entry or {}).get("litellm_params") or {}
        if not name:
            problems.append("entry with empty model_name")
            continue
        if name in seen:
            problems.append(f"duplicate model_name: {name}")
        seen.add(name)
        if not params.get("model"):
            problems.append(f"{name}: litellm_params.model is missing")
    return problems


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "litellm_config.yaml"
    found = validate_config(path)
    for p in found:
        print(f"[litellm-config] {p}", file=sys.stderr)
    sys.exit(1 if found else 0)
