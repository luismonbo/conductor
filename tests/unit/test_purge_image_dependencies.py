"""The purge job's image installs only pyproject's `purge` dependency group (the
Dockerfile's `purge` stage), so the group must cover everything a pass imports. If the
purge starts importing a package outside it, this fails here, not on the job's first
run in production. Design: docs/superpowers/specs/2026-10-14-retention-purge-design.md §8.3.
"""
from __future__ import annotations

import importlib.metadata as md
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]

# What a pass imports: the command and purge(), plus the modules
# orchestration/storage.py's _open_postgres and _checkpoint_serde import when they run.
_PURGE_MODULES = (
    "harness.cli.purge",
    "harness.orchestration.retention",
    "langgraph.checkpoint.postgres.aio",
    "langgraph.checkpoint.serde.jsonplus",
    "harness.adapters.postgres.pool",
    "harness.observability.run_store.postgres",
)

# Imports the modules in a fresh interpreter and prints the file of every module those
# imports loaded. What was loaded before them is left out: a venv's .pth files load
# modules at startup (setuptools' _distutils_hack), and the image has none of them.
_PRINT_LOADED_FILES = """
import importlib, sys
before = set(sys.modules)
for name in sys.argv[1:]:
    importlib.import_module(name)
for name, module in list(sys.modules.items()):
    path = getattr(module, "__file__", None)
    if name not in before and path:
        print(path)
"""


def _normalize(name: str) -> str:
    """A distribution name as PEP 503 compares them."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _loaded_files() -> list[str]:
    done = subprocess.run(
        [sys.executable, "-c", _PRINT_LOADED_FILES, *_PURGE_MODULES],
        capture_output=True, text=True, check=True, timeout=120,
    )
    return done.stdout.splitlines()


def _owning_distributions(files: list[str]) -> set[str]:
    """The installed distribution that lists each file in its RECORD.

    The project's own src/ files and the standard library belong to none.
    """
    # normpath, not resolve(): both sides come from the same sys.path entries, and
    # resolving every installed file's path takes seconds.
    owner: dict[str, str] = {}
    for dist in md.distributions():
        name = _normalize(dist.metadata["Name"])
        for file in dist.files or []:
            owner[os.path.normpath(dist.locate_file(file))] = name
    return {owner[path] for path in map(os.path.normpath, files) if path in owner}


def _group_closure(group: str) -> set[str]:
    """Every package uv.lock installs for `group`: its entries and all they depend on."""
    lock = tomllib.loads((_ROOT / "uv.lock").read_text())
    packages = {_normalize(package["name"]): package for package in lock["package"]}
    groups = packages["agent-harness"].get("dev-dependencies", {})
    assert group in groups, f"uv.lock has no {group!r} dependency group"

    closure: set[str] = set()
    extras_seen: set[tuple[str, str]] = set()
    todo = list(groups[group])
    while todo:
        dependency = todo.pop()
        name = _normalize(dependency["name"])
        package = packages[name]
        if name not in closure:
            closure.add(name)
            todo.extend(package.get("dependencies", []))
        # psycopg[binary] pulls psycopg-binary in through the extra. Markers are
        # ignored: they can only make the closure larger than one platform needs.
        for extra in dependency.get("extra", []):
            if (name, extra) not in extras_seen:
                extras_seen.add((name, extra))
                todo.extend(package.get("optional-dependencies", {}).get(extra, []))
    return closure


def test_the_purge_group_has_every_package_the_purge_imports():
    needed = _owning_distributions(_loaded_files())
    closure = _group_closure("purge")

    # The file-to-package map found the packages a pass is known to load; an empty
    # map would let any group pass.
    assert {"psycopg", "langgraph-checkpoint-postgres", "pydantic-settings"} <= needed, needed

    missing = sorted(needed - closure)
    assert not missing, f"the purge imports packages outside the purge group: {missing}"
