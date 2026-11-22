"""The purge job's image installs pyproject's `purge` dependency group minus the packages
the Dockerfile's `purge-builder` stage leaves out, and that must be exactly what a pass
imports: every one of them, or the job fails in production rather than here, and
nothing else, so the image stays as small as it can. Design:
docs/superpowers/specs/2026-10-14-retention-purge-design.md §8.3.
"""
from __future__ import annotations

import functools
import importlib.metadata as md
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

from packaging.markers import Marker

_ROOT = Path(__file__).resolve().parents[2]

# What a pass imports: the command and purge(), plus the modules
# orchestration/storage.py's _open_postgres and _checkpoint_serde import when they run.
# In the image, a real pass (a dry run, then one that deleted a thread) loaded the same
# packages as importing these (checked 2026-11-18).
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

# The image's platform, for uv.lock's environment markers (psycopg's tzdata only
# installs on Windows). Names not set here come from the interpreter running the test.
_IMAGE_PLATFORM = {
    "sys_platform": "linux",
    "platform_system": "Linux",
    "os_name": "posix",
    "platform_machine": "x86_64",
    "implementation_name": "cpython",
    "platform_python_implementation": "CPython",
}


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


# Cached: both tests need it, and it starts an interpreter.
@functools.cache
def _needed() -> frozenset[str]:
    """The packages a pass imports."""
    return frozenset(_owning_distributions(_loaded_files()))


def _group_closure(group: str) -> set[str]:
    """Every package uv.lock installs on the image for `group`: its entries and all they
    depend on, leaving out dependencies whose marker excludes the image's platform."""
    lock = tomllib.loads((_ROOT / "uv.lock").read_text())
    packages = {_normalize(package["name"]): package for package in lock["package"]}
    groups = packages["agent-harness"].get("dev-dependencies", {})
    assert group in groups, f"uv.lock has no {group!r} dependency group"

    closure: set[str] = set()
    extras_seen: set[tuple[str, str]] = set()
    todo = list(groups[group])
    while todo:
        dependency = todo.pop()
        marker = dependency.get("marker")
        if marker and not Marker(marker).evaluate(_IMAGE_PLATFORM):
            continue
        name = _normalize(dependency["name"])
        package = packages[name]
        if name not in closure:
            closure.add(name)
            todo.extend(package.get("dependencies", []))
        # psycopg[binary] pulls psycopg-binary in through the extra.
        for extra in dependency.get("extra", []):
            if (name, extra) not in extras_seen:
                extras_seen.add((name, extra))
                todo.extend(package.get("optional-dependencies", {}).get(extra, []))
    return closure


def _skipped_by_the_purge_stage() -> set[str]:
    """The packages the Dockerfile's purge-builder stage leaves out (--no-install-package)."""
    dockerfile = (_ROOT / "Dockerfile").read_text()
    stage = re.search(r"^FROM \S+ AS purge-builder$(.*?)^FROM ", dockerfile, re.M | re.S)
    assert stage, "the Dockerfile has no purge-builder stage"
    code = "\n".join(
        line for line in stage[1].splitlines() if not line.lstrip().startswith("#")
    )
    return {_normalize(name) for name in re.findall(r"--no-install-package\s+([\w.-]+)", code)}


def _image_packages() -> set[str]:
    return _group_closure("purge") - _skipped_by_the_purge_stage()


def test_the_purge_image_installs_every_package_the_purge_imports():
    needed = _needed()
    # The file-to-package map found the packages a pass is known to load; an empty
    # map would let any image pass.
    assert {"psycopg", "langgraph-checkpoint-postgres", "pydantic-settings"} <= needed, needed

    missing = sorted(needed - _image_packages())
    assert not missing, (
        f"the purge imports packages its image doesn't install: {missing}. Add them to "
        "pyproject's purge group, or stop leaving them out in the Dockerfile's "
        "purge-builder stage"
    )


def test_the_purge_image_installs_nothing_a_pass_never_imports():
    unused = sorted(_image_packages() - _needed())
    assert not unused, (
        f"the purge image installs packages a pass never imports: {unused}. Leave each "
        "out with --no-install-package in the Dockerfile's purge-builder stage"
    )
