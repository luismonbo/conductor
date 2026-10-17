"""Importing orchestration.build must stay light.

Ingestion, rag_query, scripts/rag.py and the eval scripts import it and never run an
agent, so build_agent defers LangGraph's import to the call that compiles a graph.
Nothing at module level, storage.py's imports included, may undo that.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"


def test_importing_build_does_not_load_langgraph():
    code = (
        "import sys, harness.orchestration.build; "
        "print(sorted(m for m in sys.modules if m.startswith('langgraph')))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True,
        timeout=60, env={**os.environ, "PYTHONPATH": str(_SRC)},
    )
    assert result.stdout.strip() == "[]"
