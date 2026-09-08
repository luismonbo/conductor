"""Shared Langfuse configuration gate.

One presence check reused everywhere Langfuse instrumentation could fire —
main.py's LangChain callback, the LLM adapters' raw-SDK patch, and the tool
execution span. Centralized so all three stay in lockstep; drifting would
mean e.g. LLM calls get traced but tool calls silently don't.
"""
from __future__ import annotations

import os


def langfuse_configured() -> bool:
    """True when both Langfuse keys are present in the environment.

    The SDK reads LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST
    itself; callers only need to know whether to touch Langfuse at all.
    Calling into an unconfigured client doesn't raise, but it does log an
    error on every single call — this gate keeps credential-free runs
    (tests, default dev) silent rather than noisy.
    """
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))
