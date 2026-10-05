"""Input-guard port: probabilistic yes/no checks over a visitor message.

Pure types and a Protocol — nothing here imports a vendor SDK. The policy that
turns probabilities into allow/block lives in harness.security.input_guard;
the TypeSafe (Jev) adapter lives in harness.adapters.guard.typesafe. See
docs/superpowers/specs/2026-09-27-input-guard-design.md.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Criterion:
    """What one side of a yes/no check means, pinned down with examples."""

    definition: str
    examples: tuple[str, ...] = ()


@dataclass(frozen=True)
class GuardCheck:
    """One yes/no question. Yes always means hazard."""

    id: str
    question: str
    yes: Criterion
    no: Criterion
    threshold: float  # block when P(yes) is at or above this
    refusal: str  # what the visitor sees when this check blocks


@dataclass(frozen=True)
class ClassifierResult:
    probabilities: Mapping[str, float]  # check id -> P(yes)
    model: str  # the versioned model id that answered
    request_id: str | None
    input_tokens: int


@dataclass(frozen=True)
class GuardDecision:
    """The outcome of screening one message.

    stage is "precap" (a fixed limit fired), "classifier" (the classifier
    answered), "fallback" (the classifier failed, so the message was let
    through), or "none" (nothing was evaluated: no classifier, no limit hit).
    """

    allowed: bool
    stage: str
    reason: str | None = None
    refusal: str | None = None
    probabilities: Mapping[str, float] | None = None
    battery_version: str | None = None
    model: str | None = None
    request_id: str | None = None
    latency_ms: int | None = None
    input_tokens: int | None = None
    error: str | None = None


class GuardClassifierError(Exception):
    """A classifier call failed.

    `cause` names the underlying failure (usually an exception class name).
    `misconfigured` marks failures that recur on every call until a human
    fixes something, such as a rejected API key or a retired model id.
    """

    def __init__(self, message: str, *, cause: str, misconfigured: bool = False) -> None:
        super().__init__(message)
        self.cause = cause
        self.misconfigured = misconfigured


class GuardClassifier(Protocol):
    async def classify(
        self, state: Mapping[str, str], checks: Sequence[GuardCheck]
    ) -> ClassifierResult:
        """P(yes) for every check, or raise. Never returns partial results."""
        ...

    async def aclose(self) -> None: ...
