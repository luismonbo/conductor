"""Input-guard port: probabilistic yes/no checks over a visitor message.

Pure types and a Protocol — nothing here imports a vendor SDK. The policy that
turns probabilities into allow/block lives in harness.security.input_guard;
the Microsoft-Decision-1 adapter lives in harness.adapters.guard.azure_decision. See
docs/superpowers/specs/2026-09-27-input-guard-design.md.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from harness.core.language import DEFAULT_LANGUAGE, Language


@dataclass(frozen=True)
class Criterion:
    """What one side of a yes/no check means, pinned down with examples."""

    definition: str
    examples: tuple[str, ...] = ()


@dataclass(frozen=True)
class GuardQuestion:
    """One yes/no question about a message, answered with P(yes)."""

    id: str
    question: str
    yes: Criterion
    no: Criterion


@dataclass(frozen=True)
class GuardCheck(GuardQuestion):
    """A question that blocks. Yes always means hazard."""

    threshold: float  # block when P(yes) is at or above this
    refusal: str  # the key of what the visitor sees (security/visitor_text.py)


@dataclass(frozen=True)
class ClassifierResult:
    probabilities: Mapping[str, float]  # question id -> P(yes)
    model: str  # the versioned model id that answered
    request_id: str | None
    input_tokens: int


@dataclass(frozen=True)
class GuardDecision:
    """The outcome of screening one message.

    stage is "precap" (a fixed limit fired), "classifier" (the classifier
    answered), "fallback" (the classifier failed, so the message was let
    through), or "none" (nothing was evaluated: no classifier, no limit hit).

    language is the visitor's language as the classifier judged it, English
    when it did not answer. It and language_probabilities are never stored or
    logged; probabilities holds the checks' scores only.
    """

    allowed: bool
    stage: str
    reason: str | None = None
    refusal: str | None = None  # the visitor's text, in `language`
    probabilities: Mapping[str, float] | None = None
    battery_version: str | None = None
    model: str | None = None
    request_id: str | None = None
    latency_ms: int | None = None
    input_tokens: int | None = None
    error: str | None = None
    language: Language = DEFAULT_LANGUAGE
    language_probabilities: Mapping[str, float] | None = None


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
        self, state: Mapping[str, str], questions: Sequence[GuardQuestion]
    ) -> ClassifierResult:
        """P(yes) for every question, or raise. Never returns partial results."""
        ...

    async def aclose(self) -> None: ...
