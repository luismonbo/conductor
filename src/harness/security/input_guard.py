"""Input guard policy: fixed limits, then the classifier, then thresholds.

Runs at the API boundary, before the agent ever sees the message (see
docs/superpowers/specs/2026-09-27-input-guard-design.md). It does no I/O of
its own besides the classifier call: the API layer gathers everything else
into a GuardContext.

Fails open: any classifier failure lets the message through, because the
chatbot's availability must never depend on the classifier. The fixed limits
still apply. Cancellation is never swallowed.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from harness.core.guard import (
    GuardCheck,
    GuardClassifier,
    GuardClassifierError,
    GuardDecision,
    GuardQuestion,
)
from harness.core.language import DEFAULT_LANGUAGE, Language
from harness.security.battery import (
    BATTERY,
    BATTERY_VERSION,
    LANGUAGE_QUESTIONS,
    LANGUAGE_THRESHOLD,
)
from harness.security.visitor_text import TEXT_KEYS, TextKey, visitor_text

logger = logging.getLogger(__name__)

PREVIOUS_REPLY_CHARS = 1000
NO_PREVIOUS_REPLY = "(none — this is the first message)"


def pick_language(
    probabilities: Mapping[str, float],
    *,
    threshold: float = LANGUAGE_THRESHOLD,
    questions: Mapping[Language, GuardQuestion] = LANGUAGE_QUESTIONS,
) -> Language:
    """The language whose question scored highest at or above `threshold`, else English.

    Ties go to the language listed first. A question with no score counts as 0.
    """
    best: Language = DEFAULT_LANGUAGE
    best_score = -1.0
    for code, question in questions.items():
        score = probabilities.get(question.id, 0.0)
        if score >= threshold and score > best_score:
            best, best_score = code, score
    return best


@dataclass(frozen=True)
class GuardContext:
    message: str
    system_prompt: str
    previous_reply: str | None = None  # full text; trimmed to the tail here
    tokens_today: int = 0  # input + output tokens recorded since UTC midnight


@dataclass(frozen=True)
class GuardLimits:
    """Each limit is off at 0."""

    max_message_chars: int = 0
    daily_token_budget: int = 0


class InputGuard:
    def __init__(
        self,
        classifier: GuardClassifier | None,
        limits: GuardLimits,
        timeout_seconds: float,
        checks: Sequence[GuardCheck] = BATTERY,
        battery_version: str = BATTERY_VERSION,
        language_questions: Mapping[Language, GuardQuestion] = LANGUAGE_QUESTIONS,
        language_threshold: float = LANGUAGE_THRESHOLD,
    ) -> None:
        for check in checks:
            if check.refusal not in TEXT_KEYS:
                raise ValueError(
                    f"check {check.id!r} names refusal {check.refusal!r}, which "
                    f"security/visitor_text.py does not have"
                )
        self._classifier = classifier
        self._limits = limits
        self._timeout = timeout_seconds
        self._checks = tuple(checks)
        self._battery_version = battery_version
        self._language_questions = dict(language_questions)
        self._language_threshold = language_threshold

    @property
    def classifier_enabled(self) -> bool:
        return self._classifier is not None

    @property
    def limits(self) -> GuardLimits:
        return self._limits

    async def screen(self, ctx: GuardContext) -> GuardDecision:
        blocked = self._check_limits(ctx)
        if blocked is not None:
            return blocked
        if self._classifier is None:
            return GuardDecision(allowed=True, stage="none")
        return await self._classify(ctx, self._classifier)

    async def aclose(self) -> None:
        if self._classifier is not None:
            await self._classifier.aclose()

    def _refusal(self, key: str, language: Language) -> str:
        """The visitor's text for a refusal key (checked at construction)."""
        return visitor_text(
            cast(TextKey, key), language, max_chars=self._limits.max_message_chars
        )

    def _check_limits(self, ctx: GuardContext) -> GuardDecision | None:
        limits = self._limits
        if limits.daily_token_budget and ctx.tokens_today >= limits.daily_token_budget:
            return _precap("daily_budget", self._refusal("daily_budget", DEFAULT_LANGUAGE))
        if limits.max_message_chars and len(ctx.message) > limits.max_message_chars:
            refusal = self._refusal("message_too_long", DEFAULT_LANGUAGE)
            return _precap("message_too_long", refusal)
        return None

    async def _classify(
        self, ctx: GuardContext, classifier: GuardClassifier
    ) -> GuardDecision:
        started = time.perf_counter()
        # One request carries the checks and the language questions.
        questions = (*self._checks, *self._language_questions.values())
        try:
            result = await asyncio.wait_for(
                classifier.classify(_state(ctx), questions), timeout=self._timeout
            )
        except TimeoutError:
            logger.warning(
                "input guard: classifier timed out after %.1fs; failing open",
                self._timeout,
            )
            return _fallback("TimeoutError", started)
        except GuardClassifierError as exc:
            log = logger.error if exc.misconfigured else logger.warning
            log("input guard: classifier failed (%s); failing open", exc.cause)
            return _fallback(exc.cause, started)
        except Exception as exc:
            logger.exception("input guard: unexpected classifier error; failing open")
            return _fallback(type(exc).__name__, started)

        missing = [c.id for c in self._checks if c.id not in result.probabilities]
        if missing:
            logger.error("input guard: classifier omitted %s; failing open", missing)
            return _fallback("MissingProbability", started)

        fired = next(
            (c for c in self._checks if result.probabilities[c.id] >= c.threshold), None
        )
        language_ids = {q.id for q in self._language_questions.values()}
        language_probabilities = {
            qid: p for qid, p in result.probabilities.items() if qid in language_ids
        }
        language = pick_language(
            language_probabilities,
            threshold=self._language_threshold,
            questions=self._language_questions,
        )
        return GuardDecision(
            allowed=fired is None,
            stage="classifier",
            reason=fired.id if fired else None,
            refusal=self._refusal(fired.refusal, language) if fired else None,
            probabilities={c.id: result.probabilities[c.id] for c in self._checks},
            battery_version=self._battery_version,
            model=result.model,
            request_id=result.request_id,
            latency_ms=_elapsed_ms(started),
            input_tokens=result.input_tokens,
            language=language,
            language_probabilities=language_probabilities,
        )


def _state(ctx: GuardContext) -> dict[str, str]:
    previous = (
        ctx.previous_reply[-PREVIOUS_REPLY_CHARS:]
        if ctx.previous_reply
        else NO_PREVIOUS_REPLY
    )
    return {
        "assistant_system_prompt": ctx.system_prompt,
        "previous_assistant_reply": previous,
        "visitor_message": ctx.message,
    }


def _precap(reason: str, refusal: str) -> GuardDecision:
    return GuardDecision(allowed=False, stage="precap", reason=reason, refusal=refusal)


def _fallback(error: str, started: float) -> GuardDecision:
    return GuardDecision(
        allowed=True,
        stage="fallback",
        reason="guard_unavailable",
        latency_ms=_elapsed_ms(started),
        error=error,
    )


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
