"""Input guard policy: fixed limits, then the classifier, then thresholds.

Runs at the API boundary, before the agent ever sees the message (see
docs/superpowers/specs/2026-09-27-input-guard-design.md). It does no I/O of
its own besides the classifier call: the API layer gathers everything else
into a GuardContext.

Fails open: any classifier failure lets the message through, because the
chatbot's availability must never depend on the classifier. The fixed limits
still apply. Cancellation is never swallowed.

The same request asks the visitor's language, and every refusal comes in it. A
limit that refuses first still asks the language questions alone, with the
message's first LANGUAGE_ONLY_MESSAGE_CHARS characters; when that request
fails, the refusal is English.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from harness.core.guard import (
    ClassifierResult,
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
# A limit's refusal asks for the language with only this much of the message.
LANGUAGE_ONLY_MESSAGE_CHARS = 500


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
        limit = _limit_hit(ctx, self._limits)
        if limit is not None:
            return await self._refuse_at_limit(ctx, limit)
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

    def _language(
        self, probabilities: Mapping[str, float]
    ) -> tuple[dict[str, float], Language]:
        """The language questions' scores out of a result, and the language they pick."""
        ids = {q.id for q in self._language_questions.values()}
        scores = {qid: p for qid, p in probabilities.items() if qid in ids}
        language = pick_language(
            scores, threshold=self._language_threshold, questions=self._language_questions
        )
        return scores, language

    async def _ask(
        self,
        classifier: GuardClassifier,
        state: Mapping[str, str],
        questions: Sequence[GuardQuestion],
        *,
        on_failure: str,
    ) -> tuple[ClassifierResult | None, str | None, int]:
        """One classifier call: (result, None, ms), or (None, cause, ms) when it failed.

        `on_failure` ends the log line: what the guard does instead.
        """
        started = time.perf_counter()
        try:
            result = await asyncio.wait_for(
                classifier.classify(state, questions), timeout=self._timeout
            )
        except TimeoutError:
            logger.warning(
                "input guard: classifier timed out after %.1fs; %s", self._timeout, on_failure
            )
            return None, "TimeoutError", _elapsed_ms(started)
        except GuardClassifierError as exc:
            log = logger.error if exc.misconfigured else logger.warning
            log("input guard: classifier failed (%s); %s", exc.cause, on_failure)
            return None, exc.cause, _elapsed_ms(started)
        except Exception as exc:
            logger.exception("input guard: unexpected classifier error; %s", on_failure)
            return None, type(exc).__name__, _elapsed_ms(started)
        return result, None, _elapsed_ms(started)

    async def _refuse_at_limit(self, ctx: GuardContext, reason: TextKey) -> GuardDecision:
        """A limit's refusal, in the language a language-only request finds.

        The request carries only the language questions, the start of the message
        and the end of the previous reply. Without a classifier, or when the
        request fails, the refusal is English.
        """
        if self._classifier is None:
            return GuardDecision(
                allowed=False, stage="precap", reason=reason,
                refusal=self._refusal(reason, DEFAULT_LANGUAGE),
            )
        state = {
            "previous_assistant_reply": _previous_reply_tail(ctx),
            "visitor_message": ctx.message[:LANGUAGE_ONLY_MESSAGE_CHARS],
        }
        result, error, latency_ms = await self._ask(
            self._classifier, state, tuple(self._language_questions.values()),
            on_failure="refusing in English",
        )
        scores, language = (
            self._language(result.probabilities) if result else (None, DEFAULT_LANGUAGE)
        )
        return GuardDecision(
            allowed=False,
            stage="precap",
            reason=reason,
            refusal=self._refusal(reason, language),
            model=result.model if result else None,
            request_id=result.request_id if result else None,
            latency_ms=latency_ms,
            input_tokens=result.input_tokens if result else None,
            error=error,
            language=language,
            language_probabilities=scores,
        )

    async def _classify(
        self, ctx: GuardContext, classifier: GuardClassifier
    ) -> GuardDecision:
        # One request carries the checks and the language questions.
        questions = (*self._checks, *self._language_questions.values())
        result, error, latency_ms = await self._ask(
            classifier, _state(ctx), questions, on_failure="failing open"
        )
        if result is None:
            return _fallback(error or "unknown", latency_ms)

        missing = [c.id for c in self._checks if c.id not in result.probabilities]
        if missing:
            logger.error("input guard: classifier omitted %s; failing open", missing)
            return _fallback("MissingProbability", latency_ms)

        fired = next(
            (c for c in self._checks if result.probabilities[c.id] >= c.threshold), None
        )
        scores, language = self._language(result.probabilities)
        return GuardDecision(
            allowed=fired is None,
            stage="classifier",
            reason=fired.id if fired else None,
            refusal=self._refusal(fired.refusal, language) if fired else None,
            probabilities={c.id: result.probabilities[c.id] for c in self._checks},
            battery_version=self._battery_version,
            model=result.model,
            request_id=result.request_id,
            latency_ms=latency_ms,
            input_tokens=result.input_tokens,
            language=language,
            language_probabilities=scores,
        )


def _limit_hit(ctx: GuardContext, limits: GuardLimits) -> TextKey | None:
    """The limit this message hits, named as its refusal, or None."""
    if limits.daily_token_budget and ctx.tokens_today >= limits.daily_token_budget:
        return "daily_budget"
    if limits.max_message_chars and len(ctx.message) > limits.max_message_chars:
        return "message_too_long"
    return None


def _previous_reply_tail(ctx: GuardContext) -> str:
    if ctx.previous_reply:
        return ctx.previous_reply[-PREVIOUS_REPLY_CHARS:]
    return NO_PREVIOUS_REPLY


def _state(ctx: GuardContext) -> dict[str, str]:
    return {
        "assistant_system_prompt": ctx.system_prompt,
        "previous_assistant_reply": _previous_reply_tail(ctx),
        "visitor_message": ctx.message,
    }


def _fallback(error: str, latency_ms: int) -> GuardDecision:
    return GuardDecision(
        allowed=True,
        stage="fallback",
        reason="guard_unavailable",
        latency_ms=latency_ms,
        error=error,
    )


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
