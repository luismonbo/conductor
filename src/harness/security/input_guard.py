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
from collections.abc import Sequence
from dataclasses import dataclass

from harness.core.guard import (
    GuardCheck,
    GuardClassifier,
    GuardClassifierError,
    GuardDecision,
)
from harness.security.battery import (
    BATTERY,
    BATTERY_VERSION,
    REFUSAL_DAILY_BUDGET,
    REFUSAL_MESSAGE_TOO_LONG,
    REFUSAL_THREAD_TOO_LONG,
)

logger = logging.getLogger(__name__)

PREVIOUS_REPLY_CHARS = 1000
NO_PREVIOUS_REPLY = "(none — this is the first message)"


@dataclass(frozen=True)
class GuardContext:
    message: str
    system_prompt: str
    previous_reply: str | None = None  # full text; trimmed to the tail here
    thread_turns: int = 0  # runs already recorded on this thread
    tokens_today: int = 0  # input + output tokens recorded since UTC midnight


@dataclass(frozen=True)
class GuardLimits:
    """Each limit is off at 0."""

    max_message_chars: int = 0
    max_thread_turns: int = 0
    daily_token_budget: int = 0


class InputGuard:
    def __init__(
        self,
        classifier: GuardClassifier | None,
        limits: GuardLimits,
        timeout_seconds: float,
        checks: Sequence[GuardCheck] = BATTERY,
        battery_version: str = BATTERY_VERSION,
    ) -> None:
        self._classifier = classifier
        self._limits = limits
        self._timeout = timeout_seconds
        self._checks = tuple(checks)
        self._battery_version = battery_version

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

    def _check_limits(self, ctx: GuardContext) -> GuardDecision | None:
        limits = self._limits
        if limits.daily_token_budget and ctx.tokens_today >= limits.daily_token_budget:
            return _precap("daily_budget", REFUSAL_DAILY_BUDGET)
        if limits.max_message_chars and len(ctx.message) > limits.max_message_chars:
            refusal = REFUSAL_MESSAGE_TOO_LONG.format(max_chars=limits.max_message_chars)
            return _precap("message_too_long", refusal)
        if limits.max_thread_turns and ctx.thread_turns >= limits.max_thread_turns:
            return _precap("thread_too_long", REFUSAL_THREAD_TOO_LONG)
        return None

    async def _classify(
        self, ctx: GuardContext, classifier: GuardClassifier
    ) -> GuardDecision:
        started = time.perf_counter()
        try:
            result = await asyncio.wait_for(
                classifier.classify(_state(ctx), self._checks), timeout=self._timeout
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
        return GuardDecision(
            allowed=fired is None,
            stage="classifier",
            reason=fired.id if fired else None,
            refusal=fired.refusal if fired else None,
            probabilities=dict(result.probabilities),
            battery_version=self._battery_version,
            model=result.model,
            request_id=result.request_id,
            latency_ms=_elapsed_ms(started),
            input_tokens=result.input_tokens,
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
