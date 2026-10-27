"""InputGuard policy: limits first, then the classifier, then thresholds; fails open."""
from __future__ import annotations

import asyncio
import logging

import pytest

from harness.adapters.guard.fake import FakeClassifier
from harness.core.guard import ClassifierResult, Criterion, GuardCheck, GuardClassifierError
from harness.security.battery import REFUSAL_DAILY_BUDGET, REFUSAL_THREAD_TOO_LONG
from harness.security.input_guard import (
    NO_PREVIOUS_REPLY,
    PREVIOUS_REPLY_CHARS,
    GuardContext,
    GuardLimits,
    InputGuard,
)

_C = Criterion(definition="d")
FIRST = GuardCheck(id="first", question="q1", yes=_C, no=_C, threshold=0.7, refusal="refusal-first")
SECOND = GuardCheck(
    id="second", question="q2", yes=_C, no=_C, threshold=0.8, refusal="refusal-second"
)
CHECKS = (FIRST, SECOND)


def _guard(classifier=None, limits=GuardLimits(), timeout=1.0) -> InputGuard:
    return InputGuard(classifier, limits, timeout, checks=CHECKS, battery_version="test-v")


def _ctx(message: str = "hello", **kwargs) -> GuardContext:
    return GuardContext(message=message, system_prompt="You answer questions about X.", **kwargs)


# --- fixed limits -----------------------------------------------------------

async def test_message_at_the_limit_passes_and_one_over_blocks():
    guard = _guard(limits=GuardLimits(max_message_chars=5))
    assert (await guard.screen(_ctx("x" * 5))).allowed
    decision = await guard.screen(_ctx("x" * 6))
    assert (decision.allowed, decision.stage, decision.reason) == (
        False, "precap", "message_too_long",
    )
    assert "under 5 characters" in decision.refusal


async def test_message_length_counts_characters_not_bytes():
    guard = _guard(limits=GuardLimits(max_message_chars=2000))
    assert (await guard.screen(_ctx("é" * 2000))).allowed
    assert not (await guard.screen(_ctx("é" * 2001))).allowed


async def test_thread_turn_limit_blocks_once_reached():
    guard = _guard(limits=GuardLimits(max_thread_turns=20))
    assert (await guard.screen(_ctx(thread_turns=19))).allowed
    decision = await guard.screen(_ctx(thread_turns=20))
    assert (decision.allowed, decision.reason, decision.refusal) == (
        False, "thread_too_long", REFUSAL_THREAD_TOO_LONG,
    )


async def test_daily_budget_blocks_once_reached():
    guard = _guard(limits=GuardLimits(daily_token_budget=100))
    assert (await guard.screen(_ctx(tokens_today=99))).allowed
    decision = await guard.screen(_ctx(tokens_today=100))
    assert (decision.allowed, decision.reason, decision.refusal) == (
        False, "daily_budget", REFUSAL_DAILY_BUDGET,
    )


async def test_limits_are_checked_budget_then_length_then_turns():
    guard = _guard(
        limits=GuardLimits(max_message_chars=1, max_thread_turns=1, daily_token_budget=1)
    )
    assert (await guard.screen(_ctx("xx", thread_turns=5, tokens_today=5))).reason == (
        "daily_budget"
    )
    assert (await guard.screen(_ctx("xx", thread_turns=5))).reason == "message_too_long"


async def test_zero_disables_every_limit():
    decision = await _guard(limits=GuardLimits()).screen(
        _ctx("x" * 100_000, thread_turns=10_000, tokens_today=10**9)
    )
    assert decision.allowed


async def test_a_limit_hit_never_calls_the_classifier():
    fake = FakeClassifier()
    await _guard(fake, limits=GuardLimits(max_message_chars=1)).screen(_ctx("too long"))
    assert fake.calls == []


# --- classifier and thresholds ------------------------------------------------

async def test_no_classifier_means_nothing_was_evaluated():
    decision = await _guard().screen(_ctx())
    assert (decision.allowed, decision.stage) == (True, "none")


async def test_below_every_threshold_is_allowed_with_full_metadata():
    decision = await _guard(FakeClassifier({"first": 0.69, "second": 0.79})).screen(_ctx())
    assert (decision.allowed, decision.stage, decision.reason) == (True, "classifier", None)
    assert decision.probabilities == {"first": 0.69, "second": 0.79}
    assert decision.battery_version == "test-v"
    assert decision.model == "fake-decision"
    assert decision.request_id == "fake-request"
    assert decision.latency_ms is not None


async def test_at_threshold_blocks_with_that_checks_refusal():
    decision = await _guard(FakeClassifier({"second": 0.8})).screen(_ctx())
    assert (decision.allowed, decision.reason, decision.refusal) == (
        False, "second", "refusal-second",
    )


async def test_first_check_in_order_wins_when_several_fire():
    decision = await _guard(FakeClassifier({"first": 0.9, "second": 0.99})).screen(_ctx())
    assert decision.reason == "first"
    assert decision.probabilities == {"first": 0.9, "second": 0.99}


async def test_state_carries_prompt_trimmed_previous_reply_and_message():
    fake = FakeClassifier()
    reply = "a" * 10 + "b" * PREVIOUS_REPLY_CHARS
    await _guard(fake).screen(_ctx("the message", previous_reply=reply))
    assert fake.calls == [{
        "assistant_system_prompt": "You answer questions about X.",
        "previous_assistant_reply": "b" * PREVIOUS_REPLY_CHARS,
        "visitor_message": "the message",
    }]


async def test_first_message_uses_the_no_previous_reply_sentinel():
    fake = FakeClassifier()
    await _guard(fake).screen(_ctx())
    assert fake.calls[0]["previous_assistant_reply"] == NO_PREVIOUS_REPLY


# --- fail open ------------------------------------------------------------------

class _SlowClassifier(FakeClassifier):
    async def classify(self, state, checks):
        await asyncio.sleep(1)
        return await super().classify(state, checks)


class _PartialClassifier(FakeClassifier):
    async def classify(self, state, checks):
        return ClassifierResult(
            probabilities={"first": 0.1}, model="m", request_id=None, input_tokens=0
        )


async def test_timeout_fails_open():
    decision = await _guard(_SlowClassifier(), timeout=0.01).screen(_ctx())
    assert (decision.allowed, decision.stage, decision.reason, decision.error) == (
        True, "fallback", "guard_unavailable", "TimeoutError",
    )


@pytest.mark.parametrize(
    "misconfigured,level", [(False, logging.WARNING), (True, logging.ERROR)]
)
async def test_classifier_error_fails_open_and_logs_by_severity(caplog, misconfigured, level):
    error = GuardClassifierError("boom", cause="HTTP401", misconfigured=misconfigured)
    with caplog.at_level(logging.WARNING, logger="harness.security.input_guard"):
        decision = await _guard(FakeClassifier(error=error)).screen(_ctx())
    assert (decision.allowed, decision.stage, decision.error) == (
        True, "fallback", "HTTP401",
    )
    assert caplog.records[-1].levelno == level


async def test_unexpected_error_fails_open():
    decision = await _guard(FakeClassifier(error=KeyError("bug"))).screen(_ctx())
    assert (decision.allowed, decision.stage, decision.error) == (True, "fallback", "KeyError")


async def test_missing_probability_fails_open():
    decision = await _guard(_PartialClassifier()).screen(_ctx())
    assert (decision.allowed, decision.stage, decision.error) == (
        True, "fallback", "MissingProbability",
    )


async def test_cancellation_is_never_swallowed():
    """A client disconnect must cancel the request, never turn into an allow."""
    with pytest.raises(asyncio.CancelledError):
        await _guard(FakeClassifier(error=asyncio.CancelledError())).screen(_ctx())


async def test_aclose_closes_the_classifier():
    fake = FakeClassifier()
    await _guard(fake).aclose()
    assert fake.closed


def test_exposes_whether_a_classifier_is_on_and_its_limits():
    limits = GuardLimits(max_message_chars=7)
    assert _guard(None, limits).classifier_enabled is False
    assert _guard(FakeClassifier(), limits).classifier_enabled is True
    assert _guard(None, limits).limits == limits
