"""InputGuard policy: limits first, then the classifier, then thresholds; fails open."""
from __future__ import annotations

import asyncio
import logging

import pytest

from harness.adapters.guard.fake import FakeClassifier
from harness.core.guard import ClassifierResult, Criterion, GuardCheck, GuardClassifierError
from harness.security.input_guard import (
    NO_PREVIOUS_REPLY,
    PREVIOUS_REPLY_CHARS,
    GuardContext,
    GuardLimits,
    InputGuard,
)
from harness.security.visitor_text import visitor_text

_C = Criterion(definition="d")
FIRST = GuardCheck(id="first", question="q1", yes=_C, no=_C, threshold=0.7, refusal="not_shareable")
SECOND = GuardCheck(id="second", question="q2", yes=_C, no=_C, threshold=0.8, refusal="off_topic")
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


async def test_daily_budget_blocks_once_reached():
    guard = _guard(limits=GuardLimits(daily_token_budget=100))
    assert (await guard.screen(_ctx(tokens_today=99))).allowed
    decision = await guard.screen(_ctx(tokens_today=100))
    assert (decision.allowed, decision.reason, decision.refusal) == (
        False, "daily_budget", visitor_text("daily_budget"),
    )


async def test_limits_are_checked_budget_then_length():
    guard = _guard(limits=GuardLimits(max_message_chars=1, daily_token_budget=1))
    assert (await guard.screen(_ctx("xx", tokens_today=5))).reason == "daily_budget"
    assert (await guard.screen(_ctx("xx"))).reason == "message_too_long"


async def test_zero_disables_every_limit():
    decision = await _guard(limits=GuardLimits()).screen(
        _ctx("x" * 100_000, tokens_today=10**9)
    )
    assert decision.allowed


async def test_a_limit_hit_asks_only_the_language_questions():
    fake = FakeClassifier()
    await _guard(fake, limits=GuardLimits(max_message_chars=1)).screen(_ctx("too long"))
    assert fake.asked == [("language_es", "language_it", "language_fr", "language_de")]
    assert fake.calls == [
        {"previous_assistant_reply": NO_PREVIOUS_REPLY, "visitor_message": "too long"}
    ]


async def test_a_limit_hit_asks_with_the_previous_replys_tail():
    fake = FakeClassifier()
    reply = "a" * 10 + "b" * PREVIOUS_REPLY_CHARS
    guard = _guard(fake, limits=GuardLimits(daily_token_budget=1))
    await guard.screen(_ctx("hola", previous_reply=reply, tokens_today=5))
    assert fake.calls[0]["previous_assistant_reply"] == "b" * PREVIOUS_REPLY_CHARS


async def test_a_huge_message_sends_only_its_first_500_characters():
    fake = FakeClassifier()
    await _guard(fake, limits=GuardLimits(max_message_chars=2000)).screen(_ctx("é" * 100_000))
    assert fake.calls[0]["visitor_message"] == "é" * 500


async def test_a_limit_refusal_comes_in_the_language_found():
    guard = _guard(FakeClassifier({"language_de": 0.9}), limits=GuardLimits(max_message_chars=5))
    decision = await guard.screen(_ctx("x" * 6))
    assert (decision.allowed, decision.stage, decision.reason) == (
        False, "precap", "message_too_long",
    )
    assert decision.refusal == visitor_text("message_too_long", "de", max_chars=5)
    assert decision.language == "de"
    assert (decision.model, decision.request_id, decision.input_tokens) == (
        "fake-decision", "fake-request", 0,
    )
    assert decision.latency_ms is not None
    assert (decision.probabilities, decision.battery_version, decision.error) == (None, None, None)


async def test_a_failed_language_request_refuses_in_english_and_records_why():
    error = GuardClassifierError("down", cause="HTTP503")
    guard = _guard(FakeClassifier(error=error), limits=GuardLimits(max_message_chars=1))
    decision = await guard.screen(_ctx("xx"))
    assert (decision.allowed, decision.stage, decision.reason, decision.error) == (
        False, "precap", "message_too_long", "HTTP503",
    )
    assert (decision.refusal, decision.language) == (
        visitor_text("message_too_long", max_chars=1), "en",
    )


async def test_a_slow_language_request_refuses_in_english_within_the_timeout():
    guard = _guard(_SlowClassifier(), limits=GuardLimits(daily_token_budget=1), timeout=0.01)
    decision = await asyncio.wait_for(guard.screen(_ctx(tokens_today=1)), timeout=0.5)
    assert (decision.refusal, decision.error) == (visitor_text("daily_budget"), "TimeoutError")


async def test_without_a_classifier_a_limit_refusal_is_english_and_asks_nothing():
    decision = await _guard(None, limits=GuardLimits(max_message_chars=1)).screen(_ctx("xx"))
    assert (decision.refusal, decision.model, decision.latency_ms) == (
        visitor_text("message_too_long", max_chars=1), None, None,
    )


async def test_cancellation_during_the_language_request_is_never_swallowed():
    guard = _guard(
        FakeClassifier(error=asyncio.CancelledError()), limits=GuardLimits(max_message_chars=1)
    )
    with pytest.raises(asyncio.CancelledError):
        await guard.screen(_ctx("xx"))


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
        False, "second", visitor_text("off_topic"),
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


# --- the visitor's language ----------------------------------------------------

async def test_the_language_questions_ride_in_the_checks_request():
    fake = FakeClassifier()
    await _guard(fake).screen(_ctx())
    assert fake.asked == [
        ("first", "second", "language_es", "language_it", "language_fr", "language_de")
    ]


async def test_the_decision_carries_the_language_with_its_scores_kept_apart():
    decision = await _guard(FakeClassifier({"first": 0.1, "language_it": 0.95})).screen(_ctx())
    assert decision.language == "it"
    assert decision.probabilities == {"first": 0.1, "second": 0.0}
    assert decision.language_probabilities == {
        "language_es": 0.0, "language_it": 0.95, "language_fr": 0.0, "language_de": 0.0,
    }


async def test_a_block_is_refused_in_the_visitors_language():
    decision = await _guard(FakeClassifier({"second": 0.9, "language_fr": 0.9})).screen(_ctx())
    assert decision.refusal == visitor_text("off_topic", "fr")


def test_a_check_with_an_unknown_refusal_fails_at_construction():
    bad = GuardCheck(id="x", question="q", yes=_C, no=_C, threshold=0.5, refusal="nope")
    with pytest.raises(ValueError, match="'x'"):
        InputGuard(None, GuardLimits(), 1.0, checks=(bad,))


async def test_no_classifier_or_a_failed_one_means_english():
    assert (await _guard().screen(_ctx())).language == "en"
    failing = _guard(FakeClassifier(error=KeyError("bug")))
    assert (await failing.screen(_ctx())).language == "en"


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
