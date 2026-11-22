"""The guard's language questions and how a language is picked from their scores."""
from __future__ import annotations

from harness.security.battery import BATTERY, LANGUAGE_QUESTIONS
from harness.security.input_guard import pick_language


def test_four_questions_ask_about_the_visitor_message_and_the_previous_reply():
    assert list(LANGUAGE_QUESTIONS) == ["es", "it", "fr", "de"]
    ids = [q.id for q in LANGUAGE_QUESTIONS.values()]
    assert ids == ["language_es", "language_it", "language_fr", "language_de"]
    assert not set(ids) & {check.id for check in BATTERY}
    for q in LANGUAGE_QUESTIONS.values():
        assert "`visitor_message`" in q.question
        assert "`previous_assistant_reply`" in q.yes.definition


def test_the_highest_language_at_or_above_the_threshold_wins():
    assert pick_language({"language_es": 0.9, "language_it": 0.6}, threshold=0.5) == "es"
    assert pick_language({"language_fr": 0.5}, threshold=0.5) == "fr"
    assert pick_language({"language_de": 0.49}, threshold=0.5) == "en"
    assert pick_language({}, threshold=0.5) == "en"


def test_a_tie_goes_to_the_first_language_in_order():
    assert pick_language({"language_it": 0.8, "language_es": 0.8}, threshold=0.5) == "es"


def test_the_default_threshold_is_the_calibrated_one():
    # Calibrated on Decision-1: right picks scored 0.91+, wrong languages 0.44 or less.
    assert pick_language({"language_es": 0.7}) == "es"
    assert pick_language({"language_es": 0.69}) == "en"
