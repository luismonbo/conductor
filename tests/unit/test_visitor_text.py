"""The sentences a visitor gets instead of an answer, in the five languages."""
from __future__ import annotations

import pytest

from harness.core.language import LANGUAGES
from harness.security.visitor_text import (
    TEXT_KEYS,
    VisitorLanguage,
    configured_language,
    visitor_text,
)


def test_every_text_exists_in_the_five_languages():
    for key in TEXT_KEYS:
        for language in LANGUAGES:
            assert visitor_text(key, language, max_chars=1).strip(), (key, language)


def test_the_english_kept_from_before_is_unchanged():
    assert visitor_text("off_topic") == "That's outside what I can help with here."
    assert visitor_text("daily_budget") == (
        "I've reached my limit for today. Please come back tomorrow."
    )
    assert visitor_text("message_too_long", max_chars=2000) == (
        "That message is a bit long for me. Could you keep it under 2000 characters?"
    )
    assert visitor_text("content_filter") == (
        "I can't respond to that message. Your next message will start a new conversation."
    )
    assert visitor_text("unavailable") == (
        "The assistant is unavailable right now. Please try again later."
    )
    assert visitor_text("step_limit") == (
        "I couldn't complete this within the allowed reasoning steps."
    )


def test_the_not_shareable_sentence_is_the_owners_wording():
    assert visitor_text("not_shareable") == (
        "I can't share that kind of information. If you really want to know, ask Luis."
    )


def test_a_text_comes_in_the_language_asked_for():
    assert visitor_text("not_shareable", "es") == (
        "No puedo compartir ese tipo de información. Si de verdad quieres saberlo, "
        "pregúntale a Luis."
    )
    assert visitor_text("off_topic", "de") == "Dabei kann ich dir hier nicht helfen."


def test_only_the_length_sentence_needs_a_field():
    for language in LANGUAGES:
        with pytest.raises(KeyError):
            visitor_text("message_too_long", language)
        assert "2000" in visitor_text("message_too_long", language, max_chars=2000)


def test_french_puts_a_no_break_space_before_every_question_mark():
    for key in TEXT_KEYS:
        text = visitor_text(key, "fr", max_chars=1)
        assert all(text[i - 1] == " " for i, ch in enumerate(text) if ch == "?"), key
    assert "?" in visitor_text("message_too_long", "fr", max_chars=1)


def test_an_unknown_key_is_a_key_error():
    with pytest.raises(KeyError):
        visitor_text("thread_too_long")  # type: ignore[arg-type]


def test_the_runs_language_comes_from_its_wrapper_and_defaults_to_english():
    assert configured_language({}) == "en"
    assert configured_language({"visitor_language": VisitorLanguage("fr")}) == "fr"
    # A bare string would have been copied into the checkpoint, so it is not read.
    assert configured_language({"visitor_language": "fr"}) == "en"
