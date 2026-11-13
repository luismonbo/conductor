"""The languages a visitor's refusals and errors come in.

The website agent answers in these five and in English otherwise (its prompt says
so), and the guard picks one of them for every sentence the API gives a visitor
instead of an answer. See docs/superpowers/specs/2026-11-13-refusal-messages-design.md.
"""
from __future__ import annotations

from typing import Literal

Language = Literal["en", "es", "it", "fr", "de"]

LANGUAGES: tuple[Language, ...] = ("en", "es", "it", "fr", "de")
DEFAULT_LANGUAGE: Language = "en"
