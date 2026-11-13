"""Labeled cases for the guard's language questions (evaluation/guard/datasets/language_v1.json).

Cases use an {owner} placeholder so the committed dataset never names the site
owner; the runner substitutes it (--owner). Only the "question" and "short"
cases, written in the five languages, count toward the bar. The "none",
"after_reply" and "neighbour" cases are reported, never gated.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast, get_args

from harness.core.language import LANGUAGES, Language

CaseKind = Literal["question", "short", "none", "after_reply", "neighbour"]
_KINDS = frozenset(get_args(CaseKind))
_GATED = frozenset({"question", "short"})

# 39 of 44 passes and 38 fails: py3langid, the local detector Decision-1
# replaced, got 39 of the same 44 messages right.
MIN_RIGHT_SHARE = 0.88


@dataclass(frozen=True)
class LanguageCase:
    id: str
    message: str
    expected: Language
    kind: CaseKind
    previous_reply: str | None = None


def load_language_cases(path: Path, owner: str) -> list[LanguageCase]:
    """Load and validate cases, replacing {owner} with the owner's name."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases: list[LanguageCase] = []
    seen: set[str] = set()
    for entry in raw["cases"]:
        case_id = entry["id"]
        if entry["expected"] not in LANGUAGES:
            raise ValueError(f"case {case_id!r}: unknown language {entry['expected']!r}")
        if entry["kind"] not in _KINDS:
            raise ValueError(f"case {case_id!r}: unknown kind {entry['kind']!r}")
        if case_id in seen:
            raise ValueError(f"duplicate case id {case_id!r}")
        seen.add(case_id)
        previous = entry.get("previous_reply")
        cases.append(LanguageCase(
            id=case_id,
            message=entry["message"].replace("{owner}", owner),
            expected=cast(Language, entry["expected"]),
            kind=cast(CaseKind, entry["kind"]),
            previous_reply=previous.replace("{owner}", owner) if previous else None,
        ))
    return cases


@dataclass(frozen=True)
class LanguageScore:
    right: int  # "question" and "short" cases given the right language
    total: int  # "question" and "short" cases
    wrong: list[str]  # every case given the wrong language, any kind, in dataset order
    wrong_questions: list[str]  # the "question" cases among them

    @property
    def passes_bar(self) -> bool:
        return (
            not self.wrong_questions
            and self.total > 0
            and self.right / self.total >= MIN_RIGHT_SHARE
        )


def score_language(
    cases: Sequence[LanguageCase], picks: Mapping[str, Language]
) -> LanguageScore:
    """picks maps case id -> the language the guard picked; every case must have one."""
    right = total = 0
    wrong: list[str] = []
    wrong_questions: list[str] = []
    for case in cases:
        if case.id not in picks:
            raise ValueError(f"case {case.id!r} has no language; rerun the classifier")
        is_right = picks[case.id] == case.expected
        if not is_right:
            wrong.append(case.id)
            if case.kind == "question":
                wrong_questions.append(case.id)
        if case.kind in _GATED:
            total += 1
            right += int(is_right)
    return LanguageScore(right=right, total=total, wrong=wrong, wrong_questions=wrong_questions)
