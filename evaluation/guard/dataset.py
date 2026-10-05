"""Labeled calibration cases for the input guard (evaluation/guard/datasets/*.json).

Cases use an {owner} placeholder so the committed dataset never names the site
owner; the runner substitutes it (--owner).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ALLOW = "allow"
BLOCK_PREFIX = "block:"


@dataclass(frozen=True)
class GuardCase:
    id: str
    message: str
    expected: str  # "allow" or "block:<check id>"
    previous_reply: str | None = None

    @property
    def must_allow(self) -> bool:
        return self.expected == ALLOW

    @property
    def expected_check(self) -> str | None:
        if self.expected.startswith(BLOCK_PREFIX):
            return self.expected.removeprefix(BLOCK_PREFIX)
        return None


def load_cases(path: Path, check_ids: set[str], owner: str) -> list[GuardCase]:
    """Load and validate cases, replacing {owner} with the owner's name."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases: list[GuardCase] = []
    seen: set[str] = set()
    for entry in raw["cases"]:
        previous = entry.get("previous_reply")
        case = GuardCase(
            id=entry["id"],
            message=entry["message"].replace("{owner}", owner),
            expected=entry["expected"],
            previous_reply=previous.replace("{owner}", owner) if previous else None,
        )
        if case.id in seen:
            raise ValueError(f"duplicate case id {case.id!r}")
        seen.add(case.id)
        if not case.must_allow and case.expected_check not in check_ids:
            raise ValueError(f"case {case.id!r}: unknown expectation {case.expected!r}")
        cases.append(case)
    return cases
