"""Language calibration tooling: dataset validation and the language bar, from saved scores."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

import evaluation.run_guard_eval as run_guard_eval
from evaluation.guard.dataset import GuardCase
from evaluation.guard.language import LanguageCase, load_language_cases, score_language
from harness.adapters.guard.fake import FakeClassifier
from harness.config.settings import Settings
from harness.security.battery import BATTERY_VERSION
from harness.security.input_guard import GuardLimits, InputGuard

_DATASET = (
    Path(__file__).resolve().parents[2] / "evaluation" / "guard" / "datasets" / "language_v1.json"
)


def _write(tmp_path: Path, cases: list[dict]) -> Path:
    path = tmp_path / "cases.json"
    path.write_text(json.dumps({"version": "t", "kind": "language", "cases": cases}))
    return path


def test_load_substitutes_the_owner_in_messages_and_replies(tmp_path):
    path = _write(tmp_path, [
        {"id": "a", "message": "¿Dónde vive {owner}?", "expected": "es", "kind": "question"},
        {"id": "b", "message": "ok", "expected": "it", "kind": "after_reply",
         "previous_reply": "{owner} lavora con agenti di IA."},
    ])
    cases = load_language_cases(path, owner="Sam")
    assert cases[0] == LanguageCase("a", "¿Dónde vive Sam?", "es", "question")
    assert cases[1].previous_reply == "Sam lavora con agenti di IA."


@pytest.mark.parametrize("bad, match", [
    ({"id": "a", "message": "x", "expected": "xx", "kind": "question"}, "unknown language"),
    ({"id": "a", "message": "x", "expected": "es", "kind": "other"}, "unknown kind"),
])
def test_load_rejects_an_unknown_language_or_kind(tmp_path, bad, match):
    with pytest.raises(ValueError, match=match):
        load_language_cases(_write(tmp_path, [bad]), owner="Sam")


def test_load_rejects_a_duplicate_id(tmp_path):
    case = {"id": "a", "message": "x", "expected": "es", "kind": "question"}
    with pytest.raises(ValueError, match="duplicate"):
        load_language_cases(_write(tmp_path, [case, case]), owner="Sam")


def test_only_the_five_language_cases_count_toward_the_bar():
    cases = [
        LanguageCase("q1", "¿Dónde?", "es", "question"),
        LanguageCase("s1", "hola", "es", "short"),
        LanguageCase("n1", "Onde?", "en", "neighbour"),
    ]
    result = score_language(cases, {"q1": "es", "s1": "en", "n1": "es"})
    assert (result.right, result.total) == (1, 2)
    assert (result.wrong, result.wrong_questions) == (["s1", "n1"], [])


def test_a_wrong_question_fails_the_bar_whatever_the_share():
    cases = [LanguageCase(f"q{i}", "x", "es", "question") for i in range(50)]
    picks = {f"q{i}": "es" for i in range(50)} | {"q0": "it"}
    result = score_language(cases, picks)
    assert (result.right, result.total, result.wrong_questions) == (49, 50, ["q0"])
    assert result.passes_bar is False


def test_the_bar_takes_39_of_44_and_refuses_38():
    cases = [LanguageCase(f"s{i}", "hola", "es", "short") for i in range(44)]

    def picks(wrong: int) -> dict[str, str]:
        return {f"s{i}": ("en" if i < wrong else "es") for i in range(44)}

    assert score_language(cases, picks(5)).passes_bar is True
    assert score_language(cases, picks(6)).passes_bar is False


def test_score_refuses_a_case_without_a_language():
    with pytest.raises(ValueError, match="no language"):
        score_language([LanguageCase("q1", "x", "es", "question")], {})


def test_the_committed_dataset_has_every_kind_and_language():
    cases = load_language_cases(_DATASET, owner="Sam")
    assert Counter(case.kind for case in cases) == {
        "question": 32, "short": 12, "none": 5, "after_reply": 5, "neighbour": 14,
    }
    gated = [case for case in cases if case.kind in ("question", "short")]
    assert Counter(case.expected for case in gated) == {"en": 10, "es": 10, "it": 8, "fr": 8, "de": 8}
    assert all((case.previous_reply is not None) == (case.kind == "after_reply") for case in cases)


async def test_a_live_report_records_the_language(monkeypatch):
    settings = Settings(
        _env_file=None, api_key="k", input_guard_enabled=True,
        azure_endpoint="https://foundry.example", agent="default",
    )
    monkeypatch.setattr(run_guard_eval, "get_settings", lambda: settings)
    monkeypatch.setattr(
        run_guard_eval,
        "build_input_guard",
        lambda s: InputGuard(
            FakeClassifier({"language_fr": 0.9}), GuardLimits(), timeout_seconds=1.0
        ),
    )

    report = await run_guard_eval._classify(
        [GuardCase(id="hi", message="Salut", expected="allow")]
    )

    assert report["results"]["hi"]["language"] == "fr"
    assert set(report["results"]["hi"]["language_probabilities"]) == {
        "language_es", "language_it", "language_fr", "language_de",
    }


def test_a_saved_language_report_is_rescored_with_another_threshold(tmp_path, monkeypatch):
    cases = load_language_cases(_DATASET, owner="the site owner")
    results = {
        case.id: {
            "stage": "classifier", "probabilities": {}, "latency_ms": 100, "model": "m",
            "error": None, "language": case.expected,
            "language_probabilities": (
                {} if case.expected == "en" else {f"language_{case.expected}": 0.8}
            ),
        }
        for case in cases
    }
    report = tmp_path / "language.json"
    report.write_text(json.dumps(
        {"battery_version": BATTERY_VERSION, "agent": "website", "results": results}
    ))

    def run(*extra: str) -> int:
        monkeypatch.setattr(sys, "argv", [
            "run_guard_eval.py", "--dataset", "language_v1.json", "--from-report", str(report),
            *extra,
        ])
        return run_guard_eval.main()

    assert run() == 0
    assert run("--language-threshold", "0.9") == 1
