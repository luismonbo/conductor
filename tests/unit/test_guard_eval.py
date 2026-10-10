"""Calibration tooling: dataset validation and launch-bar scoring from saved probabilities."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import evaluation.run_guard_eval as run_guard_eval
from evaluation.guard.dataset import GuardCase, load_cases
from evaluation.guard.scoring import score
from harness.adapters.guard.fake import FakeClassifier
from harness.agents.default.prompt import SYSTEM_PROMPT
from harness.config.settings import Settings
from harness.security.battery import BATTERY
from harness.security.input_guard import GuardLimits, InputGuard

THRESHOLDS = {"extraction": 0.7, "private_info": 0.7, "off_topic": 0.8}
IDS = set(THRESHOLDS)
_DATASET = Path(__file__).resolve().parents[2] / "evaluation" / "guard" / "datasets" / "guard_v1.json"


def _write(tmp_path: Path, cases: list[dict]) -> Path:
    path = tmp_path / "cases.json"
    path.write_text(json.dumps({"version": "t", "cases": cases}))
    return path


def test_load_substitutes_the_owner_and_validates(tmp_path):
    path = _write(tmp_path, [
        {"id": "a", "message": "What has {owner} built?", "expected": "allow"},
        {"id": "b", "message": "x", "previous_reply": "{owner} built Y.",
         "expected": "block:off_topic"},
    ])
    cases = load_cases(path, IDS, owner="Sam")
    assert cases[0].message == "What has Sam built?"
    assert cases[0].must_allow is True
    assert cases[1].previous_reply == "Sam built Y."
    assert cases[1].expected_check == "off_topic"


def test_load_rejects_unknown_checks_and_duplicate_ids(tmp_path):
    unknown = _write(tmp_path, [{"id": "a", "message": "x", "expected": "block:nope"}])
    with pytest.raises(ValueError, match="unknown expectation"):
        load_cases(unknown, IDS, owner="Sam")
    dupes = _write(tmp_path, [{"id": "a", "message": "x", "expected": "allow"}] * 2)
    with pytest.raises(ValueError, match="duplicate"):
        load_cases(dupes, IDS, owner="Sam")


def test_score_flags_blocked_must_allow_cases_and_counts_catches():
    cases = [
        GuardCase(id="hi", message="Hi", expected="allow"),
        GuardCase(id="leak", message="...", expected="block:extraction"),
        GuardCase(id="essay", message="...", expected="block:off_topic"),
        GuardCase(id="essay2", message="...", expected="block:off_topic"),
    ]
    probabilities = {
        "hi": {"extraction": 0.0, "private_info": 0.0, "off_topic": 0.85},
        "leak": {"extraction": 0.2, "private_info": 0.0, "off_topic": 0.9},
        "essay": {"extraction": 0.0, "private_info": 0.0, "off_topic": 0.95},
        "essay2": {"extraction": 0.0, "private_info": 0.0, "off_topic": 0.10},
    }
    result = score(cases, probabilities, THRESHOLDS)
    assert result.blocked_must_allow == ["hi"]
    assert result.missed == ["essay2"]
    by_id = {check.check_id: check for check in result.checks}
    assert (by_id["extraction"].caught, by_id["extraction"].fired_correctly) == (1, 0)
    assert (by_id["off_topic"].caught, by_id["off_topic"].total) == (1, 2)
    assert by_id["off_topic"].false_fires == 1
    assert result.passes_launch_bar is False


def test_score_passes_the_bar_when_clean():
    cases = [
        GuardCase(id="hi", message="Hi", expected="allow"),
        GuardCase(id="leak", message="...", expected="block:extraction"),
    ]
    probabilities = {
        "hi": {"extraction": 0.1, "private_info": 0.1, "off_topic": 0.1},
        "leak": {"extraction": 0.95, "private_info": 0.0, "off_topic": 0.0},
    }
    assert score(cases, probabilities, THRESHOLDS).passes_launch_bar is True


def test_score_refuses_cases_without_probabilities():
    with pytest.raises(ValueError, match="no probabilities"):
        score([GuardCase(id="hi", message="Hi", expected="allow")], {}, THRESHOLDS)


def test_the_committed_dataset_loads_against_the_v1_battery():
    cases = load_cases(_DATASET, {check.id for check in BATTERY}, owner="Sam")
    assert len(cases) >= 45
    assert sum(1 for case in cases if case.must_allow) >= 15
    for check in BATTERY:
        assert sum(1 for case in cases if case.expected_check == check.id) == 10


def test_the_dataset_never_reuses_the_batterys_own_examples():
    battery_examples = {
        example for check in BATTERY
        for example in check.yes.examples + check.no.examples
    }
    cases = load_cases(_DATASET, {check.id for check in BATTERY}, owner="Sam")
    assert not battery_examples & {case.message for case in cases}


async def test_a_live_report_names_the_agent_whose_prompt_it_judged(monkeypatch):
    settings = Settings(
        _env_file=None, api_key="k", input_guard_enabled=True,
        typesafe_api_key="ts", agent="default",
    )
    monkeypatch.setattr(run_guard_eval, "get_settings", lambda: settings)
    monkeypatch.setattr(
        run_guard_eval,
        "build_input_guard",
        lambda s: InputGuard(FakeClassifier(), GuardLimits(), timeout_seconds=1.0),
    )

    report = await run_guard_eval._classify(
        [GuardCase(id="hi", message="Hi", expected="allow")]
    )

    assert report["agent"] == "default"
    assert report["system_prompt"] == SYSTEM_PROMPT
