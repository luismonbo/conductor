"""Input-guard calibration: run the real classifier over labeled cases.

Needs the Foundry resource's Decision-1 deployment (HARNESS_AZURE_ENDPOINT, with
HARNESS_AZURE_API_KEY, or `az login` for the managed-identity path) and the agent
whose prompt the guard judges against, the entry agent HARNESS_AGENT names:

    HARNESS_INPUT_GUARD_ENABLED=true HARNESS_AGENT=<website-agent> \
        uv run python evaluation/run_guard_eval.py --owner "<owner name>"

The HARNESS_INPUT_GUARD_ENABLED=true prefix is needed when .env turns the guard
off (a real environment variable beats .env). Re-run whenever that agent's
prompt.py changes.

Every live run saves raw probabilities to evaluation/reports/ (gitignored), so
a threshold sweep re-scores a saved run without calling the classifier again:

    uv run python evaluation/run_guard_eval.py \
        --from-report evaluation/reports/guard_<stamp>.json --threshold off_topic=0.85

The language dataset (--dataset language_v1.json) scores the language the guard
picks instead, against the language bar; --language-threshold re-scores a saved
language report at another threshold.

Exit code 0 when the bar passes, 1 when it fails, 2 when the run itself is
unusable (guard disabled, a case not classified, or a stale battery).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

# Python adds the script directory (evaluation/) to sys.path[0] automatically,
# which makes evaluation/harness/ shadow src/harness/. Force src to front, and
# ensure the project root is present so evaluation.* imports resolve.
_root = Path(__file__).parent.parent
sys.path.insert(0, str(_root / "src"))
if str(_root) not in sys.path:
    sys.path.append(str(_root))

from harness.config.settings import get_settings  # noqa: E402
from harness.orchestration.build import build_input_guard, resolve_system_prompt  # noqa: E402
from harness.security.battery import BATTERY, BATTERY_VERSION, LANGUAGE_THRESHOLD  # noqa: E402
from harness.security.input_guard import GuardContext, pick_language  # noqa: E402

from evaluation.guard.dataset import GuardCase, load_cases  # noqa: E402
from evaluation.guard.language import (  # noqa: E402
    LanguageCase,
    LanguageScore,
    load_language_cases,
    score_language,
)
from evaluation.guard.scoring import GuardScore, score  # noqa: E402

_EVAL_DIR = Path(__file__).parent
_DATASETS_DIR = _EVAL_DIR / "guard" / "datasets"
_REPORTS_DIR = _EVAL_DIR / "reports"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate the input guard")
    parser.add_argument("--dataset", default="guard_v1.json")
    parser.add_argument("--owner", default="the site owner", help="Replaces {owner}")
    parser.add_argument("--from-report", type=Path, default=None,
                        help="Re-score a saved run instead of calling the classifier")
    parser.add_argument("--threshold", action="append", default=[], metavar="CHECK=VALUE",
                        help="Override one check's threshold for scoring (repeatable)")
    parser.add_argument("--language-threshold", type=float, default=None, metavar="VALUE",
                        help="Override the language threshold for scoring a language dataset")
    return parser.parse_args()


def _thresholds(overrides: list[str]) -> dict[str, float]:
    thresholds = {check.id: check.threshold for check in BATTERY}
    for item in overrides:
        check_id, _, value = item.partition("=")
        if check_id not in thresholds:
            raise SystemExit(f"unknown check {check_id!r}; known: {sorted(thresholds)}")
        thresholds[check_id] = float(value)
    return thresholds


async def _classify(cases: Sequence[GuardCase | LanguageCase]) -> dict:
    settings = get_settings()
    if not settings.input_guard_enabled:
        print("HARNESS_INPUT_GUARD_ENABLED is false; calibration needs the real classifier.")
        raise SystemExit(2)
    guard = build_input_guard(settings)
    prompt = resolve_system_prompt(settings)
    results: dict[str, dict] = {}
    try:
        for case in cases:  # sequential, so latency is realistic and rate limits are safe
            decision = await guard.screen(GuardContext(
                message=case.message, system_prompt=prompt,
                previous_reply=case.previous_reply,
            ))
            results[case.id] = {
                "stage": decision.stage,
                "probabilities": decision.probabilities,
                "latency_ms": decision.latency_ms,
                "model": decision.model,
                "error": decision.error,
                "language": decision.language,
                "language_probabilities": decision.language_probabilities,
            }
    finally:
        await guard.aclose()
    return {
        "battery_version": BATTERY_VERSION,
        "agent": settings.agent,
        "system_prompt": prompt,
        "results": results,
    }


def _print(result: GuardScore, cases: list[GuardCase], report: dict,
           thresholds: dict[str, float]) -> None:
    probabilities = {cid: r["probabilities"] for cid, r in report["results"].items()}
    print(f"\nagent: {report.get('agent', 'not recorded')}")
    print(f"thresholds: {thresholds}")
    print(f"{'check':<14}{'caught':>8}{'fired':>8}{'false fires':>13}{'catch rate':>12}")
    for check in result.checks:
        caught = f"{check.caught}/{check.total}"
        print(f"{check.check_id:<14}{caught:>8}{check.fired_correctly:>8}"
              f"{check.false_fires:>13}{check.catch_rate:>12.0%}")
    by_id = {case.id: case for case in cases}
    for label, ids in (("BLOCKED must-allow", result.blocked_must_allow),
                       ("MISSED hazard", result.missed)):
        for cid in ids:
            probs = ", ".join(f"{k}={v:.2f}" for k, v in probabilities[cid].items())
            print(f"{label}: {cid} [{probs}] {by_id[cid].message[:70]!r}")
    _print_latency(report)
    print("LAUNCH BAR:", "PASS" if result.passes_launch_bar else "FAIL")


def _print_latency(report: dict) -> None:
    latencies = [r["latency_ms"] for r in report["results"].values()
                 if r.get("latency_ms") is not None]
    if len(latencies) >= 2:
        p95 = statistics.quantiles(latencies, n=20)[-1]
        print(f"latency p50={statistics.median(latencies):.0f}ms p95={p95:.0f}ms")


def _score_language(cases: list[LanguageCase], report: dict, threshold: float) -> int:
    probabilities = {
        cid: r.get("language_probabilities") or {} for cid, r in report["results"].items()
    }
    picks = {cid: pick_language(probs, threshold=threshold) for cid, probs in probabilities.items()}
    result = score_language(cases, picks)
    _print_language(result, cases, picks, probabilities, report, threshold)
    return 0 if result.passes_bar else 1


def _print_language(result: LanguageScore, cases: list[LanguageCase], picks: dict,
                    probabilities: dict, report: dict, threshold: float) -> None:
    print(f"\nagent: {report.get('agent', 'not recorded')}")
    print(f"language threshold: {threshold}")
    print(f"five-language cases right: {result.right}/{result.total}")
    wrong = set(result.wrong)
    for case in cases:
        if case.id in wrong or case.kind in ("after_reply", "neighbour"):
            mark = "wrong" if case.id in wrong else "right"
            probs = ", ".join(f"{k}={v:.2f}" for k, v in probabilities[case.id].items())
            print(f"{mark:<6}{case.kind:<12}{case.id:<20}expected {case.expected} got "
                  f"{picks[case.id]} [{probs}] {case.message[:40]!r}")
    _print_latency(report)
    print("LANGUAGE BAR:", "PASS" if result.passes_bar else "FAIL")


def _report(
    args: argparse.Namespace, cases: Sequence[GuardCase | LanguageCase], prefix: str
) -> dict:
    """The saved report --from-report names, or a live run, saved for later re-scoring."""
    if args.from_report:
        return json.loads(args.from_report.read_text())
    report = asyncio.run(_classify(cases))
    _REPORTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = _REPORTS_DIR / f"{prefix}_{stamp}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"saved raw probabilities to {out}")
    return report


def _usable(report: dict) -> bool:
    if report["battery_version"] != BATTERY_VERSION:
        print(f"report is from battery {report['battery_version']}, current is "
              f"{BATTERY_VERSION}; rerun the classifier.")
        return False
    unclassified = [cid for cid, r in report["results"].items() if r["probabilities"] is None]
    if unclassified:
        print(f"{len(unclassified)} case(s) were not classified (e.g. {unclassified[:3]}); "
              f"rerun before scoring.")
        return False
    return True


def main() -> int:
    args = _parse_args()
    dataset = _DATASETS_DIR / args.dataset
    if json.loads(dataset.read_text(encoding="utf-8")).get("kind") == "language":
        language_cases = load_language_cases(dataset, args.owner)
        report = _report(args, language_cases, "language")
        if not _usable(report):
            return 2
        threshold = (
            LANGUAGE_THRESHOLD if args.language_threshold is None else args.language_threshold
        )
        return _score_language(language_cases, report, threshold)

    cases = load_cases(dataset, {c.id for c in BATTERY}, args.owner)
    report = _report(args, cases, "guard")
    if not _usable(report):
        return 2
    thresholds = _thresholds(args.threshold)
    probabilities = {cid: r["probabilities"] for cid, r in report["results"].items()}
    result = score(cases, probabilities, thresholds)
    _print(result, cases, report, thresholds)
    return 0 if result.passes_launch_bar else 1


if __name__ == "__main__":
    sys.exit(main())
