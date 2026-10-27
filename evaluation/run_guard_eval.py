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

Exit code 0 when the launch bar passes, 1 when it fails, 2 when the run itself
is unusable (guard disabled, a case not classified, or a stale battery).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
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
from harness.security.battery import BATTERY, BATTERY_VERSION  # noqa: E402
from harness.security.input_guard import GuardContext  # noqa: E402

from evaluation.guard.dataset import GuardCase, load_cases  # noqa: E402
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
    return parser.parse_args()


def _thresholds(overrides: list[str]) -> dict[str, float]:
    thresholds = {check.id: check.threshold for check in BATTERY}
    for item in overrides:
        check_id, _, value = item.partition("=")
        if check_id not in thresholds:
            raise SystemExit(f"unknown check {check_id!r}; known: {sorted(thresholds)}")
        thresholds[check_id] = float(value)
    return thresholds


async def _classify(cases: list[GuardCase]) -> dict:
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
    latencies = [r["latency_ms"] for r in report["results"].values()
                 if r.get("latency_ms") is not None]
    if len(latencies) >= 2:
        p95 = statistics.quantiles(latencies, n=20)[-1]
        print(f"latency p50={statistics.median(latencies):.0f}ms p95={p95:.0f}ms")
    print("LAUNCH BAR:", "PASS" if result.passes_launch_bar else "FAIL")


def main() -> int:
    args = _parse_args()
    cases = load_cases(_DATASETS_DIR / args.dataset, {c.id for c in BATTERY}, args.owner)
    if args.from_report:
        report = json.loads(args.from_report.read_text())
    else:
        report = asyncio.run(_classify(cases))
        _REPORTS_DIR.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = _REPORTS_DIR / f"guard_{stamp}.json"
        out.write_text(json.dumps(report, indent=2))
        print(f"saved raw probabilities to {out}")

    if report["battery_version"] != BATTERY_VERSION:
        print(f"report is from battery {report['battery_version']}, current is "
              f"{BATTERY_VERSION}; rerun the classifier.")
        return 2
    unclassified = [cid for cid, r in report["results"].items() if r["probabilities"] is None]
    if unclassified:
        print(f"{len(unclassified)} case(s) were not classified (e.g. {unclassified[:3]}); "
              f"rerun before scoring.")
        return 2

    thresholds = _thresholds(args.threshold)
    probabilities = {cid: r["probabilities"] for cid, r in report["results"].items()}
    result = score(cases, probabilities, thresholds)
    _print(result, cases, report, thresholds)
    return 0 if result.passes_launch_bar else 1


if __name__ == "__main__":
    sys.exit(main())
