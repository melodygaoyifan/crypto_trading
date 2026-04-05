#!/usr/bin/env python
"""
reflect_weekly.py - Offline weekly attribution analysis.

Reads attribution JSONL logs (signals + outcomes) written by
AttributionTracker and produces a structured report:
  - Per-agent hit rate, rolling confidence, regime breakdown
  - Worst-performing agents flagged
  - Regime-specific alpha leaks
  - Optional: narrative summary via Claude Haiku API

Usage:
    # Last 7 days (default)
    python scripts/reflect_weekly.py

    # Custom date range
    python scripts/reflect_weekly.py --start 20260210 --end 20260217

    # With LLM narrative (requires ANTHROPIC_API_KEY)
    python scripts/reflect_weekly.py --narrative

Output:
    logs/attribution/weekly_report_YYYYMMDD.json
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("reflect_weekly")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LOG_DIR = os.path.join(_PROJECT_ROOT, "logs", "attribution")
HIT_RATE_ALERT_THRESHOLD = 0.45
MIN_SIGNALS_FOR_ALERT = 10


# ---------------------------------------------------------------------------
# JSONL Reader
# ---------------------------------------------------------------------------

def _read_jsonl_files(
    log_dir: str,
    prefix: str,
    start_date: str,
    end_date: str,
) -> List[Dict[str, Any]]:
    """Read all JSONL files matching prefix between date range (YYYYMMDD)."""
    records: List[Dict[str, Any]] = []
    log_path = Path(log_dir)
    if not log_path.exists():
        logger.warning(f"Log directory not found: {log_dir}")
        return records

    start_dt = datetime.strptime(start_date, "%Y%m%d")
    end_dt = datetime.strptime(end_date, "%Y%m%d")

    for fpath in sorted(log_path.glob(f"{prefix}_*.jsonl")):
        # Extract date from filename: prefix_YYYYMMDD.jsonl
        date_part = fpath.stem.replace(f"{prefix}_", "")
        try:
            file_dt = datetime.strptime(date_part, "%Y%m%d")
        except ValueError:
            continue
        if file_dt < start_dt or file_dt > end_dt:
            continue

        try:
            with open(fpath, "r", encoding="utf-8") as fh:
                for line_no, line in enumerate(fh, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning(f"Bad JSON at {fpath.name}:{line_no}")
        except OSError as exc:
            logger.warning(f"Cannot read {fpath}: {exc}")

    return records


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_outcomes(outcomes: List[Dict]) -> Dict[str, Any]:
    """Aggregate outcome records into per-agent scorecards."""
    agent_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "total": 0,
        "correct": 0,
        "conf_when_correct": [],
        "conf_when_wrong": [],
        "by_regime": defaultdict(lambda: {"total": 0, "correct": 0}),
        "by_asset": defaultdict(lambda: {"total": 0, "correct": 0}),
    })

    total_outcomes = 0

    for entry in outcomes:
        asset = entry.get("asset", "UNK")
        regime = entry.get("regime", "unknown")
        for outcome in entry.get("outcomes", []):
            agent = outcome.get("agent_name", "unknown")
            correct = outcome.get("correct", False)
            conf = outcome.get("confidence", 0.0)

            stats = agent_stats[agent]
            stats["total"] += 1
            total_outcomes += 1

            if correct:
                stats["correct"] += 1
                stats["conf_when_correct"].append(conf)
            else:
                stats["conf_when_wrong"].append(conf)

            stats["by_regime"][regime]["total"] += 1
            if correct:
                stats["by_regime"][regime]["correct"] += 1

            stats["by_asset"][asset]["total"] += 1
            if correct:
                stats["by_asset"][asset]["correct"] += 1

    # Build scorecards
    scorecards = {}
    for agent, stats in agent_stats.items():
        hit_rate = stats["correct"] / stats["total"] if stats["total"] else 0.0
        avg_conf_correct = (
            sum(stats["conf_when_correct"]) / len(stats["conf_when_correct"])
            if stats["conf_when_correct"] else 0.0
        )
        avg_conf_wrong = (
            sum(stats["conf_when_wrong"]) / len(stats["conf_when_wrong"])
            if stats["conf_when_wrong"] else 0.0
        )

        regime_breakdown = {}
        for r, rd in stats["by_regime"].items():
            regime_breakdown[r] = {
                "total": rd["total"],
                "correct": rd["correct"],
                "hit_rate": round(rd["correct"] / rd["total"], 4) if rd["total"] else 0.0,
            }

        asset_breakdown = {}
        for a, ad in stats["by_asset"].items():
            asset_breakdown[a] = {
                "total": ad["total"],
                "correct": ad["correct"],
                "hit_rate": round(ad["correct"] / ad["total"], 4) if ad["total"] else 0.0,
            }

        scorecards[agent] = {
            "total_signals": stats["total"],
            "correct_signals": stats["correct"],
            "hit_rate": round(hit_rate, 4),
            "avg_confidence_when_correct": round(avg_conf_correct, 4),
            "avg_confidence_when_wrong": round(avg_conf_wrong, 4),
            "calibration_gap": round(avg_conf_correct - avg_conf_wrong, 4),
            "by_regime": dict(regime_breakdown),
            "by_asset": dict(asset_breakdown),
        }

    return {
        "total_outcome_records": total_outcomes,
        "agents_evaluated": len(scorecards),
        "scorecards": scorecards,
    }


def find_alerts(scorecards: Dict[str, Dict]) -> Dict[str, List]:
    """Identify underperformers and regime-specific alpha leaks."""
    decay_alerts = []
    regime_leaks = []
    calibration_alerts = []

    for agent, sc in scorecards.items():
        # Global underperformer
        if sc["total_signals"] >= MIN_SIGNALS_FOR_ALERT:
            if sc["hit_rate"] < HIT_RATE_ALERT_THRESHOLD:
                decay_alerts.append({
                    "agent": agent,
                    "hit_rate": sc["hit_rate"],
                    "total": sc["total_signals"],
                })

        # Per-regime alpha leaks (agent < 40% in a specific regime with 5+ signals)
        for regime, rd in sc.get("by_regime", {}).items():
            if rd["total"] >= 5 and rd["hit_rate"] < 0.40:
                regime_leaks.append({
                    "agent": agent,
                    "regime": regime,
                    "hit_rate": rd["hit_rate"],
                    "total": rd["total"],
                })

        # Calibration alert: high confidence when wrong (overconfident)
        if sc["avg_confidence_when_wrong"] > 0.7 and sc["total_signals"] >= MIN_SIGNALS_FOR_ALERT:
            calibration_alerts.append({
                "agent": agent,
                "avg_conf_when_wrong": sc["avg_confidence_when_wrong"],
                "avg_conf_when_correct": sc["avg_confidence_when_correct"],
            })

    return {
        "decay_alerts": decay_alerts,
        "regime_leaks": regime_leaks,
        "calibration_alerts": calibration_alerts,
    }


def signal_volume_summary(signals: List[Dict]) -> Dict[str, Any]:
    """Summarize signal volume: total ticks, per-asset, per-agent counts."""
    tick_count = len(signals)
    by_asset: Dict[str, int] = defaultdict(int)
    by_agent: Dict[str, int] = defaultdict(int)

    for entry in signals:
        asset = entry.get("asset", "UNK")
        by_asset[asset] += 1
        for sig in entry.get("signals", []):
            by_agent[sig.get("agent_name", "unknown")] += 1

    return {
        "total_ticks": tick_count,
        "by_asset": dict(by_asset),
        "by_agent": dict(by_agent),
    }


# ---------------------------------------------------------------------------
# LLM Narrative (optional)
# ---------------------------------------------------------------------------

def generate_narrative(report: Dict[str, Any]) -> Optional[str]:
    """Use Claude Haiku to generate a plain-English narrative summary."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.info("ANTHROPIC_API_KEY not set - skipping narrative generation")
        return None

    try:
        import anthropic
    except ImportError:
        logger.warning("anthropic package not installed - skipping narrative")
        return None

    prompt = (
        "You are a quant trading system analyst. Below is a weekly attribution "
        "report for a multi-agent crypto trading system (HMATS). "
        "Write a concise 3-paragraph summary:\n"
        "1. Overall performance: which agents are performing well/poorly\n"
        "2. Regime-specific findings: any alpha leaks in specific regimes\n"
        "3. Actionable recommendations: what to adjust next week\n\n"
        "Keep it under 200 words. Be specific with numbers.\n\n"
        f"Report data:\n```json\n{json.dumps(report, indent=2)}\n```"
    )

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    except Exception as exc:
        logger.warning(f"Haiku narrative generation failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_report(
    start_date: str,
    end_date: str,
    log_dir: str = LOG_DIR,
    with_narrative: bool = False,
) -> Dict[str, Any]:
    """Build the full weekly report."""
    logger.info(f"Building report for {start_date} - {end_date}")

    signals = _read_jsonl_files(log_dir, "signals", start_date, end_date)
    outcomes = _read_jsonl_files(log_dir, "outcomes", start_date, end_date)

    logger.info(f"Read {len(signals)} signal entries, {len(outcomes)} outcome entries")

    if not outcomes:
        logger.warning("No outcome data found - report will be empty")

    analysis = analyze_outcomes(outcomes)
    alerts = find_alerts(analysis.get("scorecards", {}))
    volume = signal_volume_summary(signals)

    report = {
        "period": {"start": start_date, "end": end_date},
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "signal_volume": volume,
        "analysis": analysis,
        "alerts": alerts,
    }

    if with_narrative:
        narrative = generate_narrative(report)
        if narrative:
            report["narrative"] = narrative

    return report


def main():
    parser = argparse.ArgumentParser(description="HMATS Weekly Attribution Report")
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date YYYYMMDD (default: 7 days ago)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date YYYYMMDD (default: today)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=LOG_DIR,
        help="Attribution log directory",
    )
    parser.add_argument(
        "--narrative",
        action="store_true",
        help="Generate LLM narrative summary (requires ANTHROPIC_API_KEY)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: logs/attribution/weekly_report_YYYYMMDD.json)",
    )

    args = parser.parse_args()

    # Default date range: last 7 days
    end_dt = datetime.now(timezone.utc) if args.end is None else datetime.strptime(args.end, "%Y%m%d")
    start_dt = (end_dt - timedelta(days=7)) if args.start is None else datetime.strptime(args.start, "%Y%m%d")

    start_str = start_dt.strftime("%Y%m%d")
    end_str = end_dt.strftime("%Y%m%d")

    report = build_report(
        start_date=start_str,
        end_date=end_str,
        log_dir=args.log_dir,
        with_narrative=args.narrative,
    )

    # Write output
    out_path = args.output
    if out_path is None:
        os.makedirs(args.log_dir, exist_ok=True)
        out_path = os.path.join(args.log_dir, f"weekly_report_{end_str}.json")

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    logger.info(f"Report written to {out_path}")

    # Print summary to stdout
    analysis = report.get("analysis", {})
    alerts = report.get("alerts", {})
    volume = report.get("signal_volume", {})

    print(f"\n{'='*60}")
    print(f"HMATS Weekly Attribution Report: {start_str} - {end_str}")
    print(f"{'='*60}")
    print(f"Signal ticks: {volume.get('total_ticks', 0)}")
    print(f"Agents evaluated: {analysis.get('agents_evaluated', 0)}")
    print(f"Total outcomes: {analysis.get('total_outcome_records', 0)}")

    scorecards = analysis.get("scorecards", {})
    if scorecards:
        print(f"\n{'Agent':<25} {'Hit%':>6} {'N':>5} {'ConfOK':>7} {'ConfBad':>7}")
        print("-" * 55)
        for agent, sc in sorted(scorecards.items(), key=lambda x: x[1]["hit_rate"], reverse=True):
            print(
                f"{agent:<25} {sc['hit_rate']*100:>5.1f}% "
                f"{sc['total_signals']:>5} "
                f"{sc['avg_confidence_when_correct']:>6.3f} "
                f"{sc['avg_confidence_when_wrong']:>6.3f}"
            )

    decay = alerts.get("decay_alerts", [])
    if decay:
        print(f"\nDECAY ALERTS ({len(decay)}):")
        for a in decay:
            print(f"  {a['agent']}: {a['hit_rate']*100:.1f}% hit rate ({a['total']} signals)")

    leaks = alerts.get("regime_leaks", [])
    if leaks:
        print(f"\nREGIME LEAKS ({len(leaks)}):")
        for a in leaks:
            print(f"  {a['agent']} in {a['regime']}: {a['hit_rate']*100:.1f}% ({a['total']} signals)")

    cal = alerts.get("calibration_alerts", [])
    if cal:
        print(f"\nCALIBRATION ALERTS ({len(cal)}):")
        for a in cal:
            print(f"  {a['agent']}: conf_wrong={a['avg_conf_when_wrong']:.3f}, conf_ok={a['avg_conf_when_correct']:.3f}")

    if report.get("narrative"):
        print(f"\nNARRATIVE:\n{report['narrative']}")

    print(f"\nFull report: {out_path}")


if __name__ == "__main__":
    main()
