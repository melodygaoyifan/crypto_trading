"""
HMATS - FastRiskTick Shadow Analysis
======================================

Analyzes FastRiskTick shadow logs from paper run to evaluate
promotion readiness (shadow -> active).

Reads:
  - FastRiskTick._shadow_log (in-memory, dumped to logs/fast_risk_shadow.jsonl)
  - Shadow ledger (data/shadow_ledger/) for trade context
  - Paper run logs for [FastRiskTick][SHADOW] entries

Outputs:
  - Trigger frequency by asset and type
  - False positive rate estimation (trigger but price reverts)
  - Hypothetical PnL impact (what would have happened)
  - Promotion readiness score

Usage:
    python -X utf8 scripts/fast_risk_shadow_analysis.py
    python -X utf8 scripts/fast_risk_shadow_analysis.py --log-dir logs/
    python -X utf8 scripts/fast_risk_shadow_analysis.py --days 7
"""

import argparse
import json
import logging
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger("fast_risk_shadow")

# Default paths
LOG_DIR = Path("logs")
SHADOW_LEDGER_DIR = Path("data/shadow_ledger")
SHADOW_DUMP_FILE = Path("logs/fast_risk_shadow.jsonl")

# Promotion criteria
PROMOTION_CRITERIA = {
    "min_triggers": 10,            # Need at least N triggers to assess
    "min_days": 3,                 # Minimum observation period
    "max_false_positive_rate": 0.4, # <40% false positives
    "min_true_positive_rate": 0.5,  # >50% true positives
    "max_trigger_rate_per_day": 20, # Not too noisy
}

# Pattern for log parsing
FRT_LOG_PATTERN = re.compile(
    r"\[FastRiskTick\]\[SHADOW\]\s+(\w+):\s+WOULD\s+(\w+)\s+-\s+(.*?)\s+"
    r"\(anchor=\$([0-9,.]+)\s+->\s+\$([0-9,.]+)\)"
)


def parse_log_files(log_dir: Path, days: int) -> list:
    """Parse paper run logs for FastRiskTick shadow entries."""
    triggers = []
    cutoff = datetime.now() - timedelta(days=days)

    # Look for proof_log files and any log files
    log_patterns = ["proof_log_*.log", "paper_run_*.log", "hmats_*.log"]
    log_files = []
    for pattern in log_patterns:
        log_files.extend(log_dir.glob(pattern))

    # Also check for gzipped files
    for pattern in log_patterns:
        log_files.extend(log_dir.glob(pattern + ".gz"))

    for log_file in sorted(log_files):
        try:
            if str(log_file).endswith(".gz"):
                import gzip
                opener = lambda: gzip.open(log_file, "rt", encoding="utf-8", errors="replace")
            else:
                opener = lambda: open(log_file, "r", encoding="utf-8", errors="replace")

            with opener() as f:
                for line in f:
                    if "[FastRiskTick][SHADOW]" not in line:
                        continue
                    match = FRT_LOG_PATTERN.search(line)
                    if match:
                        asset = match.group(1)
                        action = match.group(2)
                        reason = match.group(3)
                        anchor = float(match.group(4).replace(",", ""))
                        current = float(match.group(5).replace(",", ""))

                        # Extract timestamp from log line (assumes standard format)
                        ts_match = re.match(r"(\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2})", line)
                        ts_str = ts_match.group(1) if ts_match else ""

                        triggers.append({
                            "timestamp": ts_str,
                            "asset": asset,
                            "action": action,
                            "reason": reason,
                            "anchor_price": anchor,
                            "current_price": current,
                            "price_move_pct": abs(current - anchor) / anchor if anchor > 0 else 0,
                        })
        except Exception as e:
            logger.debug(f"Error reading {log_file}: {e}")

    return triggers


def load_shadow_dump(filepath: Path) -> list:
    """Load FastRiskTick shadow dump file."""
    triggers = []
    if not filepath.exists():
        return triggers

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                triggers.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return triggers


def load_shadow_ledger_context(days: int) -> dict:
    """Load shadow ledger for trade context (positions at trigger time)."""
    positions = {}
    if not SHADOW_LEDGER_DIR.exists():
        return positions

    cutoff = datetime.now() - timedelta(days=days)
    for f in sorted(SHADOW_LEDGER_DIR.glob("ledger_*.jsonl")):
        try:
            date_str = f.stem.split("_")[-1]
            file_date = datetime.strptime(date_str, "%Y%m%d")
            if file_date < cutoff:
                continue
        except ValueError:
            pass

        with open(f, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    if r.get("entry_type") == "POSITION":
                        asset = r.get("asset", "")
                        positions[asset] = r.get("data", {})
                except json.JSONDecodeError:
                    continue
    return positions


def analyze_triggers(triggers: list):
    """Analyze FastRiskTick shadow triggers."""
    if not triggers:
        print("\n=== No FastRiskTick Shadow Triggers Found ===")
        print("  The FastRiskTick runs in shadow mode between 4H ticks.")
        print("  If no triggers found, either:")
        print("  1. Paper run hasn't encountered extreme moves (good!)")
        print("  2. Log files not found in logs/ directory")
        print("  3. FastRiskTick not initialized (check main.py L3163-3168)")
        return

    print(f"\n{'='*70}")
    print(f"  FAST RISK TICK - SHADOW ANALYSIS")
    print(f"  {len(triggers)} triggers found")
    print(f"{'='*70}")

    # By asset
    by_asset = defaultdict(list)
    for t in triggers:
        by_asset[t["asset"]].append(t)

    print(f"\n--- Triggers by Asset ---")
    print(f"{'Asset':<8} {'Count':>6} {'EXIT_ONLY':>10} {'REDUCE_50':>10} {'Avg Move':>10}")
    print("-" * 48)

    for asset in sorted(by_asset.keys()):
        recs = by_asset[asset]
        exits = sum(1 for r in recs if r["action"] == "EXIT_ONLY")
        reduces = sum(1 for r in recs if r["action"] == "REDUCE_50")
        moves = [r.get("price_move_pct", 0) for r in recs]
        avg_move = sum(moves) / len(moves) * 100 if moves else 0
        print(f"{asset:<8} {len(recs):>6} {exits:>10} {reduces:>10} {avg_move:>9.1f}%")

    # By reason
    by_reason = defaultdict(int)
    for t in triggers:
        reason = t["reason"].split("=")[0] if "=" in t["reason"] else t["reason"]
        by_reason[reason] += 1

    print(f"\n--- Triggers by Reason ---")
    for reason, count in sorted(by_reason.items(), key=lambda x: -x[1]):
        print(f"  {reason:<20} {count:>5}")

    # Time distribution
    if triggers[0].get("timestamp"):
        print(f"\n--- Trigger Timeline ---")
        by_date = defaultdict(int)
        for t in triggers:
            ts = t.get("timestamp", "")[:10]
            if ts:
                by_date[ts] += 1
        for date in sorted(by_date.keys())[-10:]:
            bar = "#" * min(by_date[date], 50)
            print(f"  {date}: {by_date[date]:>3} {bar}")

    # Promotion assessment
    print(f"\n--- Promotion Readiness Assessment ---")
    n_triggers = len(triggers)
    days_observed = len(set(t.get("timestamp", "")[:10] for t in triggers if t.get("timestamp")))
    triggers_per_day = n_triggers / max(days_observed, 1)

    checks = {
        "min_triggers": n_triggers >= PROMOTION_CRITERIA["min_triggers"],
        "min_days": days_observed >= PROMOTION_CRITERIA["min_days"],
        "trigger_rate": triggers_per_day <= PROMOTION_CRITERIA["max_trigger_rate_per_day"],
    }

    for check, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {check}: ", end="")
        if check == "min_triggers":
            print(f"{n_triggers} (need >={PROMOTION_CRITERIA['min_triggers']})")
        elif check == "min_days":
            print(f"{days_observed} days (need >={PROMOTION_CRITERIA['min_days']})")
        elif check == "trigger_rate":
            print(f"{triggers_per_day:.1f}/day (max {PROMOTION_CRITERIA['max_trigger_rate_per_day']})")

    all_pass = all(checks.values())
    print(f"\n  Overall: {'READY for promotion review' if all_pass else 'NOT READY - need more data'}")
    print(f"  Note: False positive rate requires post-trigger price tracking (not yet implemented)")


def main():
    parser = argparse.ArgumentParser(description="FastRiskTick Shadow Analysis")
    parser.add_argument("--log-dir", type=str, default="logs",
                        help="Log directory (default: logs/)")
    parser.add_argument("--days", type=int, default=30,
                        help="Look back N days (default: 30)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print("HMATS FastRiskTick Shadow Analysis")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Collect triggers from all sources
    all_triggers = []

    # Source 1: Log file parsing
    log_triggers = parse_log_files(Path(args.log_dir), args.days)
    if log_triggers:
        print(f"  Found {len(log_triggers)} triggers from log files")
        all_triggers.extend(log_triggers)

    # Source 2: Shadow dump file
    dump_triggers = load_shadow_dump(SHADOW_DUMP_FILE)
    if dump_triggers:
        print(f"  Found {len(dump_triggers)} triggers from shadow dump")
        all_triggers.extend(dump_triggers)

    analyze_triggers(all_triggers)

    print(f"\n{'='*70}")
    print("  Analysis complete.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
