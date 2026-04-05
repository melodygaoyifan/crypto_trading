"""
HMATS - Execution Quality Report
==================================

Reads execution quality JSONL logs and shadow ledger data.
Produces slippage/fill stats by asset and session, worst executions,
and PA vs legacy comparison.

Usage:
    python -X utf8 scripts/execution_quality_report.py
    python -X utf8 scripts/execution_quality_report.py --days 7
    python -X utf8 scripts/execution_quality_report.py --source shadow_ledger
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger("execution_quality_report")

# Data directories
EXEC_LOG_DIR = Path("data/execution_logs")
SHADOW_LEDGER_DIR = Path("data/shadow_ledger")


def load_execution_logs(days: int = 30) -> list:
    """Load execution quality JSONL logs."""
    records = []
    if not EXEC_LOG_DIR.exists():
        logger.info(f"No execution logs directory: {EXEC_LOG_DIR}")
        return records

    cutoff = datetime.now() - timedelta(days=days)
    for f in sorted(EXEC_LOG_DIR.glob("execution_quality_*.jsonl")):
        try:
            # Extract date from filename
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
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def load_shadow_ledger(days: int = 30) -> list:
    """Load shadow ledger JSONL entries."""
    records = []
    if not SHADOW_LEDGER_DIR.exists():
        logger.info(f"No shadow ledger directory: {SHADOW_LEDGER_DIR}")
        return records

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
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def analyze_execution_logs(records: list):
    """Analyze execution quality log records."""
    if not records:
        print("\n=== No Execution Quality Logs Found ===")
        print("  (Paper run may not have produced execution_quality logs yet)")
        print(f"  Expected location: {EXEC_LOG_DIR}/execution_quality_YYYYMMDD.jsonl")
        return

    print(f"\n{'='*70}")
    print(f"  EXECUTION QUALITY REPORT - {len(records)} records")
    print(f"{'='*70}")

    # Group by asset
    by_asset = defaultdict(list)
    for r in records:
        by_asset[r.get("asset", "UNKNOWN")].append(r)

    print(f"\n--- Slippage & Fill Stats by Asset ---")
    print(f"{'Asset':<8} {'Count':>6} {'Avg Slip':>10} {'Med Slip':>10} {'Fill%':>8} {'Avg Time':>10}")
    print("-" * 55)

    for asset in sorted(by_asset.keys()):
        recs = by_asset[asset]
        slippages = [r.get("realized_slippage_bps", 0) for r in recs]
        fill_ratios = [r.get("fill_ratio", 1.0) for r in recs]
        fill_times = [r.get("time_to_fill_sec", 0) for r in recs]

        avg_slip = sum(slippages) / len(slippages) if slippages else 0
        med_slip = sorted(slippages)[len(slippages) // 2] if slippages else 0
        avg_fill = sum(fill_ratios) / len(fill_ratios) * 100 if fill_ratios else 0
        avg_time = sum(fill_times) / len(fill_times) if fill_times else 0

        print(f"{asset:<8} {len(recs):>6} {avg_slip:>9.1f}bp {med_slip:>9.1f}bp "
              f"{avg_fill:>7.1f}% {avg_time:>9.1f}s")

    # Worst 10 executions
    print(f"\n--- Worst 10 Executions (by realized slippage) ---")
    sorted_by_slip = sorted(records, key=lambda r: r.get("realized_slippage_bps", 0), reverse=True)
    for i, r in enumerate(sorted_by_slip[:10]):
        ts = r.get("timestamp", "?")[:19]
        asset = r.get("asset", "?")
        slip = r.get("realized_slippage_bps", 0)
        expected = r.get("expected_slippage_bps", 0)
        fill_r = r.get("fill_ratio", 0) * 100
        print(f"  {i+1:>2}. [{ts}] {asset:<6} slip={slip:>6.1f}bp (expected {expected:.1f}bp) fill={fill_r:.0f}%")

    # PA vs Legacy comparison
    pa_records = [r for r in records if r.get("execution_source") == "rl" or r.get("rl_advice_used")]
    legacy_records = [r for r in records if not r.get("rl_advice_used") and r.get("execution_source") != "rl"]

    if pa_records or legacy_records:
        print(f"\n--- PA vs Legacy Execution ---")
        for label, recs in [("PA/RL", pa_records), ("Legacy", legacy_records)]:
            if not recs:
                print(f"  {label}: no records")
                continue
            slips = [r.get("realized_slippage_bps", 0) for r in recs]
            avg_s = sum(slips) / len(slips)
            fills = [r.get("fill_ratio", 1.0) for r in recs]
            avg_f = sum(fills) / len(fills) * 100
            print(f"  {label:<8}: n={len(recs):>5}, avg_slip={avg_s:>6.1f}bp, avg_fill={avg_f:.1f}%")


def analyze_shadow_ledger(records: list):
    """Analyze shadow ledger records for execution quality."""
    if not records:
        print("\n=== No Shadow Ledger Data ===")
        return

    # Separate entry types
    intents = [r for r in records if r.get("entry_type") == "INTENT"]
    positions = [r for r in records if r.get("entry_type") == "POSITION"]
    fills = [r for r in records if r.get("entry_type") == "FILL"]

    print(f"\n{'='*70}")
    print(f"  SHADOW LEDGER ANALYSIS - {len(records)} total entries")
    print(f"{'='*70}")
    print(f"  INTENT: {len(intents)}  |  POSITION: {len(positions)}  |  FILL: {len(fills)}  |  Other: {len(records) - len(intents) - len(positions) - len(fills)}")

    # Intent analysis by asset
    by_asset = defaultdict(list)
    for r in intents:
        by_asset[r.get("asset", "UNKNOWN")].append(r)

    print(f"\n--- Intents by Asset ---")
    print(f"{'Asset':<8} {'Count':>6} {'Entries':>8} {'Exits':>8} {'Avg Conf':>10} {'Avg |Dir|':>10}")
    print("-" * 55)

    for asset in sorted(by_asset.keys()):
        recs = by_asset[asset]
        entries = sum(1 for r in recs if r.get("data", {}).get("is_entry", False))
        exits = len(recs) - entries
        confs = [r.get("data", {}).get("confidence", 0) for r in recs]
        dirs = [abs(r.get("data", {}).get("direction", 0)) for r in recs]
        avg_conf = sum(confs) / len(confs) if confs else 0
        avg_dir = sum(dirs) / len(dirs) if dirs else 0
        print(f"{asset:<8} {len(recs):>6} {entries:>8} {exits:>8} {avg_conf:>9.3f} {avg_dir:>9.3f}")

    # Position changes with PnL
    if positions:
        pnl_records = [r for r in positions if r.get("data", {}).get("realized_pnl", 0) != 0]
        if pnl_records:
            total_pnl = sum(r["data"]["realized_pnl"] for r in pnl_records)
            winners = sum(1 for r in pnl_records if r["data"]["realized_pnl"] > 0)
            print(f"\n--- Realized PnL Summary ---")
            print(f"  Trades with PnL: {len(pnl_records)}")
            print(f"  Total PnL: {total_pnl:+.4f}")
            print(f"  Win rate: {winners}/{len(pnl_records)} ({winners/len(pnl_records)*100:.1f}%)")

    # Session analysis
    sessions = defaultdict(int)
    for r in records:
        sessions[r.get("session_id", "unknown")] += 1
    print(f"\n--- Sessions ---")
    for sid in sorted(sessions.keys())[-5:]:
        print(f"  {sid}: {sessions[sid]} entries")


def main():
    parser = argparse.ArgumentParser(description="Execution Quality Report")
    parser.add_argument("--days", type=int, default=30, help="Look back N days (default: 30)")
    parser.add_argument("--source", choices=["all", "exec_logs", "shadow_ledger"],
                        default="all", help="Data source (default: all)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print(f"HMATS Execution Quality Report")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Lookback: {args.days} days")

    if args.source in ("all", "exec_logs"):
        exec_records = load_execution_logs(args.days)
        analyze_execution_logs(exec_records)

    if args.source in ("all", "shadow_ledger"):
        shadow_records = load_shadow_ledger(args.days)
        analyze_shadow_ledger(shadow_records)

    print(f"\n{'='*70}")
    print("  Report complete.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
