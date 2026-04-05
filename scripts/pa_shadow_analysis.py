"""
HMATS - Passive-Aggressive Executor Shadow Analysis
=====================================================

Compares PA executor recommendations vs actual execution results.
Reads paper run logs for PA_SHADOW/PA_ACTIVE entries and correlates
with execution quality and shadow ledger data.

Usage:
    python -X utf8 scripts/pa_shadow_analysis.py
    python -X utf8 scripts/pa_shadow_analysis.py --days 14
    python -X utf8 scripts/pa_shadow_analysis.py --log-dir logs/
"""

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger("pa_shadow_analysis")

# Default paths
LOG_DIR = Path("logs")
EXEC_LOG_DIR = Path("data/execution_logs")
SHADOW_LEDGER_DIR = Path("data/shadow_ledger")

# Log patterns for PA decisions
# [PA_SHADOW] BTC: PASSIVE limit offset=1.2bps (vpin=0.35, conf=0.5, alpha=60.0bps)
PA_LOG_PATTERN = re.compile(
    r"\[(PA_SHADOW|PA_ACTIVE)\]\s+(\w+(?:/\w+)?):\s+(\w+)\s+"
    r"(.*?)(?:\s+\(.*?\))?"
    r"$"
)

# Simpler pattern to catch PA lines
PA_SIMPLE_PATTERN = re.compile(
    r"\[(PA_SHADOW|PA_ACTIVE)\]\s+(\S+?):\s+(.*)"
)


def parse_log_files(log_dir: Path, days: int) -> list:
    """Parse paper run logs for PA executor entries."""
    decisions = []
    cutoff = datetime.now() - timedelta(days=days)

    log_patterns = ["proof_log_*.log", "paper_run_*.log", "hmats_*.log"]
    log_files = []
    for pattern in log_patterns:
        log_files.extend(log_dir.glob(pattern))
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
                    if "PA_SHADOW" not in line and "PA_ACTIVE" not in line:
                        continue

                    match = PA_SIMPLE_PATTERN.search(line)
                    if match:
                        mode = match.group(1)
                        asset = match.group(2).rstrip(":")
                        details = match.group(3)

                        # Extract timestamp
                        ts_match = re.match(r"(\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2})", line)
                        ts_str = ts_match.group(1) if ts_match else ""

                        # Determine execution mode from details
                        exec_mode = "UNKNOWN"
                        if "PASSIVE" in details.upper() or "limit" in details.lower() or "maker" in details.lower():
                            exec_mode = "PASSIVE"
                        elif "AGGRESSIVE" in details.upper() or "market" in details.lower() or "taker" in details.lower():
                            exec_mode = "AGGRESSIVE"

                        decisions.append({
                            "timestamp": ts_str,
                            "mode": mode,  # PA_SHADOW or PA_ACTIVE
                            "asset": asset.replace("/USD", ""),
                            "exec_mode": exec_mode,
                            "details": details.strip(),
                        })
        except Exception as e:
            logger.debug(f"Error reading {log_file}: {e}")

    return decisions


def load_execution_logs(days: int) -> list:
    """Load execution quality logs for correlation."""
    records = []
    if not EXEC_LOG_DIR.exists():
        return records

    cutoff = datetime.now() - timedelta(days=days)
    for f in sorted(EXEC_LOG_DIR.glob("execution_quality_*.jsonl")):
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


def analyze_pa_decisions(decisions: list, exec_records: list):
    """Analyze PA executor shadow decisions."""
    if not decisions:
        print("\n=== No PA Executor Decisions Found in Logs ===")
        print("  PA decisions are logged as [PA_SHADOW] or [PA_ACTIVE] in proof logs.")
        print("  If no entries found, either:")
        print("  1. PA executor not initialized (check config pa_executor_config)")
        print("  2. Log files not in expected location")
        print("  3. Paper run hasn't produced PA decisions yet")
        return

    print(f"\n{'='*70}")
    print(f"  PA EXECUTOR SHADOW ANALYSIS - {len(decisions)} decisions")
    print(f"{'='*70}")

    # Mode breakdown
    shadow = [d for d in decisions if d["mode"] == "PA_SHADOW"]
    active = [d for d in decisions if d["mode"] == "PA_ACTIVE"]
    print(f"\n  SHADOW mode: {len(shadow)} decisions")
    print(f"  ACTIVE mode: {len(active)} decisions")

    # By asset
    by_asset = defaultdict(list)
    for d in decisions:
        by_asset[d["asset"]].append(d)

    print(f"\n--- Decisions by Asset ---")
    print(f"{'Asset':<8} {'Count':>6} {'Passive':>8} {'Aggressive':>10} {'Unknown':>8}")
    print("-" * 45)

    for asset in sorted(by_asset.keys()):
        recs = by_asset[asset]
        passive = sum(1 for r in recs if r["exec_mode"] == "PASSIVE")
        aggressive = sum(1 for r in recs if r["exec_mode"] == "AGGRESSIVE")
        unknown = sum(1 for r in recs if r["exec_mode"] == "UNKNOWN")
        print(f"{asset:<8} {len(recs):>6} {passive:>8} {aggressive:>10} {unknown:>8}")

    # Overall passive/aggressive ratio
    total_passive = sum(1 for d in decisions if d["exec_mode"] == "PASSIVE")
    total_aggressive = sum(1 for d in decisions if d["exec_mode"] == "AGGRESSIVE")
    total = total_passive + total_aggressive
    if total > 0:
        print(f"\n--- Passive/Aggressive Ratio ---")
        print(f"  Passive:    {total_passive:>5} ({total_passive/total*100:.1f}%)")
        print(f"  Aggressive: {total_aggressive:>5} ({total_aggressive/total*100:.1f}%)")
        print(f"  Target: ~70% passive for cost savings")
        if total_passive / total > 0.65:
            print(f"  Status: GOOD - passive ratio above 65%")
        elif total_passive / total > 0.50:
            print(f"  Status: OK - passive ratio could be higher")
        else:
            print(f"  Status: REVIEW - aggressive ratio too high (consider tuning thresholds)")

    # Time distribution
    print(f"\n--- Decision Timeline ---")
    by_date = defaultdict(lambda: {"passive": 0, "aggressive": 0, "total": 0})
    for d in decisions:
        ts = d.get("timestamp", "")[:10]
        if ts:
            by_date[ts]["total"] += 1
            if d["exec_mode"] == "PASSIVE":
                by_date[ts]["passive"] += 1
            elif d["exec_mode"] == "AGGRESSIVE":
                by_date[ts]["aggressive"] += 1

    for date in sorted(by_date.keys())[-10:]:
        stats = by_date[date]
        p_bar = "P" * min(stats["passive"], 30)
        a_bar = "A" * min(stats["aggressive"], 30)
        print(f"  {date}: {stats['total']:>3} [{p_bar}{a_bar}]")

    # Correlation with execution quality if available
    if exec_records:
        print(f"\n--- Execution Quality Correlation ---")
        pa_fills = [r for r in exec_records if r.get("rl_advice_used")]
        legacy_fills = [r for r in exec_records if not r.get("rl_advice_used")]

        if pa_fills and legacy_fills:
            pa_slip = sum(r.get("realized_slippage_bps", 0) for r in pa_fills) / len(pa_fills)
            legacy_slip = sum(r.get("realized_slippage_bps", 0) for r in legacy_fills) / len(legacy_fills)
            improvement = legacy_slip - pa_slip
            print(f"  PA avg slippage:     {pa_slip:>6.1f} bps (n={len(pa_fills)})")
            print(f"  Legacy avg slippage: {legacy_slip:>6.1f} bps (n={len(legacy_fills)})")
            print(f"  Improvement:         {improvement:>+6.1f} bps")
        else:
            print(f"  Not enough comparative data (PA fills: {len(pa_fills)}, Legacy: {len(legacy_fills)})")
    else:
        print(f"\n  (No execution quality logs for correlation analysis)")

    # Promotion recommendation
    print(f"\n--- Promotion Recommendation ---")
    if len(shadow) > 0 and len(active) == 0:
        print(f"  Status: SHADOW ONLY - PA executor is recommendation-only")
        if len(shadow) >= 20:
            if total > 0 and total_passive / total > 0.5:
                print(f"  Recommendation: Consider ACTIVE promotion (sufficient data, healthy ratio)")
            else:
                print(f"  Recommendation: Tune thresholds before promotion (aggressive ratio too high)")
        else:
            print(f"  Recommendation: Need more shadow data ({len(shadow)}/20 minimum)")
    elif len(active) > 0:
        print(f"  Status: ACTIVE - PA executor is driving execution")
        print(f"  Monitor: slippage savings vs legacy baseline")


def main():
    parser = argparse.ArgumentParser(description="PA Executor Shadow Analysis")
    parser.add_argument("--log-dir", type=str, default="logs",
                        help="Log directory (default: logs/)")
    parser.add_argument("--days", type=int, default=30,
                        help="Look back N days (default: 30)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print("HMATS PA Executor Shadow Analysis")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    decisions = parse_log_files(Path(args.log_dir), args.days)
    exec_records = load_execution_logs(args.days)

    analyze_pa_decisions(decisions, exec_records)

    print(f"\n{'='*70}")
    print("  Analysis complete.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
