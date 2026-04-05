"""
Paper Run Health Monitor - reads proof logs + shadow ledger to produce health reports.

Usage:
    python scripts/paper_run_monitor.py                   # latest session
    python scripts/paper_run_monitor.py --hours 24        # last 24 hours
    python scripts/paper_run_monitor.py --session 20260208_123456   # specific session

No new external dependencies - stdlib + json only.
"""

import argparse
import gzip
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Project root (two levels up from scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
PROOF_LOG_DIR = LOGS_DIR
EVENT_LOG_DIR = LOGS_DIR / "events"
SHADOW_LEDGER_DIR = PROJECT_ROOT / "data" / "shadow_ledger"

# ── Proof log parsing ────────────────────────────────────────────────

# Regex for the v3.6.1 proof log line produced by TradeIntentV36.to_proof_log()
PROOF_RE = re.compile(
    r"\[PROOF\]\s*"
    r"(?P<timestamp>\S+)\s*\|\s*"
    r"v(?P<version>[\d.]+)\s*\|\s*"
    r"DataValid=(?P<data_valid>\w+)\s*\|\s*"
    r"NO_TRADE_internal=(?P<no_trade>\w+)\s*\|\s*"
    r"OPP_internal=(?P<opp>\w+)\s*\|\s*"
    r"AlphaGate=\{pass=(?P<alpha_pass>\w+),est=(?P<alpha_est>[\d.-]+),thresh=(?P<alpha_thresh>[\d.-]+)\}\s*\|\s*"
    r"Tranche=\{action=(?P<tranche_action>\w+),level=(?P<tranche_level>\d+)\}\s*\|\s*"
    r"Conflict=(?P<conflict>\w+)\s*\|\s*"
    r"Mode=(?P<mode>\w+)\s*\|\s*"
    r"Quant=(?P<quant>[^|]+)\s*\|\s*"
    r"DRL=(?P<drl>\w+)\s*\|\s*"
    r"FailureMemory=\{caution=(?P<caution>\w+),boost=(?P<boost>[\d.-]+)\}\s*\|\s*"
    r"SOL_dom=\{active=(?P<sol_dom_active>\w+),TTL=(?P<sol_dom_ttl>\d+)\}\s*\|\s*"
    r"Deadlock=(?P<deadlock>\w+)\s*\|\s*"
    r"Intent=\{asset=(?P<asset>\w+),dir=(?P<dir>[+-]?[\d.]+),exp=(?P<exp>[^}]+)\}\s*\|\s*"
    r"Exec=(?P<exec>\w+)"
)

# Enriched tail:  | regime=X conf=Y power=Z | strategy=X sig=Y str=Z | SOL_Dominance: ACTIVE (source=FALLBACK)
REGIME_RE = re.compile(r"regime=(?P<regime>\S+)\s+conf=(?P<conf>[\d.]+)\s+power=(?P<power>[\d.]+)")
STRATEGY_RE = re.compile(r"strategy=(?P<strategy>\S+)\s+sig=(?P<sig>[+-]?[\d.]+)\s+str=(?P<str>[\d.]+)")
SOL_DOM_RE = re.compile(r"SOL_Dominance:\s*(?P<status>ACTIVE|INACTIVE)\s*\(source=(?P<source>\w+)\)")
WARMUP_RE = re.compile(r"\[WARMUP tick (\d+)\]")


def parse_proof_line(line: str) -> Optional[Dict[str, Any]]:
    """Parse a single proof log line into a structured dict."""
    m = PROOF_RE.search(line)
    if not m:
        return None

    entry = {
        "timestamp": m.group("timestamp"),
        "asset": m.group("asset"),
        "data_valid": m.group("data_valid") == "True",
        "alpha_pass": m.group("alpha_pass") == "True",
        "alpha_est": float(m.group("alpha_est")),
        "alpha_thresh": float(m.group("alpha_thresh")),
        "tranche_action": m.group("tranche_action"),
        "tranche_level": int(m.group("tranche_level")),
        "mode": m.group("mode"),
        "direction": float(m.group("dir")),
        "exposure": m.group("exp"),
        "exec_mode": m.group("exec"),
        "sol_dom_active": m.group("sol_dom_active") == "True",
        "sol_dom_ttl": int(m.group("sol_dom_ttl")),
        "is_warmup": bool(WARMUP_RE.search(line)),
    }

    # Parse enriched regime/strategy/sol_dominance tail
    rm = REGIME_RE.search(line)
    if rm:
        entry["regime"] = rm.group("regime")
        entry["regime_conf"] = float(rm.group("conf"))
        entry["regime_power"] = float(rm.group("power"))

    sm = STRATEGY_RE.search(line)
    if sm:
        entry["strategy"] = sm.group("strategy")
        entry["strategy_signal"] = float(sm.group("sig"))
        entry["strategy_strength"] = float(sm.group("str"))

    sdm = SOL_DOM_RE.search(line)
    if sdm:
        entry["sol_dom_fallback_status"] = sdm.group("status")
        entry["sol_dom_fallback_source"] = sdm.group("source")

    # Determine if this tick resulted in a trade or FLATTEN
    exp_str = entry["exposure"]
    try:
        # exposure is formatted as "12.3%" or "0.0%"
        exp_pct = float(exp_str.replace("%", ""))
    except (ValueError, AttributeError):
        exp_pct = 0.0
    entry["exposure_pct"] = exp_pct
    entry["is_flatten"] = entry["tranche_action"] == "FLATTEN" or exp_pct == 0.0
    entry["is_trade"] = not entry["is_flatten"] and entry["alpha_pass"]

    return entry


def load_proof_logs(hours: Optional[float] = None, session: Optional[str] = None) -> List[Dict]:
    """Load and parse proof log files from logs/ directory."""
    entries = []

    if not PROOF_LOG_DIR.exists():
        return entries

    # Find matching files - proof_log_*.log + ALWAYS include hmats.log / stderr
    # [FIX-MONITOR] Previously only used fallbacks when proof_log files were absent.
    # But proof logs are now written to hmats.log/stderr, not separate files.
    log_files = sorted(PROOF_LOG_DIR.glob("proof_log_*.log"))
    _fallbacks = [PROOF_LOG_DIR / "hmats.log", PROOF_LOG_DIR / "paper_run_stderr.log"]
    for fb in _fallbacks:
        if fb.exists() and fb not in log_files:
            log_files.append(fb)
    if not log_files:
        return entries

    if session:
        log_files = [f for f in log_files if session in f.name]
    elif hours:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        filtered = []
        for f in log_files:
            # Extract timestamp from filename: proof_log_YYYYMMDD_HHMMSS.log
            m = re.search(r"proof_log_(\d{8}_\d{6})", f.name)
            if m:
                try:
                    ts = datetime.strptime(m.group(1), "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
                    if ts >= cutoff:
                        filtered.append(f)
                except ValueError:
                    filtered.append(f)
            else:
                filtered.append(f)
        log_files = filtered
    else:
        # Default: latest file only
        log_files = log_files[-1:]

    for log_file in log_files:
        try:
            with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line or "[PROOF]" not in line:
                        continue
                    parsed = parse_proof_line(line)
                    if parsed:
                        parsed["source_file"] = log_file.name
                        entries.append(parsed)
        except Exception as e:
            print(f"  [WARN] Failed to read {log_file}: {e}", file=sys.stderr)

    return entries


# ── Shadow ledger parsing ────────────────────────────────────────────

def load_shadow_ledger(hours: Optional[float] = None) -> List[Dict]:
    """Load shadow ledger JSONL entries."""
    entries = []

    if not SHADOW_LEDGER_DIR.exists():
        return entries

    ledger_files = sorted(SHADOW_LEDGER_DIR.glob("ledger_*.jsonl"))
    if not ledger_files:
        return entries

    if hours:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        filtered = []
        for f in ledger_files:
            m = re.search(r"ledger_(\d{8})", f.name)
            if m:
                try:
                    ts = datetime.strptime(m.group(1), "%Y%m%d").replace(tzinfo=timezone.utc)
                    if ts >= cutoff - timedelta(days=1):  # include day overlap
                        filtered.append(f)
                except ValueError:
                    filtered.append(f)
        ledger_files = filtered
    else:
        ledger_files = ledger_files[-2:]  # last 2 days

    for lf in ledger_files:
        try:
            with open(lf, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            print(f"  [WARN] Failed to read {lf}: {e}", file=sys.stderr)

    return entries


# ── Health report generation ─────────────────────────────────────────

def generate_health_report(
    proof_entries: List[Dict],
    ledger_entries: List[Dict],
) -> Dict[str, Any]:
    """Analyze proof logs + shadow ledger and return structured health report."""

    report: Dict[str, Any] = {}

    # ── Section 1: Tick Activity ──
    total_ticks = len(proof_entries)
    warmup_ticks = sum(1 for e in proof_entries if e.get("is_warmup"))
    active_ticks = total_ticks - warmup_ticks
    assets_seen = set(e["asset"] for e in proof_entries)
    ticks_per_asset = Counter(e["asset"] for e in proof_entries)

    report["tick_activity"] = {
        "total_ticks": total_ticks,
        "warmup_ticks": warmup_ticks,
        "active_ticks": active_ticks,
        "assets": dict(ticks_per_asset),
    }

    if total_ticks == 0:
        report["health_verdict"] = "NO_DATA"
        report["alerts"] = ["No proof log entries found. Is the paper run active?"]
        return report

    # ── Section 2: Trade Activity ──
    post_warmup = [e for e in proof_entries if not e.get("is_warmup")]
    trades = [e for e in post_warmup if e.get("is_trade")]
    flattens = [e for e in post_warmup if e.get("is_flatten")]
    trades_per_asset = Counter(e["asset"] for e in trades)
    flatten_per_asset = Counter(e["asset"] for e in flattens)

    flatten_rate = len(flattens) / max(len(post_warmup), 1)
    # Project 30-day trade count (based on 4H ticks = 6 per day per asset)
    if post_warmup:
        ticks_per_day = 6 * len(assets_seen)
        days_observed = max(len(post_warmup) / ticks_per_day, 1 / 6)
        trades_per_day = len(trades) / days_observed
        projected_30d = trades_per_day * 30
    else:
        projected_30d = 0

    report["trade_activity"] = {
        "trades": len(trades),
        "flattens": len(flattens),
        "flatten_rate": f"{flatten_rate:.1%}",
        "trades_per_asset": dict(trades_per_asset),
        "flatten_per_asset": dict(flatten_per_asset),
        "projected_30d_trades": round(projected_30d, 1),
    }

    # ── Section 3: Veto Analysis ──
    veto_reasons = Counter()
    for e in post_warmup:
        if e.get("is_flatten"):
            ta = e.get("tranche_action", "UNKNOWN")
            if ta == "FLATTEN":
                veto_reasons["TRANCHE_FLATTEN"] += 1
            elif not e.get("alpha_pass"):
                veto_reasons["ALPHA_GATE"] += 1
            else:
                veto_reasons["OTHER_FLATTEN"] += 1

    report["veto_analysis"] = {
        "total_vetoes": len(flattens),
        "reasons": dict(veto_reasons.most_common(10)),
    }

    # ── Section 4: Regime Distribution ──
    regime_counts = Counter(e.get("regime", "UNKNOWN") for e in post_warmup if "regime" in e)
    regime_total = sum(regime_counts.values())
    regime_dist = {
        k: f"{v}/{regime_total} ({v / max(regime_total, 1):.0%})"
        for k, v in regime_counts.most_common()
    }

    report["regime_distribution"] = regime_dist

    # ── Section 5: Alpha Health ──
    alpha_stats: Dict[str, Any] = {}
    for asset in assets_seen:
        asset_entries = [e for e in post_warmup if e["asset"] == asset]
        alphas = [e["alpha_est"] for e in asset_entries if "alpha_est" in e]
        thresholds = [e["alpha_thresh"] for e in asset_entries if "alpha_thresh" in e]
        if alphas:
            mean_alpha = sum(alphas) / len(alphas)
            max_alpha = max(alphas)
            min_alpha = min(alphas)
            pass_count = sum(1 for a, t in zip(alphas, thresholds) if a >= t)
            pass_rate = pass_count / len(alphas)
            mean_thresh = sum(thresholds) / len(thresholds) if thresholds else 0
            alpha_stats[asset] = {
                "mean_bps": round(mean_alpha, 1),
                "max_bps": round(max_alpha, 1),
                "min_bps": round(min_alpha, 1),
                "mean_threshold_bps": round(mean_thresh, 1),
                "pass_rate": f"{pass_rate:.0%}",
                "gap_bps": round(mean_alpha - mean_thresh, 1),
            }

    report["alpha_health"] = alpha_stats

    # ── Section 6: Strategy Selection ──
    strategy_counts = Counter(e.get("strategy", "UNKNOWN") for e in post_warmup if "strategy" in e)
    strategy_total = sum(strategy_counts.values())
    strategy_dist = {
        k: f"{v}/{strategy_total} ({v / max(strategy_total, 1):.0%})"
        for k, v in strategy_counts.most_common()
    }

    # Strategy signal strength stats
    strategy_strength: Dict[str, List[float]] = defaultdict(list)
    for e in post_warmup:
        if "strategy" in e and "strategy_strength" in e:
            strategy_strength[e["strategy"]].append(e["strategy_strength"])
    strategy_avg_str = {
        k: round(sum(v) / len(v), 3) for k, v in strategy_strength.items()
    }

    report["strategy_selection"] = {
        "distribution": strategy_dist,
        "avg_strength": strategy_avg_str,
    }

    # ── Section 7: FLATTEN Analysis ──
    flatten_by_asset: Dict[str, Dict] = {}
    for asset in assets_seen:
        asset_pw = [e for e in post_warmup if e["asset"] == asset]
        asset_flat = [e for e in asset_pw if e.get("is_flatten")]
        if asset_pw:
            rate = len(asset_flat) / len(asset_pw)
            flatten_by_asset[asset] = {
                "flatten_count": len(asset_flat),
                "total_ticks": len(asset_pw),
                "rate": f"{rate:.0%}",
                "consecutive_flattens": _max_consecutive(asset_pw, "is_flatten"),
            }

    report["flatten_analysis"] = flatten_by_asset

    # ── Section 8: SOL Dominance Tracking ──
    sol_entries = [e for e in post_warmup if e["asset"] == "SOL"]
    sol_dom_active_count = sum(1 for e in sol_entries if e.get("sol_dom_active"))
    sol_fb_active = sum(1 for e in sol_entries if e.get("sol_dom_fallback_status") == "ACTIVE")
    sol_fb_inactive = sum(1 for e in sol_entries if e.get("sol_dom_fallback_status") == "INACTIVE")

    report["sol_dominance"] = {
        "total_sol_ticks": len(sol_entries),
        "v36_dom_active": sol_dom_active_count,
        "fallback_active": sol_fb_active,
        "fallback_inactive": sol_fb_inactive,
        "fallback_trigger_rate": (
            f"{sol_fb_active / max(sol_fb_active + sol_fb_inactive, 1):.0%}"
        ),
    }

    # ── Section 9: Shadow Ledger Stats ──
    if ledger_entries:
        entry_types = Counter(e.get("entry_type", "UNKNOWN") for e in ledger_entries)
        ledger_assets = Counter(e.get("asset", "UNKNOWN") for e in ledger_entries)
        report["shadow_ledger"] = {
            "total_entries": len(ledger_entries),
            "by_type": dict(entry_types.most_common()),
            "by_asset": dict(ledger_assets.most_common()),
        }
    else:
        report["shadow_ledger"] = {"total_entries": 0, "note": "No ledger files found"}

    # ── Section 10: Health Verdict ──
    alerts = []

    # Alert: High flatten rate
    if flatten_rate > 0.95 and active_ticks > 6:
        alerts.append(f"CRITICAL: Flatten rate {flatten_rate:.0%} - system not trading")
    elif flatten_rate > 0.80 and active_ticks > 6:
        alerts.append(f"WARNING: High flatten rate {flatten_rate:.0%}")

    # Alert: Zero trades
    if len(trades) == 0 and active_ticks > 0:
        alerts.append("CRITICAL: Zero trades across all ticks")

    # Alert: Low alpha pass rate per asset
    for asset, stats in alpha_stats.items():
        rate_str = stats.get("pass_rate", "0%")
        rate_val = float(rate_str.replace("%", "")) / 100
        if rate_val < 0.1 and ticks_per_asset.get(asset, 0) > 3:
            alerts.append(f"WARNING: {asset} alpha pass rate {rate_str} - signals too weak")

    # Alert: Projected trades too low for StatisticalPromotionGate
    if projected_30d < 30:
        alerts.append(
            f"INFO: Projected 30d trades = {projected_30d:.0f} "
            f"(need 30+ for DRL promotion)"
        )

    # Alert: Single regime dominance
    if regime_counts and regime_total > 6:
        top_regime, top_count = regime_counts.most_common(1)[0]
        if top_count / regime_total > 0.90:
            alerts.append(
                f"INFO: Regime stuck in {top_regime} ({top_count}/{regime_total})"
            )

    # Alert: SOL dominance never triggering
    if len(sol_entries) > 6 and sol_fb_active == 0:
        alerts.append("INFO: SOL Dominance fallback never triggered (conservative gate working)")

    verdict = "HEALTHY"
    if any("CRITICAL" in a for a in alerts):
        verdict = "CRITICAL"
    elif any("WARNING" in a for a in alerts):
        verdict = "DEGRADED"

    report["health_verdict"] = verdict
    report["alerts"] = alerts

    return report


def _max_consecutive(entries: List[Dict], key: str) -> int:
    """Find max consecutive True values for a given key."""
    max_run = 0
    current_run = 0
    for e in entries:
        if e.get(key):
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    return max_run


# ── Report printing ──────────────────────────────────────────────────

def print_report(report: Dict[str, Any]) -> None:
    """Pretty-print the health report to stdout."""
    W = 60  # column width

    def header(title: str):
        print(f"\n{'=' * W}")
        print(f"  {title}")
        print(f"{'=' * W}")

    def kv(key: str, val: Any, indent: int = 2):
        print(f"{' ' * indent}{key:<30} {val}")

    header("HMATS PAPER RUN HEALTH REPORT")
    print(f"  Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Verdict:   {report.get('health_verdict', 'UNKNOWN')}")

    # Alerts
    alerts = report.get("alerts", [])
    if alerts:
        header("ALERTS")
        for a in alerts:
            print(f"  * {a}")
    else:
        header("ALERTS")
        print("  (none)")

    # Tick Activity
    ta = report.get("tick_activity", {})
    header("TICK ACTIVITY")
    kv("Total ticks:", ta.get("total_ticks", 0))
    kv("Warmup ticks:", ta.get("warmup_ticks", 0))
    kv("Active ticks:", ta.get("active_ticks", 0))
    for asset, count in sorted(ta.get("assets", {}).items()):
        kv(f"  {asset}:", count)

    # Trade Activity
    tra = report.get("trade_activity", {})
    header("TRADE ACTIVITY")
    kv("Trades:", tra.get("trades", 0))
    kv("Flattens:", tra.get("flattens", 0))
    kv("Flatten rate:", tra.get("flatten_rate", "N/A"))
    kv("Projected 30d trades:", tra.get("projected_30d_trades", 0))
    for asset, count in sorted(tra.get("trades_per_asset", {}).items()):
        kv(f"  {asset} trades:", count)

    # Veto Analysis
    va = report.get("veto_analysis", {})
    header("VETO ANALYSIS")
    kv("Total vetoes:", va.get("total_vetoes", 0))
    for reason, count in va.get("reasons", {}).items():
        kv(f"  {reason}:", count)

    # Regime Distribution
    rd = report.get("regime_distribution", {})
    header("REGIME DISTRIBUTION")
    for regime, stat in rd.items():
        kv(f"  {regime}:", stat)

    # Alpha Health
    ah = report.get("alpha_health", {})
    header("ALPHA HEALTH (per asset)")
    for asset, stats in sorted(ah.items()):
        print(f"\n  {asset}:")
        kv("Mean alpha:", f"{stats['mean_bps']} bps", indent=4)
        kv("Max alpha:", f"{stats['max_bps']} bps", indent=4)
        kv("Min alpha:", f"{stats['min_bps']} bps", indent=4)
        kv("Threshold:", f"{stats['mean_threshold_bps']} bps", indent=4)
        kv("Pass rate:", stats["pass_rate"], indent=4)
        kv("Gap (mean-thresh):", f"{stats['gap_bps']} bps", indent=4)

    # Strategy Selection
    ss = report.get("strategy_selection", {})
    header("STRATEGY SELECTION")
    for strat, stat in ss.get("distribution", {}).items():
        avg = ss.get("avg_strength", {}).get(strat, "?")
        kv(f"  {strat}:", f"{stat}  (avg_strength={avg})")

    # FLATTEN Analysis
    fa = report.get("flatten_analysis", {})
    header("FLATTEN ANALYSIS (per asset)")
    for asset, stats in sorted(fa.items()):
        print(f"\n  {asset}:")
        kv("Flatten count:", stats["flatten_count"], indent=4)
        kv("Total ticks:", stats["total_ticks"], indent=4)
        kv("Rate:", stats["rate"], indent=4)
        kv("Max consecutive:", stats["consecutive_flattens"], indent=4)

    # SOL Dominance
    sd = report.get("sol_dominance", {})
    header("SOL DOMINANCE TRACKING")
    kv("Total SOL ticks:", sd.get("total_sol_ticks", 0))
    kv("V3.6 dom active:", sd.get("v36_dom_active", 0))
    kv("Fallback active:", sd.get("fallback_active", 0))
    kv("Fallback inactive:", sd.get("fallback_inactive", 0))
    kv("Fallback trigger rate:", sd.get("fallback_trigger_rate", "N/A"))

    # Shadow Ledger
    sl = report.get("shadow_ledger", {})
    header("SHADOW LEDGER")
    kv("Total entries:", sl.get("total_entries", 0))
    if sl.get("note"):
        kv("Note:", sl["note"])
    for entry_type, count in sl.get("by_type", {}).items():
        kv(f"  {entry_type}:", count)

    print(f"\n{'=' * W}")
    print(f"  END OF REPORT - Verdict: {report.get('health_verdict', 'UNKNOWN')}")
    print(f"{'=' * W}\n")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="HMATS Paper Run Health Monitor")
    parser.add_argument("--hours", type=float, default=None, help="Look back N hours")
    parser.add_argument("--session", type=str, default=None, help="Specific session ID (YYYYMMDD_HHMMSS)")
    parser.add_argument("--json", action="store_true", help="Output as JSON instead of pretty-print")
    args = parser.parse_args()

    print(f"[Monitor] Loading proof logs from: {PROOF_LOG_DIR}")
    proof_entries = load_proof_logs(hours=args.hours, session=args.session)
    print(f"[Monitor] Loaded {len(proof_entries)} proof log entries")

    print(f"[Monitor] Loading shadow ledger from: {SHADOW_LEDGER_DIR}")
    ledger_entries = load_shadow_ledger(hours=args.hours)
    print(f"[Monitor] Loaded {len(ledger_entries)} ledger entries")

    report = generate_health_report(proof_entries, ledger_entries)

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print_report(report)


if __name__ == "__main__":
    main()
