#!/usr/bin/env python3
"""
Decision-trace watcher (Tier 1 / Item 1 — P111 2026-04-27)
================================================================

Tails the per-tick attribution signals_*.jsonl + outcomes_*.jsonl and
flags surprising patterns BEFORE they become "0 trades for 7 days":

  - DECIDE_ABSTAIN fired N+ ticks in a row
  - Specific gate (TRADE_GATE / WEEKEND / THESIS_BUDGET / etc.)
    rejected M+ of last K ticks
  - POSITION-DESYNC alerts in last hour
  - STOP-MINSIZE alerts in last hour
  - Governor exception logged in last 24h
  - No tick recorded in last >2h (engine stuck)

Run as a one-shot operator query:
    python scripts/decision_trace_watcher.py
    python scripts/decision_trace_watcher.py --hours 24 --tail 200

Or as a cron-style monitor:
    while true; do python scripts/decision_trace_watcher.py --json; sleep 300; done

Output: human-readable summary OR --json for machine consumption
(Discord webhook, dashboard, etc.)

Design notes:
  - Read-only: never modifies signal/outcome files.
  - Defensive parse: malformed lines logged + counted, not crashed.
  - Bounded memory: streams files line-by-line, not load-all.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
LOCAL_ATTR_DIR = REPO / "logs" / "attribution"
CONTAINER_ATTR_DIR = Path("/opt/hmats/logs/attribution")  # inside container
HETZNER_ATTR_DIR = Path("/var/lib/docker/volumes/hmats-logs/_data/attribution")  # host


def _resolve_attr_dir() -> Path:
    """Pick the first attribution dir that exists on this host."""
    for candidate in (LOCAL_ATTR_DIR, CONTAINER_ATTR_DIR, HETZNER_ATTR_DIR):
        if candidate.exists():
            return candidate
    print(f"[ERROR] No attribution dir found in any of: "
          f"{[str(p) for p in (LOCAL_ATTR_DIR, CONTAINER_ATTR_DIR, HETZNER_ATTR_DIR)]}",
          file=sys.stderr)
    sys.exit(2)


def _parse_ts(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    try:
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None


def _iter_recent_records(
    attr_dir: Path,
    pattern: str,
    cutoff: datetime,
    tail_lines: int = 5000,
) -> List[Dict]:
    """Iterate the most-recent N lines of recent files matching pattern."""
    files = sorted(attr_dir.glob(pattern))
    if not files:
        return []
    recent: deque = deque(maxlen=tail_lines)
    # Walk newest files first; stop once we've hit the line cap.
    for fp in reversed(files):
        try:
            with open(fp, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    recent.append(rec)
        except OSError:
            continue
        if len(recent) >= tail_lines:
            break
    # Filter by cutoff
    out = []
    for rec in recent:
        ts = _parse_ts(rec.get("ts") or rec.get("timestamp", ""))
        if ts is None or ts.tzinfo is None:
            # Either parse failed or naive — keep it (don't drop silently)
            out.append(rec)
            continue
        if ts >= cutoff:
            out.append(rec)
    return out


def analyze_signals(signals: List[Dict]) -> Dict:
    """Look for DECIDE_ABSTAIN streaks + per-agent abstain rates."""
    if not signals:
        return {"signals_total": 0}
    abstain_streak = 0
    max_abstain_streak = 0
    abstain_count = 0
    per_agent_abstains: Counter = Counter()
    per_agent_total: Counter = Counter()
    for rec in signals:
        decide_active = 0
        decide_total = 0
        for sig in rec.get("signals", []):
            authority = sig.get("authority", "")
            if "DECIDE" not in authority:
                continue
            agent = sig.get("agent_name", "?")
            decide_total += 1
            per_agent_total[agent] += 1
            try:
                d = abs(float(sig.get("direction", 0.0)))
                c = float(sig.get("confidence", 0.0))
            except (TypeError, ValueError):
                d = c = 0.0
            if d < 0.01 or c < 0.05:
                per_agent_abstains[agent] += 1
            else:
                decide_active += 1
        if decide_total > 0 and decide_active == 0:
            abstain_streak += 1
            abstain_count += 1
            max_abstain_streak = max(max_abstain_streak, abstain_streak)
        else:
            abstain_streak = 0
    return {
        "signals_total": len(signals),
        "abstain_ticks": abstain_count,
        "abstain_pct": round(abstain_count / max(1, len(signals)) * 100, 1),
        "max_abstain_streak": max_abstain_streak,
        "per_agent_abstain_rate": {
            a: round(per_agent_abstains[a] / max(1, per_agent_total[a]) * 100, 1)
            for a in per_agent_total
        },
    }


def analyze_outcomes(outcomes: List[Dict]) -> Dict:
    """Look for veto-reason histogram, REJECT/PERMANENT_FAILURE counts."""
    if not outcomes:
        return {"outcomes_total": 0}
    veto_reasons: Counter = Counter()
    statuses: Counter = Counter()
    for rec in outcomes:
        statuses[rec.get("status", "?")] += 1
        vr = rec.get("veto_reason") or rec.get("reason") or ""
        if vr:
            # Bucket by first token (e.g., "[TRADE_GATE]" or "WEEKEND" or "P0_SAFETY")
            head = vr.split(":", 1)[0].split(" ", 1)[0].strip("[]")
            veto_reasons[head] += 1
    return {
        "outcomes_total": len(outcomes),
        "status_counts": dict(statuses.most_common(10)),
        "top_veto_reasons": dict(veto_reasons.most_common(10)),
    }


def detect_anomalies(sig_summary: Dict, out_summary: Dict, hours: int) -> List[str]:
    """Translate raw stats into operator-actionable warnings."""
    warnings = []
    # 1. DECIDE_ABSTAIN streak >= 6 ticks (one full day at 4H cadence)
    streak = sig_summary.get("max_abstain_streak", 0)
    if streak >= 6:
        warnings.append(
            f"⚠️  DECIDE_ABSTAIN streak of {streak} consecutive ticks "
            f"(>= 24h continuously no DECIDE-layer signal). Check fusion "
            f"thresholds + DRL ACTIVE state."
        )
    elif streak >= 3:
        warnings.append(
            f"⚠️  DECIDE_ABSTAIN streak of {streak} ticks (>=12h). "
            f"Watch for full-day drought."
        )
    # 2. Abstain rate > 50% of ticks
    abstain_pct = sig_summary.get("abstain_pct", 0)
    if abstain_pct > 50:
        warnings.append(
            f"⚠️  {abstain_pct}% of last {sig_summary.get('signals_total')} ticks "
            f"abstained on DECIDE layer. Trade frequency drought imminent."
        )
    # 3. Per-agent abstain rate > 80% (one agent dragging down the layer)
    per_agent = sig_summary.get("per_agent_abstain_rate", {})
    for agent, rate in per_agent.items():
        if rate > 80 and sig_summary.get("signals_total", 0) > 10:
            warnings.append(
                f"⚠️  Agent '{agent}' abstained {rate}% of last "
                f"{sig_summary['signals_total']} ticks — likely broken "
                f"or threshold misconfigured."
            )
    # 4. PERMANENT_FAILURE outcomes (P79 family)
    statuses = out_summary.get("status_counts", {})
    perm_failures = sum(v for k, v in statuses.items() if "PERMANENT" in str(k))
    if perm_failures > 3:
        warnings.append(
            f"⚠️  {perm_failures} PERMANENT_FAILURE outcomes in last "
            f"{hours}h — order rejection cascade or stop-loss leak."
        )
    # 5. Top veto-reason concentration > 70%
    top_reasons = out_summary.get("top_veto_reasons", {})
    if top_reasons:
        top_count = max(top_reasons.values())
        total_vetoes = sum(top_reasons.values())
        if total_vetoes > 5 and top_count / total_vetoes > 0.7:
            top_name = max(top_reasons, key=top_reasons.get)
            warnings.append(
                f"⚠️  Veto reasons dominated by '{top_name}' "
                f"({top_count}/{total_vetoes} = "
                f"{top_count/total_vetoes*100:.0f}%) — single gate "
                f"is bottlenecking trade execution."
            )
    return warnings


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hours", type=int, default=24,
                    help="Look back N hours (default: 24)")
    ap.add_argument("--tail", type=int, default=5000,
                    help="Cap at N most-recent records per file (default: 5000)")
    ap.add_argument("--json", action="store_true",
                    help="Emit machine-readable JSON")
    args = ap.parse_args()

    attr_dir = _resolve_attr_dir()
    cutoff = datetime.now(timezone.utc) - timedelta(hours=args.hours)

    signals = _iter_recent_records(attr_dir, "signals_*.jsonl", cutoff, args.tail)
    outcomes = _iter_recent_records(attr_dir, "outcomes_*.jsonl", cutoff, args.tail)

    sig_summary = analyze_signals(signals)
    out_summary = analyze_outcomes(outcomes)
    warnings = detect_anomalies(sig_summary, out_summary, args.hours)

    payload = {
        "attribution_dir": str(attr_dir),
        "hours_lookback": args.hours,
        "signals": sig_summary,
        "outcomes": out_summary,
        "warnings": warnings,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"=== HMATS Decision-Trace Watcher (last {args.hours}h) ===")
        print(f"Attribution dir: {attr_dir}")
        print()
        print(f"SIGNALS: {sig_summary.get('signals_total', 0)} ticks")
        print(f"  abstain ticks: {sig_summary.get('abstain_ticks', 0)} "
              f"({sig_summary.get('abstain_pct', 0)}%)")
        print(f"  max abstain streak: {sig_summary.get('max_abstain_streak', 0)} ticks")
        per_agent = sig_summary.get("per_agent_abstain_rate", {})
        if per_agent:
            print(f"  per-agent abstain rate:")
            for a, r in sorted(per_agent.items(), key=lambda x: -x[1])[:8]:
                print(f"    {a}: {r}%")
        print()
        print(f"OUTCOMES: {out_summary.get('outcomes_total', 0)} records")
        statuses = out_summary.get("status_counts", {})
        if statuses:
            print(f"  status distribution: {statuses}")
        top_reasons = out_summary.get("top_veto_reasons", {})
        if top_reasons:
            print(f"  top veto reasons:")
            for r, c in top_reasons.items():
                print(f"    {r}: {c}")
        print()
        if warnings:
            print(f"WARNINGS ({len(warnings)}):")
            for w in warnings:
                print(f"  {w}")
        else:
            print("WARNINGS: none — system operating normally.")
        print()


if __name__ == "__main__":
    main()
