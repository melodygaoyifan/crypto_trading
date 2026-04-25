"""
Gate-rejection forensic analysis.

Reads `data/shadow_ledger/ledger_*.jsonl` and `data/trade_attribution.jsonl`
to answer the question that no static audit can: "why did the system make
only N trades in the last M days?"

Output:

1. Top-N rejection reasons (Pareto): which gate is killing the most trades?
2. Per-asset / per-regime / per-mode breakdown of the top blocker.
3. Per-strategy fire rate: which strategies were even SELECTED, vs always
   rejected pre-fusion?
4. DECIDE-agent contribution audit: when DRL is ACTIVE, did its direction
   actually drive any of the gate-passed intents?
5. The "1 trade" anatomy: for each successful fill, what was different about
   that tick's signals vs the 99 nearest rejections?

Usage:
    python -X utf8 scripts/gate_rejection_analysis.py [--days N] [--asset BTC]

[P41 2026-04-25] Added after operator reported 30-day live run = 1 fill,
241 gate rejections, despite multiple audits "fixing" the wiring. Static
audits look at code; this one looks at runtime decisions.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
LEDGER_DIR = REPO_ROOT / "data" / "shadow_ledger"
ATTRIBUTION_PATH = REPO_ROOT / "data" / "trade_attribution.jsonl"


def _load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _iter_ledger(days: int) -> Iterable[Dict[str, Any]]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    if not LEDGER_DIR.exists():
        return
    for ledger_file in sorted(LEDGER_DIR.glob("ledger_*.jsonl")):
        # Filename: ledger_YYYYMMDD.jsonl
        try:
            file_date_str = ledger_file.stem.replace("ledger_", "")
            file_date = datetime.strptime(file_date_str, "%Y%m%d").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if file_date < cutoff - timedelta(days=1):
            continue
        for entry in _load_jsonl(ledger_file):
            ts = entry.get("timestamp")
            if not ts:
                continue
            try:
                entry_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if entry_dt.tzinfo is None:
                    entry_dt = entry_dt.replace(tzinfo=timezone.utc)
            except (ValueError, AttributeError):
                continue
            if entry_dt < cutoff:
                continue
            yield entry


def _bucket_reason(reason: str) -> str:
    """Group similar rejection reasons into broader buckets."""
    if not reason:
        return "<empty>"
    reason_lower = reason.lower()
    if "weekend" in reason_lower:
        return "WEEKEND_GATE"
    if "all_conflict_flat" in reason_lower or "signal_conflict" in reason_lower:
        return "ALL_CONFLICT_FLAT"
    if "alpha gate" in reason_lower or "alpha_gate" in reason_lower or "min_alpha" in reason_lower:
        return "ALPHA_GATE"
    if "quiet_accumulation" in reason_lower:
        return "QUIET_ACCUMULATION"
    if "best_of_n_hold" in reason_lower or "best of n hold" in reason_lower:
        return "BEST_OF_N_HOLD"
    if "black_swan" in reason_lower or "black swan" in reason_lower:
        return "BLACK_SWAN"
    if "no_trade" in reason_lower or "no trade" in reason_lower:
        return "NO_TRADE_MODE"
    if "risk veto" in reason_lower or "risk_veto" in reason_lower:
        return "RISK_VETO"
    if "force.*flat" in reason_lower or "forced flat" in reason_lower:
        return "FORCED_FLAT"
    if "circuit" in reason_lower:
        return "CIRCUIT_BREAKER"
    if "stale" in reason_lower:
        return "STALE_DATA"
    if "liquidity" in reason_lower:
        return "LIQUIDITY"
    if "cooldown" in reason_lower:
        return "COOLDOWN"
    if "drawdown" in reason_lower:
        return "DRAWDOWN"
    if "thesis" in reason_lower:
        return "THESIS_BUDGET"
    if "tranche" in reason_lower:
        return "TRANCHE"
    return reason[:60]  # raw, truncated


def analyze(days: int = 30, asset_filter: Optional[str] = None) -> Dict[str, Any]:
    rejects: List[Dict[str, Any]] = []
    fills: List[Dict[str, Any]] = []
    other: Counter = Counter()

    for entry in _iter_ledger(days):
        et = entry.get("entry_type", "")
        if asset_filter and entry.get("asset") != asset_filter:
            continue
        if et == "GATE_REJECT":
            rejects.append(entry)
        elif et in ("FILL", "ORDER_FILLED", "EXECUTION"):
            fills.append(entry)
        else:
            other[et] += 1

    # 1. Pareto of rejection reasons (bucketed)
    reason_counter: Counter = Counter()
    raw_reason_counter: Counter = Counter()
    for r in rejects:
        raw = (r.get("data") or {}).get("rejection_reason", "")
        reason_counter[_bucket_reason(raw)] += 1
        raw_reason_counter[raw[:80]] += 1

    # 2. Per-asset breakdown of TOP bucket
    top_bucket = reason_counter.most_common(1)[0][0] if reason_counter else None
    asset_breakdown: Counter = Counter()
    regime_breakdown: Counter = Counter()
    mode_breakdown: Counter = Counter()
    strategy_breakdown: Counter = Counter()
    for r in rejects:
        raw = (r.get("data") or {}).get("rejection_reason", "")
        if _bucket_reason(raw) != top_bucket:
            continue
        asset_breakdown[r.get("asset", "?")] += 1
        details = (r.get("data") or {}).get("gate_details") or {}
        regime_breakdown[details.get("regime", "?")] += 1
        mode_breakdown[details.get("mode", "?")] += 1
        strategy_breakdown[details.get("strategy", "?")] += 1

    # 3. DRL contribution audit
    drl_observations = []
    for r in rejects + fills:
        details = (r.get("data") or {}).get("gate_details") or {}
        if "drl_action" in details or "drl_confidence" in details:
            drl_observations.append({
                "ts": r.get("timestamp"),
                "asset": r.get("asset"),
                "type": r.get("entry_type"),
                "drl_dir": details.get("drl_action"),
                "drl_conf": details.get("drl_confidence"),
                "passed": r.get("entry_type") != "GATE_REJECT",
            })
    drl_avg_dir = (
        sum(abs(d["drl_dir"]) for d in drl_observations if d["drl_dir"] is not None)
        / max(1, sum(1 for d in drl_observations if d["drl_dir"] is not None))
    )
    drl_avg_conf = (
        sum(d["drl_conf"] for d in drl_observations if d["drl_conf"] is not None)
        / max(1, sum(1 for d in drl_observations if d["drl_conf"] is not None))
    )

    # 4. Strategy fire rate
    strategy_seen: Counter = Counter()
    strategy_fill: Counter = Counter()
    for r in rejects + fills:
        details = (r.get("data") or {}).get("gate_details") or {}
        s = details.get("strategy", "?")
        strategy_seen[s] += 1
        if r.get("entry_type") in ("FILL", "ORDER_FILLED", "EXECUTION"):
            strategy_fill[s] += 1

    # 5b. STALE_DATA breakdown (P43 — needs data_age + stale_sources fields)
    stale_sources_pareto = Counter()
    stale_age_buckets = Counter()
    stale_orderbook_status = Counter()
    stale_freshness_mode = Counter()
    n_stale_with_diag_fields = 0
    for r in rejects:
        raw = (r.get("data") or {}).get("rejection_reason", "") or ""
        if "stale" not in raw.lower():
            continue
        details = (r.get("data") or {}).get("gate_details") or {}
        # Only count entries that have the P43 enrichment fields populated
        if details.get("data_age_seconds") is not None or details.get("stale_sources"):
            n_stale_with_diag_fields += 1
        srcs = details.get("stale_sources") or []
        if isinstance(srcs, list) and srcs:
            for s in srcs:
                stale_sources_pareto[s] += 1
        else:
            stale_sources_pareto["<unknown — no diag fields, pre-P43 ledger>"] += 1
        age = details.get("data_age_seconds")
        if age is not None:
            try:
                age_f = float(age)
                if age_f < 60:
                    stale_age_buckets["0-60s"] += 1
                elif age_f < 300:
                    stale_age_buckets["60-300s (1-5min)"] += 1
                elif age_f < 1800:
                    stale_age_buckets["300-1800s (5-30min)"] += 1
                elif age_f < 7200:
                    stale_age_buckets["1800-7200s (30min-2h)"] += 1
                else:
                    stale_age_buckets[">7200s (>2h)"] += 1
            except (ValueError, TypeError):
                pass
        ob_stale = details.get("orderbook_stale")
        ob_fb = details.get("orderbook_fallback_reason", "")
        if ob_stale is not None:
            stale_orderbook_status[f"ob_stale={ob_stale}, fb={ob_fb!r}"] += 1
        fm = details.get("freshness_mode", "")
        if fm:
            stale_freshness_mode[fm] += 1

    # 5c. STRUCTURE-block breakdown (P44 — needs structure_* gate_details fields)
    struct_side_pareto = Counter()
    struct_gap_buckets = Counter()
    struct_strategy_pareto = Counter()
    struct_edge_vs_min = Counter()
    n_struct_with_diag = 0
    for r in rejects:
        raw = (r.get("data") or {}).get("rejection_reason", "") or ""
        if "structure" not in raw.lower():
            continue
        details = (r.get("data") or {}).get("gate_details") or {}
        side = details.get("structure_side")
        if side:
            n_struct_with_diag += 1
            struct_side_pareto[side] += 1
        gap = details.get("structure_gap_bps")
        gap_limit = details.get("structure_gap_limit_bps")
        if gap is not None and gap_limit:
            try:
                gap_f = float(gap)
                limit_f = float(gap_limit)
                ratio = gap_f / max(1e-9, limit_f)
                if ratio < 1.5:
                    struct_gap_buckets["near (gap < 1.5×limit)"] += 1
                elif ratio < 3.0:
                    struct_gap_buckets["mid (1.5-3×limit)"] += 1
                elif ratio < 10.0:
                    struct_gap_buckets["far (3-10×limit)"] += 1
                else:
                    struct_gap_buckets["very far (>10×limit)"] += 1
            except (ValueError, TypeError):
                pass
        sid = details.get("structure_strategy_id")
        if sid:
            struct_strategy_pareto[sid] += 1
        edge = details.get("structure_edge_bps")
        min_edge = details.get("structure_min_edge_bps")
        if edge is not None and min_edge is not None:
            try:
                if float(edge) >= float(min_edge):
                    struct_edge_vs_min["edge >= min_edge (gap is the issue)"] += 1
                else:
                    struct_edge_vs_min["edge < min_edge (insufficient alpha)"] += 1
            except (ValueError, TypeError):
                pass

    # 5. Fill anatomy (compare each fill against its 5 nearest rejections by time)
    fill_anatomy = []
    for f in fills[:10]:  # limit to 10 fills
        f_ts = f.get("timestamp", "")
        f_details = (f.get("data") or {}).get("gate_details") or {}
        nearest = sorted(
            rejects,
            key=lambda r: abs(_ts_diff(f_ts, r.get("timestamp", ""))),
        )[:5]
        fill_anatomy.append({
            "fill_ts": f_ts,
            "fill_asset": f.get("asset"),
            "fill_strategy": f_details.get("strategy"),
            "fill_alpha": f_details.get("alpha_estimated_bps"),
            "fill_threshold": f_details.get("alpha_threshold_bps"),
            "fill_drl_dir": f_details.get("drl_action"),
            "fill_drl_conf": f_details.get("drl_confidence"),
            "nearest_rejects": [
                {
                    "ts": n.get("timestamp"),
                    "reason": _bucket_reason((n.get("data") or {}).get("rejection_reason", "")),
                }
                for n in nearest
            ],
        })

    return {
        "window_days": days,
        "asset_filter": asset_filter,
        "total_rejects": len(rejects),
        "total_fills": len(fills),
        "other_event_types": dict(other),
        "rejection_pareto_bucketed": reason_counter.most_common(20),
        "rejection_pareto_raw": raw_reason_counter.most_common(15),
        "top_bucket": top_bucket,
        "top_bucket_by_asset": asset_breakdown.most_common(),
        "top_bucket_by_regime": regime_breakdown.most_common(),
        "top_bucket_by_mode": mode_breakdown.most_common(),
        "top_bucket_by_strategy": strategy_breakdown.most_common(),
        "drl_observations": {
            "n_with_drl_field": len(drl_observations),
            "avg_abs_direction": round(drl_avg_dir, 3),
            "avg_confidence": round(drl_avg_conf, 3),
            "n_passed_with_strong_drl": sum(
                1 for d in drl_observations
                if d["passed"] and d["drl_dir"] is not None and abs(d["drl_dir"]) >= 0.5
            ),
            "n_rejected_with_strong_drl": sum(
                1 for d in drl_observations
                if not d["passed"] and d["drl_dir"] is not None and abs(d["drl_dir"]) >= 0.5
            ),
        },
        "strategy_fire_rate": {
            s: {
                "seen": strategy_seen[s],
                "filled": strategy_fill[s],
                "fill_pct": round(100 * strategy_fill[s] / max(1, strategy_seen[s]), 2),
            }
            for s in strategy_seen
        },
        "fill_anatomy": fill_anatomy,
        "stale_data_breakdown": {
            "n_stale_rejections": sum(1 for r in rejects
                                       if "stale" in ((r.get("data") or {}).get("rejection_reason", "") or "").lower()),
            "n_with_diag_fields_p43": n_stale_with_diag_fields,
            "by_source": stale_sources_pareto.most_common(),
            "by_age_bucket": stale_age_buckets.most_common(),
            "by_orderbook_status": stale_orderbook_status.most_common(),
            "by_freshness_mode": stale_freshness_mode.most_common(),
        },
        "structure_breakdown": {
            "n_struct_rejections": sum(1 for r in rejects
                                        if "structure" in ((r.get("data") or {}).get("rejection_reason", "") or "").lower()),
            "n_with_diag_fields_p44": n_struct_with_diag,
            "by_side": struct_side_pareto.most_common(),
            "by_gap_bucket": struct_gap_buckets.most_common(),
            "by_strategy": struct_strategy_pareto.most_common(),
            "by_edge_vs_min": struct_edge_vs_min.most_common(),
        },
    }


def _ts_diff(a: str, b: str) -> float:
    """Diff in seconds between two ISO timestamps. Returns inf on parse error."""
    try:
        ta = datetime.fromisoformat(a.replace("Z", "+00:00"))
        tb = datetime.fromisoformat(b.replace("Z", "+00:00"))
        if ta.tzinfo is None:
            ta = ta.replace(tzinfo=timezone.utc)
        if tb.tzinfo is None:
            tb = tb.replace(tzinfo=timezone.utc)
        return (ta - tb).total_seconds()
    except Exception:
        return float("inf")


def render(report: Dict[str, Any]) -> str:
    lines = []
    lines.append("=" * 78)
    lines.append(
        f"GATE REJECTION ANALYSIS — last {report['window_days']} days"
        + (f" — asset={report['asset_filter']}" if report['asset_filter'] else "")
    )
    lines.append("=" * 78)
    lines.append("")
    lines.append(f"Total fills:     {report['total_fills']}")
    lines.append(f"Total rejects:   {report['total_rejects']}")
    lines.append(f"Other events:    {report['other_event_types']}")
    if report['total_rejects'] + report['total_fills']:
        fill_pct = 100 * report['total_fills'] / max(1, (report['total_rejects'] + report['total_fills']))
        lines.append(f"Fill rate:       {fill_pct:.2f}% ({report['total_fills']}/{report['total_fills']+report['total_rejects']})")
    lines.append("")

    lines.append("-" * 78)
    lines.append("REJECTION PARETO (bucketed)")
    lines.append("-" * 78)
    for reason, count in report['rejection_pareto_bucketed']:
        pct = 100 * count / max(1, report['total_rejects'])
        bar = "#" * min(50, int(pct / 2))
        lines.append(f"  {reason:30} {count:5}  {pct:5.1f}%  {bar}")
    lines.append("")

    lines.append("-" * 78)
    lines.append(f"TOP BLOCKER ({report['top_bucket']}) — breakdown")
    lines.append("-" * 78)
    lines.append(f"  By asset:    {dict(report['top_bucket_by_asset'])}")
    lines.append(f"  By regime:   {dict(report['top_bucket_by_regime'])}")
    lines.append(f"  By mode:     {dict(report['top_bucket_by_mode'])}")
    lines.append(f"  By strategy: {dict(report['top_bucket_by_strategy'])}")
    lines.append("")

    lines.append("-" * 78)
    lines.append("DRL CONTRIBUTION AUDIT")
    lines.append("-" * 78)
    drl = report['drl_observations']
    lines.append(f"  Ledger entries with drl_action field: {drl['n_with_drl_field']}")
    lines.append(f"  Avg |drl_direction|: {drl['avg_abs_direction']}")
    lines.append(f"  Avg drl_confidence:  {drl['avg_confidence']}")
    lines.append(f"  Strong DRL signals that PASSED: {drl['n_passed_with_strong_drl']}")
    lines.append(f"  Strong DRL signals that REJECTED: {drl['n_rejected_with_strong_drl']}")
    if drl['n_rejected_with_strong_drl'] > drl['n_passed_with_strong_drl'] * 5:
        lines.append("  ⚠ DRL strong signals are mostly being REJECTED downstream — DECIDE authority may not be effective")
    lines.append("")

    lines.append("-" * 78)
    lines.append("STRATEGY FIRE RATE")
    lines.append("-" * 78)
    for s, stats in sorted(report['strategy_fire_rate'].items(),
                           key=lambda kv: -kv[1]['seen']):
        lines.append(f"  {s:20} seen={stats['seen']:5}  filled={stats['filled']:3}  rate={stats['fill_pct']:5.2f}%")
    lines.append("")

    lines.append("-" * 78)
    lines.append("STALE_DATA BREAKDOWN (P43 — needs ledger entries with diag fields)")
    lines.append("-" * 78)
    sd = report.get('stale_data_breakdown', {})
    lines.append(f"  Total stale rejects: {sd.get('n_stale_rejections', 0)}")
    lines.append(f"  With P43 diag fields: {sd.get('n_with_diag_fields_p43', 0)}")
    if sd.get('by_source'):
        lines.append("  By stale source:")
        for src, count in sd['by_source']:
            lines.append(f"    {src:35} {count}")
    if sd.get('by_age_bucket'):
        lines.append("  By data_age_seconds bucket:")
        for bucket, count in sd['by_age_bucket']:
            lines.append(f"    {bucket:30} {count}")
    if sd.get('by_orderbook_status'):
        lines.append("  By orderbook status:")
        for status, count in sd['by_orderbook_status']:
            lines.append(f"    {status:50} {count}")
    if sd.get('by_freshness_mode'):
        lines.append(f"  Freshness modes: {dict(sd['by_freshness_mode'])}")
    lines.append("")

    lines.append("-" * 78)
    lines.append("STRUCTURE-BLOCK BREAKDOWN (P44 — needs ledger entries with diag fields)")
    lines.append("-" * 78)
    sb = report.get('structure_breakdown', {})
    lines.append(f"  Total structure rejects: {sb.get('n_struct_rejections', 0)}")
    lines.append(f"  With P44 diag fields: {sb.get('n_with_diag_fields_p44', 0)}")
    if sb.get('by_side'):
        lines.append("  By side:")
        for side, count in sb['by_side']:
            lines.append(f"    {side:10} {count}")
    if sb.get('by_gap_bucket'):
        lines.append("  By gap distance (vs configured limit):")
        for bucket, count in sb['by_gap_bucket']:
            lines.append(f"    {bucket:30} {count}")
    if sb.get('by_strategy'):
        lines.append("  By strategy:")
        for s, count in sb['by_strategy']:
            lines.append(f"    {s:25} {count}")
    if sb.get('by_edge_vs_min'):
        lines.append("  Net edge vs min_edge requirement:")
        for status, count in sb['by_edge_vs_min']:
            lines.append(f"    {status:50} {count}")
    lines.append("")

    lines.append("-" * 78)
    lines.append("FILL ANATOMY (each fill vs nearest 5 rejects)")
    lines.append("-" * 78)
    if not report['fill_anatomy']:
        lines.append("  (no fills in window)")
    for f in report['fill_anatomy']:
        lines.append(f"  Fill: {f['fill_asset']} {f['fill_ts']}")
        lines.append(f"        strategy={f['fill_strategy']} alpha={f['fill_alpha']} thresh={f['fill_threshold']}")
        lines.append(f"        drl_dir={f['fill_drl_dir']} drl_conf={f['fill_drl_conf']}")
        lines.append(f"        Nearest rejects: {[r['reason'] for r in f['nearest_rejects']]}")
        lines.append("")

    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--days", type=int, default=30, help="Lookback window")
    p.add_argument("--asset", type=str, default=None, help="Filter to one asset")
    p.add_argument("--json", action="store_true", help="Output raw JSON instead of rendered report")
    args = p.parse_args()

    report = analyze(days=args.days, asset_filter=args.asset)
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(render(report))


if __name__ == "__main__":
    main()
