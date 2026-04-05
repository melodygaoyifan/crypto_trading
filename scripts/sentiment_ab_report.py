"""
Sentiment A/B keep/kill report from local diagnostics and shadow-ledger artifacts.

Usage:
    python -X utf8 scripts/sentiment_ab_report.py
    python -X utf8 scripts/sentiment_ab_report.py --days 14
    python -X utf8 scripts/sentiment_ab_report.py --json
"""

from __future__ import annotations

import argparse
import ast
import json
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DIAGNOSTICS_DIR = DATA_DIR / "diagnostics"
SHADOW_LEDGER_DIR = DATA_DIR / "shadow_ledger"
STEP15_STATUS = DATA_DIR / "step15_status.json"
UTC = timezone.utc

DEFAULT_DAYS = 30
DEFAULT_ASSOCIATION_MINUTES = 30
MIN_TREATMENT_REALIZED_EVENTS = 5
MIN_REFERENCE_REALIZED_EVENTS = 5

GROUP_TREATMENT = "treatment"
GROUP_CONTROL = "control"
GROUP_BASELINE = "baseline"

VERDICT_KEEP = "KEEP"
VERDICT_KILL = "KILL"
VERDICT_MONITOR = "MONITOR"
VERDICT_INSUFFICIENT_REALIZED_SAMPLE = "INSUFFICIENT_REALIZED_SAMPLE"
VERDICT_INSUFFICIENT_REFERENCE_SAMPLE = "INSUFFICIENT_REFERENCE_SAMPLE"
VERDICT_NO_TREATMENT_EVIDENCE = "NO_TREATMENT_EVIDENCE"


@dataclass
class Verdict:
    status: str
    reason: str
    reference_group: str
    keep_threshold_pct: float
    treatment_realized_events: int
    reference_realized_events: int
    treatment_total_pnl_usd: float
    reference_total_pnl_usd: float
    pnl_delta_pct: Optional[float]


def parse_dt(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(UTC)
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)
    except Exception:
        return None


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, "", "N/A"):
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, "", "N/A"):
            return default
        return int(value)
    except Exception:
        return default


def pct_ratio(num: float, den: float) -> Optional[float]:
    if den <= 0:
        return None
    return num / den


def pct_delta(current: float, reference: float) -> Optional[float]:
    if abs(reference) <= 1e-12:
        return None
    return (current - reference) / abs(reference)


def parse_structured_output(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        value = ast.literal_eval(text)
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_diagnostics(days: int, anchor: Optional[datetime] = None) -> list[dict[str, Any]]:
    anchor_dt = anchor or datetime.now(UTC)
    cutoff = anchor_dt - timedelta(days=days)
    rows: list[dict[str, Any]] = []
    for path in sorted(DIAGNOSTICS_DIR.glob("diag_*.json")):
        mtime = datetime.fromtimestamp(path.stat().st_mtime, UTC)
        if mtime < cutoff:
            continue
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        tick_dt = parse_dt(obj.get("tick_time")) or mtime
        if tick_dt < cutoff:
            continue
        components = obj.get("components", {})
        llm_component = components.get("llm_sentiment", {})
        llm_output = parse_structured_output(llm_component.get("output", {}))
        det_output = parse_structured_output(components.get("det_sentiment", {}).get("output", {}))
        called = bool(llm_component.get("called", False))
        tradeable = bool(llm_component.get("consumed", False))
        source = str(llm_output.get("source", "none") or "none")
        status = str(llm_output.get("status", "unknown") or "unknown")
        if tradeable:
            group = GROUP_TREATMENT
        else:
            group = GROUP_CONTROL
        rows.append(
            {
                "asset": str(obj.get("asset", "")),
                "tick_dt": tick_dt,
                "path": str(path.relative_to(ROOT)),
                "group": group,
                "called": called,
                "tradeable": tradeable,
                "source": source,
                "status": status,
                "headline_count": safe_int(llm_output.get("headlines", 0)),
                "direction": safe_float(llm_output.get("combined_dir", 0.0)),
                "confidence": safe_float(llm_output.get("conf", 0.0)),
                "sentiment_zscore": safe_float(det_output.get("zscore", 0.0)),
                "alpha_gate_passed": bool(obj.get("final_intent", {}).get("alpha_gate_passed", False)),
                "actionable": bool(obj.get("final_intent", {}).get("is_actionable", False)),
                "veto_active": bool(obj.get("final_intent", {}).get("veto_active", False)),
                "final_direction": safe_float(obj.get("final_intent", {}).get("direction", 0.0)),
            }
        )
    rows.sort(key=lambda item: (item["asset"], item["tick_dt"]))
    return rows


def load_fills(days: int, anchor: Optional[datetime] = None) -> list[dict[str, Any]]:
    anchor_dt = anchor or datetime.now(UTC)
    cutoff = anchor_dt - timedelta(days=days)
    rows: list[dict[str, Any]] = []
    for path in sorted(SHADOW_LEDGER_DIR.glob("ledger_*.jsonl")):
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue
        for line in lines:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("entry_type") != "FILL":
                continue
            dt = parse_dt(obj.get("timestamp"))
            if not dt or dt < cutoff:
                continue
            data = obj.get("data", {})
            rows.append(
                {
                    "asset": str(obj.get("asset", "")),
                    "dt": dt,
                    "side": str(data.get("side", "")).upper(),
                    "size": abs(safe_float(data.get("size", 0.0))),
                    "realized_pnl": safe_float(data.get("realized_pnl", 0.0)),
                    "price": safe_float(data.get("price", 0.0)),
                    "path": str(path.relative_to(ROOT)),
                }
            )
    rows.sort(key=lambda item: item["dt"])
    return rows


def associate_diagnostics(
    diagnostics: list[dict[str, Any]],
    max_association_gap: timedelta,
) -> dict[str, tuple[list[datetime], list[dict[str, Any]]]]:
    by_asset: dict[str, tuple[list[datetime], list[dict[str, Any]]]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in diagnostics:
        grouped[row["asset"]].append(row)
    for asset, rows in grouped.items():
        by_asset[asset] = ([row["tick_dt"] for row in rows], rows)
    return by_asset


def match_diag_group(
    diag_index: dict[str, tuple[list[datetime], list[dict[str, Any]]]],
    asset: str,
    dt: datetime,
    max_association_gap: timedelta,
) -> Optional[str]:
    times_rows = diag_index.get(asset)
    if not times_rows:
        return None
    times, rows = times_rows
    idx = bisect_right(times, dt) - 1
    if idx < 0:
        return None
    if dt - times[idx] > max_association_gap:
        return None
    return rows[idx]["group"]


def reconstruct_trade_groups(
    fills: list[dict[str, Any]],
    diag_index: dict[str, tuple[list[datetime], list[dict[str, Any]]]],
    max_association_gap: timedelta,
) -> dict[str, Any]:
    positions: dict[str, float] = defaultdict(float)
    lots: dict[str, list[dict[str, Any]]] = defaultdict(list)

    entry_fills = Counter()
    realized_events = Counter()
    realized_pnl = Counter()
    mixed_realized_events = 0
    mixed_realized_pnl = 0.0
    unattributed_realized_events = 0
    unattributed_realized_pnl = 0.0

    for fill in fills:
        asset = fill["asset"]
        size = fill["size"]
        if size <= 0:
            continue
        fill_sign = 1.0 if fill["side"] == "BUY" else -1.0
        current_position = positions[asset]
        same_direction = current_position == 0 or (current_position > 0 and fill_sign > 0) or (current_position < 0 and fill_sign < 0)

        if same_direction:
            if abs(fill["realized_pnl"]) > 1e-12:
                unattributed_realized_events += 1
                unattributed_realized_pnl += fill["realized_pnl"]
            group = match_diag_group(diag_index, asset, fill["dt"], max_association_gap) or GROUP_BASELINE
            entry_fills[group] += 1
            lots[asset].append({"direction": fill_sign, "remaining": size, "group": group})
            positions[asset] = current_position + fill_sign * size
            continue

        remaining = size
        closing_direction = 1.0 if current_position > 0 else -1.0
        group_pnl = Counter()
        asset_lots = lots[asset]
        while remaining > 1e-12 and asset_lots and asset_lots[0]["direction"] == closing_direction:
            lot = asset_lots[0]
            used = min(lot["remaining"], remaining)
            if abs(fill["realized_pnl"]) > 1e-12:
                group_pnl[lot["group"]] += fill["realized_pnl"] * (used / size)
            lot["remaining"] -= used
            remaining -= used
            if lot["remaining"] <= 1e-12:
                asset_lots.pop(0)

        if group_pnl:
            if len(group_pnl) == 1:
                only_group = next(iter(group_pnl))
                realized_events[only_group] += 1
            else:
                mixed_realized_events += 1
                mixed_realized_pnl += sum(group_pnl.values())
            for group, pnl_share in group_pnl.items():
                realized_pnl[group] += pnl_share
        elif abs(fill["realized_pnl"]) > 1e-12:
            unattributed_realized_events += 1
            unattributed_realized_pnl += fill["realized_pnl"]

        positions[asset] = current_position + fill_sign * size

        if abs(positions[asset]) > 1e-12 and remaining > 1e-12:
            group = match_diag_group(diag_index, asset, fill["dt"], max_association_gap) or GROUP_BASELINE
            entry_fills[group] += 1
            lots[asset].append({"direction": fill_sign, "remaining": remaining, "group": group})

    return {
        "entry_fills": dict(entry_fills),
        "realized_events": dict(realized_events),
        "realized_pnl_usd": dict(realized_pnl),
        "mixed_realized_events": mixed_realized_events,
        "mixed_realized_pnl_usd": mixed_realized_pnl,
        "unattributed_realized_events": unattributed_realized_events,
        "unattributed_realized_pnl_usd": unattributed_realized_pnl,
        "all_fills": len(fills),
        "realized_fill_events_total": sum(1 for fill in fills if abs(fill["realized_pnl"]) > 1e-12),
    }


def summarize_diagnostics(diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(diagnostics)
    called = sum(1 for row in diagnostics if row["called"])
    treatment = [row for row in diagnostics if row["group"] == GROUP_TREATMENT]
    control = [row for row in diagnostics if row["group"] == GROUP_CONTROL]

    def _group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "count": len(rows),
            "alpha_gate_pass_rate": pct_ratio(sum(1 for row in rows if row["alpha_gate_passed"]), len(rows)),
            "actionable_rate": pct_ratio(sum(1 for row in rows if row["actionable"]), len(rows)),
            "nonzero_direction_rate": pct_ratio(sum(1 for row in rows if abs(row["final_direction"]) > 1e-12), len(rows)),
            "avg_confidence": (sum(row["confidence"] for row in rows) / len(rows)) if rows else None,
        }

    by_asset: dict[str, dict[str, Any]] = {}
    assets = sorted({row["asset"] for row in diagnostics})
    for asset in assets:
        asset_rows = [row for row in diagnostics if row["asset"] == asset]
        by_asset[asset] = {
            "total": len(asset_rows),
            "tradeable": sum(1 for row in asset_rows if row["tradeable"]),
            "tradeable_rate": pct_ratio(sum(1 for row in asset_rows if row["tradeable"]), len(asset_rows)),
        }

    return {
        "total": total,
        "called": called,
        "called_rate": pct_ratio(called, total),
        "treatment_count": len(treatment),
        "control_count": len(control),
        "treatment_rate": pct_ratio(len(treatment), total),
        "source_mix": dict(Counter(row["source"] for row in diagnostics)),
        "status_mix": dict(Counter(row["status"] for row in diagnostics)),
        "groups": {
            GROUP_TREATMENT: _group_summary(treatment),
            GROUP_CONTROL: _group_summary(control),
        },
        "by_asset": by_asset,
    }


def choose_reference_group(trade_summary: dict[str, Any]) -> str:
    realized_events = trade_summary["realized_events"]
    baseline_count = safe_int(realized_events.get(GROUP_BASELINE, 0))
    control_count = safe_int(realized_events.get(GROUP_CONTROL, 0))
    if baseline_count >= MIN_REFERENCE_REALIZED_EVENTS:
        return GROUP_BASELINE
    if control_count >= MIN_REFERENCE_REALIZED_EVENTS:
        return GROUP_CONTROL
    return GROUP_BASELINE if baseline_count >= control_count else GROUP_CONTROL


def build_verdict(trade_summary: dict[str, Any], diag_summary: dict[str, Any]) -> Verdict:
    realized_events = trade_summary["realized_events"]
    realized_pnl = trade_summary["realized_pnl_usd"]

    treatment_count = safe_int(realized_events.get(GROUP_TREATMENT, 0))
    treatment_total = safe_float(realized_pnl.get(GROUP_TREATMENT, 0.0))
    reference_group = choose_reference_group(trade_summary)
    reference_count = safe_int(realized_events.get(reference_group, 0))
    reference_total = safe_float(realized_pnl.get(reference_group, 0.0))
    delta = pct_delta(treatment_total, reference_total)

    if safe_int(diag_summary["treatment_count"], 0) == 0:
        return Verdict(
            status=VERDICT_NO_TREATMENT_EVIDENCE,
            reason="No tradeable llm_sentiment diagnostics observed in the analysis window.",
            reference_group=reference_group,
            keep_threshold_pct=0.03,
            treatment_realized_events=treatment_count,
            reference_realized_events=reference_count,
            treatment_total_pnl_usd=treatment_total,
            reference_total_pnl_usd=reference_total,
            pnl_delta_pct=delta,
        )

    if treatment_count < MIN_TREATMENT_REALIZED_EVENTS:
        return Verdict(
            status=VERDICT_INSUFFICIENT_REALIZED_SAMPLE,
            reason=f"treatment_realized_events={treatment_count} < {MIN_TREATMENT_REALIZED_EVENTS}",
            reference_group=reference_group,
            keep_threshold_pct=0.03,
            treatment_realized_events=treatment_count,
            reference_realized_events=reference_count,
            treatment_total_pnl_usd=treatment_total,
            reference_total_pnl_usd=reference_total,
            pnl_delta_pct=delta,
        )

    if reference_count < MIN_REFERENCE_REALIZED_EVENTS:
        return Verdict(
            status=VERDICT_INSUFFICIENT_REFERENCE_SAMPLE,
            reason=f"{reference_group}_realized_events={reference_count} < {MIN_REFERENCE_REALIZED_EVENTS}",
            reference_group=reference_group,
            keep_threshold_pct=0.03,
            treatment_realized_events=treatment_count,
            reference_realized_events=reference_count,
            treatment_total_pnl_usd=treatment_total,
            reference_total_pnl_usd=reference_total,
            pnl_delta_pct=delta,
        )

    if delta is not None and delta > 0.03:
        return Verdict(
            status=VERDICT_KEEP,
            reason=f"Treatment PnL delta vs {reference_group} = {delta * 100:.2f}% > 3.00%",
            reference_group=reference_group,
            keep_threshold_pct=0.03,
            treatment_realized_events=treatment_count,
            reference_realized_events=reference_count,
            treatment_total_pnl_usd=treatment_total,
            reference_total_pnl_usd=reference_total,
            pnl_delta_pct=delta,
        )

    if delta is not None and delta < 0.0:
        return Verdict(
            status=VERDICT_KILL,
            reason=f"Treatment PnL delta vs {reference_group} = {delta * 100:.2f}% < 0%",
            reference_group=reference_group,
            keep_threshold_pct=0.03,
            treatment_realized_events=treatment_count,
            reference_realized_events=reference_count,
            treatment_total_pnl_usd=treatment_total,
            reference_total_pnl_usd=reference_total,
            pnl_delta_pct=delta,
        )

    return Verdict(
        status=VERDICT_MONITOR,
        reason=f"Treatment PnL delta vs {reference_group} did not clear the +3% keep threshold.",
        reference_group=reference_group,
        keep_threshold_pct=0.03,
        treatment_realized_events=treatment_count,
        reference_realized_events=reference_count,
        treatment_total_pnl_usd=treatment_total,
        reference_total_pnl_usd=reference_total,
        pnl_delta_pct=delta,
    )


def build_report(days: int = DEFAULT_DAYS, association_minutes: int = DEFAULT_ASSOCIATION_MINUTES) -> dict[str, Any]:
    now = datetime.now(UTC)
    diagnostics = load_diagnostics(days, anchor=now)
    fills = load_fills(days, anchor=now)
    diag_index = associate_diagnostics(diagnostics, timedelta(minutes=association_minutes))
    diag_summary = summarize_diagnostics(diagnostics)
    trade_summary = reconstruct_trade_groups(
        fills=fills,
        diag_index=diag_index,
        max_association_gap=timedelta(minutes=association_minutes),
    )
    verdict = build_verdict(trade_summary, diag_summary)
    step15_status = load_json(STEP15_STATUS)

    warnings: list[str] = []
    if diag_summary["source_mix"] and "haiku" not in diag_summary["source_mix"]:
        warnings.append("No direct Haiku source observed in diagnostics; treatment is currently fallback/metrics-only.")
    if safe_int(trade_summary["entry_fills"].get(GROUP_TREATMENT, 0)) == 0:
        warnings.append("No treatment-tagged entry fills were observed in the analysis window.")
    if safe_int(trade_summary.get("unattributed_realized_events", 0), 0) > 0:
        warnings.append(
            "Some realized PnL fill events could not be reliably attributed back to an entry sentiment group; "
            "keep/kill uses only attributed realized events."
        )
    if safe_int(step15_status.get("realized_pnl", {}).get("closed_trade_count", 0)) == 0:
        warnings.append("Current runtime step15_status still reports zero closed trades.")

    return {
        "generated_at": now.isoformat(),
        "window": {
            "days": days,
            "association_minutes": association_minutes,
            "start_at": (now - timedelta(days=days)).isoformat(),
            "end_at": now.isoformat(),
            "baseline_definition": "baseline = fills/trades without a nearby tradeable llm_sentiment diagnostic",
            "control_definition": "control = diagnostics present but llm_sentiment was not tradeable/consumed",
            "treatment_definition": "treatment = diagnostics where llm_sentiment was called and consumed as tradeable alpha",
        },
        "diagnostics": diag_summary,
        "trades": trade_summary,
        "runtime_snapshot": {
            "step15_status_loaded": bool(step15_status),
            "run_started_at": step15_status.get("run", {}).get("started_at"),
            "current_closed_trade_count": step15_status.get("realized_pnl", {}).get("closed_trade_count"),
            "current_tradeable_assets": step15_status.get("sentiment", {}).get("tradeable_assets", []),
        },
        "verdict": asdict(verdict),
        "warnings": warnings,
    }


def fmt_pct(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def fmt_money(value: float) -> str:
    return f"{value:+.2f}"


def render_text(report: dict[str, Any]) -> str:
    diagnostics = report["diagnostics"]
    trades = report["trades"]
    verdict = report["verdict"]
    lines = [
        "HMATS Sentiment A/B Report",
        f"Generated: {report['generated_at']}",
        f"Window: {report['window']['days']}d, association={report['window']['association_minutes']}m",
        "",
        "Coverage",
        f"- diagnostics: total={diagnostics['total']} called={diagnostics['called']} called_rate={fmt_pct(diagnostics['called_rate'])}",
        f"- treatment/control: treatment={diagnostics['treatment_count']} ({fmt_pct(diagnostics['treatment_rate'])}) control={diagnostics['control_count']}",
        f"- source_mix: {diagnostics['source_mix']}",
        f"- treatment alpha_gate/actionable: {fmt_pct(diagnostics['groups'][GROUP_TREATMENT]['alpha_gate_pass_rate'])} / {fmt_pct(diagnostics['groups'][GROUP_TREATMENT]['actionable_rate'])}",
        f"- control alpha_gate/actionable: {fmt_pct(diagnostics['groups'][GROUP_CONTROL]['alpha_gate_pass_rate'])} / {fmt_pct(diagnostics['groups'][GROUP_CONTROL]['actionable_rate'])}",
        "",
        "Trade Linkage",
        f"- entry fills by group: {trades['entry_fills']}",
        f"- realized events by group: {trades['realized_events']} (mixed={trades['mixed_realized_events']})",
        f"- realized pnl by group: { {k: round(v, 4) for k, v in trades['realized_pnl_usd'].items()} }",
        f"- unattributed realized events: {trades['unattributed_realized_events']} pnl={round(trades['unattributed_realized_pnl_usd'], 4)}",
        f"- all fills={trades['all_fills']} realized_fill_events_total={trades['realized_fill_events_total']}",
        "",
        "Verdict",
        f"- status: {verdict['status']}",
        f"- reason: {verdict['reason']}",
        f"- reference_group: {verdict['reference_group']}",
        f"- treatment pnl vs reference: {fmt_money(verdict['treatment_total_pnl_usd'])} vs {fmt_money(verdict['reference_total_pnl_usd'])}",
        f"- pnl_delta_pct: {fmt_pct(verdict['pnl_delta_pct'])} (keep threshold={verdict['keep_threshold_pct'] * 100:.0f}%)",
    ]
    if report["runtime_snapshot"]["step15_status_loaded"]:
        lines.extend(
            [
                "",
                "Runtime Snapshot",
                f"- run_started_at: {report['runtime_snapshot']['run_started_at']}",
                f"- current_closed_trade_count: {report['runtime_snapshot']['current_closed_trade_count']}",
                f"- current_tradeable_assets: {report['runtime_snapshot']['current_tradeable_assets']}",
            ]
        )
    if report["warnings"]:
        lines.append("")
        lines.append("Warnings")
        for warning in report["warnings"]:
            lines.append(f"- {warning}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a local sentiment A/B keep/kill report from diagnostics and shadow-ledger artifacts.")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS, help=f"Look back N days (default: {DEFAULT_DAYS})")
    parser.add_argument(
        "--association-minutes",
        type=int,
        default=DEFAULT_ASSOCIATION_MINUTES,
        help=f"Maximum gap between a fill and its preceding diagnostic (default: {DEFAULT_ASSOCIATION_MINUTES}m)",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    args = parser.parse_args()

    report = build_report(days=args.days, association_minutes=args.association_minutes)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=False))
    else:
        print(render_text(report))


if __name__ == "__main__":
    main()
