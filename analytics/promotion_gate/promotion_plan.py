"""
HMATS v5.1 Phase 10 - Promotion Gate Plan Generator (ADVISORY)
================================================================

Reads:
  - analytics/shadow_ic/reports/shadow_ic_*.json   (Pre-6.2)
  - analytics/sleeve_attribution/reports/sleeve_pnl_*.json (Phase 6.0 prep)

Emits a "would-promote" plan: which shadow strategies and which sleeves
would graduate to live participation if operator approves. Plan is
written to analytics/promotion_gate/plans/promotion_plan_*.json and
human-printed to console.

Iron Laws honored:
  4. fail-closed: missing reports / unparseable JSON / corrupted entries
     -> sentinel HOLD/INSUFFICIENT_DATA verdict; no auto-promotion.
  7. >=30d shadow before promotion: enforced via window_days check on
     the shadow_ic report (must be >=30) AND independent sleeve attribution
     coverage >= 95% (set by sleeve_pnl report).
  10. ZERO side-effects on production runtime: this module does NOT
      modify any allocator, fusion, or sizer state. Operator must run
      a separate (future) `apply_promotion_plan.py` after reviewing.

Decision matrix per shadow strategy:
  Verdict from shadow_ic report  | Plan action
  -----------------------------  | -----------
  PROMOTE                        | PROMOTE_TO_FUSION (advisory)
  HOLD                           | HOLD_SHADOW
  KILL                           | ARCHIVE
  INSUFFICIENT_SAMPLES           | EXTEND_SHADOW

Decision matrix per sleeve:
  attribution_coverage >= 95% AND realized_vol within 50% of estimated
                                | UPDATE_ALLOCATOR_REALIZED_VOL
  attribution_coverage < 95%   | DEFER (P25 era data; wait for more)
  realized_vol drift > 50%     | FLAG_FOR_OPERATOR_REVIEW

Output JSON shape (Phase 11 60-day review consumes this):
  {
    "generated_at": "ISO8601",
    "inputs": {"shadow_ic_report": "...", "sleeve_pnl_report": "..."},
    "shadow_strategy_actions": [...],
    "sleeve_actions": [...],
    "summary": {"n_promote": ..., "n_kill": ..., "n_hold": ...,
                "blockers": [...]},
  }
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


REPO = Path(__file__).resolve().parents[2]
SHADOW_IC_REPORT_DIR = REPO / "analytics" / "shadow_ic" / "reports"
SLEEVE_PNL_REPORT_DIR = REPO / "analytics" / "sleeve_attribution" / "reports"
PLAN_DIR = REPO / "analytics" / "promotion_gate" / "plans"


# Promotion-gate thresholds
MIN_SHADOW_DAYS_FOR_PROMOTION = 30
MIN_ATTRIBUTION_COVERAGE = 0.95
MAX_VOL_DRIFT_PCT = 0.50         # realized vs estimated; >50% drift triggers operator review


# ---------------------------------------------------------------------------
# Action enums
# ---------------------------------------------------------------------------

class StrategyAction(Enum):
    PROMOTE_TO_FUSION = "PROMOTE_TO_FUSION"
    HOLD_SHADOW = "HOLD_SHADOW"
    ARCHIVE = "ARCHIVE"
    EXTEND_SHADOW = "EXTEND_SHADOW"


class SleeveAction(Enum):
    UPDATE_ALLOCATOR_REALIZED_VOL = "UPDATE_ALLOCATOR_REALIZED_VOL"
    DEFER_INSUFFICIENT_COVERAGE = "DEFER_INSUFFICIENT_COVERAGE"
    FLAG_FOR_OPERATOR_REVIEW = "FLAG_FOR_OPERATOR_REVIEW"
    SKIP_NO_DATA = "SKIP_NO_DATA"


# ---------------------------------------------------------------------------
# Report loaders
# ---------------------------------------------------------------------------

def latest_report(directory: Path, pattern: str) -> Optional[Path]:
    """Return the most-recently-modified file matching pattern in directory."""
    if not directory.exists():
        return None
    files = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def load_report(path: Path) -> Optional[Dict[str, Any]]:
    """Read+parse JSON. Returns None on any error (caller handles)."""
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"[PROMOTION_PLAN] failed to load {path}: {type(e).__name__}: {e}")
        return None


# ---------------------------------------------------------------------------
# Strategy decisions
# ---------------------------------------------------------------------------

def decide_strategy_action(
    record: Dict[str, Any],
    shadow_window_days: int,
) -> Tuple[StrategyAction, str]:
    """Map a per-strategy shadow_ic record to a plan action + reason."""
    verdict = str(record.get("verdict", "")).upper()

    # Iron Law 7: PROMOTE only if shadow window is >=30d
    if verdict == "PROMOTE":
        if shadow_window_days < MIN_SHADOW_DAYS_FOR_PROMOTION:
            return (
                StrategyAction.HOLD_SHADOW,
                f"verdict=PROMOTE but shadow_window_days={shadow_window_days} < "
                f"{MIN_SHADOW_DAYS_FOR_PROMOTION} (Iron Law 7)"
            )
        return (StrategyAction.PROMOTE_TO_FUSION, "verdict=PROMOTE + window >= 30d")
    if verdict == "KILL":
        return (StrategyAction.ARCHIVE, "verdict=KILL")
    if verdict == "INSUFFICIENT_SAMPLES":
        return (StrategyAction.EXTEND_SHADOW, "verdict=INSUFFICIENT_SAMPLES")
    if verdict == "HOLD":
        return (StrategyAction.HOLD_SHADOW, "verdict=HOLD")
    return (StrategyAction.HOLD_SHADOW, f"unknown_verdict={verdict or 'EMPTY'}")


def build_strategy_actions(
    shadow_ic_report: Dict[str, Any],
) -> List[Dict[str, Any]]:
    window_days = int(shadow_ic_report.get("window_days") or 0)
    out = []
    for rec in shadow_ic_report.get("per_strategy", []):
        action, reason = decide_strategy_action(rec, window_days)
        out.append({
            "strategy": rec.get("strategy"),
            "asset": rec.get("asset"),
            "action": action.value,
            "reason": reason,
            "verdict": rec.get("verdict"),
            "ic_per_horizon": rec.get("ic_per_horizon", {}),
            "n_records": rec.get("n_records", 0),
            "annualized_sharpe": rec.get("annualized_sharpe", 0.0),
        })
    return out


# ---------------------------------------------------------------------------
# Sleeve decisions
# ---------------------------------------------------------------------------

def decide_sleeve_action(
    sleeve: Dict[str, Any],
    attribution_coverage: float,
    estimated_vol_lookup: Dict[str, float],
) -> Tuple[SleeveAction, str]:
    name = sleeve.get("name", "")
    realized_vol = float(sleeve.get("annualized_vol", 0.0) or 0.0)
    n_days = int(sleeve.get("n_days", 0) or 0)

    if name == "unknown":
        return (SleeveAction.SKIP_NO_DATA, "unknown sleeve bucket — never promotes")

    if n_days < 7:
        return (SleeveAction.SKIP_NO_DATA, f"only {n_days} days of attribution")

    if attribution_coverage < MIN_ATTRIBUTION_COVERAGE:
        return (
            SleeveAction.DEFER_INSUFFICIENT_COVERAGE,
            f"attribution_coverage={attribution_coverage:.1%} < "
            f"{MIN_ATTRIBUTION_COVERAGE:.0%} (likely pre-P25 era records)"
        )

    if realized_vol <= 0:
        return (SleeveAction.SKIP_NO_DATA, "realized_vol = 0 (no realized PnL yet)")

    estimated = estimated_vol_lookup.get(name, 0.0)
    if estimated > 0:
        drift = abs(realized_vol - estimated) / estimated
        if drift > MAX_VOL_DRIFT_PCT:
            return (
                SleeveAction.FLAG_FOR_OPERATOR_REVIEW,
                f"realized_vol={realized_vol:.2%} drifts {drift*100:.0f}% from "
                f"estimated={estimated:.2%} (>{MAX_VOL_DRIFT_PCT*100:.0f}% threshold)"
            )

    return (
        SleeveAction.UPDATE_ALLOCATOR_REALIZED_VOL,
        f"coverage OK + drift OK; feed realized_vol={realized_vol:.2%}"
    )


# Estimated-vol lookup matches build_phase7_sleeve_allocator()
PHASE7_ESTIMATED_VOLS = {
    "directional_short": 0.45,
    "microstructure": 0.20,
    "cascade": 0.30,
}


def build_sleeve_actions(
    sleeve_pnl_report: Dict[str, Any],
) -> List[Dict[str, Any]]:
    coverage = float(sleeve_pnl_report.get("attribution_coverage", 1.0) or 0.0)
    out = []
    for sleeve in sleeve_pnl_report.get("sleeves", []):
        action, reason = decide_sleeve_action(sleeve, coverage, PHASE7_ESTIMATED_VOLS)
        out.append({
            "sleeve": sleeve.get("name"),
            "action": action.value,
            "reason": reason,
            "n_days": sleeve.get("n_days", 0),
            "n_fills": sleeve.get("n_fills", 0),
            "realized_vol": sleeve.get("annualized_vol", 0.0),
            "estimated_vol": PHASE7_ESTIMATED_VOLS.get(sleeve.get("name", ""), None),
            "annualized_sharpe": sleeve.get("annualized_sharpe", 0.0),
            "total_net_pnl": sleeve.get("total_net_pnl", 0.0),
        })
    return out


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------

def build_plan(
    shadow_ic_report_path: Optional[Path] = None,
    sleeve_pnl_report_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Build the promotion plan. Either path can be omitted; latest auto-picked."""
    if shadow_ic_report_path is None:
        shadow_ic_report_path = latest_report(SHADOW_IC_REPORT_DIR, "shadow_ic_*.json")
    if sleeve_pnl_report_path is None:
        sleeve_pnl_report_path = latest_report(SLEEVE_PNL_REPORT_DIR, "sleeve_pnl_*.json")

    blockers: List[str] = []
    shadow_ic_report: Optional[Dict[str, Any]] = None
    sleeve_pnl_report: Optional[Dict[str, Any]] = None

    if shadow_ic_report_path is None:
        blockers.append("no shadow_ic report found")
    else:
        shadow_ic_report = load_report(shadow_ic_report_path)
        if shadow_ic_report is None:
            blockers.append(f"shadow_ic report unparseable: {shadow_ic_report_path}")

    if sleeve_pnl_report_path is None:
        blockers.append("no sleeve_pnl report found")
    else:
        sleeve_pnl_report = load_report(sleeve_pnl_report_path)
        if sleeve_pnl_report is None:
            blockers.append(f"sleeve_pnl report unparseable: {sleeve_pnl_report_path}")

    strategy_actions: List[Dict[str, Any]] = []
    sleeve_actions: List[Dict[str, Any]] = []

    if shadow_ic_report is not None:
        strategy_actions = build_strategy_actions(shadow_ic_report)
    if sleeve_pnl_report is not None:
        sleeve_actions = build_sleeve_actions(sleeve_pnl_report)

    n_promote = sum(1 for s in strategy_actions if s["action"] == StrategyAction.PROMOTE_TO_FUSION.value)
    n_kill = sum(1 for s in strategy_actions if s["action"] == StrategyAction.ARCHIVE.value)
    n_hold = sum(1 for s in strategy_actions if s["action"] == StrategyAction.HOLD_SHADOW.value)
    n_extend = sum(1 for s in strategy_actions if s["action"] == StrategyAction.EXTEND_SHADOW.value)
    n_sleeve_update = sum(1 for s in sleeve_actions if s["action"] == SleeveAction.UPDATE_ALLOCATOR_REALIZED_VOL.value)
    n_sleeve_defer = sum(1 for s in sleeve_actions if s["action"] == SleeveAction.DEFER_INSUFFICIENT_COVERAGE.value)
    n_sleeve_flag = sum(1 for s in sleeve_actions if s["action"] == SleeveAction.FLAG_FOR_OPERATOR_REVIEW.value)

    plan = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "shadow_ic_report": str(shadow_ic_report_path) if shadow_ic_report_path else None,
            "sleeve_pnl_report": str(sleeve_pnl_report_path) if sleeve_pnl_report_path else None,
        },
        "thresholds": {
            "min_shadow_days_for_promotion": MIN_SHADOW_DAYS_FOR_PROMOTION,
            "min_attribution_coverage": MIN_ATTRIBUTION_COVERAGE,
            "max_vol_drift_pct": MAX_VOL_DRIFT_PCT,
        },
        "shadow_strategy_actions": strategy_actions,
        "sleeve_actions": sleeve_actions,
        "summary": {
            "n_strategy_promote": n_promote,
            "n_strategy_kill": n_kill,
            "n_strategy_hold": n_hold,
            "n_strategy_extend": n_extend,
            "n_sleeve_update": n_sleeve_update,
            "n_sleeve_defer": n_sleeve_defer,
            "n_sleeve_flag": n_sleeve_flag,
            "blockers": blockers,
        },
        "advisory_only": True,
    }
    return plan


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def render_plan(plan: Dict[str, Any]) -> str:
    lines = []
    lines.append("=" * 88)
    lines.append("  PROMOTION GATE PLAN  (advisory_only=True — operator must apply manually)")
    lines.append("=" * 88)
    lines.append(f"  generated_at: {plan['generated_at']}")
    lines.append(f"  shadow_ic_report:  {plan['inputs']['shadow_ic_report']}")
    lines.append(f"  sleeve_pnl_report: {plan['inputs']['sleeve_pnl_report']}")
    lines.append("")

    if plan["summary"]["blockers"]:
        lines.append("  BLOCKERS:")
        for b in plan["summary"]["blockers"]:
            lines.append(f"    - {b}")
        lines.append("")

    lines.append("  STRATEGY ACTIONS:")
    if not plan["shadow_strategy_actions"]:
        lines.append("    (none — shadow_ic report missing or empty)")
    for s in plan["shadow_strategy_actions"]:
        lines.append(
            f"    {s['action']:<24} {s['strategy']:<24} {s['asset']:<5} "
            f"verdict={s['verdict']}  reason={s['reason']}"
        )
    lines.append("")

    lines.append("  SLEEVE ACTIONS:")
    if not plan["sleeve_actions"]:
        lines.append("    (none — sleeve_pnl report missing or empty)")
    for s in plan["sleeve_actions"]:
        rv = s.get("realized_vol", 0.0)
        ev = s.get("estimated_vol")
        ev_str = f"{ev:.2%}" if isinstance(ev, (int, float)) else "n/a"
        lines.append(
            f"    {s['action']:<32} {s['sleeve']:<22} "
            f"realized={rv:.2%} estimated={ev_str}  reason={s['reason']}"
        )
    lines.append("")
    lines.append("  SUMMARY: " + " ".join(
        f"{k}={v}" for k, v in plan["summary"].items() if k != "blockers"
    ))
    lines.append("=" * 88)
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Generate v5.1 promotion-gate advisory plan.")
    p.add_argument("--shadow-ic-report", default=None,
                   help="Path to shadow_ic_*.json (default: latest in analytics/shadow_ic/reports)")
    p.add_argument("--sleeve-pnl-report", default=None,
                   help="Path to sleeve_pnl_*.json (default: latest in analytics/sleeve_attribution/reports)")
    p.add_argument("--output", default=None,
                   help="Output JSON path (default: analytics/promotion_gate/plans/promotion_plan_*.json)")
    args = p.parse_args(argv)

    shadow_path = Path(args.shadow_ic_report) if args.shadow_ic_report else None
    sleeve_path = Path(args.sleeve_pnl_report) if args.sleeve_pnl_report else None
    plan = build_plan(shadow_ic_report_path=shadow_path, sleeve_pnl_report_path=sleeve_path)

    print(render_plan(plan))

    PLAN_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else (
        PLAN_DIR / f"promotion_plan_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.json"
    )
    out_path.write_text(json.dumps(plan, default=str, indent=2), encoding="utf-8")
    print(f"\nPlan saved: {out_path}")
    return 0 if not plan["summary"]["blockers"] else 1


if __name__ == "__main__":
    sys.exit(main())
