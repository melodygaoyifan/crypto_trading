"""Phase 10 promotion-gate plan generator unit tests.

Verifies Iron Laws 4 and 7:
  - missing/malformed reports never raise; produce HOLD/SKIP plan
  - PROMOTE blocked when shadow window < 30d
  - sleeve update blocked when attribution_coverage < 95%
  - vol drift > 50% routes to FLAG_FOR_OPERATOR_REVIEW
  - "unknown" sleeve never promotes
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from analytics.promotion_gate.promotion_plan import (
    MAX_VOL_DRIFT_PCT,
    MIN_ATTRIBUTION_COVERAGE,
    MIN_SHADOW_DAYS_FOR_PROMOTION,
    PHASE7_ESTIMATED_VOLS,
    SleeveAction,
    StrategyAction,
    build_plan,
    build_sleeve_actions,
    build_strategy_actions,
    decide_sleeve_action,
    decide_strategy_action,
    latest_report,
    load_report,
)


# ---------- decide_strategy_action ------------------------------------------

def test_decide_strategy_promote_when_long_window():
    rec = {"verdict": "PROMOTE", "strategy": "ofi", "asset": "BTC"}
    action, reason = decide_strategy_action(rec, shadow_window_days=30)
    assert action is StrategyAction.PROMOTE_TO_FUSION


def test_decide_strategy_promote_blocked_when_short_window():
    """Iron Law 7: PROMOTE downgraded to HOLD when window < 30d."""
    rec = {"verdict": "PROMOTE"}
    action, reason = decide_strategy_action(rec, shadow_window_days=14)
    assert action is StrategyAction.HOLD_SHADOW
    assert "Iron Law 7" in reason


def test_decide_strategy_kill_maps_to_archive():
    rec = {"verdict": "KILL"}
    action, _ = decide_strategy_action(rec, shadow_window_days=14)
    assert action is StrategyAction.ARCHIVE


def test_decide_strategy_insufficient_extends():
    rec = {"verdict": "INSUFFICIENT_SAMPLES"}
    action, _ = decide_strategy_action(rec, shadow_window_days=14)
    assert action is StrategyAction.EXTEND_SHADOW


def test_decide_strategy_hold_passes_through():
    rec = {"verdict": "HOLD"}
    action, _ = decide_strategy_action(rec, shadow_window_days=14)
    assert action is StrategyAction.HOLD_SHADOW


def test_decide_strategy_unknown_verdict_holds():
    rec = {"verdict": ""}
    action, reason = decide_strategy_action(rec, shadow_window_days=14)
    assert action is StrategyAction.HOLD_SHADOW
    assert "unknown" in reason.lower() or "EMPTY" in reason


# ---------- decide_sleeve_action --------------------------------------------

def test_decide_sleeve_unknown_skipped():
    sleeve = {"name": "unknown", "annualized_vol": 0.20, "n_days": 30}
    action, _ = decide_sleeve_action(sleeve, attribution_coverage=1.0,
                                      estimated_vol_lookup={})
    assert action is SleeveAction.SKIP_NO_DATA


def test_decide_sleeve_too_few_days():
    sleeve = {"name": "directional_short", "annualized_vol": 0.30, "n_days": 5}
    action, _ = decide_sleeve_action(sleeve, attribution_coverage=1.0,
                                      estimated_vol_lookup=PHASE7_ESTIMATED_VOLS)
    assert action is SleeveAction.SKIP_NO_DATA


def test_decide_sleeve_low_coverage_defers():
    sleeve = {"name": "directional_short", "annualized_vol": 0.30, "n_days": 30}
    action, reason = decide_sleeve_action(sleeve, attribution_coverage=0.50,
                                           estimated_vol_lookup=PHASE7_ESTIMATED_VOLS)
    assert action is SleeveAction.DEFER_INSUFFICIENT_COVERAGE
    assert "50.0%" in reason and "95%" in reason


def test_decide_sleeve_zero_realized_vol_skipped():
    sleeve = {"name": "directional_short", "annualized_vol": 0.0, "n_days": 30}
    action, _ = decide_sleeve_action(sleeve, attribution_coverage=1.0,
                                      estimated_vol_lookup=PHASE7_ESTIMATED_VOLS)
    assert action is SleeveAction.SKIP_NO_DATA


def test_decide_sleeve_drift_flagged():
    """realized 0.80 vs estimated 0.45 → drift ~78% → FLAG."""
    sleeve = {"name": "directional_short", "annualized_vol": 0.80, "n_days": 30}
    action, reason = decide_sleeve_action(sleeve, attribution_coverage=1.0,
                                           estimated_vol_lookup=PHASE7_ESTIMATED_VOLS)
    assert action is SleeveAction.FLAG_FOR_OPERATOR_REVIEW
    assert "drifts" in reason


def test_decide_sleeve_clean_path_updates_allocator():
    """realized 0.50 vs estimated 0.45 → drift ~11% < 50% threshold → UPDATE."""
    sleeve = {"name": "directional_short", "annualized_vol": 0.50, "n_days": 30}
    action, _ = decide_sleeve_action(sleeve, attribution_coverage=0.97,
                                      estimated_vol_lookup=PHASE7_ESTIMATED_VOLS)
    assert action is SleeveAction.UPDATE_ALLOCATOR_REALIZED_VOL


def test_decide_sleeve_unknown_sleeve_with_no_estimate_passes_drift_check():
    """If sleeve isn't in lookup (estimated=0), drift check is skipped."""
    sleeve = {"name": "new_sleeve", "annualized_vol": 1.20, "n_days": 30}
    action, _ = decide_sleeve_action(sleeve, attribution_coverage=1.0,
                                      estimated_vol_lookup={})
    assert action is SleeveAction.UPDATE_ALLOCATOR_REALIZED_VOL


# ---------- build_strategy_actions / build_sleeve_actions ------------------

def test_build_strategy_actions_passes_window_days():
    report = {
        "window_days": 14,
        "per_strategy": [
            {"verdict": "PROMOTE", "strategy": "ofi", "asset": "BTC",
             "n_records": 100, "annualized_sharpe": 1.0,
             "ic_per_horizon": {"4": 0.06}}
        ]
    }
    actions = build_strategy_actions(report)
    assert len(actions) == 1
    # PROMOTE downgraded because window=14 < 30
    assert actions[0]["action"] == StrategyAction.HOLD_SHADOW.value


def test_build_strategy_actions_promote_passes_at_30d():
    report = {
        "window_days": 30,
        "per_strategy": [
            {"verdict": "PROMOTE", "strategy": "ofi", "asset": "BTC"}
        ]
    }
    actions = build_strategy_actions(report)
    assert actions[0]["action"] == StrategyAction.PROMOTE_TO_FUSION.value


def test_build_sleeve_actions_uses_coverage():
    report = {
        "attribution_coverage": 0.40,
        "sleeves": [
            {"name": "directional_short", "annualized_vol": 0.45, "n_days": 30}
        ]
    }
    actions = build_sleeve_actions(report)
    assert len(actions) == 1
    assert actions[0]["action"] == SleeveAction.DEFER_INSUFFICIENT_COVERAGE.value


# ---------- end-to-end build_plan -------------------------------------------

def _write_shadow_ic(tmp_path: Path, **overrides) -> Path:
    p = tmp_path / "shadow_ic_test.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window_days": 30,
        "horizons_bars": [4, 12, 24],
        "n_records": 200,
        "per_strategy": [
            {"strategy": "ofi", "asset": "BTC", "verdict": "PROMOTE",
             "n_records": 100, "ic_per_horizon": {"4": 0.06},
             "annualized_sharpe": 0.7},
            {"strategy": "vpin_spike", "asset": "BTC", "verdict": "KILL",
             "n_records": 100, "ic_per_horizon": {"4": 0.01},
             "annualized_sharpe": -0.2},
            {"strategy": "kyle_lambda", "asset": "ETH", "verdict": "INSUFFICIENT_SAMPLES",
             "n_records": 5, "ic_per_horizon": {}},
        ],
    }
    payload.update(overrides)
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def _write_sleeve_pnl(tmp_path: Path, **overrides) -> Path:
    p = tmp_path / "sleeve_pnl_test.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window_days": 30,
        "n_total_fills": 100,
        "attribution_coverage": 0.97,
        "initial_capital": 10000.0,
        "sleeves": [
            {"name": "directional_short", "n_days": 30, "n_fills": 80,
             "total_net_pnl": 200.0, "annualized_vol": 0.50,
             "annualized_sharpe": 0.6},
            {"name": "microstructure", "n_days": 5, "n_fills": 5,
             "total_net_pnl": 5.0, "annualized_vol": 0.18,
             "annualized_sharpe": 0.4},
        ],
    }
    payload.update(overrides)
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def test_build_plan_end_to_end(tmp_path: Path):
    sic = _write_shadow_ic(tmp_path)
    spnl = _write_sleeve_pnl(tmp_path)
    plan = build_plan(shadow_ic_report_path=sic, sleeve_pnl_report_path=spnl)
    assert plan["advisory_only"] is True
    assert plan["summary"]["n_strategy_promote"] == 1
    assert plan["summary"]["n_strategy_kill"] == 1
    assert plan["summary"]["n_strategy_extend"] == 1
    assert plan["summary"]["n_sleeve_update"] == 1   # directional_short
    # microstructure has only 5 days → SKIP_NO_DATA
    assert any(s["action"] == SleeveAction.SKIP_NO_DATA.value
               for s in plan["sleeve_actions"] if s["sleeve"] == "microstructure")
    assert plan["summary"]["blockers"] == []


def test_build_plan_missing_reports_yields_blockers(tmp_path: Path):
    plan = build_plan(
        shadow_ic_report_path=tmp_path / "nonexistent_shadow.json",
        sleeve_pnl_report_path=tmp_path / "nonexistent_sleeve.json",
    )
    # Both reports unreadable → 2 blockers
    assert len(plan["summary"]["blockers"]) == 2
    assert plan["shadow_strategy_actions"] == []
    assert plan["sleeve_actions"] == []
    assert plan["advisory_only"] is True  # never auto-promote


def test_build_plan_only_shadow_report(tmp_path: Path):
    sic = _write_shadow_ic(tmp_path)
    plan = build_plan(
        shadow_ic_report_path=sic,
        sleeve_pnl_report_path=tmp_path / "nope.json",
    )
    assert len(plan["summary"]["blockers"]) == 1
    assert len(plan["shadow_strategy_actions"]) == 3
    assert plan["sleeve_actions"] == []


def test_build_plan_with_low_coverage_defers_all_sleeves(tmp_path: Path):
    sic = _write_shadow_ic(tmp_path)
    spnl = _write_sleeve_pnl(tmp_path, attribution_coverage=0.30)
    plan = build_plan(shadow_ic_report_path=sic, sleeve_pnl_report_path=spnl)
    # directional_short with 30d but coverage=30% → DEFER
    direct = next(s for s in plan["sleeve_actions"] if s["sleeve"] == "directional_short")
    assert direct["action"] == SleeveAction.DEFER_INSUFFICIENT_COVERAGE.value


def test_build_plan_short_window_blocks_promotion(tmp_path: Path):
    sic = _write_shadow_ic(tmp_path, window_days=14)
    spnl = _write_sleeve_pnl(tmp_path)
    plan = build_plan(shadow_ic_report_path=sic, sleeve_pnl_report_path=spnl)
    # PROMOTE was in shadow_ic but window=14 → downgraded to HOLD
    assert plan["summary"]["n_strategy_promote"] == 0
    assert plan["summary"]["n_strategy_hold"] >= 1


# ---------- latest_report ---------------------------------------------------

def test_latest_report_returns_none_for_empty_dir(tmp_path: Path):
    assert latest_report(tmp_path / "missing", "*.json") is None
    assert latest_report(tmp_path, "*.json") is None


def test_latest_report_picks_newest(tmp_path: Path):
    p1 = tmp_path / "a.json"
    p1.write_text("{}", encoding="utf-8")
    import time
    time.sleep(0.05)
    p2 = tmp_path / "b.json"
    p2.write_text("{}", encoding="utf-8")
    assert latest_report(tmp_path, "*.json") == p2


# ---------- load_report fail-closed ----------------------------------------

def test_load_report_returns_none_on_malformed(tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text("{ not json", encoding="utf-8")
    assert load_report(p) is None


def test_load_report_returns_none_on_missing(tmp_path: Path):
    assert load_report(tmp_path / "missing.json") is None


# ---------- Iron Law 7+10: no production runtime side-effect ---------------

def test_promotion_plan_does_not_import_runtime_state():
    src = Path("analytics/promotion_gate/promotion_plan.py").read_text(encoding="utf-8")
    # Iron Law 10 (added in this phase): zero runtime side effects.
    # Promotion plan must NOT import live sizers, fusion, allocator instances.
    assert "from risk.unified_position_sizer" not in src
    assert "from risk.sleeve_allocator_v5_1" not in src
    assert "from signals.authority_fusion" not in src
    assert "import main" not in src
    # And must declare itself advisory
    assert '"advisory_only": True' in src
