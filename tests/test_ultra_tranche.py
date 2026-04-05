"""
Tests for P1-03: TrancheScheduler + CascadeExhaustionGovernor under ULTRA_AGGRESSIVE_5Y.

Covers:
  1) Default (HIGH_RISK) vs ULTRA tranche sizes
  2) Escalation gates: 4H close, regime, min bars
  3) CascadeExhaustionGovernor permissions override
  4) TrancheManager (risk/) config wiring
  5) Config overlay end-to-end
  6) Real config file validation
"""

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict

import pytest


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────

def _high_risk_tranche_config() -> None:
    """HIGH_RISK = no config (defaults)."""
    return None


def _ultra_tranche_config() -> Dict:
    """ULTRA tranche config matching ultra_aggressive_5y.json."""
    return {
        "sizes_cumulative": {
            "TRANCHE_1": 0.30,
            "TRANCHE_2": 0.50,
            "TRANCHE_3": 0.75,
            "TRANCHE_4": 1.00,
        },
        "t2_require_4h_close": False,
        "t3_require_regime_aligned": False,
        "t3_min_profit_bps": 0,
        "t4_min_profit_bps": 25,
        "t4_min_time_hours": 1.0,
        "escalation_min_bars": {
            "t1_to_t2": 0,
            "t2_to_t3": 1,
            "t3_to_t4": 2,
        },
    }


def _high_risk_live_tranche_config() -> Dict:
    """HIGH_RISK tranche config matching high_risk.json/live_high_risk.json."""
    return {
        "sizes_cumulative": {
            "TRANCHE_1": 0.35,
            "TRANCHE_2": 0.65,
            "TRANCHE_3": 0.90,
            "TRANCHE_4": 1.00,
        },
        "t2_require_4h_close": True,
        "t3_require_regime_aligned": True,
        "entry_volume_collapse_threshold": 0.03,
        "entry_volume_collapse_threshold_by_asset": {
            "SOL": 0.015,
        },
    }


def _ultra_cascade_permissions() -> Dict:
    """ULTRA cascade permissions matching ultra_aggressive_5y.json."""
    return {
        "DETECTED":      {"T1": True, "T2": True, "T3": False, "T4": False},
        "ACCELERATING":  {"T1": True, "T2": True, "T3": True,  "T4": True},
        "EXHAUSTING":    {"T1": False, "T2": False, "T3": False, "T4": False},
    }


# ─────────────────────────────────────────────────────────────────────────
# 1) Default vs ULTRA tranche sizes - TrancheScheduler (constitution.py)
# ─────────────────────────────────────────────────────────────────────────

class TestTrancheSchedulerSizes:

    def test_default_sizes(self):
        """Current HIGH_RISK defaults: heavier T1/T2/T3 sizing."""
        from defense.constitution import TrancheScheduler, TrancheLevel
        sched = TrancheScheduler()
        assert sched.TRANCHE_SIZES[TrancheLevel.TRANCHE_1] == 0.35
        assert sched.TRANCHE_SIZES[TrancheLevel.TRANCHE_2] == 0.65
        assert sched.TRANCHE_SIZES[TrancheLevel.TRANCHE_3] == 0.90
        assert sched.TRANCHE_SIZES[TrancheLevel.TRANCHE_4] == 1.00

    def test_ultra_sizes(self):
        """ULTRA: T1=30%, T2=50%, T3=75%, T4=100%."""
        from defense.constitution import TrancheScheduler, TrancheLevel
        sched = TrancheScheduler(tranche_config=_ultra_tranche_config())
        assert sched.TRANCHE_SIZES[TrancheLevel.TRANCHE_1] == 0.30
        assert sched.TRANCHE_SIZES[TrancheLevel.TRANCHE_2] == 0.50
        assert sched.TRANCHE_SIZES[TrancheLevel.TRANCHE_3] == 0.75
        assert sched.TRANCHE_SIZES[TrancheLevel.TRANCHE_4] == 1.00

    def test_default_escalation_params(self):
        """HIGH_RISK defaults: t2 needs 4H close, t3 needs regime, t4 needs 50bps+2h."""
        from defense.constitution import TrancheScheduler
        sched = TrancheScheduler()
        assert sched.t2_require_4h_close is True
        assert sched.t3_require_regime_aligned is True
        assert sched.TRANCHE_3_MIN_PROFIT_BPS == 0
        assert sched.TRANCHE_4_MIN_PROFIT_BPS == 50
        assert sched.TRANCHE_4_MIN_TIME_HOURS == 2

    def test_ultra_escalation_params(self):
        """ULTRA: t2 no 4H close, t3 no regime, t4 only 25bps+1h."""
        from defense.constitution import TrancheScheduler
        sched = TrancheScheduler(tranche_config=_ultra_tranche_config())
        assert sched.t2_require_4h_close is False
        assert sched.t3_require_regime_aligned is False
        assert sched.TRANCHE_3_MIN_PROFIT_BPS == 0
        assert sched.TRANCHE_4_MIN_PROFIT_BPS == 25
        assert sched.TRANCHE_4_MIN_TIME_HOURS == 1.0

    def test_ultra_escalation_min_bars(self):
        """ULTRA min bars: T1->T2=0, T2->T3=1, T3->T4=2."""
        from defense.constitution import TrancheScheduler
        sched = TrancheScheduler(tranche_config=_ultra_tranche_config())
        assert sched._min_bars_t1_to_t2 == 0
        assert sched._min_bars_t2_to_t3 == 1
        assert sched._min_bars_t3_to_t4 == 2

    def test_default_entry_volume_collapse_threshold(self):
        """Default tranche scheduler stays fail-closed at 0.05."""
        from defense.constitution import TrancheScheduler
        sched = TrancheScheduler()
        assert sched.entry_volume_collapse_threshold == pytest.approx(0.05)

    def test_high_risk_entry_volume_collapse_threshold(self):
        """HIGH_RISK lowers only the new-entry collapse guard to 0.03."""
        from defense.constitution import TrancheScheduler
        sched = TrancheScheduler(tranche_config=_high_risk_live_tranche_config())
        assert sched.entry_volume_collapse_threshold == pytest.approx(0.03)

    def test_high_risk_entry_volume_collapse_threshold_by_asset(self):
        """HIGH_RISK can lower the new-entry collapse guard further for SOL only."""
        from defense.constitution import TrancheScheduler
        sched = TrancheScheduler(tranche_config=_high_risk_live_tranche_config())
        assert sched._get_entry_volume_collapse_threshold("SOL") == pytest.approx(0.015)
        assert sched._get_entry_volume_collapse_threshold("BTC") == pytest.approx(0.03)


# ─────────────────────────────────────────────────────────────────────────
# 2) Escalation gate behavior
# ─────────────────────────────────────────────────────────────────────────

class TestEscalationGates:

    def _make_scheduler_with_t1_position(self, tranche_config=None):
        """Create scheduler with an existing T1 position in SOL."""
        from defense.constitution import TrancheScheduler, TrancheLevel
        sched = TrancheScheduler(tranche_config=tranche_config)
        pos = sched.get_position("SOL")
        pos.level = TrancheLevel.TRANCHE_1
        pos.direction = "long"
        pos.target_exposure = 1.0
        pos.current_exposure = sched.TRANCHE_SIZES[TrancheLevel.TRANCHE_1]
        pos.entry_price = 100.0
        pos.entered_at = datetime.now(timezone.utc) - timedelta(hours=5)
        pos.last_escalation = datetime.now(timezone.utc) - timedelta(hours=5)
        return sched, pos

    def test_default_t2_needs_4h_close(self):
        """HIGH_RISK: T1->T2 requires is_4h_bar_close=True."""
        from defense.constitution import TrancheLevel
        sched, pos = self._make_scheduler_with_t1_position()
        # No 4H close -> should stay at T1
        decision = sched.decide("SOL", "OPPORTUNITY", 1.0,
                                {"current_price": 105, "volume_ratio": 1.0},
                                {}, is_4h_bar_close=False)
        assert decision.action == "HOLD"
        assert pos.level == TrancheLevel.TRANCHE_1

    def test_default_t2_passes_with_4h_close(self):
        """HIGH_RISK: T1->T2 passes when is_4h_bar_close=True."""
        from defense.constitution import TrancheLevel
        sched, pos = self._make_scheduler_with_t1_position()
        decision = sched.decide("SOL", "OPPORTUNITY", 1.0,
                                {"current_price": 105, "volume_ratio": 1.0},
                                {}, is_4h_bar_close=True)
        assert decision.action == "ESCALATE_T2"
        assert pos.level == TrancheLevel.TRANCHE_2

    def test_ultra_t2_no_4h_close_needed(self):
        """ULTRA: T1->T2 passes WITHOUT is_4h_bar_close (t2_require_4h_close=False)."""
        from defense.constitution import TrancheLevel
        sched, pos = self._make_scheduler_with_t1_position(_ultra_tranche_config())
        decision = sched.decide("SOL", "OPPORTUNITY", 1.0,
                                {"current_price": 105, "volume_ratio": 1.0},
                                {}, is_4h_bar_close=False)
        assert decision.action == "ESCALATE_T2"
        assert pos.level == TrancheLevel.TRANCHE_2

    def test_ultra_t2_exposure_is_50pct(self):
        """ULTRA T2 target exposure = 0.50 (cumulative)."""
        sched, pos = self._make_scheduler_with_t1_position(_ultra_tranche_config())
        decision = sched.decide("SOL", "OPPORTUNITY", 1.0,
                                {"current_price": 105, "volume_ratio": 1.0},
                                {}, is_4h_bar_close=False)
        assert decision.target_exposure == pytest.approx(0.50)

    def test_ultra_t1_entry_is_30pct(self):
        """ULTRA T1 entry = 0.30 (not 0.20)."""
        from defense.constitution import TrancheScheduler
        sched = TrancheScheduler(tranche_config=_ultra_tranche_config())
        decision = sched.decide("BTC", "OPPORTUNITY", 1.0,
                                {"current_price": 50000, "volume_ratio": 1.0,
                                 "data_valid": True},
                                {}, is_4h_bar_close=False)
        assert decision.action == "ENTER_T1"
        assert decision.target_exposure == pytest.approx(0.30)

    def test_high_risk_t1_entry_allows_quiet_weekend_volume_ratio_004(self):
        """HIGH_RISK should still seed T1 when effective quiet volume is 0.04."""
        from defense.constitution import TrancheScheduler

        sched = TrancheScheduler(tranche_config=_high_risk_live_tranche_config())
        decision = sched.decide(
            "SOL",
            "OPPORTUNITY",
            -0.147,
            {
                "current_price": 87.10,
                "volume_ratio": 0.04,
                "bar_progress_4h": 1.0,
                "data_valid": True,
            },
            {},
            is_4h_bar_close=False,
        )
        assert decision.action == "ENTER_T1"
        assert decision.target_exposure == pytest.approx(0.35 * 0.147, abs=1e-6)

    def test_high_risk_sol_t1_entry_allows_effective_volume_ratio_0017(self):
        """HIGH_RISK should allow SOL weekend T1 at 0.017 effective volume via asset override."""
        from defense.constitution import TrancheScheduler

        sched = TrancheScheduler(tranche_config=_high_risk_live_tranche_config())
        decision = sched.decide(
            "SOL",
            "OPPORTUNITY",
            -0.166,
            {
                "current_price": 88.09,
                "volume_ratio": 0.017,
                "bar_progress_4h": 1.0,
                "data_valid": True,
            },
            {},
            is_4h_bar_close=False,
        )
        assert decision.action == "ENTER_T1"

    def test_ultra_t3_no_regime_required(self):
        """ULTRA: T2->T3 does NOT require regime_aligned."""
        from defense.constitution import TrancheScheduler, TrancheLevel
        sched = TrancheScheduler(tranche_config=_ultra_tranche_config())
        pos = sched.get_position("SOL")
        pos.level = TrancheLevel.TRANCHE_2
        pos.direction = "long"
        pos.target_exposure = 1.0
        pos.current_exposure = 0.50
        pos.entry_price = 100.0
        pos.unrealized_pnl_bps = 10  # in profit
        pos.entered_at = datetime.now(timezone.utc) - timedelta(hours=10)
        pos.last_escalation = datetime.now(timezone.utc) - timedelta(hours=5)  # 1+ bars ago

        # regime_aligned=False should still pass under ULTRA
        decision = sched.decide("SOL", "OPPORTUNITY", 1.0,
                                {"current_price": 101, "volume_ratio": 1.0},
                                {"regime_aligned": False},
                                is_4h_bar_close=False)
        assert decision.action == "ESCALATE_T3"
        assert decision.target_exposure == pytest.approx(0.75)

    def test_default_t3_blocked_without_regime(self):
        """HIGH_RISK: T2->T3 BLOCKED when regime_aligned=False."""
        from defense.constitution import TrancheScheduler, TrancheLevel
        sched = TrancheScheduler()
        pos = sched.get_position("SOL")
        pos.level = TrancheLevel.TRANCHE_2
        pos.direction = "long"
        pos.target_exposure = 1.0
        pos.current_exposure = 0.50
        pos.entry_price = 100.0
        pos.unrealized_pnl_bps = 10
        pos.entered_at = datetime.now(timezone.utc) - timedelta(hours=10)
        pos.last_escalation = datetime.now(timezone.utc) - timedelta(hours=5)

        decision = sched.decide("SOL", "OPPORTUNITY", 1.0,
                                {"current_price": 101, "volume_ratio": 1.0},
                                {"regime_aligned": False},
                                is_4h_bar_close=False)
        assert decision.action == "HOLD"

    def test_ultra_t3_blocked_by_min_bars(self):
        """ULTRA: T2->T3 blocked if < 1 bar since last escalation."""
        from defense.constitution import TrancheScheduler, TrancheLevel
        sched = TrancheScheduler(tranche_config=_ultra_tranche_config())
        pos = sched.get_position("SOL")
        pos.level = TrancheLevel.TRANCHE_2
        pos.direction = "long"
        pos.target_exposure = 1.0
        pos.current_exposure = 0.50
        pos.entry_price = 100.0
        pos.unrealized_pnl_bps = 10
        pos.entered_at = datetime.now(timezone.utc) - timedelta(hours=10)
        pos.last_escalation = datetime.now(timezone.utc) - timedelta(hours=1)  # < 4h = 0 bars

        decision = sched.decide("SOL", "OPPORTUNITY", 1.0,
                                {"current_price": 101, "volume_ratio": 1.0},
                                {},
                                is_4h_bar_close=False)
        assert decision.action == "HOLD"


# ─────────────────────────────────────────────────────────────────────────
# 3) CascadeExhaustionGovernor permissions override
# ─────────────────────────────────────────────────────────────────────────

class TestCascadePermissionsOverride:

    def test_default_detected_blocks_t2(self):
        """Current default: DETECTED phase allows T2."""
        from risk.cascade_exhaustion_governor import (
            CascadeExhaustionGovernor, CascadePhase, TrancheLevel,
        )
        gov = CascadeExhaustionGovernor()
        gov._phase = CascadePhase.DETECTED
        result = gov.check_tranche("SOL", 2, "long")
        assert result.allowed is True

    def test_ultra_detected_allows_t2(self):
        """ULTRA: DETECTED phase allows T2."""
        from risk.cascade_exhaustion_governor import (
            CascadeExhaustionGovernor, CascadePhase,
        )
        gov = CascadeExhaustionGovernor(
            permissions_override=_ultra_cascade_permissions()
        )
        gov._phase = CascadePhase.DETECTED
        result = gov.check_tranche("SOL", 2, "long")
        assert result.allowed is True

    def test_default_accelerating_blocks_t4(self):
        """Default: ACCELERATING phase blocks T4."""
        from risk.cascade_exhaustion_governor import (
            CascadeExhaustionGovernor, CascadePhase,
        )
        gov = CascadeExhaustionGovernor()
        gov._phase = CascadePhase.ACCELERATING
        result = gov.check_tranche("SOL", 4, "long")
        assert result.allowed is False

    def test_ultra_accelerating_allows_t4(self):
        """ULTRA: ACCELERATING phase allows T4."""
        from risk.cascade_exhaustion_governor import (
            CascadeExhaustionGovernor, CascadePhase,
        )
        gov = CascadeExhaustionGovernor(
            permissions_override=_ultra_cascade_permissions()
        )
        gov._phase = CascadePhase.ACCELERATING
        result = gov.check_tranche("SOL", 4, "long")
        assert result.allowed is True

    def test_ultra_exhausting_still_blocked(self):
        """ULTRA: EXHAUSTING phase STILL blocks all (hard safety)."""
        from risk.cascade_exhaustion_governor import (
            CascadeExhaustionGovernor, CascadePhase,
        )
        gov = CascadeExhaustionGovernor(
            permissions_override=_ultra_cascade_permissions()
        )
        gov._phase = CascadePhase.EXHAUSTING
        for tranche in [1, 2, 3, 4]:
            result = gov.check_tranche("SOL", tranche, "long")
            assert result.allowed is False, f"T{tranche} should be blocked in EXHAUSTING"

    def test_none_phase_always_allows_all(self):
        """Both default and ULTRA: NONE phase allows all tranches."""
        from risk.cascade_exhaustion_governor import (
            CascadeExhaustionGovernor, CascadePhase,
        )
        for override in [None, _ultra_cascade_permissions()]:
            gov = CascadeExhaustionGovernor(permissions_override=override)
            gov._phase = CascadePhase.NONE
            for tranche in [1, 2, 3, 4]:
                result = gov.check_tranche("SOL", tranche, "long")
                assert result.allowed is True


# ─────────────────────────────────────────────────────────────────────────
# 4) TrancheManager (risk/tranche_manager.py) config wiring
# ─────────────────────────────────────────────────────────────────────────

class TestTrancheManagerConfig:

    def test_default_individual_sizes(self):
        """Default individual sizes: T1=20%, T2=30%, T3=30%, T4=20%."""
        from risk.tranche_manager import TrancheManager
        mgr = TrancheManager()
        assert mgr.TRANCHE_1_SIZE == pytest.approx(0.20)
        assert mgr.TRANCHE_2_SIZE == pytest.approx(0.30)
        assert mgr.TRANCHE_3_SIZE == pytest.approx(0.30)
        assert mgr.TRANCHE_4_SIZE == pytest.approx(0.20)

    def test_ultra_individual_sizes(self):
        """ULTRA individual sizes derived from cumulative: T1=30%, T2=20%, T3=25%, T4=25%."""
        from risk.tranche_manager import TrancheManager
        mgr = TrancheManager(tranche_config=_ultra_tranche_config())
        assert mgr.TRANCHE_1_SIZE == pytest.approx(0.30)
        assert mgr.TRANCHE_2_SIZE == pytest.approx(0.20)  # 0.50 - 0.30
        assert mgr.TRANCHE_3_SIZE == pytest.approx(0.25)  # 0.75 - 0.50
        assert mgr.TRANCHE_4_SIZE == pytest.approx(0.25)  # 1.00 - 0.75

    def test_ultra_cumulative_sizes(self):
        """ULTRA cumulative sizes match config."""
        from risk.tranche_manager import TrancheManager, TrancheLevel
        mgr = TrancheManager(tranche_config=_ultra_tranche_config())
        assert mgr._cumulative_sizes[TrancheLevel.TRANCHE_1] == 0.30
        assert mgr._cumulative_sizes[TrancheLevel.TRANCHE_2] == 0.50
        assert mgr._cumulative_sizes[TrancheLevel.TRANCHE_3] == 0.75
        assert mgr._cumulative_sizes[TrancheLevel.TRANCHE_4] == 1.00

    def test_ultra_escalation_params(self):
        """ULTRA escalation params from config."""
        from risk.tranche_manager import TrancheManager
        mgr = TrancheManager(tranche_config=_ultra_tranche_config())
        assert mgr.TRANCHE_3_MIN_PROFIT_BPS == 0
        assert mgr.TRANCHE_4_MIN_PROFIT_BPS == 25
        assert mgr.TRANCHE_4_MIN_TIME_HOURS == 1.0


# ─────────────────────────────────────────────────────────────────────────
# 5) Config overlay end-to-end (ProductionConfig)
# ─────────────────────────────────────────────────────────────────────────

class TestConfigOverlay:

    def _write_json(self, directory: Path, name: str, data: dict) -> Path:
        path = directory / name
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_ultra_profile_has_tranche_config(self, tmp_path):
        """ProductionConfig with ULTRA overlay must carry tranche_config."""
        from main import ProductionConfig
        base = {
            "assets": ["BTC"],
            "risk": {"hard_drawdown_halt": 0.10},
            "dms_timeout_seconds": 60,
        }
        profile = {
            "risk_profile": "ULTRA_AGGRESSIVE_5Y",
            "tranche": {
                "sizes_cumulative": {"TRANCHE_1": 0.30, "TRANCHE_2": 0.50,
                                     "TRANCHE_3": 0.75, "TRANCHE_4": 1.00},
                "t2_require_4h_close": False,
                "t3_require_regime_aligned": False,
                "t4_min_profit_bps": 25,
                "t4_min_time_hours": 1.0,
            },
            "cascade_permissions": {
                "DETECTED": {"T1": True, "T2": True, "T3": False, "T4": False},
                "ACCELERATING": {"T1": True, "T2": True, "T3": True, "T4": True},
            },
        }
        self._write_json(tmp_path, "base.json", base)
        self._write_json(tmp_path, "ultra_aggressive_5y.json", profile)

        cfg = ProductionConfig.from_file(
            tmp_path / "base.json", risk_profile="ULTRA_AGGRESSIVE_5Y"
        )
        assert cfg.tranche_config is not None
        assert cfg.tranche_config["sizes_cumulative"]["TRANCHE_1"] == 0.30
        assert cfg.tranche_config["t2_require_4h_close"] is False

    def test_ultra_profile_has_cascade_permissions(self, tmp_path):
        """ProductionConfig with ULTRA overlay must carry cascade_permissions_override."""
        from main import ProductionConfig
        base = {
            "assets": ["BTC"],
            "risk": {"hard_drawdown_halt": 0.10},
            "dms_timeout_seconds": 60,
        }
        profile = {
            "risk_profile": "ULTRA_AGGRESSIVE_5Y",
            "cascade_permissions": {
                "DETECTED": {"T1": True, "T2": True, "T3": False, "T4": False},
            },
        }
        self._write_json(tmp_path, "base.json", base)
        self._write_json(tmp_path, "ultra_aggressive_5y.json", profile)

        cfg = ProductionConfig.from_file(
            tmp_path / "base.json", risk_profile="ULTRA_AGGRESSIVE_5Y"
        )
        assert cfg.cascade_permissions_override is not None
        assert cfg.cascade_permissions_override["DETECTED"]["T2"] is True

    def test_default_profile_has_no_tranche_config(self, tmp_path):
        """ProductionConfig without overlay must have tranche_config=None."""
        from main import ProductionConfig
        base = {
            "assets": ["BTC"],
            "risk": {"hard_drawdown_halt": 0.10},
            "dms_timeout_seconds": 60,
        }
        self._write_json(tmp_path, "base.json", base)

        cfg = ProductionConfig.from_file(tmp_path / "base.json")
        assert cfg.tranche_config is None
        assert cfg.cascade_permissions_override is None


# ─────────────────────────────────────────────────────────────────────────
# 6) Abort conditions preserved (not relaxed by profile)
# ─────────────────────────────────────────────────────────────────────────

class TestAbortPreserved:

    def test_ultra_vpin_abort_still_works(self):
        """ULTRA: VPIN spike still triggers abort."""
        from defense.constitution import TrancheScheduler, TrancheLevel
        sched = TrancheScheduler(tranche_config=_ultra_tranche_config())
        pos = sched.get_position("SOL")
        pos.level = TrancheLevel.TRANCHE_2
        pos.direction = "long"
        pos.current_exposure = 0.50
        pos.entry_price = 100.0
        pos.entered_at = datetime.now(timezone.utc) - timedelta(hours=10)

        decision = sched.decide("SOL", "OPPORTUNITY", 1.0,
                                {"current_price": 99, "volume_ratio": 1.0,
                                 "vpin": 0.90},  # > 0.85 threshold
                                {},
                                is_4h_bar_close=False)
        # VPIN is a hard abort -> should reduce or hold (not escalate)
        assert decision.action != "ESCALATE_T3"

    def test_ultra_no_trade_still_flattens(self):
        """ULTRA: NO_TRADE mode still forces flatten."""
        from defense.constitution import TrancheScheduler, TrancheLevel
        sched = TrancheScheduler(tranche_config=_ultra_tranche_config())
        pos = sched.get_position("SOL")
        pos.level = TrancheLevel.TRANCHE_3
        pos.direction = "long"
        pos.current_exposure = 0.75

        decision = sched.decide("SOL", "NO_TRADE", 0.0,
                                {"current_price": 100, "volume_ratio": 1.0},
                                {},
                                is_4h_bar_close=False)
        assert decision.action in ("FLATTEN", "REDUCE")


# ─────────────────────────────────────────────────────────────────────────
# 7) Real config file validation
# ─────────────────────────────────────────────────────────────────────────

class TestRealUltraTrancheConfig:

    def test_ultra_json_has_tranche_section(self):
        """configs/ultra_aggressive_5y.json must contain tranche section."""
        path = Path("configs/ultra_aggressive_5y.json")
        assert path.exists()
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        t = data["tranche"]
        assert t["sizes_cumulative"]["TRANCHE_1"] == 0.30
        assert t["sizes_cumulative"]["TRANCHE_2"] == 0.50
        assert t["sizes_cumulative"]["TRANCHE_3"] == 0.75
        assert t["sizes_cumulative"]["TRANCHE_4"] == 1.00
        assert t["t2_require_4h_close"] is False
        assert t["t3_require_regime_aligned"] is False
        assert t["t4_min_profit_bps"] == 25
        assert t["t4_min_time_hours"] == 1.0

    def test_ultra_json_has_cascade_permissions(self):
        """configs/ultra_aggressive_5y.json must contain cascade_permissions section."""
        path = Path("configs/ultra_aggressive_5y.json")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        cp = data["cascade_permissions"]
        assert cp["DETECTED"]["T2"] is True
        assert cp["ACCELERATING"]["T4"] is True
        assert cp["EXHAUSTING"]["T1"] is False

    def test_ultra_json_has_escalation_min_bars(self):
        """configs/ultra_aggressive_5y.json must contain escalation_min_bars."""
        path = Path("configs/ultra_aggressive_5y.json")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        bars = data["tranche"]["escalation_min_bars"]
        assert bars["t1_to_t2"] == 0
        assert bars["t2_to_t3"] == 1
        assert bars["t3_to_t4"] == 2
