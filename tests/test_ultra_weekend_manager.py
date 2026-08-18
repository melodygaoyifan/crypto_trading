"""
Tests for P1-04: Weekend manager under ULTRA_AGGRESSIVE_5Y profile.

Covers:
  1) RiskVetoClassifier weekend soft veto cap override
  2) IntegratedWeekendLiquidityManager config override
  3) Default (HIGH_RISK) behavior preserved when no config
  4) Hard vetoes still work during weekends
  5) is_weekend detection logic
  6) Config JSON validation
  7) ProductionConfig parsing
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
from unittest.mock import patch

import pytest


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────

def _ultra_weekend_config() -> Dict:
    """ULTRA weekend config matching ultra_aggressive_5y.json."""
    return {
        "weekend_soft_veto_cap": 1.00,
        "weekend_max_exposure": 1.00,
        "weekend_leverage_cap": 3.0,
        "reduce_friday_evening": False,
        "per_asset_weekend_caps": {
            "BTC": 0.60,
            "ETH": 0.60,
            "SOL": 0.80,
        },
        "reduce_at_hours": 999,
        "flatten_at_hours": 999,
        "min_alpha_multiplier_weekend": 1.0,
        "min_confidence_weekend": 0.50,
    }


def _no_config() -> None:
    """HIGH_RISK = no weekend config."""
    return None


def _weekend_state():
    from liquidity.weekend_manager import LiquidityQuality, TradingSession, WeekendState

    return WeekendState(
        session=TradingSession.WEEKEND,
        is_weekend=True,
        hours_until_normal=42.0,
        hours_into_weekend=6.0,
        liquidity_quality=LiquidityQuality.REDUCED,
        position_multiplier=0.50,
        spread_multiplier=1.50,
    )


# ─────────────────────────────────────────────────────────────────────────
# Test: RiskVetoClassifier weekend cap override
# ─────────────────────────────────────────────────────────────────────────

class TestRiskVetoClassifierWeekend:
    """RiskVetoClassifier weekend soft veto cap override."""

    def test_default_weekend_cap_is_040(self):
        """Without config, WEEKEND_LIQUIDITY cap = 0.40."""
        from defense.production_reliability import RiskVetoClassifier, SoftVetoCondition
        clf = RiskVetoClassifier()
        assert clf.SOFT_EXPOSURE_CAPS[SoftVetoCondition.WEEKEND_LIQUIDITY] == pytest.approx(0.40)

    def test_ultra_weekend_cap_is_100(self):
        """With ULTRA config, WEEKEND_LIQUIDITY cap = 1.00 (disabled)."""
        from defense.production_reliability import RiskVetoClassifier, SoftVetoCondition
        clf = RiskVetoClassifier(weekend_config=_ultra_weekend_config())
        assert clf.SOFT_EXPOSURE_CAPS[SoftVetoCondition.WEEKEND_LIQUIDITY] == pytest.approx(1.00)

    def test_ultra_weekend_classify_allows_full_exposure(self):
        """With ULTRA config and is_weekend=True, soft veto caps at 100% (no reduction)."""
        from defense.production_reliability import RiskVetoClassifier
        clf = RiskVetoClassifier(weekend_config=_ultra_weekend_config())
        result = clf.classify(
            mode="OPPORTUNITY",
            is_weekend=True,
        )
        # Weekend is the only soft condition, cap should be 1.0
        assert result.exposure_cap >= 1.0 or result.veto_type.value == "NONE"

    def test_default_weekend_classify_caps_at_040(self):
        """Without config, is_weekend=True caps at 40%."""
        from defense.production_reliability import RiskVetoClassifier
        clf = RiskVetoClassifier()
        result = clf.classify(
            mode="OPPORTUNITY",
            is_weekend=True,
        )
        assert result.exposure_cap == pytest.approx(0.40)

    def test_other_soft_caps_unchanged(self):
        """ULTRA weekend config only changes WEEKEND_LIQUIDITY, not others."""
        from defense.production_reliability import RiskVetoClassifier, SoftVetoCondition
        clf = RiskVetoClassifier(weekend_config=_ultra_weekend_config())
        assert clf.SOFT_EXPOSURE_CAPS[SoftVetoCondition.ELEVATED_DVOL] == pytest.approx(0.50)
        assert clf.SOFT_EXPOSURE_CAPS[SoftVetoCondition.LIQUIDITY_WARNING] == pytest.approx(0.60)
        assert clf.SOFT_EXPOSURE_CAPS[SoftVetoCondition.MISSING_CRITICAL_KEYS] == pytest.approx(0.30)

    def test_custom_cap_value(self):
        """Arbitrary weekend_soft_veto_cap value works."""
        from defense.production_reliability import RiskVetoClassifier, SoftVetoCondition
        cfg = {"weekend_soft_veto_cap": 0.75}
        clf = RiskVetoClassifier(weekend_config=cfg)
        assert clf.SOFT_EXPOSURE_CAPS[SoftVetoCondition.WEEKEND_LIQUIDITY] == pytest.approx(0.75)


class TestWeekendOverrideRules:
    def test_opportunity_alpha_multiplier_can_be_overridden_by_asset(self):
        from liquidity.weekend_manager import WeekendOverrideRules

        blocked, reason = WeekendOverrideRules.should_override_entry(
            _weekend_state(),
            estimated_alpha_bps=9.0,
            confidence=0.60,
            asset="BTC",
            system_mode="OPPORTUNITY",
            weekend_config={
                "opportunity_alpha_multiplier_weekend": 0.45,
                "opportunity_alpha_multiplier_weekend_by_asset": {"BTC": 0.25},
                "opportunity_confidence_weekend": 0.45,
            },
        )

        assert blocked is False
        assert reason == ""

    def test_opportunity_confidence_can_be_overridden_by_asset(self):
        from liquidity.weekend_manager import WeekendOverrideRules

        blocked, reason = WeekendOverrideRules.should_override_entry(
            _weekend_state(),
            estimated_alpha_bps=9.0,
            confidence=0.32,
            asset="BTC",
            system_mode="OPPORTUNITY",
            weekend_config={
                "opportunity_alpha_multiplier_weekend": 0.25,
                "opportunity_confidence_weekend": 0.45,
                "opportunity_confidence_weekend_by_asset": {"BTC": 0.30},
            },
        )

        assert blocked is False
        assert reason == ""

    def test_opportunity_alpha_multiplier_falls_back_to_global_when_asset_missing(self):
        """[P42] base bps lowered 33 → 20, so the calculated min for ETH
        falls back to (20 * 0.45) = 9 bps. Alpha=5 is below → reject."""
        from liquidity.weekend_manager import WeekendOverrideRules

        blocked, reason = WeekendOverrideRules.should_override_entry(
            _weekend_state(),
            estimated_alpha_bps=5.0,
            confidence=0.60,
            asset="ETH",
            system_mode="OPPORTUNITY",
            weekend_config={
                "opportunity_alpha_multiplier_weekend": 0.45,
                "opportunity_alpha_multiplier_weekend_by_asset": {"BTC": 0.25},
                "opportunity_confidence_weekend": 0.45,
            },
        )

        assert blocked is True
        assert "Weekend alpha 5bps < min 9bps" in reason

    def test_weekend_min_alpha_bps_config_overrides_base(self):
        """[P42] Operator can override the base bps via config —
        tightens or loosens the floor without editing constants."""
        from liquidity.weekend_manager import WeekendOverrideRules

        # Operator config sets base=10, mult=1.0 → min = 10 bps
        blocked, reason = WeekendOverrideRules.should_override_entry(
            _weekend_state(),
            estimated_alpha_bps=8.0,
            confidence=0.60,
            asset="BTC",
            system_mode="NORMAL",
            weekend_config={
                "weekend_min_alpha_bps": 10.0,
                "min_alpha_multiplier_weekend": 1.0,
                "min_confidence_weekend": 0.50,
            },
        )
        assert blocked is True
        assert "min 10bps" in reason

        # Same alpha, looser base (5 bps) → not blocked
        blocked, reason = WeekendOverrideRules.should_override_entry(
            _weekend_state(),
            estimated_alpha_bps=8.0,
            confidence=0.60,
            asset="BTC",
            system_mode="NORMAL",
            weekend_config={
                "weekend_min_alpha_bps": 5.0,
                "min_alpha_multiplier_weekend": 1.0,
                "min_confidence_weekend": 0.50,
            },
        )
        assert blocked is False

    def test_weekend_default_floor_is_20_bps(self):
        """[P42] Default base is 20 bps (was hardcoded 33). With default
        mult=1.0, the floor is 20 bps for NORMAL mode."""
        from liquidity.weekend_manager import WeekendOverrideRules

        blocked, reason = WeekendOverrideRules.should_override_entry(
            _weekend_state(),
            estimated_alpha_bps=15.0,  # below 20 bps default floor
            confidence=0.60,
            asset="BTC",
            system_mode="NORMAL",
            weekend_config={"min_confidence_weekend": 0.50},  # only conf override
        )
        assert blocked is True
        assert "min 20bps" in reason


# ─────────────────────────────────────────────────────────────────────────
# Test: Hard vetoes still block during weekends
# ─────────────────────────────────────────────────────────────────────────

class TestHardVetoesPreserved:
    """Hard vetoes remain non-negotiable even with ULTRA weekend config."""

    def test_drawdown_hard_veto_on_weekend(self):
        """Drawdown >= 20% (UL-5 unleash) triggers HARD veto even with ULTRA weekend config.

        [P58 2026-04-25] Was 0.12 — but RiskVetoClassifier.HARD_THRESHOLDS["drawdown"]
        was widened from 0.10 to 0.20 in UL-5 unleash. Updated to 0.22 to test
        actual current HARD-threshold semantics.
        """
        from defense.production_reliability import RiskVetoClassifier
        clf = RiskVetoClassifier(weekend_config=_ultra_weekend_config())
        result = clf.classify(
            mode="OPPORTUNITY",
            drawdown=0.22,  # was 0.12, now > 0.20 HARD threshold
            is_weekend=True,
        )
        assert result.veto_type.value == "HARD"
        assert not result.allows_trade

    def test_correlation_crisis_hard_veto_on_weekend(self):
        """Correlation >= 0.98 (UL-4 unleash) triggers HARD veto on weekend.

        [P58 2026-04-25] Was 0.96 — UL-4 widened HARD threshold from 0.95 to 0.98.
        Updated to 0.99 to test actual HARD-threshold semantics.
        """
        from defense.production_reliability import RiskVetoClassifier
        clf = RiskVetoClassifier(weekend_config=_ultra_weekend_config())
        result = clf.classify(
            mode="OPPORTUNITY",
            correlation=0.99,  # was 0.96, now > 0.98 HARD threshold
            is_weekend=True,
        )
        assert result.veto_type.value == "HARD"

    def test_data_integrity_hard_veto_on_weekend(self):
        """data_valid=False triggers HARD veto on weekend."""
        from defense.production_reliability import RiskVetoClassifier
        clf = RiskVetoClassifier(weekend_config=_ultra_weekend_config())
        result = clf.classify(
            mode="OPPORTUNITY",
            data_valid=False,
            is_weekend=True,
        )
        assert result.veto_type.value == "HARD"

    def test_flash_crash_hard_veto_on_weekend(self):
        """Flash crash triggers HARD veto on weekend."""
        from defense.production_reliability import RiskVetoClassifier
        clf = RiskVetoClassifier(weekend_config=_ultra_weekend_config())
        result = clf.classify(
            mode="OPPORTUNITY",
            flash_crash_active=True,
            is_weekend=True,
        )
        assert result.veto_type.value == "HARD"


# ─────────────────────────────────────────────────────────────────────────
# Test: IntegratedWeekendLiquidityManager config override
# ─────────────────────────────────────────────────────────────────────────

class TestIntegratedLiquidityManagerConfig:
    """IntegratedWeekendLiquidityManager accepts weekend_config overrides."""

    def test_default_caps(self):
        """Without config, defaults are conservative.

        [P42 2026-04-25] mult lowered 2.0 → 1.0 to match operator's live
        config (no surprise default for 24/7 crypto). Base bps now also
        configurable; default 20 (was hardcoded 33)."""
        from liquidity.integrated_manager import IntegratedWeekendLiquidityManager
        mgr = IntegratedWeekendLiquidityManager()
        assert mgr.config.cap_weekend == pytest.approx(0.50)
        assert mgr.config.reduce_at_hours == 12
        assert mgr.config.flatten_at_hours == 36
        assert mgr.config.min_alpha_multiplier_weekend == pytest.approx(1.0)
        assert mgr.config.weekend_min_alpha_bps == pytest.approx(20.0)

    def test_ultra_caps(self):
        """With ULTRA config, caps are relaxed."""
        from liquidity.integrated_manager import IntegratedWeekendLiquidityManager
        mgr = IntegratedWeekendLiquidityManager(weekend_config=_ultra_weekend_config())
        assert mgr.config.cap_weekend == pytest.approx(1.00)
        assert mgr.config.reduce_at_hours == 999
        assert mgr.config.flatten_at_hours == 999
        assert mgr.config.min_alpha_multiplier_weekend == pytest.approx(1.0)
        assert mgr.config.min_confidence_weekend == pytest.approx(0.50)

    def test_ultra_per_asset_caps(self):
        """Per-asset weekend caps are overridden."""
        from liquidity.integrated_manager import IntegratedWeekendLiquidityManager
        mgr = IntegratedWeekendLiquidityManager(weekend_config=_ultra_weekend_config())
        assert mgr.config.asset_cap_weekend["BTC"] == pytest.approx(0.60)
        assert mgr.config.asset_cap_weekend["ETH"] == pytest.approx(0.60)
        assert mgr.config.asset_cap_weekend["SOL"] == pytest.approx(0.80)

    def test_default_per_asset_caps(self):
        """Default per-asset caps are conservative."""
        from liquidity.integrated_manager import IntegratedWeekendLiquidityManager
        mgr = IntegratedWeekendLiquidityManager()
        assert mgr.config.asset_cap_weekend["BTC"] == pytest.approx(0.30)
        assert mgr.config.asset_cap_weekend["SOL"] == pytest.approx(0.25)

    def test_partial_config_only_overrides_specified(self):
        """Only specified fields are overridden, others keep defaults."""
        from liquidity.integrated_manager import IntegratedWeekendLiquidityManager
        partial = {"weekend_max_exposure": 0.80}
        mgr = IntegratedWeekendLiquidityManager(weekend_config=partial)
        assert mgr.config.cap_weekend == pytest.approx(0.80)
        # Not overridden - keep defaults
        assert mgr.config.reduce_at_hours == 12
        assert mgr.config.flatten_at_hours == 36


# ─────────────────────────────────────────────────────────────────────────
# Test: Singleton accepts config on first call
# ─────────────────────────────────────────────────────────────────────────

class TestSingletonConfig:
    """Singleton get_* functions accept config on first call."""

    def test_risk_veto_classifier_singleton(self):
        """get_risk_veto_classifier passes weekend_config on first call."""
        import defense.production_reliability as mod
        # Reset singleton
        mod._risk_veto_classifier = None
        try:
            from defense.production_reliability import SoftVetoCondition
            clf = mod.get_risk_veto_classifier(weekend_config=_ultra_weekend_config())
            assert clf.SOFT_EXPOSURE_CAPS[SoftVetoCondition.WEEKEND_LIQUIDITY] == pytest.approx(1.00)
        finally:
            mod._risk_veto_classifier = None

    def test_integrated_liquidity_manager_singleton(self):
        """get_integrated_liquidity_manager passes weekend_config on first call."""
        from liquidity.integrated_manager import (
            get_integrated_liquidity_manager,
            reset_integrated_liquidity_manager,
        )
        reset_integrated_liquidity_manager()
        try:
            mgr = get_integrated_liquidity_manager(weekend_config=_ultra_weekend_config())
            assert mgr.config.cap_weekend == pytest.approx(1.00)
            assert mgr.config.reduce_at_hours == 999
        finally:
            reset_integrated_liquidity_manager()


# ─────────────────────────────────────────────────────────────────────────
# Test: RiskVetoClassifier weekend soft cap
# [P58 2026-04-25] Renamed from TestHardSoftVetoClassifierPatch and
# repointed at the live RiskVetoClassifier — old HardSoftVetoClassifier
# was retired to archive/defense/production_reliability_patches.py.
# Same behavior is tested against the live class.
# ─────────────────────────────────────────────────────────────────────────

class TestRiskVetoClassifierWeekendCap:
    """Verify weekend soft veto cap defaults + ULTRA override on the live classifier."""

    def test_default_weekend_cap_040(self):
        """Default WEEKEND_LIQUIDITY cap = 0.40 (no weekend_config override)."""
        from defense.production_reliability import (
            RiskVetoClassifier, SoftVetoCondition,
        )
        clf = RiskVetoClassifier()  # no weekend_config → default 0.40
        assert clf.SOFT_EXPOSURE_CAPS[SoftVetoCondition.WEEKEND_LIQUIDITY] == pytest.approx(0.40)

    def test_ultra_weekend_cap_overrides(self):
        """ULTRA config (weekend_soft_veto_cap=1.00) overrides default."""
        from defense.production_reliability import (
            RiskVetoClassifier, SoftVetoCondition,
        )
        clf = RiskVetoClassifier(weekend_config=_ultra_weekend_config())
        assert clf.SOFT_EXPOSURE_CAPS[SoftVetoCondition.WEEKEND_LIQUIDITY] == pytest.approx(1.00)


# ─────────────────────────────────────────────────────────────────────────
# Test: is_weekend detection logic
# ─────────────────────────────────────────────────────────────────────────

class TestIsWeekendDetection:
    """_is_weekend_now() logic (tested via direct datetime mocking)."""

    def _check_weekend(self, fake_dt: datetime) -> bool:
        """Replicate _is_weekend_now() logic for testability."""
        wd = fake_dt.weekday()
        h = fake_dt.hour
        if wd == 4 and h >= 21:
            return True
        if wd == 5:
            return True
        if wd == 6 and h < 21:
            return True
        return False

    def test_friday_20_not_weekend(self):
        """Friday 20:00 UTC is NOT weekend."""
        dt = datetime(2026, 2, 20, 20, 0)  # Friday
        assert dt.weekday() == 4
        assert not self._check_weekend(dt)

    def test_friday_21_is_weekend(self):
        """Friday 21:00 UTC IS weekend."""
        dt = datetime(2026, 2, 20, 21, 0)  # Friday
        assert self._check_weekend(dt)

    def test_saturday_is_weekend(self):
        """All of Saturday IS weekend."""
        dt = datetime(2026, 2, 21, 12, 0)  # Saturday
        assert dt.weekday() == 5
        assert self._check_weekend(dt)

    def test_sunday_morning_is_weekend(self):
        """Sunday 10:00 UTC IS weekend."""
        dt = datetime(2026, 2, 22, 10, 0)  # Sunday
        assert dt.weekday() == 6
        assert self._check_weekend(dt)

    def test_sunday_21_not_weekend(self):
        """Sunday 21:00 UTC is NOT weekend (normal resumes)."""
        dt = datetime(2026, 2, 22, 21, 0)  # Sunday
        assert not self._check_weekend(dt)

    def test_monday_not_weekend(self):
        """Monday is NOT weekend."""
        dt = datetime(2026, 2, 23, 12, 0)  # Monday
        assert dt.weekday() == 0
        assert not self._check_weekend(dt)

    def test_wednesday_not_weekend(self):
        """Wednesday is NOT weekend."""
        dt = datetime(2026, 2, 25, 15, 0)  # Wednesday
        assert dt.weekday() == 2
        assert not self._check_weekend(dt)


# ─────────────────────────────────────────────────────────────────────────
# Test: Config JSON validation
# ─────────────────────────────────────────────────────────────────────────

class TestRealUltraWeekendConfig:
    """Validate the actual ultra_aggressive_5y.json weekend_manager section."""

    @pytest.fixture
    def ultra_config(self) -> Dict:
        path = Path(__file__).parent.parent / "configs" / "ultra_aggressive_5y.json"
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def test_has_weekend_manager_section(self, ultra_config):
        """JSON has weekend_manager section."""
        assert "weekend_manager" in ultra_config

    def test_weekend_soft_veto_cap(self, ultra_config):
        """weekend_soft_veto_cap = 1.00."""
        wc = ultra_config["weekend_manager"]
        assert wc["weekend_soft_veto_cap"] == 1.00

    def test_weekend_max_exposure(self, ultra_config):
        """weekend_max_exposure = 1.00."""
        wc = ultra_config["weekend_manager"]
        assert wc["weekend_max_exposure"] == 1.00

    def test_weekend_leverage_cap(self, ultra_config):
        """weekend_leverage_cap = 3.0."""
        wc = ultra_config["weekend_manager"]
        assert wc["weekend_leverage_cap"] == 3.0

    def test_reduce_friday_disabled(self, ultra_config):
        """reduce_friday_evening = false."""
        wc = ultra_config["weekend_manager"]
        assert wc["reduce_friday_evening"] is False

    def test_reduce_flatten_at_disabled(self, ultra_config):
        """reduce_at_hours and flatten_at_hours effectively disabled."""
        wc = ultra_config["weekend_manager"]
        assert wc["reduce_at_hours"] >= 999
        assert wc["flatten_at_hours"] >= 999

    def test_per_asset_caps(self, ultra_config):
        """Per-asset weekend caps match profile."""
        wc = ultra_config["weekend_manager"]
        caps = wc["per_asset_weekend_caps"]
        assert caps["BTC"] == 0.60
        assert caps["ETH"] == 0.60
        assert caps["SOL"] == 0.80

    def test_alpha_and_confidence_relaxed(self, ultra_config):
        """Alpha multiplier and confidence thresholds are relaxed."""
        wc = ultra_config["weekend_manager"]
        assert wc["min_alpha_multiplier_weekend"] == 1.0
        assert wc["min_confidence_weekend"] == 0.50


# ─────────────────────────────────────────────────────────────────────────
# Test: ProductionConfig from_file parsing
# ─────────────────────────────────────────────────────────────────────────

class TestProductionConfigWeekend:
    """ProductionConfig correctly parses weekend_config from profile."""

    def test_ultra_has_weekend_config(self):
        """ULTRA profile sets weekend_config on ProductionConfig."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from main import ProductionConfig
        path = Path(__file__).parent.parent / "configs" / "ultra_aggressive_5y.json"
        config = ProductionConfig.from_file(path)
        assert config.weekend_config is not None
        assert config.weekend_config["weekend_soft_veto_cap"] == 1.00

    def test_default_has_no_weekend_config(self):
        """Default config has no weekend_config."""
        from main import ProductionConfig
        path = Path(__file__).parent.parent / "configs" / "cloud_production.json"
        if not path.exists():
            pytest.skip("cloud_production.json not found")
        config = ProductionConfig.from_file(path)
        assert config.weekend_config is None
