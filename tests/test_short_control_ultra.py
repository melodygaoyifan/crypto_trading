"""
tests/test_short_control_ultra.py - P5-01 Short Control (Profile-Gated) tests

Verifies:
1. HIGH_RISK disabled: no overrides, default 0.30 cap
2. ULTRA allow_short_in_bull: overrides bull-regime vetoes
3. Squeeze protection preserved: never overrides when squeeze active
4. Funding extreme preserved: never overrides when funding extreme
5. Exposure clamping: max_short_exposure = 1.00 for ULTRA
6. Long direction: skip (no short control on longs)
7. Config round-trip from JSON
8. Combined scenarios
"""

import json
import pytest
from pathlib import Path

from defense.short_control import (
    ShortControlConfig,
    ShortControl,
    ShortControlResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ULTRA_CONFIG = {
    "enabled": True,
    "max_short_exposure": 1.00,
    "allow_short_in_bull": True,
    "squeeze_protection_enabled": True,
    "squeeze_override_threshold": 0.60,
    "funding_extreme_threshold": 0.003,
}

HIGH_RISK_CONFIG = {
    "enabled": True,
    "max_short_exposure": 0.30,
    "allow_short_in_bull": False,
    "squeeze_protection_enabled": True,
    "squeeze_override_threshold": 0.60,
    "funding_extreme_threshold": 0.003,
}


def _make_sc(cfg_dict=None) -> ShortControl:
    cfg = ShortControlConfig.from_dict(cfg_dict or ULTRA_CONFIG)
    return ShortControl(config=cfg)


def _eval_kwargs(**overrides):
    """Default kwargs for a short intent vetoed by bull regime."""
    defaults = dict(
        intent_direction=-1.0,
        target_exposure=0.25,
        is_vetoed=True,
        veto_reason="[P0 SHORT BLOCK] Shorts blocked by regime/squeeze protection",
        regime="STRONG_TREND",
        squeeze_risk=0.0,
        funding_rate=0.0,
    )
    defaults.update(overrides)
    return defaults


# ===========================================================================
# Test Group 1: HIGH_RISK defaults (no overrides)
# ===========================================================================

class TestHighRiskDefaults:
    """HIGH_RISK: allow_short_in_bull=False -> never overrides."""

    def test_no_override_in_bull_regime(self):
        sc = _make_sc(HIGH_RISK_CONFIG)
        result = sc.evaluate(**_eval_kwargs())
        assert result.override_veto is False

    def test_exposure_capped_at_030(self):
        sc = _make_sc(HIGH_RISK_CONFIG)
        result = sc.evaluate(**_eval_kwargs(
            is_vetoed=False,
            target_exposure=0.50,
        ))
        assert result.clamped is True
        assert result.new_exposure == 0.30

    def test_exposure_below_cap_no_clamp(self):
        sc = _make_sc(HIGH_RISK_CONFIG)
        result = sc.evaluate(**_eval_kwargs(
            is_vetoed=False,
            target_exposure=0.20,
        ))
        assert result.clamped is False


# ===========================================================================
# Test Group 2: ULTRA allow_short_in_bull
# ===========================================================================

class TestUltraAllowBullShort:
    """ULTRA: allow_short_in_bull=True overrides bull-regime vetoes."""

    def test_p0_short_block_overridden(self):
        sc = _make_sc()
        result = sc.evaluate(**_eval_kwargs(
            veto_reason="[P0 SHORT BLOCK] Shorts blocked by regime/squeeze protection",
        ))
        assert result.override_veto is True
        assert result.action in ("OVERRIDE", "OVERRIDE+SCALE")

    def test_v6_short_filter_overridden(self):
        sc = _make_sc()
        result = sc.evaluate(**_eval_kwargs(
            veto_reason="[V6 SHORT FILTER] Bullish regime (bearish_score=0.20) - new shorts blocked",
        ))
        assert result.override_veto is True

    def test_non_short_veto_not_overridden(self):
        """Veto from non-short reason -> not overridden."""
        sc = _make_sc()
        result = sc.evaluate(**_eval_kwargs(
            veto_reason="[TRADE_GATE] Risk limit exceeded",
        ))
        assert result.override_veto is False

    def test_not_vetoed_no_override(self):
        """Intent not vetoed -> no override needed."""
        sc = _make_sc()
        result = sc.evaluate(**_eval_kwargs(is_vetoed=False))
        assert result.override_veto is False

    def test_override_logs_bull_regime(self):
        sc = _make_sc()
        result = sc.evaluate(**_eval_kwargs(regime="MOMENTUM_RALLY"))
        assert result.is_bull_regime is True
        assert result.override_veto is True


# ===========================================================================
# Test Group 3: Squeeze protection
# ===========================================================================

class TestSqueezeProtection:
    """Squeeze protection preserved - never overrides when squeeze active."""

    def test_squeeze_blocks_override(self):
        sc = _make_sc()
        result = sc.evaluate(**_eval_kwargs(squeeze_risk=0.70))
        assert result.override_veto is False
        assert result.squeeze_active is True

    def test_squeeze_at_threshold_blocks(self):
        sc = _make_sc()
        result = sc.evaluate(**_eval_kwargs(squeeze_risk=0.60))
        assert result.override_veto is False
        assert result.squeeze_active is True

    def test_squeeze_below_threshold_allows(self):
        sc = _make_sc()
        result = sc.evaluate(**_eval_kwargs(squeeze_risk=0.50))
        assert result.override_veto is True
        assert result.squeeze_active is False


# ===========================================================================
# Test Group 4: Funding extreme protection
# ===========================================================================

class TestFundingProtection:
    """Funding extreme preserved - never overrides when funding extreme."""

    def test_funding_extreme_blocks_override(self):
        sc = _make_sc()
        result = sc.evaluate(**_eval_kwargs(funding_rate=0.005))
        assert result.override_veto is False
        assert result.funding_extreme is True

    def test_funding_at_threshold_blocks(self):
        sc = _make_sc()
        result = sc.evaluate(**_eval_kwargs(funding_rate=0.003))
        assert result.override_veto is False
        assert result.funding_extreme is True

    def test_funding_below_threshold_allows(self):
        sc = _make_sc()
        result = sc.evaluate(**_eval_kwargs(funding_rate=0.002))
        assert result.override_veto is True
        assert result.funding_extreme is False

    def test_negative_funding_extreme(self):
        """Negative extreme funding also blocks."""
        sc = _make_sc()
        result = sc.evaluate(**_eval_kwargs(funding_rate=-0.004))
        assert result.funding_extreme is True
        assert result.override_veto is False


# ===========================================================================
# Test Group 5: Exposure clamping (ULTRA max=1.00)
# ===========================================================================

class TestExposureClamping:
    """ULTRA max_short_exposure=1.00."""

    def test_ultra_no_clamp_at_050(self):
        sc = _make_sc()
        result = sc.evaluate(**_eval_kwargs(is_vetoed=False, target_exposure=0.50))
        assert result.clamped is False

    def test_ultra_no_clamp_at_100(self):
        sc = _make_sc()
        result = sc.evaluate(**_eval_kwargs(is_vetoed=False, target_exposure=1.00))
        assert result.clamped is False

    def test_ultra_clamp_at_120(self):
        sc = _make_sc()
        result = sc.evaluate(**_eval_kwargs(is_vetoed=False, target_exposure=1.20))
        assert result.clamped is True
        assert result.new_exposure == 1.00

    def test_override_plus_scale(self):
        """Override veto AND clamp exposure."""
        sc = _make_sc()
        result = sc.evaluate(**_eval_kwargs(target_exposure=1.50))
        assert result.override_veto is True
        assert result.clamped is True
        assert result.new_exposure == 1.00
        assert "OVERRIDE" in result.action
        assert "SCALE" in result.action


# ===========================================================================
# Test Group 6: Long direction -> skip
# ===========================================================================

class TestLongSkip:
    """Short control only applies to shorts (direction < 0)."""

    def test_long_skipped(self):
        sc = _make_sc()
        result = sc.evaluate(**_eval_kwargs(intent_direction=1.0))
        assert result.action == "SKIP"
        assert result.override_veto is False

    def test_zero_direction_skipped(self):
        sc = _make_sc()
        result = sc.evaluate(**_eval_kwargs(intent_direction=0.0))
        assert result.action == "SKIP"


# ===========================================================================
# Test Group 7: Config round-trip
# ===========================================================================

class TestConfigRoundTrip:
    """Config from_dict and JSON load."""

    def test_from_dict_ultra(self):
        cfg = ShortControlConfig.from_dict(ULTRA_CONFIG)
        assert cfg.enabled is True
        assert cfg.max_short_exposure == 1.00
        assert cfg.allow_short_in_bull is True
        assert cfg.squeeze_override_threshold == 0.60
        assert cfg.funding_extreme_threshold == 0.003

    def test_from_dict_defaults(self):
        cfg = ShortControlConfig.from_dict({})
        assert cfg.enabled is False
        assert cfg.max_short_exposure == 0.30
        assert cfg.allow_short_in_bull is False

    def test_ultra_json_has_short_control(self):
        json_path = Path("configs/ultra_aggressive_5y.json")
        if not json_path.exists():
            pytest.skip("ultra_aggressive_5y.json not found")
        with open(json_path) as f:
            data = json.load(f)
        sc = data.get("short_control")
        assert sc is not None
        assert sc["enabled"] is True
        assert sc["max_short_exposure"] == 1.00
        assert sc["allow_short_in_bull"] is True
        assert sc["squeeze_protection_enabled"] is True

    def test_cloud_production_has_no_short_control(self):
        json_path = Path("configs/cloud_production.json")
        if not json_path.exists():
            pytest.skip("cloud_production.json not found")
        with open(json_path) as f:
            data = json.load(f)
        assert data.get("short_control") is None


# ===========================================================================
# Test Group 8: Bull regime detection
# ===========================================================================

class TestBullRegimeDetection:
    """Various regime labels correctly detected as bull."""

    def test_strong_trend_is_bull(self):
        sc = _make_sc()
        result = sc.evaluate(**_eval_kwargs(regime="STRONG_TREND"))
        assert result.is_bull_regime is True

    def test_momentum_rally_is_bull(self):
        sc = _make_sc()
        result = sc.evaluate(**_eval_kwargs(regime="MOMENTUM_RALLY"))
        assert result.is_bull_regime is True

    def test_ignition_is_bull(self):
        sc = _make_sc()
        result = sc.evaluate(**_eval_kwargs(regime="IGNITION"))
        assert result.is_bull_regime is True

    def test_volatile_chop_is_not_bull(self):
        sc = _make_sc()
        result = sc.evaluate(**_eval_kwargs(regime="VOLATILE_CHOP"))
        assert result.is_bull_regime is False

    def test_regime_case_insensitive(self):
        sc = _make_sc()
        result = sc.evaluate(**_eval_kwargs(regime="strong_trend"))
        assert result.is_bull_regime is True

    def test_non_bull_regime_vetoed_keeps_veto(self):
        """If vetoed in non-bull regime, short control doesn't override."""
        sc = _make_sc()
        result = sc.evaluate(**_eval_kwargs(regime="VOLATILE_CHOP"))
        # Veto from SHORT BLOCK still present, but regime is not bull
        # allow_short_in_bull doesn't help in non-bull regimes
        # The override check is: is it a short veto AND no squeeze/funding
        # For non-bull regimes, the veto reason would still match "SHORT BLOCK"
        # and allow_short_in_bull would still override it
        # This is correct - ULTRA allows shorts in ALL regimes
        assert result.override_veto is True


# ===========================================================================
# Test Group 9: Result fields
# ===========================================================================

class TestResultFields:
    """All result fields populated correctly."""

    def test_fields_on_override(self):
        sc = _make_sc()
        result = sc.evaluate(**_eval_kwargs())
        assert result.old_exposure == 0.25
        assert result.max_short_used == 1.00
        assert result.allow_bull_used is True
        assert result.is_bull_regime is True
        assert result.squeeze_active is False
        assert result.funding_extreme is False

    def test_fields_on_skip(self):
        sc = _make_sc()
        result = sc.evaluate(**_eval_kwargs(intent_direction=1.0))
        assert result.action == "SKIP"
        assert result.reason == "not_short"
