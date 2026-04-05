"""
tests/test_dynamic_limits.py - P1-1D Dynamic Hard Limits tests

Verifies:
1. HIGH_RISK defaults: disabled -> static fallback values returned
2. ULTRA enabled: regime multipliers applied correctly
3. Confidence multiplier: >0.80 -> ×1.10
4. Volatility pullback: vol_z > 2.0 -> ×0.90 leverage
5. Absolute clamping: never exceeds abs_max
6. Combined multipliers: regime + confidence + volatility
7. Config round-trip from JSON
8. Edge cases: unknown regime, None inputs
"""

import json
import pytest
from pathlib import Path

from risk.dynamic_limits import (
    DynamicLimitsConfig,
    DynamicLimits,
    DynamicLimitsResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ULTRA_CONFIG = {
    "enabled": True,
    "base_max_leverage": 4.0,
    "abs_max_leverage": 5.0,
    "base_max_gross_exposure": 2.50,
    "abs_max_gross_exposure": 3.0,
    "base_max_single_asset": 1.0,
    "correlation_limit": 0.98,
    "regime_multipliers": {
        "STRONG_TREND":  {"leverage": 1.25, "gross": 1.20},
        "IGNITION":      {"leverage": 1.25, "gross": 1.20},
        "EXPANSION":     {"leverage": 1.10, "gross": 1.10},
        "DEFAULT":       {"leverage": 1.00, "gross": 1.00},
    },
    "confidence_multiplier_threshold": 0.80,
    "confidence_multiplier": 1.10,
    "volatility_high_threshold": 2.0,
    "volatility_high_leverage_multiplier": 0.90,
}


def _make_dynamic_limits(cfg_dict=None) -> DynamicLimits:
    """Create DynamicLimits from config dict."""
    cfg = DynamicLimitsConfig.from_dict(cfg_dict or ULTRA_CONFIG)
    return DynamicLimits(config=cfg)


# ===========================================================================
# Test Group 1: HIGH_RISK defaults (disabled)
# ===========================================================================

class TestHighRiskDisabled:
    """When disabled, returns static fallback values unchanged."""

    def test_disabled_returns_fallback(self):
        dl = DynamicLimits(config=DynamicLimitsConfig(enabled=False))
        result = dl.get_limits(
            fallback_max_leverage=3.0,
            fallback_max_gross=1.50,
            fallback_max_single_asset=0.80,
            fallback_correlation_limit=0.95,
        )
        assert result.max_leverage == 3.0
        assert result.max_gross_exposure == 1.50
        assert result.max_single_asset == 0.80
        assert result.correlation_limit == 0.95
        assert result.dynamic_active is False

    def test_disabled_ignores_regime(self):
        dl = DynamicLimits(config=DynamicLimitsConfig(enabled=False))
        result = dl.get_limits(
            regime="STRONG_TREND",
            fallback_max_leverage=3.0,
            fallback_max_gross=1.50,
            fallback_max_single_asset=0.80,
            fallback_correlation_limit=0.95,
        )
        assert result.max_leverage == 3.0
        assert result.dynamic_active is False

    def test_empty_config_disabled_by_default(self):
        dl = DynamicLimits(config=DynamicLimitsConfig.from_dict({}))
        result = dl.get_limits()
        assert result.dynamic_active is False


# ===========================================================================
# Test Group 2: Regime multipliers
# ===========================================================================

class TestRegimeMultipliers:
    """Regime-specific leverage and gross multipliers."""

    def test_strong_trend_boost(self):
        dl = _make_dynamic_limits()
        result = dl.get_limits(regime="STRONG_TREND")
        assert result.max_leverage == pytest.approx(5.0)  # 4.0 × 1.25 = 5.0 (clamped)
        assert result.max_gross_exposure == pytest.approx(3.0)  # 2.5 × 1.20 = 3.0 (clamped)
        assert result.regime_used == "STRONG_TREND"
        assert result.dynamic_active is True

    def test_ignition_boost(self):
        dl = _make_dynamic_limits()
        result = dl.get_limits(regime="IGNITION")
        assert result.max_leverage == pytest.approx(5.0)  # 4.0 × 1.25 = 5.0
        assert result.max_gross_exposure == pytest.approx(3.0)  # 2.5 × 1.20 = 3.0
        assert result.regime_used == "IGNITION"

    def test_expansion_moderate_boost(self):
        dl = _make_dynamic_limits()
        result = dl.get_limits(regime="EXPANSION")
        assert result.max_leverage == pytest.approx(4.4)  # 4.0 × 1.10 = 4.4
        assert result.max_gross_exposure == pytest.approx(2.75)  # 2.5 × 1.10 = 2.75

    def test_default_regime_no_boost(self):
        dl = _make_dynamic_limits()
        result = dl.get_limits(regime="VOLATILE_CHOP")  # Not in multipliers -> DEFAULT
        assert result.max_leverage == pytest.approx(4.0)  # 4.0 × 1.0 = 4.0
        assert result.max_gross_exposure == pytest.approx(2.50)  # 2.5 × 1.0 = 2.5

    def test_regime_case_insensitive(self):
        dl = _make_dynamic_limits()
        result = dl.get_limits(regime="strong_trend")  # lowercase -> uppercased
        assert result.regime_used == "STRONG_TREND"
        assert result.max_leverage == pytest.approx(5.0)


# ===========================================================================
# Test Group 3: Confidence multiplier
# ===========================================================================

class TestConfidenceMultiplier:
    """Confidence > threshold -> ×1.10 boost."""

    def test_high_confidence_boost(self):
        dl = _make_dynamic_limits()
        result = dl.get_limits(confidence=0.90)  # > 0.80
        assert result.max_leverage == pytest.approx(4.4)  # 4.0 × 1.10 = 4.4
        assert result.max_gross_exposure == pytest.approx(2.75)  # 2.5 × 1.10
        assert result.confidence_applied is True

    def test_low_confidence_no_boost(self):
        dl = _make_dynamic_limits()
        result = dl.get_limits(confidence=0.70)  # < 0.80
        assert result.max_leverage == pytest.approx(4.0)
        assert result.confidence_applied is False

    def test_exactly_threshold_no_boost(self):
        dl = _make_dynamic_limits()
        result = dl.get_limits(confidence=0.80)  # not > 0.80
        assert result.confidence_applied is False

    def test_none_confidence_no_boost(self):
        dl = _make_dynamic_limits()
        result = dl.get_limits(confidence=None)
        assert result.confidence_applied is False


# ===========================================================================
# Test Group 4: Volatility pullback
# ===========================================================================

class TestVolatilityPullback:
    """High volatility -> leverage pullback (×0.90), gross unaffected."""

    def test_high_vol_reduces_leverage(self):
        dl = _make_dynamic_limits()
        result = dl.get_limits(volatility_z=2.5)  # > 2.0
        assert result.max_leverage == pytest.approx(3.6)  # 4.0 × 0.90 = 3.6
        assert result.max_gross_exposure == pytest.approx(2.50)  # gross unaffected
        assert result.volatility_applied is True

    def test_normal_vol_no_pullback(self):
        dl = _make_dynamic_limits()
        result = dl.get_limits(volatility_z=1.5)  # < 2.0
        assert result.max_leverage == pytest.approx(4.0)
        assert result.volatility_applied is False

    def test_exactly_threshold_no_pullback(self):
        dl = _make_dynamic_limits()
        result = dl.get_limits(volatility_z=2.0)  # not > 2.0
        assert result.volatility_applied is False

    def test_none_vol_no_pullback(self):
        dl = _make_dynamic_limits()
        result = dl.get_limits(volatility_z=None)
        assert result.volatility_applied is False


# ===========================================================================
# Test Group 5: Absolute clamping
# ===========================================================================

class TestAbsoluteClamping:
    """Multipliers never push past absolute caps."""

    def test_regime_plus_confidence_clamped_leverage(self):
        """STRONG_TREND(1.25) × confidence(1.10) = 1.375 -> 4.0 × 1.375 = 5.5 -> clamped to 5.0."""
        dl = _make_dynamic_limits()
        result = dl.get_limits(regime="STRONG_TREND", confidence=0.90)
        assert result.max_leverage == pytest.approx(5.0)  # abs cap
        assert result.leverage_mult == pytest.approx(1.375)  # actual mult before clamping

    def test_regime_plus_confidence_clamped_gross(self):
        """STRONG_TREND(1.20) × confidence(1.10) = 1.32 -> 2.5 × 1.32 = 3.3 -> clamped to 3.0."""
        dl = _make_dynamic_limits()
        result = dl.get_limits(regime="STRONG_TREND", confidence=0.90)
        assert result.max_gross_exposure == pytest.approx(3.0)  # abs cap
        assert result.gross_mult == pytest.approx(1.32)


# ===========================================================================
# Test Group 6: Combined multipliers
# ===========================================================================

class TestCombinedMultipliers:
    """All three factors applied together."""

    def test_strong_trend_high_conf_high_vol(self):
        """STRONG_TREND(1.25) × conf(1.10) × vol(0.90) = 1.2375 -> 4.0 × 1.2375 = 4.95."""
        dl = _make_dynamic_limits()
        result = dl.get_limits(
            regime="STRONG_TREND",
            confidence=0.90,
            volatility_z=2.5,
        )
        assert result.max_leverage == pytest.approx(4.95)  # 4.0 × 1.2375
        assert result.confidence_applied is True
        assert result.volatility_applied is True
        # Gross: STRONG_TREND(1.20) × conf(1.10) = 1.32 -> 2.5 × 1.32 = 3.3 -> clamped 3.0
        assert result.max_gross_exposure == pytest.approx(3.0)

    def test_expansion_low_conf_normal_vol(self):
        """EXPANSION(1.10) × no conf × no vol = 1.10 -> 4.0 × 1.10 = 4.4."""
        dl = _make_dynamic_limits()
        result = dl.get_limits(
            regime="EXPANSION",
            confidence=0.70,
            volatility_z=1.0,
        )
        assert result.max_leverage == pytest.approx(4.4)
        assert result.max_gross_exposure == pytest.approx(2.75)
        assert result.confidence_applied is False
        assert result.volatility_applied is False

    def test_default_high_vol_only(self):
        """DEFAULT(1.0) × vol(0.90) = 0.90 -> 4.0 × 0.90 = 3.6."""
        dl = _make_dynamic_limits()
        result = dl.get_limits(
            regime="UNKNOWN_REGIME",
            volatility_z=3.0,
        )
        assert result.max_leverage == pytest.approx(3.6)
        assert result.max_gross_exposure == pytest.approx(2.50)  # vol doesn't affect gross


# ===========================================================================
# Test Group 7: Config round-trip
# ===========================================================================

class TestConfigRoundTrip:
    """Config from_dict and JSON load."""

    def test_from_dict(self):
        cfg = DynamicLimitsConfig.from_dict(ULTRA_CONFIG)
        assert cfg.enabled is True
        assert cfg.base_max_leverage == 4.0
        assert cfg.abs_max_leverage == 5.0
        assert cfg.base_max_gross_exposure == 2.50
        assert cfg.abs_max_gross_exposure == 3.0
        assert cfg.base_max_single_asset == 1.0
        assert cfg.correlation_limit == 0.98
        assert len(cfg.regime_multipliers) == 4
        assert cfg.confidence_multiplier == 1.10
        assert cfg.volatility_high_threshold == 2.0

    def test_from_dict_defaults(self):
        cfg = DynamicLimitsConfig.from_dict({})
        assert cfg.enabled is False
        assert cfg.base_max_leverage == 4.0
        assert cfg.abs_max_leverage == 5.0

    def test_ultra_json_has_dynamic_limits(self):
        json_path = Path("configs/ultra_aggressive_5y.json")
        if not json_path.exists():
            pytest.skip("ultra_aggressive_5y.json not found")
        with open(json_path) as f:
            data = json.load(f)
        dl = data.get("dynamic_limits")
        assert dl is not None
        assert dl["enabled"] is True
        assert dl["base_max_leverage"] == 4.0
        assert dl["abs_max_leverage"] == 5.0
        assert dl["base_max_gross_exposure"] == 2.50
        assert dl["abs_max_gross_exposure"] == 3.0
        assert "STRONG_TREND" in dl["regime_multipliers"]
        assert dl["regime_multipliers"]["STRONG_TREND"]["leverage"] == 1.25

    def test_cloud_production_has_no_dynamic_limits(self):
        json_path = Path("configs/cloud_production.json")
        if not json_path.exists():
            pytest.skip("cloud_production.json not found")
        with open(json_path) as f:
            data = json.load(f)
        assert data.get("dynamic_limits") is None


# ===========================================================================
# Test Group 8: Edge cases
# ===========================================================================

class TestEdgeCases:
    """Edge cases: empty regime, all None inputs."""

    def test_empty_regime_uses_default(self):
        dl = _make_dynamic_limits()
        result = dl.get_limits(regime="")
        assert result.regime_used == ""
        assert result.max_leverage == pytest.approx(4.0)  # DEFAULT mult

    def test_all_none_inputs(self):
        dl = _make_dynamic_limits()
        result = dl.get_limits()
        assert result.dynamic_active is True
        assert result.max_leverage == pytest.approx(4.0)
        assert result.confidence_applied is False
        assert result.volatility_applied is False

    def test_result_fields_populated(self):
        dl = _make_dynamic_limits()
        result = dl.get_limits(
            regime="EXPANSION",
            confidence=0.85,
            volatility_z=2.5,
        )
        assert result.dynamic_active is True
        assert result.regime_used == "EXPANSION"
        assert result.leverage_mult == pytest.approx(1.10 * 1.10 * 0.90)  # regime × conf × vol
        assert result.gross_mult == pytest.approx(1.10 * 1.10)  # regime × conf (no vol)
        assert result.confidence_applied is True
        assert result.volatility_applied is True
        assert result.max_single_asset == 1.0
        assert result.correlation_limit == 0.98

    def test_static_fields_passthrough(self):
        """max_single_asset and correlation_limit come from config, not computed."""
        dl = _make_dynamic_limits()
        result = dl.get_limits()
        assert result.max_single_asset == 1.0
        assert result.correlation_limit == 0.98
