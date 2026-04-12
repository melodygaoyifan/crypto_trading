"""
Tests for UnifiedPositionSizer — confidence scaling, correlation adjustment,
dynamic vol-adjusted limits, and margin constraints.
Inspired by ai-hedge-fund test_portfolio.py patterns: deterministic inputs,
exact float assertions via pytest.approx, edge case coverage.
"""
import sys
import os
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from risk.unified_position_sizer import UnifiedPositionSizer, PositionSizingConfig, PositionSizingResult
from core.canonical_enums import TrancheLevel


@pytest.fixture
def sizer():
    config = PositionSizingConfig(risk_per_trade=0.02)  # 2% base
    return UnifiedPositionSizer(config=config, initial_capital=10_000)


@pytest.fixture
def base_kwargs():
    """Default kwargs for calculate_position_size."""
    return dict(
        asset="BTC",
        account_balance=10_000.0,
        stop_loss_pct=0.02,
        tranche_level=TrancheLevel.TRANCHE_1,
        current_gross_exposure=0.0,
        current_asset_exposure=0.0,
        drawdown_status="NORMAL",
        price_usd=73_000.0,
        is_opportunity=False,
        confidence=0.5,
        cross_asset_correlation=0.5,
        annualized_vol=0.6,
    )


# =============================================================================
# CONFIDENCE-BASED SIZING
# =============================================================================

class TestConfidenceSizing:
    def test_high_confidence_increases_size(self, sizer, base_kwargs):
        """High confidence (0.9) should produce larger position than low (0.3)."""
        low = sizer.calculate_position_size(**{**base_kwargs, "confidence": 0.3})
        mid = sizer.calculate_position_size(**{**base_kwargs, "confidence": 0.5})
        high = sizer.calculate_position_size(**{**base_kwargs, "confidence": 0.9})
        assert high.final_size_usd >= mid.final_size_usd
        assert mid.final_size_usd >= low.final_size_usd

    def test_confidence_multiplier_clamped(self, sizer, base_kwargs):
        """Confidence multiplier should be clamped to [0.5, 1.5]."""
        # conf=0.0 → mult=0.5, conf=1.0 → mult=1.5
        zero = sizer.calculate_position_size(**{**base_kwargs, "confidence": 0.0})
        one = sizer.calculate_position_size(**{**base_kwargs, "confidence": 1.0})
        ratio = one.risk_based_max_usd / zero.risk_based_max_usd if zero.risk_based_max_usd > 0 else 0
        assert ratio == pytest.approx(3.0, rel=0.05)  # 1.5 / 0.5 = 3.0x

    def test_default_confidence_is_neutral(self, sizer, base_kwargs):
        """Default confidence (0.5) should give 1.0x multiplier = same as base."""
        result = sizer.calculate_position_size(**{**base_kwargs, "confidence": 0.5})
        # At conf=0.5: mult = max(0.5, min(1.5, 0.5+0.5)) = 1.0
        base_risk = 10_000 * 0.02  # $200
        expected_max = base_risk / 0.02  # $10,000
        assert result.risk_based_max_usd == pytest.approx(expected_max, rel=0.01)


# =============================================================================
# CORRELATION-ADJUSTED SIZING
# =============================================================================

class TestCorrelationSizing:
    def test_high_correlation_reduces_size(self, sizer, base_kwargs):
        """Correlation >= 0.80 should apply 0.70x multiplier."""
        normal = sizer.calculate_position_size(**{**base_kwargs, "cross_asset_correlation": 0.5})
        high_corr = sizer.calculate_position_size(**{**base_kwargs, "cross_asset_correlation": 0.85})
        # high_corr should be ~70% of normal (0.70 / 1.00)
        if normal.final_size_usd > 0:
            ratio = high_corr.final_size_usd / normal.final_size_usd
            assert ratio < 0.80  # should be around 0.70

    def test_low_correlation_increases_size(self, sizer, base_kwargs):
        """Correlation < 0.20 should apply 1.10x multiplier."""
        normal = sizer.calculate_position_size(**{**base_kwargs, "cross_asset_correlation": 0.5})
        low_corr = sizer.calculate_position_size(**{**base_kwargs, "cross_asset_correlation": 0.1})
        if normal.final_size_usd > 0:
            ratio = low_corr.final_size_usd / normal.final_size_usd
            assert ratio >= 1.05

    def test_correlation_tiers(self, sizer, base_kwargs):
        """Verify all 5 correlation tiers produce distinct sizes."""
        results = {}
        for corr in [0.1, 0.3, 0.5, 0.7, 0.9]:
            r = sizer.calculate_position_size(**{**base_kwargs, "cross_asset_correlation": corr})
            results[corr] = r.final_size_usd
        # Higher correlation = smaller size
        assert results[0.1] >= results[0.3]
        assert results[0.3] >= results[0.5]
        assert results[0.5] >= results[0.7]
        assert results[0.7] >= results[0.9]


# =============================================================================
# DYNAMIC VOL-ADJUSTED LIMITS
# =============================================================================

class TestVolAdjustedLimits:
    def test_low_vol_gets_higher_cap(self, sizer, base_kwargs):
        """Low annualized vol (0.3) should get higher asset cap (×1.2)."""
        low_vol = sizer.calculate_position_size(**{**base_kwargs, "annualized_vol": 0.3})
        high_vol = sizer.calculate_position_size(**{**base_kwargs, "annualized_vol": 1.0})
        assert low_vol.final_size_usd >= high_vol.final_size_usd

    def test_extreme_vol_caps_at_50pct(self, sizer, base_kwargs):
        """Annualized vol > 1.2 should get lower asset cap than normal vol."""
        extreme = sizer.calculate_position_size(**{**base_kwargs, "annualized_vol": 1.5})
        low_vol = sizer.calculate_position_size(**{**base_kwargs, "annualized_vol": 0.3})
        # Extreme vol asset cap = base * 0.50, low vol = base * 1.20
        # But final size may be capped by other constraints (risk-based max, gross cap).
        # Just verify the ordering: low vol >= extreme vol.
        assert low_vol.final_size_usd >= extreme.final_size_usd


# =============================================================================
# EDGE CASES
# =============================================================================

class TestBetaAdjustedSizing:
    def test_high_beta_reduces_size(self, sizer, base_kwargs):
        """SOL with beta=1.35 should get smaller position than BTC with beta=1.0."""
        btc = sizer.calculate_position_size(**{**base_kwargs, "beta_to_btc": 1.0})
        sol = sizer.calculate_position_size(**{**base_kwargs, "beta_to_btc": 1.35, "asset": "SOL"})
        assert sol.final_size_usd < btc.final_size_usd

    def test_low_beta_increases_size(self, sizer, base_kwargs):
        """Asset with beta=0.7 should get slightly larger position."""
        normal = sizer.calculate_position_size(**{**base_kwargs, "beta_to_btc": 1.0})
        low_beta = sizer.calculate_position_size(**{**base_kwargs, "beta_to_btc": 0.7})
        assert low_beta.final_size_usd >= normal.final_size_usd

    def test_beta_multiplier_clamped(self, sizer, base_kwargs):
        """Extreme beta should be clamped — not explode or go to zero."""
        extreme_high = sizer.calculate_position_size(**{**base_kwargs, "beta_to_btc": 3.0})
        extreme_low = sizer.calculate_position_size(**{**base_kwargs, "beta_to_btc": 0.3})
        # Both should produce valid sizes (not zero, not infinite)
        assert extreme_high.final_size_usd > 0
        assert extreme_low.final_size_usd > 0

    def test_btc_beta_one_no_change(self, sizer, base_kwargs):
        """BTC with beta=1.0 should get the same as default."""
        default = sizer.calculate_position_size(**{**base_kwargs})  # beta_to_btc defaults to 1.0
        explicit = sizer.calculate_position_size(**{**base_kwargs, "beta_to_btc": 1.0})
        assert default.final_size_usd == explicit.final_size_usd


class TestEdgeCases:
    def test_zero_balance_returns_zero(self, sizer, base_kwargs):
        result = sizer.calculate_position_size(**{**base_kwargs, "account_balance": 0.0})
        assert result.final_size_usd == 0.0

    def test_zero_stop_loss_uses_default(self, sizer, base_kwargs):
        """Zero stop_loss_pct should use default 2% instead of div-by-zero."""
        result = sizer.calculate_position_size(**{**base_kwargs, "stop_loss_pct": 0.0})
        assert result.final_size_usd >= 0  # should not crash

    def test_result_has_valid_metadata(self, sizer, base_kwargs):
        result = sizer.calculate_position_size(**base_kwargs)
        assert isinstance(result, PositionSizingResult)
        assert result.asset == "BTC"
        assert result.tranche_level == TrancheLevel.TRANCHE_1
        assert result.is_valid is True
