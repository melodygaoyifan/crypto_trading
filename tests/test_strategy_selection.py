"""
Tests for Best-of-N strategy selection — verifying that Hurst, vol_regime,
convexity, z-score, and consistency scoring correctly modulate strategy fitness.
"""
import sys
import os
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data_mgmt.market_data_pipeline import MarketDataPipeline


def _select(**kwargs):
    """Helper: run _compute_strategy_signals with defaults for unspecified params."""
    defaults = dict(
        rsi_sig=0.0, macd_sig=0.0, bb_sig=0.0, ema_sig=0.0,
        adx_mult=1.0, rsi_val=50.0, adx_val=20.0, bb_pos=0.0,
        volume_ratio=1.0, regime="QUIET_ACCUMULATION",
        hurst=0.5, vol_regime=1.0, z_score_50=0.0,
        mtf_momentum=0.0, convexity=1.0, vol_z_score=0.0,
        vov_spike_ratio=1.0, rsi_consistency=0.0,
    )
    defaults.update(kwargs)
    return MarketDataPipeline._compute_strategy_signals(None, **defaults)


class TestHoldStrategy:
    def test_hold_wins_in_choppy_market(self):
        """Low ADX + neutral RSI + low volume → HOLD should win."""
        s = _select(adx_val=10, rsi_val=50, bb_pos=0.1, volume_ratio=0.5,
                    regime="QUIET_ACCUMULATION")
        best = max(s, key=lambda k: s[k]["strength"])
        assert best == "hold"

    def test_hold_boosted_by_turkey_problem(self):
        """vol_regime < 0.7 → HOLD gets extra +0.20."""
        normal = _select(adx_val=12, rsi_val=50, volume_ratio=0.6,
                         vol_regime=1.0, regime="QUIET_ACCUMULATION")
        turkey = _select(adx_val=12, rsi_val=50, volume_ratio=0.6,
                         vol_regime=0.6, regime="QUIET_ACCUMULATION")
        assert turkey["hold"]["strength"] > normal["hold"]["strength"]

    def test_hold_boosted_by_hurst_random_walk(self):
        """Hurst ≈ 0.5 (random walk) → HOLD gets +0.10."""
        rw = _select(adx_val=12, rsi_val=50, volume_ratio=0.6,
                     hurst=0.50, regime="QUIET_ACCUMULATION")
        trending = _select(adx_val=12, rsi_val=50, volume_ratio=0.6,
                           hurst=0.70, regime="QUIET_ACCUMULATION")
        assert rw["hold"]["strength"] > trending["hold"]["strength"]

    def test_hold_boosted_by_vov_spike(self):
        """VoV spike > 2.0 → HOLD gets +0.15."""
        stable = _select(adx_val=12, rsi_val=50, volume_ratio=0.6,
                         vov_spike_ratio=1.0, regime="QUIET_ACCUMULATION")
        spike = _select(adx_val=12, rsi_val=50, volume_ratio=0.6,
                        vov_spike_ratio=2.5, regime="QUIET_ACCUMULATION")
        assert spike["hold"]["strength"] > stable["hold"]["strength"]

    def test_hold_reduced_in_opportunity_vol(self):
        """vol_regime 1.3-2.0 (opportunity) → HOLD gets -0.05."""
        opp = _select(adx_val=12, rsi_val=50, volume_ratio=0.6,
                      vol_regime=1.5, regime="QUIET_ACCUMULATION")
        normal = _select(adx_val=12, rsi_val=50, volume_ratio=0.6,
                         vol_regime=1.0, regime="QUIET_ACCUMULATION")
        assert opp["hold"]["strength"] <= normal["hold"]["strength"]

    def test_hold_loses_in_trending_regime(self):
        """In MOMENTUM_RALLY, HOLD should NOT win over momentum."""
        s = _select(ema_sig=0.8, macd_sig=0.6, adx_val=35,
                    regime="MOMENTUM_RALLY", hurst=0.7)
        best = max(s, key=lambda k: s[k]["strength"])
        assert best != "hold"


class TestHurstModulation:
    def test_low_hurst_boosts_mean_revert(self):
        """H < 0.5 → mean_revert fitness ×1.2."""
        mr_low_h = _select(rsi_val=25, rsi_sig=0.5, bb_pos=-0.9,
                           regime="MEAN_REVERT", hurst=0.3)
        mr_high_h = _select(rsi_val=25, rsi_sig=0.5, bb_pos=-0.9,
                            regime="MEAN_REVERT", hurst=0.7)
        assert mr_low_h["mean_revert"]["strength"] > mr_high_h["mean_revert"]["strength"]

    def test_high_hurst_boosts_momentum(self):
        """H > 0.5 → momentum fitness ×1.2."""
        mom_high_h = _select(ema_sig=0.8, macd_sig=0.5, adx_val=30,
                             regime="MOMENTUM_RALLY", hurst=0.7)
        mom_low_h = _select(ema_sig=0.8, macd_sig=0.5, adx_val=30,
                            regime="MOMENTUM_RALLY", hurst=0.3)
        assert mom_high_h["momentum"]["strength"] > mom_low_h["momentum"]["strength"]


class TestConvexityModulation:
    def test_convex_long_boosts_mr(self):
        """Convexity > 1.3 + LONG MR signal → ×1.10 boost."""
        convex = _select(rsi_val=25, rsi_sig=0.5, bb_pos=-0.9,
                         regime="MEAN_REVERT", convexity=1.5)
        neutral = _select(rsi_val=25, rsi_sig=0.5, bb_pos=-0.9,
                          regime="MEAN_REVERT", convexity=1.0)
        assert convex["mean_revert"]["strength"] > neutral["mean_revert"]["strength"]

    def test_concave_long_penalizes_mr(self):
        """Convexity < 0.7 + LONG MR signal → ×0.85 penalty."""
        concave = _select(rsi_val=25, rsi_sig=0.5, bb_pos=-0.9,
                          regime="MEAN_REVERT", convexity=0.6)
        neutral = _select(rsi_val=25, rsi_sig=0.5, bb_pos=-0.9,
                          regime="MEAN_REVERT", convexity=1.0)
        assert concave["mean_revert"]["strength"] < neutral["mean_revert"]["strength"]


class TestZScoreBBJoint:
    def test_extreme_zscore_bb_boosts_mr_long(self):
        """z < -2 AND bb_pos < -0.6 → strong LONG MR signal."""
        extreme = _select(rsi_val=45, z_score_50=-2.5, bb_pos=-0.8,
                          regime="MEAN_REVERT")
        mild = _select(rsi_val=45, z_score_50=-0.5, bb_pos=-0.2,
                       regime="MEAN_REVERT")
        assert extreme["mean_revert"]["strength"] > mild["mean_revert"]["strength"]


class TestVolZScoreBreakout:
    def test_compressed_vol_boosts_breakout(self):
        """vol_z < -1.5 → volume_breakout ×1.15."""
        compressed = _select(macd_sig=0.5, volume_ratio=2.0,
                             regime="MOMENTUM_RALLY", vol_z_score=-2.0)
        normal = _select(macd_sig=0.5, volume_ratio=2.0,
                         regime="MOMENTUM_RALLY", vol_z_score=0.0)
        assert compressed["volume_breakout"]["strength"] > normal["volume_breakout"]["strength"]


class TestConsistencyScoring:
    def test_high_rsi_consistency_boosts_mr(self):
        """RSI consistency > 50% → MR signal ×1.15."""
        consistent = _select(rsi_val=28, rsi_sig=0.4,
                             regime="MEAN_REVERT", rsi_consistency=0.6)
        sporadic = _select(rsi_val=28, rsi_sig=0.4,
                           regime="MEAN_REVERT", rsi_consistency=0.2)
        assert consistent["mean_revert"]["strength"] > sporadic["mean_revert"]["strength"]


class TestMTFMomentumBlend:
    def test_aligned_mtf_strengthens_momentum(self):
        """Positive TA + positive MTF → momentum signal is boosted vs conflicting MTF."""
        aligned = _select(ema_sig=0.6, macd_sig=0.4, adx_val=30,
                          regime="MOMENTUM_RALLY", mtf_momentum=0.05)
        conflict = _select(ema_sig=0.6, macd_sig=0.4, adx_val=30,
                           regime="MOMENTUM_RALLY", mtf_momentum=-0.05)
        # Aligned MTF should produce stronger signal than conflicting MTF
        assert aligned["momentum"]["strength"] > conflict["momentum"]["strength"]

    def test_conflicting_mtf_dampens_momentum(self):
        """Positive TA + negative MTF → dampened momentum."""
        conflict = _select(ema_sig=0.6, macd_sig=0.4, adx_val=30,
                           regime="MOMENTUM_RALLY", mtf_momentum=-0.05)
        aligned = _select(ema_sig=0.6, macd_sig=0.4, adx_val=30,
                          regime="MOMENTUM_RALLY", mtf_momentum=0.05)
        assert conflict["momentum"]["strength"] < aligned["momentum"]["strength"]
