"""
Tests for _compute_advanced_metrics — Hurst exponent, vol regime, kurtosis,
tail ratio, convexity, VoV spike, multi-TF momentum, z-score.
Uses synthetic data with KNOWN statistical properties.
"""
import sys
import os
import pytest
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data_mgmt.market_data_pipeline import MarketDataPipeline


def _compute(closes, volumes=None):
    """Helper: run _compute_advanced_metrics with minimal setup."""
    if volumes is None:
        volumes = np.ones(len(closes)) * 100_000_000
    return MarketDataPipeline._compute_advanced_metrics(
        np.array(closes, dtype=np.float64),
        np.array(volumes, dtype=np.float64),
        np,
    )


# =============================================================================
# HURST EXPONENT
# =============================================================================

class TestHurst:
    def test_trending_series_has_high_hurst(self):
        """A pure uptrend should have H > 0.5 (trending)."""
        closes = [100.0 + i * 0.5 for i in range(200)]  # linear uptrend
        m = _compute(closes)
        assert m["hurst"] > 0.55, f"Expected H > 0.55 for trend, got {m['hurst']}"

    def test_mean_reverting_series_has_low_hurst(self):
        """A strongly anti-correlated series should have H < trending series."""
        np.random.seed(123)
        # Strongly mean-reverting: flip sign each bar
        closes = [100.0]
        for i in range(199):
            delta = -0.8 * (closes[-1] - 100.0) + np.random.normal(0, 0.1)
            closes.append(max(50.0, closes[-1] + delta))
        m_mr = _compute(closes)
        # Compare against a trending series
        trending = [100.0 + i * 0.5 for i in range(200)]
        m_trend = _compute(trending)
        assert m_mr["hurst"] < m_trend["hurst"], (
            f"MR Hurst ({m_mr['hurst']:.3f}) should be < trending ({m_trend['hurst']:.3f})"
        )

    def test_short_series_returns_default(self):
        """Series < 40 bars should return default H = 0.5."""
        m = _compute([100.0 + i for i in range(20)])
        assert m["hurst"] == 0.5

    def test_hurst_bounded_0_1(self):
        """Hurst should always be in [0, 1]."""
        np.random.seed(42)
        closes = 100.0 * np.cumprod(1 + np.random.normal(0, 0.02, 200))
        m = _compute(closes)
        assert 0.0 <= m["hurst"] <= 1.0


# =============================================================================
# VOLATILITY REGIME
# =============================================================================

class TestVolRegime:
    def test_stable_vol_near_one(self):
        """Constant-vol series should have vol_regime ≈ 1.0."""
        np.random.seed(42)
        closes = 100.0 * np.cumprod(1 + np.random.normal(0, 0.015, 200))
        m = _compute(closes)
        assert 0.5 < m["vol_regime"] < 2.0

    def test_vol_compression_detected(self):
        """Series with compressed recent vol vs history should have ratio < 1."""
        np.random.seed(42)
        # First 150 bars: high vol
        high_vol = 100.0 * np.cumprod(1 + np.random.normal(0, 0.03, 150))
        # Last 50 bars: low vol
        low_vol = high_vol[-1] * np.cumprod(1 + np.random.normal(0, 0.005, 50))
        closes = np.concatenate([high_vol, low_vol])
        m = _compute(closes)
        assert m["vol_regime"] < 0.7, f"Expected turkey problem (< 0.7), got {m['vol_regime']}"

    def test_vol_expansion_detected(self):
        """Series with expanded recent vol should have ratio > 1."""
        np.random.seed(42)
        low_vol = 100.0 * np.cumprod(1 + np.random.normal(0, 0.005, 150))
        high_vol = low_vol[-1] * np.cumprod(1 + np.random.normal(0, 0.04, 50))
        closes = np.concatenate([low_vol, high_vol])
        m = _compute(closes)
        assert m["vol_regime"] > 1.3, f"Expected elevated vol (> 1.3), got {m['vol_regime']}"


# =============================================================================
# KURTOSIS & TAIL RISK
# =============================================================================

class TestTailRisk:
    def test_normal_returns_kurtosis_near_3(self):
        """Normal distribution returns should have kurtosis ≈ 3."""
        np.random.seed(42)
        closes = 100.0 * np.cumprod(1 + np.random.normal(0, 0.01, 200))
        m = _compute(closes)
        assert 1.5 < m["kurtosis"] < 5.0, f"Expected ~3 kurtosis, got {m['kurtosis']}"

    def test_fat_tails_high_kurtosis(self):
        """T-distribution (fat tails) should have kurtosis > 3."""
        np.random.seed(42)
        # t-distribution with df=3 has theoretical kurtosis = inf
        rets = np.random.standard_t(df=3, size=200) * 0.01
        closes = 100.0 * np.cumprod(1 + rets)
        m = _compute(closes)
        assert m["kurtosis"] > 3.0, f"Expected fat tails (K > 3), got {m['kurtosis']}"

    def test_skewness_sign(self):
        """Positive-skew series should have positive skewness."""
        np.random.seed(42)
        # Log-normal returns are positively skewed
        rets = np.random.lognormal(0, 0.02, 200) - 1.0
        closes = 100.0 * np.cumprod(1 + rets)
        m = _compute(closes)
        assert m["skewness"] > 0, f"Expected positive skew, got {m['skewness']}"

    def test_tail_ratio_positive(self):
        """Tail ratio should always be positive."""
        np.random.seed(42)
        closes = 100.0 * np.cumprod(1 + np.random.normal(0, 0.015, 200))
        m = _compute(closes)
        assert m["tail_ratio"] > 0


# =============================================================================
# CONVEXITY (P1-1)
# =============================================================================

class TestConvexity:
    def test_symmetric_returns_near_one(self):
        """Normal distribution has symmetric up/down → convexity ≈ 1.0."""
        np.random.seed(42)
        closes = 100.0 * np.cumprod(1 + np.random.normal(0, 0.015, 200))
        m = _compute(closes)
        assert 0.7 < m["convexity"] < 1.3

    def test_uptrend_biased_convexity(self):
        """Strong uptrend: avg up > avg down → convexity > 1.0."""
        np.random.seed(42)
        rets = np.random.normal(0.005, 0.01, 200)  # positive mean
        closes = 100.0 * np.cumprod(1 + rets)
        m = _compute(closes)
        assert m["convexity"] > 1.0


# =============================================================================
# VOV SPIKE (P1-2)
# =============================================================================

class TestVoVSpike:
    def test_stable_vov_near_one(self):
        """Stable volatility-of-volatility should have spike ratio near 1."""
        np.random.seed(42)
        closes = 100.0 * np.cumprod(1 + np.random.normal(0, 0.015, 200))
        m = _compute(closes)
        assert 0.3 < m["vov_spike_ratio"] < 3.0

    def test_short_series_default(self):
        """Series < 80 bars should return default 1.0."""
        m = _compute([100.0 + i * 0.1 for i in range(50)])
        assert m["vov_spike_ratio"] == 1.0


# =============================================================================
# MULTI-TIMEFRAME MOMENTUM
# =============================================================================

class TestMTFMomentum:
    def test_uptrend_positive_momentum(self):
        """Pure uptrend should have positive MTF momentum."""
        closes = [100.0 + i * 0.3 for i in range(600)]
        m = _compute(closes)
        assert m["mtf_momentum"] > 0

    def test_downtrend_negative_momentum(self):
        """Pure downtrend should have negative MTF momentum."""
        closes = [100.0 - i * 0.1 for i in range(600)]
        m = _compute(closes)
        assert m["mtf_momentum"] < 0

    def test_short_series_still_computes(self):
        """42+ bars should compute 7d momentum at minimum."""
        closes = [100.0 + i * 0.5 for i in range(50)]
        m = _compute(closes)
        assert m["mtf_momentum"] > 0


# =============================================================================
# Z-SCORE
# =============================================================================

class TestZScore:
    def test_price_at_mean_zscore_zero(self):
        """Price at 50-bar SMA should have z ≈ 0."""
        closes = [100.0] * 60  # constant price
        m = _compute(closes)
        assert abs(m["z_score_50"]) < 0.01

    def test_price_above_mean_positive_z(self):
        """Price above SMA50 should have positive z-score."""
        closes = [100.0] * 50 + [110.0]  # last bar above mean
        m = _compute(closes)
        assert m["z_score_50"] > 0

    def test_price_below_mean_negative_z(self):
        """Price below SMA50 should have negative z-score."""
        closes = [100.0] * 50 + [90.0]
        m = _compute(closes)
        assert m["z_score_50"] < 0


# =============================================================================
# VOL Z-SCORE (P1-3)
# =============================================================================

class TestVolZScore:
    def test_compressed_vol_negative_z(self):
        """Recent vol << historical vol → vol_z_score < 0."""
        np.random.seed(42)
        high_vol = 100.0 * np.cumprod(1 + np.random.normal(0, 0.03, 150))
        low_vol = high_vol[-1] * np.cumprod(1 + np.random.normal(0, 0.003, 50))
        closes = np.concatenate([high_vol, low_vol])
        m = _compute(closes)
        assert m["vol_z_score"] < 0

    def test_short_series_default(self):
        """Series < 84 bars should return existing vol_z as baseline."""
        m = _compute([100.0 + i for i in range(50)])
        assert "vol_z_score" in m
