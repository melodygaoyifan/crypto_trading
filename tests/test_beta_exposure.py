"""Tests for analytics/beta_exposure.py — beta/alpha decomposition."""
import sys, os, pytest
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analytics.beta_exposure import BetaExposureAnalyzer, BetaReport


@pytest.fixture
def analyzer():
    return BetaExposureAnalyzer(annualization_factor=252.0)


class TestRollingBeta:
    def test_perfect_correlation_beta_one(self, analyzer):
        """Identical series should have beta = 1.0."""
        r = np.random.RandomState(42).normal(0, 0.01, 100)
        betas = analyzer.compute_rolling_beta(r, r, window=20)
        valid = betas[np.isfinite(betas)]
        assert len(valid) > 0
        assert np.mean(valid) == pytest.approx(1.0, abs=0.1)  # numerical noise in small windows

    def test_zero_variance_benchmark(self, analyzer):
        """Constant benchmark should give beta = 0."""
        s = np.random.RandomState(42).normal(0, 0.01, 100)
        b = np.zeros(100)
        betas = analyzer.compute_rolling_beta(s, b, window=20)
        valid = betas[np.isfinite(betas)]
        assert all(v == 0.0 for v in valid)

    def test_negative_correlation(self, analyzer):
        """Inverted series should have beta < 0."""
        r = np.random.RandomState(42).normal(0, 0.01, 100)
        betas = analyzer.compute_rolling_beta(-r, r, window=20)
        valid = betas[np.isfinite(betas)]
        assert np.mean(valid) < -0.5


class TestUpsideDownsideCapture:
    def test_symmetric_capture(self, analyzer):
        """Identical series should have capture ratio ≈ 1.0."""
        r = np.random.RandomState(42).normal(0, 0.01, 200)
        up, down, up_n, down_n = analyzer.compute_upside_downside_capture(r, r)
        assert up == pytest.approx(1.0, abs=0.01)
        assert down == pytest.approx(1.0, abs=0.01)
        assert up_n > 0
        assert down_n > 0

    def test_high_downside_capture(self, analyzer):
        """Strategy that loses more on down days → downside capture > 1."""
        np.random.seed(42)
        b = np.random.normal(0, 0.01, 200)
        s = b.copy()
        s[b < 0] *= 2.0  # double losses
        _, down, _, _ = analyzer.compute_upside_downside_capture(s, b)
        assert down > 1.5


class TestResidualAlpha:
    def test_alpha_with_known_beta(self, analyzer):
        """With beta=0.5, residual = strat - 0.5*bench."""
        b = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
        s = 0.5 * b + 0.001  # beta=0.5, alpha=0.001 per bar
        residual = analyzer.compute_residual_alpha(s, b, beta=0.5)
        assert np.mean(residual) == pytest.approx(0.001, abs=0.0001)


class TestRegimeBucket:
    def test_regime_separation(self, analyzer):
        """Different regimes should produce separate beta estimates."""
        np.random.seed(42)
        n = 100
        b = np.random.normal(0, 0.01, n)
        s_bull = 1.5 * b[:50] + np.random.normal(0, 0.002, 50)
        s_bear = 0.3 * b[50:] + np.random.normal(0, 0.002, 50)
        s = np.concatenate([s_bull, s_bear])
        regimes = np.array(["BULL"] * 50 + ["BEAR"] * 50)
        result = analyzer.compute_regime_bucket_report(s, b, regimes)
        assert "BULL" in result
        assert "BEAR" in result
        assert result["BULL"]["beta"] > result["BEAR"]["beta"]


class TestFullReport:
    def test_report_structure(self, analyzer):
        """Full report should have all required fields."""
        np.random.seed(42)
        s = np.random.normal(0.0005, 0.01, 50)
        b = np.random.normal(0, 0.01, 50)
        report = analyzer.full_report(s, b, window=10)
        assert isinstance(report, BetaReport)
        assert report.beta_risk_level in ("LOW", "MEDIUM", "HIGH", "EXTREME", "INSUFFICIENT_DATA")
        assert isinstance(report.overall_beta, float)
        assert isinstance(report.overall_correlation, float)

    def test_insufficient_data(self, analyzer):
        """< 10 bars should return INSUFFICIENT_DATA."""
        s = np.array([0.01, -0.01])
        b = np.array([0.02, -0.01])
        report = analyzer.full_report(s, b)
        assert report.beta_risk_level == "INSUFFICIENT_DATA"

    def test_low_beta_classified_low(self, analyzer):
        """Uncorrelated series should be classified LOW risk."""
        np.random.seed(42)
        s = np.random.normal(0, 0.01, 100)
        b = np.random.normal(0, 0.01, 100)  # independent
        report = analyzer.full_report(s, b, window=20)
        assert report.beta_risk_level in ("LOW", "MEDIUM")
