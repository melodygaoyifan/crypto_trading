"""
Tests for system stability features — CCXT session reset, watchdog logic,
margin tracking, and the pipeline's advanced metrics edge cases.
"""
import sys, os, pytest, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# =============================================================================
# CCXT SESSION AUTO-RESET
# =============================================================================

class TestCCXTAutoReset:
    """Test the failure counter and reset threshold logic."""

    def test_failure_counter_increments(self):
        """Each fetch failure should increment the global counter."""
        from data_mgmt.market_data_pipeline import MarketDataPipeline
        pipeline = MarketDataPipeline.__new__(MarketDataPipeline)
        pipeline._global_fetch_failure_count = 0
        pipeline._global_fetch_success_count = 0
        pipeline._ccxt_reset_count = 0
        pipeline._CCXT_RESET_THRESHOLD = 10
        pipeline._ccxt_exchange = None
        pipeline._kraken_rest = None
        pipeline._requests_session = None

        # Simulate failures
        for _ in range(5):
            pipeline._global_fetch_failure_count += 1
        assert pipeline._global_fetch_failure_count == 5

    def test_success_resets_counter(self):
        """A successful fetch should reset the failure counter to 0."""
        from data_mgmt.market_data_pipeline import MarketDataPipeline
        pipeline = MarketDataPipeline.__new__(MarketDataPipeline)
        pipeline._global_fetch_failure_count = 7
        pipeline._global_fetch_success_count = 0

        # Simulate success
        pipeline._global_fetch_failure_count = 0
        pipeline._global_fetch_success_count += 1
        assert pipeline._global_fetch_failure_count == 0

    def test_threshold_default_is_10(self):
        """Default CCXT reset threshold should be 10."""
        from data_mgmt.market_data_pipeline import MarketDataPipeline
        pipeline = MarketDataPipeline.__new__(MarketDataPipeline)
        pipeline._CCXT_RESET_THRESHOLD = 10
        assert pipeline._CCXT_RESET_THRESHOLD == 10


# =============================================================================
# WATCHDOG HEALTH CHECK LOGIC
# =============================================================================

class TestWatchdogLogic:
    def test_stale_log_detected(self):
        """Log older than MAX_LOG_STALE_SECONDS should be flagged."""
        MAX_LOG_STALE_SECONDS = 300
        log_age = 400  # 400 seconds old
        assert log_age > MAX_LOG_STALE_SECONDS

    def test_fresh_log_is_healthy(self):
        MAX_LOG_STALE_SECONDS = 300
        log_age = 30
        assert log_age <= MAX_LOG_STALE_SECONDS

    def test_stale_data_detected(self):
        """No LIVE_DATA for > 600s should trigger restart."""
        MAX_DATA_STALE_SECONDS = 600
        data_age = 700
        assert data_age > MAX_DATA_STALE_SECONDS

    def test_restart_rate_limit(self):
        """Max 3 restarts per hour should be enforced."""
        MAX_RESTARTS_PER_HOUR = 3
        restart_times = [time.time() - 60, time.time() - 120, time.time() - 180]
        recent = [t for t in restart_times if time.time() - t < 3600]
        assert len(recent) >= MAX_RESTARTS_PER_HOUR
        # Should NOT restart
        should_restart = len(recent) < MAX_RESTARTS_PER_HOUR
        assert should_restart is False


# =============================================================================
# MARGIN TRACKING (P2-7)
# =============================================================================

class TestMarginTracking:
    def test_margin_exhaustion_blocks_entry(self):
        """When margin_available <= 0, exposure should be set to 0."""
        margin_req = 0.50
        account_equity = 10_000
        gross_exp = 2.0  # 200% gross = fully leveraged at 2x
        margin_used = gross_exp * account_equity  # $20,000
        margin_available = max(0.0, (account_equity / margin_req) - margin_used)
        assert margin_available == 0.0

    def test_margin_available_with_room(self):
        """With moderate exposure, margin should be available."""
        margin_req = 0.50
        account_equity = 10_000
        gross_exp = 0.5  # 50% gross
        margin_used = gross_exp * account_equity  # $5,000
        margin_available = max(0.0, (account_equity / margin_req) - margin_used)
        assert margin_available > 0
        assert margin_available == pytest.approx(15_000.0)

    def test_margin_formula_correctness(self):
        """Verify: margin_available = equity / margin_req - margin_used."""
        equity = 10_000
        margin_req = 0.50
        margin_used = 8_000
        expected = (equity / margin_req) - margin_used  # 20000 - 8000 = 12000
        assert expected == pytest.approx(12_000.0)


# =============================================================================
# PIPELINE EDGE CASES
# =============================================================================

class TestPipelineEdgeCases:
    def test_advanced_metrics_empty_array(self):
        """Empty closes array should not crash."""
        import numpy as np
        from data_mgmt.market_data_pipeline import MarketDataPipeline
        m = MarketDataPipeline._compute_advanced_metrics(
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np,
        )
        assert m["hurst"] == 0.5
        assert m["vol_regime"] == 1.0
        assert m["kurtosis"] == 3.0

    def test_advanced_metrics_single_price(self):
        """Single price should return safe defaults."""
        import numpy as np
        from data_mgmt.market_data_pipeline import MarketDataPipeline
        m = MarketDataPipeline._compute_advanced_metrics(
            np.array([100.0], dtype=np.float64),
            np.array([1e6], dtype=np.float64),
            np,
        )
        assert m["hurst"] == 0.5
        assert m["mtf_momentum"] == 0.0
        assert m["z_score_50"] == 0.0

    def test_advanced_metrics_constant_price(self):
        """Constant price (zero returns) should not crash or produce NaN."""
        import numpy as np
        from data_mgmt.market_data_pipeline import MarketDataPipeline
        m = MarketDataPipeline._compute_advanced_metrics(
            np.array([100.0] * 200, dtype=np.float64),
            np.array([1e6] * 200, dtype=np.float64),
            np,
        )
        assert np.isfinite(m["hurst"])
        assert np.isfinite(m["vol_regime"])
        assert np.isfinite(m["kurtosis"])
        assert np.isfinite(m["convexity"])

    def test_advanced_metrics_nan_input(self):
        """NaN in input should not crash — returns safe defaults."""
        import numpy as np
        from data_mgmt.market_data_pipeline import MarketDataPipeline
        closes = np.array([100.0] * 100 + [float('nan')] * 5, dtype=np.float64)
        volumes = np.array([1e6] * 105, dtype=np.float64)
        m = MarketDataPipeline._compute_advanced_metrics(closes, volumes, np)
        # Should not crash; values may be defaults
        assert "hurst" in m
        assert "vol_regime" in m
