"""
test_sol_feed_nonetype_p132.py — ticker None-field crash regression (P132)
================================================================================

Production bug 2026-04-28: SOL [LIVE_DATA] failing 100% of ticks since 23:24
restart with `TypeError: float() argument must be a string or a real number,
not 'NoneType'`. Root cause: lines 1597-1600 in market_data_pipeline.py used
`float(ticker.get("last", 0))` — the dict.get(k, default) only returns
default when the key is MISSING. When the key is PRESENT but value is None
(common in CCXT/Kraken responses for low-liquidity moments), it returns
None, then float(None) raises TypeError.

Fix uses _safe_float(value, default) which catches TypeError and returns
default. Same shape as the existing trade-field handling at lines 1821-1822.

This test mocks a CCXT ticker dict with None fields and verifies:
  1. _safe_float handles None correctly
  2. The fixed code path doesn't crash on None ticker fields
  3. OHLCV bars with None entries are SKIPPED, not crash-causing
"""
from __future__ import annotations

import math

import pytest


class TestSafeFloat:
    """The shared helper used to fix P132. Verify it handles None / NaN /
    non-numeric without crash."""

    def test_safe_float_handles_none(self):
        from data_mgmt.market_data_pipeline import MarketDataPipeline
        # Static method — call on class
        assert MarketDataPipeline._safe_float(None, 1.5) == 1.5
        assert MarketDataPipeline._safe_float(None, 0.0) == 0.0

    def test_safe_float_handles_nan(self):
        from data_mgmt.market_data_pipeline import MarketDataPipeline
        assert MarketDataPipeline._safe_float(float("nan"), 0.0) == 0.0

    def test_safe_float_handles_inf(self):
        from data_mgmt.market_data_pipeline import MarketDataPipeline
        assert MarketDataPipeline._safe_float(float("inf"), 0.0) == 0.0
        assert MarketDataPipeline._safe_float(float("-inf"), 1.0) == 1.0

    def test_safe_float_handles_non_numeric_string(self):
        from data_mgmt.market_data_pipeline import MarketDataPipeline
        assert MarketDataPipeline._safe_float("not_a_number", 7.5) == 7.5

    def test_safe_float_passes_real_numbers(self):
        from data_mgmt.market_data_pipeline import MarketDataPipeline
        assert MarketDataPipeline._safe_float(3.14, 0.0) == 3.14
        assert MarketDataPipeline._safe_float(0, 1.0) == 0.0  # zero is valid
        assert MarketDataPipeline._safe_float("1.5", 0.0) == 1.5  # numeric str

    def test_safe_float_handles_empty_string(self):
        from data_mgmt.market_data_pipeline import MarketDataPipeline
        # Empty string causes ValueError in float() — falls back to default
        assert MarketDataPipeline._safe_float("", 2.0) == 2.0


class TestP132SourceContract:
    """Verify the P132 fix is present + the old vulnerable pattern is gone."""

    def test_ticker_fields_use_safe_float(self):
        """Lines 1597-1600 (the original crash site) must use _safe_float,
        not raw float() on .get() with None-vulnerable defaults."""
        import inspect
        from data_mgmt.market_data_pipeline import MarketDataPipeline
        src = inspect.getsource(MarketDataPipeline._fetch_live_data)
        # The fix uses _safe_float on the ticker fields
        assert "_safe_float(ticker.get(\"last\")" in src, (
            "P132 regression: ticker.get('last') no longer uses _safe_float. "
            "float(None) will crash again on Kraken null responses."
        )
        assert "_safe_float(ticker.get(\"quoteVolume\")" in src, (
            "P132 regression: quoteVolume no longer uses _safe_float."
        )

    def test_ohlcv_bars_use_safe_float(self):
        """OHLCV bar parsing uses _safe_float + skips malformed bars."""
        import inspect
        from data_mgmt.market_data_pipeline import MarketDataPipeline
        src = inspect.getsource(MarketDataPipeline._fetch_live_data)
        assert "_safe_float(bar[4]" in src or "_safe_float(bar[5]" in src, (
            "P132 regression: OHLCV bar field parsing no longer uses "
            "_safe_float. None bar fields will crash the loop."
        )

    def test_p132_marker_present(self):
        """Comment markers preserved so future operators know why the
        non-obvious _safe_float pattern is used here."""
        import inspect
        from data_mgmt.market_data_pipeline import MarketDataPipeline
        src = inspect.getsource(MarketDataPipeline._fetch_live_data)
        assert "P132" in src, (
            "P132 marker comments removed; future grep won't find context."
        )

    def test_exception_handler_includes_streak(self):
        """The improved exception handler must surface the failure streak
        so operators can distinguish 1-tick blips from 100-tick outages."""
        import inspect
        from data_mgmt.market_data_pipeline import MarketDataPipeline
        src = inspect.getsource(MarketDataPipeline._fetch_live_data)
        assert "streak=" in src and "exc_info" in src, (
            "P132 improvement reverted: exception handler no longer "
            "logs streak count + traceback. Future failures will lose "
            "the call site context (the original P132 bug was hidden "
            "for hours because the log only had the str(e) message)."
        )


class TestRegressionSimulation:
    """Direct simulation: feed a Kraken-style ticker with None fields
    through the fixed _safe_float pipeline. Pre-fix would crash; post-fix
    should yield 0 (synthetic fallback semantics)."""

    def test_kraken_null_last_doesnt_crash(self):
        """Realistic Kraken ticker with last=None (the SOL bug shape)."""
        from data_mgmt.market_data_pipeline import MarketDataPipeline
        ticker = {
            "last": None,           # ← the bug trigger
            "quoteVolume": 12345.6,
            "bid": 86.5,
            "ask": 86.6,
            "timestamp": 1700000000000,
        }
        # This is what the fixed code does:
        cp = MarketDataPipeline._safe_float(ticker.get("last"), 0.0)
        vol = MarketDataPipeline._safe_float(ticker.get("quoteVolume"), 0.0)
        bid = MarketDataPipeline._safe_float(ticker.get("bid"), cp)
        ask = MarketDataPipeline._safe_float(ticker.get("ask"), cp)
        assert cp == 0.0
        assert vol == 12345.6
        assert bid == 86.5
        assert ask == 86.6

    def test_kraken_all_null_fields_doesnt_crash(self):
        """Worst case: every ticker field is None. Should return zeros,
        not crash."""
        from data_mgmt.market_data_pipeline import MarketDataPipeline
        ticker = {"last": None, "quoteVolume": None, "bid": None, "ask": None}
        cp = MarketDataPipeline._safe_float(ticker.get("last"), 0.0)
        vol = MarketDataPipeline._safe_float(ticker.get("quoteVolume"), 0.0)
        bid = MarketDataPipeline._safe_float(ticker.get("bid"), cp)
        ask = MarketDataPipeline._safe_float(ticker.get("ask"), cp)
        # All fall back to 0
        assert cp == 0.0 and vol == 0.0 and bid == 0.0 and ask == 0.0

    def test_ohlcv_bar_with_none_fields_skipped(self):
        """Bars with None close are SKIPPED, not crash-causing."""
        from data_mgmt.market_data_pipeline import MarketDataPipeline
        ohlcv = [
            [1700000000000, 86.0, 86.5, 85.5, 86.2, 1234.0],   # valid
            [1700000060000, None, None, None, None, None],     # all-None bar
            [1700000120000, 86.5, 87.0, 86.0, 86.8, 5678.0],   # valid
        ]
        # Simulate the fixed loop:
        valid_count = 0
        for bar in ohlcv:
            if bar is None or len(bar) < 6:
                continue
            close = MarketDataPipeline._safe_float(bar[4], 0.0)
            if close <= 0:
                continue
            valid_count += 1
        assert valid_count == 2  # 2 of 3 bars survive


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
