"""
HMATS Shared Test Fixtures
==========================
Central pytest configuration with reusable fixtures for all test modules.
Inspired by ai-hedge-fund conftest.py pattern: deterministic, minimal, no external deps.
"""
import sys
import os
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock

# Ensure project root is on path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# =============================================================================
# MARKET DATA FIXTURES
# =============================================================================

@pytest.fixture
def prices():
    """Simple price dict for 3 crypto assets."""
    return {"BTC": 73000.0, "ETH": 2300.0, "SOL": 85.0}


@pytest.fixture
def ohlcv_bars_btc():
    """200 synthetic 4H BTC OHLCV bars with realistic price action.
    Starts at $70K, trends up to ~$73K with noise."""
    np.random.seed(42)
    n = 200
    returns = np.random.normal(0.0005, 0.015, n)  # slight uptrend, ~1.5% per bar vol
    close = 70000.0 * np.cumprod(1 + returns)
    high = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n)))
    open_ = np.roll(close, 1)
    open_[0] = 70000.0
    volume = np.random.uniform(50_000_000, 150_000_000, n)
    return {
        "close": close, "high": high, "low": low,
        "open": open_, "volume": volume, "n": n,
    }


@pytest.fixture
def ohlcv_bars_sol():
    """200 synthetic 4H SOL OHLCV bars — choppy consolidation (for HOLD tests)."""
    np.random.seed(99)
    n = 200
    returns = np.random.normal(0.0001, 0.025, n)  # near-zero drift, higher vol
    close = 85.0 * np.cumprod(1 + returns)
    high = close * (1 + np.abs(np.random.normal(0, 0.008, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.008, n)))
    open_ = np.roll(close, 1)
    open_[0] = 85.0
    volume = np.random.uniform(10_000_000, 30_000_000, n)
    return {
        "close": close, "high": high, "low": low,
        "open": open_, "volume": volume, "n": n,
    }


@pytest.fixture
def market_data_template(prices):
    """Minimal market_data dict matching constitution schema requirements."""
    return {
        "asset": "BTC",
        "current_price": prices["BTC"],
        "data_valid": True,
        "data_age_seconds": 1.0,
        "data_age_sec": 1.0,
        "regime_state": "QUIET_ACCUMULATION",
        "regime_confidence": 0.85,
        "dvol_zscore": 0.0,
        "vpin": 0.35,
        "correlation_btc_eth_sol": 0.75,
        "orderbook_depth_1pct_usd": 5_000_000,
        "spread_bps": 1.0,
        "volume_ratio": 1.0,
        "annualized_volatility": 0.65,
        "hurst_exponent": 0.5,
        "vol_regime": 1.0,
        "kurtosis": 3.0,
        "skewness": 0.0,
        "tail_ratio": 1.0,
        "z_score_50": 0.0,
        "convexity": 1.0,
        "vol_spike_ratio": 1.0,
        "vov_spike_ratio": 1.0,
        "vol_z_score": 0.0,
        "black_swan_sentinel": 3,
        "contrarian_opportunity": False,
        "_fear_greed_value": 50,
        "min_alpha_bps": 5.0,
        "min_alpha_bps_short": 3.0,
        "quiet_accum_min_direction": 0.08,
        "quiet_accum_min_direction_short": 0.12,
        "borderline_alpha_pass_epsilon_bps": 2.0,
        "borderline_alpha_pass_epsilon_bps_short": 1.5,
        "rsi_consistency": 0.0,
        "bb_consistency": 0.0,
    }


# =============================================================================
# POSITION / PORTFOLIO FIXTURES
# =============================================================================

@pytest.fixture
def empty_position_state():
    """No position — fresh entry scenario."""
    return {
        "has_position": False,
        "current_exposure": 0.0,
        "direction": 0,
        "tranche_level": 0,
    }


@pytest.fixture
def long_position_state():
    """Existing LONG BTC T1 position."""
    return {
        "has_position": True,
        "current_exposure": 0.15,
        "direction": 1,
        "tranche_level": 1,
        "entry_price": 71000.0,
        "strategy": "momentum",
        "entry_time": "2026-04-10T12:00:00+00:00",
        "entry_tick": 10,
        "mode_at_entry": "NORMAL",
    }


# =============================================================================
# AGENT SIGNAL FIXTURES
# =============================================================================

@pytest.fixture
def bullish_agent_signals():
    """Strong bullish quant signal."""
    return {
        "quant_direction": 0.75,
        "quant_confidence": 0.85,
        "primary_strategy": "momentum",
        "regime_confidence": 0.90,
        "effective_alpha_direction": 0.75,
        "strategy_agreement": 0.75,
        "drl_direction": 0.0,
        "drl_confidence": 0.0,
    }


@pytest.fixture
def hold_agent_signals():
    """HOLD signal (no edge detected)."""
    return {
        "quant_direction": 0.0,
        "quant_confidence": 0.50,
        "primary_strategy": "hold",
        "regime_confidence": 0.90,
        "effective_alpha_direction": 0.0,
        "strategy_agreement": 0.0,
    }


# =============================================================================
# MOCK HELPERS
# =============================================================================

@pytest.fixture
def mock_kraken_exchange():
    """Mock CCXT Kraken exchange that returns deterministic data."""
    exchange = MagicMock()
    exchange.fetch_ticker.return_value = {
        "last": 73000.0, "bid": 72990.0, "ask": 73010.0,
        "baseVolume": 1500.0, "quoteVolume": 109_500_000.0,
        "timestamp": 1712800000000,
    }
    exchange.fetch_ohlcv.return_value = [
        [1712800000000, 72500.0, 73200.0, 72400.0, 73000.0, 100_000_000.0]
        for _ in range(100)
    ]
    exchange.fetch_order_book.return_value = {
        "bids": [[72990.0, 10.0]], "asks": [[73010.0, 10.0]],
    }
    return exchange
