"""
Tests for agents/short_bias_agent.py - production-readiness checks.

Groups:
1. Short-only invariant (P0)
2. Observation period ingests history (P0)
3. Multi-asset isolation (P0)
4. Time-window correctness (P0)
5. Diagnostics completeness (P1)
6. Input normalization (P1)
7. Backward compatibility
"""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from agents.short_bias_agent import (
    ShortBiasAgent,
    ShortBiasConfig,
    ShortBiasSignal,
    ShortSignalType,
    ShortSignalComponent,
    get_short_bias_agent,
    reset_short_bias_agent,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extreme_market(asset="BTC", price=200.0):
    """Market data for a parabolic top scenario."""
    return {
        "asset": asset,
        "price": price,
        "high_24h": price * 1.05,
        "low_24h": price * 0.90,
        "volume": 5_000_000,
        "rsi": 85.0,
    }


def _extreme_derivatives():
    return {
        "funding_rate": 0.0012,
        "open_interest": 2_000_000_000,
        "liquidations_long": 50_000_000,
        "liquidations_short": 5_000_000,
    }


def _extreme_sentiment():
    return {"fear_greed_index": 0.90, "social_sentiment": 0.85}


def _extreme_onchain():
    return {"whale_flow_usd": -100_000_000, "exchange_inflow": 80_000_000}


def _normal_market(asset="BTC"):
    return {
        "asset": asset,
        "price": 100.0,
        "high_24h": 102.0,
        "low_24h": 98.0,
        "volume": 1_000_000,
        "rsi": 50.0,
    }


def _normal_derivatives():
    return {
        "funding_rate": 0.0001,
        "open_interest": 1_000_000_000,
        "liquidations_long": 5_000_000,
        "liquidations_short": 5_000_000,
    }


def _normal_sentiment():
    return {"fear_greed_index": 0.50, "social_sentiment": 0.50}


def _make_agent(**config_kwargs) -> ShortBiasAgent:
    """Create agent with no observation period for testing."""
    defaults = {"min_observation_minutes": 0}
    defaults.update(config_kwargs)
    return ShortBiasAgent(ShortBiasConfig(**defaults))


# ---------------------------------------------------------------------------
# 1. Short-Only Invariant (P0)
# ---------------------------------------------------------------------------

class TestShortOnlyInvariant:
    """P0: direction must NEVER be positive."""

    def test_extreme_market_direction_not_positive(self):
        """Full extreme scenario: direction must be -1 or 0."""
        agent = _make_agent()
        signal = agent.analyze(
            _extreme_market(), _extreme_derivatives(),
            _extreme_sentiment(), _extreme_onchain(),
        )
        assert signal.direction <= 0

    def test_normal_market_direction_not_positive(self):
        """Normal scenario: direction must be 0."""
        agent = _make_agent()
        signal = agent.analyze(_normal_market(), _normal_derivatives(), _normal_sentiment())
        assert signal.direction <= 0

    def test_to_agent_signal_dict_never_positive(self):
        """to_agent_signal_dict direction key must never be positive."""
        agent = _make_agent()
        signal = agent.analyze(
            _extreme_market(), _extreme_derivatives(),
            _extreme_sentiment(), _extreme_onchain(),
        )
        d = signal.to_agent_signal_dict()
        assert d["direction"] <= 0

    def test_is_actionable_only_when_short(self):
        """is_actionable() should only be True when direction == -1."""
        sig = ShortBiasSignal(direction=0.0, confidence=0.8)
        assert sig.is_actionable() is False

        sig2 = ShortBiasSignal(direction=-1.0, confidence=0.5)
        assert sig2.is_actionable() is True

    def test_no_long_direction_after_many_calls(self):
        """Stress: 20 calls with varied data, direction never positive."""
        agent = _make_agent(signal_cooldown_minutes=0)
        for i in range(20):
            price = 100 + i * 10
            m = _extreme_market(price=price)
            sig = agent.analyze(m, _extreme_derivatives(), _extreme_sentiment(), _extreme_onchain())
            assert sig.direction <= 0, f"Iteration {i}: direction={sig.direction}"


# ---------------------------------------------------------------------------
# 2. Observation Period Ingests History (P0)
# ---------------------------------------------------------------------------

class TestObservationIngestion:
    """P0: History must accumulate during observation period."""

    def test_history_grows_during_observation(self):
        """Calls during observation period should still record price history."""
        agent = ShortBiasAgent(ShortBiasConfig(min_observation_minutes=60))

        # 5 calls during observation (within 60 min)
        for i in range(5):
            agent.analyze({"asset": "BTC", "price": 100 + i, "volume": 1e6})

        # History should have 5 entries
        assert len(agent._price_history["BTC"]) == 5

    def test_history_available_after_observation(self):
        """After observation period ends, lookback data should be usable."""
        agent = ShortBiasAgent(ShortBiasConfig(min_observation_minutes=1))  # 1 minute

        # Feed data during observation
        for i in range(10):
            agent.analyze({"asset": "BTC", "price": 100 + i, "volume": 1e6})

        # Manually expire observation
        agent._observation_start["BTC"] = datetime.now(timezone.utc) - timedelta(minutes=2)

        # Now analyze should use the accumulated history
        sig = agent.analyze(
            {"asset": "BTC", "price": 120, "volume": 5e6, "rsi": 85, "high_24h": 125, "low_24h": 100},
            _extreme_derivatives(), _extreme_sentiment(), _extreme_onchain(),
        )
        # Should NOT be gated, and should have full history
        assert "observation" not in str(sig.reasons)
        assert len(agent._price_history["BTC"]) == 11  # 10 + 1

    def test_gated_signal_has_gate_reason(self):
        """During observation, signal should have gate info."""
        agent = ShortBiasAgent(ShortBiasConfig(min_observation_minutes=60))
        sig = agent.analyze({"asset": "BTC", "price": 100, "volume": 1e6})
        assert sig.direction == 0
        assert any("observation" in r for r in sig.reasons)
        assert "gates" in sig.meta
        assert len(sig.meta["gates"]) > 0


# ---------------------------------------------------------------------------
# 3. Multi-Asset Isolation (P0)
# ---------------------------------------------------------------------------

class TestMultiAssetIsolation:
    """P0: BTC data must not affect ETH/SOL analysis."""

    def test_btc_history_not_in_eth(self):
        """BTC price history must not appear in ETH analysis."""
        agent = _make_agent()

        # Feed BTC with rising prices
        for i in range(10):
            agent.analyze({"asset": "BTC", "price": 100 + i * 10, "volume": 1e6})

        # Feed ETH with flat prices
        for i in range(10):
            agent.analyze({"asset": "ETH", "price": 50, "volume": 5e5})

        assert len(agent._price_history["BTC"]) == 10
        assert len(agent._price_history["ETH"]) == 10

        # BTC prices should be 100-190, ETH should be all 50
        btc_prices = [v for _, v in agent._price_history["BTC"]]
        eth_prices = [v for _, v in agent._price_history["ETH"]]
        assert all(p >= 100 for p in btc_prices)
        assert all(p == 50 for p in eth_prices)

    def test_per_asset_cooldown_isolation(self):
        """BTC cooldown should not affect ETH."""
        agent = _make_agent(signal_cooldown_minutes=120)

        # Trigger BTC signal
        sig_btc = agent.analyze(
            _extreme_market("BTC"), _extreme_derivatives(),
            _extreme_sentiment(), _extreme_onchain(),
        )

        # BTC should be in cooldown now
        sig_btc2 = agent.analyze(
            _extreme_market("BTC"), _extreme_derivatives(),
            _extreme_sentiment(), _extreme_onchain(),
        )
        if sig_btc.is_actionable():
            assert any("cooldown" in r.lower() for r in sig_btc2.reasons)

        # ETH should NOT be in cooldown
        sig_eth = agent.analyze(
            _extreme_market("ETH"), _extreme_derivatives(),
            _extreme_sentiment(), _extreme_onchain(),
            asset="ETH",
        )
        assert not any("cooldown" in r.lower() for r in sig_eth.reasons)

    def test_per_asset_consecutive_counter(self):
        """Consecutive signal counter must be per-asset."""
        agent = _make_agent(signal_cooldown_minutes=0, max_consecutive_signals=2)

        # Trigger BTC twice
        agent.analyze(_extreme_market("BTC"), _extreme_derivatives(), _extreme_sentiment(), _extreme_onchain())
        agent.analyze(_extreme_market("BTC"), _extreme_derivatives(), _extreme_sentiment(), _extreme_onchain())

        btc_consec = agent._consecutive_signals.get("BTC", 0)
        eth_consec = agent._consecutive_signals.get("ETH", 0)

        # BTC should have accumulated, ETH should be 0 or not exist
        assert eth_consec == 0

    def test_asset_inferred_from_market_data(self):
        """Asset should be inferred from market_data['asset'] if not explicit."""
        agent = _make_agent()
        agent.analyze({"asset": "SOL", "price": 25, "volume": 1e6})
        assert "SOL" in agent._price_history
        assert "BTC" not in agent._price_history  # should not create BTC

    def test_asset_explicit_overrides_market_data(self):
        """Explicit asset= param should override market_data['asset']."""
        agent = _make_agent()
        agent.analyze({"asset": "BTC", "price": 100, "volume": 1e6}, asset="ETH")
        assert "ETH" in agent._price_history
        assert "BTC" not in agent._price_history


# ---------------------------------------------------------------------------
# 4. Time-Window Correctness (P0)
# ---------------------------------------------------------------------------

class TestTimeWindowCorrectness:
    """P0: Lookbacks must use timestamps, not indices."""

    def test_parabolic_uses_24h_timestamp_not_index(self):
        """Parabolic price_change should use sample at now-24h, not index -24."""
        agent = _make_agent()

        base_time = datetime.now(timezone.utc) - timedelta(hours=48)

        # Feed prices at known times
        # t=-48h: price=100
        agent._ensure_asset("BTC", base_time)
        agent._price_history["BTC"].append((base_time, 100.0))

        # t=-24h: price=110
        t_24h = base_time + timedelta(hours=24)
        agent._price_history["BTC"].append((t_24h, 110.0))

        # t=-12h: price=115
        t_12h = base_time + timedelta(hours=36)
        agent._price_history["BTC"].append((t_12h, 115.0))

        # Current price=130 -> 24h change should be (130-110)/110 ≈ 18.2%
        # NOT (130-115)/115 ≈ 13% (index -24 would be wrong anyway)
        # The _get_value_at_or_before for now-24h should find 110
        agent._observation_start["BTC"] = base_time  # bypass observation

        # Manually verify the lookback
        now = datetime.now(timezone.utc)
        target = now - timedelta(hours=24)
        val = agent._get_value_at_or_before(agent._price_history["BTC"], target)
        assert val == 110.0, f"Expected 110.0, got {val}"

    def test_funding_trend_uses_config_window(self):
        """Funding trend should use funding_trend_window_hours, not fixed index."""
        agent = _make_agent(funding_trend_window_hours=12)
        agent._ensure_asset("BTC", datetime.now(timezone.utc))

        # Add funding entries at specific times
        base = datetime.now(timezone.utc) - timedelta(hours=24)
        for h in range(25):
            t = base + timedelta(hours=h)
            agent._funding_history["BTC"].append((t, {
                "funding_rate": 0.0001 * (h + 1),
                "open_interest": 1e9,
            }))

        # Get lookback for 12 hours
        now = datetime.now(timezone.utc)
        window = agent._get_lookback_series(agent._funding_history["BTC"], now, 12)
        # Should only contain entries from last 12 hours
        assert len(window) <= 13
        assert all((now - ts).total_seconds() <= 12 * 3600 + 60 for ts, _ in window)

    def test_get_value_at_or_before_returns_correct(self):
        """Helper should return the value at or just before target time."""
        agent = _make_agent()
        agent._ensure_asset("BTC", datetime.now(timezone.utc))

        base = datetime.now(timezone.utc) - timedelta(hours=10)
        hist = agent._price_history["BTC"]
        hist.append((base, 100.0))
        hist.append((base + timedelta(hours=5), 110.0))
        hist.append((base + timedelta(hours=10), 120.0))

        # Target at base+3h -> should get 100.0 (before 5h mark)
        val = agent._get_value_at_or_before(hist, base + timedelta(hours=3))
        assert val == 100.0

        # Target at base+7h -> should get 110.0
        val = agent._get_value_at_or_before(hist, base + timedelta(hours=7))
        assert val == 110.0


# ---------------------------------------------------------------------------
# 5. Diagnostics Completeness (P1)
# ---------------------------------------------------------------------------

class TestDiagnostics:
    """P1: to_agent_signal_dict must include structured meta."""

    def test_meta_in_signal_dict(self):
        """to_agent_signal_dict must include meta key."""
        agent = _make_agent()
        sig = agent.analyze(
            _extreme_market(), _extreme_derivatives(),
            _extreme_sentiment(), _extreme_onchain(),
        )
        d = sig.to_agent_signal_dict()
        assert "meta" in d
        assert "asset" in d
        assert "asof_ts" in d

    def test_meta_has_components(self):
        """Meta should include component breakdown."""
        agent = _make_agent()
        sig = agent.analyze(
            _extreme_market(), _extreme_derivatives(),
            _extreme_sentiment(), _extreme_onchain(),
        )
        meta = sig.meta
        assert "components" in meta
        assert "PARABOLIC_TOP" in meta["components"]
        assert "EXTREME_FUNDING" in meta["components"]

    def test_meta_has_missing_inputs(self):
        """Missing inputs should be listed in meta."""
        agent = _make_agent()
        sig = agent.analyze(_normal_market())  # no derivatives, sentiment, onchain
        meta = sig.meta
        assert "missing_inputs" in meta
        assert "derivatives_data" in meta["missing_inputs"]
        assert "sentiment_data" in meta["missing_inputs"]
        assert "onchain_data" in meta["missing_inputs"]

    def test_meta_has_gates_when_gated(self):
        """Gated signal should have gate info in meta."""
        agent = ShortBiasAgent(ShortBiasConfig(min_observation_minutes=60))
        sig = agent.analyze({"asset": "BTC", "price": 100, "volume": 1e6})
        assert "gates" in sig.meta
        assert len(sig.meta["gates"]) > 0
        assert "observation" in sig.meta["gates"][0]

    def test_meta_has_normalization_for_fear_greed(self):
        """When sentiment is provided, normalization info should be in meta."""
        agent = _make_agent()
        sig = agent.analyze(
            _normal_market(),
            sentiment_data={"fear_greed_index": 85},  # 0-100 scale
        )
        meta = sig.meta
        assert "normalization" in meta
        assert "fear_greed" in meta["normalization"]
        fg_info = meta["normalization"]["fear_greed"]
        assert fg_info["raw"] == 85
        assert fg_info["interpretation"] == "assumed_0_100_scale"
        assert 0 <= fg_info["normalized"] <= 1

    def test_meta_has_triggered_components(self):
        """Triggered components list should be present."""
        agent = _make_agent()
        sig = agent.analyze(
            _extreme_market(), _extreme_derivatives(),
            _extreme_sentiment(), _extreme_onchain(),
        )
        assert "triggered_components" in sig.meta

    def test_meta_has_coverage_ratio(self):
        """Coverage ratio should reflect available inputs."""
        agent = _make_agent()
        sig = agent.analyze(_normal_market())  # only market data
        assert "coverage_ratio" in sig.meta
        # 3 of 5 inputs missing
        assert sig.meta["coverage_ratio"] <= 0.6

    def test_backward_compat_keys_preserved(self):
        """Existing keys in to_agent_signal_dict must be preserved."""
        sig = ShortBiasSignal(direction=-1.0, confidence=0.5)
        d = sig.to_agent_signal_dict()
        assert "direction" in d
        assert "confidence" in d
        assert "veto_active" in d
        assert "veto_direction" in d
        assert "leverage_cap" in d
        assert "exit_pressure" in d
        assert "tranche_advice" in d


# ---------------------------------------------------------------------------
# 6. Input Normalization (P1)
# ---------------------------------------------------------------------------

class TestInputNormalization:
    """P1: Auto-detect and normalize fear/greed scale."""

    def test_fear_greed_0_100_scale(self):
        """fear_greed_index=85 (0-100 scale) should normalize to 0.85."""
        val, interp = ShortBiasAgent._normalize_fear_greed(85)
        assert val == pytest.approx(0.85)
        assert interp == "assumed_0_100_scale"

    def test_fear_greed_0_1_scale(self):
        """fear_greed_index=0.85 (0-1 scale) should stay 0.85."""
        val, interp = ShortBiasAgent._normalize_fear_greed(0.85)
        assert val == pytest.approx(0.85)
        assert interp == "assumed_0_1_scale"

    def test_fear_greed_clamped(self):
        """Values > 100 should clamp to 1.0."""
        val, _ = ShortBiasAgent._normalize_fear_greed(150)
        assert val == 1.0

    def test_fear_greed_negative_clamped(self):
        """Negative values should clamp to 0.0."""
        val, _ = ShortBiasAgent._normalize_fear_greed(-5)
        assert val == 0.0

    def test_funding_rate_passthrough(self):
        """Funding rate should pass through with unit annotation."""
        val, unit = ShortBiasAgent._normalize_funding_rate(0.0005)
        assert val == 0.0005
        assert unit == "per_8h"

    def test_normalization_recorded_in_meta(self):
        """Both normalizations should appear in signal meta."""
        agent = _make_agent()
        sig = agent.analyze(
            _normal_market(),
            {"funding_rate": 0.001, "open_interest": 1e9, "liquidations_long": 0, "liquidations_short": 0},
            {"fear_greed_index": 75},
        )
        norm = sig.meta.get("normalization", {})
        assert "fear_greed" in norm
        assert "funding_rate" in norm


# ---------------------------------------------------------------------------
# 7. Backward Compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    """Verify existing public API contracts are preserved."""

    def test_singleton_pattern(self):
        """get/reset singleton works."""
        reset_short_bias_agent()
        a = get_short_bias_agent()
        b = get_short_bias_agent()
        assert a is b
        reset_short_bias_agent()

    def test_analyze_no_kwargs(self):
        """analyze(market_data) with no kwargs should work."""
        agent = _make_agent()
        sig = agent.analyze({"price": 100, "volume": 1e6})
        assert sig.direction <= 0

    def test_analyze_all_positional(self):
        """analyze(m, d, s, o) with all 4 positional args should work."""
        agent = _make_agent()
        sig = agent.analyze(
            {"price": 100, "volume": 1e6},
            {"funding_rate": 0.0001},
            {"fear_greed_index": 0.5},
            {"whale_flow_usd": 0},
        )
        assert sig.direction <= 0

    def test_signal_has_legacy_fields(self):
        """ShortBiasSignal must have all legacy fields."""
        sig = ShortBiasSignal()
        assert hasattr(sig, "direction")
        assert hasattr(sig, "confidence")
        assert hasattr(sig, "primary_trigger")
        assert hasattr(sig, "components")
        assert hasattr(sig, "veto_active")
        assert hasattr(sig, "veto_direction")
        assert hasattr(sig, "timestamp")
        assert hasattr(sig, "suggested_stop_pct")
        assert hasattr(sig, "max_exposure_pct")
        assert hasattr(sig, "reasons")

    def test_get_stats_backward_compat(self):
        """get_stats() with no args should return combined stats."""
        agent = _make_agent()
        agent.analyze({"asset": "BTC", "price": 100, "volume": 1e6})
        stats = agent.get_stats()
        assert "total_signals" in stats
        assert "config" in stats

    def test_get_stats_per_asset(self):
        """get_stats(asset='BTC') should return BTC-specific stats."""
        agent = _make_agent()
        agent.analyze({"asset": "BTC", "price": 100, "volume": 1e6})
        stats = agent.get_stats("BTC")
        assert "total_signals" in stats
        assert "history_lengths" in stats

    def test_reset_state_all(self):
        """reset_state() clears all gating state."""
        agent = _make_agent(signal_cooldown_minutes=0)
        agent.analyze(
            _extreme_market("BTC"), _extreme_derivatives(),
            _extreme_sentiment(), _extreme_onchain(),
        )
        agent.reset_state()
        assert agent._consecutive_signals.get("BTC", 0) == 0

    def test_reset_state_per_asset(self):
        """reset_state(asset='BTC') clears only BTC state."""
        agent = _make_agent(signal_cooldown_minutes=0)
        agent.analyze(
            _extreme_market("BTC"), _extreme_derivatives(),
            _extreme_sentiment(), _extreme_onchain(),
        )
        agent.analyze(
            _extreme_market("ETH"), _extreme_derivatives(),
            _extreme_sentiment(), _extreme_onchain(),
            asset="ETH",
        )
        agent.reset_state("BTC")
        assert agent._consecutive_signals.get("BTC", 0) == 0
        # ETH should be unaffected
        # (ETH consecutive may be 0 or 1 depending on signal)

    def test_exception_in_analyze_returns_neutral(self):
        """Even if analysis crashes internally, a valid signal is returned."""
        agent = _make_agent()
        # Corrupt internal state to force error
        agent._price_history = None  # will cause AttributeError
        sig = agent.analyze({"price": 100, "volume": 1e6})
        assert sig.direction == 0
        assert sig.confidence == 0
        assert "internal_error" in str(sig.reasons) or "error" in str(sig.meta)


# ---------------------------------------------------------------------------
# Signal Logic Correctness
# ---------------------------------------------------------------------------

class TestSignalLogic:
    """Verify signal computation correctness."""

    def test_funding_arb_short_boost(self):
        """Extreme positive funding should boost short confidence."""
        agent = _make_agent(signal_cooldown_minutes=0)
        # First call to build history
        sig1 = agent.analyze(
            _extreme_market(), _extreme_derivatives(),
            _extreme_sentiment(), _extreme_onchain(),
        )

        # Second call with extreme funding
        extreme_deriv = {
            "funding_rate": 0.003,  # 0.3% > 0.24% threshold
            "open_interest": 2e9,
            "liquidations_long": 50e6,
            "liquidations_short": 5e6,
        }
        sig2 = agent.analyze(
            _extreme_market(), extreme_deriv,
            _extreme_sentiment(), _extreme_onchain(),
        )
        if sig2.is_actionable():
            assert any("FUNDING_ARB" in r for r in sig2.reasons)

    def test_regime_quiet_penalty(self):
        """Quiet regime should penalize confidence."""
        agent = _make_agent(signal_cooldown_minutes=0)
        sig_normal = agent.analyze(
            _extreme_market(), _extreme_derivatives(),
            _extreme_sentiment(), _extreme_onchain(),
        )

        agent2 = _make_agent(signal_cooldown_minutes=0)
        sig_quiet = agent2.analyze(
            _extreme_market(), _extreme_derivatives(),
            _extreme_sentiment(), _extreme_onchain(),
            regime_state="QUIET", regime_confidence=0.8,
        )

        if sig_normal.is_actionable() and sig_quiet.is_actionable():
            assert sig_quiet.confidence <= sig_normal.confidence
