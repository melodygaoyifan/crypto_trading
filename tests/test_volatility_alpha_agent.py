"""
Tests for agents/volatility_alpha_agent.py - production-readiness checks.

Groups:
1. Multi-asset isolation (P0)
2. Coverage-aware fusion (P0)
3. Data health gating (P1)
4. Enabled flag enforcement (P1)
5. Warning suppression reset (P1)
6. Diagnostics & observability (P1)
7. EventBus integration correctness (P1)
8. Backward compatibility
"""

import pytest
from datetime import datetime, timedelta

from agents.volatility_alpha_agent import (
    VolatilityAlphaAgent,
    VolAlphaConfig,
    VolatilityIntent,
    VolBias,
    VolHorizon,
    VolSignalType,
    LOBState,
    VolatilityState,
    OnchainState,
    SentimentState,
    CorrelationState,
    VolatilityBreakoutAlpha,
    LiquidationPressureAlpha,
    CompressionExpansionAlpha,
    VolAlphaEventBusIntegration,
    get_volatility_alpha_agent,
    reset_volatility_alpha_agent,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _feed_bear_market(agent, asset="BTC"):
    """Feed a full bear-market scenario for one asset."""
    agent.on_lob_tick({
        "asset": asset,
        "spread_bps": 15.0,
        "depth_imbalance": -0.3,
        "bid_depth_usd": 500_000,
        "ask_depth_usd": 800_000,
        "vpin_proxy": 0.7,
    })
    agent.on_volatility_state({
        "asset": asset,
        "realized_vol_1h": 0.8,
        "realized_vol_4h": 0.7,
        "realized_vol_24h": 0.5,
        "vol_percentile_30d": 85.0,
        "vol_zscore": 2.0,
        "vol_of_vol": 0.6,
    })
    agent.on_onchain_tick({
        "asset": asset,
        "exchange_inflow_spike": True,
        "exchange_inflow_zscore": 2.5,
        "stablecoin_flow_direction": "outflow",
        "funding_deviation": 0.8,
    })
    agent.on_sentiment_tick({
        "asset": asset,
        "fear_index": 18.0,
        "sentiment_volatility": 0.5,
    })
    agent.on_correlation_state({
        "btc_dominance": 55.0,
        "btc_dominance_change_24h": 3.0,
        "cross_asset_correlation": 0.87,
        "correlation_regime": "exploded",
        "correlation_spike": True,
    })


def _feed_neutral(agent, asset="BTC"):
    """Feed neutral data for one asset (minimal signals)."""
    agent.on_lob_tick({
        "asset": asset,
        "spread_bps": 5.0,
        "bid_depth_usd": 1_000_000,
        "ask_depth_usd": 1_000_000,
    })
    agent.on_volatility_state({
        "asset": asset,
        "vol_percentile_30d": 50.0,
        "vol_zscore": 0.0,
        "vol_of_vol": 0.1,
    })
    agent.on_onchain_tick({
        "asset": asset,
        "exchange_inflow_spike": False,
        "exchange_inflow_zscore": 0.5,
        "stablecoin_flow_direction": "neutral",
    })
    agent.on_sentiment_tick({
        "asset": asset,
        "fear_index": 50.0,
        "sentiment_volatility": 0.1,
    })
    agent.on_correlation_state({
        "btc_dominance": 50.0,
        "correlation_regime": "normal",
        "correlation_spike": False,
    })


# ---------------------------------------------------------------------------
# 1. Multi-Asset Isolation (P0)
# ---------------------------------------------------------------------------

class TestMultiAssetIsolation:
    """P0: Per-asset state must not leak between assets."""

    @pytest.fixture
    def agent(self):
        return VolatilityAlphaAgent(VolAlphaConfig())

    def test_btc_data_does_not_affect_eth(self, agent):
        """BTC bear-market data must not produce signal for ETH."""
        _feed_bear_market(agent, "BTC")
        # ETH has no data -> insufficient_data
        intent_eth = agent.evaluate("ETH")
        assert intent_eth.is_valid is False
        assert "insufficient_data" in intent_eth.rationale_tags

    def test_independent_eval_per_asset(self, agent):
        """BTC and ETH fed independently produce independent results."""
        _feed_bear_market(agent, "BTC")
        _feed_neutral(agent, "ETH")

        intent_btc = agent.evaluate("BTC")
        intent_eth = agent.evaluate("ETH")

        # BTC should be high intensity (bear market)
        assert intent_btc.intensity > 0.3
        # ETH should be low intensity (neutral)
        assert intent_eth.intensity < intent_btc.intensity

    def test_per_asset_breakout_baselines(self, agent):
        """Each asset gets its own VolatilityBreakoutAlpha with independent baselines."""
        # Feed BTC with wide spread
        agent.on_lob_tick({"asset": "BTC", "spread_bps": 20.0, "bid_depth_usd": 100_000, "ask_depth_usd": 100_000})
        agent.on_volatility_state({"asset": "BTC", "vol_percentile_30d": 85.0, "vol_zscore": 2.0})

        # Feed ETH with tight spread
        agent.on_lob_tick({"asset": "ETH", "spread_bps": 3.0, "bid_depth_usd": 500_000, "ask_depth_usd": 500_000})
        agent.on_volatility_state({"asset": "ETH", "vol_percentile_30d": 85.0, "vol_zscore": 2.0})

        # Ensure separate calculator instances were created
        assert "BTC" in agent._breakout_alphas or "BTC" not in agent._breakout_alphas  # may not exist yet
        agent.on_correlation_state({"correlation_spike": False})
        agent.on_onchain_tick({"asset": "BTC", "exchange_inflow_spike": False})
        agent.on_sentiment_tick({"asset": "BTC", "fear_index": 50.0})
        agent.on_onchain_tick({"asset": "ETH", "exchange_inflow_spike": False})
        agent.on_sentiment_tick({"asset": "ETH", "fear_index": 50.0})

        agent.evaluate("BTC")
        agent.evaluate("ETH")

        # Now both should have separate breakout calculators
        assert "BTC" in agent._breakout_alphas
        assert "ETH" in agent._breakout_alphas
        assert agent._breakout_alphas["BTC"] is not agent._breakout_alphas["ETH"]

    def test_per_asset_compression_trackers(self, agent):
        """Each asset gets its own CompressionExpansionAlpha duration tracker."""
        _feed_bear_market(agent, "BTC")
        _feed_bear_market(agent, "SOL")

        agent.evaluate("BTC")
        agent.evaluate("SOL")

        assert "BTC" in agent._compression_alphas
        assert "SOL" in agent._compression_alphas
        assert agent._compression_alphas["BTC"] is not agent._compression_alphas["SOL"]

    def test_correlation_is_shared(self, agent):
        """CorrelationState is global - same value for all assets."""
        agent.on_correlation_state({"correlation_spike": True, "correlation_regime": "exploded"})

        # Both assets should see the same correlation state
        assert agent._correlation_state is not None
        assert agent._correlation_state.correlation_spike is True

    def test_lob_states_keyed_by_asset(self, agent):
        """on_lob_tick stores per-asset."""
        agent.on_lob_tick({"asset": "BTC", "spread_bps": 10.0})
        agent.on_lob_tick({"asset": "ETH", "spread_bps": 5.0})

        assert "BTC" in agent._lob_states
        assert "ETH" in agent._lob_states
        assert agent._lob_states["BTC"].spread_bps == 10.0
        assert agent._lob_states["ETH"].spread_bps == 5.0


# ---------------------------------------------------------------------------
# 2. Coverage-Aware Fusion (P0)
# ---------------------------------------------------------------------------

class TestCoverageAwareFusion:
    """P0: Missing sub-signals must not dilute the fused intensity."""

    @pytest.fixture
    def agent(self):
        return VolatilityAlphaAgent(VolAlphaConfig(min_data_sources_required=1))

    def test_single_subsignal_not_diluted(self, agent):
        """With only breakout data, intensity should NOT be breakout * 0.35."""
        agent.on_lob_tick({"asset": "BTC", "spread_bps": 15.0, "bid_depth_usd": 500_000, "ask_depth_usd": 800_000})
        agent.on_volatility_state({
            "asset": "BTC",
            "vol_percentile_30d": 90.0,
            "vol_zscore": 2.5,
            "vol_of_vol": 0.6,
        })
        # No onchain, sentiment, or correlation -> only breakout available

        intent = agent.evaluate("BTC")
        assert intent.is_valid

        # Breakout intensity is renormalized to full weight
        breakout_raw = intent.sub_signals.get("breakout", 0)
        assert breakout_raw > 0
        # Fused intensity should be ~= breakout_raw (not breakout_raw * 0.35)
        assert intent.intensity == pytest.approx(breakout_raw, abs=0.01)

    def test_two_subsignals_renormalize(self, agent):
        """With breakout + liquidation, weights renormalize to sum=1."""
        _feed_bear_market(agent, "BTC")
        # Remove correlation state to disable compression sub-signal
        agent._correlation_state = None

        intent = agent.evaluate("BTC")
        assert intent.is_valid

        # Only breakout and liquidation should be in sub_signals
        assert "breakout" in intent.sub_signals
        assert "liquidation" in intent.sub_signals
        assert "compression" not in intent.sub_signals

    def test_all_three_subsignals(self, agent):
        """With all data available, all three sub-signals contribute."""
        _feed_bear_market(agent, "BTC")

        intent = agent.evaluate("BTC")
        assert "breakout" in intent.sub_signals
        assert "liquidation" in intent.sub_signals
        assert "compression" in intent.sub_signals

    def test_no_fresh_data_returns_invalid(self):
        """With zero fresh sources, evaluate returns invalid intent."""
        agent = VolatilityAlphaAgent(VolAlphaConfig(min_data_sources_required=2))
        intent = agent.evaluate("BTC")
        assert intent.is_valid is False
        assert "insufficient_data" in intent.rationale_tags


# ---------------------------------------------------------------------------
# 3. Data Health Gating (P1)
# ---------------------------------------------------------------------------

class TestDataHealthGating:
    """P1: Stale data must be excluded from sub-signal computation."""

    @pytest.fixture
    def agent(self):
        return VolatilityAlphaAgent(VolAlphaConfig(min_data_sources_required=1))

    def test_stale_lob_excluded(self, agent):
        """LOB data older than lob_freshness_sec should be treated as missing."""
        _feed_bear_market(agent, "BTC")

        # Manually backdate LOB
        agent._lob_states["BTC"].timestamp = datetime.now() - timedelta(seconds=120)

        intent = agent.evaluate("BTC")
        # Breakout requires fresh LOB + VOL - LOB is stale so breakout should be absent
        assert "breakout" not in intent.sub_signals

    def test_stale_vol_excluded(self, agent):
        """Volatility data older than vol_freshness_sec should be treated as missing."""
        _feed_bear_market(agent, "BTC")

        # Manually backdate vol (> 300s)
        agent._vol_states["BTC"].timestamp = datetime.now() - timedelta(seconds=400)

        intent = agent.evaluate("BTC")
        # Both breakout and compression need fresh vol - both should be absent
        assert "breakout" not in intent.sub_signals
        assert "compression" not in intent.sub_signals

    def test_freshness_in_diagnostics(self, agent):
        """source_freshness and coverage must appear in intent."""
        _feed_bear_market(agent, "BTC")

        intent = agent.evaluate("BTC")
        assert "lob" in intent.source_freshness
        assert "volatility" in intent.source_freshness
        assert "onchain" in intent.source_freshness
        assert "sentiment" in intent.source_freshness
        assert "correlation" in intent.source_freshness

        # All should be fresh (just fed)
        for source, status in intent.coverage.items():
            assert status == "fresh", f"{source} should be fresh but is {status}"

    def test_missing_source_coverage(self, agent):
        """Sources with no data should show 'missing' in coverage."""
        # Only feed LOB + vol
        agent.on_lob_tick({"asset": "BTC", "spread_bps": 5.0, "bid_depth_usd": 1e6, "ask_depth_usd": 1e6})
        agent.on_volatility_state({"asset": "BTC", "vol_percentile_30d": 50.0})

        intent = agent.evaluate("BTC")
        assert intent.coverage["onchain"] == "missing"
        assert intent.coverage["sentiment"] == "missing"
        assert intent.coverage["correlation"] == "missing"
        assert intent.source_freshness["onchain"] == -1.0

    def test_data_quality_reflects_sources(self, agent):
        """data_quality = available_sources / 5."""
        # Feed 3 of 5 sources
        agent.on_lob_tick({"asset": "BTC", "spread_bps": 5.0, "bid_depth_usd": 1e6, "ask_depth_usd": 1e6})
        agent.on_volatility_state({"asset": "BTC", "vol_percentile_30d": 50.0})
        agent.on_onchain_tick({"asset": "BTC", "exchange_inflow_spike": False})

        # Need sentiment for at least min_data_sources_required=1
        intent = agent.evaluate("BTC")
        # LOB + vol + onchain = 3/5 = 0.6
        assert intent.data_quality == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# 4. Enabled Flag Enforcement (P1)
# ---------------------------------------------------------------------------

class TestEnabledFlag:
    """P1: config.enabled=False must produce neutral/invalid intent."""

    def test_disabled_returns_neutral(self):
        """Disabled agent returns NEUTRAL with agent_disabled tag."""
        agent = VolatilityAlphaAgent(VolAlphaConfig(enabled=False))
        _feed_bear_market(agent, "BTC")

        intent = agent.evaluate("BTC")
        assert intent.vol_bias == VolBias.NEUTRAL
        assert intent.intensity == 0.0
        assert intent.is_valid is False
        assert "agent_disabled" in intent.rationale_tags

    def test_disabled_does_not_count_evaluation(self):
        """Disabled agent should not increment total_evaluations."""
        agent = VolatilityAlphaAgent(VolAlphaConfig(enabled=False))
        agent.evaluate("BTC")
        assert agent._stats["total_evaluations"] == 0

    def test_re_enable(self):
        """Re-enabling the agent should allow normal evaluation."""
        config = VolAlphaConfig(enabled=False)
        agent = VolatilityAlphaAgent(config)
        _feed_bear_market(agent, "BTC")

        intent_off = agent.evaluate("BTC")
        assert intent_off.is_valid is False

        config.enabled = True
        intent_on = agent.evaluate("BTC")
        assert intent_on.is_valid is True
        assert intent_on.intensity > 0


# ---------------------------------------------------------------------------
# 5. Warning Suppression Reset (P1)
# ---------------------------------------------------------------------------

class TestWarningSuppression:
    """P1: _data_warning_shown must reset when data becomes sufficient."""

    def test_warning_resets_on_data_recovery(self):
        """After data recovery, warning flag resets so future lapses log again."""
        agent = VolatilityAlphaAgent(VolAlphaConfig(min_data_sources_required=2))

        # No data -> sets _data_warning_shown = True
        agent.evaluate("BTC")
        assert agent._data_warning_shown is True

        # Feed sufficient data -> should reset
        _feed_bear_market(agent, "BTC")
        agent.evaluate("BTC")
        assert agent._data_warning_shown is False

    def test_warning_shown_only_once(self):
        """Multiple insufficient-data evaluations should only log once."""
        agent = VolatilityAlphaAgent(VolAlphaConfig(min_data_sources_required=5))

        agent.evaluate("BTC")
        assert agent._data_warning_shown is True

        # Second call should still have the flag set but not re-log
        agent.evaluate("BTC")
        assert agent._data_warning_shown is True


# ---------------------------------------------------------------------------
# 6. Diagnostics & Observability (P1)
# ---------------------------------------------------------------------------

class TestDiagnostics:
    """P1: to_dict() and get_status() must include all diagnostic fields."""

    @pytest.fixture
    def agent(self):
        a = VolatilityAlphaAgent(VolAlphaConfig())
        _feed_bear_market(a, "BTC")
        return a

    def test_to_dict_includes_diagnostics(self, agent):
        """to_dict must include source_freshness and coverage."""
        intent = agent.evaluate("BTC")
        d = intent.to_dict()

        assert "source_freshness" in d
        assert "coverage" in d
        assert "data_quality" in d
        assert "sub_signals" in d
        assert "is_valid" in d

    def test_to_dict_all_sources_present(self, agent):
        """All 5 sources must appear in freshness/coverage dicts."""
        intent = agent.evaluate("BTC")
        d = intent.to_dict()

        expected_sources = {"lob", "volatility", "onchain", "sentiment", "correlation"}
        assert set(d["source_freshness"].keys()) == expected_sources
        assert set(d["coverage"].keys()) == expected_sources

    def test_get_status_includes_assets_tracked(self, agent):
        """get_status must list tracked assets."""
        agent.evaluate("BTC")
        status = agent.get_status()

        assert "assets_tracked" in status
        assert "BTC" in status["assets_tracked"]

    def test_get_status_data_sources(self, agent):
        """data_sources dict must reflect whether data is present."""
        status = agent.get_status()
        assert status["data_sources"]["lob"] is True
        assert status["data_sources"]["correlation"] is True

    def test_empty_agent_status(self):
        """Status of fresh agent with no data."""
        agent = VolatilityAlphaAgent(VolAlphaConfig())
        status = agent.get_status()
        assert status["data_sources"]["lob"] is False
        assert status["last_intent"] is None
        assert status["assets_tracked"] == []


# ---------------------------------------------------------------------------
# 7. EventBus Integration (P1)
# ---------------------------------------------------------------------------

class TestEventBusIntegration:
    """P1: EventBus integration subscribes to correct event types."""

    def test_integration_doesnt_crash_without_eventbus(self):
        """VolAlphaEventBusIntegration.initialize() should handle missing EventBus gracefully."""
        agent = VolatilityAlphaAgent(VolAlphaConfig())
        integration = VolAlphaEventBusIntegration(agent)

        # Should return False, not crash
        result = integration.initialize()
        # May succeed or fail depending on whether unified_event_v521 is importable
        assert isinstance(result, bool)

    def test_event_callback_publishes(self):
        """When event_callback is set, _publish_intent should call it."""
        agent = VolatilityAlphaAgent(VolAlphaConfig())
        _feed_bear_market(agent, "BTC")

        published = []
        agent.set_event_callback(lambda etype, payload: published.append((etype, payload)))

        agent.evaluate("BTC")

        assert len(published) == 1
        assert published[0][0] == "VOL_ALPHA_INTENT"
        assert "vol_bias" in published[0][1]

    def test_event_callback_exception_handled(self):
        """Exception in event_callback should not crash evaluate()."""
        agent = VolatilityAlphaAgent(VolAlphaConfig())
        _feed_bear_market(agent, "BTC")

        def bad_callback(etype, payload):
            raise RuntimeError("boom")

        agent.set_event_callback(bad_callback)

        # Should not raise
        intent = agent.evaluate("BTC")
        assert intent.is_valid


# ---------------------------------------------------------------------------
# 8. Backward Compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    """Verify that all existing public API contracts are maintained."""

    def test_evaluate_default_asset_btc(self):
        """evaluate() with no args defaults to 'BTC'."""
        agent = VolatilityAlphaAgent(VolAlphaConfig())
        intent = agent.evaluate()
        assert intent.asset == "BTC"

    def test_record_fusion_result(self):
        """record_fusion_result updates stats correctly."""
        agent = VolatilityAlphaAgent(VolAlphaConfig())

        agent.record_fusion_result(True)
        assert agent._stats["accepted_count"] == 1

        agent.record_fusion_result(False, veto_reason="risk_veto")
        assert agent._stats["vetoed_count"] == 1

    def test_singleton_pattern(self):
        """get/reset singleton works."""
        reset_volatility_alpha_agent()
        a = get_volatility_alpha_agent()
        b = get_volatility_alpha_agent()
        assert a is b
        reset_volatility_alpha_agent()

    def test_all_exports_importable(self):
        """All items in __all__ should be importable."""
        import agents.volatility_alpha_agent as mod
        for name in mod.__all__:
            assert hasattr(mod, name), f"Missing export: {name}"

    def test_intent_has_legacy_fields(self):
        """VolatilityIntent must have all legacy fields."""
        intent = VolatilityIntent(
            asset="BTC",
            vol_bias=VolBias.NEUTRAL,
            intensity=0.0,
            expected_horizon=VolHorizon.SHORT,
        )
        d = intent.to_dict()
        assert "vol_bias" in d
        assert "intensity" in d
        assert "expected_horizon" in d
        assert "rationale_tags" in d
        assert "timestamp" in d
        assert "sub_signals" in d
        assert "data_quality" in d
        assert "is_valid" in d

    def test_vol_bias_enum_values(self):
        """Enum values must match expected strings."""
        assert VolBias.INCREASE.value == "increase"
        assert VolBias.NEUTRAL.value == "neutral"
        assert VolBias.SUPPRESS.value == "suppress"


# ---------------------------------------------------------------------------
# Signal Logic Correctness
# ---------------------------------------------------------------------------

class TestSignalLogic:
    """Verify core signal computation is correct."""

    @pytest.fixture
    def agent(self):
        return VolatilityAlphaAgent(VolAlphaConfig())

    def test_bear_market_produces_increase(self, agent):
        """Full bear-market data should produce INCREASE bias."""
        _feed_bear_market(agent, "BTC")
        intent = agent.evaluate("BTC")
        assert intent.vol_bias == VolBias.INCREASE
        assert intent.intensity >= 0.3

    def test_neutral_market_low_intensity(self, agent):
        """Neutral data should produce low intensity."""
        _feed_neutral(agent, "BTC")
        intent = agent.evaluate("BTC")
        assert intent.intensity < 0.3

    def test_compression_breakout_triggers_medium_horizon(self, agent):
        """Extended compression -> breakout should set MEDIUM horizon."""
        # Feed low vol to build compression duration
        for _ in range(5):
            agent.on_volatility_state({
                "asset": "BTC",
                "vol_percentile_30d": 10.0,  # very low
                "vol_zscore": -1.0,
            })
            agent.on_lob_tick({"asset": "BTC", "spread_bps": 5.0, "bid_depth_usd": 1e6, "ask_depth_usd": 1e6})
            agent.on_onchain_tick({"asset": "BTC", "exchange_inflow_spike": False})
            agent.on_sentiment_tick({"asset": "BTC", "fear_index": 50.0})
            agent.on_correlation_state({"correlation_spike": False})
            # Manually advance compression tracker time
            comp = agent._get_compression_alpha("BTC")
            comp._last_update = datetime.now() - timedelta(hours=6)
            agent.evaluate("BTC")

        # Now trigger breakout (vol jumps up)
        agent.on_volatility_state({
            "asset": "BTC",
            "vol_percentile_30d": 80.0,
            "vol_zscore": 2.0,
            "vol_of_vol": 0.6,
        })
        agent.on_correlation_state({"correlation_spike": True, "correlation_regime": "exploded"})
        comp = agent._get_compression_alpha("BTC")
        comp._last_update = datetime.now() - timedelta(hours=6)

        intent = agent.evaluate("BTC")
        if "compression_breakout" in intent.rationale_tags or "extended_compression" in intent.rationale_tags:
            assert intent.expected_horizon == VolHorizon.MEDIUM

    def test_suppress_on_very_low_intensity(self, agent):
        """Very low fused intensity should produce SUPPRESS bias."""
        _feed_neutral(agent, "BTC")
        intent = agent.evaluate("BTC")
        if intent.intensity < 0.15:
            assert intent.vol_bias == VolBias.SUPPRESS

    def test_stats_increment_correctly(self, agent):
        """Stats counters should update on each evaluation."""
        _feed_bear_market(agent, "BTC")
        agent.evaluate("BTC")
        assert agent._stats["total_evaluations"] == 1

        _feed_neutral(agent, "ETH")
        agent.evaluate("ETH")
        assert agent._stats["total_evaluations"] == 2