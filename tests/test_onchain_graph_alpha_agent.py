"""
Tests for agents/onchain_graph_alpha.py - OnChainGraphAlphaAgent v6 wrapper.

Groups:
  1. No-engine gating (2 tests) - no engine, missing engine produces neutral
  2. No-data gating (2 tests) - engine exists but no signals -> neutral
  3. Signal generation (3 tests) - active signals -> valid output with expected keys
  4. Freshness gating (1 test) - stale snapshot -> neutral
  5. to_agent_signal_dict (2 tests) - caching, re-generation on asset change
"""
import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from agents.onchain_graph_alpha import (
    OnChainGraphAlphaAgent,
    OnChainGraphAlphaEngine,
    OnChainAlphaSignal,
    OnChainSignalType,
    reset_onchain_alpha_engine,
)


# ============================================================================
# HELPERS
# ============================================================================

_REQUIRED_KEYS = {
    "onchain_graph_direction",
    "onchain_graph_confidence",
    "onchain_graph_data_quality",
    "onchain_graph_valid",
    "onchain_graph_reasons",
    "onchain_graph_n_signals",
    "asset",
}


def _make_signal(
    direction: float = 0.5,
    confidence: float = 0.7,
    strength: float = 0.8,
    signal_type: OnChainSignalType = OnChainSignalType.WHALE_ACCUMULATION,
    age_seconds: float = 10.0,
) -> OnChainAlphaSignal:
    """Create a mock OnChainAlphaSignal with a recent timestamp."""
    return OnChainAlphaSignal(
        timestamp=datetime.now() - timedelta(seconds=age_seconds),
        signal_type=signal_type,
        symbol="SOL",
        direction=direction,
        confidence=confidence,
        strength=strength,
        reasoning="test signal",
    )


def _make_engine_with_signals(n: int = 2) -> OnChainGraphAlphaEngine:
    """Create an engine with n active signals."""
    engine = OnChainGraphAlphaEngine()
    engine.active_signals = [_make_signal() for _ in range(n)]
    return engine


# ============================================================================
# GROUP 1: No-engine gating (2 tests)
# ============================================================================

class TestNoEngineGating:
    def test_no_engine_returns_neutral(self):
        agent = OnChainGraphAlphaAgent(engine=None)
        sig = agent.generate_signal("SOL")
        assert sig["onchain_graph_direction"] == 0.0
        assert sig["onchain_graph_confidence"] == 0.0
        assert sig["onchain_graph_valid"] is False
        assert "no_engine" in sig["onchain_graph_reasons"]

    def test_all_required_keys_present_on_neutral(self):
        agent = OnChainGraphAlphaAgent(engine=None)
        sig = agent.generate_signal("BTC")
        for key in _REQUIRED_KEYS:
            assert key in sig, f"Missing key: {key}"
        assert sig["asset"] == "BTC"


# ============================================================================
# GROUP 2: No-data gating (2 tests)
# ============================================================================

class TestNoDataGating:
    def test_empty_engine_returns_neutral(self):
        engine = OnChainGraphAlphaEngine()
        assert len(engine.active_signals) == 0
        agent = OnChainGraphAlphaAgent(engine=engine)
        sig = agent.generate_signal("SOL")
        assert sig["onchain_graph_valid"] is False
        assert "no_onchain_data" in sig["onchain_graph_reasons"]

    def test_gating_reason_recorded(self):
        engine = OnChainGraphAlphaEngine()
        agent = OnChainGraphAlphaAgent(engine=engine)
        agent.generate_signal("SOL")
        assert agent._last_gating_reason == "no_data"


# ============================================================================
# GROUP 3: Signal generation (3 tests)
# ============================================================================

class TestSignalGeneration:
    def test_valid_signal_has_correct_fields(self):
        engine = _make_engine_with_signals(3)
        agent = OnChainGraphAlphaAgent(engine=engine)
        sig = agent.generate_signal("SOL")
        assert sig["onchain_graph_valid"] is True
        assert sig["onchain_graph_n_signals"] == 3
        assert 0.0 <= sig["onchain_graph_confidence"] <= 1.0
        assert -1.0 <= sig["onchain_graph_direction"] <= 1.0
        assert sig["onchain_graph_data_quality"] > 0.0

    def test_direction_matches_engine_aggregate(self):
        engine = OnChainGraphAlphaEngine()
        engine.active_signals = [
            _make_signal(direction=0.8, confidence=0.9, strength=1.0),
        ]
        agent = OnChainGraphAlphaAgent(engine=engine, directional_alpha_enabled=True)
        sig = agent.generate_signal("SOL")
        assert sig["onchain_graph_direction"] > 0.0

    def test_reasons_contain_signal_types(self):
        engine = OnChainGraphAlphaEngine()
        engine.active_signals = [
            _make_signal(signal_type=OnChainSignalType.EXCHANGE_INFLOW),
            _make_signal(signal_type=OnChainSignalType.DEX_VOLUME_SPIKE),
        ]
        agent = OnChainGraphAlphaAgent(engine=engine)
        sig = agent.generate_signal("SOL")
        assert "exchange_inflow" in sig["onchain_graph_reasons"]
        assert "dex_volume_spike" in sig["onchain_graph_reasons"]


# ============================================================================
# GROUP 4: Freshness gating (1 test)
# ============================================================================

class TestFreshnessGating:
    def test_stale_snapshot_returns_neutral(self):
        engine = OnChainGraphAlphaEngine()
        # Signal from 5 hours ago (> 4h MAX_SNAPSHOT_AGE_SECONDS)
        engine.active_signals = [_make_signal(age_seconds=18000)]
        agent = OnChainGraphAlphaAgent(engine=engine)
        sig = agent.generate_signal("SOL")
        assert sig["onchain_graph_valid"] is False
        assert "stale_snapshot" in sig["onchain_graph_reasons"]
        assert "stale" in (agent._last_gating_reason or "")


# ============================================================================
# GROUP 5: to_agent_signal_dict (2 tests)
# ============================================================================

class TestToAgentSignalDict:
    def test_caches_last_signal(self):
        engine = _make_engine_with_signals(2)
        agent = OnChainGraphAlphaAgent(engine=engine)
        sig1 = agent.generate_signal("SOL")
        sig2 = agent.to_agent_signal_dict("SOL")
        assert sig1 is sig2

    def test_regenerates_on_different_asset(self):
        engine = _make_engine_with_signals(2)
        agent = OnChainGraphAlphaAgent(engine=engine)
        sig_sol = agent.generate_signal("SOL")
        assert sig_sol["asset"] == "SOL"
        sig_btc = agent.to_agent_signal_dict("BTC")
        assert sig_btc["asset"] == "BTC"


# ============================================================================
# GROUP 6: directional_alpha_enabled flag (3 tests)
# ============================================================================

class TestDirectionalAlphaEnabled:
    """v6.4: directional_alpha_enabled=False zeroes direction/confidence."""

    def test_disabled_zeroes_direction(self):
        engine = _make_engine_with_signals(3)
        agent = OnChainGraphAlphaAgent(engine=engine, directional_alpha_enabled=False)
        sig = agent.generate_signal("SOL")
        assert sig["onchain_graph_direction"] == 0.0
        assert sig["onchain_graph_confidence"] == 0.0
        # But should still be "valid" (engine has data)
        assert sig["onchain_graph_valid"] is True

    def test_enabled_preserves_direction(self):
        engine = OnChainGraphAlphaEngine()
        engine.active_signals = [
            _make_signal(direction=0.8, confidence=0.9, strength=1.0),
        ]
        agent = OnChainGraphAlphaAgent(engine=engine, directional_alpha_enabled=True)
        sig = agent.generate_signal("SOL")
        assert sig["onchain_graph_direction"] != 0.0

    def test_default_is_disabled(self):
        agent = OnChainGraphAlphaAgent()
        assert agent.directional_alpha_enabled is False


# ============================================================================
# GROUP 7: get_feature_snapshot v6.4 (4 tests)
# ============================================================================

class TestGetFeatureSnapshot:
    """v6.4 feature snapshot without forced direction."""

    def test_no_engine_returns_missing(self):
        agent = OnChainGraphAlphaAgent(engine=None)
        snap = agent.get_feature_snapshot("SOL")
        assert snap["available"] is False
        assert snap["data_quality"] == "missing"
        assert snap["whale_activity_count"] == 0
        assert "provenance" in snap
        assert snap["asof_ts"].endswith("Z")

    def test_with_signals_available_true(self):
        engine = _make_engine_with_signals(2)
        agent = OnChainGraphAlphaAgent(engine=engine)
        snap = agent.get_feature_snapshot("SOL")
        assert snap["available"] is True
        assert snap["active_signals"] == 2
        assert snap["data_quality"] in ("ok", "stale", "missing")

    def test_stale_signals_quality_stale(self):
        engine = OnChainGraphAlphaEngine()
        engine.active_signals = [_make_signal(age_seconds=18000)]
        agent = OnChainGraphAlphaAgent(engine=engine)
        snap = agent.get_feature_snapshot("SOL")
        assert snap["data_quality"] == "stale"

    def test_empty_engine_quality_missing(self):
        engine = OnChainGraphAlphaEngine()
        agent = OnChainGraphAlphaAgent(engine=engine)
        snap = agent.get_feature_snapshot("SOL")
        assert snap["data_quality"] == "missing"
        assert snap["available"] is False
