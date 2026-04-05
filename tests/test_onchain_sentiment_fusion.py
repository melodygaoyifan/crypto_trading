"""
Tests for agents/onchain_sentiment_fusion.py v6.4 alignment.

Groups:
  1. enabled=False default (3 tests) - disabled noop, neutral context, no weight
  2. build_context_packet (3 tests) - disabled, keys, quality string
  3. to_agent_signal_dict (4 tests) - keys, no weight, metadata, disabled
"""
import pytest
from datetime import datetime

from agents.onchain_sentiment_fusion import (
    OnChainSentimentAlphaEngine,
    AlphaEngineConfig,
)


# ============================================================================
# HELPERS
# ============================================================================

def _make_engine(**kw) -> OnChainSentimentAlphaEngine:
    cfg = AlphaEngineConfig()
    return OnChainSentimentAlphaEngine(config=cfg, **kw)


# ============================================================================
# GROUP 1: enabled=False default (3 tests)
# ============================================================================

class TestEnabledDefault:
    def test_default_is_disabled(self):
        engine = _make_engine()
        assert engine.enabled is False

    def test_disabled_context_neutral(self):
        engine = _make_engine(enabled=False)
        pkt = engine.build_context_packet("SOL")
        assert pkt["onchain_whale_net_flow"] == 0.0
        assert pkt["onchain_exchange_net_flow"] == 0.0
        assert pkt["data_quality_str"] == "disabled"

    def test_enabled_explicit(self):
        engine = _make_engine(enabled=True)
        assert engine.enabled is True


# ============================================================================
# GROUP 2: build_context_packet (3 tests)
# ============================================================================

class TestBuildContextPacket:
    def test_disabled_returns_all_required_keys(self):
        engine = _make_engine(enabled=False)
        pkt = engine.build_context_packet("BTC")
        required = {
            "onchain_whale_net_flow", "onchain_exchange_net_flow",
            "onchain_smart_money_bias", "sentiment_fear_greed",
            "sentiment_social_score", "sentiment_news_score",
            "funding_rate", "data_quality", "source",
        }
        for k in required:
            assert k in pkt, f"Missing key: {k}"

    def test_disabled_fear_greed_neutral(self):
        engine = _make_engine(enabled=False)
        pkt = engine.build_context_packet("ETH")
        assert pkt["sentiment_fear_greed"] == 50  # neutral

    def test_disabled_source_correct(self):
        engine = _make_engine(enabled=False)
        pkt = engine.build_context_packet("SOL")
        assert pkt["source"] == "onchain_sentiment_context"


# ============================================================================
# GROUP 3: to_agent_signal_dict (4 tests)
# ============================================================================

class TestToAgentSignalDict:
    def test_no_weight_key(self):
        """v6.4 rule: no 'weight' key emitted."""
        engine = _make_engine(enabled=False)
        sig = engine.to_agent_signal_dict("SOL")
        assert "weight" not in sig
        assert "ocs_weight" not in sig

    def test_direction_always_zero(self):
        """v6: no pre-fused direction/confidence."""
        engine = _make_engine(enabled=False)
        sig = engine.to_agent_signal_dict("SOL")
        assert sig["ocs_direction"] == 0.0
        assert sig["ocs_confidence"] == 0.0

    def test_v64_metadata_present(self):
        engine = _make_engine(enabled=False)
        sig = engine.to_agent_signal_dict("SOL")
        assert "data_quality" in sig
        assert "asof_ts" in sig
        assert sig["asof_ts"].endswith("Z")
        assert "data_age_seconds" in sig
        assert "provenance" in sig
        assert "asset" in sig

    def test_disabled_data_quality_disabled(self):
        engine = _make_engine(enabled=False)
        sig = engine.to_agent_signal_dict("SOL")
        assert sig["data_quality"] == "disabled"
