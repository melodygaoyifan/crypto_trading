"""
Tests for agents/model_alpha_agent.py v6 alignment.

Groups:
  1. UnifiedState.from_runtime (5 tests) - field extraction, coverage, missing keys
  2. Freshness gating (3 tests) - stale data, low coverage, happy path
  3. to_agent_signals (4 tests) - key format, neutral fallback, weight injection
  4. Weight controllers (3 tests) - safe clamping, bad input, defaults
  5. Torch-missing fallback (2 tests) - mock model, predict works
  6. Singleton lifecycle (2 tests) - get/reset
"""
import pytest
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from agents.model_alpha_agent import (
    ModelAlphaAgent,
    ModelIntent,
    UnifiedState,
    IntentDirection,
    ModelType,
    get_model_alpha_agent,
    reset_model_alpha_agent,
)


# ============================================================================
# HELPERS
# ============================================================================

def _make_agent(**kw) -> ModelAlphaAgent:
    kw.setdefault("allow_mock_model", True)
    return ModelAlphaAgent(**kw)


def _runtime_market_data(**overrides) -> dict:
    """Minimal valid v6 runtime market_data dict."""
    d = {
        "btc_price": 95000.0,
        "eth_price": 3400.0,
        "sol_price": 185.0,
        "timestamp": datetime.now(timezone.utc),
        "data_age_seconds": 1.0,
        "fear_greed": 45,
        "volatility_4h": 0.02,
    }
    d.update(overrides)
    return d


_AGENT_SIGNAL_REQUIRED_KEYS = {
    "model_alpha_direction",
    "model_alpha_confidence",
    "model_alpha_raw_prediction",
    "model_alpha_horizon",
    "model_alpha_weight",
    "model_alpha_asof_timestamp",
    "model_alpha_data_quality",
    "model_alpha_reasons",
}


# ============================================================================
# GROUP 1: UnifiedState.from_runtime (5 tests)
# ============================================================================

class TestFromRuntime:
    """Build UnifiedState from runtime dicts."""

    def test_happy_path_all_prices(self):
        md = _runtime_market_data()
        state = UnifiedState.from_runtime("BTC", md)
        assert state.prices["BTC"] == 95000.0
        assert state.prices["ETH"] == 3400.0
        assert state.prices["SOL"] == 185.0
        assert state._coverage == 1.0
        assert state._missing == [] or "data_age_seconds" not in state._missing

    def test_missing_price_lowers_coverage(self):
        md = _runtime_market_data()
        del md["sol_price"]
        state = UnifiedState.from_runtime("BTC", md)
        assert "SOL" not in state.prices
        assert "SOL_price" in state._missing
        # Coverage is now primary-asset based; missing sibling assets should not
        # penalize a BTC runtime payload.
        assert state._coverage == 1.0

    def test_nested_dict_prices(self):
        md = {
            "BTC": {"open": 94000, "high": 96000, "low": 93000, "close": 95500},
            "ETH": {"open": 3300, "high": 3500, "low": 3200, "close": 3400},
            "SOL": {"open": 180, "high": 190, "low": 175, "close": 185},
            "timestamp": datetime.now(timezone.utc),
            "data_age_seconds": 2.0,
        }
        state = UnifiedState.from_runtime("BTC", md)
        assert state.prices["BTC"] == 95500
        assert state.prices["SOL"] == 185

    def test_missing_timestamp_recorded(self):
        md = _runtime_market_data()
        del md["timestamp"]
        state = UnifiedState.from_runtime("BTC", md)
        assert "timestamp" in state._missing
        # Should still produce a valid state with utcnow fallback
        assert isinstance(state.timestamp, datetime)

    def test_agent_signals_feed_lob(self):
        md = _runtime_market_data()
        signals = {"order_book_imbalance": 0.35, "spread_bps": 4.2}
        state = UnifiedState.from_runtime("BTC", md, agent_signals=signals)
        assert state.lob_imbalance["BTC"] == 0.35
        assert state.spread_bps["BTC"] == 4.2


# ============================================================================
# GROUP 2: Freshness gating (3 tests)
# ============================================================================

class TestFreshnessGating:
    """Stale / incomplete data must produce None + diagnostics."""

    def test_stale_data_returns_none(self):
        agent = _make_agent()
        md = _runtime_market_data(data_age_seconds=60.0)
        state = UnifiedState.from_runtime("BTC", md)
        intent = agent.generate_intent(state, "BTC")
        assert intent is None
        assert "stale" in (agent._last_gating_reason or "")

    def test_low_coverage_returns_none(self):
        agent = _make_agent()
        # Only provide timestamp - no prices at all
        state = UnifiedState.from_runtime("BTC", {"timestamp": datetime.now(timezone.utc), "data_age_seconds": 0})
        assert state._coverage == pytest.approx(2 / 3, rel=1e-6)
        intent = agent.generate_intent(state, "BTC")
        assert intent is None
        assert "stale" not in (agent._last_gating_reason or "")

    def test_fresh_complete_produces_intent(self):
        agent = _make_agent()
        md = _runtime_market_data(
            data_age_seconds=1.0,
            btc_returns_1h=0.02,
            btc_volatility=0.5,
        )
        state = UnifiedState.from_runtime("BTC", md)
        intent = agent.generate_intent(state, "BTC")
        # May still be None if confidence is too low, but not gated
        assert agent._last_gating_reason is None or "stale" not in (agent._last_gating_reason or "")


# ============================================================================
# GROUP 3: to_agent_signals (4 tests)
# ============================================================================

class TestToAgentSignals:
    """Agent signals output format and gating."""

    def test_all_required_keys_present(self):
        agent = _make_agent()
        md = _runtime_market_data()
        sig = agent.to_agent_signals("BTC", market_data=md)
        for key in _AGENT_SIGNAL_REQUIRED_KEYS:
            assert key in sig, f"Missing key: {key}"

    def test_neutral_on_no_state(self):
        agent = _make_agent()
        sig = agent.to_agent_signals("BTC")
        assert sig["model_alpha_direction"] == 0.0
        assert sig["model_alpha_confidence"] == 0.0
        assert sig["model_alpha_data_quality"] == 0.0
        assert any("gated" in r for r in sig["model_alpha_reasons"])

    def test_intent_to_agent_signals_format(self):
        intent = ModelIntent(
            asset="ETH",
            direction=IntentDirection.LONG,
            confidence=0.75,
            horizon="medium",
            raw_prediction=0.42,
            explanation_tokens=["bullish", "momentum"],
        )
        sig = intent.to_agent_signals()
        assert sig["model_alpha_direction"] == 1.0
        assert sig["model_alpha_confidence"] == 0.75
        assert sig["model_alpha_horizon"] == "medium"

    def test_weight_injected_into_signals(self):
        agent = _make_agent()
        agent.update_weight(0.6)
        md = _runtime_market_data(
            btc_returns_1h=0.05,
            btc_returns_24h=0.1,
        )
        sig = agent.to_agent_signals("BTC", market_data=md)
        # Weight in output should match agent weight (or 0 if gated)
        assert sig["model_alpha_weight"] in (0.6, 0.0)


# ============================================================================
# GROUP 4: Weight controllers (3 tests)
# ============================================================================

class TestWeightControllers:
    """Weight setters must never hard-fail."""

    def test_set_max_weight_clamps(self):
        agent = _make_agent()
        agent.set_max_weight(2.0)
        assert agent._max_weight == 1.0
        agent.set_max_weight(-0.5)
        assert agent._max_weight == 0.0

    def test_update_weight_respects_max(self):
        agent = _make_agent()
        agent.set_max_weight(0.5)
        agent.update_weight(0.8)
        assert agent._current_weight == 0.5

    def test_bad_input_does_not_crash(self):
        agent = _make_agent()
        agent.set_max_weight("invalid")  # type: ignore
        assert agent._max_weight == 1.0  # unchanged
        agent.update_weight(None)  # type: ignore
        assert agent._current_weight == 1.0  # unchanged


# ============================================================================
# GROUP 5: Torch-missing fallback (2 tests)
# ============================================================================

class TestTorchFallback:
    """Mock model works when torch model file is absent."""

    def test_mock_model_loaded(self):
        agent = _make_agent(model_path="nonexistent_path")
        assert agent._model_loaded is True  # falls back to mock

    def test_mock_predict_returns_values(self):
        agent = _make_agent()
        import numpy as np
        features = np.zeros(13, dtype=np.float32)
        pred, conf = agent._model.predict(features)
        assert isinstance(pred, float)
        assert isinstance(conf, float)
        assert -1.0 <= pred <= 1.0
        assert 0.0 <= conf <= 1.0


# ============================================================================
# GROUP 6: Singleton lifecycle (2 tests)
# ============================================================================

class TestSingleton:
    def test_singleton_returns_same(self):
        reset_model_alpha_agent()
        a1 = get_model_alpha_agent()
        a2 = get_model_alpha_agent()
        assert a1 is a2
        reset_model_alpha_agent()

    def test_reset_creates_fresh(self):
        reset_model_alpha_agent()
        a1 = get_model_alpha_agent()
        reset_model_alpha_agent()
        a2 = get_model_alpha_agent()
        assert a1 is not a2
        reset_model_alpha_agent()


# ============================================================================
# GROUP 7: allow_mock_model flag v6.4 (3 tests)
# ============================================================================

class TestAllowMockModel:
    """Current runtime prefers real local models when artifacts exist."""

    def test_default_prefers_real_model_when_available(self):
        """`_model_loaded` must track whether the artifacts are actually on disk.

        [P165] This asserted `_model_loaded is True` unconditionally, but
        `models/` is gitignored (`.gitignore:59`) — the DT checkpoints live on the
        `hmats-models` Docker volume, not in a checkout. So the test passed only
        on a machine that had trained locally, and failed everywhere else for a
        reason unrelated to the code.

        Tie the assertion to the premise instead of hardcoding one side of it: on
        the server (artifacts present) it still pins "real model gets loaded", and
        here it pins the fail-soft contract the warning at `model_alpha_agent.py`
        :462 promises — neutral signals, never a mock, when `allow_mock_model` is
        False.
        """
        agent = ModelAlphaAgent(model_path="nonexistent_path")
        artifacts_present = Path(agent.sequence_model_path).exists()
        assert agent._model_loaded is artifacts_present, (
            f"artifacts at {agent.sequence_model_path} "
            f"{'exist' if artifacts_present else 'are absent'} but "
            f"_model_loaded={agent._model_loaded}"
        )
        # Either way, with no state fed in the agent must emit a neutral signal —
        # a missing model must never become a nonzero direction.
        sig = agent.to_agent_signals("BTC")
        assert sig["model_alpha_direction"] == 0.0
        assert sig["model_alpha_confidence"] == 0.0

    def test_allow_mock_enables_model(self):
        agent = ModelAlphaAgent(allow_mock_model=True)
        assert agent._model_loaded is True

    def test_no_state_still_gates_even_when_model_loaded(self):
        agent = ModelAlphaAgent(model_path="nonexistent_path")
        sig = agent.to_agent_signals("BTC")
        assert "gated:no_state" in str(sig.get("model_alpha_reasons", []))


# ============================================================================
# GROUP 8: v6.4 metadata fields (3 tests)
# ============================================================================

class TestV64Metadata:
    """v6.4 standard metadata on agent outputs."""

    def test_neutral_signals_have_metadata(self):
        agent = _make_agent()
        sig = agent.to_agent_signals("BTC")
        # Even neutral signals must have v6.4 metadata
        assert "model_alpha_reasons" in sig
        assert isinstance(sig["model_alpha_reasons"], list)

    def test_gated_reason_in_reasons(self):
        agent = _make_agent()
        sig = agent.to_agent_signals("BTC")  # no market_data -> gated
        assert any("gated" in r for r in sig["model_alpha_reasons"])

    def test_data_quality_clamp(self):
        agent = _make_agent()
        md = _runtime_market_data()
        sig = agent.to_agent_signals("BTC", market_data=md)
        dq = sig.get("model_alpha_data_quality", 0.0)
        assert 0.0 <= dq <= 1.0
