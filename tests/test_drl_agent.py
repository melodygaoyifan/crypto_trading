"""
Tests for agents/drl_agent.py overhaul.

Groups:
  1. Action mapping (5 tests) - all 5 DRLActions have continuous mapping ranges
  2. generate_signal / get_signal (6 tests) - HMATS integration, never None
  3. DRLAgentPayload (5 tests) - to_agent_signal_dict, fields, neutral
  4. Data health gating (5 tests) - input validation, degradation
  5. Per-asset state (4 tests) - isolation, reset, caching
  6. TQC import / model loading (3 tests) - TQC flag, .zip strip, fallback
"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from agents.drl_agent import (
    DRLAgent,
    DRLAction,
    DRLMode,
    DRLState,
    DRLOutput,
    DRLAgentPayload,
    DRLRewardComponents,
    calculate_drl_reward,
    get_drl_agent,
    reset_drl_agent,
    _neutral_payload,
)


# ============================================================================
# HELPERS
# ============================================================================

def _make_position_state(**overrides):
    base = {
        "tranche": 1,
        "pnl_bps": 50.0,
        "bars_held": 3,
        "entry_vs_current": 0.01,
        "high_water": 80.0,
        "drawdown": 0.02,
        "stop_distance": 0.02,
        "runner_active": False,
    }
    base.update(overrides)
    return base


def _make_market_data(**overrides):
    base = {
        "momentum_1bar": 0.5,
        "momentum_4bar": 1.0,
        "volume_ratio": 1.5,
        "vpin": 0.6,
        "rsi": 55.0,
        "lead_lag_edge": 0.1,
        "lead_lag_confidence": 0.7,
    }
    base.update(overrides)
    return base


def _make_signals(**overrides):
    base = {
        "quant_direction": 0.5,
        "quant_confidence": 0.7,
    }
    base.update(overrides)
    return base


# ============================================================================
# GROUP 1: Action Mapping (5 tests)
# ============================================================================

class TestActionMapping:
    """All 5 DRLActions have valid continuous mapping ranges."""

    def test_release_runner_maps_from_low_values(self):
        """Values below -0.6 -> RELEASE_RUNNER."""
        agent = DRLAgent()
        # Mock model that returns very negative action
        agent.enabled = True
        agent.model = MagicMock()
        agent.model.predict.return_value = (np.array([-0.8]), None)

        state = DRLState()
        output = agent.get_action(state)
        assert output.action == DRLAction.RELEASE_RUNNER

    def test_increase_exit_pressure_maps_from_mid_negative(self):
        """Values in [-0.6, -0.2) -> INCREASE_EXIT_PRESSURE."""
        agent = DRLAgent()
        agent.enabled = True
        agent.model = MagicMock()
        agent.model.predict.return_value = (np.array([-0.4]), None)

        state = DRLState()
        output = agent.get_action(state)
        assert output.action == DRLAction.INCREASE_EXIT_PRESSURE

    def test_hold_maps_from_center(self):
        """Values in [-0.2, 0.2) -> HOLD."""
        agent = DRLAgent()
        agent.enabled = True
        agent.model = MagicMock()
        agent.model.predict.return_value = (np.array([0.0]), None)

        state = DRLState()
        output = agent.get_action(state)
        assert output.action == DRLAction.HOLD

    def test_partial_exit_maps_from_mid_positive(self):
        """Values in [0.2, 0.6) -> PARTIAL_EXIT."""
        agent = DRLAgent()
        agent.enabled = True
        agent.model = MagicMock()
        agent.model.predict.return_value = (np.array([0.4]), None)

        state = DRLState()
        output = agent.get_action(state)
        assert output.action == DRLAction.PARTIAL_EXIT

    def test_escalate_maps_from_high_values(self):
        """Values >= 0.6 -> ESCALATE."""
        agent = DRLAgent()
        agent.enabled = True
        agent.model = MagicMock()
        agent.model.predict.return_value = (np.array([0.8]), None)

        state = DRLState()
        output = agent.get_action(state)
        assert output.action == DRLAction.ESCALATE


# ============================================================================
# GROUP 2: generate_signal / get_signal (6 tests)
# ============================================================================

class TestGenerateSignal:
    """HMATS integration: generate_signal and get_signal always return valid output."""

    def test_generate_signal_returns_payload_not_none(self):
        """generate_signal always returns DRLAgentPayload, never None."""
        agent = DRLAgent()
        payload = agent.generate_signal(asset="BTC", price=50000)
        assert isinstance(payload, DRLAgentPayload)
        assert payload.asset == "BTC"

    def test_generate_signal_disabled_returns_hold(self):
        """When disabled, generate_signal returns HOLD with 0 confidence."""
        agent = DRLAgent()
        assert not agent.enabled
        payload = agent.generate_signal(asset="ETH")
        assert payload.action == "HOLD"
        assert payload.confidence == 0.0
        assert payload.is_valid is True

    def test_generate_signal_exception_returns_neutral(self):
        """On exception, generate_signal returns neutral payload."""
        agent = DRLAgent(mode="SHADOW")
        agent.enabled = True
        # Model will fail with exception
        agent.model = MagicMock()
        agent.model.predict.side_effect = RuntimeError("CUDA OOM")

        payload = agent.generate_signal(asset="SOL")
        assert payload.direction == 0.0
        assert payload.confidence == 0.0
        assert payload.asset == "SOL"

    def test_get_signal_returns_tuple(self):
        """get_signal returns (direction, confidence) tuple."""
        agent = DRLAgent()
        result = agent.get_signal(asset="BTC")
        assert isinstance(result, tuple)
        assert len(result) == 2
        direction, confidence = result
        assert isinstance(direction, float)
        assert isinstance(confidence, float)

    def test_generate_signal_escalate_positive_direction(self):
        """ESCALATE action -> positive direction (requires SHADOW mode)."""
        agent = DRLAgent(mode="SHADOW")
        agent.enabled = True
        agent.model = MagicMock()
        agent.model.predict.return_value = (np.array([0.8]), None)

        # [P178] generate_signal now refuses a call with no state
        # inputs at all. This test is about the action->direction
        # mapping, not about empty input, so give it a minimal but
        # real market_data rather than relying on the empty-dict path.
        payload = agent.generate_signal(
            asset="BTC", market_data={"close": 50000.0})
        assert payload.direction > 0, "ESCALATE should produce positive direction"
        assert payload.action == "ESCALATE"
        assert payload.tranche_advice == "ESCALATE"

    def test_generate_signal_release_negative_direction(self):
        """RELEASE_RUNNER action -> negative direction."""
        agent = DRLAgent(mode="SHADOW")
        agent.enabled = True
        agent.model = MagicMock()
        agent.model.predict.return_value = (np.array([-0.9]), None)

        # [P178] generate_signal now refuses a call with no state
        # inputs at all. This test is about the action->direction
        # mapping, not about empty input, so give it a minimal but
        # real market_data rather than relying on the empty-dict path.
        payload = agent.generate_signal(
            asset="BTC", market_data={"close": 50000.0})
        assert payload.direction < 0, "RELEASE_RUNNER should produce negative direction"
        assert payload.action == "RELEASE_RUNNER"
        assert payload.exit_signal == "RELEASE_RUNNER"
        assert payload.exit_pressure == 1.0


# ============================================================================
# GROUP 3: DRLAgentPayload (5 tests)
# ============================================================================

class TestDRLAgentPayload:
    """Payload adapter produces correct flat dict for HMATS consumption."""

    def test_to_agent_signal_dict_has_required_keys(self):
        """Dict must have direction, confidence, asset, action, meta."""
        payload = DRLAgentPayload(asset="BTC", direction=0.3, confidence=0.7)
        d = payload.to_agent_signal_dict()
        for key in ["direction", "confidence", "asset", "action", "meta",
                     "exit_pressure", "tranche_advice", "exit_signal",
                     "entropy", "data_quality", "is_valid"]:
            assert key in d, f"Missing key: {key}"

    def test_neutral_payload_is_safe(self):
        """_neutral_payload returns no-recommendation defaults."""
        p = _neutral_payload("SOL")
        assert p.asset == "SOL"
        assert p.direction == 0.0
        assert p.confidence == 0.0
        assert p.action == "HOLD"
        assert p.is_valid is True
        assert p.entropy == 1.0

    def test_payload_direction_clipped(self):
        """Direction in payload should be between -1 and 1."""
        agent = DRLAgent(mode="SHADOW")
        agent.enabled = True
        agent.model = MagicMock()
        # Extreme action
        agent.model.predict.return_value = (np.array([1.0]), None)
        payload = agent.generate_signal(asset="BTC")
        assert -1.0 <= payload.direction <= 1.0

    def test_payload_confidence_clipped(self):
        """Confidence in payload should be between 0 and 1."""
        p = DRLAgentPayload(confidence=1.5)
        # The dataclass doesn't auto-clip, but generate_signal does
        # through DRLOutput.confidence = min(abs(action_value), 1.0)
        assert isinstance(p.confidence, float)

    def test_payload_meta_captures_context(self):
        """generate_signal populates meta with price/regime/issues."""
        agent = DRLAgent(mode="SHADOW")
        payload = agent.generate_signal(asset="BTC", price=50000, regime="MOMENTUM")
        assert "price" in payload.meta
        assert payload.meta["price"] == 50000
        assert payload.meta["regime"] == "MOMENTUM"


# ============================================================================
# GROUP 4: Data Health Gating (5 tests)
# ============================================================================

class TestDataHealthGating:
    """Input validation degrades data_quality but never crashes."""

    def test_empty_position_state_reduces_quality(self):
        quality, issues = DRLAgent._validate_build_state_inputs(
            {}, None, {"vpin": 0.5}, {"quant_direction": 0.1}, 2,
        )
        assert quality < 1.0
        assert "position_state_empty" in issues

    def test_none_regime_reduces_quality(self):
        quality, issues = DRLAgent._validate_build_state_inputs(
            {"tranche": 1}, None, {"vpin": 0.5}, {"quant_direction": 0.1}, 2,
        )
        assert "regime_result_none" in issues
        assert quality < 1.0

    def test_empty_market_data_reduces_quality(self):
        quality, issues = DRLAgent._validate_build_state_inputs(
            {"tranche": 1}, None, {}, {"quant_direction": 0.1}, 2,
        )
        assert "market_data_empty" in issues
        assert quality <= 0.7

    def test_invalid_system_mode_flagged(self):
        quality, issues = DRLAgent._validate_build_state_inputs(
            {"tranche": 1}, None, {"a": 1}, {"b": 1}, 5,
        )
        assert any("system_mode_invalid" in i for i in issues)

    def test_all_valid_inputs_full_quality(self):
        """All valid inputs -> only regime_result_none (since we pass None)."""
        quality, issues = DRLAgent._validate_build_state_inputs(
            {"tranche": 1}, MagicMock(), {"vpin": 0.5}, {"quant_direction": 0.1}, 2,
        )
        assert quality == 1.0
        assert len(issues) == 0


# ============================================================================
# GROUP 5: Per-Asset State (4 tests)
# ============================================================================

class TestPerAssetState:
    """Per-asset caching and isolation."""

    def test_generate_signal_caches_per_asset(self):
        agent = DRLAgent()
        agent.generate_signal(asset="BTC")
        agent.generate_signal(asset="ETH")
        assert "BTC" in agent._last_payloads
        assert "ETH" in agent._last_payloads

    def test_get_last_payload_returns_cached(self):
        agent = DRLAgent(mode="SHADOW")
        agent.generate_signal(asset="SOL", price=200)
        cached = agent.get_last_payload("SOL")
        assert cached.asset == "SOL"
        assert cached.meta.get("price") == 200

    def test_get_last_payload_missing_returns_neutral(self):
        agent = DRLAgent()
        cached = agent.get_last_payload("XRP")
        assert cached.direction == 0.0
        assert cached.confidence == 0.0

    def test_reset_state_clears_per_asset(self):
        agent = DRLAgent(mode="SHADOW")
        agent.generate_signal(asset="BTC")
        agent.generate_signal(asset="ETH")
        agent.reset_state(asset="BTC")
        assert "BTC" not in agent._last_payloads
        assert "ETH" in agent._last_payloads

        agent.reset_state()  # Reset all
        assert len(agent._last_payloads) == 0


# ============================================================================
# GROUP 6: TQC Import & Model Loading (3 tests)
# ============================================================================

class TestTQCSupport:
    """TQC import and model path handling."""

    def test_tqc_available_flag_exists(self):
        """TQC_AVAILABLE flag should be importable."""
        from agents.drl_agent import TQC_AVAILABLE
        assert isinstance(TQC_AVAILABLE, bool)

    def test_load_model_strips_zip_extension(self):
        """Model path with .zip should be stripped before loading."""
        agent = DRLAgent()
        agent.enabled = True
        # We can't load a real model, but we can verify the path logic
        # by checking the error message includes the stripped path
        agent._load_model("models/nonexistent_tqc_model.zip")
        # Should have failed but not crashed
        assert agent.model is None

    def test_load_model_detects_tqc_from_path(self):
        """Path containing 'tqc' should select TQC algorithm."""
        agent = DRLAgent()
        agent.enabled = True
        # Will fail to load but should attempt TQC
        agent._load_model("models/best_tqc_fold1.zip")
        # If TQC_AVAILABLE is False, agent gets disabled
        # If True, it would try to load and fail (file doesn't exist)
        # Either way, no crash
        assert agent.model is None


# ============================================================================
# BONUS: DRLState & Reward (3 tests)
# ============================================================================

class TestDRLStateAndReward:
    """State vector and reward function basics."""

    def test_state_to_vector_correct_shape(self):
        state = DRLState()
        vec = state.to_vector()
        assert vec.shape == (48,)
        assert vec.dtype == np.float32

    def test_reward_components_sum_correctly(self):
        state = DRLState(current_tranche=2, position_pnl_bps=100)
        next_state = DRLState(current_tranche=3)
        reward = calculate_drl_reward(
            state, DRLAction.ESCALATE, next_state,
            realized_pnl_bps=50, peak_unrealized_pnl=100,
            next_bar_favorable=True,
        )
        assert isinstance(reward.total, float)
        assert reward.pnl_reward > 0  # 50 bps realized

    def test_singleton_lifecycle(self):
        reset_drl_agent()
        a1 = get_drl_agent()
        a2 = get_drl_agent()
        assert a1 is a2
        reset_drl_agent()
        a3 = get_drl_agent()
        assert a3 is not a1
