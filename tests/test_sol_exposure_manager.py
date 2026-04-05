"""
Tests for agents/sol_exposure_manager.py - production overhaul.

Groups:
  1. SOL_DOMINANT priority gross scaling (D)
  2. NO_TRADE "no increase" rule (C)
  3. Mode hysteresis (B)
  4. Correlation collapse (existing + configurable)
  5. Fail-closed on bad inputs (H)
"""

import math
import time
import pytest

from agents.sol_exposure_manager import (
    SOLExposureManager,
    SOLExposureConfig,
    ExposureMode,
    PortfolioExposureState,
    _MODE_RISK_ORDER,
)


# ---------------------------------------------------------------------------
# Stubs - lightweight, no heavy imports
# ---------------------------------------------------------------------------

class StubCrackWindow:
    def __init__(self, valid=False, direction="long", remaining_strength=0.8):
        self._valid = valid
        self.direction = direction
        self.remaining_strength = remaining_strength

    def is_valid(self):
        return self._valid


class StubConfidence:
    def __init__(self, value=0.7):
        self._value = value

    def overall(self):
        return self._value


class StubRegimeState:
    """Minimal stub for EnhancedRegimeState."""

    def __init__(
        self,
        no_trade=False,
        opportunity=False,
        beta_sol_btc=1.0,
        volatility_regime="normal",
        correlation_btc_eth_sol=0.5,
        crack_window=None,
        confidence=None,
    ):
        self._no_trade = no_trade
        self._opportunity = opportunity
        self.beta_sol_btc = beta_sol_btc
        self.volatility_regime = volatility_regime
        self.correlation_btc_eth_sol = correlation_btc_eth_sol
        self.crack_window = crack_window or StubCrackWindow()
        self.confidence = confidence or StubConfidence()

    def is_no_trade(self):
        return self._no_trade

    def is_opportunity_active(self):
        return self._opportunity


class StubIntent:
    """Minimal stub for TradeIntent."""

    def __init__(self, target_exposure=0.0):
        self.target_exposure = target_exposure


def _sol_dominant_regime(flow=0.7):
    """Build a regime state that qualifies for SOL_DOMINANT."""
    return StubRegimeState(
        opportunity=True,
        beta_sol_btc=1.5,
        crack_window=StubCrackWindow(valid=True, direction="long"),
        confidence=StubConfidence(0.8),
    )


# ===================================================================
# GROUP 1: SOL_DOMINANT priority gross scaling
# ===================================================================

class TestSOLDominantGrossScaling:
    """In SOL_DOMINANT, SOL should keep full allocation and BTC/ETH get scaled."""

    def test_sol_preserved_btc_eth_scaled(self):
        """SOL=1.0 should stay ~1.0; BTC/ETH capped to fit total_gross=1.3."""
        cfg = SOLExposureConfig(
            enable_mode_hysteresis=False,  # skip hysteresis for direct testing
            prefer_sol_in_gross_scaling=True,
        )
        mgr = SOLExposureManager(cfg)
        regime = _sol_dominant_regime()

        targets = mgr.calculate_target_exposures(
            btc_intent=StubIntent(0.8),
            eth_intent=StubIntent(0.8),
            sol_intent=StubIntent(1.0),
            regime_state=regime,
            flow_alignment=0.7,
        )

        # SOL should be at or near 1.0 (may be capped by SOL_DOMINANT limits)
        assert targets["SOL"] >= 0.7, f"SOL too low: {targets['SOL']}"
        # BTC and ETH should be scaled down
        gross = abs(targets["BTC"]) + abs(targets["ETH"]) + abs(targets["SOL"])
        assert gross <= 1.3 + 1e-9, f"Gross too high: {gross}"

    def test_sol_dominant_btc_eth_capped_at_0_2(self):
        """When SOL>0.7 in SOL_DOMINANT, BTC/ETH get hard-capped at 0.2."""
        cfg = SOLExposureConfig(enable_mode_hysteresis=False)
        mgr = SOLExposureManager(cfg)
        regime = _sol_dominant_regime()

        targets = mgr.calculate_target_exposures(
            btc_intent=StubIntent(0.5),
            eth_intent=StubIntent(0.5),
            sol_intent=StubIntent(1.0),
            regime_state=regime,
            flow_alignment=0.7,
        )

        assert abs(targets["BTC"]) <= 0.2 + 1e-9
        assert abs(targets["ETH"]) <= 0.2 + 1e-9

    def test_proportional_scaling_in_normal_mode(self):
        """In NORMAL mode, gross scaling is proportional (all assets scaled equally)."""
        cfg = SOLExposureConfig(enable_mode_hysteresis=False)
        mgr = SOLExposureManager(cfg)
        regime = StubRegimeState()  # NORMAL

        targets = mgr.calculate_target_exposures(
            btc_intent=StubIntent(0.8),
            eth_intent=StubIntent(0.8),
            sol_intent=StubIntent(0.8),
            regime_state=regime,
            flow_alignment=0.0,
        )

        gross = abs(targets["BTC"]) + abs(targets["ETH"]) + abs(targets["SOL"])
        assert gross <= 1.5 + 1e-9
        # All should be scaled by the same factor
        if targets["BTC"] != 0:
            ratio = targets["ETH"] / targets["BTC"]
            assert abs(ratio - 1.0) < 0.01, "Scaling should be proportional"

    def test_decision_records_sol_priority(self):
        """get_last_decision() should record gross_scaled_sol_priority."""
        cfg = SOLExposureConfig(enable_mode_hysteresis=False)
        mgr = SOLExposureManager(cfg)
        regime = _sol_dominant_regime()

        mgr.calculate_target_exposures(
            btc_intent=StubIntent(0.8),
            eth_intent=StubIntent(0.8),
            sol_intent=StubIntent(1.0),
            regime_state=regime,
            flow_alignment=0.7,
        )

        decision = mgr.get_last_decision()
        assert "gross_scaled_sol_priority" in decision["reasons"]


# ===================================================================
# GROUP 2: NO_TRADE "no increase" rule
# ===================================================================

class TestNoTradeNoIncrease:
    """NO_TRADE should not allow absolute exposure to increase."""

    def test_no_trade_blocks_increase(self):
        """Intents that increase abs exposure should be blocked."""
        cfg = SOLExposureConfig(enable_mode_hysteresis=False)
        mgr = SOLExposureManager(cfg)

        # Set current state: moderate exposure
        mgr.state.btc_exposure = 0.2
        mgr.state.eth_exposure = 0.2
        mgr.state.sol_exposure = 0.5

        regime = StubRegimeState(no_trade=True)

        targets = mgr.calculate_target_exposures(
            btc_intent=StubIntent(1.0),
            eth_intent=StubIntent(0.8),
            sol_intent=StubIntent(1.0),
            regime_state=regime,
            flow_alignment=0.0,
        )

        # Abs exposure should NOT increase
        assert abs(targets["BTC"]) <= abs(0.2) + 1e-9
        assert abs(targets["ETH"]) <= abs(0.2) + 1e-9
        assert abs(targets["SOL"]) <= abs(0.5) + 1e-9

    def test_no_trade_allows_reduction(self):
        """Intents that reduce abs exposure should be allowed."""
        cfg = SOLExposureConfig(enable_mode_hysteresis=False)
        mgr = SOLExposureManager(cfg)

        mgr.state.btc_exposure = 0.5
        mgr.state.eth_exposure = 0.5
        mgr.state.sol_exposure = 0.8

        regime = StubRegimeState(no_trade=True)

        targets = mgr.calculate_target_exposures(
            btc_intent=StubIntent(0.1),
            eth_intent=StubIntent(0.1),
            sol_intent=StubIntent(0.2),
            regime_state=regime,
            flow_alignment=0.0,
        )

        # Reduction allowed
        assert abs(targets["BTC"]) <= 0.1 + 1e-9
        assert abs(targets["ETH"]) <= 0.1 + 1e-9
        assert abs(targets["SOL"]) <= 0.2 + 1e-9

    def test_no_trade_zero_state_blocks_new_positions(self):
        """If current state is all-zero, NO_TRADE should prevent new positions."""
        cfg = SOLExposureConfig(enable_mode_hysteresis=False)
        mgr = SOLExposureManager(cfg)
        # State is default zeros

        regime = StubRegimeState(no_trade=True)

        targets = mgr.calculate_target_exposures(
            btc_intent=StubIntent(0.5),
            eth_intent=StubIntent(0.3),
            sol_intent=StubIntent(0.8),
            regime_state=regime,
            flow_alignment=0.0,
        )

        assert targets["BTC"] == 0.0
        assert targets["ETH"] == 0.0
        assert targets["SOL"] == 0.0

    def test_no_trade_recorded_in_decision(self):
        """Decision metadata should record no_trade_capped."""
        cfg = SOLExposureConfig(enable_mode_hysteresis=False)
        mgr = SOLExposureManager(cfg)
        mgr.state.sol_exposure = 0.3

        regime = StubRegimeState(no_trade=True)

        mgr.calculate_target_exposures(
            btc_intent=StubIntent(0.0),
            eth_intent=StubIntent(0.0),
            sol_intent=StubIntent(0.8),  # tries to increase from 0.3
            regime_state=regime,
            flow_alignment=0.0,
        )

        decision = mgr.get_last_decision()
        assert decision["no_trade_capped"] is True
        assert "no_trade_no_increase" in decision["reasons"]

    def test_no_trade_disabled_when_flag_off(self):
        """force_no_trade_no_increase=False should revert to old behavior."""
        cfg = SOLExposureConfig(
            enable_mode_hysteresis=False,
            force_no_trade_no_increase=False,
        )
        mgr = SOLExposureManager(cfg)
        # State is zero

        regime = StubRegimeState(no_trade=True)

        targets = mgr.calculate_target_exposures(
            btc_intent=StubIntent(0.3),
            eth_intent=StubIntent(0.3),
            sol_intent=StubIntent(0.3),
            regime_state=regime,
            flow_alignment=0.0,
        )

        # With flag off, NO_TRADE just gives CONSERVATIVE limits (0.5 each)
        # Intents of 0.3 should pass through
        assert targets["SOL"] > 0.0


# ===================================================================
# GROUP 3: Mode hysteresis
# ===================================================================

class TestModeHysteresis:
    """Fast de-risk, slow upgrade with dwell time and confirmations."""

    def test_immediate_de_risk(self):
        """Switching to lower risk should happen immediately."""
        cfg = SOLExposureConfig(
            enable_mode_hysteresis=True,
            min_mode_hold_seconds=300,
        )
        mgr = SOLExposureManager(cfg)

        # Start in AGGRESSIVE mode
        mgr.state.exposure_mode = ExposureMode.AGGRESSIVE
        mgr._last_mode_change_mono = time.monotonic()

        # Request CONSERVATIVE (de-risk)
        result = mgr._apply_hysteresis(ExposureMode.CONSERVATIVE)
        assert result == ExposureMode.CONSERVATIVE

    def test_upgrade_blocked_before_hold_time(self):
        """Upgrade should be blocked if min_mode_hold_seconds not elapsed."""
        cfg = SOLExposureConfig(
            enable_mode_hysteresis=True,
            min_mode_hold_seconds=300,
        )
        mgr = SOLExposureManager(cfg)

        # Start in NORMAL, just changed
        mgr.state.exposure_mode = ExposureMode.NORMAL
        mgr._last_mode_change_mono = time.monotonic()

        # Request AGGRESSIVE (upgrade)
        result = mgr._apply_hysteresis(ExposureMode.AGGRESSIVE)
        assert result == ExposureMode.NORMAL  # blocked

    def test_upgrade_allowed_after_hold_time(self):
        """Upgrade should succeed after hold time elapses."""
        cfg = SOLExposureConfig(
            enable_mode_hysteresis=True,
            min_mode_hold_seconds=1,  # 1 second for test speed
        )
        mgr = SOLExposureManager(cfg)

        mgr.state.exposure_mode = ExposureMode.NORMAL
        mgr._last_mode_change_mono = time.monotonic() - 2  # 2 seconds ago

        result = mgr._apply_hysteresis(ExposureMode.AGGRESSIVE)
        assert result == ExposureMode.AGGRESSIVE

    def test_sol_dominant_requires_confirmations(self):
        """SOL_DOMINANT upgrade needs N consecutive requests."""
        cfg = SOLExposureConfig(
            enable_mode_hysteresis=True,
            min_mode_hold_seconds=0,  # no hold time for this test
            sol_dominant_confirmations_required=3,
        )
        mgr = SOLExposureManager(cfg)

        mgr.state.exposure_mode = ExposureMode.AGGRESSIVE
        mgr._last_mode_change_mono = time.monotonic() - 1000

        # Request 1: should hold
        result = mgr._apply_hysteresis(ExposureMode.SOL_DOMINANT)
        assert result == ExposureMode.AGGRESSIVE
        assert mgr._sol_dominant_consecutive == 1

        # Request 2: still hold
        result = mgr._apply_hysteresis(ExposureMode.SOL_DOMINANT)
        assert result == ExposureMode.AGGRESSIVE
        assert mgr._sol_dominant_consecutive == 2

        # Request 3: approved
        result = mgr._apply_hysteresis(ExposureMode.SOL_DOMINANT)
        assert result == ExposureMode.SOL_DOMINANT

    def test_confirmation_counter_resets(self):
        """If a non-SOL_DOMINANT request comes in, counter resets."""
        cfg = SOLExposureConfig(
            enable_mode_hysteresis=True,
            min_mode_hold_seconds=0,
            sol_dominant_confirmations_required=3,
        )
        mgr = SOLExposureManager(cfg)
        mgr.state.exposure_mode = ExposureMode.AGGRESSIVE
        mgr._last_mode_change_mono = time.monotonic() - 1000

        mgr._apply_hysteresis(ExposureMode.SOL_DOMINANT)
        assert mgr._sol_dominant_consecutive == 1

        # Interrupted by AGGRESSIVE request
        mgr._apply_hysteresis(ExposureMode.AGGRESSIVE)
        assert mgr._sol_dominant_consecutive == 0

    def test_hysteresis_disabled(self):
        """With enable_mode_hysteresis=False, upgrades are immediate."""
        cfg = SOLExposureConfig(enable_mode_hysteresis=False)
        mgr = SOLExposureManager(cfg)

        mgr.state.exposure_mode = ExposureMode.NORMAL
        mgr._last_mode_change_mono = time.monotonic()  # just now

        result = mgr._apply_hysteresis(ExposureMode.AGGRESSIVE)
        assert result == ExposureMode.AGGRESSIVE

    def test_same_mode_no_change(self):
        """Requesting the same mode should return it unchanged."""
        cfg = SOLExposureConfig(enable_mode_hysteresis=True)
        mgr = SOLExposureManager(cfg)
        mgr.state.exposure_mode = ExposureMode.NORMAL

        result = mgr._apply_hysteresis(ExposureMode.NORMAL)
        assert result == ExposureMode.NORMAL


# ===================================================================
# GROUP 4: Correlation collapse
# ===================================================================

class TestCorrelationCollapse:
    """Correlation collapse reduction triggers correctly."""

    def test_high_corr_same_direction_reduces(self):
        """High correlation + all same direction -> exposure reduced."""
        cfg = SOLExposureConfig(
            enable_mode_hysteresis=False,
            correlation_collapse_threshold=0.85,
            correlation_collapse_floor=0.60,
        )
        mgr = SOLExposureManager(cfg)
        regime = StubRegimeState(correlation_btc_eth_sol=0.92)

        targets = mgr.calculate_target_exposures(
            btc_intent=StubIntent(0.5),
            eth_intent=StubIntent(0.5),
            sol_intent=StubIntent(0.5),
            regime_state=regime,
            flow_alignment=0.0,
        )

        # All 0.5 = same direction + high corr -> should be reduced
        assert targets["BTC"] < 0.5
        assert targets["ETH"] < 0.5
        assert targets["SOL"] < 0.5

        decision = mgr.get_last_decision()
        assert "correlation_collapse_reduction" in decision["reasons"]
        assert decision["scaling"]["correlation_reduction_factor"] < 1.0

    def test_low_corr_no_reduction(self):
        """Low correlation should NOT trigger reduction."""
        cfg = SOLExposureConfig(enable_mode_hysteresis=False)
        mgr = SOLExposureManager(cfg)
        regime = StubRegimeState(correlation_btc_eth_sol=0.5)

        targets = mgr.calculate_target_exposures(
            btc_intent=StubIntent(0.5),
            eth_intent=StubIntent(0.5),
            sol_intent=StubIntent(0.5),
            regime_state=regime,
            flow_alignment=0.0,
        )

        # No correlation reduction - check decision
        decision = mgr.get_last_decision()
        assert decision["scaling"]["correlation_reduction_factor"] == 1.0

    def test_mixed_direction_no_reduction(self):
        """Mixed long/short should NOT trigger reduction even with high corr."""
        cfg = SOLExposureConfig(enable_mode_hysteresis=False)
        mgr = SOLExposureManager(cfg)
        regime = StubRegimeState(correlation_btc_eth_sol=0.95)

        targets = mgr.calculate_target_exposures(
            btc_intent=StubIntent(0.5),
            eth_intent=StubIntent(-0.5),
            sol_intent=StubIntent(0.5),
            regime_state=regime,
            flow_alignment=0.0,
        )

        decision = mgr.get_last_decision()
        assert decision["scaling"]["correlation_reduction_factor"] == 1.0

    def test_reduction_floor_respected(self):
        """Reduction should never go below correlation_collapse_floor."""
        cfg = SOLExposureConfig(
            enable_mode_hysteresis=False,
            correlation_collapse_threshold=0.85,
            correlation_collapse_floor=0.60,
        )
        mgr = SOLExposureManager(cfg)
        # Extreme correlation = 1.0
        regime = StubRegimeState(correlation_btc_eth_sol=1.0)

        targets = mgr.calculate_target_exposures(
            btc_intent=StubIntent(0.5),
            eth_intent=StubIntent(0.5),
            sol_intent=StubIntent(0.5),
            regime_state=regime,
            flow_alignment=0.0,
        )

        decision = mgr.get_last_decision()
        assert decision["scaling"]["correlation_reduction_factor"] >= 0.60


# ===================================================================
# GROUP 5: Fail-closed on bad inputs
# ===================================================================

class TestFailClosed:
    """Bad inputs should degrade gracefully, not crash."""

    def test_none_regime_state(self):
        """regime_state=None should return safe dict with BTC/ETH/SOL."""
        cfg = SOLExposureConfig(enable_mode_hysteresis=False)
        mgr = SOLExposureManager(cfg)

        targets = mgr.calculate_target_exposures(
            btc_intent=StubIntent(0.5),
            eth_intent=StubIntent(0.5),
            sol_intent=StubIntent(0.5),
            regime_state=None,
            flow_alignment=0.0,
        )

        assert set(targets.keys()) == {"BTC", "ETH", "SOL"}
        for v in targets.values():
            assert isinstance(v, float)
            assert not math.isnan(v)

        decision = mgr.get_last_decision()
        assert "regime_state_is_none" in decision["reasons"]

    def test_missing_attributes_on_regime(self):
        """Regime with missing attributes should fallback to NORMAL."""

        class BadRegime:
            pass  # no methods at all

        cfg = SOLExposureConfig(enable_mode_hysteresis=False)
        mgr = SOLExposureManager(cfg)

        targets = mgr.calculate_target_exposures(
            btc_intent=StubIntent(0.3),
            eth_intent=StubIntent(0.3),
            sol_intent=StubIntent(0.3),
            regime_state=BadRegime(),
            flow_alignment=0.0,
        )

        assert set(targets.keys()) == {"BTC", "ETH", "SOL"}
        decision = mgr.get_last_decision()
        assert "regime_invalid_fallback_normal" in decision["reasons"]

    def test_nan_correlation_skips_collapse_check(self):
        """NaN correlation should skip collapse check, not crash."""
        cfg = SOLExposureConfig(enable_mode_hysteresis=False)
        mgr = SOLExposureManager(cfg)
        regime = StubRegimeState(correlation_btc_eth_sol=float("nan"))

        targets = mgr.calculate_target_exposures(
            btc_intent=StubIntent(0.5),
            eth_intent=StubIntent(0.5),
            sol_intent=StubIntent(0.5),
            regime_state=regime,
            flow_alignment=0.0,
        )

        assert set(targets.keys()) == {"BTC", "ETH", "SOL"}
        decision = mgr.get_last_decision()
        assert "correlation_is_nan" in decision["reasons"]
        assert "correlation_invalid_skip_check" in decision["reasons"]

    def test_flow_alignment_clamped(self):
        """flow_alignment > 1.0 should be clamped, not crash."""
        cfg = SOLExposureConfig(enable_mode_hysteresis=False)
        mgr = SOLExposureManager(cfg)
        regime = StubRegimeState()

        targets = mgr.calculate_target_exposures(
            btc_intent=StubIntent(0.3),
            eth_intent=StubIntent(0.3),
            sol_intent=StubIntent(0.3),
            regime_state=regime,
            flow_alignment=5.0,
        )

        assert set(targets.keys()) == {"BTC", "ETH", "SOL"}
        decision = mgr.get_last_decision()
        assert "flow_alignment_clamped" in decision["reasons"]

    def test_none_intents_return_zeros(self):
        """All None intents should produce zero exposures."""
        cfg = SOLExposureConfig(enable_mode_hysteresis=False)
        mgr = SOLExposureManager(cfg)
        regime = StubRegimeState()

        targets = mgr.calculate_target_exposures(
            btc_intent=None,
            eth_intent=None,
            sol_intent=None,
            regime_state=regime,
            flow_alignment=0.0,
        )

        assert targets["BTC"] == 0.0
        assert targets["ETH"] == 0.0
        assert targets["SOL"] == 0.0

    def test_nan_flow_alignment(self):
        """NaN flow alignment should be sanitized to 0.0."""
        cfg = SOLExposureConfig(enable_mode_hysteresis=False)
        mgr = SOLExposureManager(cfg)
        regime = StubRegimeState()

        targets = mgr.calculate_target_exposures(
            btc_intent=StubIntent(0.3),
            eth_intent=StubIntent(0.3),
            sol_intent=StubIntent(0.3),
            regime_state=regime,
            flow_alignment=float("nan"),
        )

        assert set(targets.keys()) == {"BTC", "ETH", "SOL"}
        decision = mgr.get_last_decision()
        assert "flow_alignment_nan" in decision["reasons"]

    def test_intent_with_bad_attribute(self):
        """Intent missing target_exposure should degrade to 0.0."""

        class BadIntent:
            pass

        cfg = SOLExposureConfig(enable_mode_hysteresis=False)
        mgr = SOLExposureManager(cfg)
        regime = StubRegimeState()

        targets = mgr.calculate_target_exposures(
            btc_intent=BadIntent(),
            eth_intent=None,
            sol_intent=StubIntent(0.5),
            regime_state=regime,
            flow_alignment=0.0,
        )

        assert targets["BTC"] == 0.0  # bad intent -> 0.0
        assert targets["SOL"] > 0.0   # good intent -> passes

    def test_get_last_decision_returns_dict(self):
        """get_last_decision() should always return a dict."""
        mgr = SOLExposureManager()
        # Before any calculation
        decision = mgr.get_last_decision()
        assert isinstance(decision, dict)

    def test_backward_compat_state_to_dict(self):
        """state.to_dict() should still work."""
        mgr = SOLExposureManager()
        d = mgr.get_state().to_dict()
        assert "btc_exposure" in d
        assert "exposure_mode" in d


# ===================================================================
# Bonus: Baseline allocation (E)
# ===================================================================

class TestBaselineAllocation:
    """Baseline allocation opt-in behavior."""

    def test_baseline_off_by_default(self):
        """Default config should NOT apply baseline."""
        mgr = SOLExposureManager()
        regime = StubRegimeState()

        targets = mgr.calculate_target_exposures(
            btc_intent=None,
            eth_intent=None,
            sol_intent=None,
            regime_state=regime,
            flow_alignment=0.0,
        )

        assert targets["BTC"] == 0.0
        assert targets["ETH"] == 0.0
        assert targets["SOL"] == 0.0

    def test_baseline_applied_when_enabled(self):
        """With use_baseline_when_no_intent=True, zero intents get baseline."""
        cfg = SOLExposureConfig(
            enable_mode_hysteresis=False,
            use_baseline_when_no_intent=True,
        )
        mgr = SOLExposureManager(cfg)
        regime = StubRegimeState()  # NORMAL mode

        targets = mgr.calculate_target_exposures(
            btc_intent=None,
            eth_intent=None,
            sol_intent=None,
            regime_state=regime,
            flow_alignment=0.0,
        )

        assert targets["BTC"] > 0.0
        assert targets["SOL"] > 0.0
        decision = mgr.get_last_decision()
        assert "baseline_applied" in decision["reasons"]

    def test_baseline_not_applied_in_conservative(self):
        """Baseline should NOT apply in CONSERVATIVE mode (fail-closed)."""
        cfg = SOLExposureConfig(
            enable_mode_hysteresis=False,
            use_baseline_when_no_intent=True,
        )
        mgr = SOLExposureManager(cfg)
        regime = StubRegimeState(no_trade=True)  # -> CONSERVATIVE

        targets = mgr.calculate_target_exposures(
            btc_intent=None,
            eth_intent=None,
            sol_intent=None,
            regime_state=regime,
            flow_alignment=0.0,
        )

        assert targets["BTC"] == 0.0
        assert targets["ETH"] == 0.0
        assert targets["SOL"] == 0.0