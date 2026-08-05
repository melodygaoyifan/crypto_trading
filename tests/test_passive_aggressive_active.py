"""
tests/test_passive_aggressive_active.py - P2-03 PA Executor ACTIVE mode tests

Verifies:
1. SHADOW vs ACTIVE (default) mode switch via config
2. shadow_mode attribute exists and is correct
3. Price improvement clamping to [min, max] bounds
4. Toxic flow ABORT threshold (configurable)
5. Edge-insufficient gating (abort_on_insufficient_edge)
6. Config round-trip from JSON
7. No config -> ACTIVE; an unrecognised mode degrades to SHADOW (P165)
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from execution.passive_aggressive import (
    PAExecutorConfig,
    PassiveAggressiveExecutor,
    ExecutionDecision,
    ExecutionMode,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_executor(mode="SHADOW", **overrides) -> PassiveAggressiveExecutor:
    """Create PA executor with specified config."""
    defaults = dict(
        mode=mode,
        max_price_improve_bps=8.0,
        min_price_improve_bps=2.0,
        toxic_flow_abort_threshold=0.85,
        abort_on_insufficient_edge=True,
    )
    defaults.update(overrides)
    config = PAExecutorConfig(**defaults)
    return PassiveAggressiveExecutor(config=config)


def _decide(executor, vpin=0.35, urgency="NORMAL", signal=0.5,
            confidence=0.7, alpha_bps=200.0, price=50000.0, qty=0.1):
    """Call decide() with reasonable defaults."""
    return executor.decide(
        asset="BTC",
        signal=signal,
        confidence=confidence,
        vpin=vpin,
        urgency=urgency,
        predicted_alpha_bps=alpha_bps,
        current_price=price,
        target_qty=qty,
    )


# ===========================================================================
# Test Group 1: SHADOW vs ACTIVE mode switch
# ===========================================================================

class TestShadowVsActive:
    """shadow_mode attribute correctly reflects config."""

    def test_default_no_config_is_active(self):
        """No config -> ACTIVE, and any unknown mode falls back to SHADOW.

        [P165] Was `test_default_no_config_is_shadow`. `PAExecutorConfig.mode`
        defaults to "ACTIVE" (`passive_aggressive.py:472`) and `from_dict`
        likewise (`:481`) — the SHADOW default this asserted survived only in
        the class docstring, which has now been corrected. The property worth
        guarding is the fail-safe direction of the derivation: `shadow_mode =
        (mode != "ACTIVE")`, so a typo or a retired mode name degrades to
        SHADOW (decision logged, legacy path executes) rather than silently
        taking over the hot path.
        """
        executor = PassiveAggressiveExecutor()
        assert executor.shadow_mode is False
        assert PassiveAggressiveExecutor(
            PAExecutorConfig(mode="ACTIVE_TYPO")).shadow_mode is True

    def test_shadow_config_is_shadow(self):
        """Explicit SHADOW config -> shadow_mode=True."""
        executor = _make_executor(mode="SHADOW")
        assert executor.shadow_mode is True

    def test_active_config_is_active(self):
        """ACTIVE config -> shadow_mode=False."""
        executor = _make_executor(mode="ACTIVE")
        assert executor.shadow_mode is False

    def test_shadow_mode_attr_exists(self):
        """shadow_mode attribute always exists (no AttributeError)."""
        executor = PassiveAggressiveExecutor()
        assert hasattr(executor, 'shadow_mode')


# ===========================================================================
# Test Group 2: Price improvement clamping
# ===========================================================================

class TestPriceImprovementClamping:
    """price_improvement_bps clamped to [min, max] config bounds."""

    def test_normal_conditions_clamped_to_min(self):
        """Normal conditions: PA sets 0.5bps -> clamped to min=2.0bps."""
        executor = _make_executor(
            mode="ACTIVE",
            min_price_improve_bps=2.0,
            max_price_improve_bps=8.0,
        )
        result = _decide(executor, vpin=0.35, urgency="NORMAL")
        # Normal conditions set 0.5bps -> clamped up to 2.0
        assert result.price_improvement_bps == pytest.approx(2.0, abs=0.1)

    def test_elevated_vpin_within_bounds(self):
        """Elevated VPIN: PA sets 2.0bps -> within bounds, unchanged."""
        executor = _make_executor(
            mode="ACTIVE",
            min_price_improve_bps=2.0,
            max_price_improve_bps=8.0,
        )
        result = _decide(executor, vpin=0.55)
        assert 2.0 <= result.price_improvement_bps <= 8.0

    def test_max_clamp_applied(self):
        """Custom max clamp prevents excessive price improvement."""
        executor = _make_executor(
            mode="ACTIVE",
            min_price_improve_bps=1.0,
            max_price_improve_bps=3.0,
        )
        # High toxicity asset -> sets 2.0bps, but max=3.0 so OK
        result = _decide(executor, vpin=0.55)
        assert result.price_improvement_bps <= 3.0

    def test_aggressive_mode_no_clamp(self):
        """AGGRESSIVE mode has price_improvement_bps=0 -> no clamping."""
        executor = _make_executor(mode="ACTIVE")
        result = _decide(executor, urgency="CRITICAL")
        assert result.execution_mode == ExecutionMode.AGGRESSIVE
        # AGGRESSIVE: price_improvement_bps stays at 0 (not clamped)
        assert result.price_improvement_bps == 0.0


# ===========================================================================
# Test Group 3: Toxic flow ABORT
# ===========================================================================

class TestToxicFlowAbort:
    """Extreme VPIN triggers ABORT based on configurable threshold."""

    def test_extreme_vpin_triggers_abort(self):
        """VPIN > toxic_flow_abort_threshold -> ABORT."""
        executor = _make_executor(
            mode="ACTIVE",
            toxic_flow_abort_threshold=0.85,
        )
        result = _decide(executor, vpin=0.90, urgency="NORMAL")
        assert result.execution_mode == ExecutionMode.ABORT

    def test_vpin_below_threshold_no_abort(self):
        """VPIN below threshold -> no ABORT."""
        executor = _make_executor(
            mode="ACTIVE",
            toxic_flow_abort_threshold=0.85,
        )
        result = _decide(executor, vpin=0.80, urgency="NORMAL")
        assert result.execution_mode != ExecutionMode.ABORT

    def test_critical_urgency_overrides_abort(self):
        """CRITICAL urgency bypasses ABORT even with extreme VPIN."""
        executor = _make_executor(
            mode="ACTIVE",
            toxic_flow_abort_threshold=0.85,
        )
        result = _decide(executor, vpin=0.90, urgency="CRITICAL")
        assert result.execution_mode == ExecutionMode.AGGRESSIVE

    def test_custom_threshold_respected(self):
        """Custom abort threshold (0.75) triggers at lower VPIN."""
        executor = _make_executor(
            mode="ACTIVE",
            toxic_flow_abort_threshold=0.75,
        )
        result = _decide(executor, vpin=0.78, urgency="NORMAL")
        assert result.execution_mode == ExecutionMode.ABORT


# ===========================================================================
# Test Group 4: Edge-insufficient gating
# ===========================================================================

class TestEdgeGating:
    """abort_on_insufficient_edge controls edge check behavior."""

    def test_default_aborts_on_insufficient_edge(self):
        """Default: insufficient edge -> HOLD (early return)."""
        executor = _make_executor(
            mode="ACTIVE",
            abort_on_insufficient_edge=True,
        )
        # alpha=5bps, friction=36bps (26+10), need > 3×36 = 108 -> 5 < 108 -> HOLD
        result = _decide(executor, alpha_bps=5.0)
        assert result.action == "HOLD"
        assert "Insufficient edge" in result.reason

    def test_ultra_skips_edge_abort(self):
        """ULTRA: abort_on_insufficient_edge=False -> proceeds past edge check."""
        executor = _make_executor(
            mode="ACTIVE",
            abort_on_insufficient_edge=False,
        )
        result = _decide(executor, alpha_bps=5.0)
        # Should NOT be HOLD - it proceeds through mode selection
        assert result.action != "HOLD" or "Insufficient edge" not in result.reason

    def test_sufficient_edge_proceeds_regardless(self):
        """Sufficient edge always proceeds regardless of config."""
        for abort_flag in [True, False]:
            executor = _make_executor(
                mode="ACTIVE",
                abort_on_insufficient_edge=abort_flag,
            )
            result = _decide(executor, alpha_bps=200.0)
            # 200 > 108 -> sufficient, action is BUY (signal=0.5 > 0.2)
            assert result.action == "BUY"


# ===========================================================================
# Test Group 5: Config round-trip
# ===========================================================================

class TestConfigRoundTrip:
    """Config loads correctly from dict and JSON."""

    def test_from_dict_with_active_config(self):
        """from_dict loads ACTIVE mode config."""
        d = {
            "mode": "ACTIVE",
            "max_price_improve_bps": 6.0,
            "min_price_improve_bps": 1.5,
            "toxic_flow_abort_threshold": 0.80,
            "abort_on_insufficient_edge": False,
        }
        config = PAExecutorConfig.from_dict(d)
        assert config.mode == "ACTIVE"
        assert config.max_price_improve_bps == 6.0
        assert config.min_price_improve_bps == 1.5
        assert config.toxic_flow_abort_threshold == 0.80
        assert config.abort_on_insufficient_edge is False

    def test_from_dict_defaults_match_the_dataclass(self):
        """[P165] `from_dict({})` must agree with `PAExecutorConfig()` field-for-field.

        Was `test_from_dict_defaults_to_shadow`, pinning the retired "SHADOW"
        default. Rather than re-pin "ACTIVE" (which would rot the same way the
        next time the mode is promoted), assert the invariant that actually
        matters: `from_dict` restates every default as a literal in its
        `d.get(...)` calls, so it can silently disagree with the dataclass the
        moment one side is edited. Parity is the thing to guard.
        """
        import dataclasses
        assert dataclasses.asdict(PAExecutorConfig.from_dict({})) == \
            dataclasses.asdict(PAExecutorConfig())

    def test_ultra_aggressive_json_has_pa_config(self):
        """ultra_aggressive_5y.json contains passive_aggressive section."""
        json_path = Path("configs/ultra_aggressive_5y.json")
        if not json_path.exists():
            pytest.skip("ultra_aggressive_5y.json not found")
        with open(json_path) as f:
            data = json.load(f)
        pa = data.get("passive_aggressive")
        assert pa is not None
        assert pa["mode"] == "ACTIVE"
        assert pa["max_price_improve_bps"] == 8
        assert pa["min_price_improve_bps"] == 2

    def test_default_config_is_active(self):
        """[P165] Default is ACTIVE — see test_default_no_config_is_active."""
        config = PAExecutorConfig()
        assert config.mode == "ACTIVE"
        assert config.abort_on_insufficient_edge is True


# ===========================================================================
# Test Group 6: Mode selection logic
# ===========================================================================

class TestModeSelection:
    """PA decide() produces correct execution modes."""

    def test_critical_urgency_aggressive(self):
        """CRITICAL urgency -> AGGRESSIVE."""
        executor = _make_executor(mode="ACTIVE")
        result = _decide(executor, urgency="CRITICAL")
        assert result.execution_mode == ExecutionMode.AGGRESSIVE

    def test_high_urgency_hybrid(self):
        """HIGH urgency -> HYBRID."""
        executor = _make_executor(mode="ACTIVE")
        result = _decide(executor, urgency="HIGH", vpin=0.35)
        assert result.execution_mode == ExecutionMode.HYBRID

    def test_toxic_vpin_aggressive(self):
        """VPIN > 0.7 (but below abort thresh) -> AGGRESSIVE."""
        executor = _make_executor(mode="ACTIVE", toxic_flow_abort_threshold=0.85)
        result = _decide(executor, vpin=0.75, urgency="NORMAL")
        assert result.execution_mode == ExecutionMode.AGGRESSIVE

    def test_normal_conditions_passive(self):
        """Normal VPIN + NORMAL urgency -> PASSIVE."""
        executor = _make_executor(mode="ACTIVE")
        result = _decide(executor, vpin=0.35, urgency="NORMAL")
        assert result.execution_mode == ExecutionMode.PASSIVE

    def test_weak_signal_hold(self):
        """Signal too weak (0.1) -> HOLD action."""
        executor = _make_executor(mode="ACTIVE")
        result = _decide(executor, signal=0.1)
        assert result.action == "HOLD"
        assert "weak" in result.reason.lower()


# ===========================================================================
# Test Group 7: Backward compatibility
# ===========================================================================

class TestBackwardCompat:
    """No config / SHADOW mode preserves pre-P2-03 behavior."""

    def test_no_config_produces_valid_decision(self):
        """No config still produces valid ExecutionDecision."""
        executor = PassiveAggressiveExecutor()
        result = _decide(executor)
        assert isinstance(result, ExecutionDecision)
        assert result.asset == "BTC"

    def test_shadow_mode_same_as_no_config(self):
        """SHADOW config produces identical decisions to no config."""
        executor_none = PassiveAggressiveExecutor()
        executor_shadow = _make_executor(mode="SHADOW")

        # Both should produce same mode selection logic
        result_none = _decide(executor_none, vpin=0.35, urgency="NORMAL")
        result_shadow = _decide(executor_shadow, vpin=0.35, urgency="NORMAL")

        assert result_none.execution_mode == result_shadow.execution_mode
        assert result_none.action == result_shadow.action

    def test_fill_monitor_works_in_both_modes(self):
        """FillSlopeMonitor present in both modes."""
        for mode in ["SHADOW", "ACTIVE"]:
            executor = _make_executor(mode=mode)
            assert executor.fill_monitor is not None
