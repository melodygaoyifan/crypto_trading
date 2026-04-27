"""
test_property_invariants.py — Hypothesis property tests for hot paths (P117)
================================================================================

Auto-generates edge cases for the 4 most safety-critical pure functions and
asserts STRUCTURAL invariants that must hold for ANY input. Catches the bug
class that fixed-example tests miss: the "I never thought to test this input"
class.

Hot paths covered:
  1. _classify_kraken_order_error  — error string -> (category, guidance)
  2. _compute_effective_weekend_confidence  — DRL substitution rule
  3. RiskVetoClassifier.classify  — multi-input HARD/SOFT/PASS classifier
  4. AuthorityFusionEngine.fuse  — agent_signals -> FusionResult

Invariants tested per function are documented inline. The point is NOT to
test what the functions DO (other tests cover that) but to assert that they
NEVER VIOLATE structural rules — e.g. classifier never returns an unknown
category, fusion never produces direction outside [-1, 1], etc.

Hypothesis runs ~100 randomly generated cases per @given test by default;
shrinking finds the minimal failing input.
"""
from __future__ import annotations

import math

import pytest
from hypothesis import given, settings, strategies as st, assume


# =====================================================================
# 1. _classify_kraken_order_error
# =====================================================================

class TestClassifyKrakenError:
    """Invariants:
      - Always returns a 2-tuple (category, guidance)
      - category is always one of {"PERMANENT", "TRANSIENT", "UNKNOWN"}
      - guidance is always a string (never None)
      - PREFLIGHT_* prefix => PERMANENT (P93)
      - Empty / None input => doesn't crash
    """

    @given(st.text(min_size=0, max_size=500))
    @settings(max_examples=200)
    def test_always_returns_valid_tuple(self, error_str):
        from execution.execution_manager import ExecutionManager
        cat, guidance = ExecutionManager._classify_kraken_order_error(error_str)
        assert cat in {"PERMANENT", "TRANSIENT", "UNKNOWN"}, (
            f"Unknown category {cat!r} for input {error_str!r}"
        )
        assert isinstance(guidance, str), (
            f"guidance must be str, got {type(guidance).__name__} for {error_str!r}"
        )

    @given(st.sampled_from([
        "PREFLIGHT_BELOW_MIN_SIZE", "PREFLIGHT_WRONG_SIDE",
        "INSUFFICIENT_SPOT_BALANCE",
    ]), st.text(min_size=0, max_size=200))
    def test_preflight_prefix_always_permanent(self, prefix, suffix):
        """P93 invariant: our own pre-flight rejections are PERMANENT
        regardless of what suffix appears after the prefix."""
        from execution.execution_manager import ExecutionManager
        err = f"{prefix}: {suffix}"
        cat, _ = ExecutionManager._classify_kraken_order_error(err)
        assert cat == "PERMANENT", (
            f"P93 violation: {err!r} classified as {cat}, expected PERMANENT"
        )

    @given(st.sampled_from([
        "EAPI:Invalid key",
        "EAPI:Invalid signature",
        "EAPI:Invalid permissions",
        "EGeneral:Permission denied",
    ]))
    def test_auth_errors_permanent(self, err):
        """Auth/permission errors must be PERMANENT — retry can't fix."""
        from execution.execution_manager import ExecutionManager
        cat, _ = ExecutionManager._classify_kraken_order_error(err)
        assert cat == "PERMANENT"

    def test_empty_input_safe(self):
        from execution.execution_manager import ExecutionManager
        cat, g = ExecutionManager._classify_kraken_order_error("")
        assert cat in {"PERMANENT", "TRANSIENT", "UNKNOWN"}
        assert isinstance(g, str)

    def test_none_input_safe(self):
        """None should not crash — code uses `s = error_str or ""`."""
        from execution.execution_manager import ExecutionManager
        cat, g = ExecutionManager._classify_kraken_order_error(None)
        assert cat in {"PERMANENT", "TRANSIENT", "UNKNOWN"}


# =====================================================================
# 2. _compute_effective_weekend_confidence (P46)
# =====================================================================

class _MockIntent:
    """Minimal duck-typed intent for the confidence helper."""
    def __init__(self, quant_confidence: float):
        self.quant_confidence = quant_confidence


class TestEffectiveWeekendConfidence:
    """Invariants per P46:
      - Output is always >= quant_confidence (substitution only HELPS, never hurts)
      - Output is always in [0, 1]
      - When DRL not ACTIVE, output == quant_confidence
      - When DRL conf < quant conf, output == quant_confidence
      - When |drl_dir| < 0.5, output == quant_confidence
    """

    def _runner(self):
        """Build a minimal runner-like object to call the helper.
        The function is a method but uses no `self` state — bypass __init__.
        """
        from main import HMATSProductionRunner
        return HMATSProductionRunner.__new__(HMATSProductionRunner)

    @given(
        quant_conf=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        drl_dir=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
        drl_conf=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        drl_auth=st.sampled_from(["ACTIVE", "SHADOW", "DISABLED", "EXIT_ONLY"]),
    )
    @settings(max_examples=200, deadline=None)
    def test_output_never_less_than_quant_conf(
        self, quant_conf, drl_dir, drl_conf, drl_auth,
    ):
        runner = self._runner()
        intent = _MockIntent(quant_conf)
        signals = {
            "drl_authority_level": drl_auth,
            "drl_direction": drl_dir,
            "drl_confidence": drl_conf,
        }
        result = runner._compute_effective_weekend_confidence(intent, signals, "BTC")
        assert result >= quant_conf - 1e-9, (
            f"P46 violation: substitution made conf WORSE: "
            f"quant={quant_conf}, drl_dir={drl_dir}, drl_conf={drl_conf}, "
            f"drl_auth={drl_auth}, result={result}"
        )

    @given(
        quant_conf=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        drl_dir=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
        drl_conf=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )
    @settings(max_examples=100, deadline=None)
    def test_drl_disabled_returns_quant(
        self, quant_conf, drl_dir, drl_conf,
    ):
        """When DRL not ACTIVE, no substitution — return quant_conf."""
        runner = self._runner()
        intent = _MockIntent(quant_conf)
        signals = {
            "drl_authority_level": "DISABLED",
            "drl_direction": drl_dir,
            "drl_confidence": drl_conf,
        }
        result = runner._compute_effective_weekend_confidence(intent, signals, "BTC")
        assert abs(result - quant_conf) < 1e-9, (
            f"DRL=DISABLED but result diverged from quant_conf: "
            f"quant={quant_conf}, result={result}"
        )

    @given(
        quant_conf=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        drl_dir=st.floats(min_value=-0.49, max_value=0.49, allow_nan=False),
        drl_conf=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )
    @settings(max_examples=100, deadline=None)
    def test_weak_drl_direction_no_substitution(
        self, quant_conf, drl_dir, drl_conf,
    ):
        """|drl_dir| < 0.5 => no substitution per P19/P20 thresholds."""
        runner = self._runner()
        intent = _MockIntent(quant_conf)
        signals = {
            "drl_authority_level": "ACTIVE",
            "drl_direction": drl_dir,
            "drl_confidence": drl_conf,
        }
        result = runner._compute_effective_weekend_confidence(intent, signals, "BTC")
        assert abs(result - quant_conf) < 1e-9, (
            f"P46 invariant violated: |drl_dir|={abs(drl_dir)} < 0.5 but "
            f"substitution occurred: quant={quant_conf}, drl_conf={drl_conf}, "
            f"result={result}"
        )


# =====================================================================
# 3. RiskVetoClassifier.classify
# =====================================================================

class TestRiskVetoClassifierProperties:
    """Invariants:
      - HARD threshold breach => HARD veto (regardless of other inputs)
      - PASS only when no HARD and no SOFT trips
      - Output veto_level is always one of {"HARD", "SOFT", "NONE"}
      - exposure_cap when SOFT is in [0, 1]
    """

    @given(
        drawdown=st.floats(min_value=0.20, max_value=1.0, allow_nan=False),
        mode=st.sampled_from(["NORMAL", "OPPORTUNITY", "DEFENSIVE"]),
    )
    @settings(max_examples=100, deadline=None)
    def test_drawdown_above_hard_threshold_always_hard(self, drawdown, mode):
        """UL-5: drawdown >= 0.20 must always be HARD veto."""
        from defense.production_reliability import RiskVetoClassifier
        classifier = RiskVetoClassifier()
        result = classifier.classify(mode=mode, drawdown=drawdown)
        assert result.veto_type.value == "HARD", (
            f"P50 violation: drawdown={drawdown} mode={mode} should be HARD, "
            f"got {result.veto_type.value}"
        )

    @given(
        correlation=st.floats(min_value=0.98, max_value=1.0, allow_nan=False),
        mode=st.sampled_from(["NORMAL", "OPPORTUNITY", "DEFENSIVE"]),
    )
    @settings(max_examples=100, deadline=None)
    def test_correlation_crisis_always_hard(self, correlation, mode):
        """UL-4: correlation >= 0.98 must always be HARD veto."""
        from defense.production_reliability import RiskVetoClassifier
        classifier = RiskVetoClassifier()
        result = classifier.classify(mode=mode, correlation=correlation)
        assert result.veto_type.value == "HARD"

    @given(
        dvol=st.floats(min_value=5.0, max_value=20.0, allow_nan=False),
        mode=st.sampled_from(["NORMAL", "OPPORTUNITY", "DEFENSIVE"]),
    )
    @settings(max_examples=100, deadline=None)
    def test_extreme_dvol_always_hard(self, dvol, mode):
        """dvol_zscore >= 5.0 must always be HARD."""
        from defense.production_reliability import RiskVetoClassifier
        classifier = RiskVetoClassifier()
        result = classifier.classify(mode=mode, dvol_zscore=dvol)
        assert result.veto_type.value == "HARD"

    @given(mode=st.sampled_from(["NORMAL", "OPPORTUNITY", "DEFENSIVE"]))
    def test_clean_inputs_pass(self, mode):
        """All inputs at safe levels => PASS."""
        from defense.production_reliability import RiskVetoClassifier
        classifier = RiskVetoClassifier()
        result = classifier.classify(
            mode=mode,
            drawdown=0.05,
            dvol_zscore=1.0,
            correlation=0.5,
            liquidity_usd=1_000_000,
            signal_conflict_score=0.0,
            data_valid=True,
            execution_failures=0,
            flash_crash_active=False,
            is_weekend=False,
        )
        assert result.veto_type.value == "NONE", (
            f"Clean inputs should be NONE, got {result.veto_type.value} mode={mode}"
        )

    @given(
        drawdown=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        dvol=st.floats(min_value=0.0, max_value=20.0, allow_nan=False),
        correlation=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        signal_conflict=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        exec_fails=st.integers(min_value=0, max_value=10),
        flash=st.booleans(),
        data_valid=st.booleans(),
        mode=st.sampled_from(["NORMAL", "OPPORTUNITY", "DEFENSIVE"]),
    )
    @settings(max_examples=300, deadline=None)
    def test_output_always_valid_level(
        self, drawdown, dvol, correlation, signal_conflict,
        exec_fails, flash, data_valid, mode,
    ):
        from defense.production_reliability import RiskVetoClassifier
        classifier = RiskVetoClassifier()
        result = classifier.classify(
            mode=mode, drawdown=drawdown, dvol_zscore=dvol,
            correlation=correlation, signal_conflict_score=signal_conflict,
            execution_failures=exec_fails, flash_crash_active=flash,
            data_valid=data_valid,
        )
        assert result.veto_type.value in {"HARD", "SOFT", "NONE"}, (
            f"Invalid veto_level {result.veto_type.value!r}"
        )
        # Exposure cap sanity (0 <= cap <= 1)
        if hasattr(result, "exposure_cap"):
            cap = result.exposure_cap
            if cap is not None:
                assert 0.0 <= cap <= 1.0, (
                    f"exposure_cap out of range: {cap}"
                )


# =====================================================================
# 4. AuthorityFusionEngine.fuse
# =====================================================================

class TestFusionEngineProperties:
    """Invariants:
      - direction in [-1.0, 1.0]
      - confidence in [0.0, 1.0]
      - target_exposure in [0.0, 1.0]
      - max_tranche_tier in [0, 4]
      - vetoes_active is always a list
      - Never raises on any input combo of agent signals
      - NO_TRADE mode => direction == 0.0 (P0 invariant)
    """

    def _make_engine(self):
        from signals.authority_fusion import AuthorityFusionEngine
        return AuthorityFusionEngine()

    def _make_ctx(self, mode_name="NORMAL"):
        from signals.authority_fusion import FusionContext
        from core.canonical_enums import SystemMode
        from market.phase_detector import RegimePhase
        mode_enum = getattr(SystemMode, mode_name)
        return FusionContext(
            mode=mode_enum,
            regime_phase=RegimePhase.UNDEFINED,
            data_valid=True,
            drl_enabled=True,
            regime="NEUTRAL_DRIFT",
            asset="BTC",
            current_price=50000.0,
        )

    def _make_signal(self, direction, confidence):
        from signals.authority_fusion import AgentSignal
        return AgentSignal(direction=direction, confidence=confidence)

    @given(
        quant_dir=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
        quant_conf=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        drl_dir=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
        drl_conf=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )
    @settings(max_examples=200, deadline=None)
    def test_fusion_output_in_bounds(
        self, quant_dir, quant_conf, drl_dir, drl_conf,
    ):
        engine = self._make_engine()
        signals = {
            "quant": self._make_signal(quant_dir, quant_conf),
            "drl": self._make_signal(drl_dir, drl_conf),
        }
        result = engine.fuse(signals, self._make_ctx())
        assert -1.0 <= result.direction <= 1.0, (
            f"direction out of bounds: {result.direction}"
        )
        assert 0.0 <= result.confidence <= 1.0, (
            f"confidence out of bounds: {result.confidence}"
        )
        assert 0.0 <= result.target_exposure <= 1.0, (
            f"target_exposure out of bounds: {result.target_exposure}"
        )
        assert 0 <= result.max_tranche_tier <= 4, (
            f"max_tranche_tier out of bounds: {result.max_tranche_tier}"
        )
        assert isinstance(result.vetoes_active, list)

    @given(
        quant_dir=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
        quant_conf=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )
    @settings(max_examples=50, deadline=None)
    def test_no_trade_mode_zero_direction(self, quant_dir, quant_conf):
        """P0 invariant: NO_TRADE mode must produce direction==0
        regardless of input signals."""
        engine = self._make_engine()
        signals = {"quant": self._make_signal(quant_dir, quant_conf)}
        result = engine.fuse(signals, self._make_ctx(mode_name="NO_TRADE"))
        assert result.direction == 0.0, (
            f"NO_TRADE produced direction={result.direction} — P0 violation"
        )
        assert result.target_exposure == 0.0

    def test_empty_signals_no_crash(self):
        """Fusion with zero agent inputs must not crash."""
        engine = self._make_engine()
        result = engine.fuse({}, self._make_ctx())
        assert -1.0 <= result.direction <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
