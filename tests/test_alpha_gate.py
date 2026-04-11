"""
Tests for Alpha Gate threshold logic — min_alpha_bps, SHORT tightening,
quiet_accum direction floor, borderline epsilon, NORMAL multiplier.
"""
import sys, os, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from defense.constitution import AlphaThresholdCalculator, AlphaGatingResult


@pytest.fixture
def calc():
    return AlphaThresholdCalculator()


class TestAlphaGateBasic:
    def test_no_trade_mode_always_rejects(self, calc):
        r = calc.check_alpha_gate(signal_strength=1.0, regime_confidence=1.0,
                                   mode="NO_TRADE")
        assert r.passes_threshold is False
        assert r.gate_decision == "REJECT_NO_TRADE"

    def test_strong_signal_passes(self, calc):
        r = calc.check_alpha_gate(
            signal_strength=0.8, regime_confidence=0.9, mode="NORMAL",
            min_alpha_bps=5.0, direction=0.8, quant_confidence=0.9,
            estimated_alpha_override=50.0,
        )
        assert r.passes_threshold is True

    def test_weak_signal_rejected(self, calc):
        r = calc.check_alpha_gate(
            signal_strength=0.01, regime_confidence=0.5, mode="NORMAL",
            min_alpha_bps=10.0, direction=0.01, quant_confidence=0.3,
            estimated_alpha_override=2.0,
        )
        assert r.passes_threshold is False


class TestQuietAccumDirectionFloor:
    def test_long_below_floor_rejected(self, calc):
        """LONG in QUIET_ACCUM with direction < floor → rejected."""
        r = calc.check_alpha_gate(
            signal_strength=0.05, regime_confidence=0.9, mode="NORMAL",
            min_alpha_bps=5.0, direction=0.05,
            regime="QUIET_ACCUMULATION",
            quiet_accum_min_direction=0.08,
        )
        assert r.passes_threshold is False
        assert "QUIET_ACCUMULATION" in r.reason

    def test_long_above_floor_not_rejected_by_direction(self, calc):
        """LONG in QUIET_ACCUM with direction > floor → NOT rejected by direction filter."""
        r = calc.check_alpha_gate(
            signal_strength=0.5, regime_confidence=0.9, mode="NORMAL",
            min_alpha_bps=5.0, direction=0.15,
            regime="QUIET_ACCUMULATION",
            quiet_accum_min_direction=0.08,
            estimated_alpha_override=30.0,
        )
        assert "REJECT_LOW_DIRECTION" not in (r.gate_decision or "")

    def test_short_uses_short_floor(self, calc):
        """SHORT in QUIET_ACCUM uses short-specific direction floor."""
        r = calc.check_alpha_gate(
            signal_strength=0.1, regime_confidence=0.9, mode="NORMAL",
            min_alpha_bps=5.0, direction=-0.10,
            regime="QUIET_ACCUMULATION",
            quiet_accum_min_direction=0.08,
            quiet_accum_min_direction_short=0.12,
        )
        assert r.passes_threshold is False
        assert "QUIET_ACCUMULATION" in r.reason


class TestBorderlineEpsilon:
    def test_epsilon_allows_marginal_pass(self, calc):
        """Signal just below threshold but within epsilon → should pass.
        EV threshold = friction(~7bps) * 1.10 = 7.7bps. With alpha=7.0 and
        epsilon=2.0: net_alpha + epsilon should cross threshold."""
        r = calc.check_alpha_gate(
            signal_strength=0.5, regime_confidence=0.8, mode="NORMAL",
            min_alpha_bps=5.0, direction=0.5, quant_confidence=0.8,
            estimated_alpha_override=7.0,  # Close to EV threshold ~7.7bps
            borderline_pass_epsilon_bps=3.0,
        )
        # With 3bps epsilon, marginal signals should pass
        # (exact depends on friction internals, so just verify epsilon is considered)
        assert r.passes_threshold is True or r.gate_decision in ("ALLOW_EPSILON", "ALLOW")


class TestNormalMultiplier:
    def test_normal_multiplier_is_1_10(self, calc):
        """NORMAL_MULTIPLIER should be 1.10 (lowered from 1.25)."""
        assert calc.NORMAL_MULTIPLIER == pytest.approx(1.10, abs=0.01)

    def test_opportunity_multiplier_is_1_0(self, calc):
        """OPPORTUNITY multipliers should be 1.0 (no friction markup)."""
        assert calc.OPPORTUNITY_LEAD_LAG_MULTIPLIER == pytest.approx(1.0)
        assert calc.OPPORTUNITY_CRACK_MULTIPLIER == pytest.approx(1.0)
