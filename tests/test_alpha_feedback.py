"""
ALPHA-FEEDBACK (2026-06-13) regression tests.

The live alpha estimate is `signal_edge_bps = |quant_dir| * 65`, passed to the
gate as estimated_alpha_override. Live-IC analysis showed the quant signal is
NOISE (IC -0.018) and the system's measured alpha is negative (-62%/yr), yet the
override path ignored the realized-hit-rate feedback the system already tracks
(update_hit_rate -> _rolling_hit_rate). This fix applies the same
performance_factor = 0.5 + 0.5*hit_rate (range [0.5, 1.0]) to the override path,
so the estimate self-corrects toward realized outcomes and the gate tightens when
the signal stops working. Floor 0.5 prevents starving trading.
"""

import pytest

from defense.constitution import AlphaThresholdCalculator


@pytest.fixture
def calc():
    c = AlphaThresholdCalculator()
    c._alpha_feedback_enabled = True
    return c


class TestAlphaFeedback:
    def test_haircut_applied_at_neutral_hit_rate(self, calc):
        calc._rolling_hit_rate = 0.5  # factor = 0.75
        r = calc.check_alpha_gate(
            signal_strength=0.5, regime_confidence=0.8, mode="NORMAL",
            min_alpha_bps=5.0, direction=0.5, quant_confidence=0.7,
            estimated_alpha_override=40.0,
        )
        assert r.estimated_alpha_bps == pytest.approx(40.0 * 0.75, rel=1e-6)

    def test_winning_signal_no_haircut(self, calc):
        calc._rolling_hit_rate = 1.0  # factor = 1.0
        r = calc.check_alpha_gate(
            signal_strength=0.5, regime_confidence=0.8, mode="NORMAL",
            min_alpha_bps=5.0, direction=0.5, quant_confidence=0.7,
            estimated_alpha_override=40.0,
        )
        assert r.estimated_alpha_bps == pytest.approx(40.0, rel=1e-6)

    def test_losing_signal_floored_at_half(self, calc):
        calc._rolling_hit_rate = 0.0  # factor = 0.5 (floor, never lower)
        r = calc.check_alpha_gate(
            signal_strength=0.5, regime_confidence=0.8, mode="NORMAL",
            min_alpha_bps=5.0, direction=0.5, quant_confidence=0.7,
            estimated_alpha_override=40.0,
        )
        assert r.estimated_alpha_bps == pytest.approx(20.0, rel=1e-6)

    def test_lower_hit_rate_tightens_gate(self, calc):
        # an override that passes when winning should fail when losing, near the edge
        kwargs = dict(signal_strength=0.5, regime_confidence=0.8, mode="NORMAL",
                      min_alpha_bps=5.0, direction=0.5, quant_confidence=0.7)
        calc._rolling_hit_rate = 1.0
        hi = calc.check_alpha_gate(estimated_alpha_override=16.0, **kwargs)
        calc._rolling_hit_rate = 0.3  # factor 0.65 -> 16 -> 10.4bps
        lo = calc.check_alpha_gate(estimated_alpha_override=16.0, **kwargs)
        assert hi.estimated_alpha_bps > lo.estimated_alpha_bps
        # the haircut should move the net-alpha margin in the tightening direction
        assert lo.expected_net_alpha_bps < hi.expected_net_alpha_bps

    def test_disabled_flag_no_haircut(self, calc):
        calc._alpha_feedback_enabled = False
        calc._rolling_hit_rate = 0.3
        r = calc.check_alpha_gate(
            signal_strength=0.5, regime_confidence=0.8, mode="NORMAL",
            min_alpha_bps=5.0, direction=0.5, quant_confidence=0.7,
            estimated_alpha_override=40.0,
        )
        assert r.estimated_alpha_bps == pytest.approx(40.0, rel=1e-6)

    def test_env_var_controls_default(self, monkeypatch):
        monkeypatch.setenv("HMATS_ALPHA_FEEDBACK", "0")
        c = AlphaThresholdCalculator()
        assert c._alpha_feedback_enabled is False
        monkeypatch.setenv("HMATS_ALPHA_FEEDBACK", "1")
        c2 = AlphaThresholdCalculator()
        assert c2._alpha_feedback_enabled is True


class TestAlphaFeedbackSourceGuard:
    def test_override_path_consumes_hit_rate(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parents[1] / "defense" / "constitution.py").read_text(encoding="utf-8")
        # the override branch must apply the performance factor (was previously dropped)
        assert "ALPHA-FEEDBACK" in src
        assert "_perf_factor = 0.5 + (self._rolling_hit_rate * 0.5)" in src
