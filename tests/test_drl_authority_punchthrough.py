"""
Regression tests for the DRL ACTIVE authority punch-through family of fixes.

P12 [FIXED 2026-04-24]: 2-agent conflict score 0.7 was force-promoted to 1.0
  HARD VETO. Threshold aligned to >= 0.9 (3-agent conflict only).

P19 [FIXED 2026-04-24]: BEST_OF_N_HOLD short-circuit set intent.direction=0
  before fusion when TA Best-of-N picked 'hold', nullifying DRL's DECIDE
  authority post-2026-04-22 promotion. Punch-through added: if DRL is ACTIVE
  and emits |dir|>=0.5 / conf>=0.3, skip the hold short-circuit.

P20 [FIXED 2026-04-24]: _compute_effective_alpha_direction early-returned
  with quant_direction (=0 when quant abstains), making DRL's DECIDE signal
  invisible to alpha gate. Substitution added: if quant abstains and DRL is
  ACTIVE+strong, use DRL direction as effective alpha input.

P46 [FIXED 2026-04-25]: weekend gate read intent.quant_confidence which is
  clamped to [0.40, 0.70] for HOLD-strategy and dampened to 0.33-0.42 by
  downstream multipliers. DRL's actual confidence (0.40+) was invisible.
  Substitution added: if DRL ACTIVE + |dir|>=0.5 + drl_conf>=0.3, use
  max(quant_conf, drl_conf) as effective weekend confidence input.

These tests guard against silent regressions of the punch-through logic
(e.g., threshold drift, removal of the override branch in a refactor).
"""
import sys, os
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from integration.integration_v36 import HMATSv36Engine, TradeIntentV36
from main import HMATSProductionRunner


# =============================================================================
# Helpers — bypass heavy __init__ for unit-level testing
# =============================================================================

def _make_engine():
    engine = HMATSv36Engine.__new__(HMATSv36Engine)
    engine._last_alpha_result = None
    return engine


def _make_runner(alpha_gate_config=None):
    runner = HMATSProductionRunner.__new__(HMATSProductionRunner)
    runner.config = SimpleNamespace(alpha_gate_config=alpha_gate_config or {})
    return runner


# =============================================================================
# P19 — BEST_OF_N_HOLD DRL punch-through
# =============================================================================

class TestP19BestOfNHoldPunchthrough:
    """When primary_strategy='hold' AND DRL is ACTIVE+strong, skip the hold
    short-circuit so fusion + alpha gate can act on DRL's direction."""

    def _hold_signals(self, drl_auth="DISABLED", drl_dir=0.0, drl_conf=0.0):
        return {
            "primary_strategy": "hold",
            "quant_direction": 0.0,
            "drl_authority_level": drl_auth,
            "drl_direction": drl_dir,
            "drl_confidence": drl_conf,
        }

    def test_drl_disabled_applies_hold(self):
        """Baseline: DRL DISABLED → hold short-circuit fires."""
        engine = _make_engine()
        intent = TradeIntentV36(asset="BTC", system_mode="NORMAL")
        applied = engine._maybe_apply_pre_alpha_hold(
            asset="BTC",
            intent=intent,
            agent_signals=self._hold_signals(drl_auth="DISABLED"),
            market_data={"regime_state": "QUIET_ACCUMULATION", "current_exposure": 0.0},
            position_state={"current_exposure": 0.0},
        )
        assert applied is True
        assert intent.direction == 0.0
        assert intent.tranche_action == "HOLD"
        assert engine._last_alpha_result.gate_decision == "BEST_OF_N_HOLD"

    def test_drl_active_strong_punches_through(self):
        """DRL ACTIVE + |dir|=0.9 + conf=0.7 → punch-through, return False
        (defer to fusion)."""
        engine = _make_engine()
        intent = TradeIntentV36(asset="BTC", system_mode="NORMAL")
        applied = engine._maybe_apply_pre_alpha_hold(
            asset="BTC",
            intent=intent,
            agent_signals=self._hold_signals(drl_auth="ACTIVE", drl_dir=-0.9, drl_conf=0.7),
            market_data={"regime_state": "QUIET_ACCUMULATION", "current_exposure": 0.0},
            position_state={"current_exposure": 0.0},
        )
        assert applied is False, "DRL ACTIVE + strong should skip hold short-circuit"
        # Intent untouched — defers to fusion
        assert intent.tranche_action == ""

    def test_drl_active_strong_long_punches_through(self):
        """Mirror of above but LONG direction."""
        engine = _make_engine()
        intent = TradeIntentV36(asset="ETH", system_mode="NORMAL")
        applied = engine._maybe_apply_pre_alpha_hold(
            asset="ETH",
            intent=intent,
            agent_signals=self._hold_signals(drl_auth="ACTIVE", drl_dir=0.85, drl_conf=0.5),
            market_data={"regime_state": "QUIET_ACCUMULATION", "current_exposure": 0.0},
            position_state={"current_exposure": 0.0},
        )
        assert applied is False

    def test_drl_active_weak_direction_applies_hold(self):
        """DRL ACTIVE but |dir|=0.4 < 0.5 threshold → hold still applies."""
        engine = _make_engine()
        intent = TradeIntentV36(asset="BTC", system_mode="NORMAL")
        applied = engine._maybe_apply_pre_alpha_hold(
            asset="BTC",
            intent=intent,
            agent_signals=self._hold_signals(drl_auth="ACTIVE", drl_dir=0.4, drl_conf=0.7),
            market_data={"regime_state": "QUIET_ACCUMULATION", "current_exposure": 0.0},
            position_state={"current_exposure": 0.0},
        )
        assert applied is True
        assert intent.direction == 0.0

    def test_drl_active_low_confidence_applies_hold(self):
        """DRL ACTIVE + strong dir but conf=0.2 < 0.3 → hold still applies."""
        engine = _make_engine()
        intent = TradeIntentV36(asset="BTC", system_mode="NORMAL")
        applied = engine._maybe_apply_pre_alpha_hold(
            asset="BTC",
            intent=intent,
            agent_signals=self._hold_signals(drl_auth="ACTIVE", drl_dir=0.9, drl_conf=0.2),
            market_data={"regime_state": "QUIET_ACCUMULATION", "current_exposure": 0.0},
            position_state={"current_exposure": 0.0},
        )
        assert applied is True
        assert intent.direction == 0.0

    def test_drl_advise_does_not_punch_through(self):
        """DRL must be ACTIVE specifically — ADVISE doesn't override hold."""
        engine = _make_engine()
        intent = TradeIntentV36(asset="BTC", system_mode="NORMAL")
        applied = engine._maybe_apply_pre_alpha_hold(
            asset="BTC",
            intent=intent,
            agent_signals=self._hold_signals(drl_auth="ADVISE", drl_dir=0.9, drl_conf=0.7),
            market_data={"regime_state": "QUIET_ACCUMULATION", "current_exposure": 0.0},
            position_state={"current_exposure": 0.0},
        )
        assert applied is True


# =============================================================================
# P20 — effective_alpha_direction DRL substitution
# =============================================================================

class TestP20EffectiveAlphaDrlSubstitution:
    """When quant abstains (|qd|<0.03) AND DRL is ACTIVE+strong, the alpha
    gate input must use DRL direction instead of zero."""

    def test_drl_disabled_returns_zero_when_quant_abstains(self):
        """Baseline: quant=0, DRL DISABLED → returns 0."""
        runner = _make_runner()
        result = runner._compute_effective_alpha_direction(
            "BTC",
            {
                "quant_direction": 0.0,
                "regime_state": "QUIET_ACCUMULATION",
                "drl_authority_level": "DISABLED",
                "drl_direction": 0.0,
                "drl_confidence": 0.0,
            },
            {"regime_state": "QUIET_ACCUMULATION"},
        )
        assert result["direction"] == 0.0
        assert result["sources"] == []

    def test_drl_active_strong_substitutes_for_abstaining_quant(self):
        """quant=0, DRL ACTIVE + |dir|=0.9 + conf=0.5 → returns DRL dir."""
        runner = _make_runner()
        result = runner._compute_effective_alpha_direction(
            "BTC",
            {
                "quant_direction": 0.0,
                "regime_state": "QUIET_ACCUMULATION",
                "drl_authority_level": "ACTIVE",
                "drl_direction": -0.9,
                "drl_confidence": 0.5,
            },
            {"regime_state": "QUIET_ACCUMULATION"},
        )
        assert result["direction"] == -0.9
        assert result["sources"] and "drl" in result["sources"][0]
        assert result["context_boost"] > 0.0

    def test_drl_active_weak_direction_does_not_substitute(self):
        """quant=0, DRL ACTIVE but |dir|=0.4 < 0.5 → returns 0 (no sub)."""
        runner = _make_runner()
        result = runner._compute_effective_alpha_direction(
            "BTC",
            {
                "quant_direction": 0.0,
                "regime_state": "QUIET_ACCUMULATION",
                "drl_authority_level": "ACTIVE",
                "drl_direction": 0.4,
                "drl_confidence": 0.8,
            },
            {"regime_state": "QUIET_ACCUMULATION"},
        )
        assert result["direction"] == 0.0
        assert "drl" not in (result["sources"][0] if result["sources"] else "")

    def test_drl_active_low_confidence_does_not_substitute(self):
        """quant=0, DRL ACTIVE + strong dir but conf=0.2 < 0.3 → returns 0."""
        runner = _make_runner()
        result = runner._compute_effective_alpha_direction(
            "BTC",
            {
                "quant_direction": 0.0,
                "regime_state": "QUIET_ACCUMULATION",
                "drl_authority_level": "ACTIVE",
                "drl_direction": 0.9,
                "drl_confidence": 0.2,
            },
            {"regime_state": "QUIET_ACCUMULATION"},
        )
        assert result["direction"] == 0.0

    def test_quant_active_does_not_trigger_drl_substitution(self):
        """quant_dir=0.5 (non-abstaining) → quant path used, no DRL sub."""
        runner = _make_runner()
        result = runner._compute_effective_alpha_direction(
            "BTC",
            {
                "quant_direction": 0.5,
                "regime_state": "QUIET_ACCUMULATION",
                "drl_authority_level": "ACTIVE",
                "drl_direction": -0.9,  # Opposite direction!
                "drl_confidence": 0.7,
            },
            {"regime_state": "QUIET_ACCUMULATION"},
        )
        # Substitution must NOT flip direction — quant_dir wins when non-abstaining
        assert result["direction"] >= 0.5 - 1e-9
        # Sources will not start with "drl:..."
        assert not any(s.startswith("drl:") for s in result["sources"])

    def test_drl_advise_authority_does_not_substitute(self):
        """DRL must be ACTIVE — ADVISE doesn't trigger substitution."""
        runner = _make_runner()
        result = runner._compute_effective_alpha_direction(
            "BTC",
            {
                "quant_direction": 0.0,
                "regime_state": "QUIET_ACCUMULATION",
                "drl_authority_level": "ADVISE",
                "drl_direction": 0.9,
                "drl_confidence": 0.7,
            },
            {"regime_state": "QUIET_ACCUMULATION"},
        )
        assert result["direction"] == 0.0


# =============================================================================
# P46 — Weekend gate DRL confidence substitution
# =============================================================================

class TestP46WeekendDrlConfSubstitution:
    """When DRL is ACTIVE+strong, weekend gate must receive max(quant_conf,
    drl_conf) instead of just quant_conf — same shape as P20's alpha-direction
    substitution."""

    def _make_intent(self, quant_conf=0.4):
        return SimpleNamespace(quant_confidence=quant_conf)

    def test_drl_disabled_returns_quant_conf(self):
        runner = _make_runner()
        intent = self._make_intent(quant_conf=0.42)
        result = runner._compute_effective_weekend_confidence(
            intent,
            {
                "drl_authority_level": "DISABLED",
                "drl_direction": 0.0,
                "drl_confidence": 0.0,
            },
            "BTC",
        )
        assert result == pytest.approx(0.42)

    def test_drl_active_strong_substitutes_higher_drl_conf(self):
        """The bug case from April 25: quant_conf=0.33, drl_conf=0.44.
        Substitution should swap to 0.44."""
        runner = _make_runner()
        intent = self._make_intent(quant_conf=0.33)
        result = runner._compute_effective_weekend_confidence(
            intent,
            {
                "drl_authority_level": "ACTIVE",
                "drl_direction": -0.93,
                "drl_confidence": 0.44,
            },
            "BTC",
        )
        assert result == pytest.approx(0.44)

    def test_drl_active_weak_direction_does_not_substitute(self):
        """DRL ACTIVE but |dir|=0.4 < 0.5 → no substitution."""
        runner = _make_runner()
        intent = self._make_intent(quant_conf=0.33)
        result = runner._compute_effective_weekend_confidence(
            intent,
            {
                "drl_authority_level": "ACTIVE",
                "drl_direction": 0.4,
                "drl_confidence": 0.6,
            },
            "BTC",
        )
        assert result == pytest.approx(0.33)

    def test_drl_active_low_conf_does_not_substitute(self):
        """DRL ACTIVE but drl_conf=0.2 < 0.3 threshold → no substitution."""
        runner = _make_runner()
        intent = self._make_intent(quant_conf=0.33)
        result = runner._compute_effective_weekend_confidence(
            intent,
            {
                "drl_authority_level": "ACTIVE",
                "drl_direction": 0.9,
                "drl_confidence": 0.2,
            },
            "BTC",
        )
        assert result == pytest.approx(0.33)

    def test_drl_conf_lower_than_quant_does_not_substitute(self):
        """If quant_conf already >= drl_conf, no swap (substitution should
        only ever IMPROVE confidence, not lower it)."""
        runner = _make_runner()
        intent = self._make_intent(quant_conf=0.7)
        result = runner._compute_effective_weekend_confidence(
            intent,
            {
                "drl_authority_level": "ACTIVE",
                "drl_direction": 0.9,
                "drl_confidence": 0.5,
            },
            "BTC",
        )
        assert result == pytest.approx(0.7)

    def test_drl_advise_authority_does_not_substitute(self):
        """DRL must be ACTIVE specifically — ADVISE doesn't trigger swap."""
        runner = _make_runner()
        intent = self._make_intent(quant_conf=0.33)
        result = runner._compute_effective_weekend_confidence(
            intent,
            {
                "drl_authority_level": "ADVISE",
                "drl_direction": 0.9,
                "drl_confidence": 0.7,
            },
            "BTC",
        )
        assert result == pytest.approx(0.33)


# =============================================================================
# P12 — signal_conflict score promotion threshold
# =============================================================================

class TestP12SignalConflictThreshold:
    """The 2-agent vs 3-agent conflict thresholds must stay aligned with
    constitution.py:441-444 — only 3-agent conflict (score >= 0.9) escalates
    to HARD VETO. The previous bug promoted 2-agent score 0.7 to 1.0."""

    def test_threshold_at_veto_call_site_is_geq_09(self):
        """Source-level guard: the signal_conflict_score=... line that feeds
        risk_veto_classifier.classify must use >= 0.9 to suppress 2-agent
        conflict over-promotion. The cosmetic intent.signal_conflict_detected
        flag at line 1825 uses > 0.5 intentionally (audit, not veto)."""
        path = os.path.join(
            os.path.dirname(__file__), "..", "integration", "integration_v36.py"
        )
        with open(path, "r", encoding="utf-8") as fh:
            src = fh.read()

        # Locate the signal_conflict_score=... line that feeds risk classifier.
        # Anchor: must contain `signal_conflict_score=1.0 if (...trigger_scores...
        # signal_conflict... >= 0.9` together on consecutive lines.
        assert "signal_conflict_score=1.0" in src, (
            "P12 regression: signal_conflict_score promotion line missing"
        )
        # Threshold check on the same expression. Catches drift to >= 0.7 or > 0.5.
        veto_window = src.split("signal_conflict_score=1.0", 1)[1][:300]
        assert (
            'signal_conflict", 0) >= 0.9' in veto_window
            or "signal_conflict', 0) >= 0.9" in veto_window
        ), (
            "P12 threshold drift: veto-path signal_conflict must require >= 0.9 "
            f"(found context: {veto_window[:200]!r})"
        )
