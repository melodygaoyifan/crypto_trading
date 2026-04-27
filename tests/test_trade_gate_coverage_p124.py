"""
test_trade_gate_coverage_p124.py — dynamic-dispatch reject coverage (P124)
================================================================================

P121 (Tier A3) covered the 3 statically-dispatched untested gates
(DRL_OVERCONFIDENT / VOLUME_CONTRACTING / STRUCTURE_INVALID). The decision-
trace coverage analyzer flagged 10 more reject reasons as REACHABLE-but-
not-covered, but tracing them showed:

  REAL gates lacking tests (4):
    - DVOL_SPIKE                    (Gate 2 EMERGENCY_FLAT, line 635)
    - OPPORTUNITY_BUDGET_EXCEEDED   (Gate 7 governor dispatch, line 713)
    - REGIME_TRANSITION_BLOCK       (Gate 7 governor dispatch, line 714)
    - CASCADE_EXHAUSTED             (Gate 7 governor dispatch, line 715)

  DEAD enum members (no fire site anywhere — 7):
    - INSUFFICIENT_EDGE
    - MACRO_RISK
    - POSITION_LIMIT
    - DRAWDOWN_LIMIT
    - REGIME_UNSTABLE
    - SOL_NETWORK_STRESS
    - DATA_HEALTH_DEGRADED  (RejectReason member but only DATA_HEALTH_CRITICAL +
                              DATA_HEALTH_EXIT_ONLY actually fire)

This file adds 4 behavioral tests for the real gates + 1 housekeeping test
documenting the 7 dead members so a future cleanup commit can remove them
without losing the inventory.
"""
from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest


def _make_proposal(**overrides):
    from defense.trade_gate import TradeProposal
    defaults = dict(
        asset="BTC", side="long", size=Decimal("100"),
        expected_alpha_bps=Decimal("50"),
        signal_source="quant", confidence=0.6,
        regime="STEADY_UPTREND", regime_confidence=0.8,
        drl_weight=0.4, dvol_current=10.0, dvol_zscore=0.5,
        volume_ratio=1.5, is_structure_breakout=True, price_direction="up",
    )
    defaults.update(overrides)
    return TradeProposal(**defaults)


def _make_gate(*, with_governor=False, governor_result=None):
    """Construct TradeGate with all upstream gates passing.
    Optionally inject a governor that returns a specific veto."""
    import time
    from defense.trade_gate import TradeGate, TradeGateConfig
    cfg = TradeGateConfig()
    gate = TradeGate(cfg)
    gate._data_health_enabled = False
    gate._data_health_snapshot = None
    now = time.time()
    for source in ("price", "orderbook", "vpin"):
        gate.stale_guard.update_timestamp(source, now)

    if with_governor:
        gate._governor_enabled = True
        gov = MagicMock()
        gov.check_all.return_value = governor_result
        gate._governor_integration = gov

    return gate


def _governor_result(vetoed_by_value: str, reason: str = "test_veto"):
    """Build a GovernorCheckResult that vetoes the trade."""
    from defense.governor_integration import (
        GovernorCheckResult, GovernorVetoSource,
    )
    return GovernorCheckResult(
        allowed=False,
        vetoed_by=getattr(GovernorVetoSource, vetoed_by_value),
        reason=reason,
        modifications={},
    )


# =====================================================================
# DVOL_SPIKE — Gate 2 EMERGENCY_FLAT (line 635)
# =====================================================================

class TestDvolSpikeReject:
    """DVOL z-score >= dvol_zscore_emergency (default 5.0) trips
    EMERGENCY_FLAT mode. Result.reason should be DVOL_SPIKE."""

    def test_extreme_dvol_zscore_triggers_emergency_flat(self):
        from defense.trade_gate import (
            RejectReason, GateDecision,
        )
        gate = _make_gate()
        # Trigger emergency mode by passing dvol > 5.0 (dvol_zscore_emergency).
        # The DVOLDefense.update() call in evaluate() sets emergency_mode=True
        # when dvol_zscore >= 5.0, then check() returns (False, "EMERGENCY_FLAT").
        proposal = _make_proposal(
            dvol_current=15.0,
            dvol_zscore=6.5,  # WELL above 5.0 emergency threshold
        )
        result = gate.evaluate(proposal)
        assert result.reason == RejectReason.DVOL_SPIKE, (
            f"Expected DVOL_SPIKE, got {result.reason}. "
            f"Details: {result.details}"
        )
        assert result.decision == GateDecision.EMERGENCY_FLAT, (
            f"Expected EMERGENCY_FLAT decision, got {result.decision}"
        )


# =====================================================================
# Governor dispatch — OPPORTUNITY_BUDGET_EXCEEDED / REGIME_TRANSITION_BLOCK /
# CASCADE_EXHAUSTED (Gate 7, line 713-715 dispatch table)
# =====================================================================

class TestGovernorDispatchRejects:
    """Each GovernorVetoSource maps to its corresponding RejectReason
    via the reason_map dict at trade_gate.py:712-715."""

    def test_opportunity_budget_dispatch(self):
        from defense.trade_gate import RejectReason
        gov_result = _governor_result("OPPORTUNITY_BUDGET",
                                      reason="weekly budget exhausted")
        gate = _make_gate(with_governor=True, governor_result=gov_result)
        proposal = _make_proposal()
        result = gate.evaluate(proposal)
        assert result.reason == RejectReason.OPPORTUNITY_BUDGET_EXCEEDED, (
            f"Governor OPPORTUNITY_BUDGET should map to "
            f"RejectReason.OPPORTUNITY_BUDGET_EXCEEDED, got {result.reason}. "
            f"Dispatch table at trade_gate.py:712 may have drifted."
        )

    def test_regime_transition_dispatch(self):
        from defense.trade_gate import RejectReason
        gov_result = _governor_result("REGIME_TRANSITION",
                                      reason="regime in transition buffer")
        gate = _make_gate(with_governor=True, governor_result=gov_result)
        proposal = _make_proposal()
        result = gate.evaluate(proposal)
        assert result.reason == RejectReason.REGIME_TRANSITION_BLOCK, (
            f"Governor REGIME_TRANSITION should map to "
            f"RejectReason.REGIME_TRANSITION_BLOCK, got {result.reason}."
        )

    def test_cascade_exhaustion_dispatch(self):
        from defense.trade_gate import RejectReason
        gov_result = _governor_result("CASCADE_EXHAUSTION",
                                      reason="cascade governor exhausted")
        gate = _make_gate(with_governor=True, governor_result=gov_result)
        proposal = _make_proposal()
        result = gate.evaluate(proposal)
        assert result.reason == RejectReason.CASCADE_EXHAUSTED, (
            f"Governor CASCADE_EXHAUSTION should map to "
            f"RejectReason.CASCADE_EXHAUSTED, got {result.reason}."
        )


# =====================================================================
# Dead enum member inventory — flag for future cleanup
# =====================================================================

class TestDeadRejectReasonInventory:
    """7 RejectReason enum members exist but have ZERO fire sites in the
    codebase. Documented here so a future cleanup commit can remove them
    without losing the inventory.

    If one of these starts firing, this test should FAIL and the operator
    should both (a) add a behavioral test + (b) remove the member from
    DEAD_AS_OF_P124.
    """

    DEAD_AS_OF_P124 = {
        "INSUFFICIENT_EDGE",
        "MACRO_RISK",
        "POSITION_LIMIT",
        "DRAWDOWN_LIMIT",
        "REGIME_UNSTABLE",
        "SOL_NETWORK_STRESS",
        "DATA_HEALTH_DEGRADED",
    }

    def test_dead_members_still_dead(self):
        """Each dead member has no _reject(...) call site in defense/.
        If grep finds one, it's no longer dead — add a real test."""
        import subprocess
        from pathlib import Path
        repo_root = Path(__file__).resolve().parent.parent
        defense_dir = repo_root / "defense"
        # Crude grep: search for `RejectReason.<NAME>` in defense/
        for member in self.DEAD_AS_OF_P124:
            count = 0
            for py in defense_dir.rglob("*.py"):
                try:
                    txt = py.read_text(encoding="utf-8-sig")
                except (UnicodeDecodeError, OSError):
                    continue
                count += txt.count(f"RejectReason.{member}")
            # Subtract the enum DEFINITION itself (one occurrence in
            # trade_gate.py from the `MEMBER = auto()` line — but that's
            # the bare member name, not `RejectReason.MEMBER`)
            assert count == 0, (
                f"RejectReason.{member} now has {count} reference(s) in "
                f"defense/ — was DEAD as of P124. Either: (a) add a "
                f"behavioral test like P124 above + remove from "
                f"DEAD_AS_OF_P124, or (b) the new reference is a typo and "
                f"should be reverted."
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
