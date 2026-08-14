"""[P265] The GateDecision contract, and the P0 DVOL time bomb.

`TradeGate.evaluate` emits ALLOW / REDUCE / REJECT / EMERGENCY_FLAT — and its
two consumers mishandled non-ALLOW decisions in OPPOSITE directions:

  * p0_safety_integrator handled only REJECT/EXIT_ONLY/CLIP, so
    EMERGENCY_FLAT (the DVOL z>=5 response) and REDUCE fell through the elif
    chain as full-size ALLOW — an emergency-flatten decision read as a pass.
    Its CLIP branch also read `gate_result.clipped_size`, a field that does
    not exist (the dataclass field is `adjusted_size`).
  * main.py treated ANY non-ALLOW as a full veto: REDUCE ("trade at 50-75%
    size", carrying reason=NONE) became a bare "[TRADE_GATE] NONE" veto that
    the sleeve translator classified as veto_flat — a size-reduction advisory
    LIQUIDATED the routed book.

And CHECK 5 (DVOL override) was a time bomb: it called
`dvol_controller.get_execution_mode()` — a method that DOES NOT EXIST — so the
day a `market_data["dvol"]` producer appeared, the AttributeError would hit
FIX-29's fail-closed handler and veto EVERY tick (flattening the sleeve via
the P265b classification). Even past that, it compared `mode.value` (auto()
ints) against "HALT"/"REDUCED" — strings matching neither the enum's names
nor values.
"""

import types
from decimal import Decimal

import pytest

from defense.execution_guards import DVOLOverrideController, ExecutionMode
from defense.p0_safety_integrator import P0SafetyIntegrator
from defense.trade_gate import GateDecision

from main import SLEEVE_HOLD, sleeve_direction_from_intent


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _gate_result(decision, reason_name="NONE", reason_value="none",
                 adjusted_size=Decimal("0.05")):
    return types.SimpleNamespace(
        decision=decision,
        reason=types.SimpleNamespace(name=reason_name, value=reason_value),
        details={},
        adjusted_size=adjusted_size,
    )


def _integrator(gate_decision=None, dvol_mode=None, **gate_kw):
    p0 = P0SafetyIntegrator(config={})
    # Null every component so only the one under test runs.
    p0.risk_controller = None
    p0.trade_gate = None
    p0.stale_guard = None
    p0.dvol_controller = None
    p0.rate_limiter = None
    p0.human_override = None
    p0.shadow_ledger = None
    if gate_decision is not None:
        p0.trade_gate = types.SimpleNamespace(
            check=lambda **kw: _gate_result(gate_decision, **gate_kw))
    if dvol_mode is not None:
        p0.dvol_controller = types.SimpleNamespace(
            update=lambda v: (True, dvol_mode, "test"))
    return p0


def _check(p0, dvol=0.0, is_entry=True, size=0.10):
    return p0.check_pre_execution(
        asset="BTC", direction=1, size=size, is_entry=is_entry, dvol=dvol)


# ---------------------------------------------------------------------------
# 1. p0 consumer: EMERGENCY_FLAT / REDUCE / DELAY no longer pass as ALLOW
# ---------------------------------------------------------------------------

class TestP0GateDecisionBranches:
    def test_emergency_flat_blocks_the_entry_and_keeps_exits_open(self):
        r = _check(_integrator(gate_decision=GateDecision.EMERGENCY_FLAT))
        assert r.allow_trade is False, (
            "EMERGENCY_FLAT fell through as a full-size ALLOW — an "
            "emergency-flatten decision read as a pass (P265)")
        assert r.allow_exit is True, (
            "the whole point of EMERGENCY_FLAT is exiting")

    def test_reduce_clips_to_the_gates_adjusted_size(self):
        r = _check(_integrator(gate_decision=GateDecision.REDUCE,
                               adjusted_size=Decimal("0.05")), size=0.10)
        assert r.allow_trade is True
        assert float(r.clipped_size) == pytest.approx(0.05), (
            f"clipped_size={r.clipped_size} — REDUCE passed at full size")

    def test_delay_blocks_the_entry_this_pass(self):
        r = _check(_integrator(gate_decision=GateDecision.DELAY))
        assert r.allow_trade is False
        assert r.allow_exit is True

    def test_reject_still_blocks(self):
        r = _check(_integrator(gate_decision=GateDecision.REJECT))
        assert r.allow_trade is False

    def test_allow_still_allows(self):
        r = _check(_integrator(gate_decision=GateDecision.ALLOW))
        assert r.allow_trade is True


# ---------------------------------------------------------------------------
# 2. CHECK 5: the DVOL override actually works when fed
# ---------------------------------------------------------------------------

class TestDVOLCheckFive:
    def test_flat_only_blocks_trade_allows_exit(self):
        r = _check(_integrator(dvol_mode=ExecutionMode.FLAT_ONLY), dvol=105.0)
        assert r.allow_trade is False
        assert r.allow_exit is True

    def test_defensive_clips_to_half(self):
        r = _check(_integrator(dvol_mode=ExecutionMode.DEFENSIVE),
                   dvol=85.0, size=0.10)
        assert float(r.clipped_size) == pytest.approx(0.05)
        assert r.allow_trade is True

    def test_normal_mode_passes(self):
        r = _check(_integrator(dvol_mode=ExecutionMode.NORMAL), dvol=50.0)
        assert r.allow_trade is True

    def test_a_real_controller_does_not_explode_the_chain(self):
        # The time bomb: get_execution_mode() does not exist on the REAL
        # controller — the old code would AttributeError into FIX-29's
        # fail-closed handler and veto every tick. The real API is
        # update(dvol) -> (override_active, ExecutionMode, reason).
        p0 = _integrator()
        p0.dvol_controller = DVOLOverrideController()
        r = _check(p0, dvol=50.0)
        assert r.allow_trade is True, (
            f"a REAL DVOLOverrideController at normal vol blocked the trade: "
            f"{r.reason} — the CHECK-5 call contract is broken again")

    def test_the_old_method_still_does_not_exist(self):
        # If someone adds get_execution_mode(), this documents that CHECK 5
        # must keep using update()'s 3-tuple — do not resurrect the old call.
        assert not hasattr(DVOLOverrideController(), "get_execution_mode")

    def test_real_controller_update_contract(self):
        ok, mode, reason = DVOLOverrideController().update(50.0)
        assert isinstance(mode, ExecutionMode)
        assert isinstance(reason, str)


# ---------------------------------------------------------------------------
# 3. main.py consumer: REDUCE/DELAY hold the sleeve instead of flattening it
# ---------------------------------------------------------------------------

def _intent(veto_reason):
    return types.SimpleNamespace(direction=0.9, target_exposure=0.3,
                                 veto_active=True, veto_reason=veto_reason)


class TestSleeveClassificationOfGateDecisions:
    def test_reduce_hold_marker_holds(self):
        d, r = sleeve_direction_from_intent(
            _intent("[TRADE_GATE] GATE_SIZE_OR_DELAY_HOLD "
                    "(REDUCE: NONE adjusted_size=0.05)"), 0.0)
        assert d is SLEEVE_HOLD, (
            "a size-reduction advisory flattened the routed book (P265)")

    def test_trade_gate_stale_data_holds(self):
        d, _ = sleeve_direction_from_intent(
            _intent("[TRADE_GATE] STALE_DATA sources=['ticker'] "
                    "ob_stale=True ob_fb= ob_age=412"), 0.0)
        assert d is SLEEVE_HOLD, (
            "the trade gate's own freshness veto is the data-unknown class — "
            "it flattened instead of holding")

    def test_trade_gate_crash_holds(self):
        d, _ = sleeve_direction_from_intent(
            _intent("[TRADE_GATE_ERROR] KeyError: 'x'"), 0.0)
        assert d is SLEEVE_HOLD, (
            "a trade-gate CRASH (no information at all) liquidated the book")

    def test_a_substantive_gate_rejection_still_flattens(self):
        d, r = sleeve_direction_from_intent(
            _intent("[TRADE_GATE] DVOL_EXTREME"), 0.0)
        assert d == 0.0
        assert r.startswith("veto_flat")
