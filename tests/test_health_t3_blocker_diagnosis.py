"""[P155] T3 must name WHICH `is_actionable` clause blocked the trade.

Live evidence 2026-08-04:
    [HEALTH_T3] CRITICAL: SOL intent actionable — BLOCKED 312 consecutive ticks
    — system cannot trade! Check VETO_CHAIN logs for root cause.

312 consecutive 4H ticks ≈ 52 days. The streak only accumulates when the
strategy is NOT 'hold' (see `_t3_intent_actionable`), so SOL was producing a
real directional signal every tick and the intent never became actionable. The
alert told the operator to read VETO_CHAIN logs — which are silent when the
actual blocker is a collapsed `target_exposure` or a sub-threshold direction.
That misdirection is why this ran unnoticed for ~7 weeks.

`TradeIntentV36.is_actionable` is a 3-clause conjunction:
    not veto_active AND |direction| > dir_thresh AND target_exposure > exp_thresh
These tests pin that the CRITICAL message names every failing clause.
"""

import pytest

from core.health_validator import PerTickInvariantChecker
from integration.integration_v36 import TradeIntentV36


def _intent(**kw):
    """A blocked-but-signalling intent: non-'hold' strategy, not actionable."""
    kw.setdefault("asset", "SOL")
    kw.setdefault("quant_strategy_id", "mean_revert")
    kw.setdefault("direction", 0.42)
    kw.setdefault("target_exposure", 0.20)
    return TradeIntentV36(**kw)


def _critical_after_streak(intent, n=10, asset="SOL"):
    """Drive the checker to its CRITICAL threshold and return that message."""
    c = PerTickInvariantChecker()
    msg = None
    for _ in range(n):
        msg = c._t3_intent_actionable(asset, intent).detail
    return msg


# ---------------------------------------------------------------------------
# Each clause must be named
# ---------------------------------------------------------------------------

def test_veto_is_named_with_its_reason():
    i = _intent(veto_active=True, veto_reason="CORRELATION_CAP breached")
    assert not i.is_actionable
    msg = _critical_after_streak(i)
    assert "VETO_ACTIVE" in msg
    assert "CORRELATION_CAP breached" in msg


def test_zero_exposure_is_named_not_blamed_on_veto():
    """The case the old message actively misdiagnosed: veto chain totally clean,
    sizing collapsed to ~0, operator sent to read silent VETO_CHAIN logs."""
    i = _intent(target_exposure=0.0)
    assert not i.is_actionable
    msg = _critical_after_streak(i)
    assert "ZERO_EXPOSURE" in msg
    assert "VETO_ACTIVE" not in msg
    assert "target_exposure=0.0000" in msg


def test_weak_direction_is_named():
    i = _intent(direction=0.03)  # below the 0.10 base threshold
    assert not i.is_actionable
    msg = _critical_after_streak(i)
    assert "WEAK_DIRECTION" in msg
    assert "0.0300" in msg


def test_all_failing_clauses_reported_not_just_the_first():
    i = _intent(veto_active=True, veto_reason="r", direction=0.01, target_exposure=0.0)
    msg = _critical_after_streak(i)
    assert "VETO_ACTIVE" in msg
    assert "WEAK_DIRECTION" in msg
    assert "ZERO_EXPOSURE" in msg


# ---------------------------------------------------------------------------
# Threshold mirroring must track is_actionable, not a hardcoded copy
# ---------------------------------------------------------------------------

def test_opportunity_mode_uses_the_relaxed_direction_threshold():
    """dir=0.07 is blocked at the 0.10 base floor but PASSES the 0.05
    OPPORTUNITY floor — the blocker must not claim WEAK_DIRECTION there."""
    i = _intent(direction=0.07, target_exposure=0.0,
                alpha_gate_passed=True, system_mode="OPPORTUNITY")
    msg = _critical_after_streak(i)
    assert "WEAK_DIRECTION" not in msg
    assert "ZERO_EXPOSURE" in msg


def test_opportunity_short_threshold_override_is_honoured():
    i = _intent(direction=-0.07, target_exposure=0.20,
                alpha_gate_passed=True, system_mode="OPPORTUNITY",
                opportunity_actionable_direction_threshold_short=0.20)
    assert not i.is_actionable
    msg = _critical_after_streak(i)
    assert "WEAK_DIRECTION" in msg
    assert "0.2000" in msg  # the override, not the 0.05 default


def test_custom_exposure_threshold_is_honoured():
    i = _intent(target_exposure=0.03, alpha_gate_passed=True,
                system_mode="OPPORTUNITY",
                opportunity_actionable_exposure_threshold=0.05)
    assert not i.is_actionable
    msg = _critical_after_streak(i)
    assert "ZERO_EXPOSURE" in msg


def test_blocker_agrees_with_is_actionable_across_a_grid():
    """No combination may report 'UNEXPLAINED' — that sentinel means the
    blocker logic has drifted away from the is_actionable property."""
    for veto in (True, False):
        for d in (0.0, 0.05, 0.11, -0.42):
            for exp in (0.0, 0.005, 0.30):
                i = _intent(veto_active=veto, direction=d, target_exposure=exp)
                if i.is_actionable:
                    continue
                assert "UNEXPLAINED" not in PerTickInvariantChecker._actionable_blocker(i)


# ---------------------------------------------------------------------------
# Streak semantics + robustness (the checker must never break a tick)
# ---------------------------------------------------------------------------

def test_hold_resets_the_streak_so_312_means_a_real_signal():
    """Pins the interpretation of the live log: a 312 streak cannot be 'hold'."""
    c = PerTickInvariantChecker()
    blocked = _intent(target_exposure=0.0)
    for _ in range(12):
        c._t3_intent_actionable("SOL", blocked)
    assert c._consecutive_blocked["SOL"] == 12

    c._t3_intent_actionable("SOL", _intent(quant_strategy_id="hold", target_exposure=0.0))
    assert c._consecutive_blocked["SOL"] == 0


def test_actionable_intent_resets_and_passes():
    c = PerTickInvariantChecker()
    good = _intent()
    assert good.is_actionable
    assert c._t3_intent_actionable("SOL", good).status == "PASS"
    assert c._consecutive_blocked["SOL"] == 0


def test_warn_tier_also_carries_the_diagnosis():
    """5-9 ticks is where this should be caught — the WARN must be actionable too."""
    c = PerTickInvariantChecker()
    i = _intent(target_exposure=0.0)
    for _ in range(5):
        r = c._t3_intent_actionable("SOL", i)
    assert r.status == "WARN"
    assert "ZERO_EXPOSURE" in r.detail


@pytest.mark.parametrize("bad", [None, object()])
def test_never_raises_on_a_malformed_intent(bad):
    out = PerTickInvariantChecker._actionable_blocker(bad)
    assert isinstance(out, str) and out
