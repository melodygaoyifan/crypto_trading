"""[P162] A venue-routing skip must not be recorded as a veto.

Root cause of the phantom "312 consecutive blocked ticks" alarm:

`_process_4h_tick_inner` treats ANY non-fill from `execute_intent_v2` as a
veto — it sets `veto_active=True` and `target_exposure=0.0` on the intent
*after* execution has already been attempted. Since the 2026-06-13 cutover all
three assets are Coinbase-routed and Kraken-flat, so `execute_intent_v2`
returns

    {"status": "SKIPPED", "reason": "coinbase_routed_no_kraken_entry"}

on EVERY tick by design (P152 — the Coinbase sleeve is the sole directional
driver, the Kraken path correctly declines the entry). The post-execution stamp
then flipped an intent that had passed all seven gate layers into a
non-actionable one, and `PerTickInvariantChecker._t3_intent_actionable` counted
it as blocked. 312/312 ticks. HEALTH_T3 was measuring a retired code path, and
P155/P155b/P155c were spent diagnosing a blocker that did not exist.

The distinction these tests pin: **"this path had nothing to do" is not
"this trade was blocked."** A real veto (risk, margin, rejection, error) must
still latch, still zero exposure, and still block — otherwise this fix would
silently disarm a genuine safety response.
"""

import pytest

from core.execution_service import BENIGN_EXEC_SKIP_REASONS, is_benign_exec_skip
from core.health_validator import PerTickInvariantChecker
from integration.integration_v36 import TradeIntentV36


# --- the exact payloads execute_intent_v2 returns -------------------------
P152_ROUTING_SKIP = {
    "status": "SKIPPED",
    "reason": "coinbase_routed_no_kraken_entry",
    "asset": "SOL",
}
NO_POSITION_SKIP = {"status": "SKIPPED", "reason": "No active position to close"}
VERIFY_MODE_SKIP = {"status": "SKIPPED", "reason": "Verify mode"}
RISK_VETO = {"status": "SKIPPED", "reason": "risk_veto_max_exposure"}
REJECTED = {"status": "REJECTED", "reason": "insufficient_margin"}
ERRORED = {"status": "ERROR", "error_message": "connection reset"}


def _apply_non_fill(intent, exec_result):
    """Mirror of the post-execution branch in main.py (~13527-13560).

    Kept in lockstep with production by construction: it calls the SAME
    `is_benign_exec_skip` classifier the runner calls, so a change to the
    benign set is reflected here automatically. Only the two side-effect
    branches are duplicated.
    """
    reason = str(
        exec_result.get("reason")
        or exec_result.get("error_message")
        or exec_result.get("status")
        or "EXECUTION_SKIPPED"
    )
    if is_benign_exec_skip(exec_result):
        intent.execution_skip_reason = reason
    else:
        intent.veto_active = True
        intent.veto_reason = (
            reason if reason.startswith("[") else f"[EXECUTION] {reason}"
        )
        intent.target_exposure = 0.0
    return intent


def _signalling_intent(**kw):
    """An intent that passed every gate: real direction, real exposure."""
    kw.setdefault("asset", "SOL")
    kw.setdefault("quant_strategy_id", "momentum")
    kw.setdefault("direction", 0.45)
    kw.setdefault("target_exposure", 0.30)
    return TradeIntentV36(**kw)


# --- classifier ------------------------------------------------------------
@pytest.mark.parametrize("payload", [P152_ROUTING_SKIP, NO_POSITION_SKIP])
def test_no_op_paths_are_benign(payload):
    assert is_benign_exec_skip(payload) is True


@pytest.mark.parametrize(
    "payload", [VERIFY_MODE_SKIP, RISK_VETO, REJECTED, ERRORED]
)
def test_real_blocks_are_not_benign(payload):
    """Anything that is not an explicitly-listed no-op must fail closed."""
    assert is_benign_exec_skip(payload) is False


def test_benign_set_requires_skipped_status():
    """A benign *reason* on a non-SKIPPED status is still a real failure."""
    assert is_benign_exec_skip(
        {"status": "ERROR", "reason": "coinbase_routed_no_kraken_entry"}
    ) is False


def test_benign_set_is_explicit_and_small():
    """Guard against the set quietly growing into a blanket suppression.

    Every addition must be a deliberate decision, because each one removes a
    class of non-fill from the veto path and from the T3 blocked-tick alarm.
    """
    assert BENIGN_EXEC_SKIP_REASONS == frozenset(
        {"coinbase_routed_no_kraken_entry", "No active position to close"}
    )


# --- effect on the intent --------------------------------------------------
def test_routing_skip_leaves_intent_actionable():
    """The regression itself: P152 must not zero exposure or latch a veto."""
    intent = _apply_non_fill(_signalling_intent(), P152_ROUTING_SKIP)

    assert intent.veto_active is False
    assert intent.veto_reason == ""
    assert intent.target_exposure == pytest.approx(0.30)
    assert intent.execution_skip_reason == "coinbase_routed_no_kraken_entry"
    assert intent.is_actionable is True


def test_real_veto_still_latches_and_zeroes_exposure():
    """The fix must not disarm genuine safety responses."""
    intent = _apply_non_fill(_signalling_intent(), RISK_VETO)

    assert intent.veto_active is True
    assert intent.veto_reason == "[EXECUTION] risk_veto_max_exposure"
    assert intent.target_exposure == 0.0
    assert intent.execution_skip_reason == ""
    assert intent.is_actionable is False


# --- effect on the HEALTH_T3 streak (the alarm that misfired) --------------
def test_t3_streak_does_not_accumulate_on_routing_skips():
    """312 consecutive routing skips must leave the streak at zero."""
    checker = PerTickInvariantChecker()

    for _ in range(312):
        intent = _apply_non_fill(_signalling_intent(), P152_ROUTING_SKIP)
        check = checker._t3_intent_actionable("SOL", intent)

    assert checker._consecutive_blocked.get("SOL", 0) == 0
    assert check.status == "PASS"


def test_t3_still_escalates_on_real_vetoes():
    """A genuine repeated block must still reach CRITICAL."""
    checker = PerTickInvariantChecker()

    for _ in range(12):
        intent = _apply_non_fill(_signalling_intent(), RISK_VETO)
        check = checker._t3_intent_actionable("SOL", intent)

    assert checker._consecutive_blocked["SOL"] == 12
    assert check.status == "CRITICAL"
