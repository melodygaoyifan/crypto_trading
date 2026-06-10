"""
test_chaos_state_corruption.py — state-write/restore corruption scenarios
==========================================================================

Recreates the P85 cascade conditions: SIGKILL-truncated state files,
corrupt JSON on restore, missing fields after dataclass schema drift.

Asserts the system either:
  (a) gracefully degrades (load failure → safe defaults + WARN), OR
  (b) fails-CLOSED with explicit operator-action message

NEVER:
  (x) silently loads partial/wrong state
  (y) crashes the process (would compose with restart: always → cascade)
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.chaos.harness import (
    captured_logs, temp_state_dir, write_corrupt_json,
    assert_warn_in_logs,
)


# =====================================================================
# Scenario 4: AutoRecoveryGate corrupt state file (P92 regression)
# =====================================================================

class TestChaosAutoRecoveryGateCorruption:
    """P92: corrupt halt-state file MUST synthesize a halt
    (STATE_CORRUPTION_DETECTED) instead of silently treating it as
    'no halt = trading allowed.' Pre-P92: empty HaltState returned."""

    def test_truncated_json_synthesizes_halt(self):
        from risk.auto_recovery_gate import AutoRecoveryGate, AutoRecoveryConfig
        with temp_state_dir() as tmp:
            corrupt = tmp / "halt.json"
            write_corrupt_json(corrupt)  # truncated mid-write

            with captured_logs() as logs:
                gate = AutoRecoveryGate(
                    config=AutoRecoveryConfig(),
                    state_path=corrupt,
                )

            # State must be synthesized halt, not empty
            assert gate._state.halt_reason == "STATE_CORRUPTION_DETECTED", (
                f"P92 regression: corrupt state silently loaded as empty. "
                f"halt_reason = {gate._state.halt_reason!r}"
            )
            assert gate._state.halt_since_ts > 0, (
                "Synthesized halt should have non-zero timestamp"
            )
            # CRITICAL log fired
            assert_warn_in_logs(logs, "STATE_CORRUPTION_DETECTED",
                                context="P92 should log CRITICAL on corruption")

    def test_missing_state_file_returns_empty(self):
        """No file = fresh start (correct), not corruption."""
        from risk.auto_recovery_gate import AutoRecoveryGate, AutoRecoveryConfig
        with temp_state_dir() as tmp:
            never_existed = tmp / "no_such_file.json"
            gate = AutoRecoveryGate(
                config=AutoRecoveryConfig(),
                state_path=never_existed,
            )
            assert gate._state.halt_reason == "", (
                "Missing file should return empty HaltState (fresh start), "
                f"not synthesized halt. Got: {gate._state.halt_reason!r}"
            )


# =====================================================================
# Scenario 5: state_persistence atomic write under SIGKILL race
# =====================================================================

class TestChaosStatePersistenceAtomic:
    """P83: state_persistence.save_state must atomic-rename + fsync.
    If process dies between write and rename, partial file must NOT
    be loadable as valid state."""

    def test_truncated_state_file_load_safe(self):
        """If a previous save was killed mid-write (truncated tmp file),
        load_state must return None / safe-default, not crash with
        JSONDecodeError."""
        from core.state_persistence import load_state
        with temp_state_dir() as tmp:
            corrupt = tmp / "state.json"
            write_corrupt_json(corrupt, '{"a": 1, "b":')  # truncated

            # Should not crash
            try:
                result = load_state(str(corrupt))
            except json.JSONDecodeError as e:
                pytest.fail(
                    f"load_state raised JSONDecodeError instead of returning "
                    f"safe default: {e}. Caller would crash → cascade."
                )
            # Either None or empty dict — both safe
            assert result is None or result == {} or isinstance(result, dict), (
                f"load_state returned unexpected type {type(result)}: {result}"
            )

    def test_atomic_save_then_load_roundtrip(self):
        """Sanity: P83 atomic write should be lossless."""
        from core.state_persistence import save_state, load_state
        with temp_state_dir() as tmp:
            target = tmp / "roundtrip.json"
            payload = {"key": "value", "num": 42, "list": [1, 2, 3]}
            save_state(str(target), payload)
            assert target.exists(), "save_state didn't write the file"
            loaded = load_state(str(target))
            assert loaded == payload, (
                f"Roundtrip mismatch: saved {payload}, loaded {loaded}"
            )


# =====================================================================
# Scenario 6: Dataclass schema drift on restore (missing field)
# =====================================================================

class TestChaosSchemaDriftOnRestore:
    """If a saved state file was written with an older dataclass schema
    that's missing a field added in a newer version, restoration must
    NOT crash. Either fill in the missing field with a default, or fail
    loudly with explicit message."""

    def test_partial_state_missing_new_field(self):
        """Simulate: state file written when ConditionStatus had only
        2 fields; current code expects 5. Restoration must degrade."""
        from risk.regime_transition_buffer import (
            RegimeTransitionBuffer, ConditionStatus,
        )
        with temp_state_dir() as tmp:
            # Hand-craft a partial state from "old" schema
            partial = {
                "state": "INACTIVE",
                "state_entry_time": None,
                # Missing: any newer fields added post-creation
            }
            # Should not crash on from_dict / restore
            buf = RegimeTransitionBuffer()
            try:
                if hasattr(buf, "from_dict"):
                    buf.from_dict(partial)
            except (TypeError, KeyError) as e:
                pytest.fail(
                    f"from_dict crashed on partial schema: {type(e).__name__}: "
                    f"{e}. Expected graceful degradation."
                )

    def test_corrupt_value_type_in_state(self):
        """If state has wrong type (string where int expected), restore
        must not crash with TypeError."""
        from core.state_persistence import load_state
        with temp_state_dir() as tmp:
            target = tmp / "wrong_types.json"
            target.write_text(
                '{"counter": "not_a_number", "active": "not_a_bool"}',
                encoding="utf-8",
            )
            # load_state itself returns dict — type coercion is caller's job.
            # Verify the load doesn't crash:
            result = load_state(str(target))
            assert isinstance(result, dict), (
                f"load_state should return dict even on wrong-type values, "
                f"got {type(result)}"
            )


# =====================================================================
# Scenario 7: RestartCount cascade (the actual P85 incident shape)
# =====================================================================

class TestChaosRestartCascade:
    """P85: missing self.shadow_ledger.frozen_allocations attribute
    caused 10 container restarts in 6 minutes. The fix made the read
    defensive (getattr with default + WARN). Verify the pattern is
    still in place."""

    def test_startup_reconciler_defensive_read_present(self):
        """Verify the P85 defensive getattr pattern wasn't removed."""
        import inspect
        try:
            from defense import startup_reconciler
            src = inspect.getsource(startup_reconciler)
        except (ImportError, OSError):
            pytest.skip("startup_reconciler not importable")
        # The P85 fix added getattr(self.shadow_ledger, 'frozen_allocations', None)
        assert ("getattr" in src and "frozen_allocations" in src), (
            "P85 defensive pattern removed: getattr(self.shadow_ledger, "
            "'frozen_allocations', ...) missing from startup_reconciler. "
            "Cascade can recur if shadow_ledger writer breaks."
        )

    def test_startup_reconciler_no_sys_exit_on_attr_error(self):
        """P85 lesson: never sys.exit() on missing internal attribute,
        composes with restart: always to weaponize."""
        import inspect
        try:
            from defense import startup_reconciler
            src = inspect.getsource(startup_reconciler)
        except (ImportError, OSError):
            pytest.skip()
        # No raise SystemExit / sys.exit() in the reconciler
        assert "sys.exit" not in src and "SystemExit" not in src, (
            "P85 lesson violated: startup_reconciler contains sys.exit "
            "or raise SystemExit. This composes with docker restart: "
            "always to amplify a single attribute error into a 10-restart "
            "outage loop."
        )


# =====================================================================
# Scenario 7b: P85 architectural follow-up — frozen_allocations + replay
# =====================================================================

class TestP85ArchitecturalFrozenAllocations:
    """P85 architectural (2026-06-09): the proper fix for the original
    P85 cascade — ShadowLedgerWriter now exposes frozen_allocations as
    a Set[str] populated by record_order, emptied by record_fill /
    release_order, and seedable from JSONL history via
    replay_frozen_allocations_from_jsonl().

    The defensive getattr in startup_reconciler stays as belt-and-
    suspenders (older module versions, replay failures) but the
    architectural contract is now satisfied: a fresh process can
    actually classify orphan exchange orders correctly after a
    one-call replay.
    """

    def _fresh_writer(self):
        import tempfile
        from defense.shadow_ledger_jsonl import ShadowLedgerWriter
        tmp = tempfile.mkdtemp()
        sl = ShadowLedgerWriter(output_dir=tmp, auto_flush=False)
        return sl, tmp

    def test_frozen_allocations_attribute_exists(self):
        """The architectural goal — attribute is present and is a Set."""
        sl, tmp = self._fresh_writer()
        try:
            assert hasattr(sl, 'frozen_allocations'), (
                "P85 architectural follow-up reverted: ShadowLedgerWriter "
                "no longer exposes frozen_allocations. The defensive "
                "getattr in startup_reconciler degrades to skip-orphan-"
                "cancel which is safe but loses the orphan-detection "
                "feature."
            )
            assert isinstance(sl.frozen_allocations, set), (
                f"frozen_allocations must be a Set, got {type(sl.frozen_allocations)}"
            )
            assert hasattr(sl, '_frozen_allocations_replayed'), (
                "replay flag missing — reconciler can't tell whether the "
                "set is authoritative or unseeded."
            )
            assert sl._frozen_allocations_replayed is False, (
                "fresh process must NOT claim replay-complete by default; "
                "otherwise reconciler treats an empty set as authoritative "
                "and cancels every legitimate order."
            )
        finally:
            sl.close()
            import shutil; shutil.rmtree(tmp, ignore_errors=True)

    def test_record_order_populates_set(self):
        sl, tmp = self._fresh_writer()
        try:
            sl.record_order(asset='SOL', order_id='ORD-1', side='BUY',
                            size=1.0, order_type='LIMIT')
            sl.record_order(asset='BTC', order_id='ORD-2', side='SELL',
                            size=0.01, order_type='MARKET')
            assert sl.frozen_allocations == {'ORD-1', 'ORD-2'}, (
                f"record_order should add to frozen_allocations; "
                f"got {sl.frozen_allocations}"
            )
        finally:
            sl.close()
            import shutil; shutil.rmtree(tmp, ignore_errors=True)

    def test_record_fill_removes_from_set(self):
        sl, tmp = self._fresh_writer()
        try:
            sl.record_order(asset='SOL', order_id='ORD-1', side='BUY',
                            size=1.0, order_type='LIMIT')
            sl.record_fill(asset='SOL', order_id='ORD-1', fill_id='F-1',
                           side='BUY', size=1.0, price=63.0)
            assert 'ORD-1' not in sl.frozen_allocations, (
                "record_fill must release the order_id; otherwise the "
                "reconciler counts filled orders as outstanding."
            )
        finally:
            sl.close()
            import shutil; shutil.rmtree(tmp, ignore_errors=True)

    def test_release_order_explicit_cancel(self):
        sl, tmp = self._fresh_writer()
        try:
            sl.record_order(asset='SOL', order_id='ORD-1', side='BUY',
                            size=1.0, order_type='LIMIT')
            assert sl.release_order('ORD-1') is True, "should report tracked"
            assert sl.frozen_allocations == set()
            assert sl.release_order('ORD-1') is False, "idempotent release"
            assert sl.release_order('') is False, "empty id is no-op"
            assert sl.release_order(None) is False, "None id is no-op"
        finally:
            sl.close()
            import shutil; shutil.rmtree(tmp, ignore_errors=True)

    def test_falsy_order_id_does_not_pollute_set(self):
        """Paper-mode synthetic flows may emit None/'' order_ids. Tracking
        them would corrupt the set (every None collapses to one entry)
        and confuse the reconciler."""
        sl, tmp = self._fresh_writer()
        try:
            sl.record_order(asset='SOL', order_id=None, side='BUY',
                            size=1.0, order_type='LIMIT')
            sl.record_order(asset='SOL', order_id='', side='BUY',
                            size=1.0, order_type='LIMIT')
            assert sl.frozen_allocations == set(), (
                "falsy order_id leaked into frozen_allocations"
            )
        finally:
            sl.close()
            import shutil; shutil.rmtree(tmp, ignore_errors=True)

    def test_replay_seeds_from_jsonl(self):
        """The replay closes the original P85 cascade gap: a fresh
        process can rebuild outstanding-order state from JSONL history."""
        from defense.shadow_ledger_jsonl import ShadowLedgerWriter
        import tempfile, shutil

        tmp = tempfile.mkdtemp()
        try:
            # Process 1: open + fill some orders
            sl1 = ShadowLedgerWriter(output_dir=tmp, auto_flush=False)
            sl1.record_order(asset='SOL', order_id='ORD-A', side='BUY',
                             size=1.0, order_type='LIMIT')
            sl1.record_order(asset='BTC', order_id='ORD-B', side='SELL',
                             size=0.01, order_type='MARKET')
            sl1.record_fill(asset='BTC', order_id='ORD-B', fill_id='F-B',
                            side='SELL', size=0.01, price=77000.0)
            sl1.flush()
            sl1.close()

            # Process 2: fresh init — empty by default, replay seeds
            sl2 = ShadowLedgerWriter(output_dir=tmp, auto_flush=False)
            assert sl2.frozen_allocations == set(), "fresh start must be empty"
            assert sl2._frozen_allocations_replayed is False

            n = sl2.replay_frozen_allocations_from_jsonl(days_back=1)
            assert sl2._frozen_allocations_replayed is True, (
                "replay must flip the seeded flag — reconciler depends on it"
            )
            assert 'ORD-A' in sl2.frozen_allocations, (
                f"unfilled ORD-A should be replayed as outstanding: "
                f"{sl2.frozen_allocations}"
            )
            assert 'ORD-B' not in sl2.frozen_allocations, (
                f"filled ORD-B should NOT be replayed: {sl2.frozen_allocations}"
            )
            assert n == 1
            sl2.close()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_reconciler_uses_replay_before_orphan_classification(self):
        """Source-level assertion: startup_reconciler must call
        replay_frozen_allocations_from_jsonl AND must gate orphan
        cancellation on _frozen_allocations_replayed. Otherwise a
        fresh process classifies every legitimate exchange order as
        orphan and cancels them all — worse than P85."""
        import inspect
        from defense import startup_reconciler
        src = inspect.getsource(startup_reconciler)
        assert 'replay_frozen_allocations_from_jsonl' in src, (
            "P85-arch: startup_reconciler does not call replay before "
            "classifying orphans. A fresh process would cancel every "
            "legitimate open order at startup."
        )
        assert '_frozen_allocations_replayed' in src, (
            "P85-arch: reconciler must check the replayed flag. Without "
            "it, an empty (unseeded) set looks the same as 'no outstanding "
            "orders' — every exchange order looks orphan."
        )


# =====================================================================
# Scenario 8: Margin-position close paths must plumb leverage (P138)
# =====================================================================

class TestChaosMarginClosePathLeverage:
    """P138: spot/margin instrument mismatch in close paths.

    A SHORT position can only exist on Kraken via margin. The entry path
    stores `regime_leverage` on `_paper_positions[asset]` and passes
    `leverage=` to `execute_order`. The three OFF-BAND watchdog close
    paths (FastRiskTick exit, CORR-0 reduction, EMERGENCY_FLAT) used to
    omit leverage entirely, routing margin closes as spot orders that
    (a) couldn't actually net the margin position and (b) failed the
    spot-balance clamp when free quote was low.

    These assertions pin the plumbing in place so a future refactor
    can't silently strip it again.
    """

    def _runner_source(self):
        """Read main.py once; close paths all live here."""
        from pathlib import Path
        p = Path(__file__).resolve().parents[2] / "main.py"
        if not p.exists():
            pytest.skip("main.py not at expected location")
        # main.py has a BOM; use utf-8-sig
        return p.read_text(encoding="utf-8-sig")

    def _exec_mgr_source(self):
        import inspect
        try:
            from execution import execution_manager
        except ImportError:
            pytest.skip("execution_manager not importable")
        return inspect.getsource(execution_manager)

    def test_fast_risk_tick_close_reads_regime_leverage(self):
        """_handle_fast_risk_action must read regime_leverage from pos
        and pass it to execute_order. Otherwise margin shorts get spot
        BUY-to-close which leaves the short open."""
        src = self._runner_source()
        # Find the FastRiskTick handler block
        idx = src.find("def _handle_fast_risk_action")
        assert idx >= 0, "_handle_fast_risk_action not found in main.py"
        # Search forward to the next def to bound the method body
        body_end = src.find("\n    async def ", idx + 1)
        if body_end < 0:
            body_end = src.find("\n    def ", idx + 1)
        body = src[idx:body_end if body_end > 0 else idx + 8000]

        assert 'pos.get("regime_leverage"' in body or "pos.get('regime_leverage'" in body, (
            "P138 plumbing missing: _handle_fast_risk_action does not read "
            "regime_leverage from the position. Margin shorts will close as "
            "spot orders and the watchdog will alert-storm until P110 backoff."
        )
        assert "leverage=" in body, (
            "P138 plumbing missing: _handle_fast_risk_action does not pass "
            "leverage= to execute_order."
        )

    def test_crisis_reduction_close_reads_regime_leverage(self):
        """_crisis_position_reduction (CORR-0) must plumb leverage the
        same way as FastRiskTick. Same bug shape, same fix."""
        src = self._runner_source()
        idx = src.find("def _crisis_position_reduction")
        assert idx >= 0, "_crisis_position_reduction not found in main.py"
        body_end = src.find("\n    async def ", idx + 1)
        if body_end < 0:
            body_end = src.find("\n    def ", idx + 1)
        body = src[idx:body_end if body_end > 0 else idx + 8000]

        assert 'pos.get("regime_leverage"' in body or "pos.get('regime_leverage'" in body, (
            "P138 plumbing missing: _crisis_position_reduction does not read "
            "regime_leverage from the position."
        )
        assert "leverage=" in body, (
            "P138 plumbing missing: _crisis_position_reduction does not pass "
            "leverage= to execute_order."
        )

    def test_emergency_flatten_close_reads_regime_leverage(self):
        """_emergency_flatten / trigger_emergency_flatten DEAD_MAN_SWITCH
        path must also plumb leverage. If the dead-man fires on a margin
        portfolio, spot-only flatten orders would leave shorts open."""
        src = self._runner_source()
        # The actual close block is in _emergency_flatten / trigger_emergency_flatten
        # — search for the EMERGENCY_FLAT tick_id marker (unique).
        idx = src.find('tick_id="EMERGENCY_FLAT"')
        assert idx >= 0, "EMERGENCY_FLAT close block not found in main.py"
        # Pull a window around it to inspect
        body = src[max(0, idx - 1500):idx + 500]

        assert 'pos.get("regime_leverage"' in body or "pos.get('regime_leverage'" in body, (
            "P138 plumbing missing: _emergency_flatten does not read "
            "regime_leverage from the position. DEAD_MAN_SWITCH on a margin "
            "portfolio will issue spot close orders that don't net the shorts."
        )
        assert "leverage=" in body, (
            "P138 plumbing missing: _emergency_flatten does not pass "
            "leverage= to execute_order."
        )

    def test_clamp_skips_when_leverage_gt_1(self):
        """_clamp_size_to_balance_v2 must short-circuit (return size, '')
        when leverage > 1. Spot balance is the wrong constraint for a
        margin order — Kraken validates collateral server-side."""
        from unittest.mock import MagicMock
        from execution.execution_manager import ExecutionManager

        em = ExecutionManager.__new__(ExecutionManager)
        em.exchange = MagicMock()
        em.exchange.market.return_value = {"limits": {"amount": {"min": 0.02}}}
        # Mimic the production incident: $0.12 USDT free, trying to BUY 12 SOL
        em.exchange.fetch_balance.return_value = {
            "free": {"USDT": 0.12}, "used": {"USDT": 0.0}
        }
        em.logger = MagicMock()

        class _BuySide:
            value = "BUY"

        # leverage > 1 → skip clamp, pass full size through
        size, msg = em._clamp_size_to_balance_v2(
            "SOL/USDT", _BuySide(), 12.0, 63.0, MagicMock(), leverage=2
        )
        assert size == 12.0, (
            f"P138 regression: clamp returned {size} for leverage=2 instead "
            f"of original 12.0. Margin closes will be wrongly clamped to dust."
        )
        assert msg == "", (
            f"P138 regression: clamp emitted message {msg!r} for leverage>1. "
            f"Should be empty (no clamp applied)."
        )

        # leverage=None (spot) → still rejects on $0.12 USDT
        size_spot, msg_spot = em._clamp_size_to_balance_v2(
            "SOL/USDT", _BuySide(), 12.0, 63.0, MagicMock(), leverage=None
        )
        assert size_spot == 0.0, (
            "P138 contract violated: spot path (leverage=None) must still "
            "reject when free quote is below dust threshold."
        )
        assert "USDT" in msg_spot, (
            f"Spot rejection message missing USDT context: {msg_spot!r}"
        )

    def test_execute_order_forwards_leverage_to_clamp(self):
        """execute_order must call _clamp_size_to_balance_v2 with leverage=
        kwarg. Otherwise the clamp can't tell margin from spot and the
        skip-path is dead code."""
        src = self._exec_mgr_source()
        # The call to the clamp from execute_order
        idx = src.find("self._clamp_size_to_balance_v2(")
        assert idx >= 0, "_clamp_size_to_balance_v2 call site not found"
        # Read ~6 lines after to inspect kwargs
        snippet = src[idx:idx + 200]
        assert "leverage=leverage" in snippet, (
            "P138 plumbing missing: execute_order does not pass leverage to "
            "_clamp_size_to_balance_v2. Without this, the skip-when-margin "
            "logic is unreachable."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
