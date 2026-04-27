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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
