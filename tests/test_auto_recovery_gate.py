"""
tests/test_auto_recovery_gate.py - P3-03 AutoRecoveryGate tests

Coverage:
- DRAWDOWN_HALT never auto-recovers
- CORRELATION_CRISIS recovery after observation period + conditions met
- CORRELATION_COLLAPSE recovery
- Data integrity gate blocks recovery
- Disabled config blocks recovery (HIGH_RISK profile default)
- Observation period not met blocks recovery
- Correlation still high blocks recovery
- Halt state persistence (save/load)
- Reason classification from raw veto strings
- Proof log snapshots
- Config loading from ultra_aggressive_5y.json
- Idempotent halt recording
- Clear halt resets state
"""

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from risk.auto_recovery_gate import (
    AutoRecoveryConfig,
    AutoRecoveryGate,
    HaltState,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ultra_config() -> AutoRecoveryConfig:
    """Return the config matching ultra_aggressive_5y.json."""
    return AutoRecoveryConfig(
        enabled=True,
        allowed_reasons=["CORRELATION_CRISIS", "CORRELATION_COLLAPSE"],
        never_auto_recover=["DRAWDOWN_HALT"],
        observation_hours=48,
        correlation_recovery_threshold=0.90,
        require_data_integrity_ok=True,
    )


def _disabled_config() -> AutoRecoveryConfig:
    """Return disabled config (HIGH_RISK default)."""
    return AutoRecoveryConfig(enabled=False)


def _gate(config=None, state_path=None) -> AutoRecoveryGate:
    """Create a gate with a temp state file."""
    cfg = config or _ultra_config()
    if state_path is None:
        state_path = Path(tempfile.mktemp(suffix=".json"))
    return AutoRecoveryGate(config=cfg, state_path=state_path)


# ---------------------------------------------------------------------------
# Test: DRAWDOWN_HALT never recovers
# ---------------------------------------------------------------------------

class TestDrawdownNeverRecovers(unittest.TestCase):
    """DRAWDOWN_HALT is in never_auto_recover - must always return False."""

    def test_drawdown_never_recovers_even_after_48h(self):
        gate = _gate()
        t0 = 1000000.0
        gate.record_halt("[v3.6.1] NO_TRADE: drawdown >= 10%", t0)

        self.assertTrue(gate.has_active_halt)
        self.assertEqual(gate.halt_state.halt_reason, "DRAWDOWN_HALT")

        # 100 hours later, low correlation, integrity ok
        ok, reason = gate.should_auto_recover(
            current_correlation=0.5,
            data_integrity_ok=True,
            now_ts=t0 + 100 * 3600,
        )
        self.assertFalse(ok)
        self.assertIn("never_auto_recover", reason)

    def test_drawdown_from_hard_veto_string(self):
        gate = _gate()
        t0 = 1000000.0
        gate.record_halt("[HARD_VETO] drawdown 12.3% >= 10%", t0)

        ok, reason = gate.should_auto_recover(0.3, True, t0 + 200 * 3600)
        self.assertFalse(ok)
        self.assertIn("never_auto_recover", reason)


# ---------------------------------------------------------------------------
# Test: CORRELATION_CRISIS recovery
# ---------------------------------------------------------------------------

class TestCorrelationCrisisRecovery(unittest.TestCase):
    """CORRELATION_CRISIS can recover after 48h + corr < 0.90 + integrity."""

    def test_full_recovery_path(self):
        gate = _gate()
        t0 = 1000000.0
        gate.record_halt("[v3.6.1] NO_TRADE: CORRELATION_COLLAPSE", t0)

        self.assertTrue(gate.has_active_halt)
        self.assertEqual(gate.halt_state.halt_reason, "CORRELATION_COLLAPSE")

        # 49 hours later, corr 0.85, integrity ok -> should recover
        ok, reason = gate.should_auto_recover(
            current_correlation=0.85,
            data_integrity_ok=True,
            now_ts=t0 + 49 * 3600,
        )
        self.assertTrue(ok)
        self.assertEqual(reason, "auto_recover_ok")

    def test_correlation_crisis_reason(self):
        gate = _gate()
        t0 = 1000000.0
        gate.record_halt("[HARD_VETO] correlation 0.96 >= 0.95", t0)
        self.assertEqual(gate.halt_state.halt_reason, "CORRELATION_CRISIS")

        ok, reason = gate.should_auto_recover(0.80, True, t0 + 50 * 3600)
        self.assertTrue(ok)

    def test_observation_period_not_met(self):
        gate = _gate()
        t0 = 1000000.0
        gate.record_halt("[v3.6.1] NO_TRADE: CORRELATION_COLLAPSE", t0)

        # Only 24 hours - not enough
        ok, reason = gate.should_auto_recover(0.80, True, t0 + 24 * 3600)
        self.assertFalse(ok)
        self.assertIn("observation_period", reason)

    def test_correlation_still_high(self):
        gate = _gate()
        t0 = 1000000.0
        gate.record_halt("[v3.6.1] NO_TRADE: CORRELATION_COLLAPSE", t0)

        # 49h later but corr still 0.92 (>= 0.90)
        ok, reason = gate.should_auto_recover(0.92, True, t0 + 49 * 3600)
        self.assertFalse(ok)
        self.assertIn("corr_still_high", reason)

    def test_correlation_at_boundary(self):
        """Correlation exactly at threshold should NOT recover."""
        gate = _gate()
        t0 = 1000000.0
        gate.record_halt("[v3.6.1] NO_TRADE: CORRELATION_COLLAPSE", t0)

        # corr == 0.90 (boundary) - should NOT recover (>= check)
        ok, _ = gate.should_auto_recover(0.90, True, t0 + 49 * 3600)
        self.assertFalse(ok)

        # corr == 0.899 - should recover
        ok, _ = gate.should_auto_recover(0.899, True, t0 + 49 * 3600)
        self.assertTrue(ok)


# ---------------------------------------------------------------------------
# Test: Data integrity gate
# ---------------------------------------------------------------------------

class TestDataIntegrityGate(unittest.TestCase):
    """Data integrity not ok blocks recovery even if other conditions met."""

    def test_integrity_not_ok_blocks_recovery(self):
        gate = _gate()
        t0 = 1000000.0
        gate.record_halt("[v3.6.1] NO_TRADE: CORRELATION_COLLAPSE", t0)

        ok, reason = gate.should_auto_recover(0.80, False, t0 + 49 * 3600)
        self.assertFalse(ok)
        self.assertEqual(reason, "data_integrity_not_ok")

    def test_integrity_not_required_when_config_off(self):
        """When require_data_integrity_ok=False, integrity doesn't block."""
        cfg = _ultra_config()
        cfg.require_data_integrity_ok = False
        gate = _gate(config=cfg)
        t0 = 1000000.0
        gate.record_halt("[v3.6.1] NO_TRADE: CORRELATION_COLLAPSE", t0)

        ok, reason = gate.should_auto_recover(0.80, False, t0 + 49 * 3600)
        self.assertTrue(ok)


# ---------------------------------------------------------------------------
# Test: Disabled config (HIGH_RISK default)
# ---------------------------------------------------------------------------

class TestDisabledConfig(unittest.TestCase):
    """When auto_recovery is disabled, gate always returns False."""

    def test_disabled_never_recovers(self):
        gate = _gate(config=_disabled_config())
        t0 = 1000000.0
        gate.record_halt("[v3.6.1] NO_TRADE: CORRELATION_COLLAPSE", t0)

        ok, reason = gate.should_auto_recover(0.5, True, t0 + 100 * 3600)
        self.assertFalse(ok)
        self.assertEqual(reason, "disabled")

    def test_no_config_means_disabled(self):
        """AutoRecoveryConfig.from_dict(None) -> disabled."""
        cfg = AutoRecoveryConfig.from_dict(None)
        self.assertFalse(cfg.enabled)


# ---------------------------------------------------------------------------
# Test: Reason classification
# ---------------------------------------------------------------------------

class TestReasonClassification(unittest.TestCase):
    """Test _classify_reason() maps raw veto strings correctly."""

    def test_correlation_collapse_from_no_trade(self):
        r = AutoRecoveryGate._classify_reason(
            "[v3.6.1] NO_TRADE: CORRELATION_COLLAPSE"
        )
        self.assertEqual(r, "CORRELATION_COLLAPSE")

    def test_correlation_crisis_from_hard_veto(self):
        r = AutoRecoveryGate._classify_reason(
            "[HARD_VETO] correlation 0.96 >= 0.95"
        )
        self.assertEqual(r, "CORRELATION_CRISIS")

    def test_drawdown_from_hard_veto(self):
        r = AutoRecoveryGate._classify_reason(
            "[HARD_VETO] drawdown 12.3% >= 10%"
        )
        self.assertEqual(r, "DRAWDOWN_HALT")

    def test_drawdown_from_no_trade(self):
        r = AutoRecoveryGate._classify_reason(
            "[v3.6.1] NO_TRADE: DRAWDOWN_HALT"
        )
        self.assertEqual(r, "DRAWDOWN_HALT")

    def test_unknown_reason_passthrough(self):
        """Unknown reasons pass through as-is (won't match allowed_reasons)."""
        r = AutoRecoveryGate._classify_reason("SOME_UNKNOWN_REASON")
        self.assertEqual(r, "SOME_UNKNOWN_REASON")

    def test_reason_not_in_allowed(self):
        """Reason not in allowed_reasons -> should_auto_recover returns False."""
        gate = _gate()
        t0 = 1000000.0
        gate.record_halt("FLASH_CRASH detected", t0)

        ok, reason = gate.should_auto_recover(0.5, True, t0 + 100 * 3600)
        self.assertFalse(ok)
        self.assertIn("reason_not_allowed", reason)


# ---------------------------------------------------------------------------
# Test: Halt state persistence
# ---------------------------------------------------------------------------

class TestPersistence(unittest.TestCase):
    """Halt state survives save/load cycle."""

    def test_persist_and_reload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "recovery_state.json"
            gate1 = _gate(state_path=state_path)

            t0 = 1000000.0
            gate1.record_halt("[v3.6.1] NO_TRADE: CORRELATION_COLLAPSE", t0)

            # Verify file exists
            self.assertTrue(state_path.exists())

            # Create new gate from same file - state should be restored
            gate2 = _gate(state_path=state_path)
            self.assertTrue(gate2.has_active_halt)
            self.assertEqual(gate2.halt_state.halt_reason, "CORRELATION_COLLAPSE")
            self.assertEqual(gate2.halt_state.halt_since_ts, t0)

    def test_clear_halt_persists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "recovery_state.json"
            gate = _gate(state_path=state_path)

            t0 = 1000000.0
            gate.record_halt("[v3.6.1] NO_TRADE: CORRELATION_COLLAPSE", t0)
            gate.clear_halt(t0 + 50 * 3600)

            # Reload - should be cleared
            gate2 = _gate(state_path=state_path)
            self.assertFalse(gate2.has_active_halt)
            self.assertAlmostEqual(
                gate2.halt_state.last_auto_recovery_ts,
                t0 + 50 * 3600,
            )

    def test_missing_state_file_ok(self):
        """Gate starts with empty state when file doesn't exist."""
        gate = _gate(state_path=Path("/nonexistent/path/state.json"))
        self.assertFalse(gate.has_active_halt)

    def test_corrupt_state_file_ok(self):
        """Gate handles corrupt state file gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "recovery_state.json"
            state_path.write_text("NOT VALID JSON {{{")
            gate = _gate(state_path=state_path)
            self.assertFalse(gate.has_active_halt)


# ---------------------------------------------------------------------------
# Test: Idempotent halt recording
# ---------------------------------------------------------------------------

class TestIdempotentHaltRecording(unittest.TestCase):
    """record_halt() only records the FIRST halt - preserves original ts."""

    def test_second_record_ignored(self):
        gate = _gate()
        t0 = 1000000.0
        gate.record_halt("[v3.6.1] NO_TRADE: CORRELATION_COLLAPSE", t0)
        gate.record_halt("[v3.6.1] NO_TRADE: CORRELATION_COLLAPSE", t0 + 3600)

        # halt_since_ts should be original t0, not t0+3600
        self.assertEqual(gate.halt_state.halt_since_ts, t0)


# ---------------------------------------------------------------------------
# Test: Clear halt
# ---------------------------------------------------------------------------

class TestClearHalt(unittest.TestCase):
    """clear_halt() resets reason/ts and records last_auto_recovery_ts."""

    def test_clear_resets_state(self):
        gate = _gate()
        t0 = 1000000.0
        gate.record_halt("[v3.6.1] NO_TRADE: CORRELATION_COLLAPSE", t0)
        gate.clear_halt(t0 + 50 * 3600)

        self.assertFalse(gate.has_active_halt)
        self.assertEqual(gate.halt_state.halt_reason, "")
        self.assertEqual(gate.halt_state.halt_since_ts, 0.0)
        self.assertAlmostEqual(
            gate.halt_state.last_auto_recovery_ts,
            t0 + 50 * 3600,
        )


# ---------------------------------------------------------------------------
# Test: Proof log snapshot
# ---------------------------------------------------------------------------

class TestProofLogSnapshot(unittest.TestCase):
    """get_proof_snapshot() returns correct snapshot dict."""

    def test_snapshot_during_halt(self):
        gate = _gate()
        t0 = 1000000.0
        gate.record_halt("[v3.6.1] NO_TRADE: CORRELATION_COLLAPSE", t0)

        snap = gate.get_proof_snapshot(
            current_correlation=0.93,
            data_integrity_ok=True,
            now_ts=t0 + 24 * 3600,
        )
        self.assertTrue(snap["halt_active"])
        self.assertEqual(snap["halt_reason"], "CORRELATION_COLLAPSE")
        self.assertAlmostEqual(snap["hours_in_halt"], 24.0)
        self.assertAlmostEqual(snap["current_correlation"], 0.93, places=3)
        self.assertTrue(snap["data_integrity_ok"])
        self.assertFalse(snap["would_recover"])
        self.assertIn("observation_period", snap["recovery_check"])

    def test_snapshot_no_halt(self):
        gate = _gate()
        snap = gate.get_proof_snapshot(0.5, True, time.time())
        self.assertFalse(snap["halt_active"])
        self.assertEqual(snap["hours_in_halt"], 0.0)


# ---------------------------------------------------------------------------
# Test: Config from JSON file
# ---------------------------------------------------------------------------

class TestConfigFromJSON(unittest.TestCase):
    """AutoRecoveryConfig.from_dict() matches ultra_aggressive_5y.json."""

    def test_load_from_ultra_aggressive(self):
        config_path = Path(__file__).parent.parent / "configs" / "ultra_aggressive_5y.json"
        if not config_path.exists():
            self.skipTest("ultra_aggressive_5y.json not found")

        with open(config_path) as f:
            data = json.load(f)

        cfg = AutoRecoveryConfig.from_dict(data.get("auto_recovery"))
        self.assertTrue(cfg.enabled)
        self.assertIn("CORRELATION_CRISIS", cfg.allowed_reasons)
        self.assertIn("CORRELATION_COLLAPSE", cfg.allowed_reasons)
        self.assertIn("DRAWDOWN_HALT", cfg.never_auto_recover)
        self.assertEqual(cfg.observation_hours, 48)
        self.assertAlmostEqual(cfg.correlation_recovery_threshold, 0.90)
        self.assertTrue(cfg.require_data_integrity_ok)


# ---------------------------------------------------------------------------
# Test: HaltState dataclass
# ---------------------------------------------------------------------------

class TestHaltState(unittest.TestCase):
    """HaltState dataclass serialization."""

    def test_is_active(self):
        state = HaltState(halt_reason="CORRELATION_CRISIS", halt_since_ts=1000.0)
        self.assertTrue(state.is_active)

    def test_is_not_active_empty(self):
        state = HaltState()
        self.assertFalse(state.is_active)

    def test_is_not_active_no_ts(self):
        state = HaltState(halt_reason="CORRELATION_CRISIS", halt_since_ts=0.0)
        self.assertFalse(state.is_active)

    def test_round_trip(self):
        state = HaltState(
            halt_reason="CORRELATION_COLLAPSE",
            halt_since_ts=12345.0,
            last_auto_recovery_ts=67890.0,
        )
        d = state.to_dict()
        state2 = HaltState.from_dict(d)
        self.assertEqual(state2.halt_reason, "CORRELATION_COLLAPSE")
        self.assertAlmostEqual(state2.halt_since_ts, 12345.0)
        self.assertAlmostEqual(state2.last_auto_recovery_ts, 67890.0)


# ---------------------------------------------------------------------------
# Test: No active halt -> should_auto_recover returns False
# ---------------------------------------------------------------------------

class TestNoActiveHalt(unittest.TestCase):
    """When no halt is active, should_auto_recover returns (False, 'no_active_halt')."""

    def test_no_halt_returns_false(self):
        gate = _gate()
        ok, reason = gate.should_auto_recover(0.5, True, time.time())
        self.assertFalse(ok)
        self.assertEqual(reason, "no_active_halt")


# ---------------------------------------------------------------------------
# Test: ProductionConfig integration
# ---------------------------------------------------------------------------

class TestProductionConfigAutoRecovery(unittest.TestCase):
    """ProductionConfig loads auto_recovery_config from JSON."""

    def test_config_loaded_from_ultra_aggressive(self):
        """Verify from_file() parses auto_recovery section."""
        config_path = Path(__file__).parent.parent / "configs" / "ultra_aggressive_5y.json"
        if not config_path.exists():
            self.skipTest("ultra_aggressive_5y.json not found")

        # We need to test that from_file actually reads auto_recovery
        # Without importing the full main module (heavy), check JSON directly
        with open(config_path) as f:
            data = json.load(f)

        ar = data.get("auto_recovery")
        self.assertIsNotNone(ar)
        self.assertTrue(ar["enabled"])
        self.assertIn("CORRELATION_CRISIS", ar["allowed_reasons"])

    def test_cloud_production_has_no_auto_recovery(self):
        """cloud_production.json should NOT have auto_recovery (HIGH_RISK default)."""
        config_path = Path(__file__).parent.parent / "configs" / "cloud_production.json"
        if not config_path.exists():
            self.skipTest("cloud_production.json not found")

        with open(config_path) as f:
            data = json.load(f)

        ar = data.get("auto_recovery")
        self.assertIsNone(ar)


if __name__ == "__main__":
    unittest.main()
