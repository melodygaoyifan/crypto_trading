"""
Test harness for DRLPromotionGate (CLAUDE.md non-negotiable rule #4).

The audit found this safety-critical module had zero direct tests.

Covers:
- Initialization defaults (DISABLED, demotion enabled)
- Manual promote() to each valid level + invalid level rejection
- Trade recording: PnL accumulates only when drl_contributed=True
- Auto-demotion path 1: 5 consecutive losses → demote
- Auto-demotion path 2: 15% drawdown from peak → demote
- Auto-demotion path 3 [PATCH-4]: zero-peak with negative equity → demote
  (closes the silent-bypass bug where DRL could lose forever without
  triggering drawdown demotion)
- EXIT_ONLY can also demote (not just ACTIVE — [FIX-DA3])
- Max demotions in window → DISABLED (manual recovery only)
- Auto-recovery after 3-day cooldown (configurable)
- Auto-recovery does NOT lift DISABLED
- State persistence: save → load round-trip
- Single losing trade does NOT demote
"""
import sys, os
import json
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from drl.promotion_gate import DRLPromotionGate, DemotionConfig, DRLTradeRecord


@pytest.fixture
def gate(tmp_path):
    """Fresh gate with state file in tmp_path so tests don't pollute disk."""
    state_file = tmp_path / "drl_state.json"
    return DRLPromotionGate(state_file=str(state_file))


# =============================================================================
# Initialization
# =============================================================================

class TestInitialization:
    def test_defaults_to_disabled(self, gate):
        assert gate.get_authority_level() == "DISABLED"

    @pytest.mark.skip(
        reason="[TEST-CLEANUP 2026-06-09] Per drl/promotion_gate.py:36, "
               "DemotionConfig.enable was permanently set to False per operator "
               "directive 2026-04-30 (HEALTH_T2 incident). Auto-demotion is "
               "structurally disabled; test pre-dates the directive."
    )
    def test_demotion_enabled_by_default(self, gate):
        assert gate.demotion_config.enable is True

    def test_default_consecutive_loss_threshold_is_5(self, gate):
        assert gate.demotion_config.consecutive_loss_threshold == 5

    def test_default_drawdown_trigger_is_15pct(self, gate):
        assert gate.demotion_config.drawdown_trigger_pct == 0.15

    def test_default_recovery_period_is_3_days(self, gate):
        assert gate.demotion_config.recovery_period_days == 3

    def test_default_max_demotions_is_3(self, gate):
        assert gate.demotion_config.max_demotions_before_disable == 3

    def test_state_file_loads_existing_state(self, tmp_path):
        state_file = tmp_path / "preexisting.json"
        state_file.write_text(json.dumps({
            "authority_level": "ACTIVE",
            "peak_equity": 100.0,
            "current_equity": 80.0,
        }), encoding="utf-8")
        g = DRLPromotionGate(state_file=str(state_file))
        assert g.get_authority_level() == "ACTIVE"

    def test_invalid_persisted_level_falls_back_to_disabled(self, tmp_path):
        state_file = tmp_path / "corrupt.json"
        state_file.write_text(json.dumps({"authority_level": "BANANA"}), encoding="utf-8")
        g = DRLPromotionGate(state_file=str(state_file))
        assert g.get_authority_level() == "DISABLED"

    def test_aware_demoted_at_loaded_as_naive_no_typeerror(self, tmp_path):
        """[P39] State file with tz-aware ISO timestamp (e.g. saved by a
        future-tz-aware version) must load and not crash naive comparisons.

        Repro: write `_demoted_at` with `+00:00` suffix → fromisoformat
        returns aware → naive `datetime.now() >= aware` raises TypeError.
        Fix: tz-strip on load.
        """
        state_file = tmp_path / "tz_aware.json"
        # Past-deadline aware timestamp
        state_file.write_text(json.dumps({
            "authority_level": "EXIT_ONLY",
            "demoted_at": "2026-04-20T12:00:00+00:00",  # aware
            "peak_equity": 0.0,
            "current_equity": 0.0,
            "demotion_history": [
                {"timestamp": "2026-04-19T08:00:00+00:00",
                 "from": "ACTIVE", "to": "EXIT_ONLY", "reason": "test"},
            ],
        }), encoding="utf-8")
        g = DRLPromotionGate(state_file=str(state_file))
        # Must not raise TypeError
        level = g.get_authority_level()
        assert level in ("ACTIVE", "EXIT_ONLY", "DISABLED")
        # get_status walks demotion_history with fromisoformat — also must not raise
        status = g.get_status()
        assert isinstance(status["recent_demotions"], int)


# =============================================================================
# Manual promotion
# =============================================================================

class TestManualPromote:
    def test_promote_to_shadow(self, gate):
        gate.promote("SHADOW")
        assert gate.get_authority_level() == "SHADOW"

    def test_promote_to_active(self, gate):
        gate.promote("ACTIVE")
        assert gate.get_authority_level() == "ACTIVE"

    def test_promote_invalid_level_rejected(self, gate):
        gate.promote("ACTIVE")
        gate.promote("MEGA_ULTRA")  # not in VALID_LEVELS
        # Stays at ACTIVE
        assert gate.get_authority_level() == "ACTIVE"

    def test_promote_clears_demoted_at(self, gate):
        gate.promote("ACTIVE")
        gate._demoted_at = datetime.now()  # simulate prior demotion
        gate.promote("ACTIVE")  # re-promote
        assert gate._demoted_at is None


# =============================================================================
# Trade recording
# =============================================================================

class TestTradeRecording:
    def test_records_appended(self, gate):
        gate.promote("ACTIVE")
        gate.record_trade(pnl=10.0, drl_contributed=True)
        gate.record_trade(pnl=-5.0, drl_contributed=True)
        assert len(gate._trade_history) == 2

    def test_drl_contributed_updates_equity(self, gate):
        gate.promote("ACTIVE")
        gate.record_trade(pnl=10.0, drl_contributed=True)
        gate.record_trade(pnl=20.0, drl_contributed=True)
        assert gate._current_equity_contribution == 30.0
        assert gate._peak_equity_contribution == 30.0

    def test_non_drl_trade_does_not_update_equity(self, gate):
        gate.promote("ACTIVE")
        gate.record_trade(pnl=100.0, drl_contributed=False)
        assert gate._current_equity_contribution == 0.0
        # But trade history still records it
        assert len(gate._trade_history) == 1

    def test_peak_only_increases(self, gate):
        gate.promote("ACTIVE")
        gate.record_trade(pnl=50.0)
        # 5% DD stays under 15% drawdown threshold → no demotion, peak preserved
        gate.record_trade(pnl=-2.5)
        assert gate._peak_equity_contribution == 50.0
        assert gate._current_equity_contribution == pytest.approx(47.5)


# =============================================================================
# Auto-demotion: consecutive losses
# =============================================================================

_AUTO_DEMOTION_DISABLED = pytest.mark.skip(
    reason="[TEST-CLEANUP 2026-06-09] Auto-demotion permanently disabled per "
           "operator directive 2026-04-30 (drl/promotion_gate.py:36 + :196 + "
           "auto-recovery short-circuit at :90). All demotion/recovery tests "
           "in this file assert behavior that no longer fires. Promote() and "
           "level inspection still tested in TestManualPromote + "
           "TestStatePersistence (save/restore round-trip)."
)


@_AUTO_DEMOTION_DISABLED
class TestConsecutiveLossDemotion:
    """Note: tests must build up positive equity FIRST (peak > 0), otherwise
    the zero-peak-loss demotion path (P-PATCH-4) fires on trade 1 and
    consecutive-loss path never gets to count to 5. Loss size is also kept
    well under 15% of peak so the drawdown path doesn't fire either."""

    def test_5_consecutive_losses_demotes_active(self, gate):
        gate.promote("ACTIVE")
        # Establish big peak so 5 small losses stay under 15% DD threshold
        gate.record_trade(pnl=1000.0)
        for _ in range(5):
            gate.record_trade(pnl=-1.0)
        assert gate.get_authority_level() == "EXIT_ONLY"
        assert gate._demoted_at is not None

    def test_4_losses_does_not_demote(self, gate):
        gate.promote("ACTIVE")
        gate.record_trade(pnl=1000.0)
        for _ in range(4):
            gate.record_trade(pnl=-1.0)
        assert gate.get_authority_level() == "ACTIVE"

    def test_loss_streak_broken_does_not_demote(self, gate):
        gate.promote("ACTIVE")
        gate.record_trade(pnl=1000.0)
        gate.record_trade(pnl=-1.0)
        gate.record_trade(pnl=-1.0)
        gate.record_trade(pnl=5.0)  # win breaks streak
        gate.record_trade(pnl=-1.0)
        gate.record_trade(pnl=-1.0)
        gate.record_trade(pnl=-1.0)
        # Last 5: -1, 5, -1, -1, -1 → not all losses
        assert gate.get_authority_level() == "ACTIVE"

    def test_exit_only_also_demotes_on_consecutive_losses(self, gate):
        """[FIX-DA3] EXIT_ONLY can also degrade. Demotion fires; _demoted_at set."""
        gate.promote("EXIT_ONLY")
        gate.record_trade(pnl=1000.0)
        for _ in range(5):
            gate.record_trade(pnl=-1.0)
        assert gate._demoted_at is not None

    def test_disabled_does_not_demote(self, gate):
        # DISABLED isn't in (ACTIVE, EXIT_ONLY), so demotion check is skipped.
        gate.promote("DISABLED")
        for _ in range(5):
            gate.record_trade(pnl=-10.0)
        assert gate._demoted_at is None


# =============================================================================
# Auto-demotion: drawdown
# =============================================================================

@_AUTO_DEMOTION_DISABLED
class TestDrawdownDemotion:
    def test_15pct_drawdown_demotes(self, gate):
        gate.promote("ACTIVE")
        # Build peak of 100
        gate.record_trade(pnl=100.0)
        # Now lose 16% from peak → should demote
        gate.record_trade(pnl=-16.0)
        assert gate._demoted_at is not None
        assert gate.get_authority_level() == "EXIT_ONLY"

    def test_10pct_drawdown_does_not_demote(self, gate):
        gate.promote("ACTIVE")
        gate.record_trade(pnl=100.0)
        gate.record_trade(pnl=-10.0)  # 10% DD, below 15%
        assert gate.get_authority_level() == "ACTIVE"

    def test_zero_peak_with_negative_equity_demotes(self, gate):
        """[PATCH-4] Closes silent bypass: peak=0 and current<0 used to
        skip the drawdown check entirely. Now: any loss from zero peak
        triggers demotion (only-lost-money path)."""
        gate.promote("ACTIVE")
        # Single loss with zero prior peak
        gate.record_trade(pnl=-10.0)
        # Note: 1 loss is NOT enough for consecutive_losses (needs 5),
        # but the zero-peak path should fire on the same call.
        assert gate._demoted_at is not None


# =============================================================================
# Auto-recovery
# =============================================================================

@_AUTO_DEMOTION_DISABLED
class TestAutoRecovery:
    def test_recovery_after_cooldown(self, gate):
        gate.promote("ACTIVE")
        gate.record_trade(pnl=1000.0)
        for _ in range(5):
            gate.record_trade(pnl=-1.0)
        assert gate._demoted_at is not None
        # Backdate _demoted_at past the recovery deadline
        gate._demoted_at = datetime.now() - timedelta(days=4)
        level = gate.get_authority_level()  # triggers _auto_recover
        assert level == "ACTIVE"
        assert gate._demoted_at is None

    def test_recovery_does_not_lift_disabled(self, gate):
        """Force DISABLED state then verify recovery is a no-op."""
        gate.promote("ACTIVE")
        # Trigger 4 demotions in window using the zero-peak path:
        # each loss with peak=0 → demote (no need to build peak).
        for cycle in range(4):
            gate.record_trade(pnl=-10.0)
            gate.promote("ACTIVE")
        # Final trigger to actually land DISABLED
        gate.record_trade(pnl=-10.0)
        assert gate.get_authority_level() == "DISABLED"
        # Backdate; recovery should NOT lift DISABLED
        gate._demoted_at = datetime.now() - timedelta(days=4)
        assert gate.get_authority_level() == "DISABLED"

    def test_recovery_resets_equity_tracking(self, gate):
        gate.promote("ACTIVE")
        gate.record_trade(pnl=1000.0)
        for _ in range(5):
            gate.record_trade(pnl=-1.0)
        gate._demoted_at = datetime.now() - timedelta(days=4)
        gate.get_authority_level()  # triggers recovery
        assert gate._peak_equity_contribution == 0.0
        assert gate._current_equity_contribution == 0.0


# =============================================================================
# Max demotions → DISABLED
# =============================================================================

@_AUTO_DEMOTION_DISABLED
class TestMaxDemotionsDisable:
    def test_3_demotions_in_14d_disables(self, gate):
        """After 3 demotions in 14-day window, next demotion sets DISABLED
        (not EXIT_ONLY) — manual intervention required."""
        gate.promote("ACTIVE")
        for cycle in range(3):
            for _ in range(5):
                gate.record_trade(pnl=-10.0)
            gate.promote("ACTIVE")  # re-promote
        # 3 demotions in history. 4th demotion → DISABLED.
        for _ in range(5):
            gate.record_trade(pnl=-10.0)
        assert gate.get_authority_level() == "DISABLED"

    def test_old_demotions_outside_window_dont_count(self, gate):
        """Demotions older than demotion_window_days (14) shouldn't count
        toward the max-demotions-before-disable threshold."""
        gate.promote("ACTIVE")
        # Manually inject 3 stale demotion records
        old_ts = (datetime.now() - timedelta(days=30)).isoformat()
        for _ in range(3):
            gate._demotion_history.append({
                "timestamp": old_ts, "from": "ACTIVE",
                "to": "EXIT_ONLY", "reason": "old",
            })
        # Single trade triggers zero-peak demotion exactly once.
        gate.record_trade(pnl=-10.0)
        # Should land EXIT_ONLY (only 1 in-window demotion), NOT DISABLED.
        assert gate.get_authority_level() == "EXIT_ONLY"


# =============================================================================
# State persistence
# =============================================================================

class TestStatePersistence:
    def test_save_load_round_trip(self, tmp_path):
        state_file = tmp_path / "round_trip.json"
        g1 = DRLPromotionGate(state_file=str(state_file))
        g1.promote("ACTIVE")
        g1.record_trade(pnl=50.0)
        # Small loss stays under 15% drawdown so no demotion fires
        g1.record_trade(pnl=-2.0)

        g2 = DRLPromotionGate(state_file=str(state_file))
        assert g2.get_authority_level() == "ACTIVE"
        assert g2._peak_equity_contribution == 50.0
        assert g2._current_equity_contribution == pytest.approx(48.0)

    @_AUTO_DEMOTION_DISABLED
    def test_demotion_history_persisted(self, tmp_path):
        state_file = tmp_path / "history.json"
        g1 = DRLPromotionGate(state_file=str(state_file))
        g1.promote("ACTIVE")
        # Single zero-peak loss is enough to demote once
        g1.record_trade(pnl=-10.0)
        # demotion happened
        g2 = DRLPromotionGate(state_file=str(state_file))
        assert len(g2._demotion_history) >= 1


# =============================================================================
# Authority enum compatibility
# =============================================================================

class TestAuthorityEnumShim:
    def test_get_authority_returns_enum(self, gate):
        gate.promote("ACTIVE")
        from integration.integration_v36 import DRLAuthorityLevel
        result = gate.get_authority()
        assert isinstance(result, DRLAuthorityLevel)
        assert result.value == "ACTIVE"

    def test_has_exit_authority(self, gate):
        gate.promote("ACTIVE")
        assert gate.has_exit_authority() is True
        gate.promote("EXIT_ONLY")
        assert gate.has_exit_authority() is True
        gate.promote("SHADOW")
        assert gate.has_exit_authority() is False
        gate.promote("DISABLED")
        assert gate.has_exit_authority() is False

    def test_is_shadow_mode(self, gate):
        gate.promote("SHADOW")
        assert gate.is_shadow_mode() is True
        gate.promote("ACTIVE")
        assert gate.is_shadow_mode() is False
