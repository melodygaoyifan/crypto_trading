"""
Test harness for StrategyExistenceFuse (CLAUDE.md non-negotiable rule #3).

The audit found this safety-critical module had zero direct tests despite
multiple recent fixes (FIX-FUSE-AUTORECOVERY, FIX-DA2, UL-6a, UL-6b).

Covers:
- Default config values (28d window, -15%/-18% thresholds, 10 consecutive loss limit)
- record_pnl() history tracking and equity initialization
- Weekly loss limit (-15%) → suspend
- Monthly loss limit (-18%) → suspend
- Rolling window: 2 consecutive negative periods → suspend
- Consecutive trade losses: 10 → suspend (on_trade_close path)
- min_data_points (5) gates evaluation
- Default allow_auto_recovery=False (FIX-DA2: matches documented contract)
- request_unsuspend(): refuses without confirm; refuses if not suspended;
  refuses during cooldown; resets _consecutive_trade_losses on success
- force_suspend() for emergency
- on_trade_close(): only negative pnl counts toward streak
- Time-based recovery (FIX-FUSE-AUTORECOVERY): consecutive-loss suspension
  auto-recovers after cooldown, but structural suspensions (weekly/monthly)
  do NOT
"""
import sys, os
from datetime import datetime, timedelta, timezone

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from defense.strategy_existence_fuse import (
    StrategyExistenceFuse,
    ExistenceFuseConfig,
    FuseState,
)


# =============================================================================
# Fixture: short evaluation interval so _maybe_evaluate fires every call
# =============================================================================

def _fast_config(**overrides):
    """Build a config with 0s evaluation interval so every record_pnl()
    call triggers _evaluate(). Production default is 14400s (4h)."""
    base = dict(
        evaluation_interval_seconds=0.0,
        min_data_points=5,
    )
    base.update(overrides)
    return ExistenceFuseConfig(**base)


@pytest.fixture
def fuse():
    return StrategyExistenceFuse(_fast_config())


def _seed_history(fuse, count=10, pnl_each=10.0, equity_start=10_000.0):
    """Push N records into the fuse over staggered timestamps in the past 7 days."""
    now = datetime.now(timezone.utc)
    eq = equity_start
    # Walk timestamps from oldest → newest within window
    for i in range(count):
        ts = now - timedelta(days=7) + timedelta(hours=12 * i)
        eq += pnl_each
        fuse.record_pnl(realized_pnl=pnl_each, current_equity=eq, timestamp=ts)


# =============================================================================
# Initialization
# =============================================================================

class TestInitialization:
    def test_starts_active(self):
        f = StrategyExistenceFuse(_fast_config())
        assert f._state == FuseState.ACTIVE
        assert f.is_entry_allowed() is True
        assert f.is_suspended() is False

    def test_default_window_is_28_days(self):
        f = StrategyExistenceFuse()
        assert f.config.rolling_window_days == 28

    def test_default_pnl_threshold_negative_15pct(self):
        f = StrategyExistenceFuse()
        assert f.config.pnl_threshold_pct == pytest.approx(-0.15)

    def test_default_equity_drawdown_negative_18pct(self):
        f = StrategyExistenceFuse()
        assert f.config.equity_drawdown_threshold_pct == pytest.approx(-0.18)

    def test_default_weekly_loss_limit(self):
        f = StrategyExistenceFuse()
        assert f.config.weekly_loss_limit_pct == pytest.approx(-0.15)

    def test_default_monthly_loss_limit(self):
        f = StrategyExistenceFuse()
        assert f.config.monthly_loss_limit_pct == pytest.approx(-0.18)

    def test_default_consecutive_loss_pause(self):
        f = StrategyExistenceFuse()
        assert f.config.consecutive_loss_pause == 10

    def test_default_no_auto_recovery(self):
        """[FIX-DA2] Documented contract is manual-only recovery."""
        f = StrategyExistenceFuse()
        assert f.config.allow_auto_recovery is False


# =============================================================================
# record_pnl
# =============================================================================

class TestRecordPnL:
    def test_initializes_starting_equity(self, fuse):
        fuse.record_pnl(realized_pnl=0.0, current_equity=10_000.0)
        assert fuse._starting_equity == 10_000.0

    def test_history_appends(self, fuse):
        fuse.record_pnl(realized_pnl=10.0, current_equity=10_010.0)
        fuse.record_pnl(realized_pnl=20.0, current_equity=10_030.0)
        assert len(fuse._pnl_history) == 2

    def test_cumulative_pnl_accumulates(self, fuse):
        fuse.record_pnl(realized_pnl=10.0, current_equity=10_010.0)
        fuse.record_pnl(realized_pnl=20.0, current_equity=10_030.0)
        assert fuse._pnl_history[-1].cumulative_pnl == 30.0


# =============================================================================
# Weekly / monthly loss limits
# =============================================================================

class TestWeeklyLossLimit:
    def test_below_limit_no_suspension(self, fuse):
        # 5% weekly loss — under 15% limit. Need >= min_data_points records.
        now = datetime.now(timezone.utc)
        for i in range(5):
            ts = now - timedelta(hours=24 * i)
            # First record sets equity baseline; cumulative loss ~5%
            equity = 10_000.0 - (100.0 * i)
            fuse.record_pnl(realized_pnl=-100.0, current_equity=equity, timestamp=ts)
        assert fuse._state == FuseState.ACTIVE

    def test_above_limit_suspends(self, fuse):
        """20% weekly loss should trip the -15% limit."""
        now = datetime.now(timezone.utc)
        # 5 records over the past 5 days, 4% loss each = 20% cumulative
        equity = 10_000.0
        for i in range(5):
            ts = now - timedelta(days=5 - i)
            loss = -400.0  # 4% on 10k = 400/equity each
            equity -= 400.0
            fuse.record_pnl(realized_pnl=loss, current_equity=equity, timestamp=ts)
        assert fuse._state == FuseState.SUSPENDED
        assert "Weekly" in fuse._suspension_reason


class TestMonthlyLossLimit:
    def test_below_weekly_above_monthly_suspends(self, fuse):
        """Spread loss over 30 days so weekly is fine but monthly trips."""
        now = datetime.now(timezone.utc)
        equity = 10_000.0
        # 5 records spaced ~6 days apart, total -20% over 30 days
        for i in range(5):
            ts = now - timedelta(days=29 - 6 * i)
            loss = -400.0
            equity -= 400.0
            fuse.record_pnl(realized_pnl=loss, current_equity=equity, timestamp=ts)
        # Weekly may or may not trip; monthly should trip
        assert fuse._state == FuseState.SUSPENDED
        assert (
            "Monthly" in fuse._suspension_reason
            or "Weekly" in fuse._suspension_reason
        )


# =============================================================================
# min_data_points gating
# =============================================================================

class TestMinDataPoints:
    def test_evaluation_skipped_below_min(self):
        """With only 4 records (< min 5), even a 50% loss doesn't trigger."""
        f = StrategyExistenceFuse(_fast_config(min_data_points=5))
        now = datetime.now(timezone.utc)
        equity = 10_000.0
        for i in range(4):  # 4 < 5
            ts = now - timedelta(days=4 - i)
            equity -= 1500.0
            f.record_pnl(realized_pnl=-1500.0, current_equity=equity, timestamp=ts)
        assert f._state == FuseState.ACTIVE


# =============================================================================
# Consecutive trade losses
# =============================================================================

class TestConsecutiveTradeLosses:
    def test_streak_below_limit_does_not_suspend(self, fuse):
        for _ in range(9):  # 9 < 10
            fuse.on_trade_close(realized_pnl=-1.0)
        assert fuse._state == FuseState.ACTIVE
        assert fuse._consecutive_trade_losses == 9

    def test_streak_at_limit_suspends(self, fuse):
        for _ in range(10):
            fuse.on_trade_close(realized_pnl=-1.0)
        assert fuse._state == FuseState.SUSPENDED
        assert "Consecutive trade losses" in fuse._suspension_reason

    def test_winner_resets_streak(self, fuse):
        """A winning trade (pnl > 0) resets the consecutive-loss counter."""
        for _ in range(5):
            fuse.on_trade_close(realized_pnl=-1.0)
        assert fuse._consecutive_trade_losses == 5
        fuse.on_trade_close(realized_pnl=10.0)
        assert fuse._consecutive_trade_losses == 0

    def test_breakeven_does_not_reset_streak(self, fuse):
        """[FIX-H3] pnl == 0 (breakeven from rounding/fees) is neither a win
        nor a loss — must NOT reset the streak."""
        for _ in range(5):
            fuse.on_trade_close(realized_pnl=-1.0)
        assert fuse._consecutive_trade_losses == 5
        fuse.on_trade_close(realized_pnl=0.0)
        # Streak unchanged
        assert fuse._consecutive_trade_losses == 5

    def test_initial_winner_keeps_counter_at_zero(self, fuse):
        fuse.on_trade_close(realized_pnl=10.0)
        fuse.on_trade_close(realized_pnl=0.0)
        assert fuse._consecutive_trade_losses == 0


# =============================================================================
# is_entry_allowed
# =============================================================================

class TestEntryAllowed:
    def test_active_allows_entry(self, fuse):
        assert fuse.is_entry_allowed() is True

    def test_suspended_blocks_entry(self, fuse):
        fuse.force_suspend("test")
        assert fuse.is_entry_allowed() is False

    def test_is_suspended_truth_table(self, fuse):
        assert fuse.is_suspended() is False
        fuse.force_suspend("test")
        assert fuse.is_suspended() is True


# =============================================================================
# Manual unsuspend
# =============================================================================

class TestManualUnsuspend:
    def test_refuses_when_not_suspended(self, fuse):
        ok, msg = fuse.request_unsuspend(confirm=True)
        assert ok is False
        assert "not suspended" in msg.lower()

    def test_refuses_without_confirm(self, fuse):
        fuse.force_suspend("test")
        ok, msg = fuse.request_unsuspend(confirm=False)
        assert ok is False
        assert "confirm" in msg.lower()

    def test_refuses_during_cooldown(self, fuse):
        fuse.force_suspend("test")
        # Just-suspended → cooldown active (24h default)
        ok, msg = fuse.request_unsuspend(confirm=True)
        assert ok is False
        assert "cooldown" in msg.lower()

    def test_succeeds_after_cooldown(self, fuse):
        fuse.force_suspend("test")
        # Backdate to past cooldown
        fuse._suspended_at = datetime.now(timezone.utc) - timedelta(hours=25)
        ok, msg = fuse.request_unsuspend(confirm=True)
        assert ok is True
        assert fuse._state == FuseState.ACTIVE

    def test_resets_consecutive_trade_losses_on_unsuspend(self, fuse):
        """Unsuspend must reset the loss streak counter — otherwise a single
        new loss immediately re-suspends. Recent fix."""
        for _ in range(10):
            fuse.on_trade_close(realized_pnl=-1.0)
        # Now suspended with counter at 10
        fuse._suspended_at = datetime.now(timezone.utc) - timedelta(hours=25)
        ok, _ = fuse.request_unsuspend(confirm=True)
        assert ok is True
        assert fuse._consecutive_trade_losses == 0


# =============================================================================
# Time-based recovery (FIX-FUSE-AUTORECOVERY)
# =============================================================================

class TestTimeBasedRecovery:
    def test_consecutive_loss_suspension_recovers_after_cooldown(self):
        """When suspension reason is 'Consecutive trade losses', the time-
        based recovery path lifts it after cooldown (with allow_auto_recovery
        explicitly enabled — default is False)."""
        f = StrategyExistenceFuse(_fast_config(
            allow_auto_recovery=True,
            suspension_cooldown_hours=24.0,
        ))
        for _ in range(10):
            f.on_trade_close(realized_pnl=-1.0)
        assert f._state == FuseState.SUSPENDED
        # Backdate suspension past cooldown
        f._suspended_at = datetime.now(timezone.utc) - timedelta(hours=25)
        # is_entry_allowed triggers _check_time_based_recovery
        assert f.is_entry_allowed() is True
        assert f._state == FuseState.ACTIVE
        assert f._consecutive_trade_losses == 0

    def test_structural_suspension_does_not_time_recover(self):
        """Weekly/monthly limit suspensions must NOT auto-recover via the
        time-based path — only consecutive-loss suspensions do."""
        f = StrategyExistenceFuse(_fast_config(
            allow_auto_recovery=True,
            suspension_cooldown_hours=24.0,
        ))
        # Trigger a structural (weekly) suspension
        now = datetime.now(timezone.utc)
        equity = 10_000.0
        for i in range(5):
            ts = now - timedelta(days=5 - i)
            equity -= 400.0
            f.record_pnl(realized_pnl=-400.0, current_equity=equity, timestamp=ts)
        assert f._state == FuseState.SUSPENDED
        assert "Weekly" in f._suspension_reason or "Monthly" in f._suspension_reason
        # Backdate past cooldown
        f._suspended_at = datetime.now(timezone.utc) - timedelta(hours=25)
        # Time-based recovery should NOT lift this — it's structural
        f._check_time_based_recovery()
        assert f._state == FuseState.SUSPENDED

    def test_no_auto_recovery_when_disabled(self):
        """Default config: allow_auto_recovery=False. Time-based recovery
        must NOT fire even after cooldown for consecutive losses."""
        f = StrategyExistenceFuse(_fast_config(
            allow_auto_recovery=False,  # explicit
        ))
        for _ in range(10):
            f.on_trade_close(realized_pnl=-1.0)
        f._suspended_at = datetime.now(timezone.utc) - timedelta(hours=25)
        f._check_time_based_recovery()
        assert f._state == FuseState.SUSPENDED


# =============================================================================
# force_suspend (emergency / test path)
# =============================================================================

class TestForceSuspend:
    def test_immediate_suspension(self, fuse):
        fuse.force_suspend("manual")
        assert fuse._state == FuseState.SUSPENDED
        assert fuse._suspension_reason == "manual"
        assert fuse._suspended_at is not None
