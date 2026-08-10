"""
================================================================================
STRATEGY EXISTENCE FUSE - Time-Based Strategy Suspension
================================================================================
Version: 1.0.0
Purpose: Force strategy hibernation when cumulative performance is negative
         over a rolling time window.

DESIGN PRINCIPLE:
    This is a FUSE, not a steering wheel.
    - It does NOT judge market conditions
    - It does NOT use signals, RSI, regime, or any market data
    - It ONLY looks at: "Has this strategy been losing money over time?"

SUSPENSION CRITERIA:
    IF rolling_window_pnl < 0 for CONSECUTIVE evaluation periods
    THEN strategy enters SUSPENDED state

SUSPENSION BEHAVIOR:
    - NO new entries allowed
    - NO position increases allowed  
    - ALLOWED: Risk-driven exits, position reductions
    - ALLOWED: Engineering safety actions

RECOVERY:
    - NO automatic recovery
    - Requires explicit human confirmation (manual_unsuspend=True)

================================================================================
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple
from collections import deque
from enum import Enum, auto

logger = logging.getLogger("HMATS.StrategyExistenceFuse")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ExistenceFuseConfig:
    """
    Configuration for strategy existence fuse.
    
    Conservative defaults designed for protection, not optimization.
    """
    
    # === ROLLING WINDOW ===
    # How far back to look (in calendar days)
    rolling_window_days: int = 28  # 4 weeks

    # Minimum data points required before fuse can trigger
    min_data_points: int = 5  # At least 5 PnL records

    # === TRIGGER CONDITIONS ===
    # Trigger if cumulative PnL in window is below this (as fraction of starting equity)
    # [UNLEASH] UL-6a: was -0.05 -> -0.15 (short systems accumulate neg PnL in bull phases)
    pnl_threshold_pct: float = -0.15  # [UNLEASH] was -0.05 -> -0.15

    # Alternative: trigger if equity dropped by this much from window start
    # [UNLEASH] UL-6a: was -0.08 -> -0.18
    equity_drawdown_threshold_pct: float = -0.18  # [UNLEASH] was -0.08 -> -0.18

    # How many consecutive evaluation periods must be negative
    consecutive_negative_periods: int = 2  # 2 consecutive checks

    # === WEEKLY / MONTHLY LOSS LIMITS (CHECK-118) ===
    # [PARAM-FIX] UNLEASH v2 UL-6a: widened from -8%/-10% - short systems accumulate neg PnL in bull phases
    weekly_loss_limit_pct: float = -0.15   # -15% weekly loss -> suspend [UNLEASH v2]
    monthly_loss_limit_pct: float = -0.18  # -18% monthly loss -> suspend [UNLEASH v2]

    # === CONSECUTIVE TRADE LOSS LIMIT ===
    # [PARAM-FIX] UNLEASH v2 UL-6a: widened from 5 - short 3-5 loss streaks are normal in bull phases
    consecutive_loss_pause: int = 10      # 10 consecutive losing trades -> suspend [UNLEASH v2]

    # === EVALUATION FREQUENCY ===
    # How often to evaluate (in seconds)
    evaluation_interval_seconds: float = 14400.0  # Every 4 hours (1 tick)
    
    # === RECOVERY ===
    # [P253] History, in order, because the two prior comments contradicted
    # each other: [FIX-FUSE-AUTORECOVERY] flipped this True after the Mar
    # 23-26 incident (suspended 3.5 days with no way to resume); [FIX-DA2]
    # then flipped it back to False because auto-recovery violated the
    # documented contract ("NO automatic recovery, requires human
    # confirmation" — Non-Negotiable Rule #3: manual recovery only).
    # False IS the current, deliberate value: recovery is manual via
    # request_unsuspend(). Consequence: _check_time_based_recovery returns
    # early and _auto_recover is intentionally unreachable — the Mar-26
    # Catch-22 is accepted as the price of the manual-recovery contract.
    allow_auto_recovery: bool = False

    # If auto recovery allowed, require this many positive periods
    recovery_positive_periods: int = 2  # [FIX-FUSE-AUTORECOVERY] was 4 → 2 (require 2 positive 4H evaluations = 8h of positive PnL)

    # Cooldown after suspension before any evaluation (hours)
    # [FIX-FUSE-AUTORECOVERY] 48h → 24h. 48h was too long — missed entire market moves
    suspension_cooldown_hours: float = 24.0  # [FIX-FUSE-AUTORECOVERY] was 48.0 → 24.0


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class FuseState(Enum):
    """Fuse state enumeration."""
    ACTIVE = auto()      # Strategy is allowed to trade
    SUSPENDED = auto()   # Strategy is suspended - no new entries
    COOLDOWN = auto()    # In post-suspension cooldown


@dataclass
class PnLRecord:
    """Single PnL observation."""
    timestamp: datetime
    realized_pnl: float      # Realized PnL since last record
    cumulative_pnl: float    # Cumulative PnL since start
    equity: float            # Current equity
    trade_count: int = 0     # Number of trades in this period


@dataclass
class FuseStatus:
    """Current fuse status for external queries."""
    state: FuseState
    is_suspended: bool
    suspension_reason: str
    suspended_at: Optional[datetime]
    
    # Window metrics
    window_pnl: float
    window_pnl_pct: float
    window_start_equity: float
    current_equity: float
    equity_change_pct: float
    
    # Consecutive tracking
    consecutive_negative_periods: int
    
    # Recovery info
    can_auto_recover: bool
    manual_unsuspend_required: bool

    # Weekly / Monthly loss tracking (CHECK-118)
    weekly_loss_pct: Optional[float] = None
    monthly_loss_pct: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            "state": self.state.name,
            "is_suspended": self.is_suspended,
            "suspension_reason": self.suspension_reason,
            "suspended_at": self.suspended_at.isoformat() if self.suspended_at else None,
            "window_pnl": self.window_pnl,
            "window_pnl_pct": self.window_pnl_pct,
            "equity_change_pct": self.equity_change_pct,
            "consecutive_negative_periods": self.consecutive_negative_periods,
            "manual_unsuspend_required": self.manual_unsuspend_required,
            "weekly_loss_pct": self.weekly_loss_pct,
            "monthly_loss_pct": self.monthly_loss_pct,
        }


# =============================================================================
# STRATEGY EXISTENCE FUSE
# =============================================================================

class StrategyExistenceFuse:
    """
    Strategy Existence Fuse - Time-based strategy suspension mechanism.
    
    This is a PROTECTION mechanism, not a trading decision.
    
    Usage:
        fuse = StrategyExistenceFuse()
        
        # On each trade completion
        fuse.record_pnl(realized_pnl=100.0, current_equity=95000.0)
        
        # Before generating any new entry signal
        if not fuse.is_entry_allowed():
            return  # Strategy suspended
        
        # Check status
        status = fuse.get_status()
        if status.is_suspended:
            logger.warning(f"Strategy suspended: {status.suspension_reason}")
    """
    
    def __init__(self, config: Optional[ExistenceFuseConfig] = None):
        self.config = config or ExistenceFuseConfig()
        
        # State
        self._state = FuseState.ACTIVE
        self._suspended_at: Optional[datetime] = None
        self._suspension_reason: str = ""
        
        # PnL history
        self._pnl_history: deque = deque(maxlen=500)  # ~500 records
        self._starting_equity: Optional[float] = None
        
        # Consecutive tracking
        self._consecutive_negative: int = 0
        self._consecutive_positive: int = 0
        
        # Evaluation tracking
        self._last_evaluation: Optional[datetime] = None
        self._evaluation_count: int = 0
        
        # Consecutive trade loss tracking (F-092)
        self._consecutive_trade_losses: int = 0

        # Manual override
        self._manual_unsuspend_requested: bool = False
        
        logger.info(
            f"[EXISTENCE_FUSE] Initialized: "
            f"window={self.config.rolling_window_days}d, "
            f"pnl_threshold={self.config.pnl_threshold_pct:.1%}, "
            f"equity_threshold={self.config.equity_drawdown_threshold_pct:.1%}"
        )
    
    # =========================================================================
    # PUBLIC API
    # =========================================================================
    
    def record_pnl(
        self,
        realized_pnl: float,
        current_equity: float,
        trade_count: int = 1,
        timestamp: Optional[datetime] = None
    ):
        """
        Record a PnL observation.
        
        Call this after each trade completion or at regular intervals.
        
        Args:
            realized_pnl: Realized PnL since last record (can be negative)
            current_equity: Current account equity
            trade_count: Number of trades in this period
            timestamp: Observation timestamp (default: now)
        """
        now = timestamp or datetime.now(timezone.utc)
        
        # Initialize starting equity
        if self._starting_equity is None:
            self._starting_equity = current_equity
            logger.info(f"[EXISTENCE_FUSE] Starting equity set: ${current_equity:,.2f}")
        
        # Calculate cumulative
        if self._pnl_history:
            cumulative = self._pnl_history[-1].cumulative_pnl + realized_pnl
        else:
            cumulative = realized_pnl
        
        # Record
        record = PnLRecord(
            timestamp=now,
            realized_pnl=realized_pnl,
            cumulative_pnl=cumulative,
            equity=current_equity,
            trade_count=trade_count,
        )
        self._pnl_history.append(record)
        
        logger.debug(
            f"[EXISTENCE_FUSE] PnL recorded: "
            f"realized={realized_pnl:+.2f}, cumulative={cumulative:+.2f}, "
            f"equity=${current_equity:,.2f}"
        )
        
        # Evaluate if enough time has passed
        self._maybe_evaluate()
    
    def is_entry_allowed(self) -> bool:
        """
        Check if new entries are allowed.

        Returns:
            True if strategy can open new positions
            False if strategy is suspended
        """
        # [FIX-FUSE-AUTORECOVERY] Check time-based recovery BEFORE evaluation
        # so that consecutive loss suspensions can self-heal after cooldown.
        self._check_time_based_recovery()

        # Evaluate first
        self._maybe_evaluate()

        return self._state == FuseState.ACTIVE
    
    def is_suspended(self) -> bool:
        """Check if strategy is currently suspended."""
        return self._state in [FuseState.SUSPENDED, FuseState.COOLDOWN]
    
    def get_status(self) -> FuseStatus:
        """Get current fuse status."""
        
        # Calculate window metrics
        window_pnl, window_start_equity, current_equity = self._calculate_window_metrics()
        
        window_pnl_pct = 0.0
        equity_change_pct = 0.0
        
        if window_start_equity and window_start_equity > 0:
            window_pnl_pct = window_pnl / window_start_equity
            if current_equity:
                equity_change_pct = (current_equity - window_start_equity) / window_start_equity
        
        return FuseStatus(
            state=self._state,
            is_suspended=self.is_suspended(),
            suspension_reason=self._suspension_reason,
            suspended_at=self._suspended_at,
            window_pnl=window_pnl,
            window_pnl_pct=window_pnl_pct,
            window_start_equity=window_start_equity or 0.0,
            current_equity=current_equity or 0.0,
            equity_change_pct=equity_change_pct,
            consecutive_negative_periods=self._consecutive_negative,
            can_auto_recover=self.config.allow_auto_recovery,
            manual_unsuspend_required=not self.config.allow_auto_recovery,
            weekly_loss_pct=self._calculate_period_loss_pct(days=7),
            monthly_loss_pct=self._calculate_period_loss_pct(days=30),
        )
    
    def request_unsuspend(self, confirm: bool = False) -> Tuple[bool, str]:
        """
        Request manual unsuspension.
        
        Args:
            confirm: Must be True to actually unsuspend
            
        Returns:
            (success, message)
        """
        if self._state == FuseState.ACTIVE:
            return False, "Strategy is not suspended"
        
        if not confirm:
            return False, "Must set confirm=True to unsuspend"
        
        # Check cooldown
        if self._suspended_at:
            cooldown_end = self._suspended_at + timedelta(hours=self.config.suspension_cooldown_hours)
            if datetime.now(timezone.utc) < cooldown_end:
                remaining = cooldown_end - datetime.now(timezone.utc)
                return False, f"Cooldown active: {remaining.total_seconds()/3600:.1f}h remaining"
        
        # Unsuspend
        self._state = FuseState.ACTIVE
        self._consecutive_negative = 0
        self._suspension_reason = ""
        # Reset streak counter — human has reviewed and confirmed restart.
        # Without this, a single additional loss would immediately re-trigger
        # the fuse (counter was at limit when suspended).
        self._consecutive_trade_losses = 0

        logger.warning(
            f"[EXISTENCE_FUSE] [WARN]️ MANUAL UNSUSPEND: "
            f"Strategy reactivated by human confirmation"
        )
        
        return True, "Strategy unsuspended"
    
    def force_suspend(self, reason: str = "Manual suspension"):
        """Force immediate suspension (for testing or emergency)."""
        self._suspend(reason)

    def on_trade_close(self, realized_pnl: float):
        """
        Called on each trade close to track consecutive losses.

        Separate from record_pnl() which tracks periodic equity snapshots.
        This tracks individual trade outcomes for streak detection.
        """
        if realized_pnl < 0:
            self._consecutive_trade_losses += 1
            if (self._state == FuseState.ACTIVE
                    and self._consecutive_trade_losses >= self.config.consecutive_loss_pause):
                self._suspend(
                    f"Consecutive trade losses: {self._consecutive_trade_losses} "
                    f"(limit: {self.config.consecutive_loss_pause})"
                )
        elif realized_pnl > 0:
            # [FIX-H3] Only reset on TRUE profit, not breakeven (pnl=0).
            # Breakeven trades (due to rounding/fees) shouldn't break a loss streak.
            self._consecutive_trade_losses = 0

    @property
    def consecutive_trade_losses(self) -> int:
        """Current consecutive trade loss count."""
        return self._consecutive_trade_losses

    # =========================================================================
    # INTERNAL LOGIC
    # =========================================================================
    
    def _maybe_evaluate(self):
        """Evaluate if enough time has passed since last evaluation."""
        now = datetime.now(timezone.utc)
        
        if self._last_evaluation:
            elapsed = (now - self._last_evaluation).total_seconds()
            if elapsed < self.config.evaluation_interval_seconds:
                return  # Not time yet
        
        self._evaluate()
        self._last_evaluation = now
    
    def _evaluate(self):
        """
        Core evaluation logic.
        
        This is the ONLY place where suspension decisions are made.
        """
        self._evaluation_count += 1
        
        # Need minimum data
        if len(self._pnl_history) < self.config.min_data_points:
            logger.debug(
                f"[EXISTENCE_FUSE] Skipping evaluation: "
                f"insufficient data ({len(self._pnl_history)}/{self.config.min_data_points})"
            )
            return
        
        # Calculate window metrics
        window_pnl, window_start_equity, current_equity = self._calculate_window_metrics()
        
        if window_start_equity is None or window_start_equity <= 0:
            return
        
        # Calculate percentages
        pnl_pct = window_pnl / window_start_equity
        equity_change_pct = (current_equity - window_start_equity) / window_start_equity if current_equity else 0
        
        logger.info(
            f"[EXISTENCE_FUSE] Evaluation #{self._evaluation_count}: "
            f"window_pnl={pnl_pct:+.2%}, equity_change={equity_change_pct:+.2%}, "
            f"state={self._state.name}"
        )
        
        # --- CHECK-118: Weekly / Monthly loss limits ---
        if self._state == FuseState.ACTIVE:
            weekly_pct = self._calculate_period_loss_pct(days=7)
            if weekly_pct is not None and weekly_pct < self.config.weekly_loss_limit_pct:
                self._suspend(
                    f"Weekly loss limit breached: {weekly_pct:+.2%} "
                    f"(limit: {self.config.weekly_loss_limit_pct:+.2%})"
                )
                return

            monthly_pct = self._calculate_period_loss_pct(days=30)
            if monthly_pct is not None and monthly_pct < self.config.monthly_loss_limit_pct:
                self._suspend(
                    f"Monthly loss limit breached: {monthly_pct:+.2%} "
                    f"(limit: {self.config.monthly_loss_limit_pct:+.2%})"
                )
                return

        # Check rolling window conditions
        is_period_negative = (
            pnl_pct < self.config.pnl_threshold_pct or
            equity_change_pct < self.config.equity_drawdown_threshold_pct
        )

        if is_period_negative:
            self._consecutive_negative += 1
            self._consecutive_positive = 0

            logger.warning(
                f"[EXISTENCE_FUSE] Negative period #{self._consecutive_negative}: "
                f"pnl={pnl_pct:+.2%}, equity_change={equity_change_pct:+.2%}"
            )

            # Check if should suspend
            if self._state == FuseState.ACTIVE:
                if self._consecutive_negative >= self.config.consecutive_negative_periods:
                    reason = (
                        f"Rolling {self.config.rolling_window_days}d performance negative: "
                        f"PnL={pnl_pct:+.2%}, Equity={equity_change_pct:+.2%}, "
                        f"Consecutive negative periods={self._consecutive_negative}"
                    )
                    self._suspend(reason)
        else:
            self._consecutive_positive += 1
            self._consecutive_negative = 0
            
            # Check auto-recovery (if enabled)
            if self._state == FuseState.SUSPENDED and self.config.allow_auto_recovery:
                if self._consecutive_positive >= self.config.recovery_positive_periods:
                    self._auto_recover()
    
    def _calculate_window_metrics(self) -> Tuple[float, Optional[float], Optional[float]]:
        """
        Calculate metrics for the rolling window.

        Returns:
            (window_pnl, window_start_equity, current_equity)
        """
        if not self._pnl_history:
            return 0.0, None, None

        now = datetime.now(timezone.utc)
        window_start = now - timedelta(days=self.config.rolling_window_days)
        # Strip to naive for comparison with legacy records
        window_start_naive = window_start.replace(tzinfo=None)

        # Filter to window (handle timezone-naive/aware comparison)
        window_records = []
        for r in self._pnl_history:
            # Make both timestamps comparable by removing timezone info if present
            record_ts = r.timestamp.replace(tzinfo=None) if r.timestamp.tzinfo else r.timestamp
            if record_ts >= window_start_naive:
                window_records.append(r)

        if not window_records:
            return 0.0, None, None

        # Sum PnL in window
        window_pnl = sum(r.realized_pnl for r in window_records)

        # Get start and end equity
        window_start_equity = window_records[0].equity - window_records[0].realized_pnl
        current_equity = window_records[-1].equity
        
        return window_pnl, window_start_equity, current_equity

    def _calculate_period_loss_pct(self, days: int) -> Optional[float]:
        """Calculate PnL % over the last N days. Returns None if insufficient data."""
        if not self._pnl_history:
            return None
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=days)
        # [FIX-M6] Use aware datetimes consistently. Strip tzinfo on BOTH sides
        # to handle legacy naive records (pre-timezone-fix era).
        _cutoff_naive = cutoff.replace(tzinfo=None)
        period_records = [
            r for r in self._pnl_history
            if (r.timestamp.replace(tzinfo=None) if r.timestamp.tzinfo else r.timestamp) >= _cutoff_naive
        ]
        if not period_records:
            return None
        start_equity = period_records[0].equity - period_records[0].realized_pnl
        if start_equity <= 0:
            return None
        period_pnl = sum(r.realized_pnl for r in period_records)
        return period_pnl / start_equity

    def _suspend(self, reason: str):
        """Enter suspended state."""
        self._state = FuseState.SUSPENDED
        self._suspended_at = datetime.now(timezone.utc)
        self._suspension_reason = reason
        
        logger.critical(
            f"[STRATEGY_SUSPENDED] ⛔ FUSE BLOWN\n"
            f"  Reason: {reason}\n"
            f"  Time: {self._suspended_at.isoformat()}\n"
            f"  Action: NO NEW ENTRIES ALLOWED\n"
            f"  Recovery: MANUAL CONFIRMATION REQUIRED"
        )
    
    def _auto_recover(self):
        """Auto-recover from suspension (if enabled)."""
        # Check cooldown
        if self._suspended_at:
            cooldown_end = self._suspended_at + timedelta(hours=self.config.suspension_cooldown_hours)
            if datetime.now(timezone.utc) < cooldown_end:
                return  # Still in cooldown

        self._state = FuseState.ACTIVE
        self._suspension_reason = ""
        self._consecutive_trade_losses = 0  # [FIX-FUSE-AUTORECOVERY] Reset loss streak on recovery

        logger.warning(
            f"[EXISTENCE_FUSE] Auto-recovery: "
            f"Strategy reactivated after {self._consecutive_positive} positive periods"
        )

    def _check_time_based_recovery(self):
        """[FIX-FUSE-AUTORECOVERY] Time-based recovery for consecutive loss suspensions.

        Problem: during suspension no trades execute → no new PnL → window metrics
        don't improve → consecutive_positive stays 0 → standard auto-recovery never
        fires. This is a Catch-22. System stayed suspended 3.5 days (Mar 23-26).

        Fix: after cooldown_hours, if suspension was caused by consecutive losses
        (not by structural drawdown), auto-resume with reset loss counter.
        Structural suspensions (weekly/monthly limits, rolling window) still
        require window improvement or manual unsuspend.
        """
        if self._state != FuseState.SUSPENDED:
            return
        if not self.config.allow_auto_recovery:
            return
        if not self._suspended_at:
            return

        cooldown_end = self._suspended_at + timedelta(hours=self.config.suspension_cooldown_hours)
        if datetime.now(timezone.utc) < cooldown_end:
            return

        # Only auto-resume for consecutive loss suspensions (not structural)
        if "Consecutive trade losses" in self._suspension_reason:
            hours_suspended = (datetime.now(timezone.utc) - self._suspended_at).total_seconds() / 3600
            self._state = FuseState.ACTIVE
            self._consecutive_trade_losses = 0
            self._consecutive_negative = 0
            logger.warning(
                f"[FIX-FUSE-AUTORECOVERY] Time-based recovery after {hours_suspended:.0f}h: "
                f"consecutive loss suspension cleared. "
                f"Previous reason: {self._suspension_reason}"
            )
            self._suspension_reason = ""
    
    # =========================================================================
    # PERSISTENCE (optional, for restarts)
    # =========================================================================
    
    def to_dict(self) -> Dict:
        """Serialize state for persistence."""
        return {
            "state": self._state.name,
            "suspended_at": self._suspended_at.isoformat() if self._suspended_at else None,
            "suspension_reason": self._suspension_reason,
            "starting_equity": self._starting_equity,
            "consecutive_negative": self._consecutive_negative,
            "consecutive_positive": self._consecutive_positive,
            "consecutive_trade_losses": self._consecutive_trade_losses,
            "evaluation_count": self._evaluation_count,
            "pnl_history": [
                {
                    "timestamp": r.timestamp.isoformat(),
                    "realized_pnl": r.realized_pnl,
                    "cumulative_pnl": r.cumulative_pnl,
                    "equity": r.equity,
                    "trade_count": r.trade_count,
                }
                # [P253] Was [-50:]. At the live cadence of 6 records/day
                # (one per 4H tick, P209 feed), 50 records is ~8 days — so
                # every restart silently TRUNCATED the "28-day" rolling
                # window to a week and the monthly limit never had a full
                # month to look at. 28d needs ~168; 30d monthly needs ~180;
                # 400 covers both with margin at trivial file cost.
                for r in list(self._pnl_history)[-400:]
            ]
        }
    
    def from_dict(self, data: Dict):
        """Restore state from persistence."""
        self._state = FuseState[data.get("state", "ACTIVE")]
        
        if data.get("suspended_at"):
            self._suspended_at = datetime.fromisoformat(data["suspended_at"])
        
        self._suspension_reason = data.get("suspension_reason", "")
        self._starting_equity = data.get("starting_equity")
        self._consecutive_negative = data.get("consecutive_negative", 0)
        self._consecutive_positive = data.get("consecutive_positive", 0)
        self._consecutive_trade_losses = data.get("consecutive_trade_losses", 0)
        self._evaluation_count = data.get("evaluation_count", 0)
        
        # Restore PnL history
        for record_data in data.get("pnl_history", []):
            record = PnLRecord(
                timestamp=datetime.fromisoformat(record_data["timestamp"]),
                realized_pnl=record_data["realized_pnl"],
                cumulative_pnl=record_data["cumulative_pnl"],
                equity=record_data["equity"],
                trade_count=record_data.get("trade_count", 0),
            )
            self._pnl_history.append(record)
        
        logger.info(
            f"[EXISTENCE_FUSE] State restored: "
            f"state={self._state.name}, history_records={len(self._pnl_history)}"
        )


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_existence_fuse: Optional[StrategyExistenceFuse] = None


def get_existence_fuse(config: Optional[ExistenceFuseConfig] = None) -> StrategyExistenceFuse:
    """Get or create the existence fuse singleton."""
    global _existence_fuse
    if _existence_fuse is None:
        _existence_fuse = StrategyExistenceFuse(config)
    elif config is not None and _existence_fuse.config != config:
        logger.info("[EXISTENCE_FUSE] Config changed; refreshing singleton instance")
        _existence_fuse = StrategyExistenceFuse(config)
    return _existence_fuse


def reset_existence_fuse():
    """Reset the existence fuse singleton (for testing)."""
    global _existence_fuse
    _existence_fuse = None


# =============================================================================
# VERIFICATION / TEST
# =============================================================================

if __name__ == "__main__":
    """
    Minimal verification example.
    
    Run: python -m defense.strategy_existence_fuse
    """
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    
    print("=" * 70)
    print("STRATEGY EXISTENCE FUSE - VERIFICATION TEST")
    print("=" * 70)
    
    # Test 1: Consecutive losses trigger suspension
    print("\n--- TEST 1: Consecutive losses should trigger suspension ---")
    
    config = ExistenceFuseConfig(
        rolling_window_days=7,
        min_data_points=3,
        pnl_threshold_pct=-0.02,  # -2% (lower threshold for testing)
        consecutive_negative_periods=2,
        evaluation_interval_seconds=0,  # Immediate evaluation for testing
    )
    
    fuse = StrategyExistenceFuse(config)
    
    # Simulate losses - more severe
    equity = 100000.0
    print(f"Starting equity: ${equity:,.2f}")
    
    # Each loss triggers evaluation, need 2 consecutive negative periods
    losses = [-1000, -1200, -1500, -1300, -1000]  # Total: -6000 = -6%
    for i, pnl in enumerate(losses):
        equity += pnl
        # Spread timestamps to ensure evaluation happens each time
        from datetime import datetime, timedelta, timezone
        ts = datetime.now(timezone.utc) + timedelta(hours=i*5)
        fuse.record_pnl(
            realized_pnl=pnl,
            current_equity=equity,
            timestamp=ts
        )
        print(f"  Trade {i+1}: PnL={pnl:+}, Equity=${equity:,.2f}, Suspended={fuse.is_suspended()}")
    
    status = fuse.get_status()
    print(f"\nFinal status:")
    print(f"  State: {status.state.name}")
    print(f"  Suspended: {status.is_suspended}")
    print(f"  Reason: {status.suspension_reason}")
    print(f"  Window PnL: {status.window_pnl_pct:+.2%}")
    
    assert fuse.is_suspended(), "TEST 1 FAILED: Should be suspended after consecutive losses"
    print("✓ TEST 1 PASSED: Suspension triggered correctly")
    
    # Test 2: Mixed results should NOT trigger
    print("\n--- TEST 2: Mixed results should NOT trigger suspension ---")
    
    reset_existence_fuse()
    fuse2 = StrategyExistenceFuse(config)
    
    equity = 100000.0
    mixed = [500, -300, 400, -200, 300]  # Net positive
    for i, pnl in enumerate(mixed):
        equity += pnl
        ts = datetime.now(timezone.utc) + timedelta(hours=i*5)
        fuse2.record_pnl(
            realized_pnl=pnl,
            current_equity=equity,
            timestamp=ts
        )
        print(f"  Trade {i+1}: PnL={pnl:+}, Equity=${equity:,.2f}, Suspended={fuse2.is_suspended()}")
    
    assert not fuse2.is_suspended(), "TEST 2 FAILED: Should NOT be suspended with mixed results"
    print("✓ TEST 2 PASSED: No suspension with positive/mixed results")
    
    # Test 3: Entry check
    print("\n--- TEST 3: Entry check ---")
    
    # Use the suspended fuse from Test 1
    allowed = fuse.is_entry_allowed()
    print(f"  Entry allowed (suspended fuse): {allowed}")
    assert not allowed, "TEST 3 FAILED: Entry should be blocked when suspended"
    
    allowed2 = fuse2.is_entry_allowed()
    print(f"  Entry allowed (active fuse): {allowed2}")
    assert allowed2, "TEST 3 FAILED: Entry should be allowed when active"
    
    print("✓ TEST 3 PASSED: Entry check works correctly")
    
    # Test 4: Manual unsuspend
    print("\n--- TEST 4: Manual unsuspend (with cooldown bypass for test) ---")
    
    fuse.config.suspension_cooldown_hours = 0  # Bypass cooldown for test
    success, msg = fuse.request_unsuspend(confirm=False)
    print(f"  Without confirm: success={success}, msg={msg}")
    assert not success, "Should fail without confirm"
    
    success, msg = fuse.request_unsuspend(confirm=True)
    print(f"  With confirm: success={success}, msg={msg}")
    assert success, "Should succeed with confirm"
    assert fuse.is_entry_allowed(), "Should be active after unsuspend"
    
    print("✓ TEST 4 PASSED: Manual unsuspend works correctly")
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)
