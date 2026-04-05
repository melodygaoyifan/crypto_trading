"""
================================================================================
HMATS v3.5 - FAILURE-AWARE META-DECISION MEMORY
================================================================================
Version: 3.5.0
Upgrade From: v3.4 Meta-Decision Memory
Purpose: Track OPPORTUNITY outcomes with failure awareness for adaptive aggressiveness

CHANGES FROM v3.4:
    v3.4: Simple consecutive loss tracking
    v3.5: Rich failure analysis with magnitude and time-to-failure tracking

NEW FEATURES:
    1. Track success/failure with magnitude (not just win/loss)
    2. Track time-to-failure (how quickly did it fail)
    3. After N consecutive failed OPPORTUNITIES:
       - Raise OPPORTUNITY trigger threshold (density requirement)
       - Slow tranche escalation (delay T2, T3)
    4. Auto-restore after success

WHY THIS IMPROVES PROFITABILITY:
    - Prevents repeated entries into false breakouts
    - Doesn't become permanently conservative (auto-restores)
    - Uses magnitude/timing, not just binary outcome
    - Still aggressive when opportunities are real
================================================================================
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone, timedelta
from enum import Enum
from collections import deque
import statistics

logger = logging.getLogger(__name__)


class OpportunityOutcome(Enum):
    """Outcome classification for OPPORTUNITY trades."""
    STRONG_WIN = "STRONG_WIN"      # > +150 bps
    WIN = "WIN"                     # +50 to +150 bps
    SCRATCH = "SCRATCH"            # -50 to +50 bps
    LOSS = "LOSS"                   # -50 to -150 bps
    STRONG_LOSS = "STRONG_LOSS"    # < -150 bps
    FAST_FAILURE = "FAST_FAILURE"  # Loss within 2 bars (false breakout)


@dataclass
class OpportunityRecord:
    """Detailed record of an OPPORTUNITY trade."""
    
    trade_id: str
    asset: str
    
    # Entry context
    entry_time: datetime
    phase_at_entry: str
    opportunity_density_at_entry: float
    crack_weight_at_entry: float
    
    # Outcome
    outcome: OpportunityOutcome
    pnl_bps: float
    bars_to_outcome: int  # How many bars until exit/reversal
    
    # Time-to-failure (only for losses)
    time_to_max_adverse: Optional[int] = None  # Bars until worst point
    max_drawdown_bps: float = 0.0
    
    # Was it a false breakout?
    is_false_breakout: bool = False
    
    def to_dict(self) -> Dict:
        return {
            "trade_id": self.trade_id,
            "asset": self.asset,
            "outcome": self.outcome.value,
            "pnl_bps": self.pnl_bps,
            "bars_to_outcome": self.bars_to_outcome,
            "is_false_breakout": self.is_false_breakout,
            "density_at_entry": self.opportunity_density_at_entry,
        }


@dataclass
class FailureAwareConfig:
    """Configuration for failure-aware memory."""
    
    # Outcome thresholds
    strong_win_threshold_bps: float = 150.0
    win_threshold_bps: float = 50.0
    loss_threshold_bps: float = -50.0
    strong_loss_threshold_bps: float = -150.0
    
    # False breakout detection
    false_breakout_bars: int = 2  # Failure within 2 bars = false breakout
    false_breakout_loss_bps: float = -30.0  # Must lose at least 30bps
    
    # Consecutive failure tracking
    # [UNLEASH] UL-8f: Relaxed - short 3-loss streaks are normal in bull phases
    consecutive_failure_threshold: int = 5  # [UNLEASH] was 3 -> 5
    consecutive_false_breakout_threshold: int = 3  # [UNLEASH] was 2 -> 3

    # Adjustments after failures
    density_threshold_boost: float = 0.05  # [UNLEASH] was 0.15 -> 0.05 (gentler boost)
    tranche_delay_bars: int = 0  # [UNLEASH] was 1 -> 0 (no delay)
    max_density_boost: float = 0.30  # Cap at +30%
    max_tranche_delay: int = 2  # Cap at 2 bars delay
    
    # Recovery
    wins_to_full_restore: int = 2
    partial_restore_per_win: float = 0.5  # 50% restoration per win
    
    # Auto-restore after time
    auto_restore_hours: int = 48  # Full restore after 48 hours
    
    # Memory window
    memory_window_trades: int = 20


@dataclass
class FailureAwareModifiers:
    """Output modifiers from failure-aware memory."""
    
    # Density threshold boost (add to base requirement)
    density_boost: float = 0.0  # e.g., 0.15 means require 0.85 instead of 0.70
    
    # Tranche escalation delay
    tranche_delay_bars: int = 0  # Bars to wait before escalation
    
    # Aggressiveness multiplier (retained from v3.4)
    aggressiveness_modifier: float = 1.0
    
    # Status
    in_caution_mode: bool = False
    caution_reason: str = ""
    
    # Statistics
    recent_win_rate: float = 0.5
    false_breakout_rate: float = 0.0
    consecutive_failures: int = 0
    
    # Recovery status
    recovery_progress: float = 1.0  # 0 = full caution, 1 = normal
    
    def to_dict(self) -> Dict:
        return {
            "density_boost": self.density_boost,
            "tranche_delay_bars": self.tranche_delay_bars,
            "aggressiveness_modifier": self.aggressiveness_modifier,
            "in_caution_mode": self.in_caution_mode,
            "caution_reason": self.caution_reason,
            "recovery_progress": self.recovery_progress,
        }


class FailureAwareMetaMemory:
    """
    Failure-aware meta-decision memory for v3.5.
    
    UPGRADES FROM v3.4:
        1. Tracks magnitude, not just binary outcome
        2. Detects false breakouts (fast failures)
        3. Adjusts OPPORTUNITY threshold, not just aggressiveness
        4. Delays tranche escalation after failures
    
    USAGE:
        memory = FailureAwareMetaMemory()
        
        # Record completed OPPORTUNITY trade
        memory.record_opportunity(record)
        
        # Get modifiers before entering OPPORTUNITY
        modifiers = memory.get_modifiers()
        
        # Apply in Mode detection
        if opportunity_density >= base_threshold + modifiers.density_boost:
            mode = OPPORTUNITY
        
        # Apply in Tranche Manager
        if modifiers.tranche_delay_bars > 0:
            wait_before_escalation()
    
    WHY THIS DOESN'T BECOME CONSERVATIVE:
        1. Auto-restores after 2 wins
        2. Auto-restores after 48 hours
        3. Only affects OPPORTUNITY threshold, not all trading
        4. Magnitude-aware (big win resets faster)
    """
    
    def __init__(self, config: Optional[FailureAwareConfig] = None):
        self.config = config or FailureAwareConfig()
        
        # Trade memory
        self._records: deque = deque(maxlen=self.config.memory_window_trades)
        
        # Consecutive tracking
        self._consecutive_failures: int = 0
        self._consecutive_false_breakouts: int = 0
        self._consecutive_wins: int = 0
        
        # Current state
        self._in_caution_mode: bool = False
        self._caution_start_time: Optional[datetime] = None
        self._recovery_progress: float = 1.0
        
        # Current adjustments
        self._density_boost: float = 0.0
        self._tranche_delay: int = 0
        
        logger.info("FailureAwareMetaMemory v3.5 initialized")
    
    def record_opportunity(
        self,
        trade_id: str,
        asset: str,
        entry_time: datetime,
        phase_at_entry: str,
        opportunity_density_at_entry: float,
        crack_weight_at_entry: float,
        pnl_bps: float,
        bars_to_outcome: int,
        max_drawdown_bps: float = 0.0,
        time_to_max_adverse: Optional[int] = None,
    ):
        """
        Record an OPPORTUNITY trade outcome.
        
        This is called when an OPPORTUNITY-mode trade completes.
        """
        
        # Classify outcome
        outcome = self._classify_outcome(pnl_bps, bars_to_outcome)
        
        # Detect false breakout
        is_false_breakout = (
            pnl_bps < self.config.false_breakout_loss_bps and
            bars_to_outcome <= self.config.false_breakout_bars
        )
        
        if is_false_breakout:
            outcome = OpportunityOutcome.FAST_FAILURE
        
        record = OpportunityRecord(
            trade_id=trade_id,
            asset=asset,
            entry_time=entry_time,
            phase_at_entry=phase_at_entry,
            opportunity_density_at_entry=opportunity_density_at_entry,
            crack_weight_at_entry=crack_weight_at_entry,
            outcome=outcome,
            pnl_bps=pnl_bps,
            bars_to_outcome=bars_to_outcome,
            time_to_max_adverse=time_to_max_adverse,
            max_drawdown_bps=max_drawdown_bps,
            is_false_breakout=is_false_breakout,
        )
        
        self._records.append(record)
        
        # Update consecutive tracking
        self._update_consecutive_tracking(record)
        
        # Update adjustments
        self._update_adjustments()
        
        logger.info(
            f"FailureAwareMemory: {outcome.value} ({pnl_bps:+.0f}bps), "
            f"consecutive_failures={self._consecutive_failures}, "
            f"density_boost={self._density_boost:.2f}, "
            f"tranche_delay={self._tranche_delay}"
        )
    
    def _classify_outcome(self, pnl_bps: float, bars: int) -> OpportunityOutcome:
        """Classify trade outcome."""
        if pnl_bps >= self.config.strong_win_threshold_bps:
            return OpportunityOutcome.STRONG_WIN
        elif pnl_bps >= self.config.win_threshold_bps:
            return OpportunityOutcome.WIN
        elif pnl_bps >= self.config.loss_threshold_bps:
            return OpportunityOutcome.SCRATCH
        elif pnl_bps >= self.config.strong_loss_threshold_bps:
            return OpportunityOutcome.LOSS
        else:
            return OpportunityOutcome.STRONG_LOSS
    
    def _update_consecutive_tracking(self, record: OpportunityRecord):
        """Update consecutive win/loss tracking."""
        
        is_win = record.outcome in [OpportunityOutcome.STRONG_WIN, OpportunityOutcome.WIN]
        is_loss = record.outcome in [
            OpportunityOutcome.LOSS, 
            OpportunityOutcome.STRONG_LOSS,
            OpportunityOutcome.FAST_FAILURE,
        ]
        
        if is_win:
            self._consecutive_wins += 1
            self._consecutive_failures = 0
            self._consecutive_false_breakouts = 0
            self._handle_win(record)
        elif is_loss:
            self._consecutive_failures += 1
            self._consecutive_wins = 0
            if record.is_false_breakout:
                self._consecutive_false_breakouts += 1
            self._handle_failure(record)
        # SCRATCH doesn't reset either counter
    
    def _handle_win(self, record: OpportunityRecord):
        """Handle a winning OPPORTUNITY."""
        
        if not self._in_caution_mode:
            return
        
        # Recovery based on win magnitude
        if record.outcome == OpportunityOutcome.STRONG_WIN:
            # Strong win = faster recovery
            restore_amount = self.config.partial_restore_per_win * 1.5
        else:
            restore_amount = self.config.partial_restore_per_win
        
        self._recovery_progress = min(1.0, self._recovery_progress + restore_amount)
        
        # Check for full recovery
        if self._consecutive_wins >= self.config.wins_to_full_restore:
            self._exit_caution_mode("Consecutive wins")
        elif self._recovery_progress >= 1.0:
            self._exit_caution_mode("Recovery progress complete")
    
    def _handle_failure(self, record: OpportunityRecord):
        """Handle a failed OPPORTUNITY."""
        
        # Check if should enter caution mode
        should_enter_caution = (
            self._consecutive_failures >= self.config.consecutive_failure_threshold or
            self._consecutive_false_breakouts >= self.config.consecutive_false_breakout_threshold
        )
        
        if should_enter_caution and not self._in_caution_mode:
            self._enter_caution_mode(record)
        elif self._in_caution_mode:
            # Deepen caution
            self._deepen_caution()
    
    def _enter_caution_mode(self, trigger_record: OpportunityRecord):
        """Enter caution mode after failures."""
        
        self._in_caution_mode = True
        self._caution_start_time = datetime.now(timezone.utc)
        self._recovery_progress = 0.0
        
        # Set initial adjustments
        if trigger_record.is_false_breakout:
            # False breakout: increase density threshold more
            self._density_boost = self.config.density_threshold_boost * 1.5
            reason = f"False breakout detected ({self._consecutive_false_breakouts} consecutive)"
        else:
            self._density_boost = self.config.density_threshold_boost
            reason = f"Consecutive failures ({self._consecutive_failures})"
        
        self._tranche_delay = self.config.tranche_delay_bars
        
        logger.warning(f"FailureAwareMemory: ENTERING CAUTION MODE - {reason}")
    
    def _deepen_caution(self):
        """Deepen caution after additional failures while in caution mode."""
        
        # Increase density boost (capped)
        self._density_boost = min(
            self._density_boost + self.config.density_threshold_boost * 0.5,
            self.config.max_density_boost
        )
        
        # Increase tranche delay (capped)
        self._tranche_delay = min(
            self._tranche_delay + 1,
            self.config.max_tranche_delay
        )
        
        # Reset recovery progress
        self._recovery_progress = 0.0
        
        logger.warning(
            f"FailureAwareMemory: Deepening caution - "
            f"density_boost={self._density_boost:.2f}, delay={self._tranche_delay}"
        )
    
    def _exit_caution_mode(self, reason: str):
        """Exit caution mode and restore normal operation."""
        
        self._in_caution_mode = False
        self._caution_start_time = None
        self._recovery_progress = 1.0
        self._density_boost = 0.0
        self._tranche_delay = 0
        
        logger.info(f"FailureAwareMemory: EXITING CAUTION MODE - {reason}")
    
    def _update_adjustments(self):
        """Update adjustments based on current state."""
        
        # Check for time-based auto-restore
        if self._in_caution_mode and self._caution_start_time:
            hours_in_caution = (datetime.now(timezone.utc) - self._caution_start_time).total_seconds() / 3600
            
            if hours_in_caution >= self.config.auto_restore_hours:
                self._exit_caution_mode("Time-based auto-restore")
        
        # If in caution mode, apply partial recovery to adjustments
        if self._in_caution_mode:
            effective_density_boost = self._density_boost * (1 - self._recovery_progress)
            effective_tranche_delay = int(self._tranche_delay * (1 - self._recovery_progress))
            
            self._density_boost = effective_density_boost
            self._tranche_delay = max(0, effective_tranche_delay)
    
    def get_modifiers(self) -> FailureAwareModifiers:
        """
        Get current modifiers for OPPORTUNITY decisions.
        
        WHERE CONSUMED:
            - Mode detection: Add density_boost to threshold
            - Tranche Manager: Apply tranche_delay_bars
            - Fusion: Apply aggressiveness_modifier
        """
        
        # Refresh state
        self._update_adjustments()
        
        # Calculate statistics
        records = list(self._records)
        
        if records:
            wins = sum(1 for r in records if r.outcome in [
                OpportunityOutcome.STRONG_WIN, OpportunityOutcome.WIN
            ])
            false_breakouts = sum(1 for r in records if r.is_false_breakout)
            
            win_rate = wins / len(records)
            fb_rate = false_breakouts / len(records)
        else:
            win_rate = 0.5
            fb_rate = 0.0
        
        # Calculate aggressiveness modifier (compatible with v3.4)
        if self._in_caution_mode:
            # Reduce aggressiveness based on recovery progress
            aggressiveness = 0.7 + 0.3 * self._recovery_progress
        else:
            aggressiveness = 1.0
        
        modifiers = FailureAwareModifiers(
            density_boost=self._density_boost,
            tranche_delay_bars=self._tranche_delay,
            aggressiveness_modifier=aggressiveness,
            in_caution_mode=self._in_caution_mode,
            caution_reason=f"Failures: {self._consecutive_failures}, FB: {self._consecutive_false_breakouts}" if self._in_caution_mode else "",
            recent_win_rate=win_rate,
            false_breakout_rate=fb_rate,
            consecutive_failures=self._consecutive_failures,
            recovery_progress=self._recovery_progress,
        )
        
        return modifiers
    
    def get_statistics(self) -> Dict:
        """Get memory statistics."""
        
        records = list(self._records)
        
        if not records:
            return {"count": 0}
        
        outcomes = {}
        for r in records:
            outcomes[r.outcome.value] = outcomes.get(r.outcome.value, 0) + 1
        
        pnls = [r.pnl_bps for r in records]
        
        return {
            "count": len(records),
            "outcomes": outcomes,
            "avg_pnl": statistics.mean(pnls),
            "median_pnl": statistics.median(pnls),
            "win_rate": sum(1 for r in records if r.pnl_bps > 50) / len(records),
            "false_breakout_rate": sum(1 for r in records if r.is_false_breakout) / len(records),
            "consecutive_failures": self._consecutive_failures,
            "in_caution_mode": self._in_caution_mode,
        }
    
    def force_exit_caution(self, reason: str = "Manual override"):
        """Force exit from caution mode."""
        if self._in_caution_mode:
            self._exit_caution_mode(reason)
    
    def reset(self):
        """Full reset."""
        self._records.clear()
        self._consecutive_failures = 0
        self._consecutive_false_breakouts = 0
        self._consecutive_wins = 0
        self._in_caution_mode = False
        self._caution_start_time = None
        self._recovery_progress = 1.0
        self._density_boost = 0.0
        self._tranche_delay = 0

    def to_dict(self) -> Dict:
        """Serialize key state for restart persistence."""
        return {
            "consecutive_failures": self._consecutive_failures,
            "consecutive_false_breakouts": self._consecutive_false_breakouts,
            "consecutive_wins": self._consecutive_wins,
            "in_caution_mode": self._in_caution_mode,
            "caution_start_time": self._caution_start_time.isoformat() if self._caution_start_time else None,
            "recovery_progress": self._recovery_progress,
            "density_boost": self._density_boost,
            "tranche_delay": self._tranche_delay,
        }

    def from_dict(self, data: Dict):
        """Restore state from persistence."""
        self._consecutive_failures = data.get("consecutive_failures", 0)
        self._consecutive_false_breakouts = data.get("consecutive_false_breakouts", 0)
        self._consecutive_wins = data.get("consecutive_wins", 0)
        self._in_caution_mode = data.get("in_caution_mode", False)
        if data.get("caution_start_time"):
            try:
                self._caution_start_time = datetime.fromisoformat(data["caution_start_time"])
            except (ValueError, TypeError):
                pass
        self._recovery_progress = data.get("recovery_progress", 1.0)
        self._density_boost = data.get("density_boost", 0.0)
        self._tranche_delay = data.get("tranche_delay", 0)
        logger.info(
            f"[FAILURE_MEMORY] State restored: failures={self._consecutive_failures}, "
            f"caution={self._in_caution_mode}, boost={self._density_boost}"
        )


# Singleton
_failure_memory: Optional[FailureAwareMetaMemory] = None

def get_failure_memory() -> FailureAwareMetaMemory:
    global _failure_memory
    if _failure_memory is None:
        _failure_memory = FailureAwareMetaMemory()
    return _failure_memory

def reset_failure_memory():
    global _failure_memory
    _failure_memory = None
