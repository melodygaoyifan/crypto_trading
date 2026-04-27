"""
================================================================================
HMATS v3.4 - META-DECISION MEMORY
================================================================================
Version: 3.4.0
Type: META-LAYER (not an agent, does not generate signals)
Purpose: Track OPPORTUNITY outcomes and adaptively adjust aggressiveness

FUNCTION:
    - Track recent OPPORTUNITY trade outcomes (success/failure)
    - Temporarily reduce aggressiveness after consecutive failures
    - Automatically recover after success
    - Provide aggressiveness modifier to Fusion (NOT a signal)

CRITICAL CONSTRAINTS:
    - Does NOT generate trades or signals
    - Does NOT override Risk vetoes
    - Only MODIFIES aggressiveness within bounds
    - Auto-recovers (no permanent suppression)

WHY THIS EXISTS:
    Even good systems have losing streaks. Without meta-memory, the system
    will keep firing at full aggressiveness during drawdowns. This creates
    adaptive behavior: pull back after failures, return to normal after recovery.
================================================================================
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone, timedelta
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


class OutcomeType(Enum):
    """Trade outcome classification."""
    WIN = "WIN"           # Profit > min_win_bps
    LOSS = "LOSS"         # Loss > min_loss_bps
    SCRATCH = "SCRATCH"   # Small profit/loss (neutral)
    STOPPED = "STOPPED"   # Hit stop loss


@dataclass
class TradeOutcome:
    """Record of a completed trade."""
    
    trade_id: str
    asset: str
    direction: int  # 1 = long, -1 = short
    
    # Mode at entry
    mode_at_entry: str  # OPPORTUNITY or NORMAL
    phase_at_entry: str
    
    # PnL
    pnl_bps: float
    outcome_type: OutcomeType
    
    # Timing
    entry_time: datetime
    exit_time: datetime
    bars_held: int
    
    # Exit reason
    exit_reason: str  # scale_out, runner_release, stop_hit, signal_exit
    
    def to_dict(self) -> Dict:
        return {
            "trade_id": self.trade_id,
            "asset": self.asset,
            "mode": self.mode_at_entry,
            "pnl_bps": self.pnl_bps,
            "outcome": self.outcome_type.value,
            "bars_held": self.bars_held,
        }


@dataclass
class MetaMemoryConfig:
    """Configuration for meta-decision memory."""
    
    # Outcome classification
    min_win_bps: float = 50.0       # > 50 bps = WIN
    min_loss_bps: float = -50.0     # < -50 bps = LOSS
    
    # Memory window
    memory_window_trades: int = 20   # Track last N trades
    opportunity_memory_trades: int = 10  # OPPORTUNITY-specific
    
    # Aggressiveness adjustment
    consecutive_loss_threshold: int = 3  # After N losses, reduce
    max_aggressiveness_reduction: float = 0.4  # Max 40% reduction
    reduction_per_loss: float = 0.15  # 15% reduction per consecutive loss
    
    # Recovery
    wins_to_recover: int = 2  # N wins to fully recover
    recovery_per_win: float = 0.25  # 25% recovery per win
    
    # Time-based recovery
    auto_recovery_hours: int = 24  # Full recovery after N hours
    
    # Bounds
    min_aggressiveness: float = 0.5  # Never go below 50%
    max_aggressiveness: float = 1.0  # Normal = 100%


@dataclass
class AggressivenessModifier:
    """Output from meta-memory to Fusion."""
    
    # The modifier (0.5 to 1.0)
    modifier: float = 1.0
    
    # Explanation
    reason: str = ""
    consecutive_losses: int = 0
    recent_win_rate: float = 0.5
    
    # Recovery status
    in_recovery: bool = False
    recovery_progress: float = 1.0  # 0 = fully suppressed, 1 = fully recovered
    
    # Time info
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict:
        return {
            "modifier": self.modifier,
            "reason": self.reason,
            "consecutive_losses": self.consecutive_losses,
            "win_rate": self.recent_win_rate,
            "in_recovery": self.in_recovery,
        }


class MetaDecisionMemory:
    """
    Meta-layer for tracking outcomes and adjusting aggressiveness.
    
    USAGE:
        memory = MetaDecisionMemory()
        
        # Record completed trade
        memory.record_outcome(trade_outcome)
        
        # Get aggressiveness modifier for Fusion
        modifier = memory.get_aggressiveness_modifier()
        
        # Apply in Fusion (NOT a signal, just a modifier)
        if mode == OPPORTUNITY:
            effective_aggressiveness = base_aggressiveness * modifier.modifier
    
    CRITICAL: This is NOT an agent. It does not generate signals.
    It only provides a modifier that Fusion can use to scale aggressiveness.
    """
    
    def __init__(self, config: Optional[MetaMemoryConfig] = None):
        self.config = config or MetaMemoryConfig()
        
        # Trade memory
        self._all_trades: deque = deque(maxlen=self.config.memory_window_trades)
        self._opportunity_trades: deque = deque(maxlen=self.config.opportunity_memory_trades)
        
        # Consecutive tracking
        self._consecutive_losses: int = 0
        self._consecutive_wins: int = 0
        
        # Recovery state
        self._in_recovery: bool = False
        self._recovery_progress: float = 1.0
        self._last_loss_time: Optional[datetime] = None
        
        # Current modifier
        self._current_modifier = AggressivenessModifier()
        
        logger.info("MetaDecisionMemory initialized")
    
    def record_outcome(self, outcome: TradeOutcome):
        """
        Record a completed trade outcome.
        
        This updates the internal state and recalculates the
        aggressiveness modifier.
        """
        
        # Add to memory
        self._all_trades.append(outcome)
        
        if outcome.mode_at_entry == "OPPORTUNITY":
            self._opportunity_trades.append(outcome)
        
        # Update consecutive counters
        if outcome.outcome_type == OutcomeType.WIN:
            self._consecutive_wins += 1
            self._consecutive_losses = 0
            self._handle_win()
        
        elif outcome.outcome_type in [OutcomeType.LOSS, OutcomeType.STOPPED]:
            self._consecutive_losses += 1
            self._consecutive_wins = 0
            self._last_loss_time = datetime.now(timezone.utc)
            self._handle_loss()
        
        # SCRATCH doesn't reset counters
        
        # Recalculate modifier
        self._update_modifier()
        
        logger.info(
            f"MetaMemory recorded: {outcome.outcome_type.value} "
            f"({outcome.pnl_bps:+.0f}bps), "
            f"consecutive_losses={self._consecutive_losses}, "
            f"modifier={self._current_modifier.modifier:.2f}"
        )
    
    def _handle_win(self):
        """Handle a winning trade."""
        
        if self._in_recovery:
            # Progress recovery
            self._recovery_progress += self.config.recovery_per_win
            self._recovery_progress = min(self._recovery_progress, 1.0)
            
            if self._consecutive_wins >= self.config.wins_to_recover:
                # Full recovery
                self._in_recovery = False
                self._recovery_progress = 1.0
                logger.info("MetaMemory: Full recovery achieved after wins")
    
    def _handle_loss(self):
        """Handle a losing trade."""
        
        if self._consecutive_losses >= self.config.consecutive_loss_threshold:
            # Enter recovery mode
            self._in_recovery = True
            
            # Calculate suppression
            excess_losses = self._consecutive_losses - self.config.consecutive_loss_threshold + 1
            reduction = excess_losses * self.config.reduction_per_loss
            reduction = min(reduction, self.config.max_aggressiveness_reduction)
            
            self._recovery_progress = 1.0 - reduction
            self._recovery_progress = max(
                self._recovery_progress,
                1.0 - self.config.max_aggressiveness_reduction
            )
            
            logger.warning(
                f"MetaMemory: Entering recovery mode after {self._consecutive_losses} losses, "
                f"progress={self._recovery_progress:.2f}"
            )
    
    def _update_modifier(self):
        """Update the aggressiveness modifier."""
        
        # Check for time-based auto-recovery
        if self._in_recovery and self._last_loss_time:
            hours_since_loss = (datetime.now(timezone.utc) - self._last_loss_time).total_seconds() / 3600
            
            if hours_since_loss >= self.config.auto_recovery_hours:
                self._in_recovery = False
                self._recovery_progress = 1.0
                logger.info("MetaMemory: Auto-recovered after time elapsed")
        
        # Calculate win rate
        recent_opportunity = list(self._opportunity_trades)
        if recent_opportunity:
            wins = sum(1 for t in recent_opportunity if t.outcome_type == OutcomeType.WIN)
            win_rate = wins / len(recent_opportunity)
        else:
            win_rate = 0.5
        
        # Calculate modifier
        if self._in_recovery:
            modifier = self.config.min_aggressiveness + (
                (self.config.max_aggressiveness - self.config.min_aggressiveness) *
                self._recovery_progress
            )
            reason = f"Recovery mode: {self._consecutive_losses} consecutive losses"
        else:
            modifier = self.config.max_aggressiveness
            reason = "Normal operation"
        
        # Ensure bounds
        modifier = max(self.config.min_aggressiveness, min(modifier, self.config.max_aggressiveness))
        
        self._current_modifier = AggressivenessModifier(
            modifier=modifier,
            reason=reason,
            consecutive_losses=self._consecutive_losses,
            recent_win_rate=win_rate,
            in_recovery=self._in_recovery,
            recovery_progress=self._recovery_progress,
        )
    
    def get_aggressiveness_modifier(self) -> AggressivenessModifier:
        """
        Get current aggressiveness modifier for Fusion.
        
        WHERE CONSUMED:
            - authority_fusion_v34.py: Scales OPPORTUNITY aggressiveness
            
        HOW APPLIED:
            if mode == OPPORTUNITY:
                modifier = meta_memory.get_aggressiveness_modifier()
                target_exposure *= modifier.modifier
        
        CRITICAL: This is a MODIFIER, not a SIGNAL.
        It does not decide direction or entry.
        """
        
        # Refresh time-based recovery
        self._update_modifier()
        
        return self._current_modifier
    
    def get_stats(self) -> Dict:
        """Get memory statistics."""
        
        all_trades = list(self._all_trades)
        opp_trades = list(self._opportunity_trades)
        
        def calc_stats(trades: List[TradeOutcome]) -> Dict:
            if not trades:
                return {"count": 0, "win_rate": 0, "avg_pnl": 0}
            
            wins = sum(1 for t in trades if t.outcome_type == OutcomeType.WIN)
            avg_pnl = sum(t.pnl_bps for t in trades) / len(trades)
            
            return {
                "count": len(trades),
                "win_rate": wins / len(trades),
                "avg_pnl": avg_pnl,
            }
        
        return {
            "all_trades": calc_stats(all_trades),
            "opportunity_trades": calc_stats(opp_trades),
            "consecutive_losses": self._consecutive_losses,
            "consecutive_wins": self._consecutive_wins,
            "in_recovery": self._in_recovery,
            "current_modifier": self._current_modifier.modifier,
        }
    
    def force_reset(self):
        """
        Force reset to normal state.
        
        USE CASE: Manual intervention after system review.
        """
        self._consecutive_losses = 0
        self._consecutive_wins = 0
        self._in_recovery = False
        self._recovery_progress = 1.0
        self._update_modifier()
        logger.info("MetaMemory: Force reset to normal state")
    
    def reset(self):
        """Full reset including trade history."""
        self._all_trades.clear()
        self._opportunity_trades.clear()
        self.force_reset()


# Singleton
_meta_memory: Optional[MetaDecisionMemory] = None

def get_meta_memory() -> MetaDecisionMemory:
    global _meta_memory
    if _meta_memory is None:
        _meta_memory = MetaDecisionMemory()
    return _meta_memory

def reset_meta_memory():
    global _meta_memory
    _meta_memory = None
