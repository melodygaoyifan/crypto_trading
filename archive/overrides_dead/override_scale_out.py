"""
================================================================================
OVERRIDE 5: MINIMAL SCALE-OUT (Exit Alpha)
================================================================================
Version: 3.3-HR
Purpose: Add simple partial profit lock while letting remainder run

RULE:
    Add one non-discretionary scale-out:
    - Exit 25% of position (partial profit lock)
    - Triggered by ANY of:
        1. Momentum stall (2 consecutive bars of flat momentum)
        2. CRACK decay below 50%
        3. Position is profitable by at least 150bps
    - Remainder stays as "runner" with trailing stop

WHY THIS EXISTS:
    Entry is granular (tranches); exit should also have some structure.
    Binary exits (all-or-nothing) leave alpha on the table.
    
    This is NOT a complex trailing system. It's ONE simple rule:
    - Lock in some profit when momentum fades
    - Let the rest run with protection
    
    The goal is psychological as much as financial:
    - Reduces regret when full position exits early
    - Allows participation in extended moves

CONSTRAINTS:
    - Only ONE partial exit (not multiple scale-out levels)
    - No complex trailing logic
    - No discretionary override
================================================================================
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
from datetime import datetime
from enum import Enum

from market.high_risk_config import get_hr_config, OverrideMode

logger = logging.getLogger(__name__)


class ScaleOutStatus(Enum):
    """Scale-out status."""
    NOT_TRIGGERED = "NOT_TRIGGERED"
    TRIGGERED = "TRIGGERED"
    EXECUTED = "EXECUTED"


class ScaleOutTrigger(Enum):
    """What triggered the scale-out."""
    NONE = "NONE"
    MOMENTUM_STALL = "MOMENTUM_STALL"
    CRACK_DECAY = "CRACK_DECAY"
    PROFIT_THRESHOLD = "PROFIT_THRESHOLD"


@dataclass
class PositionMetrics:
    """Metrics for scale-out evaluation."""
    asset: str
    direction: str  # "long" or "short"
    entry_price: float
    current_price: float
    position_size: float
    
    # Profit calculation
    unrealized_pnl_bps: float = 0.0
    
    # Momentum
    momentum_values: List[float] = field(default_factory=list)  # Last N bars
    
    # CRACK
    crack_weight: float = 1.0
    
    # State
    bars_held: int = 0
    
    def calculate_pnl_bps(self) -> float:
        """Calculate unrealized P&L in basis points."""
        if self.entry_price <= 0:
            return 0.0
        
        if self.direction == "long":
            return ((self.current_price - self.entry_price) / self.entry_price) * 10000
        else:
            return ((self.entry_price - self.current_price) / self.entry_price) * 10000


@dataclass
class ScaleOutDecision:
    """Decision from scale-out evaluation."""
    
    # Core decision
    should_scale_out: bool = False
    status: ScaleOutStatus = ScaleOutStatus.NOT_TRIGGERED
    trigger: ScaleOutTrigger = ScaleOutTrigger.NONE
    
    # Scale-out details
    exit_percentage: float = 0.0
    exit_size: float = 0.0
    remaining_size: float = 0.0
    
    # Runner configuration
    runner_stop_price: float = 0.0
    runner_trail_pct: float = 0.02
    
    # Metrics that led to decision
    momentum_stalled: bool = False
    crack_below_threshold: bool = False
    profit_above_threshold: bool = False
    current_pnl_bps: float = 0.0
    
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict:
        return {
            "should_scale_out": self.should_scale_out,
            "status": self.status.value,
            "trigger": self.trigger.value,
            "exit_percentage": self.exit_percentage,
            "exit_size": self.exit_size,
            "remaining_size": self.remaining_size,
            "runner_stop_price": self.runner_stop_price,
            "current_pnl_bps": self.current_pnl_bps,
            "conditions": {
                "momentum_stalled": self.momentum_stalled,
                "crack_below_threshold": self.crack_below_threshold,
                "profit_above_threshold": self.profit_above_threshold,
            },
        }


class MinimalScaleOut:
    """
    Simple scale-out manager for exit alpha.
    
    ONE RULE:
    - Exit 25% when momentum stalls, CRACK decays, or minimum profit reached
    - Remainder runs with trailing stop
    
    USAGE:
        scale_out = MinimalScaleOut()
        
        # On each bar, evaluate
        decision = scale_out.evaluate(
            metrics=PositionMetrics(
                asset="SOL",
                direction="long",
                entry_price=180.0,
                current_price=185.0,
                position_size=1000.0,
                momentum_values=[0.05, 0.02, 0.01],  # Last 3 bars
                crack_weight=0.45,
                bars_held=3,
            )
        )
        
        if decision.should_scale_out:
            # Execute partial exit
            execute_exit(decision.exit_size)
            # Set trailing stop for runner
            set_trailing_stop(decision.runner_stop_price, decision.runner_trail_pct)
    """
    
    def __init__(self):
        self.config = get_hr_config()
        self.cfg = self.config.scale_out_config
        
        # Track which positions have already scaled out
        self._scaled_out: Dict[str, bool] = {}
        
        logger.info("MinimalScaleOut initialized")
    
    def evaluate(
        self,
        metrics: PositionMetrics,
    ) -> ScaleOutDecision:
        """
        Evaluate whether to scale out of position.
        
        Returns decision with exit details if triggered.
        """
        
        decision = ScaleOutDecision()
        
        # Check if override is enabled
        if self.config.scale_out_override != OverrideMode.ENABLED:
            return decision
        
        # Check if already scaled out of this position
        if self._scaled_out.get(metrics.asset, False):
            decision.status = ScaleOutStatus.EXECUTED
            return decision
        
        # Calculate P&L
        pnl_bps = metrics.calculate_pnl_bps()
        decision.current_pnl_bps = pnl_bps
        
        # Check trigger conditions
        triggers = self.cfg["triggers"]
        
        # Condition 1: Momentum stall
        if triggers["momentum_stall"]:
            decision.momentum_stalled = self._check_momentum_stall(metrics.momentum_values)
        
        # Condition 2: CRACK decay below threshold
        if triggers["crack_decay_below"]:
            crack_threshold = triggers["crack_decay_below"]
            decision.crack_below_threshold = metrics.crack_weight < crack_threshold
        
        # Condition 3: Minimum profit reached
        profit_threshold = triggers["profit_threshold_bps"]
        decision.profit_above_threshold = pnl_bps >= profit_threshold
        
        # Determine if scale-out triggered
        # Needs profit threshold PLUS at least one momentum/crack condition
        if decision.profit_above_threshold:
            if decision.momentum_stalled:
                decision.should_scale_out = True
                decision.trigger = ScaleOutTrigger.MOMENTUM_STALL
            elif decision.crack_below_threshold:
                decision.should_scale_out = True
                decision.trigger = ScaleOutTrigger.CRACK_DECAY
        
        # Can also trigger on just meeting profit threshold if very high
        if pnl_bps >= profit_threshold * 2:  # 300bps = automatic partial lock
            decision.should_scale_out = True
            decision.trigger = ScaleOutTrigger.PROFIT_THRESHOLD
        
        # If triggered, calculate exit details
        if decision.should_scale_out:
            decision.status = ScaleOutStatus.TRIGGERED
            decision.exit_percentage = self.cfg["partial_exit_pct"]
            decision.exit_size = metrics.position_size * decision.exit_percentage
            decision.remaining_size = metrics.position_size - decision.exit_size
            
            # Calculate trailing stop for runner
            decision.runner_trail_pct = self.cfg["runner_trail_pct"]
            decision.runner_stop_price = self._calculate_trail_stop(
                current_price=metrics.current_price,
                direction=metrics.direction,
                trail_pct=decision.runner_trail_pct,
            )
            
            logger.info(
                f"SCALE-OUT TRIGGERED for {metrics.asset}: "
                f"trigger={decision.trigger.value}, pnl={pnl_bps:.0f}bps, "
                f"exit {decision.exit_percentage:.0%} ({decision.exit_size:.2f})"
            )
        
        return decision
    
    def _check_momentum_stall(self, momentum_values: List[float]) -> bool:
        """Check if momentum has stalled."""
        
        stall_bars = self.cfg["momentum_stall_bars"]
        stall_threshold = self.cfg["momentum_stall_threshold"]
        
        if len(momentum_values) < stall_bars:
            return False
        
        # Check last N bars for low momentum
        recent = momentum_values[-stall_bars:]
        
        # All bars must have low momentum change
        for m in recent:
            if abs(m) > stall_threshold:
                return False
        
        return True
    
    def _calculate_trail_stop(
        self,
        current_price: float,
        direction: str,
        trail_pct: float,
    ) -> float:
        """Calculate trailing stop price."""
        
        if direction == "long":
            return current_price * (1 - trail_pct)
        else:
            return current_price * (1 + trail_pct)
    
    def mark_scaled_out(self, asset: str):
        """Mark position as having scaled out (prevent multiple scale-outs)."""
        self._scaled_out[asset] = True
        logger.debug(f"{asset} marked as scaled out")
    
    def reset_position(self, asset: str):
        """Reset scale-out tracking for position (when position closed)."""
        self._scaled_out.pop(asset, None)
        logger.debug(f"{asset} scale-out tracking reset")
    
    def get_runner_status(self, asset: str) -> Dict:
        """Get status of runner position."""
        return {
            "asset": asset,
            "has_scaled_out": self._scaled_out.get(asset, False),
            "runner_active": self._scaled_out.get(asset, False),
        }


# Singleton
_scale_out: Optional[MinimalScaleOut] = None


def get_minimal_scale_out() -> MinimalScaleOut:
    """Get singleton instance."""
    global _scale_out
    if _scale_out is None:
        _scale_out = MinimalScaleOut()
    return _scale_out


def reset_minimal_scale_out():
    """Reset singleton."""
    global _scale_out
    _scale_out = None
