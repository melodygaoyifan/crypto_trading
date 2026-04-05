"""
================================================================================
HMATS v3.5 - FUSION PATIENCE & EXECUTION DEADLOCK RESOLUTION
================================================================================
Version: 3.5.0
Upgrade From: v3.4 Authority Fusion
Purpose: Detect and resolve execution-induced alpha suppression

PROBLEM THIS SOLVES:
    In v3.4, if Execution repeatedly delays/downgrades:
        - Fusion generates TradeIntent
        - Execution says "wait, conditions poor"
        - Fusion generates again next bar
        - Execution says "still poor"
        - Opportunity expires, alpha lost
    
    This is "want to trade but never trades" - alpha suppression.

SOLUTION IN v3.5:
    1. Track execution delay/downgrade streaks
    2. After N consecutive delays, Fusion detects deadlock
    3. Fusion makes binary decision:
       - Force aggressive execution (if edge >> friction)
       - Abort the opportunity (don't wait indefinitely)

DETECTION THRESHOLD:
    - Consecutive delays: 2 bars (at 4H, this is 8 hours)
    - Edge vs friction comparison: edge > 3x estimated friction

DECISION LOGIC:
    if consecutive_delays >= 2:
        if signal_edge > 3 * estimated_friction:
            FORCE_AGGRESSIVE_EXECUTION
        else:
            ABORT_OPPORTUNITY

WHY THIS AVOIDS "NEVER TRADES":
    - Forces a decision after 2 bars
    - Either commits with conviction OR explicitly aborts
    - No infinite waiting
================================================================================
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)


from core.canonical_enums import DeadlockResolution  # Single source of truth


@dataclass
class ExecutionDelayRecord:
    """Record of an execution delay."""
    
    timestamp: datetime
    asset: str
    intended_action: str  # "ENTRY", "EXIT", "ESCALATION"
    delay_reason: str     # "poor_timing_score", "high_vpin", "spread_too_wide"
    timing_score: float
    estimated_friction_bps: float


@dataclass
class DeadlockState:
    """Current deadlock state for an asset."""
    
    asset: str
    consecutive_delays: int = 0
    cumulative_edge_lost_bps: float = 0.0
    first_delay_timestamp: Optional[datetime] = None
    last_delay_reason: str = ""
    
    # Signal context at first delay
    original_signal_direction: float = 0.0
    original_signal_edge_bps: float = 0.0
    
    def reset(self):
        self.consecutive_delays = 0
        self.cumulative_edge_lost_bps = 0.0
        self.first_delay_timestamp = None
        self.last_delay_reason = ""
        self.original_signal_direction = 0.0
        self.original_signal_edge_bps = 0.0


@dataclass
class DeadlockResolutionResult:
    """Result of deadlock resolution."""
    
    resolution: DeadlockResolution
    
    # Context
    consecutive_delays: int = 0
    estimated_edge_bps: float = 0.0
    estimated_friction_bps: float = 0.0
    
    # Decision factors
    edge_to_friction_ratio: float = 0.0
    force_reason: str = ""
    abort_reason: str = ""
    
    # Execution override
    force_execution_mode: Optional[str] = None  # "AGGRESSIVE" if forcing
    
    def to_dict(self) -> Dict:
        return {
            "resolution": self.resolution.value,
            "delays": self.consecutive_delays,
            "edge_bps": self.estimated_edge_bps,
            "friction_bps": self.estimated_friction_bps,
            "ratio": self.edge_to_friction_ratio,
        }


@dataclass
class FusionPatienceConfig:
    """Configuration for fusion patience."""
    
    # Deadlock detection
    max_consecutive_delays: int = 2  # After 2 delays, force decision
    
    # Edge vs friction thresholds
    force_aggressive_ratio: float = 3.0  # edge > 3x friction = force
    abort_ratio: float = 1.5             # edge < 1.5x friction = abort
    
    # Minimum edge to force
    min_force_edge_bps: float = 30.0     # Need at least 30bps edge to force
    
    # Maximum wait
    max_wait_hours: float = 12.0         # At 4H, this is 3 bars
    
    # Edge decay per bar
    edge_decay_per_bar_pct: float = 15.0  # Assume 15% edge decay per bar


class FusionPatienceManager:
    """
    Fusion patience manager for v3.5.
    
    Detects execution-induced deadlocks and forces resolution.
    
    USAGE:
        patience = FusionPatienceManager()
        
        # In Fusion, after Execution returns delay
        if execution_result.delay_bars > 0:
            patience.record_delay(
                asset="SOL",
                intended_action="ENTRY",
                delay_reason=execution_result.delay_reason,
                timing_score=execution_result.timing_score,
                estimated_friction_bps=execution_result.estimated_friction_bps,
                signal_edge_bps=fusion_signal_edge,
            )
        
        # Check for deadlock resolution
        resolution = patience.check_deadlock("SOL")
        
        if resolution.resolution == DeadlockResolution.FORCE_AGGRESSIVE:
            # Override execution mode to AGGRESSIVE
            trade_intent.execution_mode = ExecutionMode.AGGRESSIVE
            trade_intent.force_execution = True
        
        elif resolution.resolution == DeadlockResolution.ABORT_OPPORTUNITY:
            # Cancel the opportunity
            trade_intent = None
            patience.reset_state("SOL")
    
    WHERE INTEGRATED:
        - authority_fusion_v34.py: After receiving execution feasibility
        - integration_v34.py: In main decide() loop
    """
    
    def __init__(self, config: Optional[FusionPatienceConfig] = None):
        self.config = config or FusionPatienceConfig()
        
        # State by asset
        self._states: Dict[str, DeadlockState] = {}
        
        # Delay history
        self._delay_history: list = []
        
        logger.info("FusionPatienceManager v3.5 initialized")
    
    def record_delay(
        self,
        asset: str,
        intended_action: str,
        delay_reason: str,
        timing_score: float,
        estimated_friction_bps: float,
        signal_direction: float = 0.0,
        signal_edge_bps: float = 0.0,
    ):
        """
        Record an execution delay.
        
        Called when Execution returns delay > 0.
        """
        
        record = ExecutionDelayRecord(
            timestamp=datetime.now(timezone.utc),
            asset=asset,
            intended_action=intended_action,
            delay_reason=delay_reason,
            timing_score=timing_score,
            estimated_friction_bps=estimated_friction_bps,
        )
        
        self._delay_history.append(record)
        
        # Update state
        if asset not in self._states:
            self._states[asset] = DeadlockState(asset=asset)
        
        state = self._states[asset]
        state.consecutive_delays += 1
        state.last_delay_reason = delay_reason
        
        if state.consecutive_delays == 1:
            state.first_delay_timestamp = datetime.now(timezone.utc)
            state.original_signal_direction = signal_direction
            state.original_signal_edge_bps = signal_edge_bps
        
        # Accumulate edge lost (decay)
        edge_decay = state.original_signal_edge_bps * (self.config.edge_decay_per_bar_pct / 100)
        state.cumulative_edge_lost_bps += edge_decay
        
        logger.info(
            f"FusionPatience: Delay #{state.consecutive_delays} for {asset} - "
            f"{delay_reason}, timing={timing_score:.2f}"
        )
    
    def record_execution(self, asset: str):
        """
        Record successful execution - resets delay state.
        
        Called when Execution successfully fills.
        """
        
        if asset in self._states:
            state = self._states[asset]
            if state.consecutive_delays > 0:
                logger.info(
                    f"FusionPatience: {asset} executed after {state.consecutive_delays} delays"
                )
            state.reset()
    
    def check_deadlock(
        self,
        asset: str,
        current_edge_bps: float,
        current_friction_bps: float,
    ) -> DeadlockResolutionResult:
        """
        Check for deadlock and return resolution.
        
        DECISION LOGIC:
            if consecutive_delays >= 2:
                remaining_edge = original_edge - decay
                ratio = remaining_edge / friction
                
                if ratio > 3.0 and remaining_edge > 30bps:
                    FORCE_AGGRESSIVE
                elif ratio < 1.5:
                    ABORT_OPPORTUNITY
                else:
                    WAIT_ONE_MORE (but only once)
        """
        
        result = DeadlockResolutionResult(resolution=DeadlockResolution.NONE)
        
        if asset not in self._states:
            return result
        
        state = self._states[asset]
        result.consecutive_delays = state.consecutive_delays
        
        if state.consecutive_delays < self.config.max_consecutive_delays:
            return result
        
        # Calculate remaining edge
        remaining_edge = state.original_signal_edge_bps - state.cumulative_edge_lost_bps
        remaining_edge = max(0, remaining_edge)
        
        result.estimated_edge_bps = remaining_edge
        result.estimated_friction_bps = current_friction_bps
        
        # Calculate ratio
        if current_friction_bps > 0:
            ratio = remaining_edge / current_friction_bps
        else:
            ratio = float('inf') if remaining_edge > 0 else 0
        
        result.edge_to_friction_ratio = ratio
        
        # Check max wait time
        if state.first_delay_timestamp:
            hours_waiting = (datetime.now(timezone.utc) - state.first_delay_timestamp).total_seconds() / 3600
            if hours_waiting >= self.config.max_wait_hours:
                result.resolution = DeadlockResolution.ABORT_OPPORTUNITY
                result.abort_reason = f"Max wait time exceeded ({hours_waiting:.1f}h)"
                logger.warning(f"FusionPatience: ABORT {asset} - max wait exceeded")
                return result
        
        # Decision based on ratio
        if ratio >= self.config.force_aggressive_ratio and remaining_edge >= self.config.min_force_edge_bps:
            result.resolution = DeadlockResolution.FORCE_AGGRESSIVE
            result.force_execution_mode = "AGGRESSIVE"
            result.force_reason = f"Edge ({remaining_edge:.0f}bps) >> friction ({current_friction_bps:.0f}bps)"
            logger.warning(
                f"FusionPatience: FORCE AGGRESSIVE for {asset} - "
                f"ratio={ratio:.1f}, edge={remaining_edge:.0f}bps"
            )
        
        elif ratio < self.config.abort_ratio:
            result.resolution = DeadlockResolution.ABORT_OPPORTUNITY
            result.abort_reason = f"Edge ({remaining_edge:.0f}bps) too small vs friction ({current_friction_bps:.0f}bps)"
            logger.warning(
                f"FusionPatience: ABORT {asset} - ratio={ratio:.1f}, edge decayed"
            )
        
        else:
            # In between - allow one more wait if we haven't exceeded max delays by much
            if state.consecutive_delays <= self.config.max_consecutive_delays + 1:
                result.resolution = DeadlockResolution.WAIT_ONE_MORE
                logger.info(f"FusionPatience: WAIT ONE MORE for {asset} - ratio={ratio:.1f}")
            else:
                result.resolution = DeadlockResolution.ABORT_OPPORTUNITY
                result.abort_reason = f"Too many delays ({state.consecutive_delays})"
        
        return result
    
    def reset_state(self, asset: str):
        """Reset state for an asset."""
        if asset in self._states:
            self._states[asset].reset()
    
    def get_state(self, asset: str) -> Optional[DeadlockState]:
        """Get current state for an asset."""
        return self._states.get(asset)
    
    def get_statistics(self) -> Dict:
        """Get patience statistics."""
        
        return {
            "assets_with_delays": [
                a for a, s in self._states.items() if s.consecutive_delays > 0
            ],
            "total_delays_recorded": len(self._delay_history),
            "states": {
                a: {
                    "delays": s.consecutive_delays,
                    "edge_lost": s.cumulative_edge_lost_bps,
                }
                for a, s in self._states.items()
            },
        }
    
    def reset(self):
        """Full reset."""
        self._states.clear()
        self._delay_history.clear()


# Singleton
_patience_manager: Optional[FusionPatienceManager] = None

def get_patience_manager() -> FusionPatienceManager:
    global _patience_manager
    if _patience_manager is None:
        _patience_manager = FusionPatienceManager()
    return _patience_manager

def reset_patience_manager():
    global _patience_manager
    _patience_manager = None
