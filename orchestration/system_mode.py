"""
HMATS v3.2 - System Mode State Machine
=======================================
Implements the Strategy Constitution Mode rules.

Modes (precedence order):
1. NO_TRADE (highest) - Hard flat, zero execution
2. OPPORTUNITY - Aggressive window, reduced thresholds
3. NORMAL (default) - Standard operation

All transitions are explicit and testable.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timezone, timedelta
import logging

logger = logging.getLogger(__name__)


from core.canonical_enums import SystemMode  # Single source of truth (string values)


class ModeTransitionReason(Enum):
    """Reasons for mode transitions (for audit trail)."""
    # NO_TRADE entry reasons
    ALL_CONFLICT_FLAT = auto()
    DATA_INTEGRITY_FAIL = auto()
    STALE_DATA = auto()
    EXTREME_DVOL = auto()
    LIQUIDITY_CRITICAL = auto()
    CORRELATION_COLLAPSE_RISK = auto()
    FLASH_CRASH = auto()
    EXECUTION_BLOCKED = auto()
    FEED_DISAGREEMENT = auto()
    
    # OPPORTUNITY entry reasons
    CRACK_WINDOW = auto()
    LEAD_LAG_EDGE = auto()
    VOLATILITY_EXPANSION = auto()
    SENTIMENT_SHOCK = auto()
    SOL_FLOW_SURGE = auto()
    
    # Exit reasons
    TTL_EXPIRED = auto()
    EDGE_EXHAUSTED = auto()
    CONDITIONS_RESOLVED = auto()
    OPPOSING_CRACK = auto()
    
    # Default
    DEFAULT_STATE = auto()


@dataclass
class ModeState:
    """Current mode state with metadata."""
    mode: SystemMode = SystemMode.NORMAL
    entered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    reason: ModeTransitionReason = ModeTransitionReason.DEFAULT_STATE
    ttl_expires_at: Optional[datetime] = None
    decay_strength: float = 1.0  # For OPPORTUNITY decay
    
    # NO_TRADE tracking
    bars_since_trigger: int = 0
    trigger_conditions: List[str] = field(default_factory=list)
    
    # OPPORTUNITY tracking
    opportunity_triggers_active: List[str] = field(default_factory=list)
    edge_exhaustion_count: int = 0
    
    def is_no_trade(self) -> bool:
        return self.mode == SystemMode.NO_TRADE
    
    def is_opportunity(self) -> bool:
        return self.mode == SystemMode.OPPORTUNITY
    
    def is_normal(self) -> bool:
        return self.mode == SystemMode.NORMAL
    
    def time_in_mode_hours(self) -> float:
        return (datetime.now(timezone.utc) - self.entered_at).total_seconds() / 3600
    
    def is_ttl_expired(self) -> bool:
        if self.ttl_expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.ttl_expires_at


@dataclass
class ModeTransition:
    """Record of a mode transition."""
    from_mode: SystemMode
    to_mode: SystemMode
    reason: ModeTransitionReason
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    details: str = ""


class SystemModeManager:
    """
    _DEPRECATED (T24): Not instantiated by main.py or integration_v36.py.
    Mode logic lives in HMATSv36Engine._determine_system_mode().
    Kept for archive/ imports. Do not add new consumers.
    """
    
    # TTL for OPPORTUNITY mode: 8 hours
    OPPORTUNITY_TTL_HOURS = 8
    
    # Decay per 4H bar in OPPORTUNITY
    OPPORTUNITY_DECAY_PER_BAR = 0.30
    
    # Bars required to exit NO_TRADE
    NO_TRADE_EXIT_BARS = 2
    
    # Edge exhaustion threshold (consecutive low-edge packets)
    EDGE_EXHAUSTION_THRESHOLD = 3
    EDGE_EXHAUSTION_BPS = 5
    
    def __init__(self):
        self.state = ModeState()
        self.history: List[ModeTransition] = []
        self._last_4h_bar: Optional[datetime] = None
    
    def get_mode(self) -> SystemMode:
        """Get current system mode."""
        return self.state.mode
    
    def get_state(self) -> ModeState:
        """Get full mode state."""
        return self.state
    
    def check_and_update(
        self,
        no_trade_triggers: Dict[str, bool],
        opportunity_triggers: Dict[str, bool],
        lead_lag_edge_bps: float = 0,
        is_4h_bar: bool = False
    ) -> Tuple[SystemMode, Optional[ModeTransitionReason]]:
        """
        Check triggers and update mode.
        
        Called on every tick. Mode transitions happen based on:
        1. NO_TRADE triggers (immediate)
        2. OPPORTUNITY triggers (if not in NO_TRADE)
        3. TTL/decay (for OPPORTUNITY)
        4. Exit conditions
        
        Returns: (new_mode, transition_reason if changed)
        """
        
        old_mode = self.state.mode
        transition_reason = None
        
        # Update 4H bar tracking
        if is_4h_bar:
            self._on_4h_bar()
        
        # === PRIORITY 1: Check NO_TRADE triggers ===
        no_trade_triggered, nt_reason = self._check_no_trade_triggers(no_trade_triggers)
        
        if no_trade_triggered:
            if self.state.mode != SystemMode.NO_TRADE:
                self._enter_no_trade(nt_reason, no_trade_triggers)
                transition_reason = nt_reason
        
        elif self.state.mode == SystemMode.NO_TRADE:
            # Check if we can exit NO_TRADE
            if self._can_exit_no_trade(no_trade_triggers):
                self._exit_no_trade()
                transition_reason = ModeTransitionReason.CONDITIONS_RESOLVED
        
        # === PRIORITY 2: Check OPPORTUNITY (if not NO_TRADE) ===
        if self.state.mode != SystemMode.NO_TRADE:
            
            # Check for new OPPORTUNITY triggers
            opp_triggered, opp_reason = self._check_opportunity_triggers(opportunity_triggers)
            
            if opp_triggered and self.state.mode != SystemMode.OPPORTUNITY:
                self._enter_opportunity(opp_reason, opportunity_triggers)
                transition_reason = opp_reason
            
            elif self.state.mode == SystemMode.OPPORTUNITY:
                # Check for OPPORTUNITY exit conditions
                if self._should_exit_opportunity(lead_lag_edge_bps, opportunity_triggers):
                    self._exit_opportunity()
                    transition_reason = ModeTransitionReason.TTL_EXPIRED
        
        # Log transition
        if transition_reason and self.state.mode != old_mode:
            self._record_transition(old_mode, self.state.mode, transition_reason)
        
        return self.state.mode, transition_reason
    
    def _check_no_trade_triggers(
        self,
        triggers: Dict[str, bool]
    ) -> Tuple[bool, Optional[ModeTransitionReason]]:
        """
        Check NO_TRADE triggers in priority order.
        
        From Strategy Constitution Section 3:
        1. Data integrity fail (immediate)
        2. Execution blocked state (immediate)
        3. Stale data > 2s (immediate)
        4. Flash crash detection (immediate)
        5. Extreme DVOL ≥ 5.0 (immediate)
        6. All-Conflict-Flat (next 4H bar)
        7. Correlation collapse risk (next 4H bar)
        8. Liquidity critical (immediate)
        """
        
        # Priority 1: Data integrity
        if triggers.get('data_integrity_fail', False):
            return True, ModeTransitionReason.DATA_INTEGRITY_FAIL
        
        # Priority 2: Execution blocked
        if triggers.get('execution_blocked', False):
            return True, ModeTransitionReason.EXECUTION_BLOCKED
        
        # Priority 3: Stale data
        if triggers.get('stale_data', False):
            return True, ModeTransitionReason.STALE_DATA
        
        # Priority 4: Flash crash
        if triggers.get('flash_crash', False):
            return True, ModeTransitionReason.FLASH_CRASH
        
        # Priority 5: Extreme DVOL
        if triggers.get('extreme_dvol', False):
            return True, ModeTransitionReason.EXTREME_DVOL
        
        # Priority 6: All-Conflict-Flat
        if triggers.get('all_conflict_flat', False):
            return True, ModeTransitionReason.ALL_CONFLICT_FLAT
        
        # Priority 7: Correlation collapse
        if triggers.get('correlation_collapse_risk', False):
            return True, ModeTransitionReason.CORRELATION_COLLAPSE_RISK
        
        # Priority 8: Liquidity critical
        if triggers.get('liquidity_critical', False):
            return True, ModeTransitionReason.LIQUIDITY_CRITICAL
        
        # Priority 9: Feed disagreement
        if triggers.get('feed_disagreement', False):
            return True, ModeTransitionReason.FEED_DISAGREEMENT
        
        return False, None
    
    def _check_opportunity_triggers(
        self,
        triggers: Dict[str, bool]
    ) -> Tuple[bool, Optional[ModeTransitionReason]]:
        """
        Check OPPORTUNITY triggers.
        
        From Strategy Constitution Section 2:
        A) Lead-Lag Edge
        B) CRACK Window
        C) Volatility Expansion
        D) Sentiment Shock
        E) SOL Flow Surge
        """
        
        # Any trigger can enter OPPORTUNITY
        if triggers.get('crack_window', False):
            return True, ModeTransitionReason.CRACK_WINDOW
        
        if triggers.get('lead_lag_edge', False):
            return True, ModeTransitionReason.LEAD_LAG_EDGE
        
        if triggers.get('volatility_expansion', False):
            return True, ModeTransitionReason.VOLATILITY_EXPANSION
        
        if triggers.get('sentiment_shock', False):
            return True, ModeTransitionReason.SENTIMENT_SHOCK
        
        if triggers.get('sol_flow_surge', False):
            return True, ModeTransitionReason.SOL_FLOW_SURGE
        
        return False, None
    
    def _enter_no_trade(self, reason: ModeTransitionReason, triggers: Dict[str, bool]):
        """Enter NO_TRADE mode."""
        logger.warning(f"Entering NO_TRADE: {reason.name}")
        
        self.state = ModeState(
            mode=SystemMode.NO_TRADE,
            entered_at=datetime.now(timezone.utc),
            reason=reason,
            ttl_expires_at=None,  # NO_TRADE has no TTL
            bars_since_trigger=0,
            trigger_conditions=[k for k, v in triggers.items() if v]
        )
    
    def _can_exit_no_trade(self, triggers: Dict[str, bool]) -> bool:
        """
        Check if NO_TRADE exit conditions are met.
        
        Requires:
        1. All trigger conditions resolved
        2. 2 consecutive 4H bars with no re-trigger
        """
        
        # Check if any triggers still active
        any_active = any(triggers.values())
        if any_active:
            self.state.bars_since_trigger = 0
            return False
        
        # Need 2 consecutive clear bars
        return self.state.bars_since_trigger >= self.NO_TRADE_EXIT_BARS
    
    def _exit_no_trade(self):
        """Exit NO_TRADE mode to NORMAL."""
        logger.info("Exiting NO_TRADE -> NORMAL")
        
        self.state = ModeState(
            mode=SystemMode.NORMAL,
            entered_at=datetime.now(timezone.utc),
            reason=ModeTransitionReason.CONDITIONS_RESOLVED
        )
    
    def _enter_opportunity(self, reason: ModeTransitionReason, triggers: Dict[str, bool]):
        """Enter OPPORTUNITY mode."""
        logger.info(f"Entering OPPORTUNITY: {reason.name}")
        
        ttl = datetime.now(timezone.utc) + timedelta(hours=self.OPPORTUNITY_TTL_HOURS)
        
        self.state = ModeState(
            mode=SystemMode.OPPORTUNITY,
            entered_at=datetime.now(timezone.utc),
            reason=reason,
            ttl_expires_at=ttl,
            decay_strength=1.0,
            opportunity_triggers_active=[k for k, v in triggers.items() if v],
            edge_exhaustion_count=0
        )
    
    def _should_exit_opportunity(
        self,
        lead_lag_edge_bps: float,
        triggers: Dict[str, bool]
    ) -> bool:
        """
        Check OPPORTUNITY exit conditions.
        
        Exit when ANY of:
        1. TTL expired
        2. NO_TRADE trigger (handled separately)
        3. Edge exhausted (net_edge < 5bps for 3 consecutive packets)
        4. Opposing CRACK detected
        """
        
        # TTL check
        if self.state.is_ttl_expired():
            return True
        
        # Decay strength check (below 20% = expired)
        if self.state.decay_strength < 0.2:
            return True
        
        # Edge exhaustion
        if abs(lead_lag_edge_bps) < self.EDGE_EXHAUSTION_BPS:
            self.state.edge_exhaustion_count += 1
            if self.state.edge_exhaustion_count >= self.EDGE_EXHAUSTION_THRESHOLD:
                return True
        else:
            self.state.edge_exhaustion_count = 0
        
        # Opposing CRACK - if original was long and now short (or vice versa)
        # This would be detected in triggers as a new CRACK in opposite direction
        # Design decision: allow the new CRACK to replace the old one, enabling 
        # quick re-entry in volatile markets rather than forcing a mode exit
        
        return False
    
    def _exit_opportunity(self):
        """Exit OPPORTUNITY mode to NORMAL."""
        logger.info("Exiting OPPORTUNITY -> NORMAL")
        
        self.state = ModeState(
            mode=SystemMode.NORMAL,
            entered_at=datetime.now(timezone.utc),
            reason=ModeTransitionReason.TTL_EXPIRED
        )
    
    def _on_4h_bar(self):
        """Called on each 4H bar close."""
        
        # Update NO_TRADE bar counter
        if self.state.mode == SystemMode.NO_TRADE:
            self.state.bars_since_trigger += 1
            logger.info(f"NO_TRADE: {self.state.bars_since_trigger}/{self.NO_TRADE_EXIT_BARS} bars")
        
        # Apply OPPORTUNITY decay
        elif self.state.mode == SystemMode.OPPORTUNITY:
            self.state.decay_strength *= (1 - self.OPPORTUNITY_DECAY_PER_BAR)
            logger.info(f"OPPORTUNITY decay: {self.state.decay_strength:.1%} remaining")
        
        self._last_4h_bar = datetime.now(timezone.utc)
    
    def _record_transition(
        self,
        from_mode: SystemMode,
        to_mode: SystemMode,
        reason: ModeTransitionReason
    ):
        """Record mode transition for audit."""
        transition = ModeTransition(
            from_mode=from_mode,
            to_mode=to_mode,
            reason=reason,
            details=f"Triggers: {self.state.trigger_conditions or self.state.opportunity_triggers_active}"
        )
        self.history.append(transition)
        
        # Keep last 100 transitions
        if len(self.history) > 100:
            self.history = self.history[-100:]
        
        logger.info(f"Mode transition: {from_mode.name} -> {to_mode.name} ({reason.name})")
    
    def get_allowed_actions(self) -> Dict[str, bool]:
        """
        Get allowed actions for current mode.
        
        From Strategy Constitution Section 1.2.
        """
        
        if self.state.mode == SystemMode.NO_TRADE:
            return {
                'can_open_position': False,
                'can_add_tranche': False,
                'can_close_position': True,  # Must close
                'can_cancel_orders': True,   # Must cancel
                'early_entry_allowed': False,
                'aggressive_execution_allowed': False,
                'sol_dominance_eligible': False,
                'reduced_alpha_threshold': False
            }
        
        elif self.state.mode == SystemMode.OPPORTUNITY:
            return {
                'can_open_position': True,
                'can_add_tranche': True,
                'can_close_position': True,
                'can_cancel_orders': True,
                'early_entry_allowed': True,      # Tranche-1 before 4H close
                'aggressive_execution_allowed': True,
                'sol_dominance_eligible': True,
                'reduced_alpha_threshold': True
            }
        
        else:  # NORMAL
            return {
                'can_open_position': True,
                'can_add_tranche': True,
                'can_close_position': True,
                'can_cancel_orders': True,
                'early_entry_allowed': False,
                'aggressive_execution_allowed': False,  # Passive preferred
                'sol_dominance_eligible': False,
                'reduced_alpha_threshold': False
            }


# Singleton instance
_mode_manager = SystemModeManager()

def get_mode_manager() -> SystemModeManager:
    return _mode_manager
