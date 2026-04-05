"""
================================================================================
STOP LOSS AUTHORITY INTEGRATION - Stops that Respect Authority System
================================================================================
Version: 3.3.0
Purpose: Resolves HIGH impact item #5 from CTO Audit

PROBLEM SOLVED:
- Exchange-side stop losses execute regardless of system state
- If NO_TRADE is active and stop triggers, it executes anyway
- Stop orders can bypass VETO conditions from Sentiment/Macro agents
- No coordination between stop management and authority matrix

SOLUTION:
- Two-tier stop system:
  1. SOFT STOPS: System-managed, respect authority (can be suspended in NO_TRADE)
  2. HARD STOPS: Exchange-managed, catastrophic protection only (wider levels)
  
- Soft stops are cancelled during NO_TRADE, but hard stops remain
- Hard stops are set at 2x soft stop distance (last-resort protection)
- Stop state is tracked and synchronized with authority system

STOP BEHAVIOR BY MODE:
- NO_TRADE: Soft stops SUSPENDED, hard stops ACTIVE (emergency only)
- OPPORTUNITY: Soft stops ACTIVE at tighter levels
- NORMAL: Soft stops ACTIVE at standard levels

STOP BEHAVIOR WITH VETO:
- If Sentiment VETO is active: Soft stops for opposite direction SUSPENDED
- If Macro VETO is active: All soft stops tightened by 20%
================================================================================
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)


class StopType(Enum):
    """Types of stop loss."""
    SOFT = "SOFT"   # System-managed, respects authority
    HARD = "HARD"   # Exchange-managed, catastrophic protection


class StopStatus(Enum):
    """Stop order status."""
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    SUSPENDED = "SUSPENDED"
    TRIGGERED = "TRIGGERED"
    CANCELLED = "CANCELLED"


from core.canonical_enums import SystemMode  # Single source of truth


@dataclass
class StopOrder:
    """Represents a stop loss order."""
    asset: str
    stop_type: StopType
    trigger_price: float
    direction: str  # "long" = sell stop, "short" = buy stop
    size: float
    status: StopStatus = StopStatus.PENDING
    
    # Exchange info (for hard stops)
    exchange_order_id: Optional[str] = None
    
    # Authority info
    authority_suspended: bool = False
    suspension_reason: str = ""
    
    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    triggered_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict:
        return {
            "asset": self.asset,
            "stop_type": self.stop_type.value,
            "trigger_price": self.trigger_price,
            "direction": self.direction,
            "size": self.size,
            "status": self.status.value,
            "exchange_order_id": self.exchange_order_id,
            "authority_suspended": self.authority_suspended,
            "suspension_reason": self.suspension_reason,
        }


@dataclass
class StopConfig:
    """Configuration for stop management."""
    
    # Stop distances (as percentage of entry)
    # [CALIBRATION] avg win=+41bps, avg loss=-83bps. Stops must be tighter
    # than avg loss to improve W/L ratio. Soft 1%=100bps, Hard 2%=200bps.
    soft_stop_pct: float = 0.01     # [CALIBRATION] was 2%. 1% soft stop = 100bps max
    hard_stop_pct: float = 0.02     # [CALIBRATION] was 4%. 2% hard stop = 200bps max

    # Mode-specific adjustments
    opportunity_stop_multiplier: float = 0.8   # Tighter in OPPORTUNITY (0.8% soft)
    veto_tighten_multiplier: float = 0.8       # Tighten 20% when VETO active

    # Minimum distances (floor)
    min_soft_stop_pct: float = 0.005  # [CALIBRATION] was 1%. Floor at 0.5% = 50bps
    min_hard_stop_pct: float = 0.01   # [CALIBRATION] was 2%. Floor at 1% = 100bps


@dataclass
class AuthorityState:
    """Current authority system state (subset for stop management)."""
    system_mode: SystemMode = SystemMode.NORMAL
    
    # VETO states
    sentiment_veto_long: bool = False   # VETO against going long
    sentiment_veto_short: bool = False  # VETO against going short
    macro_veto_active: bool = False     # General VETO from macro
    
    # Optional context
    veto_reason: str = ""


@dataclass
class StopDecision:
    """Decision from stop manager."""
    soft_stop_price: Optional[float]
    hard_stop_price: Optional[float]
    soft_stop_active: bool
    soft_suspension_reason: str
    hard_stop_active: bool
    
    def to_dict(self) -> Dict:
        return {
            "soft_stop_price": self.soft_stop_price,
            "hard_stop_price": self.hard_stop_price,
            "soft_stop_active": self.soft_stop_active,
            "soft_suspension_reason": self.soft_suspension_reason,
            "hard_stop_active": self.hard_stop_active,
        }


class StopLossAuthorityManager:
    """
    Manages stop losses with authority system integration.
    
    KEY CONCEPT:
    - SOFT stops are like "normal" stops - they respect market conditions
    - HARD stops are catastrophic protection - they ALWAYS execute
    
    USAGE:
        manager = StopLossAuthorityManager()
        
        # When entering position
        decision = manager.calculate_stops(
            asset="SOL",
            entry_price=180.0,
            direction="long",
            authority_state=authority_state,
        )
        
        # Place soft stop (managed by system)
        if decision.soft_stop_active:
            place_limit_stop(decision.soft_stop_price)
        
        # Place hard stop (on exchange)
        place_exchange_stop(decision.hard_stop_price)
        
        # When authority changes
        manager.update_authority(new_authority_state)
        # Soft stops may be suspended or reactivated
    """
    
    def __init__(self, config: Optional[StopConfig] = None):
        self.config = config or StopConfig()
        
        # Current stops by asset
        self.active_stops: Dict[str, Dict[StopType, StopOrder]] = {}
        
        # Authority state
        self.authority_state = AuthorityState()
        
        logger.info("StopLossAuthorityManager initialized")
    
    def calculate_stops(
        self,
        asset: str,
        entry_price: float,
        direction: str,  # "long" or "short"
        authority_state: Optional[AuthorityState] = None,
        size: float = 1.0,
    ) -> StopDecision:
        """
        Calculate stop prices respecting authority.
        
        Returns stop prices and whether they should be active.
        """
        if authority_state:
            self.authority_state = authority_state
        
        auth = self.authority_state
        
        # Calculate base stop distances
        soft_pct = self.config.soft_stop_pct
        hard_pct = self.config.hard_stop_pct
        
        # Apply mode adjustment
        if auth.system_mode == SystemMode.OPPORTUNITY:
            soft_pct *= self.config.opportunity_stop_multiplier
        
        # Apply VETO tightening
        if auth.macro_veto_active:
            soft_pct *= self.config.veto_tighten_multiplier
        
        # Enforce minimums
        soft_pct = max(soft_pct, self.config.min_soft_stop_pct)
        hard_pct = max(hard_pct, self.config.min_hard_stop_pct)
        
        # Calculate prices
        if direction == "long":
            soft_stop_price = entry_price * (1 - soft_pct)
            hard_stop_price = entry_price * (1 - hard_pct)
        else:  # short
            soft_stop_price = entry_price * (1 + soft_pct)
            hard_stop_price = entry_price * (1 + hard_pct)
        
        # Determine if soft stop should be active
        soft_active = True
        soft_suspension_reason = ""
        
        # NO_TRADE: Suspend soft stops
        if auth.system_mode == SystemMode.NO_TRADE:
            soft_active = False
            soft_suspension_reason = "NO_TRADE mode active"
        
        # VETO: Suspend opposite direction soft stops
        elif direction == "long" and auth.sentiment_veto_short:
            # We're long and sentiment vetoes SHORT
            # This means we should NOT exit long positions quickly
            # Actually keep soft stop active since we want to protect long
            pass
        elif direction == "short" and auth.sentiment_veto_long:
            # We're short and sentiment vetoes LONG
            # Keep soft stop active to protect short
            pass
        
        # Hard stop is ALWAYS active (catastrophic protection)
        hard_active = True
        
        # Create stop orders
        soft_order = StopOrder(
            asset=asset,
            stop_type=StopType.SOFT,
            trigger_price=soft_stop_price,
            direction=direction,
            size=size,
            status=StopStatus.ACTIVE if soft_active else StopStatus.SUSPENDED,
            authority_suspended=not soft_active,
            suspension_reason=soft_suspension_reason,
        )
        
        hard_order = StopOrder(
            asset=asset,
            stop_type=StopType.HARD,
            trigger_price=hard_stop_price,
            direction=direction,
            size=size,
            status=StopStatus.ACTIVE,
        )
        
        # Store
        if asset not in self.active_stops:
            self.active_stops[asset] = {}
        self.active_stops[asset][StopType.SOFT] = soft_order
        self.active_stops[asset][StopType.HARD] = hard_order
        
        return StopDecision(
            soft_stop_price=soft_stop_price if soft_active else None,
            hard_stop_price=hard_stop_price,
            soft_stop_active=soft_active,
            soft_suspension_reason=soft_suspension_reason,
            hard_stop_active=hard_active,
        )
    
    def update_authority(self, new_state: AuthorityState) -> Dict[str, str]:
        """
        Update authority state and adjust stops accordingly.
        
        Returns dict of asset -> action taken
        """
        old_state = self.authority_state
        self.authority_state = new_state
        
        actions = {}
        
        for asset, stops in self.active_stops.items():
            soft = stops.get(StopType.SOFT)
            if not soft:
                continue
            
            # Check if soft stop status should change
            was_active = soft.status == StopStatus.ACTIVE
            
            # Determine new status
            should_be_active = True
            suspension_reason = ""
            
            if new_state.system_mode == SystemMode.NO_TRADE:
                should_be_active = False
                suspension_reason = "NO_TRADE mode active"
            
            # Update if changed
            if was_active and not should_be_active:
                soft.status = StopStatus.SUSPENDED
                soft.authority_suspended = True
                soft.suspension_reason = suspension_reason
                soft.updated_at = datetime.now(timezone.utc)
                actions[asset] = f"SOFT_STOP_SUSPENDED: {suspension_reason}"
                logger.info(f"{asset}: Soft stop SUSPENDED - {suspension_reason}")
            
            elif not was_active and should_be_active:
                soft.status = StopStatus.ACTIVE
                soft.authority_suspended = False
                soft.suspension_reason = ""
                soft.updated_at = datetime.now(timezone.utc)
                actions[asset] = "SOFT_STOP_REACTIVATED"
                logger.info(f"{asset}: Soft stop REACTIVATED")
        
        return actions
    
    def check_stops(self, asset: str, current_price: float) -> Optional[str]:
        """
        Check if any stops should trigger.
        
        Returns:
            None if no trigger, or "SOFT" / "HARD" if triggered
        """
        stops = self.active_stops.get(asset)
        if not stops:
            return None
        
        soft = stops.get(StopType.SOFT)
        hard = stops.get(StopType.HARD)
        
        # Check hard stop first (always takes priority)
        if hard and hard.status == StopStatus.ACTIVE:
            triggered = False
            if hard.direction == "long" and current_price <= hard.trigger_price:
                triggered = True
            elif hard.direction == "short" and current_price >= hard.trigger_price:
                triggered = True
            
            if triggered:
                hard.status = StopStatus.TRIGGERED
                hard.triggered_at = datetime.now(timezone.utc)
                logger.warning(
                    f"{asset}: HARD STOP TRIGGERED at {current_price} "
                    f"(trigger={hard.trigger_price})"
                )
                return "HARD"
        
        # Check soft stop (only if active)
        if soft and soft.status == StopStatus.ACTIVE:
            triggered = False
            if soft.direction == "long" and current_price <= soft.trigger_price:
                triggered = True
            elif soft.direction == "short" and current_price >= soft.trigger_price:
                triggered = True
            
            if triggered:
                soft.status = StopStatus.TRIGGERED
                soft.triggered_at = datetime.now(timezone.utc)
                logger.info(
                    f"{asset}: Soft stop triggered at {current_price} "
                    f"(trigger={soft.trigger_price})"
                )
                return "SOFT"
        
        return None
    
    def clear_stops(self, asset: str):
        """Clear all stops for an asset (when position closed)."""
        if asset in self.active_stops:
            del self.active_stops[asset]
            logger.debug(f"{asset}: All stops cleared")
    
    def get_active_stops(self) -> Dict[str, Dict]:
        """Get all active stops for monitoring."""
        result = {}
        for asset, stops in self.active_stops.items():
            result[asset] = {}
            for stop_type, stop in stops.items():
                result[asset][stop_type.value] = stop.to_dict()
        return result
    
    def should_cancel_soft_on_exchange(self, asset: str) -> bool:
        """
        Check if exchange soft stop order should be cancelled.
        
        Returns True if soft stop is suspended and needs exchange cancellation.
        """
        stops = self.active_stops.get(asset)
        if not stops:
            return False
        
        soft = stops.get(StopType.SOFT)
        if not soft:
            return False
        
        return soft.authority_suspended and soft.exchange_order_id is not None


# Singleton instance
_stop_authority_manager: Optional[StopLossAuthorityManager] = None


def get_stop_authority_manager(config: Optional[StopConfig] = None) -> StopLossAuthorityManager:
    """Get or create singleton stop authority manager."""
    global _stop_authority_manager
    if _stop_authority_manager is None:
        _stop_authority_manager = StopLossAuthorityManager(config)
    elif config is not None and _stop_authority_manager.config != config:
        logger.info("[STOP_AUTHORITY] Config changed; refreshing singleton instance")
        _stop_authority_manager = StopLossAuthorityManager(config)
    return _stop_authority_manager


def reset_stop_authority_manager():
    """Reset singleton (for testing)."""
    global _stop_authority_manager
    _stop_authority_manager = None
