"""
================================================================================
OVERRIDE 4: WEEKEND / HOLIDAY EXPOSURE PHILOSOPHY
================================================================================
Version: 3.3-HR
Purpose: Separate institutional calendar logic from personal crypto trading

RULE:
    For crypto assets:
    - Weekend exposure reduction applies ONLY in NORMAL mode
    - In OPPORTUNITY mode, weekend is NOT a veto
    - Same logic for holidays and low-liquidity hours

WHY THIS EXISTS:
    Crypto markets trade 24/7/365. "Weekend" is an institutional equity concept.
    
    Standard v3.3 behavior:
    - Weekend: Force 50% cap regardless of opportunity
    - Holiday: Force 30% cap regardless of opportunity
    
    This is wrong for personal crypto trading because:
    1. Major crypto moves often happen on weekends (fewer bots, less arbitrage)
    2. Weekend liquidity on major exchanges (Kraken, Binance) is adequate
    3. Personal traders can monitor 24/7 with alerts
    
    HIGH-RISK OVERRIDE:
    - NORMAL mode: Keep conservative caps (institutional discipline)
    - OPPORTUNITY mode: Trade at full capacity (capture the move)

RISK ACKNOWLEDGMENT:
    Weekend gaps CAN happen in crypto (exchange downtime, flash crashes).
    We accept this risk in exchange for not missing weekend opportunities.
    Hard stops remain in place as catastrophic protection.
================================================================================
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
from datetime import datetime

from market.high_risk_config import get_hr_config, OverrideMode
# Import from correct liquidity modules
from liquidity.weekend_manager import WeekendLiquidityManager, WeekendConfig, WeekendState
from liquidity.integrated_manager import LiquidityPeriod
# Aliases for compatibility
LiquidityConfig = WeekendConfig
LiquidityState = WeekendState

logger = logging.getLogger(__name__)


@dataclass
class WeekendOverrideState:
    """State of weekend override logic."""
    
    # Current period
    liquidity_period: LiquidityPeriod = LiquidityPeriod.NORMAL
    
    # System mode
    system_mode: str = "NORMAL"
    
    # Standard cap (from base weekend manager)
    standard_cap: float = 1.50
    
    # Override cap (what we use in OPPORTUNITY)
    override_cap: float = 1.50
    
    # Effective cap in use
    effective_cap: float = 1.50
    
    # Override status
    override_active: bool = False
    override_reason: str = ""
    
    # Per-asset caps
    btc_cap: float = 0.80
    eth_cap: float = 0.80
    sol_cap: float = 1.00
    
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict:
        return {
            "period": self.liquidity_period.value,
            "system_mode": self.system_mode,
            "standard_cap": self.standard_cap,
            "override_cap": self.override_cap,
            "effective_cap": self.effective_cap,
            "override_active": self.override_active,
            "override_reason": self.override_reason,
            "asset_caps": {
                "BTC": self.btc_cap,
                "ETH": self.eth_cap,
                "SOL": self.sol_cap,
            },
        }


class WeekendOverrideManager:
    """
    Manages weekend/holiday exposure based on system mode.
    
    PHILOSOPHY:
    - In NORMAL mode: Be conservative, respect institutional calendar
    - In OPPORTUNITY mode: Be aggressive, crypto doesn't sleep
    
    USAGE:
        manager = WeekendOverrideManager()
        
        state = manager.get_effective_caps(
            system_mode="OPPORTUNITY",
            is_sol_dominant=True,
        )
        
        # Use state.effective_cap for gross exposure limit
        # Use state.sol_cap for SOL-specific limit
    """
    
    def __init__(self, base_manager: Optional[WeekendLiquidityManager] = None):
        self.config = get_hr_config()
        self.cfg = self.config.weekend_config
        
        # Use provided manager or create new one
        self.base_manager = base_manager or WeekendLiquidityManager()
        
        self.state = WeekendOverrideState()
        
        logger.info("WeekendOverrideManager initialized")
    
    def get_effective_caps(
        self,
        system_mode: str,
        is_sol_dominant: bool = False,
        now: Optional[datetime] = None,
    ) -> WeekendOverrideState:
        """
        Get effective exposure caps considering mode.
        
        Args:
            system_mode: NORMAL, OPPORTUNITY, or NO_TRADE
            is_sol_dominant: Whether SOL dominance mode is active
            now: Optional timestamp (for testing)
        
        Returns:
            WeekendOverrideState with effective caps
        """
        
        # Get base state
        base_state = self.base_manager.get_current_state(now)
        
        self.state.liquidity_period = base_state.period
        self.state.system_mode = system_mode
        self.state.standard_cap = base_state.current_cap
        self.state.timestamp = now or datetime.utcnow()
        
        # Check if override is enabled
        if self.config.weekend_override != OverrideMode.ENABLED:
            # Use standard caps
            self.state.effective_cap = base_state.current_cap
            self.state.override_active = False
            self.state.override_reason = "Override disabled"
            self._set_asset_caps(base_state.per_asset_caps, is_sol_dominant, False)
            return self.state
        
        # NO_TRADE: Always use conservative caps
        if system_mode == "NO_TRADE":
            self.state.effective_cap = min(base_state.current_cap, 0.50)
            self.state.override_active = False
            self.state.override_reason = "NO_TRADE mode - conservative"
            self._set_asset_caps(base_state.per_asset_caps, is_sol_dominant, False)
            return self.state
        
        # NORMAL mode: Use standard (conservative) caps
        if system_mode == "NORMAL":
            self.state.effective_cap = base_state.current_cap
            self.state.override_active = False
            self.state.override_reason = "NORMAL mode - standard caps apply"
            self._set_asset_caps(base_state.per_asset_caps, is_sol_dominant, False)
            return self.state
        
        # OPPORTUNITY mode: Apply override
        if system_mode == "OPPORTUNITY":
            override_cap = self._get_opportunity_cap(base_state.period)
            
            self.state.override_cap = override_cap
            self.state.effective_cap = override_cap
            self.state.override_active = True
            self.state.override_reason = (
                f"OPPORTUNITY mode - weekend/holiday NOT a veto "
                f"({base_state.period.value}: {base_state.current_cap:.0%} -> {override_cap:.0%})"
            )
            
            self._set_asset_caps(base_state.per_asset_caps, is_sol_dominant, True)
            
            if base_state.period != LiquidityPeriod.NORMAL:
                logger.info(
                    f"Weekend override ACTIVE: {base_state.period.value} cap "
                    f"{base_state.current_cap:.0%} -> {override_cap:.0%}"
                )
            
            return self.state
        
        # Default fallback
        self.state.effective_cap = base_state.current_cap
        self.state.override_active = False
        return self.state
    
    def _get_opportunity_cap(self, period: LiquidityPeriod) -> float:
        """Get cap for OPPORTUNITY mode by period."""
        
        if period == LiquidityPeriod.WEEKEND:
            return self.cfg["opportunity_mode_cap"]
        
        elif period == LiquidityPeriod.HOLIDAY:
            return self.cfg["holiday_opportunity_cap"]
        
        elif period == LiquidityPeriod.LOW_LIQUIDITY:
            return self.cfg["low_liq_opportunity_cap"]
        
        else:  # NORMAL
            return 1.50  # Full capacity
    
    def _set_asset_caps(
        self,
        base_caps: Dict[str, float],
        is_sol_dominant: bool,
        is_override: bool,
    ):
        """Set per-asset caps."""
        
        if is_override:
            # In override mode, allow full capacity
            self.state.btc_cap = 0.80 if not is_sol_dominant else 0.30
            self.state.eth_cap = 0.80 if not is_sol_dominant else 0.30
            self.state.sol_cap = 1.00
        else:
            # Use base caps (potentially reduced by weekend/holiday)
            self.state.btc_cap = base_caps.get("BTC", 0.80)
            self.state.eth_cap = base_caps.get("ETH", 0.80)
            self.state.sol_cap = base_caps.get("SOL", 0.80)
    
    def is_weekend_reduced(self, system_mode: str) -> Tuple[bool, str]:
        """
        Check if weekend reduction is currently in effect.
        
        Returns:
            (is_reduced, reason)
        """
        state = self.get_effective_caps(system_mode)
        
        if state.liquidity_period == LiquidityPeriod.NORMAL:
            return False, "Not a weekend/holiday period"
        
        if state.override_active:
            return False, state.override_reason
        
        return True, f"{state.liquidity_period.value} reduction active"
    
    def get_period_info(self, now: Optional[datetime] = None) -> Dict:
        """Get information about current period."""
        base_state = self.base_manager.get_current_state(now)
        
        return {
            "period": base_state.period.value,
            "standard_cap": base_state.current_cap,
            "time_until_next": str(base_state.time_until_next_transition),
            "next_period": base_state.next_period.value,
            "override_available": self.config.weekend_override == OverrideMode.ENABLED,
        }


# Singleton
_weekend_override: Optional[WeekendOverrideManager] = None


def get_weekend_override_manager() -> WeekendOverrideManager:
    """Get singleton instance."""
    global _weekend_override
    if _weekend_override is None:
        _weekend_override = WeekendOverrideManager()
    return _weekend_override


def reset_weekend_override_manager():
    """Reset singleton."""
    global _weekend_override
    _weekend_override = None
