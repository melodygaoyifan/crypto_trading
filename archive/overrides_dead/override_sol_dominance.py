"""
================================================================================
OVERRIDE 1: SOL DOMINANCE > GLOBAL GROSS CAP (Selective)
================================================================================
Version: 3.3-HR
Purpose: Allow SOL to temporarily exceed global gross cap in OPPORTUNITY mode

RULE:
    When Mode = OPPORTUNITY AND SOL is the primary signal carrier:
    - Allow SOL exposure to temporarily violate global gross cap
    - Provided:
        - BTC/ETH exposure is reduced to near zero (< 10%)
        - Violation duration ≤ OPPORTUNITY TTL (max 2 bars = 8 hours)
    - After TTL expiry -> force normalization

WHY THIS EXISTS:
    Missing a high-conviction SOL move is worse than taking a controlled loss.
    SOL has higher beta, higher volatility, and higher profit potential.
    In OPPORTUNITY windows, we want FULL concentration.

SAFETY:
    - BTC/ETH must be near-flat (< 10% combined)
    - Time-limited violation
    - Forced normalization after TTL
    - Hard stop still applies
================================================================================
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum

from market.high_risk_config import get_hr_config, OverrideMode

logger = logging.getLogger(__name__)


class ViolationStatus(Enum):
    """Status of gross cap violation."""
    NO_VIOLATION = "NO_VIOLATION"
    VIOLATION_ALLOWED = "VIOLATION_ALLOWED"
    VIOLATION_ACTIVE = "VIOLATION_ACTIVE"
    FORCE_NORMALIZE = "FORCE_NORMALIZE"


@dataclass
class SolDominanceState:
    """Current SOL dominance override state."""
    
    # Whether override is active
    override_active: bool = False
    
    # Violation tracking
    violation_status: ViolationStatus = ViolationStatus.NO_VIOLATION
    violation_started: Optional[datetime] = None
    violation_bars_elapsed: int = 0
    
    # Current exposures
    sol_exposure: float = 0.0
    btc_exposure: float = 0.0
    eth_exposure: float = 0.0
    gross_exposure: float = 0.0
    
    # Standard cap
    standard_gross_cap: float = 1.50
    
    # Override cap (during violation)
    override_gross_cap: float = 1.80  # 1.50 + 0.30 violation allowance
    
    # Effective cap in use
    effective_cap: float = 1.50
    
    def to_dict(self) -> Dict:
        return {
            "override_active": self.override_active,
            "violation_status": self.violation_status.value,
            "violation_bars_elapsed": self.violation_bars_elapsed,
            "sol_exposure": self.sol_exposure,
            "btc_eth_combined": self.btc_exposure + self.eth_exposure,
            "gross_exposure": self.gross_exposure,
            "effective_cap": self.effective_cap,
        }


class SolDominanceOverride:
    """
    Manages SOL dominance override of global gross cap.
    
    USAGE:
        override = SolDominanceOverride()
        
        # Check if override allows higher exposure
        result = override.evaluate(
            system_mode="OPPORTUNITY",
            sol_signal=0.8,
            sol_beta=1.5,
            btc_exposure=0.05,
            eth_exposure=0.03,
            sol_exposure=0.90,
        )
        
        if result.override_active:
            # Use result.effective_cap instead of standard cap
            pass
    """
    
    def __init__(self):
        self.config = get_hr_config()
        self.state = SolDominanceState()
        
        logger.info("SolDominanceOverride initialized")
    
    def evaluate(
        self,
        system_mode: str,
        sol_signal: float,
        sol_beta: float,
        sol_flow: float,
        btc_exposure: float,
        eth_exposure: float,
        sol_exposure: float,
        standard_gross_cap: float = 1.50,
    ) -> SolDominanceState:
        """
        Evaluate whether SOL dominance override should be active.
        
        Returns updated state with effective cap.
        """
        
        # Check if override is enabled
        if self.config.sol_dominance_override != OverrideMode.ENABLED:
            return self._standard_state(standard_gross_cap, btc_exposure, eth_exposure, sol_exposure)
        
        cond = self.config.sol_dominance_conditions
        thresh = self.config.sol_primary_signal_thresholds
        
        # Update exposures
        self.state.btc_exposure = btc_exposure
        self.state.eth_exposure = eth_exposure
        self.state.sol_exposure = sol_exposure
        self.state.gross_exposure = abs(btc_exposure) + abs(eth_exposure) + abs(sol_exposure)
        self.state.standard_gross_cap = standard_gross_cap
        
        # Check all conditions for override activation
        conditions_met = True
        
        # Condition 1: Must be in OPPORTUNITY mode
        if cond["require_opportunity_mode"] and system_mode != "OPPORTUNITY":
            conditions_met = False
        
        # Condition 2: SOL must be primary signal carrier
        if cond["require_sol_primary_signal"]:
            if sol_signal < thresh["sol_signal_min"]:
                conditions_met = False
            if sol_beta < thresh["sol_beta_min"]:
                conditions_met = False
            if sol_flow < thresh["sol_flow_min"]:
                conditions_met = False
        
        # Condition 3: BTC/ETH must be reduced
        if cond["require_btc_eth_reduced"]:
            btc_eth_combined = abs(btc_exposure) + abs(eth_exposure)
            if btc_eth_combined > thresh["btc_eth_max_combined"]:
                conditions_met = False
        
        # Condition 4: Check violation duration
        max_bars = cond["violation_duration_bars"]
        if self.state.violation_bars_elapsed >= max_bars:
            # Force normalization
            self.state.violation_status = ViolationStatus.FORCE_NORMALIZE
            self.state.override_active = False
            self.state.effective_cap = standard_gross_cap
            
            logger.warning(
                f"SOL dominance violation TTL expired ({max_bars} bars). "
                f"Forcing normalization to {standard_gross_cap:.0%}"
            )
            return self.state
        
        # Apply override if conditions met
        if conditions_met:
            max_violation = cond["max_violation_pct"]
            self.state.override_gross_cap = standard_gross_cap * (1 + max_violation)
            self.state.effective_cap = self.state.override_gross_cap
            self.state.override_active = True
            
            # Track violation start
            if self.state.violation_status == ViolationStatus.NO_VIOLATION:
                self.state.violation_status = ViolationStatus.VIOLATION_ALLOWED
                self.state.violation_started = datetime.utcnow()
            
            if self.state.gross_exposure > standard_gross_cap:
                self.state.violation_status = ViolationStatus.VIOLATION_ACTIVE
            
            logger.info(
                f"SOL dominance override ACTIVE: cap raised to {self.state.effective_cap:.0%} "
                f"(SOL={sol_exposure:.0%}, signal={sol_signal:.2f}, beta={sol_beta:.2f})"
            )
        else:
            # Conditions not met - use standard cap
            self.state.override_active = False
            self.state.effective_cap = standard_gross_cap
            self.state.violation_status = ViolationStatus.NO_VIOLATION
            self.state.violation_started = None
            self.state.violation_bars_elapsed = 0
        
        return self.state
    
    def on_bar_close(self):
        """Called at each 4H bar close to track violation duration."""
        if self.state.violation_status in [ViolationStatus.VIOLATION_ALLOWED, 
                                            ViolationStatus.VIOLATION_ACTIVE]:
            self.state.violation_bars_elapsed += 1
            logger.debug(f"SOL violation bar count: {self.state.violation_bars_elapsed}")
    
    def force_normalize(self) -> Dict[str, float]:
        """
        Force normalization to standard cap.
        
        Returns target exposures to achieve compliance.
        """
        if self.state.gross_exposure <= self.state.standard_gross_cap:
            return {
                "BTC": self.state.btc_exposure,
                "ETH": self.state.eth_exposure,
                "SOL": self.state.sol_exposure,
            }
        
        # Scale down to fit within standard cap
        scale = self.state.standard_gross_cap / self.state.gross_exposure
        
        targets = {
            "BTC": self.state.btc_exposure * scale,
            "ETH": self.state.eth_exposure * scale,
            "SOL": self.state.sol_exposure * scale,
        }
        
        logger.warning(
            f"Forcing normalization: scale={scale:.2f}, "
            f"SOL target={targets['SOL']:.0%}"
        )
        
        # Reset violation state
        self.state.violation_status = ViolationStatus.NO_VIOLATION
        self.state.violation_started = None
        self.state.violation_bars_elapsed = 0
        self.state.override_active = False
        
        return targets
    
    def _standard_state(
        self,
        standard_cap: float,
        btc: float,
        eth: float,
        sol: float,
    ) -> SolDominanceState:
        """Return state with standard (non-override) values."""
        self.state.override_active = False
        self.state.effective_cap = standard_cap
        self.state.standard_gross_cap = standard_cap
        self.state.btc_exposure = btc
        self.state.eth_exposure = eth
        self.state.sol_exposure = sol
        self.state.gross_exposure = abs(btc) + abs(eth) + abs(sol)
        return self.state
    
    def get_sol_exposure_limit(self) -> float:
        """Get maximum allowed SOL exposure given current state."""
        if self.state.override_active:
            # In override, SOL can take up to full effective cap
            # minus whatever BTC/ETH hold
            btc_eth = abs(self.state.btc_exposure) + abs(self.state.eth_exposure)
            return self.state.effective_cap - btc_eth
        else:
            # Standard SOL cap from exposure manager
            return 1.00  # Default, will be capped by other systems


# Singleton
_sol_override: Optional[SolDominanceOverride] = None


def get_sol_dominance_override() -> SolDominanceOverride:
    """Get singleton instance."""
    global _sol_override
    if _sol_override is None:
        _sol_override = SolDominanceOverride()
    return _sol_override


def reset_sol_dominance_override():
    """Reset singleton."""
    global _sol_override
    _sol_override = None
