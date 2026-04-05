"""
================================================================================
HMATS v3.3-HR - HIGH-RISK OVERRIDE CONFIGURATION
================================================================================
Version: 3.3-HR (Personal, Aggressive Variant)
Purpose: Re-weight decision priorities for personal high-risk trading

GUIDING PRINCIPLE:
    Missing a high-conviction move is worse than taking a controlled loss.

This module contains ALL override configurations in one place.
Changes to risk philosophy should be made HERE, not scattered across modules.

OVERRIDES IMPLEMENTED:
1. SOL Dominance > Global Gross Cap (selective)
2. Fast Correlation Sentinel (exit-only)
3. CRACK Decay Softening (trend continuation)
4. Weekend Exposure Philosophy (OPPORTUNITY exemption)
5. Minimal Scale-Out (exit alpha)

CONSTRAINTS:
- No new strategies
- No new agents
- No architecture changes
- Only thresholds, precedence, and overrides
================================================================================
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
from enum import Enum


class OverrideMode(Enum):
    """Override activation modes."""
    DISABLED = "DISABLED"      # Use v3.3 base behavior
    ENABLED = "ENABLED"        # Apply high-risk override
    CONDITIONAL = "CONDITIONAL"  # Apply only when conditions met


@dataclass
class HighRiskConfig:
    """
    Master configuration for all high-risk overrides.
    
    PHILOSOPHY:
    - Aggressive alpha capture in OPPORTUNITY mode
    - Defensive only when signals decay or conflict
    - SOL is the primary profit vehicle
    """
    
    # =========================================================================
    # OVERRIDE 1: SOL Dominance > Global Gross Cap
    # =========================================================================
    # When OPPORTUNITY + SOL is primary signal carrier, allow temporary
    # violation of global gross cap.
    
    sol_dominance_override: OverrideMode = OverrideMode.ENABLED
    
    # Conditions for override activation
    sol_dominance_conditions: Dict = field(default_factory=lambda: {
        "require_opportunity_mode": True,
        "require_sol_primary_signal": True,  # SOL signal strength > BTC+ETH combined
        "require_btc_eth_reduced": True,     # BTC+ETH exposure must be < 10%
        "max_violation_pct": 0.30,           # Can exceed cap by up to 30%
        "violation_duration_bars": 2,        # Max 2 bars (8 hours) of violation
    })
    
    # What "SOL is primary signal carrier" means
    sol_primary_signal_thresholds: Dict = field(default_factory=lambda: {
        "sol_signal_min": 0.6,         # SOL signal strength minimum
        "sol_beta_min": 1.3,           # SOL/BTC beta minimum
        "sol_flow_min": 0.5,           # SOL flow alignment minimum
        "btc_eth_max_combined": 0.10,  # BTC+ETH combined exposure cap
    })
    
    # =========================================================================
    # OVERRIDE 2: Fast Correlation Sentinel (Exit-Only)
    # =========================================================================
    # Shorter correlation window active only during OPPORTUNITY for faster exits.
    # Does NOT veto entries - only accelerates exits.
    
    fast_correlation_override: OverrideMode = OverrideMode.ENABLED
    
    fast_correlation_config: Dict = field(default_factory=lambda: {
        "window_minutes": 30,          # 30-minute rolling window
        "only_in_opportunity": True,   # Only active in OPPORTUNITY mode
        "exit_only": True,             # Cannot veto entries
        "exit_threshold": 0.90,        # Fast exit when correlation > 0.90
        "standard_threshold": 0.92,    # Standard 4H threshold unchanged
    })
    
    # =========================================================================
    # OVERRIDE 3: CRACK Decay Softening
    # =========================================================================
    # During confirmed trend continuation, slow CRACK decay to stay in trade.
    
    crack_decay_override: OverrideMode = OverrideMode.ENABLED
    
    crack_decay_config: Dict = field(default_factory=lambda: {
        # Standard decay: 0.7^n per bar
        "standard_decay_rate": 0.70,
        
        # Softened decay during trend
        "trend_decay_rate": 0.85,      # Slower decay (0.85^n)
        "minimum_crack_weight": 0.40,  # OpportunityThresholds.CRACK_DECAY_FLOOR  # Floor - never decay below 40%
        
        # Trend continuation conditions
        "trend_conditions": {
            "momentum_aligned": True,   # Price momentum matches CRACK direction
            "volume_sustained": True,   # Volume > 0.8× average
            "no_reversal_signal": True, # No opposing signal from any agent
            "rsi_not_extreme": True,    # RSI between 25-75
        },
        
        # RSI bounds for "not extreme"
        "rsi_lower_bound": 25,
        "rsi_upper_bound": 75,
        "volume_floor_multiplier": 0.80,
    })
    
    # =========================================================================
    # OVERRIDE 4: Weekend/Holiday Exposure Philosophy
    # =========================================================================
    # Crypto markets don't close. Weekend reduction only applies to NORMAL mode.
    
    weekend_override: OverrideMode = OverrideMode.ENABLED
    
    weekend_config: Dict = field(default_factory=lambda: {
        # NORMAL mode: Apply standard weekend caps (50%)
        "normal_mode_cap": 0.50,
        
        # OPPORTUNITY mode: No weekend veto
        "opportunity_mode_cap": 1.50,  # Full capacity in OPPORTUNITY
        
        # Holiday behavior (same logic)
        "holiday_normal_cap": 0.30,
        "holiday_opportunity_cap": 1.20,  # Slightly reduced but not blocked
        
        # Low liquidity hours (22:00-02:00 UTC)
        "low_liq_normal_cap": 0.70,
        "low_liq_opportunity_cap": 1.20,
    })
    
    # =========================================================================
    # OVERRIDE 5: Minimal Scale-Out (Exit Alpha)
    # =========================================================================
    # One simple partial profit lock, then let the rest run.
    
    scale_out_override: OverrideMode = OverrideMode.ENABLED
    
    scale_out_config: Dict = field(default_factory=lambda: {
        # Partial exit percentage
        "partial_exit_pct": 0.25,      # Exit 25% of position
        
        # Trigger conditions (ANY of these)
        "triggers": {
            "momentum_stall": True,     # Momentum indicator flattens
            "crack_decay_below": 0.50,  # OpportunityThresholds.CRACK_ACTIVATION_WEIGHT  # CRACK weight drops below 50%
            "profit_threshold_bps": 150, # At least 150bps profit before partial
        },
        
        # Momentum stall definition
        "momentum_stall_bars": 2,       # 2 consecutive bars of flat momentum
        "momentum_stall_threshold": 0.05,  # Momentum change < 5%
        
        # Runner behavior
        "runner_stop_type": "trailing",  # Trailing stop for remainder
        "runner_trail_pct": 0.02,        # 2% trailing stop
    })
    
    # =========================================================================
    # GLOBAL HIGH-RISK SETTINGS
    # =========================================================================
    
    # Alpha thresholds (more aggressive)
    alpha_thresholds: Dict = field(default_factory=lambda: {
        "normal_mode_bps": 15,         # 15bps: Kraken+ free tier, 3:1 alpha:cost ratio
        "opportunity_mode_bps": 8,     # 8bps: aggressive in OPPORTUNITY mode
        "sol_dominant_bps": 5,         # 5bps: minimal friction for SOL dominant
    })
    
    # Tranche timing (faster escalation)
    tranche_config: Dict = field(default_factory=lambda: {
        "min_hold_bars": 1,            # Keep 1 bar minimum
        "opportunity_t2_early": True,  # Allow T2 before 4H close in OPPORTUNITY
        "t2_early_confirmation": 0.7,  # 70% structure confirmation threshold
    })
    
    def to_dict(self) -> Dict:
        return {
            "sol_dominance_override": self.sol_dominance_override.value,
            "fast_correlation_override": self.fast_correlation_override.value,
            "crack_decay_override": self.crack_decay_override.value,
            "weekend_override": self.weekend_override.value,
            "scale_out_override": self.scale_out_override.value,
            "alpha_thresholds": self.alpha_thresholds,
            "tranche_config": self.tranche_config,
        }


# Singleton instance
_hr_config: Optional[HighRiskConfig] = None


def get_hr_config() -> HighRiskConfig:
    """Get high-risk configuration singleton."""
    global _hr_config
    if _hr_config is None:
        _hr_config = HighRiskConfig()
    return _hr_config


def reset_hr_config():
    """Reset to defaults (for testing)."""
    global _hr_config
    _hr_config = None
