"""
================================================================================
HMATS v3.3-HR - HIGH-RISK OVERRIDE INTEGRATION
================================================================================
Version: 3.3-HR
Purpose: Unified integration of all high-risk overrides

This module provides a single interface to all 5 high-risk overrides:
1. SOL Dominance > Global Gross Cap
2. Fast Correlation Sentinel (Exit-Only)
3. CRACK Decay Softening
4. Weekend Override (OPPORTUNITY exemption)
5. Minimal Scale-Out

GUIDING PRINCIPLE:
    Missing a high-conviction move is worse than taking a controlled loss.

USAGE:
    from core.high_risk_integration import HighRiskOverrideLayer
    
    hr_layer = HighRiskOverrideLayer()
    
    # On each decision point
    result = hr_layer.apply_overrides(
        system_mode="OPPORTUNITY",
        sol_signal=0.8,
        sol_beta=1.5,
        ...
    )
    
    # Use result.effective_caps for position sizing
    # Check result.scale_out_decision for exit signals
================================================================================
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Any
from datetime import datetime

from .high_risk_config import get_hr_config, HighRiskConfig, OverrideMode
from overrides.override_sol_dominance import (
    SolDominanceOverride,
    SolDominanceState,
    get_sol_dominance_override,
)
from overrides.override_fast_correlation import (
    FastCorrelationSentinel,
    FastCorrelationSnapshot,
    get_fast_correlation_sentinel,
)
from overrides.override_crack_decay import (
    CrackDecaySoftener,
    CrackDecayState,
    get_crack_decay_softener,
)
from overrides.override_weekend import (
    WeekendOverrideManager,
    WeekendOverrideState,
    get_weekend_override_manager,
)
from overrides.override_scale_out import (
    MinimalScaleOut,
    ScaleOutDecision,
    PositionMetrics,
    get_minimal_scale_out,
)

logger = logging.getLogger(__name__)


@dataclass
class HighRiskOverrideResult:
    """Complete result of applying all high-risk overrides."""
    
    # Effective caps (final values to use)
    effective_gross_cap: float = 1.50
    effective_sol_cap: float = 1.00
    effective_btc_cap: float = 0.80
    effective_eth_cap: float = 0.80
    
    # Override statuses
    sol_dominance: Optional[SolDominanceState] = None
    fast_correlation: Optional[FastCorrelationSnapshot] = None
    crack_decay: Optional[CrackDecayState] = None
    weekend: Optional[WeekendOverrideState] = None
    scale_out: Optional[ScaleOutDecision] = None
    
    # Aggregated signals
    should_exit_fast: bool = False
    exit_reason: str = ""
    effective_crack_weight: float = 1.0
    
    # Active overrides count
    overrides_active: int = 0
    override_summary: str = ""
    
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict:
        return {
            "effective_caps": {
                "gross": self.effective_gross_cap,
                "SOL": self.effective_sol_cap,
                "BTC": self.effective_btc_cap,
                "ETH": self.effective_eth_cap,
            },
            "should_exit_fast": self.should_exit_fast,
            "exit_reason": self.exit_reason,
            "effective_crack_weight": self.effective_crack_weight,
            "overrides_active": self.overrides_active,
            "override_summary": self.override_summary,
            "sol_dominance": self.sol_dominance.to_dict() if self.sol_dominance else None,
            "fast_correlation": self.fast_correlation.to_dict() if self.fast_correlation else None,
            "crack_decay": self.crack_decay.to_dict() if self.crack_decay else None,
            "weekend": self.weekend.to_dict() if self.weekend else None,
            "scale_out": self.scale_out.to_dict() if self.scale_out else None,
        }


class HighRiskOverrideLayer:
    """
    Unified interface for all high-risk overrides.
    
    This layer sits between the base v3.3 safety layer and the trading logic.
    It MODIFIES caps and signals based on high-risk philosophy.
    
    CRITICAL: This does NOT disable safety features.
    - Hard stops still apply
    - NO_TRADE still respected
    - Data integrity checks still enforced
    
    It only ADJUSTS:
    - Exposure caps (more aggressive in OPPORTUNITY)
    - Exit timing (faster via correlation sentinel)
    - CRACK persistence (slower decay in trends)
    - Weekend behavior (no veto in OPPORTUNITY)
    - Scale-out logic (partial profit locks)
    """
    
    def __init__(self, config: Optional[HighRiskConfig] = None):
        self.config = config or get_hr_config()
        
        # Initialize override modules
        self.sol_dominance = get_sol_dominance_override()
        self.fast_correlation = get_fast_correlation_sentinel()
        self.crack_decay = get_crack_decay_softener()
        self.weekend = get_weekend_override_manager()
        self.scale_out = get_minimal_scale_out()
        
        logger.info("HighRiskOverrideLayer initialized")
    
    def apply_overrides(
        self,
        # System state
        system_mode: str,
        
        # SOL dominance inputs
        sol_signal: float = 0.0,
        sol_beta: float = 1.0,
        sol_flow: float = 0.0,
        btc_exposure: float = 0.0,
        eth_exposure: float = 0.0,
        sol_exposure: float = 0.0,
        
        # Correlation inputs
        standard_correlation: float = 0.0,
        
        # CRACK inputs
        crack_direction: str = "none",
        crack_bars_elapsed: int = 0,
        momentum: float = 0.0,
        volume_ratio: float = 1.0,
        rsi: float = 50.0,
        has_reversal_signal: bool = False,
        
        # Position metrics (for scale-out)
        position_metrics: Optional[PositionMetrics] = None,
        
        # Base caps (from v3.3 safety layer)
        base_gross_cap: float = 1.50,
        base_sol_cap: float = 1.00,
        
        # Is SOL dominant mode active?
        is_sol_dominant: bool = False,
    ) -> HighRiskOverrideResult:
        """
        Apply all high-risk overrides and return aggregated result.
        """
        
        result = HighRiskOverrideResult()
        overrides_active = []
        
        # =====================================================================
        # OVERRIDE 1: SOL Dominance
        # =====================================================================
        sol_state = self.sol_dominance.evaluate(
            system_mode=system_mode,
            sol_signal=sol_signal,
            sol_beta=sol_beta,
            sol_flow=sol_flow,
            btc_exposure=btc_exposure,
            eth_exposure=eth_exposure,
            sol_exposure=sol_exposure,
            standard_gross_cap=base_gross_cap,
        )
        result.sol_dominance = sol_state
        
        if sol_state.override_active:
            result.effective_gross_cap = sol_state.effective_cap
            result.effective_sol_cap = sol_state.get_sol_exposure_limit() if hasattr(sol_state, 'get_sol_exposure_limit') else base_sol_cap
            overrides_active.append("SOL_DOMINANCE")
        else:
            result.effective_gross_cap = base_gross_cap
            result.effective_sol_cap = base_sol_cap
        
        # =====================================================================
        # OVERRIDE 2: Fast Correlation (Exit-Only)
        # =====================================================================
        corr_snapshot = self.fast_correlation.get_snapshot(
            system_mode=system_mode,
            standard_correlation=standard_correlation,
        )
        result.fast_correlation = corr_snapshot
        
        if corr_snapshot.exit_triggered:
            result.should_exit_fast = True
            result.exit_reason = corr_snapshot.exit_reason
            overrides_active.append("FAST_CORR_EXIT")
        
        # =====================================================================
        # OVERRIDE 3: CRACK Decay Softening
        # =====================================================================
        if crack_direction != "none":
            crack_state = self.crack_decay.calculate_crack_weight(
                crack_direction=crack_direction,
                bars_elapsed=crack_bars_elapsed,
                momentum=momentum,
                volume_ratio=volume_ratio,
                rsi=rsi,
                has_reversal_signal=has_reversal_signal,
            )
            result.crack_decay = crack_state
            result.effective_crack_weight = crack_state.effective_weight
            
            if crack_state.softening_active:
                overrides_active.append("CRACK_SOFTENING")
        
        # =====================================================================
        # OVERRIDE 4: Weekend Override
        # =====================================================================
        weekend_state = self.weekend.get_effective_caps(
            system_mode=system_mode,
            is_sol_dominant=is_sol_dominant,
        )
        result.weekend = weekend_state
        
        if weekend_state.override_active:
            # Weekend override can raise caps above what SOL dominance set
            result.effective_gross_cap = max(
                result.effective_gross_cap,
                weekend_state.effective_cap,
            )
            result.effective_sol_cap = max(
                result.effective_sol_cap,
                weekend_state.sol_cap,
            )
            result.effective_btc_cap = weekend_state.btc_cap
            result.effective_eth_cap = weekend_state.eth_cap
            overrides_active.append("WEEKEND_OVERRIDE")
        else:
            # Use weekend caps as minimum (don't make things MORE restrictive)
            result.effective_btc_cap = min(0.80, weekend_state.btc_cap)
            result.effective_eth_cap = min(0.80, weekend_state.eth_cap)
        
        # =====================================================================
        # OVERRIDE 5: Scale-Out
        # =====================================================================
        if position_metrics:
            scale_out_decision = self.scale_out.evaluate(position_metrics)
            result.scale_out = scale_out_decision
            
            if scale_out_decision.should_scale_out:
                overrides_active.append("SCALE_OUT")
        
        # =====================================================================
        # Aggregate results
        # =====================================================================
        result.overrides_active = len(overrides_active)
        result.override_summary = ", ".join(overrides_active) if overrides_active else "None"
        
        if overrides_active:
            logger.debug(
                f"High-risk overrides: {result.override_summary}, "
                f"gross_cap={result.effective_gross_cap:.0%}, "
                f"sol_cap={result.effective_sol_cap:.0%}"
            )
        
        return result
    
    def update_price_data(
        self,
        btc_price: float,
        eth_price: float,
        sol_price: float,
    ):
        """Feed price data to fast correlation sentinel."""
        self.fast_correlation.add_prices(btc_price, eth_price, sol_price)
    
    def on_bar_close(self):
        """Called at each 4H bar close for time-based updates."""
        self.sol_dominance.on_bar_close()
    
    def mark_scale_out(self, asset: str):
        """Mark position as having scaled out."""
        self.scale_out.mark_scaled_out(asset)
    
    def reset_position(self, asset: str):
        """Reset tracking for position."""
        self.scale_out.reset_position(asset)
    
    def get_status(self) -> Dict[str, Any]:
        """Get complete status of all overrides."""
        return {
            "config": self.config.to_dict(),
            "sol_dominance_active": self.sol_dominance.state.override_active,
            "fast_correlation_active": self.fast_correlation.snapshot.is_active,
            "crack_softening_active": self.crack_decay.state.softening_active,
            "weekend_override_active": self.weekend.state.override_active,
        }


# Singleton
_hr_layer: Optional[HighRiskOverrideLayer] = None


def get_hr_override_layer() -> HighRiskOverrideLayer:
    """Get singleton high-risk override layer."""
    global _hr_layer
    if _hr_layer is None:
        _hr_layer = HighRiskOverrideLayer()
    return _hr_layer


def reset_hr_override_layer():
    """Reset singleton."""
    global _hr_layer
    _hr_layer = None
