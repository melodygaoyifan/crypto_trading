"""
HMATS v3.2 - Alpha Threshold Calculator
========================================
Implements Strategy Constitution Section 4.

Alpha thresholds by mode:
- NORMAL: alpha >= 3× friction (99 bps taker)
- OPPORTUNITY: alpha >= 1.5-2.5× friction (50-83 bps)
- NO_TRADE: no execution allowed

Includes alpha estimation heuristic and friction calculation.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from enum import Enum
import logging

from orchestration.system_mode import SystemMode

logger = logging.getLogger(__name__)


@dataclass
class FrictionComponents:
    """Friction components for alpha calculation."""
    taker_fee_bps: float = 0.0       # Kraken+ $10K/mo free tier
    maker_fee_bps: float = 0.0       # Kraken+ free tier
    slippage_bps: float = 5.0        # Expected slippage
    latency_cost_bps: float = 2.0    # Network/processing delay
    
    @property
    def total_taker(self) -> float:
        """Total friction for taker orders."""
        return self.taker_fee_bps + self.slippage_bps + self.latency_cost_bps
    
    @property
    def total_maker(self) -> float:
        """Total friction for maker orders."""
        return self.maker_fee_bps + self.slippage_bps + self.latency_cost_bps


@dataclass
class AlphaThresholds:
    """Alpha thresholds for a specific mode."""
    mode: SystemMode
    taker_threshold_bps: float
    maker_threshold_bps: float
    multiplier: float
    
    def to_dict(self) -> Dict:
        return {
            "mode": self.mode.name,
            "taker_threshold_bps": self.taker_threshold_bps,
            "maker_threshold_bps": self.maker_threshold_bps,
            "multiplier": self.multiplier
        }


@dataclass
class AlphaEstimate:
    """Estimated alpha for a signal."""
    estimated_alpha_bps: float
    signal_strength: float
    regime_confidence: float
    historical_hit_rate: float
    meets_threshold: bool
    threshold_bps: float
    margin_bps: float  # How much above/below threshold
    
    def to_dict(self) -> Dict:
        return {
            "estimated_alpha_bps": self.estimated_alpha_bps,
            "signal_strength": self.signal_strength,
            "regime_confidence": self.regime_confidence,
            "historical_hit_rate": self.historical_hit_rate,
            "meets_threshold": self.meets_threshold,
            "threshold_bps": self.threshold_bps,
            "margin_bps": self.margin_bps
        }


class AlphaThresholdCalculator:
    """
    Calculates alpha thresholds and estimates.
    
    From Strategy Constitution Section 4.
    """
    
    # Kraken friction components
    FRICTION = FrictionComponents()
    
    # Mode multipliers
    NORMAL_MULTIPLIER = 3.0
    OPPORTUNITY_LEAD_LAG_MULTIPLIER = 1.5
    OPPORTUNITY_CRACK_MULTIPLIER = 2.0
    OPPORTUNITY_OTHER_MULTIPLIER = 2.5
    
    # Alpha estimation parameters
    MAX_ALPHA_BPS = 200  # Maximum assumed alpha (200 bps)
    
    def __init__(self):
        self._rolling_hit_rate = 0.5  # Default 50%
        self._recent_trades: list = []
    
    def get_threshold(
        self,
        mode: SystemMode,
        is_maker: bool = False,
        lead_lag_edge_confirmed: bool = False,
        crack_window_active: bool = False
    ) -> AlphaThresholds:
        """
        Get alpha threshold for current mode and conditions.
        
        Args:
            mode: Current system mode
            is_maker: True if using maker orders
            lead_lag_edge_confirmed: True if lead-lag edge is confirmed
            crack_window_active: True if CRACK window is active
        
        Returns:
            AlphaThresholds with calculated values
        """
        
        friction = self.FRICTION.total_maker if is_maker else self.FRICTION.total_taker
        
        if mode == SystemMode.NO_TRADE:
            return AlphaThresholds(
                mode=mode,
                taker_threshold_bps=float('inf'),
                maker_threshold_bps=float('inf'),
                multiplier=float('inf')
            )
        
        elif mode == SystemMode.OPPORTUNITY:
            # Dynamic threshold based on opportunity type
            if lead_lag_edge_confirmed:
                multiplier = self.OPPORTUNITY_LEAD_LAG_MULTIPLIER  # 1.5×
            elif crack_window_active:
                multiplier = self.OPPORTUNITY_CRACK_MULTIPLIER  # 2.0×
            else:
                multiplier = self.OPPORTUNITY_OTHER_MULTIPLIER  # 2.5×
            
            return AlphaThresholds(
                mode=mode,
                taker_threshold_bps=self.FRICTION.total_taker * multiplier,
                maker_threshold_bps=self.FRICTION.total_maker * multiplier,
                multiplier=multiplier
            )
        
        else:  # NORMAL
            multiplier = self.NORMAL_MULTIPLIER  # 3×
            return AlphaThresholds(
                mode=mode,
                taker_threshold_bps=self.FRICTION.total_taker * multiplier,
                maker_threshold_bps=self.FRICTION.total_maker * multiplier,
                multiplier=multiplier
            )
    
    def estimate_alpha(
        self,
        signal_strength: float,
        regime_confidence: float,
        historical_hit_rate: Optional[float] = None
    ) -> float:
        """
        Estimate alpha from signal characteristics.
        
        From Strategy Constitution Section 4.3:
        - Base alpha from signal strength (max 200bps)
        - Adjusted by regime confidence
        - Adjusted by historical performance
        
        Args:
            signal_strength: -1 to 1 (from Quant/DRL fusion)
            regime_confidence: 0 to 1 (from regime navigator)
            historical_hit_rate: 0 to 1 (rolling 30-day), or None to use stored
        
        Returns:
            Estimated alpha in bps
        """
        
        if historical_hit_rate is None:
            historical_hit_rate = self._rolling_hit_rate
        
        # Base alpha from signal strength (assume 200bps max alpha)
        base_alpha_bps = abs(signal_strength) * self.MAX_ALPHA_BPS
        
        # Confidence adjustment
        adjusted_alpha = base_alpha_bps * regime_confidence
        
        # Historical performance adjustment (0.5 to 1.0 factor)
        performance_factor = 0.5 + (historical_hit_rate * 0.5)
        
        estimated_alpha = adjusted_alpha * performance_factor
        
        return estimated_alpha
    
    def should_execute(
        self,
        signal_strength: float,
        regime_confidence: float,
        mode: SystemMode,
        is_maker: bool = False,
        lead_lag_edge_confirmed: bool = False,
        crack_window_active: bool = False,
        historical_hit_rate: Optional[float] = None
    ) -> AlphaEstimate:
        """
        Check if a signal should be executed based on alpha threshold.
        
        From Strategy Constitution Section 4.4.
        
        Returns:
            AlphaEstimate with decision and details
        """
        
        # Get threshold
        thresholds = self.get_threshold(
            mode=mode,
            is_maker=is_maker,
            lead_lag_edge_confirmed=lead_lag_edge_confirmed,
            crack_window_active=crack_window_active
        )
        
        threshold = thresholds.maker_threshold_bps if is_maker else thresholds.taker_threshold_bps
        
        # Estimate alpha
        estimated_alpha = self.estimate_alpha(
            signal_strength=signal_strength,
            regime_confidence=regime_confidence,
            historical_hit_rate=historical_hit_rate
        )
        
        # Compare
        meets_threshold = estimated_alpha >= threshold
        margin = estimated_alpha - threshold
        
        return AlphaEstimate(
            estimated_alpha_bps=estimated_alpha,
            signal_strength=signal_strength,
            regime_confidence=regime_confidence,
            historical_hit_rate=historical_hit_rate or self._rolling_hit_rate,
            meets_threshold=meets_threshold,
            threshold_bps=threshold,
            margin_bps=margin
        )
    
    def update_hit_rate(self, won: bool):
        """
        Update rolling hit rate with trade outcome.
        
        Uses exponential moving average with 30-trade lookback.
        """
        
        self._recent_trades.append(1 if won else 0)
        
        # Keep last 100 trades
        if len(self._recent_trades) > 100:
            self._recent_trades = self._recent_trades[-100:]
        
        # Calculate hit rate (use last 30 for recent performance)
        recent = self._recent_trades[-30:] if len(self._recent_trades) >= 30 else self._recent_trades
        self._rolling_hit_rate = sum(recent) / len(recent) if recent else 0.5
    
    def get_friction_summary(self) -> Dict:
        """Get summary of friction components."""
        return {
            "taker_fee_bps": self.FRICTION.taker_fee_bps,
            "maker_fee_bps": self.FRICTION.maker_fee_bps,
            "slippage_bps": self.FRICTION.slippage_bps,
            "latency_cost_bps": self.FRICTION.latency_cost_bps,
            "total_taker_bps": self.FRICTION.total_taker,
            "total_maker_bps": self.FRICTION.total_maker
        }
    
    def get_thresholds_summary(self) -> Dict:
        """Get summary of thresholds by mode."""
        return {
            "NORMAL": {
                "multiplier": self.NORMAL_MULTIPLIER,
                "taker_bps": self.FRICTION.total_taker * self.NORMAL_MULTIPLIER,
                "maker_bps": self.FRICTION.total_maker * self.NORMAL_MULTIPLIER
            },
            "OPPORTUNITY_LEAD_LAG": {
                "multiplier": self.OPPORTUNITY_LEAD_LAG_MULTIPLIER,
                "taker_bps": self.FRICTION.total_taker * self.OPPORTUNITY_LEAD_LAG_MULTIPLIER,
                "maker_bps": self.FRICTION.total_maker * self.OPPORTUNITY_LEAD_LAG_MULTIPLIER
            },
            "OPPORTUNITY_CRACK": {
                "multiplier": self.OPPORTUNITY_CRACK_MULTIPLIER,
                "taker_bps": self.FRICTION.total_taker * self.OPPORTUNITY_CRACK_MULTIPLIER,
                "maker_bps": self.FRICTION.total_maker * self.OPPORTUNITY_CRACK_MULTIPLIER
            },
            "OPPORTUNITY_OTHER": {
                "multiplier": self.OPPORTUNITY_OTHER_MULTIPLIER,
                "taker_bps": self.FRICTION.total_taker * self.OPPORTUNITY_OTHER_MULTIPLIER,
                "maker_bps": self.FRICTION.total_maker * self.OPPORTUNITY_OTHER_MULTIPLIER
            },
            "NO_TRADE": {
                "multiplier": "inf",
                "taker_bps": "inf",
                "maker_bps": "inf"
            }
        }


# Singleton instance
_alpha_calculator = AlphaThresholdCalculator()

def get_alpha_calculator() -> AlphaThresholdCalculator:
    return _alpha_calculator
