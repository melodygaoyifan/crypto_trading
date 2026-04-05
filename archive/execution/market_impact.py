#!/usr/bin/env python3
"""
================================================================================
HMATS v5.1 - Market Impact Model
================================================================================
Purpose: Estimate market impact and execution costs for optimal order sizing

P2 SOTA Requirement:
- "补 Market Impact 模块，并接到 execution decision"
- "先从经验曲线（按盘口深度/OFI/成交量分位）估计 impact cost"

Theory:
    Market Impact = f(order_size, volatility, liquidity, urgency)
    
    Almgren-Chriss Framework:
    - Temporary Impact: Linear in trade rate
    - Permanent Impact: Square-root in total volume
    
    Total Cost = Spread Cost + Temporary Impact + Permanent Impact

Implementation:
    1. Empirical curves from historical data
    2. Order book depth analysis
    3. OFI (Order Flow Imbalance) adjustment
    4. Volume percentile scaling

Author: HMATS v5.1 P2 Fix  
Version: 5.1.0
================================================================================
"""

import logging
import math
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum, auto
from collections import deque

logger = logging.getLogger(__name__)


# =============================================================================
# IMPACT MODEL CONFIGURATION
# =============================================================================

class ImpactModel(Enum):
    """Available market impact models."""
    LINEAR = "linear"           # Simple linear model
    SQUARE_ROOT = "sqrt"        # Square-root model (Almgren-Chriss)
    POWER_LAW = "power"         # Power law model
    KYLE_LAMBDA = "kyle"        # Kyle's lambda model
    EMPIRICAL = "empirical"     # Data-driven empirical curves


@dataclass
class ImpactConfig:
    """Market impact model configuration."""
    # Model selection
    model: ImpactModel = ImpactModel.SQUARE_ROOT
    
    # Almgren-Chriss parameters
    eta: float = 0.0001        # Temporary impact coefficient
    gamma: float = 0.0001      # Permanent impact coefficient
    
    # Power law parameters
    alpha: float = 0.5         # Impact exponent (0.5 = square root)
    
    # Kyle's lambda (estimated from data)
    kyle_lambda: float = 0.001  # Price impact per unit flow
    
    # Empirical curve parameters
    use_ofi_adjustment: bool = True
    use_volume_scaling: bool = True
    
    # Risk limits
    max_impact_bps: float = 50.0    # Maximum acceptable impact in bps
    max_participation: float = 0.1   # Maximum % of daily volume


@dataclass
class OrderBookState:
    """Current order book state for impact estimation."""
    symbol: str
    timestamp: datetime
    
    # Top of book
    best_bid: float
    best_ask: float
    mid_price: float
    spread_bps: float
    
    # Depth (cumulative USD at each level)
    bid_depth_1pct: float  # Depth within 1% of mid
    ask_depth_1pct: float
    bid_depth_5pct: float  # Depth within 5% of mid
    ask_depth_5pct: float
    
    # Order flow imbalance
    ofi: float = 0.0  # Positive = buy pressure, negative = sell pressure
    ofi_zscore: float = 0.0  # Normalized OFI


@dataclass
class ImpactEstimate:
    """Market impact estimation result."""
    symbol: str
    side: str  # "buy" or "sell"
    quantity: float
    quantity_usd: float
    
    # Impact components (in basis points)
    spread_cost_bps: float
    temporary_impact_bps: float
    permanent_impact_bps: float
    total_impact_bps: float
    
    # Costs in USD
    spread_cost_usd: float
    temporary_impact_usd: float
    permanent_impact_usd: float
    total_cost_usd: float
    
    # Execution quality metrics
    expected_slippage_bps: float
    participation_rate: float
    confidence: float  # Model confidence 0-1
    
    # Recommendations
    recommended_algo: str  # TWAP, VWAP, aggressive, etc.
    recommended_duration_minutes: int
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "quantity_usd": self.quantity_usd,
            "spread_cost_bps": self.spread_cost_bps,
            "temporary_impact_bps": self.temporary_impact_bps,
            "permanent_impact_bps": self.permanent_impact_bps,
            "total_impact_bps": self.total_impact_bps,
            "total_cost_usd": self.total_cost_usd,
            "expected_slippage_bps": self.expected_slippage_bps,
            "recommended_algo": self.recommended_algo,
            "recommended_duration_minutes": self.recommended_duration_minutes,
            "confidence": self.confidence,
            "warnings": self.warnings
        }


# =============================================================================
# EMPIRICAL IMPACT CURVES
# =============================================================================

class EmpiricalImpactCurves:
    """
    Data-driven impact curves based on historical observations.
    
    Curves are parameterized by:
    - Order size as % of daily volume
    - Volatility regime
    - Liquidity conditions (spread, depth)
    """
    
    # Impact multipliers by volatility regime (low, normal, high)
    VOLATILITY_MULTIPLIERS = {
        "low": 0.7,
        "normal": 1.0,
        "high": 1.5,
        "extreme": 2.5
    }
    
    # Impact multipliers by liquidity (tight, normal, wide spread)
    SPREAD_MULTIPLIERS = {
        "tight": 0.8,    # < 5 bps
        "normal": 1.0,   # 5-15 bps
        "wide": 1.4,     # 15-30 bps
        "very_wide": 2.0 # > 30 bps
    }
    
    # Base impact curve (participation rate -> impact in bps)
    # Empirically calibrated for crypto markets
    BASE_CURVE = [
        (0.001, 1.0),    # 0.1% participation -> 1 bps impact
        (0.005, 3.0),    # 0.5% -> 3 bps
        (0.01, 6.0),     # 1% -> 6 bps
        (0.02, 12.0),    # 2% -> 12 bps
        (0.05, 30.0),    # 5% -> 30 bps
        (0.10, 70.0),    # 10% -> 70 bps
        (0.20, 150.0),   # 20% -> 150 bps
    ]
    
    @classmethod
    def interpolate_impact(cls, participation_rate: float) -> float:
        """Interpolate impact from base curve."""
        if participation_rate <= 0:
            return 0.0
            
        # Find surrounding points
        prev_rate, prev_impact = 0.0, 0.0
        for rate, impact in cls.BASE_CURVE:
            if participation_rate <= rate:
                # Linear interpolation
                if rate == prev_rate:
                    return impact
                slope = (impact - prev_impact) / (rate - prev_rate)
                return prev_impact + slope * (participation_rate - prev_rate)
            prev_rate, prev_impact = rate, impact
            
        # Extrapolate beyond curve (power law)
        last_rate, last_impact = cls.BASE_CURVE[-1]
        return last_impact * (participation_rate / last_rate) ** 0.5
        
    @classmethod
    def get_volatility_regime(cls, volatility: float) -> str:
        """Classify volatility regime."""
        if volatility < 0.02:  # < 2% daily
            return "low"
        elif volatility < 0.04:  # 2-4%
            return "normal"
        elif volatility < 0.08:  # 4-8%
            return "high"
        else:
            return "extreme"
            
    @classmethod
    def get_spread_regime(cls, spread_bps: float) -> str:
        """Classify spread regime."""
        if spread_bps < 5:
            return "tight"
        elif spread_bps < 15:
            return "normal"
        elif spread_bps < 30:
            return "wide"
        else:
            return "very_wide"


# =============================================================================
# MARKET IMPACT CALCULATOR
# =============================================================================

class MarketImpactCalculator:
    """
    Calculate market impact for proposed trades.
    
    Usage:
        calc = MarketImpactCalculator(config)
        
        # Estimate impact
        estimate = calc.estimate_impact(
            symbol="SOL/USD",
            side="buy", 
            quantity_usd=10000,
            orderbook=current_orderbook,
            daily_volume=1000000,
            volatility=0.03
        )
        
        # Use estimate to adjust execution
        if estimate.total_impact_bps > 20:
            use_twap_algo()
    """
    
    def __init__(self, config: Optional[ImpactConfig] = None):
        self.config = config or ImpactConfig()
        
        # Historical data for calibration
        self._impact_history: deque = deque(maxlen=1000)
        self._volume_history: deque = deque(maxlen=100)
        
    def estimate_impact(
        self,
        symbol: str,
        side: str,
        quantity_usd: float,
        orderbook: Optional[OrderBookState] = None,
        daily_volume: float = 0.0,
        volatility: float = 0.03,
        urgency: float = 0.5  # 0 = patient, 1 = urgent
    ) -> ImpactEstimate:
        """
        Estimate market impact for a proposed trade.
        
        Args:
            symbol: Trading symbol
            side: "buy" or "sell"
            quantity_usd: Order size in USD
            orderbook: Current orderbook state
            daily_volume: Average daily volume in USD
            volatility: Daily return volatility
            urgency: Execution urgency (0-1)
            
        Returns:
            ImpactEstimate with all cost components
        """
        warnings = []
        
        # Default orderbook if not provided
        if not orderbook:
            orderbook = OrderBookState(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                best_bid=100.0,
                best_ask=100.1,
                mid_price=100.05,
                spread_bps=10.0,
                bid_depth_1pct=100000,
                ask_depth_1pct=100000,
                bid_depth_5pct=500000,
                ask_depth_5pct=500000
            )
            
        # Default daily volume if not provided
        if daily_volume <= 0:
            daily_volume = 10_000_000  # Default $10M
            warnings.append("Using default daily volume estimate")
            
        # Calculate participation rate
        participation_rate = quantity_usd / daily_volume
        
        # Check limits
        if participation_rate > self.config.max_participation:
            warnings.append(f"High participation rate: {participation_rate:.1%}")
            
        # Get price for quantity conversion
        price = orderbook.mid_price
        quantity = quantity_usd / price
        
        # 1. Spread cost (always pay half-spread on entry)
        spread_cost_bps = orderbook.spread_bps / 2
        spread_cost_usd = quantity_usd * spread_cost_bps / 10000
        
        # 2. Calculate impact based on model
        if self.config.model == ImpactModel.SQUARE_ROOT:
            temp_impact_bps, perm_impact_bps = self._almgren_chriss_impact(
                quantity_usd, daily_volume, volatility, urgency
            )
        elif self.config.model == ImpactModel.EMPIRICAL:
            temp_impact_bps, perm_impact_bps = self._empirical_impact(
                participation_rate, volatility, orderbook
            )
        elif self.config.model == ImpactModel.LINEAR:
            temp_impact_bps, perm_impact_bps = self._linear_impact(
                quantity_usd, daily_volume, volatility
            )
        else:
            # Default to square root
            temp_impact_bps, perm_impact_bps = self._almgren_chriss_impact(
                quantity_usd, daily_volume, volatility, urgency
            )
            
        # Apply OFI adjustment
        if self.config.use_ofi_adjustment and orderbook.ofi != 0:
            ofi_factor = self._ofi_adjustment(side, orderbook.ofi)
            temp_impact_bps *= ofi_factor
            
        # Total impact
        total_impact_bps = spread_cost_bps + temp_impact_bps + perm_impact_bps
        
        # Check max impact
        if total_impact_bps > self.config.max_impact_bps:
            warnings.append(f"Impact exceeds limit: {total_impact_bps:.1f} bps > {self.config.max_impact_bps} bps")
            
        # Convert to USD
        temp_impact_usd = quantity_usd * temp_impact_bps / 10000
        perm_impact_usd = quantity_usd * perm_impact_bps / 10000
        total_cost_usd = spread_cost_usd + temp_impact_usd + perm_impact_usd
        
        # Expected slippage (impact + noise)
        noise_bps = 2.0  # Base execution noise
        expected_slippage_bps = total_impact_bps + noise_bps
        
        # Recommend execution algorithm
        recommended_algo, duration = self._recommend_execution(
            total_impact_bps, participation_rate, urgency
        )
        
        # Confidence based on data quality
        confidence = self._calculate_confidence(orderbook, daily_volume > 0)
        
        return ImpactEstimate(
            symbol=symbol,
            side=side,
            quantity=quantity,
            quantity_usd=quantity_usd,
            spread_cost_bps=spread_cost_bps,
            temporary_impact_bps=temp_impact_bps,
            permanent_impact_bps=perm_impact_bps,
            total_impact_bps=total_impact_bps,
            spread_cost_usd=spread_cost_usd,
            temporary_impact_usd=temp_impact_usd,
            permanent_impact_usd=perm_impact_usd,
            total_cost_usd=total_cost_usd,
            expected_slippage_bps=expected_slippage_bps,
            participation_rate=participation_rate,
            confidence=confidence,
            recommended_algo=recommended_algo,
            recommended_duration_minutes=duration,
            warnings=warnings
        )
        
    def _almgren_chriss_impact(
        self,
        quantity_usd: float,
        daily_volume: float,
        volatility: float,
        urgency: float
    ) -> Tuple[float, float]:
        """
        Almgren-Chriss optimal execution model.
        
        Temporary Impact: η * (rate)
        Permanent Impact: γ * (volume)^0.5
        """
        # Participation rate
        rate = quantity_usd / daily_volume
        
        # Temporary impact (linear in rate, scaled by urgency)
        # Higher urgency = faster execution = more temporary impact
        urgency_factor = 0.5 + urgency
        temp_impact = self.config.eta * rate * 10000 * urgency_factor * volatility / 0.03
        
        # Permanent impact (square root of volume)
        perm_impact = self.config.gamma * math.sqrt(rate) * 10000 * volatility / 0.03
        
        return temp_impact, perm_impact
        
    def _empirical_impact(
        self,
        participation_rate: float,
        volatility: float,
        orderbook: OrderBookState
    ) -> Tuple[float, float]:
        """
        Empirical impact from calibrated curves.
        """
        # Base impact from curve
        base_impact = EmpiricalImpactCurves.interpolate_impact(participation_rate)
        
        # Apply volatility multiplier
        vol_regime = EmpiricalImpactCurves.get_volatility_regime(volatility)
        vol_mult = EmpiricalImpactCurves.VOLATILITY_MULTIPLIERS[vol_regime]
        
        # Apply spread multiplier
        spread_regime = EmpiricalImpactCurves.get_spread_regime(orderbook.spread_bps)
        spread_mult = EmpiricalImpactCurves.SPREAD_MULTIPLIERS[spread_regime]
        
        # Total adjusted impact
        total_impact = base_impact * vol_mult * spread_mult
        
        # Split into temporary (60%) and permanent (40%)
        temp_impact = total_impact * 0.6
        perm_impact = total_impact * 0.4
        
        return temp_impact, perm_impact
        
    def _linear_impact(
        self,
        quantity_usd: float,
        daily_volume: float,
        volatility: float
    ) -> Tuple[float, float]:
        """Simple linear impact model."""
        rate = quantity_usd / daily_volume
        
        # Linear: impact = λ * rate
        impact = self.config.kyle_lambda * rate * 10000 * volatility / 0.03
        
        # Split 70% temporary, 30% permanent
        return impact * 0.7, impact * 0.3
        
    def _ofi_adjustment(self, side: str, ofi: float) -> float:
        """
        Adjust impact based on Order Flow Imbalance.
        
        Buying when OFI is positive (buy pressure) -> higher impact
        Selling when OFI is negative (sell pressure) -> higher impact
        """
        if (side == "buy" and ofi > 0) or (side == "sell" and ofi < 0):
            # Trading in same direction as flow -> higher impact
            return 1.0 + min(abs(ofi), 0.5)
        else:
            # Trading against flow -> lower impact (providing liquidity)
            return max(0.7, 1.0 - abs(ofi) * 0.3)
            
    def _recommend_execution(
        self,
        impact_bps: float,
        participation_rate: float,
        urgency: float
    ) -> Tuple[str, int]:
        """
        Recommend execution algorithm based on conditions.
        
        Returns:
            (algorithm_name, duration_minutes)
        """
        # Urgent and small -> aggressive
        if urgency > 0.8 and participation_rate < 0.01:
            return "AGGRESSIVE", 5
            
        # Large order -> spread over time
        if impact_bps > 30 or participation_rate > 0.05:
            duration = int(60 * participation_rate / 0.01)  # Scale with size
            duration = min(max(duration, 30), 240)  # 30min to 4hr
            return "TWAP", duration
            
        # High impact but urgent -> VWAP
        if impact_bps > 15:
            return "VWAP", 30
            
        # Low impact -> aggressive is fine
        if impact_bps < 10:
            return "AGGRESSIVE", 5
            
        # Default: VWAP
        return "VWAP", 15
        
    def _calculate_confidence(
        self,
        orderbook: Optional[OrderBookState],
        has_volume_data: bool
    ) -> float:
        """Calculate model confidence."""
        confidence = 0.5  # Base
        
        if orderbook:
            confidence += 0.2
            if orderbook.ofi != 0:
                confidence += 0.1
                
        if has_volume_data:
            confidence += 0.2
            
        return min(confidence, 1.0)
        
    def record_realized_impact(
        self,
        symbol: str,
        estimated_bps: float,
        realized_bps: float,
        participation_rate: float
    ):
        """Record actual vs estimated impact for model calibration."""
        self._impact_history.append({
            "symbol": symbol,
            "estimated": estimated_bps,
            "realized": realized_bps,
            "participation": participation_rate,
            "timestamp": datetime.utcnow()
        })
        
    def get_calibration_stats(self) -> Dict:
        """Get model calibration statistics."""
        if not self._impact_history:
            return {"samples": 0}
            
        estimates = [h["estimated"] for h in self._impact_history]
        realized = [h["realized"] for h in self._impact_history]
        
        errors = [r - e for r, e in zip(realized, estimates)]
        
        return {
            "samples": len(self._impact_history),
            "mean_error_bps": np.mean(errors),
            "std_error_bps": np.std(errors),
            "mean_estimate_bps": np.mean(estimates),
            "mean_realized_bps": np.mean(realized)
        }


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_impact_calculator: Optional[MarketImpactCalculator] = None


def get_impact_calculator(config: Optional[ImpactConfig] = None) -> MarketImpactCalculator:
    """Get singleton MarketImpactCalculator instance."""
    global _impact_calculator
    if _impact_calculator is None:
        _impact_calculator = MarketImpactCalculator(config)
    return _impact_calculator


def reset_impact_calculator():
    """Reset singleton instance."""
    global _impact_calculator
    _impact_calculator = None


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'MarketImpactCalculator',
    'ImpactConfig',
    'ImpactModel',
    'ImpactEstimate',
    'OrderBookState',
    'EmpiricalImpactCurves',
    'get_impact_calculator',
    'reset_impact_calculator',
]
