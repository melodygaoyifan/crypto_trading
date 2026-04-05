"""
================================================================================
OVERRIDE 3: CRACK DECAY SOFTENING (Trend Continuation)
================================================================================
Version: 3.3-HR
Purpose: Slow CRACK decay during confirmed trend continuation

RULE:
    During confirmed trend continuation:
    - Slow decay rate from 0.7^n to 0.85^n
    - Clamp minimum CRACK weight at 0.40
    - Pause decay entirely while strong momentum persists

TREND CONTINUATION CONDITIONS:
    - Momentum aligned with CRACK direction
    - Volume sustained (> 0.8× average)
    - No reversal signal from any agent
    - RSI not extreme (25-75)

WHY THIS EXISTS:
    Standard CRACK decay (0.7^n) is too aggressive for strong trends.
    After 3 bars: 0.7^3 = 34% weight remaining
    After 4 bars: 0.7^4 = 24% weight remaining
    
    This causes premature exit from trending moves.
    With softened decay:
    After 3 bars: 0.85^3 = 61% weight remaining
    After 4 bars: 0.85^4 = 52% weight remaining
    
    Combined with 40% floor, we stay in trends longer.
================================================================================
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
from datetime import datetime

from market.high_risk_config import get_hr_config, OverrideMode

logger = logging.getLogger(__name__)


@dataclass
class TrendConditions:
    """Current trend continuation conditions."""
    momentum_aligned: bool = False
    volume_sustained: bool = False
    no_reversal_signal: bool = True
    rsi_not_extreme: bool = True
    
    # Raw values
    momentum_value: float = 0.0
    crack_direction: str = "none"  # "long" or "short"
    volume_ratio: float = 1.0
    rsi_value: float = 50.0
    reversal_signals: int = 0
    
    @property
    def all_met(self) -> bool:
        """Check if all conditions are met."""
        return (
            self.momentum_aligned and
            self.volume_sustained and
            self.no_reversal_signal and
            self.rsi_not_extreme
        )
    
    def to_dict(self) -> Dict:
        return {
            "all_met": self.all_met,
            "momentum_aligned": self.momentum_aligned,
            "volume_sustained": self.volume_sustained,
            "no_reversal_signal": self.no_reversal_signal,
            "rsi_not_extreme": self.rsi_not_extreme,
            "momentum": self.momentum_value,
            "volume_ratio": self.volume_ratio,
            "rsi": self.rsi_value,
        }


@dataclass
class CrackDecayState:
    """Current CRACK decay state."""
    
    # Current CRACK
    crack_direction: str = "none"
    crack_bars_elapsed: int = 0
    
    # Weight calculation
    standard_weight: float = 1.0    # What standard decay would give
    adjusted_weight: float = 1.0    # What softened decay gives
    effective_weight: float = 1.0   # Final weight in use
    
    # Override status
    softening_active: bool = False
    decay_rate_used: float = 0.70
    floor_applied: bool = False
    
    # Trend conditions
    trend_conditions: TrendConditions = field(default_factory=TrendConditions)
    
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict:
        return {
            "crack_direction": self.crack_direction,
            "bars_elapsed": self.crack_bars_elapsed,
            "standard_weight": self.standard_weight,
            "adjusted_weight": self.adjusted_weight,
            "effective_weight": self.effective_weight,
            "softening_active": self.softening_active,
            "decay_rate": self.decay_rate_used,
            "floor_applied": self.floor_applied,
            "trend_conditions": self.trend_conditions.to_dict(),
        }


class CrackDecaySoftener:
    """
    Manages CRACK decay with trend-aware softening.
    
    USAGE:
        softener = CrackDecaySoftener()
        
        # On each bar
        state = softener.calculate_crack_weight(
            crack_direction="long",
            bars_elapsed=3,
            momentum=0.5,      # Positive = up momentum
            volume_ratio=1.2,  # Current / average
            rsi=65,
            has_reversal_signal=False,
        )
        
        # Use state.effective_weight instead of standard calculation
        effective_crack = crack_base_strength * state.effective_weight
    """
    
    def __init__(self):
        self.config = get_hr_config()
        self.cfg = self.config.crack_decay_config
        
        self.state = CrackDecayState()
        
        logger.info("CrackDecaySoftener initialized")
    
    def calculate_crack_weight(
        self,
        crack_direction: str,
        bars_elapsed: int,
        momentum: float,
        volume_ratio: float,
        rsi: float,
        has_reversal_signal: bool = False,
    ) -> CrackDecayState:
        """
        Calculate CRACK weight with optional softening.
        
        Args:
            crack_direction: "long" or "short"
            bars_elapsed: Number of 4H bars since CRACK triggered
            momentum: Price momentum (-1 to 1, positive = up)
            volume_ratio: Current volume / average volume
            rsi: RSI value (0-100)
            has_reversal_signal: Whether any agent has reversal signal
        
        Returns:
            CrackDecayState with effective weight
        """
        
        self.state.crack_direction = crack_direction
        self.state.crack_bars_elapsed = bars_elapsed
        
        # Calculate standard decay
        standard_rate = self.cfg["standard_decay_rate"]
        self.state.standard_weight = standard_rate ** bars_elapsed
        
        # Check if override is enabled
        if self.config.crack_decay_override != OverrideMode.ENABLED:
            self.state.effective_weight = self.state.standard_weight
            self.state.adjusted_weight = self.state.standard_weight
            self.state.softening_active = False
            self.state.decay_rate_used = standard_rate
            return self.state
        
        # Evaluate trend conditions
        conditions = self._evaluate_conditions(
            crack_direction=crack_direction,
            momentum=momentum,
            volume_ratio=volume_ratio,
            rsi=rsi,
            has_reversal_signal=has_reversal_signal,
        )
        self.state.trend_conditions = conditions
        
        # Apply softening if conditions met
        if conditions.all_met:
            # Use slower decay rate
            trend_rate = self.cfg["trend_decay_rate"]
            adjusted_weight = trend_rate ** bars_elapsed
            
            self.state.adjusted_weight = adjusted_weight
            self.state.softening_active = True
            self.state.decay_rate_used = trend_rate
            
            logger.debug(
                f"CRACK decay softened: {standard_rate}^{bars_elapsed}={self.state.standard_weight:.2f} "
                f"-> {trend_rate}^{bars_elapsed}={adjusted_weight:.2f}"
            )
        else:
            self.state.adjusted_weight = self.state.standard_weight
            self.state.softening_active = False
            self.state.decay_rate_used = standard_rate
        
        # Apply floor
        min_weight = self.cfg["minimum_crack_weight"]
        if self.state.adjusted_weight < min_weight:
            self.state.effective_weight = min_weight
            self.state.floor_applied = True
            logger.debug(f"CRACK weight floored at {min_weight}")
        else:
            self.state.effective_weight = self.state.adjusted_weight
            self.state.floor_applied = False
        
        return self.state
    
    def _evaluate_conditions(
        self,
        crack_direction: str,
        momentum: float,
        volume_ratio: float,
        rsi: float,
        has_reversal_signal: bool,
    ) -> TrendConditions:
        """Evaluate trend continuation conditions."""
        
        cond_cfg = self.cfg["trend_conditions"]
        
        conditions = TrendConditions(
            momentum_value=momentum,
            crack_direction=crack_direction,
            volume_ratio=volume_ratio,
            rsi_value=rsi,
            reversal_signals=1 if has_reversal_signal else 0,
        )
        
        # Condition 1: Momentum aligned with CRACK direction
        if cond_cfg["momentum_aligned"]:
            if crack_direction == "long":
                conditions.momentum_aligned = momentum > 0
            elif crack_direction == "short":
                conditions.momentum_aligned = momentum < 0
            else:
                conditions.momentum_aligned = False
        else:
            conditions.momentum_aligned = True
        
        # Condition 2: Volume sustained
        if cond_cfg["volume_sustained"]:
            volume_floor = self.cfg["volume_floor_multiplier"]
            conditions.volume_sustained = volume_ratio >= volume_floor
        else:
            conditions.volume_sustained = True
        
        # Condition 3: No reversal signal
        if cond_cfg["no_reversal_signal"]:
            conditions.no_reversal_signal = not has_reversal_signal
        else:
            conditions.no_reversal_signal = True
        
        # Condition 4: RSI not extreme
        if cond_cfg["rsi_not_extreme"]:
            rsi_lower = self.cfg["rsi_lower_bound"]
            rsi_upper = self.cfg["rsi_upper_bound"]
            conditions.rsi_not_extreme = rsi_lower <= rsi <= rsi_upper
        else:
            conditions.rsi_not_extreme = True
        
        return conditions
    
    def get_weight_comparison(
        self,
        bars: int = 5,
    ) -> Dict[int, Dict[str, float]]:
        """
        Get comparison of standard vs softened decay over N bars.
        
        Useful for understanding the impact of softening.
        """
        standard_rate = self.cfg["standard_decay_rate"]
        trend_rate = self.cfg["trend_decay_rate"]
        floor = self.cfg["minimum_crack_weight"]
        
        comparison = {}
        for b in range(1, bars + 1):
            standard = standard_rate ** b
            softened = max(floor, trend_rate ** b)
            comparison[b] = {
                "standard": round(standard, 3),
                "softened": round(softened, 3),
                "difference": round(softened - standard, 3),
            }
        
        return comparison
    
    def should_pause_decay(self) -> bool:
        """
        Check if decay should be completely paused.
        
        Decay is paused when:
        - All trend conditions are met
        - Weight is already at floor
        """
        return (
            self.state.softening_active and
            self.state.floor_applied
        )


# Singleton
_crack_softener: Optional[CrackDecaySoftener] = None


def get_crack_decay_softener() -> CrackDecaySoftener:
    """Get singleton instance."""
    global _crack_softener
    if _crack_softener is None:
        _crack_softener = CrackDecaySoftener()
    return _crack_softener


def reset_crack_decay_softener():
    """Reset singleton."""
    global _crack_softener
    _crack_softener = None
