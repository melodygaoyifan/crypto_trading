"""
================================================================================
HMATS v6.4.1 - False Breakout Detector
================================================================================
Purpose: Detect false breakouts using ATR + Volume + RSI proxy heuristics

This module provides a HARD VETO capability for breakout/retest entry types ONLY.
It does NOT affect other entry types (momentum, mean-revert, etc).

DESIGN:
- Lightweight, no external dependencies
- Multiple heuristics combined for confidence scoring
- Configurable thresholds for different market conditions

Author: HMATS P1 Upgrade
Version: 6.4.1
================================================================================
"""

import logging
from dataclasses import dataclass, field, fields as dataclass_fields
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import numpy as np

logger = logging.getLogger("FalseBreakoutDetector")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class FalseBreakoutConfig:
    """Configuration for false breakout detection."""
    
    # Master switch
    enabled: bool = True
    
    # Entry types to apply veto (ONLY these get vetoed)
    # v6.5.1: Removed "breakdown" - hard veto ONLY for breakout/retest
    veto_entry_types: List[str] = field(default_factory=lambda: ["breakout", "retest"])
    
    # Heuristic thresholds
    volume_median_ratio_threshold: float = 0.8  # Below this = weak volume
    atr_rejection_multiplier: float = 1.5       # Wick > 1.5*ATR = rejection signal
    rsi_divergence_threshold: float = 10.0      # RSI divergence points
    bars_to_check_reentry: int = 3              # Bars to check for range re-entry
    
    # Confidence thresholds
    min_criteria_for_veto: int = 2              # Need at least 2 signals to veto
    high_confidence_threshold: float = 0.7      # Above this = strong false breakout
    
    # Safe defaults for missing data
    default_atr: float = 0.01                   # 1% as safe default
    default_volume_ratio: float = 1.0
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FalseBreakoutConfig':
        """Load from config dict with safe defaults."""
        fb = data.get("false_breakout", {})
        defaults = cls()
        kwargs = {
            f.name: fb.get(f.name, getattr(defaults, f.name))
            for f in dataclass_fields(cls)
        }
        return cls(**kwargs)


# =============================================================================
# DETECTION RESULT
# =============================================================================

class FalseBreakoutReason(Enum):
    """Reasons why a breakout might be false."""
    VOLUME_WEAK = "volume_below_median"
    PRICE_REENTRY = "price_reentered_range"
    ATR_REJECTION = "atr_normalized_rejection"
    RSI_DIVERGENCE = "rsi_divergence"
    WICK_RATIO_HIGH = "high_wick_to_body_ratio"


@dataclass
class FalseBreakoutResult:
    """Result of false breakout detection."""
    is_false_breakout: bool
    confidence: float  # 0-1
    reasons: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    
    # Individual criterion results
    volume_weak: bool = False
    price_reentered: bool = False
    atr_rejection: bool = False
    rsi_divergence: bool = False
    wick_ratio_high: bool = False
    
    def to_dict(self) -> Dict:
        return {
            "is_false_breakout": self.is_false_breakout,
            "confidence": self.confidence,
            "reasons": self.reasons,
            "details": self.details,
            "criteria": {
                "volume_weak": self.volume_weak,
                "price_reentered": self.price_reentered,
                "atr_rejection": self.atr_rejection,
                "rsi_divergence": self.rsi_divergence,
                "wick_ratio_high": self.wick_ratio_high,
            }
        }


# =============================================================================
# DETECTOR IMPLEMENTATION
# =============================================================================

class FalseBreakoutDetector:
    """
    Detects potential false breakouts using multiple heuristics.
    
    This detector is designed to VETO only breakout/retest entries,
    allowing other entry types to pass through unaffected.
    
    FAIL-CLOSED: Missing data -> returns safe default (no veto)
    """
    
    def __init__(self, config: FalseBreakoutConfig = None):
        """
        Initialize detector.
        
        Args:
            config: Configuration (uses defaults if None)
        """
        self.config = config or FalseBreakoutConfig()
        
        # Statistics
        self._total_checks = 0
        self._vetos_issued = 0
        
        logger.info(
            f"[FALSE_BREAKOUT] Initialized: enabled={self.config.enabled}, "
            f"veto_types={self.config.veto_entry_types}"
        )
    
    def detect(
        self,
        price_series: List[float],
        atr: Optional[float] = None,
        volume_series: Optional[List[float]] = None,
        rsi_or_momentum: Optional[float] = None,
        breakout_level: Optional[float] = None,
        direction: int = 0,  # 1=long, -1=short
        entry_type: str = "unknown",
        context_tags: Optional[Dict] = None
    ) -> FalseBreakoutResult:
        """
        Detect if current situation shows signs of a false breakout.
        
        Args:
            price_series: Recent close prices (most recent last)
            atr: Average True Range
            volume_series: Recent volumes (most recent last)
            rsi_or_momentum: RSI or momentum proxy (0-100 scale)
            breakout_level: Price level that was broken
            direction: Trade direction (1=long, -1=short)
            entry_type: Type of entry (breakout, retest, momentum, etc)
            context_tags: Additional context
            
        Returns:
            FalseBreakoutResult with detection results
        """
        self._total_checks += 1
        
        # Early return if disabled or not a veto-eligible entry type
        if not self.config.enabled:
            return self._safe_result("disabled")
        
        if entry_type.lower() not in [t.lower() for t in self.config.veto_entry_types]:
            return self._safe_result(f"entry_type={entry_type} not in veto list")
        
        # Validate inputs - fail safe if insufficient data
        if not price_series or len(price_series) < 3:
            return self._safe_result("insufficient_price_data")
        
        # Calculate individual criteria
        reasons = []
        details = {}
        criteria_met = 0
        
        # Safe conversions
        prices = np.array(price_series[-20:] if len(price_series) > 20 else price_series)
        current_price = prices[-1]
        
        # Use safe defaults for missing data
        atr_value = atr if atr and atr > 0 else self.config.default_atr * current_price
        
        # 1. Volume weakness check
        volume_weak = False
        if volume_series and len(volume_series) >= 3:
            volumes = np.array(volume_series[-20:] if len(volume_series) > 20 else volume_series)
            median_volume = np.median(volumes[:-1]) if len(volumes) > 1 else volumes[0]
            current_volume = volumes[-1]
            
            if median_volume > 0:
                volume_ratio = current_volume / median_volume
                details["volume_ratio"] = volume_ratio
                
                if volume_ratio < self.config.volume_median_ratio_threshold:
                    volume_weak = True
                    criteria_met += 1
                    reasons.append(FalseBreakoutReason.VOLUME_WEAK.value)
        
        # 2. Price re-entry check (price went back into prior range)
        price_reentered = False
        if breakout_level is not None and len(prices) >= self.config.bars_to_check_reentry + 1:
            recent_prices = prices[-(self.config.bars_to_check_reentry + 1):]
            
            if direction > 0:  # Long breakout
                # Check if price dropped back below breakout level
                broke_above = any(p > breakout_level for p in recent_prices[:-1])
                now_below = current_price < breakout_level
                if broke_above and now_below:
                    price_reentered = True
                    criteria_met += 1
                    reasons.append(FalseBreakoutReason.PRICE_REENTRY.value)
            else:  # Short breakdown
                # Check if price rose back above breakout level
                broke_below = any(p < breakout_level for p in recent_prices[:-1])
                now_above = current_price > breakout_level
                if broke_below and now_above:
                    price_reentered = True
                    criteria_met += 1
                    reasons.append(FalseBreakoutReason.PRICE_REENTRY.value)
            
            details["breakout_level"] = breakout_level
            details["price_reentered"] = price_reentered
        
        # 3. ATR-normalized rejection (large wick/reversal)
        atr_rejection = False
        if len(prices) >= 3 and atr_value > 0:
            # Calculate reversal magnitude from recent high/low
            recent_high = np.max(prices[-5:])
            recent_low = np.min(prices[-5:])
            
            if direction > 0:  # Long - check rejection from highs
                rejection_magnitude = recent_high - current_price
            else:  # Short - check rejection from lows
                rejection_magnitude = current_price - recent_low
            
            normalized_rejection = rejection_magnitude / atr_value
            details["atr_rejection_ratio"] = normalized_rejection
            
            if normalized_rejection > self.config.atr_rejection_multiplier:
                atr_rejection = True
                criteria_met += 1
                reasons.append(FalseBreakoutReason.ATR_REJECTION.value)
        
        # 4. RSI divergence check
        rsi_divergence = False
        if rsi_or_momentum is not None:
            # For long breakout: RSI should be strong (>50)
            # For short breakdown: RSI should be weak (<50)
            if direction > 0 and rsi_or_momentum < (50 - self.config.rsi_divergence_threshold):
                rsi_divergence = True
                criteria_met += 1
                reasons.append(FalseBreakoutReason.RSI_DIVERGENCE.value)
            elif direction < 0 and rsi_or_momentum > (50 + self.config.rsi_divergence_threshold):
                rsi_divergence = True
                criteria_met += 1
                reasons.append(FalseBreakoutReason.RSI_DIVERGENCE.value)
            
            details["rsi_value"] = rsi_or_momentum
        
        # 5. High wick-to-body ratio (reversal candle)
        wick_ratio_high = False
        if len(prices) >= 2:
            # Approximate wick ratio from close-to-close vs high-low
            # This is a simplified check - real implementation would use OHLC
            close_change = abs(prices[-1] - prices[-2])
            if close_change > 0 and atr_value > 0:
                # If body is small relative to ATR, likely reversal
                body_to_atr = close_change / atr_value
                if body_to_atr < 0.3:  # Small body = potential reversal
                    wick_ratio_high = True
                    criteria_met += 1
                    reasons.append(FalseBreakoutReason.WICK_RATIO_HIGH.value)
                
                details["body_to_atr_ratio"] = body_to_atr
        
        # Calculate confidence based on criteria met
        max_criteria = 5
        confidence = min(1.0, criteria_met / max_criteria + 0.1 * criteria_met)
        
        # Determine if this is a false breakout
        is_false_breakout = (
            criteria_met >= self.config.min_criteria_for_veto or
            (criteria_met >= 1 and price_reentered)  # Price re-entry is strong signal
        )
        
        if is_false_breakout:
            self._vetos_issued += 1
            logger.warning(
                f"[FALSE_BREAKOUT] DETECTED: confidence={confidence:.2f}, "
                f"criteria_met={criteria_met}, reasons={reasons}"
            )
        
        return FalseBreakoutResult(
            is_false_breakout=is_false_breakout,
            confidence=confidence,
            reasons=reasons,
            details=details,
            volume_weak=volume_weak,
            price_reentered=price_reentered,
            atr_rejection=atr_rejection,
            rsi_divergence=rsi_divergence,
            wick_ratio_high=wick_ratio_high,
        )
    
    def _safe_result(self, reason: str) -> FalseBreakoutResult:
        """Return safe default result (no veto)."""
        return FalseBreakoutResult(
            is_false_breakout=False,
            confidence=0.0,
            reasons=[],
            details={"safe_skip_reason": reason}
        )
    
    def get_stats(self) -> Dict:
        """Get detection statistics."""
        return {
            "total_checks": self._total_checks,
            "vetos_issued": self._vetos_issued,
            "veto_rate": self._vetos_issued / max(1, self._total_checks),
        }
    
    def reset_stats(self):
        """Reset statistics."""
        self._total_checks = 0
        self._vetos_issued = 0


# =============================================================================
# SINGLETON ACCESSOR
# =============================================================================

_detector_instance: Optional[FalseBreakoutDetector] = None


def get_false_breakout_detector(config: FalseBreakoutConfig = None) -> FalseBreakoutDetector:
    """Get or create the singleton detector instance."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = FalseBreakoutDetector(config)
    elif config is not None and _detector_instance.config != config:
        logger.info("FalseBreakoutDetector config changed; refreshing singleton instance")
        _detector_instance = FalseBreakoutDetector(config)
    return _detector_instance


def reset_false_breakout_detector():
    """Reset the singleton instance."""
    global _detector_instance
    _detector_instance = None
