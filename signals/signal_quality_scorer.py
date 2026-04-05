"""
================================================================================
HMATS v6.4.1 - Signal Quality Scorer
================================================================================
Purpose: Compute composite signal quality score (0-100) and derive multipliers

This module aggregates multiple signal quality dimensions into a single score
that directly affects:
- conviction_multiplier: [0.6, 1.5] for sizing
- execution_urgency: [0.5, 1.5] for order aggressiveness
- allow_add: bool for pyramiding permission

DESIGN:
- Composite score from 6 dimensions
- Each dimension weighted independently
- Safe defaults for missing data
- Profit-max philosophy: avoid killing trades, prefer modulation

Author: HMATS P1 Upgrade
Version: 6.4.1
================================================================================
"""

import logging
from dataclasses import dataclass, field, fields as dataclass_fields
from typing import Dict, Optional, Any, Tuple
from enum import Enum
import math

logger = logging.getLogger("SignalQualityScorer")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SignalQualityScorerConfig:
    """Configuration for signal quality scoring."""
    
    # Master switch
    enabled: bool = True
    
    # Component weights (should sum to 100)
    # v6.4.2: Added sentiment weight, adjusted others
    weight_regime_quality: int = 25      # Regime clarity and strength
    weight_trend_alignment: int = 20     # HTF trend alignment
    weight_volatility_context: int = 15  # Vol regime appropriateness
    weight_volume_confirmation: int = 15 # Volume validation
    weight_transition_stability: int = 10 # From RegimeTransitionRisk
    weight_funding_bias: int = 5         # Crowding/funding pressure (reduced, sentiment has crowding)
    weight_sentiment: int = 10           # v6.4.2: Deterministic sentiment (direction + crowding)
    
    # Conviction multiplier bounds
    conviction_multiplier_min: float = 0.6
    conviction_multiplier_max: float = 1.5
    conviction_score_for_min: float = 30.0  # Score below this -> min multiplier
    conviction_score_for_max: float = 80.0  # Score above this -> max multiplier
    
    # Execution urgency bounds
    urgency_min: float = 0.5  # More passive (limit orders)
    urgency_max: float = 1.5  # More aggressive (market orders)
    urgency_score_threshold_low: float = 40.0
    urgency_score_threshold_high: float = 70.0
    
    # Add-on permission
    min_score_for_add: float = 60.0  # Minimum score to allow pyramiding
    min_score_for_add_by_asset: Dict[str, float] = field(default_factory=dict)
    min_score_for_add_by_asset_side: Dict[str, float] = field(default_factory=dict)
    
    # Safe defaults
    default_score_on_error: float = 50.0
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SignalQualityScorerConfig':
        """Load from config dict with safe defaults."""
        sq = data.get("signal_quality", {})
        defaults = cls()
        kwargs = {
            f.name: sq.get(f.name, getattr(defaults, f.name))
            for f in dataclass_fields(cls)
        }
        return cls(**kwargs)


# =============================================================================
# SCORING RESULT
# =============================================================================

@dataclass
class SignalQualityResult:
    """Result of signal quality scoring."""
    
    # Main outputs
    score: float  # 0-100
    conviction_multiplier: float  # [0.6, 1.5]
    execution_urgency: float  # [0.5, 1.5]
    allow_add: bool  # Pyramiding permission
    
    # Component breakdown
    breakdown: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    raw_inputs: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "score": self.score,
            "conviction_multiplier": self.conviction_multiplier,
            "execution_urgency": self.execution_urgency,
            "allow_add": self.allow_add,
            "breakdown": self.breakdown,
        }


# =============================================================================
# SCORER IMPLEMENTATION
# =============================================================================

class SignalQualityScorer:
    """
    Computes a composite signal quality score from multiple dimensions.
    
    Each dimension contributes a weighted component to the final score.
    The score is then mapped to conviction_multiplier, execution_urgency,
    and allow_add outputs.
    
    PROFIT-MAX PHILOSOPHY: 
    - We don't kill trades, we modulate them
    - Low scores -> smaller size, passive execution
    - High scores -> larger size, aggressive execution
    """
    
    def __init__(self, config: SignalQualityScorerConfig = None):
        """
        Initialize scorer.
        
        Args:
            config: Configuration (uses defaults if None)
        """
        self.config = config or SignalQualityScorerConfig()
        
        # Verify weights sum correctly (allow small tolerance)
        total_weight = (
            self.config.weight_regime_quality +
            self.config.weight_trend_alignment +
            self.config.weight_volatility_context +
            self.config.weight_volume_confirmation +
            self.config.weight_transition_stability +
            self.config.weight_funding_bias +
            self.config.weight_sentiment  # v6.4.2: Added sentiment weight
        )
        if abs(total_weight - 100) > 5:
            logger.warning(f"[SIGNAL_QUALITY] Weights sum to {total_weight}, expected 100")
        
        # Statistics
        self._total_scores = 0
        self._score_sum = 0.0
        
        logger.info(
            f"[SIGNAL_QUALITY] Initialized: enabled={self.config.enabled}, "
            f"conviction_range=[{self.config.conviction_multiplier_min}, "
            f"{self.config.conviction_multiplier_max}]"
        )
    
    def score(
        self,
        regime: str = "UNKNOWN",
        regime_confidence: float = 0.5,
        trend_direction: float = 0.0,  # -1 to +1
        signal_direction: float = 0.0,  # -1 to +1
        volatility: float = 0.0,
        volatility_regime: str = "NORMAL",
        volume_ratio: float = 1.0,  # Current vs average
        transition_stability: float = 0.5,  # From RegimeTransitionRisk
        funding_rate: float = 0.0,
        signal_side: int = 0,  # 1=long, -1=short
        sentiment: Optional[Dict] = None,  # v6.4.2: Deterministic sentiment
        **kwargs  # Additional context
    ) -> SignalQualityResult:
        """
        Compute signal quality score.
        
        Args:
            regime: Current market regime
            regime_confidence: Regime classifier confidence (0-1)
            trend_direction: HTF trend direction (-1 to +1)
            signal_direction: Signal/intent direction (-1 to +1)
            volatility: Current volatility measure
            volatility_regime: CALM/NORMAL/HIGH/EXTREME
            volume_ratio: Current volume vs recent average
            transition_stability: Regime stability score (0-1)
            funding_rate: Current funding rate
            signal_side: Trade direction (1=long, -1=short)
            sentiment: v6.4.2 Deterministic sentiment dict with direction_score, crowding_bias, etc.
            **kwargs: Additional context for future use
            
        Returns:
            SignalQualityResult with score and multipliers
        """
        if not self.config.enabled:
            return self._default_result()
        
        try:
            breakdown = {}
            asset = kwargs.get("asset")
            
            # 1. Regime Quality (0-25 points, v6.4.2 adjusted)
            regime_score = self._score_regime_quality(regime, regime_confidence)
            breakdown["regime_quality"] = regime_score
            
            # 2. Trend Alignment (0-20 points)
            trend_score = self._score_trend_alignment(
                trend_direction, signal_direction, signal_side
            )
            breakdown["trend_alignment"] = trend_score
            
            # 3. Volatility Context (0-15 points)
            vol_score = self._score_volatility_context(volatility_regime)
            breakdown["volatility_context"] = vol_score
            
            # 4. Volume Confirmation (0-15 points)
            volume_score = self._score_volume_confirmation(volume_ratio)
            breakdown["volume_confirmation"] = volume_score
            
            # 5. Transition Stability (0-10 points)
            stability_score = self._score_transition_stability(transition_stability)
            breakdown["transition_stability"] = stability_score
            
            # 6. Funding/Crowding Bias (0-5 points, v6.4.2 reduced)
            funding_score = self._score_funding_bias(funding_rate, signal_side)
            breakdown["funding_bias"] = funding_score
            
            # 7. v6.4.2: Deterministic Sentiment (0-10 points)
            sentiment_score = self._score_sentiment(sentiment, signal_side)
            breakdown["sentiment"] = sentiment_score
            
            # Total score
            total_score = (
                regime_score + trend_score + vol_score +
                volume_score + stability_score + funding_score + sentiment_score
            )
            total_score = max(0.0, min(100.0, total_score))
            
            # Derive outputs
            conviction_mult = self._compute_conviction_multiplier(total_score)
            exec_urgency = self._compute_execution_urgency(total_score)
            min_score_for_add = self._get_min_score_for_add(asset, signal_side)
            allow_add = total_score >= min_score_for_add
            
            # Track statistics
            self._total_scores += 1
            self._score_sum += total_score
            
            result = SignalQualityResult(
                score=total_score,
                conviction_multiplier=conviction_mult,
                execution_urgency=exec_urgency,
                allow_add=allow_add,
                breakdown=breakdown,
                raw_inputs={
                    "regime": regime,
                    "regime_confidence": regime_confidence,
                    "trend_direction": trend_direction,
                    "signal_direction": signal_direction,
                    "volatility_regime": volatility_regime,
                    "volume_ratio": volume_ratio,
                    "transition_stability": transition_stability,
                    "funding_rate": funding_rate,
                    "signal_side": signal_side,
                    "asset": asset,
                    "sentiment": sentiment,
                    "min_score_for_add": min_score_for_add,
                }
            )
            
            logger.debug(
                f"[SIGNAL_QUALITY] score={total_score:.1f}, "
                f"conviction={conviction_mult:.2f}, urgency={exec_urgency:.2f}, "
                f"allow_add={allow_add}, add_threshold={min_score_for_add:.1f}, "
                f"sentiment_contrib={sentiment_score:.1f}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"[SIGNAL_QUALITY] Error: {e}")
            return self._default_result()

    @staticmethod
    def _normalize_asset(asset: Optional[str]) -> str:
        return str(asset or "").upper().replace("/USD", "").replace("USD", "")

    def _get_min_score_for_add(self, asset: Optional[str], signal_side: int) -> float:
        fallback = float(self.config.min_score_for_add)
        asset_key = self._normalize_asset(asset)
        side_key = "LONG" if signal_side > 0 else "SHORT" if signal_side < 0 else ""

        if asset_key and side_key:
            by_asset_side = self.config.min_score_for_add_by_asset_side or {}
            if isinstance(by_asset_side, dict):
                try:
                    return float(by_asset_side.get(f"{asset_key}:{side_key}", fallback) or fallback)
                except Exception:
                    return fallback

        if asset_key:
            by_asset = self.config.min_score_for_add_by_asset or {}
            if isinstance(by_asset, dict):
                try:
                    return float(by_asset.get(asset_key, fallback) or fallback)
                except Exception:
                    return fallback

        return fallback
    
    def _score_regime_quality(self, regime: str, confidence: float) -> float:
        """
        Score regime quality (0-30 points).
        
        Strong, clear regimes with high confidence score highest.
        """
        max_score = self.config.weight_regime_quality
        
        # Regime strength mapping
        regime_base = {
            "STRONG_TREND": 1.0,
            "STRONG_BULL": 1.0,
            "STRONG_BEAR": 1.0,
            "TREND": 0.8,
            "MODERATE_BULL": 0.8,
            "MODERATE_BEAR": 0.8,
            "NORMAL": 0.6,
            "NEUTRAL": 0.5,
            "UNCERTAIN": 0.4,
            "TRANSITION": 0.3,
            "VOLATILE_CHOP": 0.2,
            "MEAN_REVERT": 0.3,
            "UNKNOWN": 0.3,
        }
        
        base = regime_base.get(regime.upper(), 0.4)
        
        # Confidence weighting
        confidence_factor = 0.5 + 0.5 * confidence  # [0.5, 1.0]
        
        return max_score * base * confidence_factor
    
    def _score_trend_alignment(
        self,
        trend_direction: float,
        signal_direction: float,
        signal_side: int
    ) -> float:
        """
        Score trend alignment (0-20 points).
        
        Signal aligned with HTF trend scores highest.
        """
        max_score = self.config.weight_trend_alignment
        
        # Determine effective signal direction
        effective_signal = signal_direction if signal_direction != 0 else signal_side
        
        if trend_direction == 0 or effective_signal == 0:
            return max_score * 0.5  # Neutral
        
        # Alignment check
        alignment = trend_direction * effective_signal
        
        if alignment > 0:
            # Aligned - scale by strength
            strength = min(abs(trend_direction), abs(effective_signal))
            return max_score * (0.7 + 0.3 * strength)
        else:
            # Counter-trend - penalize but don't zero out (profit-max)
            return max_score * 0.3
    
    def _score_volatility_context(self, volatility_regime: str) -> float:
        """
        Score volatility context (0-15 points).
        
        Normal/moderate volatility scores highest for trend-following.
        """
        max_score = self.config.weight_volatility_context
        
        vol_scores = {
            "CALM": 0.7,       # Low vol - good for sizing up
            "NORMAL": 1.0,    # Ideal
            "ELEVATED": 0.8,  # Still tradeable
            "HIGH": 0.5,      # Reduce size
            "EXTREME": 0.3,   # Caution
        }
        
        factor = vol_scores.get(volatility_regime.upper(), 0.6)
        return max_score * factor
    
    def _score_volume_confirmation(self, volume_ratio: float) -> float:
        """
        Score volume confirmation (0-15 points).
        
        Volume above average confirms the move.
        """
        max_score = self.config.weight_volume_confirmation
        
        if volume_ratio <= 0:
            return max_score * 0.3
        
        # Map volume ratio to score
        # < 0.5 -> poor, 0.5-1.0 -> moderate, 1.0-2.0 -> good, >2.0 -> excellent
        if volume_ratio < 0.5:
            factor = 0.3
        elif volume_ratio < 0.8:
            factor = 0.5
        elif volume_ratio < 1.0:
            factor = 0.7
        elif volume_ratio < 1.5:
            factor = 0.9
        else:
            factor = 1.0
        
        return max_score * factor
    
    def _score_transition_stability(self, stability: float) -> float:
        """
        Score regime transition stability (0-10 points).
        
        Stable regimes score higher.
        """
        max_score = self.config.weight_transition_stability
        
        # Direct mapping with floor
        stability = max(0.0, min(1.0, stability))
        factor = 0.3 + 0.7 * stability  # [0.3, 1.0]
        
        return max_score * factor
    
    def _score_funding_bias(self, funding_rate: float, signal_side: int) -> float:
        """
        Score funding/crowding bias (0-10 points).
        
        Trading against extreme funding is penalized.
        Trading with extreme funding (crowded) is also penalized.
        """
        max_score = self.config.weight_funding_bias
        
        # Funding rate thresholds (annualized)
        extreme_threshold = 0.15  # 15% annualized
        moderate_threshold = 0.05
        
        abs_funding = abs(funding_rate)
        
        if abs_funding < moderate_threshold:
            # Neutral funding - good
            return max_score * 1.0
        
        if abs_funding < extreme_threshold:
            # Moderate funding
            # Check if aligned with signal
            if signal_side != 0:
                # Positive funding + long = crowded longs (penalize)
                # Negative funding + short = crowded shorts (penalize)
                if (funding_rate > 0 and signal_side > 0) or \
                   (funding_rate < 0 and signal_side < 0):
                    return max_score * 0.6  # Crowded trade
                else:
                    return max_score * 0.9  # Counter-crowding (bonus)
            return max_score * 0.7
        
        # Extreme funding
        if signal_side != 0:
            if (funding_rate > 0 and signal_side > 0) or \
               (funding_rate < 0 and signal_side < 0):
                return max_score * 0.3  # Very crowded
            else:
                return max_score * 0.8  # Counter extreme crowding
        
        return max_score * 0.5
    
    def _score_sentiment(self, sentiment: Optional[Dict], signal_side: int) -> float:
        """
        Score deterministic sentiment (0-10 points).
        
        v6.4.2: Integrates text-based direction and crowding analysis.
        
        PROFIT-MAX: This is a score contribution, NOT a gate.
        Low scores reduce total quality but NEVER veto.
        
        Components:
        - Direction alignment with trade (50%)
        - Strength/confidence (30%)
        - Crowding consideration (20%)
        """
        max_score = self.config.weight_sentiment
        
        # No sentiment data = neutral score
        if not sentiment:
            return max_score * 0.5
        
        direction_score = sentiment.get("direction_score", 0.0)
        strength = sentiment.get("strength", 0.5)
        crowding_bias = sentiment.get("crowding_bias", 0.0)
        extreme_crowding = sentiment.get("extreme_crowding", False)
        
        score = 0.0
        
        # 1. Direction alignment (50% of weight)
        if signal_side != 0:
            alignment = direction_score * signal_side
            if alignment > 0.3:
                # Strongly aligned
                score += max_score * 0.5 * (0.7 + alignment * 0.3)
            elif alignment > 0:
                # Weakly aligned
                score += max_score * 0.5 * (0.5 + alignment * 0.5)
            elif alignment > -0.3:
                # Weakly misaligned (not a big penalty)
                score += max_score * 0.5 * (0.4 + alignment * 0.3)
            else:
                # Strongly misaligned - reduced but NOT zero
                score += max_score * 0.5 * max(0.2, 0.3 + alignment * 0.2)
        else:
            score += max_score * 0.5 * 0.5  # No signal side = neutral
        
        # 2. Strength/confidence (30% of weight)
        score += max_score * 0.3 * strength
        
        # 3. Crowding consideration (20% of weight)
        if extreme_crowding:
            # Extreme crowding = be careful with crowded side
            if signal_side != 0 and crowding_bias * signal_side > 0:
                # Trading with crowded side - lower score
                score += max_score * 0.2 * 0.3
            else:
                # Trading against crowd or no position
                score += max_score * 0.2 * 0.7
        else:
            # Normal crowding - neutral
            score += max_score * 0.2 * 0.6
        
        return max(0.0, min(max_score, score))
    
    def _compute_conviction_multiplier(self, score: float) -> float:
        """Map score to conviction multiplier."""
        low = self.config.conviction_score_for_min
        high = self.config.conviction_score_for_max
        min_mult = self.config.conviction_multiplier_min
        max_mult = self.config.conviction_multiplier_max
        
        if score <= low:
            return min_mult
        if score >= high:
            return max_mult
        
        # Linear interpolation
        ratio = (score - low) / (high - low)
        return min_mult + ratio * (max_mult - min_mult)
    
    def _compute_execution_urgency(self, score: float) -> float:
        """Map score to execution urgency."""
        low = self.config.urgency_score_threshold_low
        high = self.config.urgency_score_threshold_high
        min_urg = self.config.urgency_min
        max_urg = self.config.urgency_max
        
        if score <= low:
            return min_urg
        if score >= high:
            return max_urg
        
        # Linear interpolation
        ratio = (score - low) / (high - low)
        return min_urg + ratio * (max_urg - min_urg)
    
    def _default_result(self) -> SignalQualityResult:
        """Return safe default result."""
        return SignalQualityResult(
            score=self.config.default_score_on_error,
            conviction_multiplier=1.0,
            execution_urgency=1.0,
            allow_add=False,
            breakdown={"default": self.config.default_score_on_error},
        )
    
    def get_stats(self) -> Dict:
        """Get scoring statistics."""
        return {
            "total_scores": self._total_scores,
            "average_score": self._score_sum / max(1, self._total_scores),
        }
    
    def reset_stats(self):
        """Reset statistics."""
        self._total_scores = 0
        self._score_sum = 0.0


# =============================================================================
# SINGLETON ACCESSOR
# =============================================================================

_scorer_instance: Optional[SignalQualityScorer] = None


def get_signal_quality_scorer(config: SignalQualityScorerConfig = None) -> SignalQualityScorer:
    """Get or create the singleton scorer instance."""
    global _scorer_instance
    if _scorer_instance is None:
        _scorer_instance = SignalQualityScorer(config)
    elif config is not None and _scorer_instance.config != config:
        logger.info("SignalQualityScorer config changed; refreshing singleton instance")
        _scorer_instance = SignalQualityScorer(config)
    return _scorer_instance


def reset_signal_quality_scorer():
    """Reset the singleton instance."""
    global _scorer_instance
    _scorer_instance = None
