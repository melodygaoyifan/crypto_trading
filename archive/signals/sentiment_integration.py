"""
================================================================================
HMATS v6.4.2 - Sentiment Integration Layer
================================================================================
Purpose: Wire deterministic sentiment into downstream trading components

This module integrates DeterministicSentimentOutput into:
1. SignalQualityScorer - affects score calculation
2. ProfitMaxAdapter - affects sizing/urgency modulation

PROFIT-MAX CONSTRAINTS (ENFORCED):
- Sentiment NEVER vetoes trades
- Low sentiment strength -> smaller size, slower execution (not no trade)
- Extreme crowding -> risk modulation, NOT direction flip
- LLM output NEVER directly sets conviction to zero

WIRING FLOW:
    DeterministicSentimentEngine.compute()
        ↓
    SentimentIntegration.integrate_into_context()
        ↓
    context["sentiment"] = DeterministicSentimentOutput.to_dict()
        ↓
    SignalQualityScorer.score(..., sentiment=context["sentiment"])
        ↓
    ProfitMaxAdapter.evaluate(..., sentiment=context["sentiment"])

Author: HMATS v6.4.2 LLM/Sentiment Separation
Version: 6.4.2
================================================================================
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Any
import numpy as np

from signals.deterministic_sentiment import (
    DeterministicSentimentOutput,
    DeterministicSentimentEngine,
    get_deterministic_sentiment_engine,
)

logger = logging.getLogger(__name__)


# =============================================================================
# SENTIMENT EFFECT CONFIGURATION
# =============================================================================

@dataclass
class SentimentEffectConfig:
    """
    Configuration for how sentiment affects trading decisions.
    
    PROFIT-MAX: All effects are MODULATION, never hard gates.
    """
    # Enable/disable sentiment effects
    enabled: bool = True
    
    # Signal quality score contribution (0-10 points out of 100)
    signal_quality_weight: float = 10.0  # Max 10 points
    
    # Direction alignment bonus/penalty
    direction_alignment_bonus: float = 0.1  # +10% when aligned
    direction_misalignment_penalty: float = 0.05  # -5% when misaligned (NOT veto)
    
    # Crowding effects
    crowding_urgency_reduction: float = 0.2  # Reduce urgency by 20% when crowded
    extreme_crowding_size_mult: float = 0.7  # 70% size when extreme crowding
    
    # Strength effects
    low_strength_threshold: float = 0.3  # Below this = low confidence
    low_strength_urgency_reduction: float = 0.15  # Reduce urgency when low strength
    
    # Volatility effects
    high_sentiment_volatility_threshold: float = 0.3
    high_volatility_urgency_reduction: float = 0.1


# =============================================================================
# SENTIMENT INTEGRATION
# =============================================================================

class SentimentIntegration:
    """
    Integrates deterministic sentiment into trading pipeline.
    
    Responsibilities:
    1. Fetch sentiment from DeterministicSentimentEngine
    2. Inject into context in standard format
    3. Compute effects for signal quality and profit max
    
    DOES NOT:
    - Veto trades
    - Override regime decisions
    - Act as binary gate
    """
    
    def __init__(self, config: SentimentEffectConfig = None):
        self.config = config or SentimentEffectConfig()
        self._engine = get_deterministic_sentiment_engine()
        
        logger.info(f"[SENTIMENT_INTEGRATION] Initialized: enabled={self.config.enabled}")
    
    def integrate_into_context(
        self,
        context: Dict[str, Any],
        news_texts: Optional[list] = None,
        social_texts: Optional[list] = None,
        fear_greed_value: Optional[int] = None,
        funding_rate: Optional[float] = None,
        asset: str = "BTC",
    ) -> Dict[str, Any]:
        """
        Compute sentiment and inject into context.
        
        Args:
            context: Trading context dictionary
            news_texts: News headlines/content
            social_texts: Social media posts
            fear_greed_value: Fear & Greed index
            funding_rate: Current funding rate
            asset: Target asset
            
        Returns:
            Updated context with sentiment field
        """
        if not self.config.enabled:
            context["sentiment"] = self._engine.get_default_output().to_dict()
            return context
        
        # Compute deterministic sentiment
        sentiment = self._engine.compute(
            news_texts=news_texts,
            social_texts=social_texts,
            fear_greed_value=fear_greed_value,
            funding_rate=funding_rate,
            asset=asset,
        )
        
        # Inject into context
        context["sentiment"] = sentiment.to_dict()
        
        return context
    
    def compute_signal_quality_effect(
        self,
        sentiment: Dict[str, Any],
        signal_direction: int,  # 1=long, -1=short
    ) -> float:
        """
        Compute sentiment's contribution to signal quality score.
        
        Returns a score from 0 to signal_quality_weight (default 10).
        
        Args:
            sentiment: Sentiment dict from context
            signal_direction: Trade direction
            
        Returns:
            Score contribution (0-10)
        """
        if not self.config.enabled or not sentiment:
            return self.config.signal_quality_weight * 0.5  # Neutral default
        
        max_score = self.config.signal_quality_weight
        score = 0.0
        
        direction_score = sentiment.get("direction_score", 0.0)
        strength = sentiment.get("strength", 0.5)
        crowding_bias = sentiment.get("crowding_bias", 0.0)
        extreme_crowding = sentiment.get("extreme_crowding", False)
        
        # 1. Direction alignment (40% of weight)
        if signal_direction != 0:
            alignment = direction_score * signal_direction
            if alignment > 0:
                # Aligned - scale by alignment strength
                score += max_score * 0.4 * min(1.0, 0.5 + alignment)
            else:
                # Misaligned - reduced score but NOT zero
                score += max_score * 0.4 * max(0.2, 0.5 + alignment)
        else:
            score += max_score * 0.4 * 0.5  # Neutral
        
        # 2. Strength/confidence (30% of weight)
        score += max_score * 0.3 * strength
        
        # 3. Crowding consideration (30% of weight)
        # Moderate crowding in trade direction is concerning
        if signal_direction != 0:
            crowding_alignment = crowding_bias * signal_direction
            if crowding_alignment > 0.5:
                # Trading with crowded side - reduce score
                score += max_score * 0.3 * (1.0 - crowding_alignment * 0.5)
            elif crowding_alignment < -0.5:
                # Trading against crowded side (contrarian) - slight bonus
                score += max_score * 0.3 * min(1.0, 1.0 + abs(crowding_alignment) * 0.2)
            else:
                score += max_score * 0.3 * 0.8
        else:
            score += max_score * 0.3 * 0.5
        
        # Clamp
        return max(0.0, min(max_score, score))
    
    def compute_profit_max_effects(
        self,
        sentiment: Dict[str, Any],
        signal_direction: int,
    ) -> Dict[str, Any]:
        """
        Compute sentiment effects for ProfitMaxAdapter.
        
        Returns modulation parameters (NEVER veto).
        
        Args:
            sentiment: Sentiment dict from context
            signal_direction: Trade direction
            
        Returns:
            Dict with modulation parameters:
            - conviction_mult: float (multiplier, minimum 0.5)
            - urgency_mult: float (multiplier, minimum 0.5)
            - size_mult: float (multiplier, minimum 0.3)
            - prefer_limit: bool
            - warnings: List[str]
        """
        effects = {
            "conviction_mult": 1.0,
            "urgency_mult": 1.0,
            "size_mult": 1.0,
            "prefer_limit": False,
            "warnings": [],
        }
        
        if not self.config.enabled or not sentiment:
            return effects
        
        direction_score = sentiment.get("direction_score", 0.0)
        strength = sentiment.get("strength", 0.5)
        crowding_bias = sentiment.get("crowding_bias", 0.0)
        extreme_crowding = sentiment.get("extreme_crowding", False)
        volatility = sentiment.get("volatility", 0.0)
        
        # 1. Direction alignment effect
        if signal_direction != 0:
            alignment = direction_score * signal_direction
            if alignment > 0.3:
                # Strongly aligned - small bonus
                effects["conviction_mult"] *= (1.0 + self.config.direction_alignment_bonus)
            elif alignment < -0.3:
                # Misaligned - small penalty (NOT veto)
                effects["conviction_mult"] *= (1.0 - self.config.direction_misalignment_penalty)
                effects["warnings"].append(
                    f"[SENTIMENT] Direction misalignment: sent={direction_score:.2f}, "
                    f"trade={signal_direction}"
                )
        
        # 2. Crowding effects
        if extreme_crowding:
            effects["size_mult"] *= self.config.extreme_crowding_size_mult
            effects["urgency_mult"] *= (1.0 - self.config.crowding_urgency_reduction)
            effects["prefer_limit"] = True
            effects["warnings"].append(
                f"[SENTIMENT] Extreme crowding detected: bias={crowding_bias:.2f}"
            )
        elif abs(crowding_bias) > 0.5:
            # Moderate crowding
            effects["urgency_mult"] *= (1.0 - self.config.crowding_urgency_reduction * 0.5)
        
        # 3. Low strength effect
        if strength < self.config.low_strength_threshold:
            effects["urgency_mult"] *= (1.0 - self.config.low_strength_urgency_reduction)
            effects["warnings"].append(
                f"[SENTIMENT] Low strength: {strength:.2f}"
            )
        
        # 4. High volatility effect
        if volatility > self.config.high_sentiment_volatility_threshold:
            effects["urgency_mult"] *= (1.0 - self.config.high_volatility_urgency_reduction)
        
        # CRITICAL: Enforce minimum multipliers (NEVER zero)
        effects["conviction_mult"] = max(0.5, effects["conviction_mult"])
        effects["urgency_mult"] = max(0.5, effects["urgency_mult"])
        effects["size_mult"] = max(0.3, effects["size_mult"])
        
        return effects
    
    def get_sentiment_summary(self, sentiment: Dict[str, Any]) -> str:
        """Get human-readable sentiment summary."""
        if not sentiment:
            return "No sentiment data"
        
        direction = sentiment.get("direction_score", 0.0)
        strength = sentiment.get("strength", 0.0)
        crowding = sentiment.get("crowding_bias", 0.0)
        extreme = sentiment.get("extreme_crowding", False)
        
        # Direction label
        if direction > 0.3:
            dir_label = "bullish"
        elif direction < -0.3:
            dir_label = "bearish"
        else:
            dir_label = "neutral"
        
        # Crowding label
        if extreme:
            crowd_label = "EXTREME"
        elif abs(crowding) > 0.5:
            crowd_label = "high"
        else:
            crowd_label = "normal"
        
        return (
            f"direction={dir_label}({direction:+.2f}), "
            f"strength={strength:.2f}, "
            f"crowding={crowd_label}({crowding:+.2f})"
        )


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_integration_instance: Optional[SentimentIntegration] = None


def get_sentiment_integration(config: SentimentEffectConfig = None) -> SentimentIntegration:
    """Get or create singleton instance."""
    global _integration_instance
    if _integration_instance is None:
        _integration_instance = SentimentIntegration(config)
    return _integration_instance


def reset_sentiment_integration():
    """Reset singleton instance."""
    global _integration_instance
    _integration_instance = None


# =============================================================================
# TESTS
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    print("=" * 70)
    print("Sentiment Integration Test")
    print("=" * 70)
    
    integration = get_sentiment_integration()
    
    # Test 1: Context injection
    print("\n▶ Test 1: Context injection")
    context = {"regime": "TREND", "phase": "EXPANSION"}
    context = integration.integrate_into_context(
        context,
        news_texts=["Bitcoin breakout! 🚀", "Institutional buying accelerates"],
        fear_greed_value=65,
        funding_rate=0.0003,
    )
    print(f"  Context has sentiment: {'sentiment' in context}")
    print(f"  Direction: {context['sentiment']['direction_score']:+.3f}")
    print(f"  Crowding: {context['sentiment']['crowding_bias']:+.3f}")
    
    # Test 2: Signal quality effect
    print("\n▶ Test 2: Signal quality effect")
    score_aligned = integration.compute_signal_quality_effect(
        context["sentiment"], signal_direction=1  # Long, aligned with bullish
    )
    score_misaligned = integration.compute_signal_quality_effect(
        context["sentiment"], signal_direction=-1  # Short, misaligned
    )
    print(f"  Aligned (long): {score_aligned:.2f}")
    print(f"  Misaligned (short): {score_misaligned:.2f}")
    assert score_aligned > score_misaligned, "Aligned should score higher"
    
    # Test 3: Profit max effects
    print("\n▶ Test 3: Profit max effects")
    effects = integration.compute_profit_max_effects(
        context["sentiment"], signal_direction=1
    )
    print(f"  Conviction mult: {effects['conviction_mult']:.3f}")
    print(f"  Urgency mult: {effects['urgency_mult']:.3f}")
    print(f"  Size mult: {effects['size_mult']:.3f}")
    print(f"  Prefer limit: {effects['prefer_limit']}")
    
    # Test 4: Extreme crowding
    print("\n▶ Test 4: Extreme crowding effect")
    context_extreme = integration.integrate_into_context(
        {},
        news_texts=["Moon! 🚀🚀🚀"],
        fear_greed_value=92,  # Extreme greed
        funding_rate=0.002,   # Very high
    )
    effects_extreme = integration.compute_profit_max_effects(
        context_extreme["sentiment"], signal_direction=1
    )
    print(f"  Extreme crowding: {context_extreme['sentiment']['extreme_crowding']}")
    print(f"  Size mult: {effects_extreme['size_mult']:.3f}")
    print(f"  Warnings: {effects_extreme['warnings']}")
    assert effects_extreme["size_mult"] < 1.0, "Should reduce size on extreme crowding"
    
    # Test 5: CRITICAL - Never veto
    print("\n▶ Test 5: CRITICAL - Never veto check")
    worst_case = {
        "direction_score": -1.0,  # Strongly bearish
        "strength": 0.1,          # Low confidence
        "crowding_bias": 1.0,     # Extreme crowded longs
        "extreme_crowding": True,
        "volatility": 0.5,
    }
    effects_worst = integration.compute_profit_max_effects(worst_case, signal_direction=1)
    print(f"  Worst case scenario:")
    print(f"    Conviction mult: {effects_worst['conviction_mult']:.3f} (min 0.5)")
    print(f"    Urgency mult: {effects_worst['urgency_mult']:.3f} (min 0.5)")
    print(f"    Size mult: {effects_worst['size_mult']:.3f} (min 0.3)")
    
    assert effects_worst["conviction_mult"] >= 0.5, "CRITICAL: conviction must be >= 0.5"
    assert effects_worst["urgency_mult"] >= 0.5, "CRITICAL: urgency must be >= 0.5"
    assert effects_worst["size_mult"] >= 0.3, "CRITICAL: size must be >= 0.3"
    
    print("\nON All tests passed! Sentiment integration is profit-max compliant.")
