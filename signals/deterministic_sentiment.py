"""
================================================================================
HMATS v6.4.2 - Deterministic Sentiment Engine
================================================================================
Purpose: Provide DETERMINISTIC, STRUCTURED sentiment analysis

CRITICAL DESIGN PRINCIPLES:
1. NO LLM calls in hot trading path
2. Rule-based text sentiment for DIRECTION
3. LunarCrush/Fear-Greed for CROWDING only (NOT direction)
4. Funding rate for CROWDING bias
5. All outputs are numeric, structured, and reproducible

OUTPUT CONTRACT (MANDATORY):
context["sentiment"] = {
    "direction_score": float in [-1.0, 1.0],      # bearish -> bullish
    "strength": float in [0.0, 1.0],               # confidence / intensity
    "delta": float,                                # short-term change
    "volatility": float,                           # rolling std of sentiment
    "crowding_bias": float in [-1.0, 1.0],         # from LunarCrush/funding
    "extreme_crowding": bool,
    "sources": Dict[str, Any]                      # per-source breakdown
}

PROFIT-MAX CONSTRAINTS:
- Sentiment NEVER vetoes trades
- Low sentiment -> smaller size, slower execution (not no trade)
- Extreme crowding -> risk modulation, NOT direction flip

Author: HMATS v6.4.2 LLM/Sentiment Separation
Version: 6.4.2
================================================================================
"""

import logging
import re
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
from collections import deque
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# STEP 18.3: SimpleSentimentCalculator — 6-signal weighted composite (spec-exact)
# =============================================================================

class SimpleSentimentCalculator:
    """6-signal weighted composite for short-biased crypto trading (Step 18.3 spec).

    This is the SPEC-EXACT implementation. DeterministicSentimentEngine below
    adds keyword analysis on top, but this class provides the pure numeric composite.
    """

    def compute(
        self,
        fear_greed_value: Optional[int],
        funding_rates: Dict[str, float],
        ls_ratios: Dict[str, float],
        oi_changes: Dict[str, float],
        liq_imbalance: Dict[str, float],
        dvol_zscore: Optional[float],
        vpin_values: Dict[str, float],
    ) -> Dict[str, Any]:
        signals = []  # (name, value, weight)

        # 1. Funding Rate (25%)
        avg_funding = np.mean(list(funding_rates.values())) if funding_rates else 0.0
        funding_signal = float(-np.clip(avg_funding * 100, -1.0, 1.0))
        signals.append(("funding", funding_signal, 0.25))

        # 2. Long/Short Ratio (20%)
        avg_ls = float(np.mean(list(ls_ratios.values()))) if ls_ratios else 1.0
        if avg_ls > 1.5:
            ls_signal = -0.6
        elif avg_ls > 1.2:
            ls_signal = -0.3
        elif avg_ls < 0.7:
            ls_signal = 0.3
        elif avg_ls < 0.8:
            ls_signal = 0.1
        else:
            ls_signal = 0.0
        signals.append(("ls_ratio", ls_signal, 0.20))

        # 3. Fear & Greed (15%)
        if fear_greed_value is not None:
            if fear_greed_value > 75:
                fg_signal = -0.6
            elif fear_greed_value > 55:
                fg_signal = -0.3
            elif fear_greed_value < 25:
                fg_signal = 0.0   # extreme fear → neutral (don't chase shorts)
            elif fear_greed_value < 45:
                fg_signal = -0.2
            else:
                fg_signal = 0.0
            signals.append(("fear_greed", fg_signal, 0.15))

        # 4. OI Change (15%)
        avg_oi_chg = float(np.mean(list(oi_changes.values()))) if oi_changes else 0.0
        if avg_oi_chg > 5.0:
            oi_signal = -0.4
        elif avg_oi_chg > 2.0:
            oi_signal = -0.2
        elif avg_oi_chg < -5.0:
            oi_signal = 0.2
        else:
            oi_signal = 0.0
        signals.append(("oi_change", oi_signal, 0.15))

        # 5. Liquidations (15%)
        avg_liq = float(np.mean(list(liq_imbalance.values()))) if liq_imbalance else 0.0
        liq_signal = float(-np.clip(avg_liq * 2, -0.8, 0.8))
        signals.append(("liquidations", liq_signal, 0.15))

        # 6. DVOL + VPIN (10%)
        vol_score = 0.0
        if dvol_zscore is not None:
            vol_score = min(abs(dvol_zscore) / 3.0, 1.0)
            if dvol_zscore > 2.0:
                signals.append(("dvol", -0.2, 0.05))
        avg_vpin = float(np.mean(list(vpin_values.values()))) if vpin_values else 0.5
        if avg_vpin > 0.7:
            signals.append(("vpin", -0.1, 0.05))

        # Composite
        if signals:
            weighted_sum = sum(s * w for _, s, w in signals)
            total_weight = sum(w for _, _, w in signals)
            composite = weighted_sum / total_weight if total_weight > 0 else 0.0
        else:
            composite = 0.0

        strength = min(abs(composite) * 2, 1.0)
        direction = "bearish" if composite < -0.15 else ("bullish" if composite > 0.15 else "neutral")

        # Crowding (独立, weighted_max per spec)
        funding_crowd = min(abs(avg_funding) * 200, 1.0)
        ls_crowd = max(0, (avg_ls - 1.0) / 1.0) if avg_ls > 1.0 else max(0, (1.0 - avg_ls) / 0.5)
        ls_crowd = min(ls_crowd, 1.0)
        oi_crowd = min(abs(avg_oi_chg) / 10.0, 1.0) if avg_oi_chg > 0 else 0.0
        crowding = max(funding_crowd * 0.4 + ls_crowd * 0.4 + oi_crowd * 0.2, 0.0)

        return {
            "composite_score": float(np.clip(composite, -1.0, 1.0)),
            "direction": direction,
            "strength": strength,
            "crowding_score": min(crowding, 1.0),
            "volatility": vol_score,
        }


# =============================================================================
# OUTPUT CONTRACT
# =============================================================================

@dataclass
class DeterministicSentimentOutput:
    """
    MANDATORY OUTPUT CONTRACT for all sentiment in HMATS.
    
    This is the ONLY format that downstream components accept.
    """
    # Direction from TEXT analysis only
    direction_score: float = 0.0  # [-1.0, 1.0] bearish -> bullish
    
    # Strength/confidence
    strength: float = 0.5  # [0.0, 1.0]
    
    # Short-term change (delta over last N periods)
    delta: float = 0.0
    
    # Volatility of sentiment (rolling std)
    volatility: float = 0.0
    
    # Crowding from LunarCrush/funding (NOT direction)
    crowding_bias: float = 0.0  # [-1.0, 1.0]
    
    # Extreme crowding flag
    extreme_crowding: bool = False
    
    # Per-source breakdown
    sources: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    is_stale: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for context injection."""
        return {
            "direction_score": self.direction_score,
            "strength": self.strength,
            "delta": self.delta,
            "volatility": self.volatility,
            "crowding_bias": self.crowding_bias,
            "extreme_crowding": self.extreme_crowding,
            "sources": self.sources,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "is_stale": self.is_stale,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DeterministicSentimentOutput':
        """Load from dictionary."""
        return cls(
            direction_score=data.get("direction_score", 0.0),
            strength=data.get("strength", 0.5),
            delta=data.get("delta", 0.0),
            volatility=data.get("volatility", 0.0),
            crowding_bias=data.get("crowding_bias", 0.0),
            extreme_crowding=data.get("extreme_crowding", False),
            sources=data.get("sources", {}),
            is_stale=data.get("is_stale", False),
        )
    
    def is_valid(self) -> bool:
        """Validate output is within expected ranges."""
        return (
            -1.0 <= self.direction_score <= 1.0 and
            0.0 <= self.strength <= 1.0 and
            -1.0 <= self.crowding_bias <= 1.0
        )


# =============================================================================
# CRYPTO KEYWORD SENTIMENT (Rule-based, Deterministic)
# =============================================================================

class CryptoKeywordSentiment:
    """
    Rule-based crypto sentiment from text.
    
    DETERMINISTIC: Same input always produces same output.
    NO LLM calls, NO randomness, NO temperature.
    
    Features:
    - Crypto-specific keywords (moon, rekt, diamond hands, etc.)
    - Negation handling ("not bullish" -> bearish)
    - Intensity modifiers ("extremely", "very")
    - Emoji support (🚀, 💀, 🐻, 🐂)
    """
    
    # Crypto-specific bullish keywords with weights
    BULLISH_STRONG = {
        "moon": 2.0, "mooning": 2.0, "bullish": 1.8, "pump": 1.5,
        "breakout": 1.5, "all-time high": 2.0, "ath": 2.0,
        "accumulation": 1.3, "rally": 1.5, "fomo": 1.2,
        "institutional": 1.5, "adoption": 1.3, "etf approved": 2.0,
        "partnership": 1.2, "upgrade": 1.1, "milestone": 1.0,
        "diamond hands": 1.5, "hodl": 1.2, "wagmi": 1.5,
        "send it": 1.3, "lfg": 1.5, "to the moon": 2.0,
    }
    
    BULLISH_MODERATE = {
        "buy": 0.8, "long": 0.9, "support": 0.7, "bounce": 0.8,
        "recovery": 0.7, "oversold": 0.8, "value": 0.6,
        "opportunity": 0.7, "dip": 0.5, "green": 0.5,
        "bullish divergence": 1.2, "higher low": 0.8,
    }
    
    BEARISH_STRONG = {
        "crash": 2.0, "rekt": 2.0, "bearish": 1.8, "dump": 1.5,
        "liquidation": 2.0, "hack": 2.0, "exploit": 2.0,
        "ban": 1.8, "regulation": 1.3, "investigation": 1.5,
        "fraud": 2.0, "rug pull": 2.5, "rugpull": 2.5,
        "bankruptcy": 2.5, "insolvent": 2.5, "scam": 2.0,
        "ngmi": 1.5, "dead cat bounce": 1.5,
    }
    
    BEARISH_MODERATE = {
        "sell": 0.8, "short": 0.9, "resistance": 0.6, "rejection": 0.8,
        "overbought": 0.8, "distribution": 0.7, "weakness": 0.6,
        "concern": 0.5, "red": 0.4, "lower high": 0.8,
        "bearish divergence": 1.2, "breakdown": 1.0,
    }
    
    # Emoji mappings
    BULLISH_EMOJI = {
        "🚀": 1.5, "🌙": 1.2, "🐂": 1.3, "💎": 1.0, "🙌": 0.8,
        "📈": 1.0, "💰": 0.8, "🔥": 0.7, "ON": 0.5, "💪": 0.6,
        "🤑": 0.8, "🎯": 0.5,
    }
    
    BEARISH_EMOJI = {
        "💀": 1.5, "🐻": 1.3, "📉": 1.0, "OFF": 0.8, "🔴": 0.6,
        "[WARN]️": 0.7, "💔": 0.6, "😱": 0.8, "🩸": 1.0,
    }
    
    # Negation words
    NEGATION_WORDS = {"not", "no", "never", "dont", "don't", "isn't", "aren't", "wasn't", "weren't", "won't", "cannot", "can't"}
    
    # Intensity modifiers
    INTENSITY_BOOST = {"extremely": 1.5, "very": 1.3, "super": 1.3, "highly": 1.2, "absolutely": 1.4, "definitely": 1.2}
    INTENSITY_REDUCE = {"slightly": 0.6, "somewhat": 0.7, "maybe": 0.5, "possibly": 0.5, "minor": 0.6}
    
    def __init__(self):
        """Initialize the keyword sentiment analyzer."""
        # Compile all patterns for efficiency
        self._all_bullish = {**self.BULLISH_STRONG, **self.BULLISH_MODERATE}
        self._all_bearish = {**self.BEARISH_STRONG, **self.BEARISH_MODERATE}
        logger.info("[DETERMINISTIC_SENTIMENT] CryptoKeywordSentiment initialized")
    
    def analyze(self, text: str) -> Tuple[float, float]:
        """
        Analyze text for crypto sentiment.
        
        DETERMINISTIC: Same text always produces same output.
        
        Args:
            text: Raw text to analyze
            
        Returns:
            Tuple of (direction_score, strength)
            direction_score: -1.0 (bearish) to +1.0 (bullish)
            strength: 0.0 to 1.0 (confidence/intensity)
        """
        if not text or not text.strip():
            return 0.0, 0.0
        
        text_lower = text.lower()
        words = text_lower.split()
        
        bullish_score = 0.0
        bearish_score = 0.0
        match_count = 0
        
        # Check text keywords
        for keyword, weight in self._all_bullish.items():
            if keyword in text_lower:
                # Check for negation
                negated = self._is_negated(text_lower, keyword)
                if negated:
                    bearish_score += weight * 0.8  # Flip with slight reduction
                else:
                    bullish_score += weight
                match_count += 1
        
        for keyword, weight in self._all_bearish.items():
            if keyword in text_lower:
                negated = self._is_negated(text_lower, keyword)
                if negated:
                    bullish_score += weight * 0.8  # Flip with slight reduction
                else:
                    bearish_score += weight
                match_count += 1
        
        # Check emojis
        for emoji, weight in self.BULLISH_EMOJI.items():
            count = text.count(emoji)
            if count > 0:
                bullish_score += weight * min(count, 3)  # Cap at 3
                match_count += count
        
        for emoji, weight in self.BEARISH_EMOJI.items():
            count = text.count(emoji)
            if count > 0:
                bearish_score += weight * min(count, 3)
                match_count += count
        
        # Apply intensity modifiers
        intensity_mult = self._get_intensity_multiplier(words)
        bullish_score *= intensity_mult
        bearish_score *= intensity_mult
        
        # Compute direction score
        total = bullish_score + bearish_score
        if total == 0:
            return 0.0, 0.0
        
        direction_score = (bullish_score - bearish_score) / max(total, 1.0)
        direction_score = max(-1.0, min(1.0, direction_score))
        
        # Compute strength based on match count and text length
        word_count = len(words)
        match_density = match_count / max(word_count, 1)
        strength = min(1.0, match_density * 5 + min(total, 5) / 10)
        
        return direction_score, strength
    
    def _is_negated(self, text: str, keyword: str) -> bool:
        """Check if keyword is negated in text."""
        # Find keyword position
        pos = text.find(keyword)
        if pos == -1:
            return False
        
        # Check words before keyword (within 3 words)
        before = text[:pos].split()[-3:]
        for word in before:
            if word in self.NEGATION_WORDS:
                return True
        return False
    
    def _get_intensity_multiplier(self, words: List[str]) -> float:
        """Get intensity multiplier from modifier words."""
        mult = 1.0
        for word in words:
            if word in self.INTENSITY_BOOST:
                mult = max(mult, self.INTENSITY_BOOST[word])
            elif word in self.INTENSITY_REDUCE:
                mult = min(mult, self.INTENSITY_REDUCE[word])
        return mult


# =============================================================================
# CROWDING ADAPTER (LunarCrush / Fear-Greed / Funding)
# =============================================================================

class CrowdingAdapter:
    """
    Adapts external sentiment providers to CROWDING signals.
    
    CRITICAL RULES:
    - LunarCrush / Fear-Greed / Funding = CROWDING, not direction
    - crowding_bias is SEPARATE from direction_score
    - Extreme crowding -> risk modulation, NOT direction flip
    - Crowding NEVER vetoes trades
    
    Mapping:
    - High Fear/Greed -> crowding_bias near extremes
    - Positive funding -> bearish crowding (longs overcrowded)
    - Negative funding -> bullish crowding (shorts overcrowded)
    """
    
    def __init__(
        self,
        extreme_fear_threshold: int = 20,
        extreme_greed_threshold: int = 80,
        extreme_funding_threshold: float = 0.001,  # 0.1% per 8h
    ):
        self.extreme_fear_threshold = extreme_fear_threshold
        self.extreme_greed_threshold = extreme_greed_threshold
        self.extreme_funding_threshold = extreme_funding_threshold
        
        # History for delta calculation
        self._crowding_history = deque(maxlen=24)
        
        logger.info("[CROWDING_ADAPTER] Initialized")
    
    def compute_crowding(
        self,
        fear_greed_value: Optional[int] = None,
        funding_rate: Optional[float] = None,
        lunarcrush_galaxy_score: Optional[float] = None,
        lunarcrush_alt_rank: Optional[int] = None,
        # [2026-04-11] L1 Step 18 completion: 4 missing signals
        oi_change_24h_pct: Optional[float] = None,
        liquidation_imbalance: Optional[float] = None,
        dvol_zscore: Optional[float] = None,
        vpin: Optional[float] = None,
    ) -> Tuple[float, bool]:
        """
        Compute crowding bias from 6 signal sources (Step 18 spec).

        Weights per Step 18.2:
          Funding Rate: 25%, LS Ratio (via LunarCrush proxy): 10%,
          Fear & Greed: 15%, OI Change: 15%, Liquidations: 15%,
          DVOL+VPIN: 10%, LunarCrush social: 10%

        IMPORTANT: This is NOT direction. Crowding measures market overcrowding.

        Returns:
            Tuple of (crowding_bias, extreme_crowding)
            crowding_bias: -1.0 (crowded short) to +1.0 (crowded long)
        """
        components = []
        weights = []

        # 1. Funding Rate (25%) — positive = longs pay = crowded longs
        if funding_rate is not None:
            funding_crowding = np.clip(funding_rate * 1000, -1.0, 1.0)
            components.append(funding_crowding)
            weights.append(0.25)

        # 2. Fear/Greed (15%) — high greed = crowded longs
        if fear_greed_value is not None:
            fg_crowding = (fear_greed_value - 50) / 50  # -1 to +1
            components.append(fg_crowding)
            weights.append(0.15)

        # 3. OI Change 24h (15%) — rising OI = leveraging up = crowded
        if oi_change_24h_pct is not None:
            # >5% OI rise = strong crowding signal
            oi_crowding = np.clip(oi_change_24h_pct / 10.0, -1.0, 1.0)
            components.append(oi_crowding)
            weights.append(0.15)

        # 4. Liquidation Imbalance (15%) — positive = more long liquidations
        if liquidation_imbalance is not None:
            # liq_imbalance > 0 = long liquidations dominate → bearish cascade
            liq_crowding = np.clip(liquidation_imbalance * 2.0, -1.0, 1.0)
            components.append(-liq_crowding)  # negative because long liqs = longs were overcrowded
            weights.append(0.15)

        # 5. DVOL + VPIN (10%) — high vol/toxicity = market stress
        _vol_component = 0.0
        _vol_has_signal = False
        if dvol_zscore is not None and abs(dvol_zscore) > 0.5:
            _vol_component += np.clip(-dvol_zscore / 3.0, -0.5, 0.5)  # high dvol = bearish stress
            _vol_has_signal = True
        if vpin is not None and vpin > 0.5:
            _vol_component += np.clip(-(vpin - 0.5) * 2.0, -0.5, 0.0)  # high VPIN = toxic flow
            _vol_has_signal = True
        if _vol_has_signal:
            components.append(np.clip(_vol_component, -1.0, 1.0))
            weights.append(0.10)

        # 6. LunarCrush social hype (10%) — fallback social signal
        if lunarcrush_galaxy_score is not None:
            lc_crowding = np.clip((lunarcrush_galaxy_score - 50) / 50, -1.0, 1.0)
            components.append(lc_crowding)
            weights.append(0.10)

        if not components:
            return 0.0, False

        # [#7] weighted_max per spec (not weighted average):
        # max(component * weight) gives heaviest-contributing signal
        # then normalize by max possible weight to stay in [-1, 1]
        weighted_components = [c * w for c, w in zip(components, weights)]
        crowding_bias = sum(weighted_components) / sum(weights) if sum(weights) > 0 else 0.0
        # Apply weighted_max: the peak contributor dominates
        max_abs_weighted = max(abs(c * w) for c, w in zip(components, weights))
        max_weight = max(weights)
        peak_crowding = max_abs_weighted / max_weight if max_weight > 0 else 0.0
        # Blend: 60% weighted_avg + 40% peak_contributor (spec: weighted_max emphasis)
        crowding_bias = 0.6 * crowding_bias + 0.4 * (peak_crowding * np.sign(sum(weighted_components)))
        crowding_bias = float(np.clip(crowding_bias, -1.0, 1.0))

        # Determine extreme crowding
        extreme_crowding = False
        if fear_greed_value is not None:
            if fear_greed_value <= self.extreme_fear_threshold or fear_greed_value >= self.extreme_greed_threshold:
                extreme_crowding = True
        if funding_rate is not None:
            if abs(funding_rate) >= self.extreme_funding_threshold:
                extreme_crowding = True
        if oi_change_24h_pct is not None and abs(oi_change_24h_pct) > 8.0:
            extreme_crowding = True
        if liquidation_imbalance is not None and abs(liquidation_imbalance) > 0.6:
            extreme_crowding = True

        # Track history
        self._crowding_history.append(crowding_bias)

        return crowding_bias, extreme_crowding
    
    def get_crowding_delta(self) -> float:
        """Get change in crowding over recent periods."""
        if len(self._crowding_history) < 2:
            return 0.0
        recent = list(self._crowding_history)
        return recent[-1] - recent[0]
    
    def get_crowding_volatility(self) -> float:
        """Get volatility (std) of crowding."""
        if len(self._crowding_history) < 3:
            return 0.0
        return float(np.std(list(self._crowding_history)))


# =============================================================================
# DETERMINISTIC SENTIMENT ENGINE (Main Interface)
# =============================================================================

class DeterministicSentimentEngine:
    """
    Main deterministic sentiment engine.
    
    Combines:
    - Rule-based text sentiment (DIRECTION)
    - Crowding adapter (CROWDING)
    
    Outputs standardized DeterministicSentimentOutput.
    
    GUARANTEES:
    - Deterministic: Same inputs -> same outputs
    - No LLM calls
    - No randomness
    - Backtestable
    """
    
    def __init__(self):
        self.keyword_analyzer = CryptoKeywordSentiment()
        self.crowding_adapter = CrowdingAdapter()
        
        # History for delta/volatility
        self._direction_history = deque(maxlen=24)
        
        logger.info("[DETERMINISTIC_SENTIMENT] Engine initialized")
    
    def compute(
        self,
        # Text inputs (for direction)
        news_texts: Optional[List[str]] = None,
        social_texts: Optional[List[str]] = None,

        # Crowding inputs (NOT direction)
        fear_greed_value: Optional[int] = None,
        funding_rate: Optional[float] = None,
        funding_rates_by_asset: Optional[Dict[str, float]] = None,
        lunarcrush_data: Optional[Dict[str, Any]] = None,

        # [2026-04-11] L1 Step 18 completion: 4 missing signals
        oi_change_24h_pct: Optional[float] = None,
        liquidation_imbalance: Optional[float] = None,
        dvol_zscore: Optional[float] = None,
        vpin: Optional[float] = None,

        # Asset focus
        asset: str = "BTC",
    ) -> DeterministicSentimentOutput:
        """
        Compute deterministic sentiment output.
        
        Args:
            news_texts: List of news headlines/content
            social_texts: List of social media posts
            fear_greed_value: Fear & Greed index (0-100)
            funding_rate: Current funding rate
            funding_rates_by_asset: Per-asset funding rates
            lunarcrush_data: LunarCrush metrics
            asset: Target asset
            
        Returns:
            DeterministicSentimentOutput with standardized format
        """
        sources = {}
        
        # === DIRECTION FROM TEXT ===
        direction_scores = []
        strengths = []
        
        # Analyze news
        if news_texts:
            news_directions = []
            news_strengths = []
            for text in news_texts:
                d, s = self.keyword_analyzer.analyze(text)
                if s > 0:  # Only count if there's signal
                    news_directions.append(d)
                    news_strengths.append(s)
            
            if news_directions:
                # Strength-weighted average
                weights = np.array(news_strengths)
                weights = weights / weights.sum() if weights.sum() > 0 else weights
                news_direction = float(np.dot(news_directions, weights))
                news_strength = float(np.mean(news_strengths))
                
                direction_scores.append(news_direction * 0.6)  # 60% weight
                strengths.append(news_strength)
                sources["news"] = {
                    "direction": news_direction,
                    "strength": news_strength,
                    "count": len(news_texts),
                }
        
        # Analyze social
        if social_texts:
            social_directions = []
            social_strengths = []
            for text in social_texts:
                d, s = self.keyword_analyzer.analyze(text)
                if s > 0:
                    social_directions.append(d)
                    social_strengths.append(s)
            
            if social_directions:
                weights = np.array(social_strengths)
                weights = weights / weights.sum() if weights.sum() > 0 else weights
                social_direction = float(np.dot(social_directions, weights))
                social_strength = float(np.mean(social_strengths))
                
                direction_scores.append(social_direction * 0.4)  # 40% weight
                strengths.append(social_strength)
                sources["social"] = {
                    "direction": social_direction,
                    "strength": social_strength,
                    "count": len(social_texts),
                }
        
        # Compute final direction
        if direction_scores:
            direction_score = float(np.sum(direction_scores))
            direction_score = np.clip(direction_score, -1.0, 1.0)
            strength = float(np.mean(strengths)) if strengths else 0.5
        else:
            direction_score = 0.0
            strength = 0.0
        
        # === CROWDING (NOT DIRECTION) ===
        effective_funding = funding_rate
        if funding_rates_by_asset and asset in funding_rates_by_asset:
            effective_funding = funding_rates_by_asset[asset]
        
        lc_galaxy = None
        lc_rank = None
        if lunarcrush_data:
            lc_galaxy = lunarcrush_data.get("galaxy_score")
            lc_rank = lunarcrush_data.get("alt_rank")
        
        crowding_bias, extreme_crowding = self.crowding_adapter.compute_crowding(
            fear_greed_value=fear_greed_value,
            funding_rate=effective_funding,
            lunarcrush_galaxy_score=lc_galaxy,
            lunarcrush_alt_rank=lc_rank,
            oi_change_24h_pct=oi_change_24h_pct,
            liquidation_imbalance=liquidation_imbalance,
            dvol_zscore=dvol_zscore,
            vpin=vpin,
        )
        
        sources["crowding"] = {
            "fear_greed": fear_greed_value,
            "funding_rate": effective_funding,
            "lunarcrush_galaxy": lc_galaxy,
            "bias": crowding_bias,
            "extreme": extreme_crowding,
        }
        
        # Track history
        self._direction_history.append(direction_score)
        
        # Compute delta and volatility
        delta = 0.0
        if len(self._direction_history) >= 2:
            delta = self._direction_history[-1] - self._direction_history[0]
        
        volatility = 0.0
        if len(self._direction_history) >= 3:
            volatility = float(np.std(list(self._direction_history)))
        
        return DeterministicSentimentOutput(
            direction_score=direction_score,
            strength=strength,
            delta=delta,
            volatility=volatility,
            crowding_bias=crowding_bias,
            extreme_crowding=extreme_crowding,
            sources=sources,
            timestamp=datetime.now(timezone.utc),
            is_stale=False,
        )
    
    def get_default_output(self) -> DeterministicSentimentOutput:
        """Get neutral default output when no data available."""
        return DeterministicSentimentOutput(
            direction_score=0.0,
            strength=0.0,
            delta=0.0,
            volatility=0.0,
            crowding_bias=0.0,
            extreme_crowding=False,
            sources={"default": True},
            is_stale=True,
        )


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_engine_instance: Optional[DeterministicSentimentEngine] = None


def get_deterministic_sentiment_engine() -> DeterministicSentimentEngine:
    """Get or create singleton engine instance."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = DeterministicSentimentEngine()
    return _engine_instance


def reset_deterministic_sentiment_engine():
    """Reset singleton instance."""
    global _engine_instance
    _engine_instance = None


# =============================================================================
# TESTS
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    print("=" * 70)
    print("Deterministic Sentiment Engine Test")
    print("=" * 70)
    
    engine = get_deterministic_sentiment_engine()
    
    # Test 1: Bullish news
    print("\n▶ Test 1: Bullish news")
    result = engine.compute(
        news_texts=[
            "Bitcoin breaks $100k as institutional adoption accelerates! 🚀",
            "ETF approved, massive inflows expected",
            "Diamond hands holding strong, WAGMI!",
        ],
        fear_greed_value=75,
        funding_rate=0.0005,
    )
    print(f"  Direction: {result.direction_score:+.3f}")
    print(f"  Strength: {result.strength:.3f}")
    print(f"  Crowding: {result.crowding_bias:+.3f}")
    print(f"  Extreme: {result.extreme_crowding}")
    assert result.direction_score > 0.3, "Should be bullish"
    
    # Test 2: Bearish news
    print("\n▶ Test 2: Bearish news")
    result = engine.compute(
        news_texts=[
            "Market crash incoming as whales dump 💀",
            "Major exchange hack reported, $500M stolen",
            "REKT! Liquidations cascade across exchanges",
        ],
        fear_greed_value=15,
        funding_rate=-0.001,
    )
    print(f"  Direction: {result.direction_score:+.3f}")
    print(f"  Strength: {result.strength:.3f}")
    print(f"  Crowding: {result.crowding_bias:+.3f}")
    print(f"  Extreme: {result.extreme_crowding}")
    assert result.direction_score < -0.3, "Should be bearish"
    assert result.extreme_crowding, "Should be extreme fear"
    
    # Test 3: Negation handling
    print("\n▶ Test 3: Negation handling")
    analyzer = CryptoKeywordSentiment()
    d1, s1 = analyzer.analyze("Bitcoin is bullish")
    d2, s2 = analyzer.analyze("Bitcoin is not bullish")
    print(f"  'Bitcoin is bullish': {d1:+.3f}")
    print(f"  'Bitcoin is not bullish': {d2:+.3f}")
    assert d1 > 0 and d2 < 0, "Negation should flip direction"
    
    # Test 4: Determinism
    print("\n▶ Test 4: Determinism check")
    reset_deterministic_sentiment_engine()
    engine1 = get_deterministic_sentiment_engine()
    r1 = engine1.compute(
        news_texts=["Bitcoin moon incoming!"],
        fear_greed_value=60,
    )
    
    reset_deterministic_sentiment_engine()
    engine2 = get_deterministic_sentiment_engine()
    r2 = engine2.compute(
        news_texts=["Bitcoin moon incoming!"],
        fear_greed_value=60,
    )
    
    print(f"  Run 1: dir={r1.direction_score:.4f}, crowd={r1.crowding_bias:.4f}")
    print(f"  Run 2: dir={r2.direction_score:.4f}, crowd={r2.crowding_bias:.4f}")
    assert r1.direction_score == r2.direction_score, "Must be deterministic"
    assert r1.crowding_bias == r2.crowding_bias, "Must be deterministic"
    
    # Test 5: Crowding without direction flip
    print("\n▶ Test 5: Extreme greed should NOT flip bullish direction")
    result = engine.compute(
        news_texts=["Bitcoin breakout confirmed! 🚀🚀🚀"],
        fear_greed_value=95,  # Extreme greed
        funding_rate=0.002,   # Very high positive funding
    )
    print(f"  Direction: {result.direction_score:+.3f} (from text)")
    print(f"  Crowding: {result.crowding_bias:+.3f} (from F&G/funding)")
    print(f"  Extreme: {result.extreme_crowding}")
    assert result.direction_score > 0, "Text direction should still be bullish"
    assert result.crowding_bias > 0.5, "Crowding should indicate overcrowded longs"
    assert result.extreme_crowding, "Should flag extreme crowding"
    
    print("\nON All tests passed!")
