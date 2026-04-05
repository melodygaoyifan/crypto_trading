"""
================================================================================
HMATS v6.5 - Macro/Crowd Adapter
================================================================================

Pure transformation layer that aggregates macro and crowd data into the
standardized context format for downstream consumers.

CRITICAL CONSTRAINTS:
1. This module performs TRANSFORMATION ONLY - NO trading decisions
2. Macro data is for EVENT WINDOW and RISK MODULATION only
3. Crowd data is for CROWDING detection, NOT direction
4. NO new hard trade gates are introduced
5. Only FALSE BREAKOUT may veto trades (existing rule)

Context Output Contract:
- context["macro"]: Event windows, macro_risk, macro_shock
- context["crowd"]: Crowding, attention, panic metrics
- context["sentiment"]: Deterministic text sentiment (from v6.4.2)

================================================================================
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# OUTPUT CONTRACT (TASK C)
# =============================================================================

@dataclass
class MacroContext:
    """
    Macro context output structure.
    
    This is for EVENT WINDOW and RISK MODULATION, NOT direction prediction.
    """
    event_window: Dict[str, Any] = field(default_factory=lambda: {
        "active": False,
        "name": None,
        "t_minus_min": 0,
        "t_plus_min": 0,
        "importance": 0,
    })
    macro_shock: float = 0.0  # [-1, 1], default 0 if unavailable
    macro_risk: float = 0.5   # [0, 1], baseline neutral
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_window": self.event_window,
            "macro_shock": self.macro_shock,
            "macro_risk": self.macro_risk,
        }


@dataclass
class CrowdContext:
    """
    Crowd context output structure.
    
    CRITICAL: These are CROWDING metrics, NOT direction signals.
    High attention/crowding = RISK, not opportunity.
    """
    attention_pressure: float = 0.5    # [0, 1]
    crowding_bias: float = 0.0         # [-1, 1], contrarian indicator
    panic_score: float = 0.0           # [0, 1]
    narrative_shift: float = 0.0       # [-1, 1]
    extreme_crowding: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "attention_pressure": self.attention_pressure,
            "crowding_bias": self.crowding_bias,
            "panic_score": self.panic_score,
            "narrative_shift": self.narrative_shift,
            "extreme_crowding": self.extreme_crowding,
        }


@dataclass
class SentimentContext:
    """
    Sentiment context (from deterministic_sentiment.py v6.4.2).
    
    This is DETERMINISTIC text sentiment only.
    """
    direction_score: float = 0.0   # [-1, 1]
    strength: float = 0.5          # [0, 1]
    delta: float = 0.0
    volatility: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "direction_score": self.direction_score,
            "strength": self.strength,
            "delta": self.delta,
            "volatility": self.volatility,
        }


@dataclass
class MacroCrowdOutput:
    """Combined output from all external data sources."""
    macro: MacroContext = field(default_factory=MacroContext)
    crowd: CrowdContext = field(default_factory=CrowdContext)
    sentiment: SentimentContext = field(default_factory=SentimentContext)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "macro": self.macro.to_dict(),
            "crowd": self.crowd.to_dict(),
            "sentiment": self.sentiment.to_dict(),
        }


# =============================================================================
# MACRO CROWD ADAPTER
# =============================================================================

class MacroCrowdAdapter:
    """
    Aggregates and transforms external data into standardized context.
    
    PURE TRANSFORMATION - NO TRADING DECISIONS.
    
    Data Sources:
    - FRED: Macro event windows, financial conditions
    - Coinglass: Derivatives crowding (funding, OI, liquidations)
    - LunarCrush: Social attention/crowding (NOT direction)
    - CryptoPanic: Panic score, narrative
    - DeterministicSentiment: Text sentiment (from v6.4.2)
    """
    
    def __init__(
        self,
        fred_feed=None,
        coinglass_feed=None,
        lunarcrush_feed=None,
        cryptopanic_feed=None,
        sentiment_engine=None,
    ):
        self._fred_feed = fred_feed
        self._coinglass_feed = coinglass_feed
        self._lunarcrush_feed = lunarcrush_feed
        self._cryptopanic_feed = cryptopanic_feed
        self._sentiment_engine = sentiment_engine
        
        logger.info("[MACRO_CROWD_ADAPTER] Initialized")
    
    # =========================================================================
    # PUBLIC API
    # =========================================================================
    
    def compute(
        self,
        symbol: str = "BTC",
        news_texts: Optional[List[str]] = None,
        social_texts: Optional[List[str]] = None,
        price_direction: Optional[int] = None,  # For alignment check
    ) -> MacroCrowdOutput:
        """
        Compute aggregated context from all data sources.
        
        Args:
            symbol: Asset symbol (BTC, ETH, SOL)
            news_texts: Optional news texts for sentiment
            social_texts: Optional social texts for sentiment
            price_direction: Current price direction for alignment (-1, 0, 1)
        
        Returns:
            MacroCrowdOutput with macro, crowd, sentiment contexts
        """
        output = MacroCrowdOutput()
        
        # === MACRO CONTEXT ===
        output.macro = self._compute_macro_context()
        
        # === CROWD CONTEXT ===
        output.crowd = self._compute_crowd_context(symbol)
        
        # === SENTIMENT CONTEXT ===
        output.sentiment = self._compute_sentiment_context(news_texts, social_texts)
        
        return output
    
    def get_context_dict(
        self,
        symbol: str = "BTC",
        **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get context as nested dictionary for injection into runtime.

        Returns safe defaults if any feed call takes too long or fails.
        """
        t0 = time.monotonic()
        _TOTAL_TIMEOUT = 15.0
        try:
            output = self.compute(symbol=symbol, **kwargs)
            elapsed = time.monotonic() - t0
            if elapsed > 5.0:
                logger.warning(
                    f"[MACRO_CROWD_ADAPTER] Slow compute: {elapsed:.1f}s for {symbol}"
                )
            if elapsed > _TOTAL_TIMEOUT:
                logger.error(
                    f"[MACRO_CROWD_ADAPTER] Exceeded {_TOTAL_TIMEOUT}s budget ({elapsed:.1f}s), returning defaults"
                )
                return MacroCrowdOutput().to_dict()
            return output.to_dict()
        except Exception as e:
            logger.warning(f"[MACRO_CROWD_ADAPTER] compute failed: {e}, using safe defaults")
            return MacroCrowdOutput().to_dict()
    
    def inject_into_context(
        self,
        context: Dict[str, Any],
        symbol: str = "BTC",
        **kwargs
    ):
        """
        Inject macro/crowd/sentiment into existing context dict.
        
        Modifies context in-place.
        """
        output = self.compute(symbol=symbol, **kwargs)
        context["macro"] = output.macro.to_dict()
        context["crowd"] = output.crowd.to_dict()
        # Note: sentiment may already exist from deterministic_sentiment
        if "sentiment" not in context:
            context["sentiment"] = output.sentiment.to_dict()
    
    # =========================================================================
    # MACRO COMPUTATION (TASK D1)
    # =========================================================================
    
    def _compute_macro_context(self) -> MacroContext:
        """
        Compute macro context from FRED data.
        
        Event window formula (TASK D1):
        - event_window.active if |t_minus_min| ≤ W (default W=180)
        - macro_risk = clamp01(1 - |t_minus_min| / W) * importance_weight
        """
        ctx = MacroContext()
        
        if not self._fred_feed:
            return ctx
        
        try:
            fred_ctx = self._fred_feed.get_macro_context()
            
            # Transfer event window
            ew = fred_ctx.get("event_window", {})
            ctx.event_window = {
                "active": ew.get("active", False),
                "name": ew.get("name"),
                "t_minus_min": ew.get("t_minus_min", 0),
                "t_plus_min": ew.get("t_plus_min", 0),
                "importance": ew.get("importance", 0),
            }
            
            # macro_shock: default 0 if Trading Economics unavailable
            ctx.macro_shock = fred_ctx.get("macro_shock", 0.0)
            
            # macro_risk from FRED
            ctx.macro_risk = fred_ctx.get("macro_risk", 0.5)
            
        except Exception as e:
            logger.warning(f"[MACRO_CROWD_ADAPTER] FRED error: {e}")
        
        return ctx
    
    # =========================================================================
    # CROWD COMPUTATION (TASK D2, D3, D4)
    # =========================================================================
    
    def _compute_crowd_context(self, symbol: str) -> CrowdContext:
        """
        Compute crowd context from Coinglass, LunarCrush, CryptoPanic.
        
        CRITICAL: These are CROWDING metrics, NOT direction signals.
        """
        ctx = CrowdContext()
        
        # === Coinglass: Derivatives crowding (D2) ===
        cg_crowding = 0.0
        cg_bias = 0.0
        
        if self._coinglass_feed:
            try:
                cg = self._coinglass_feed.get_crowd_metrics(symbol)
                cg_crowding = cg.get("crowding_score", 0.0)
                cg_bias = cg.get("funding_bias", 0.0)
            except Exception as e:
                logger.warning(f"[MACRO_CROWD_ADAPTER] Coinglass error: {e}")
        
        # === LunarCrush: Attention/crowding (D3) ===
        lc_attention = 0.5
        lc_bias = 0.0
        lc_shift = 0.0
        lc_extreme = False
        
        if self._lunarcrush_feed:
            try:
                lc = self._lunarcrush_feed.get_attention_metrics(symbol)
                lc_attention = lc.get("attention_pressure", 0.5)
                lc_bias = lc.get("crowding_bias", 0.0)
                lc_shift = lc.get("narrative_shift", 0.0)
                lc_extreme = lc.get("extreme_attention", False)
            except Exception as e:
                logger.warning(f"[MACRO_CROWD_ADAPTER] LunarCrush error: {e}")
        
        # === CryptoPanic: Panic score (D4) ===
        cp_panic = 0.0
        
        if self._cryptopanic_feed:
            try:
                cp = self._cryptopanic_feed.get_panic_metrics(symbol)
                cp_panic = cp.get("panic_score", 0.0)
            except Exception as e:
                logger.warning(f"[MACRO_CROWD_ADAPTER] CryptoPanic error: {e}")
        
        # === Aggregate ===
        
        # Attention pressure: LunarCrush primary
        ctx.attention_pressure = lc_attention
        
        # Crowding bias: Weighted combination
        # Coinglass funding_bias (derivatives) + LunarCrush (social)
        _crowding_raw = cg_bias * 0.6 + lc_bias * 0.4
        ctx.crowding_bias = float(np.clip(_crowding_raw, -1.0, 1.0)) if np.isfinite(_crowding_raw) else 0.0
        
        # Panic score: CryptoPanic primary
        ctx.panic_score = cp_panic
        
        # Narrative shift: LunarCrush primary
        ctx.narrative_shift = lc_shift
        
        # Extreme crowding: Any source extreme
        # High crowding score, high attention, or extreme flags
        ctx.extreme_crowding = (
            lc_extreme or
            cg_crowding >= 0.8 or
            lc_attention >= 0.9 or
            abs(ctx.crowding_bias) >= 0.8
        )
        
        return ctx
    
    # =========================================================================
    # SENTIMENT COMPUTATION (D5)
    # =========================================================================
    
    def _compute_sentiment_context(
        self,
        news_texts: Optional[List[str]],
        social_texts: Optional[List[str]],
    ) -> SentimentContext:
        """
        Compute deterministic text sentiment.
        
        Uses DeterministicSentimentEngine from v6.4.2.
        """
        ctx = SentimentContext()
        
        if not self._sentiment_engine:
            return ctx
        
        try:
            # Compute sentiment using v6.4.2 engine
            result = self._sentiment_engine.compute(
                news_texts=news_texts or [],
                social_texts=social_texts or [],
            )
            
            ctx.direction_score = result.direction_score
            ctx.strength = result.strength
            ctx.delta = result.delta
            ctx.volatility = result.volatility
            
        except Exception as e:
            logger.warning(f"[MACRO_CROWD_ADAPTER] Sentiment error: {e}")
        
        return ctx


# =============================================================================
# PROFIT-MAX INTEGRATION (TASK E3)
# =============================================================================

@dataclass
class MacroCrowdEffects:
    """
    Effects to apply in profit_max_adapter.
    
    CRITICAL CONSTRAINTS:
    - NO new hard trade gates
    - These are MODULATIONS only
    - Only false-breakout may veto
    """
    # Urgency modulation
    urgency_multiplier: float = 1.0  # [0.6, 1.0]
    
    # Size modulation
    size_multiplier: float = 1.0  # [0.85, 1.15]
    
    # Add-on permission
    allow_addon: bool = True
    
    # False breakout strictness
    false_breakout_strictness: int = 0  # 0, 1, 2
    
    # Execution hints
    prefer_limit: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "urgency_multiplier": self.urgency_multiplier,
            "size_multiplier": self.size_multiplier,
            "allow_addon": self.allow_addon,
            "false_breakout_strictness": self.false_breakout_strictness,
            "prefer_limit": self.prefer_limit,
        }


def compute_macro_crowd_effects(
    macro: Dict[str, Any],
    crowd: Dict[str, Any],
    sentiment: Dict[str, Any],
    price_direction: int = 0,  # -1, 0, 1
    system_bias: str = "short",  # [MACRO-FIX2] "short" | "long" | "neutral"
) -> MacroCrowdEffects:
    """
    [MACRO-FIX2] Compute profit-max effects with short-bias awareness.

    Pure conversion layer - no trade decisions, only modulations.
    system_bias="short" means: macro risk-off is FAVORABLE (crypto sells off).

    CRITICAL: NO NEW HARD GATES. ONLY FALSE BREAKOUT MAY VETO.
    """
    effects = MacroCrowdEffects()
    is_short_bias = (system_bias == "short")

    # === Macro Event Window ===
    event_window = macro.get("event_window", {})
    if event_window.get("active", False):
        macro_risk = macro.get("macro_risk", 0.5)

        if is_short_bias:
            # [MACRO-FIX2] High macro risk = favorable for shorts
            # Less urgency reduction when risk-off aligns with our bias
            effects.urgency_multiplier *= (1 - 0.2 * macro_risk)  # was 0.4
        else:
            effects.urgency_multiplier *= (1 - 0.4 * macro_risk)

        # Disable add-ons during high-risk event windows
        if macro_risk >= 0.6 and not is_short_bias:
            effects.allow_addon = False

        # Prefer limit orders during events
        effects.prefer_limit = True

    # === Macro Shock Scaling ===
    macro_shock = macro.get("macro_shock", 0.0)
    if abs(macro_shock) >= 0.4:
        if is_short_bias:
            # [MACRO-FIX2] Negative shock = market falling = aligned with shorts
            if macro_shock < -0.4:
                effects.size_multiplier *= 1.15  # Shock aligned with short bias
            elif macro_shock > 0.4:
                effects.size_multiplier *= 0.85  # Positive shock = against shorts
        else:
            # Original: check alignment with price_direction
            shock_direction = 1 if macro_shock > 0 else -1
            aligned = (shock_direction == price_direction) if price_direction != 0 else True
            if aligned:
                effects.size_multiplier *= 1.15
            else:
                effects.size_multiplier *= 0.85

    # === Extreme Crowding ===
    crowding_bias = crowd.get("crowding_bias", 0.0)
    if crowd.get("extreme_crowding", False):
        if is_short_bias:
            # [MACRO-FIX2] Direction matters: long crowding = contrarian short edge
            if crowding_bias > 0.3:
                # Longs are crowded -> good for shorts (contrarian)
                effects.urgency_multiplier *= 0.95  # Mild caution only
                effects.false_breakout_strictness += 1
            elif crowding_bias < -0.3:
                # Shorts are crowded -> squeeze risk for us
                effects.allow_addon = False
                effects.urgency_multiplier *= 0.75
                effects.false_breakout_strictness += 1
                effects.prefer_limit = True
            else:
                effects.urgency_multiplier *= 0.8
                effects.false_breakout_strictness += 1
        else:
            effects.allow_addon = False
            effects.urgency_multiplier *= 0.8
            effects.false_breakout_strictness += 1
            effects.prefer_limit = True

    # Additional crowding modulation
    if abs(crowding_bias) >= 0.6:
        effects.urgency_multiplier *= 0.9

    # === Panic Consideration ===
    panic_score = crowd.get("panic_score", 0.0)
    if is_short_bias:
        # [MACRO-FIX2] Panic = short confirmation at moderate levels
        if panic_score >= 0.9:
            # Extreme panic -> reversal risk, reduce urgency
            effects.urgency_multiplier *= 0.8
            effects.false_breakout_strictness += 1
        elif panic_score >= 0.7:
            # Moderate panic -> confirms short thesis, slight boost
            effects.size_multiplier *= 1.05
        elif panic_score >= 0.5:
            # Mild panic -> neutral-to-positive for shorts
            pass  # No adjustment
    else:
        if panic_score >= 0.7:
            effects.urgency_multiplier *= 0.85
            effects.false_breakout_strictness += 1

    # === Sentiment Direction ===
    direction_score = sentiment.get("direction_score", 0.0)
    if is_short_bias:
        # [MACRO-FIX2] Bullish crowd = contrarian short signal
        if direction_score > 0.3:
            # Crowd is bullish -> contrarian short edge, boost
            effects.size_multiplier *= (1 + 0.05 * abs(direction_score))
        elif direction_score < -0.3:
            # Crowd is bearish -> our trade is crowded, reduce
            effects.size_multiplier *= (1 - 0.10 * abs(direction_score))
    else:
        if price_direction != 0:
            alignment = direction_score * price_direction
            if alignment > 0.3:
                effects.size_multiplier *= (1 - 0.15 * abs(direction_score))
            elif alignment < -0.3:
                effects.size_multiplier *= (1 + 0.05 * abs(direction_score))

    # === Enforce Bounds ===
    effects.urgency_multiplier = max(0.6, min(1.0, effects.urgency_multiplier))
    effects.size_multiplier = max(0.85, min(1.20, effects.size_multiplier))  # [MACRO-FIX2] cap 1.15->1.20
    effects.false_breakout_strictness = min(2, effects.false_breakout_strictness)

    return effects


# =============================================================================
# SINGLETON
# =============================================================================

_adapter_instance: Optional[MacroCrowdAdapter] = None


def get_macro_crowd_adapter(
    fred_feed=None,
    coinglass_feed=None,
    lunarcrush_feed=None,
    cryptopanic_feed=None,
    sentiment_engine=None,
) -> MacroCrowdAdapter:
    """Get or create adapter singleton."""
    global _adapter_instance
    if _adapter_instance is None:
        _adapter_instance = MacroCrowdAdapter(
            fred_feed=fred_feed,
            coinglass_feed=coinglass_feed,
            lunarcrush_feed=lunarcrush_feed,
            cryptopanic_feed=cryptopanic_feed,
            sentiment_engine=sentiment_engine,
        )
    elif (
        getattr(_adapter_instance, "_fred_feed", None) is not fred_feed
        or getattr(_adapter_instance, "_coinglass_feed", None) is not coinglass_feed
        or getattr(_adapter_instance, "_lunarcrush_feed", None) is not lunarcrush_feed
        or getattr(_adapter_instance, "_cryptopanic_feed", None) is not cryptopanic_feed
        or getattr(_adapter_instance, "_sentiment_engine", None) is not sentiment_engine
    ):
        logger.info("[MACRO_CROWD] Constructor inputs changed; refreshing singleton instance")
        _adapter_instance = MacroCrowdAdapter(
            fred_feed=fred_feed,
            coinglass_feed=coinglass_feed,
            lunarcrush_feed=lunarcrush_feed,
            cryptopanic_feed=cryptopanic_feed,
            sentiment_engine=sentiment_engine,
        )
    return _adapter_instance


def reset_macro_crowd_adapter():
    """Reset singleton."""
    global _adapter_instance
    _adapter_instance = None


# =============================================================================
# TESTS
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Macro/Crowd Adapter Test")
    print("=" * 70)
    
    # Test with no feeds (defaults)
    adapter = MacroCrowdAdapter()
    output = adapter.compute("BTC")
    
    print("\nDefault Output:")
    print(f"  Macro: {output.macro.to_dict()}")
    print(f"  Crowd: {output.crowd.to_dict()}")
    print(f"  Sentiment: {output.sentiment.to_dict()}")
    
    # Test effects computation
    print("\n--- Testing Profit-Max Effects ---")
    
    # Test 1: Normal conditions
    effects = compute_macro_crowd_effects(
        macro={"event_window": {"active": False}, "macro_risk": 0.3, "macro_shock": 0.0},
        crowd={"extreme_crowding": False, "crowding_bias": 0.2, "panic_score": 0.1},
        sentiment={"direction_score": 0.3},
        price_direction=1,
    )
    print(f"\nNormal: {effects.to_dict()}")
    
    # Test 2: Event window
    effects = compute_macro_crowd_effects(
        macro={"event_window": {"active": True}, "macro_risk": 0.7, "macro_shock": 0.0},
        crowd={"extreme_crowding": False, "crowding_bias": 0.0, "panic_score": 0.0},
        sentiment={"direction_score": 0.0},
        price_direction=0,
    )
    print(f"Event window (risk=0.7): {effects.to_dict()}")
    assert not effects.allow_addon, "Add-on should be disabled"
    
    # Test 3: Extreme crowding
    effects = compute_macro_crowd_effects(
        macro={"event_window": {"active": False}, "macro_risk": 0.3, "macro_shock": 0.0},
        crowd={"extreme_crowding": True, "crowding_bias": 0.9, "panic_score": 0.3},
        sentiment={"direction_score": 0.0},
        price_direction=0,
    )
    print(f"Extreme crowding: {effects.to_dict()}")
    assert not effects.allow_addon, "Add-on should be disabled"
    assert effects.false_breakout_strictness >= 1, "Strictness should increase"
    
    # Test 4: Macro shock aligned
    effects = compute_macro_crowd_effects(
        macro={"event_window": {"active": False}, "macro_risk": 0.3, "macro_shock": 0.5},
        crowd={"extreme_crowding": False, "crowding_bias": 0.0, "panic_score": 0.0},
        sentiment={"direction_score": 0.0},
        price_direction=1,
    )
    print(f"Macro shock aligned: {effects.to_dict()}")
    assert effects.size_multiplier > 1.0, "Size should increase when aligned"
    
    print("\nON All tests passed!")
