"""
================================================================================
HMATS v6.7 - Vol Convexity Strategy (CONVEX tier)
================================================================================

Captures volatility expansion after compression (squeeze -> breakout).

When VolatilityAlphaAgent signals high compression intensity (>0.7),
generates OCO-style dual stop orders:
    - Long stop above BB upper (breakout up)
    - Short stop below BB lower (breakout down)

Only ONE side fills; the other is cancelled (cancel_other_on_fill).

Integration:
    - Consumes: VolatilityIntent from VolatilityAlphaAgent
    - Consumes: market_data (bb_upper, bb_lower, close, atr)
    - Output: DualStopOrder -> injected into agent_signals as vol_convex_*
    - Tier: CONVEX (15% budget, max 3x leverage)
    - Authority: ADVISE (signal only, no direct execution)

Safety:
    - Does NOT bypass: veto logic, risk checks, trade gate, stop logic
    - Does NOT trigger entries directly - signal consumed by fusion
    - Size capped at 5% of portfolio (configurable)
    - Minimum spread requirement: breakout range > 2× ATR
================================================================================
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class DualStopOrder:
    """OCO-style dual breakout order signal."""
    long_stop_price: float      # Buy stop above resistance
    short_stop_price: float     # Sell stop below support
    size_pct: float             # Position size as % of portfolio
    cancel_other_on_fill: bool  # OCO behavior
    spread_bps: float = 0.0    # Distance between stops in bps
    confidence: float = 0.0    # Signal confidence 0-1
    rationale: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "long_stop_price": self.long_stop_price,
            "short_stop_price": self.short_stop_price,
            "size_pct": self.size_pct,
            "cancel_other_on_fill": self.cancel_other_on_fill,
            "spread_bps": self.spread_bps,
            "confidence": self.confidence,
            "rationale": self.rationale,
        }


@dataclass
class VolConvexityConfig:
    """Vol convexity strategy configuration."""
    enabled: bool = True

    # VolAlpha intensity threshold to activate
    min_intensity: float = 0.7

    # Breakout offset from BB bands (multiplier)
    # long_stop = bb_upper * (1 + offset)
    # short_stop = bb_lower * (1 - offset)
    breakout_offset: float = 0.002  # 0.2% beyond BB

    # Position sizing (% of portfolio)
    base_size_pct: float = 0.05    # 5%
    max_size_pct: float = 0.08     # 8% max on extreme intensity

    # Minimum spread: stops must be > min_atr_mult × ATR apart
    min_atr_mult: float = 2.0

    # Confidence scaling
    base_confidence: float = 0.5
    intensity_bonus: float = 0.3   # extra conf per 0.1 intensity above threshold


# =============================================================================
# STRATEGY
# =============================================================================

class VolConvexityStrategy:
    """
    Volatility convexity strategy - breakout straddle on compression.

    Activated when VolatilityAlphaAgent detects high compression (intensity > 0.7).
    Places dual stop orders above/below Bollinger Bands.
    """

    def __init__(self, config: Optional[VolConvexityConfig] = None):
        self.config = config or VolConvexityConfig()
        self._signal_count = 0
        self._last_signal: Optional[DualStopOrder] = None
        logger.info(
            f"[VOL_CONVEXITY] Initialized: min_intensity={self.config.min_intensity}, "
            f"offset={self.config.breakout_offset}, size={self.config.base_size_pct:.0%}"
        )

    def evaluate(
        self,
        vol_intensity: float,
        market_data: Dict[str, Any],
    ) -> Optional[DualStopOrder]:
        """
        Evaluate vol convexity signal.

        Args:
            vol_intensity: VolAlpha intensity (0-1, from agent_signals)
            market_data: Dict with bb_upper, bb_lower, close, atr

        Returns:
            DualStopOrder if conditions met, None otherwise
        """
        if not self.config.enabled:
            return None

        if vol_intensity < self.config.min_intensity:
            return None

        # Extract price levels
        bb_upper = market_data.get("bb_upper", 0.0)
        bb_lower = market_data.get("bb_lower", 0.0)
        close = market_data.get("close", market_data.get("price", 0.0))
        atr = market_data.get("atr", 0.0)

        if not bb_upper or not bb_lower or not close or close <= 0:
            return None

        # Compute stop prices
        long_stop = bb_upper * (1 + self.config.breakout_offset)
        short_stop = bb_lower * (1 - self.config.breakout_offset)

        # Safety: stops must be meaningful distance apart
        spread = long_stop - short_stop
        if atr > 0 and spread < atr * self.config.min_atr_mult:
            return None

        spread_bps = (spread / close) * 10000 if close > 0 else 0

        # Confidence scales with intensity above threshold
        excess = vol_intensity - self.config.min_intensity
        confidence = min(
            1.0,
            self.config.base_confidence + excess * self.config.intensity_bonus * 10
        )

        # Size scales modestly with intensity
        size_pct = min(
            self.config.max_size_pct,
            self.config.base_size_pct + excess * 0.1
        )

        rationale = [
            f"vol_intensity={vol_intensity:.2f}",
            f"bb_spread={spread_bps:.0f}bps",
            f"long_stop={long_stop:.2f}",
            f"short_stop={short_stop:.2f}",
        ]

        signal = DualStopOrder(
            long_stop_price=long_stop,
            short_stop_price=short_stop,
            size_pct=round(size_pct, 4),
            cancel_other_on_fill=True,
            spread_bps=round(spread_bps, 1),
            confidence=round(confidence, 3),
            rationale=rationale,
        )

        self._signal_count += 1
        self._last_signal = signal

        logger.info(
            f"[VOL_CONVEXITY] Signal #{self._signal_count}: "
            f"long={long_stop:.2f} short={short_stop:.2f} "
            f"size={size_pct:.1%} conf={confidence:.2f} "
            f"spread={spread_bps:.0f}bps intensity={vol_intensity:.2f}"
        )

        return signal

    def get_status(self) -> Dict[str, Any]:
        return {
            "enabled": self.config.enabled,
            "signal_count": self._signal_count,
            "last_signal": self._last_signal.to_dict() if self._last_signal else None,
        }


# =============================================================================
# SINGLETON
# =============================================================================

_instance: Optional[VolConvexityStrategy] = None


def get_vol_convexity() -> VolConvexityStrategy:
    global _instance
    if _instance is None:
        _instance = VolConvexityStrategy()
    return _instance