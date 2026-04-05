"""
High Position + Low Volume Filter (纪律⑦).

Detects selling exhaustion: price near highs but volume declining.
When detected, reduces short exposure by 50% to avoid fighting
a potential reversal (exhausted sellers -> price may bounce).

Also detects buying exhaustion at lows for long exposure reduction.
"""
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class HPLVFilterResult:
    """Result from high-position low-volume filter."""
    triggered: bool = False
    pattern: str = ""              # "SELLING_EXHAUSTION" or "BUYING_EXHAUSTION"
    price_percentile: float = 0.0  # Where current price sits in recent range
    volume_ratio: float = 1.0      # Current volume / average volume
    exposure_multiplier: float = 1.0  # Applied to opposing direction exposure


class HighPositionLowVolumeFilter:
    """Reduce directional exposure when price is extreme but volume is drying up.

    纪律⑦: 高位放量后缩量 = 卖盘衰竭 -> 减少空头暴露
    Mirror: 低位放量后缩量 = 买盘衰竭 -> 减少多头暴露
    """

    HIGH_PERCENTILE = 0.90       # Price >= 90th percentile = "high position"
    LOW_PERCENTILE = 0.10        # Price <= 10th percentile = "low position"
    LOW_VOLUME_RATIO = 0.6       # Current volume < 60% average = "low volume"
    EXHAUSTION_MULTIPLIER = 0.5  # Reduce opposing exposure to 50%
    MIN_LOOKBACK = 20            # Minimum bars for percentile calculation

    def check(
        self,
        current_price: float,
        price_history: np.ndarray,
        volume_ratio: float,
        direction: int,
        current_exposure: float,
    ) -> HPLVFilterResult:
        """Check for exhaustion patterns and return exposure adjustment.

        Args:
            current_price: Current asset price.
            price_history: Array of recent close prices (oldest first).
            volume_ratio: current_volume / avg_volume (20-bar).
            direction: Intent direction (+1 long, -1 short).
            current_exposure: Current exposure fraction for this asset.

        Returns:
            HPLVFilterResult with potential exposure reduction.
        """
        if len(price_history) < self.MIN_LOOKBACK:
            return HPLVFilterResult()

        price_pctile = (
            np.sum(price_history <= current_price) / len(price_history)
        )
        volume_declining = volume_ratio < self.LOW_VOLUME_RATIO

        # Selling exhaustion: price near highs + volume drying up -> bad for shorts
        if (
            price_pctile >= self.HIGH_PERCENTILE
            and volume_declining
            and direction < 0  # short
        ):
            adjusted = current_exposure * self.EXHAUSTION_MULTIPLIER
            logger.info(
                f"[DISC-7] SELLING_EXHAUSTION: price_pctile={price_pctile:.2f} "
                f"vol_ratio={volume_ratio:.2f} -> short exposure "
                f"{current_exposure:.4f}->{adjusted:.4f}"
            )
            return HPLVFilterResult(
                triggered=True,
                pattern="SELLING_EXHAUSTION",
                price_percentile=price_pctile,
                volume_ratio=volume_ratio,
                exposure_multiplier=self.EXHAUSTION_MULTIPLIER,
            )

        # Buying exhaustion: price near lows + volume drying up -> bad for longs
        if (
            price_pctile <= self.LOW_PERCENTILE
            and volume_declining
            and direction > 0  # long
        ):
            adjusted = current_exposure * self.EXHAUSTION_MULTIPLIER
            logger.info(
                f"[DISC-7] BUYING_EXHAUSTION: price_pctile={price_pctile:.2f} "
                f"vol_ratio={volume_ratio:.2f} -> long exposure "
                f"{current_exposure:.4f}->{adjusted:.4f}"
            )
            return HPLVFilterResult(
                triggered=True,
                pattern="BUYING_EXHAUSTION",
                price_percentile=price_pctile,
                volume_ratio=volume_ratio,
                exposure_multiplier=self.EXHAUSTION_MULTIPLIER,
            )

        return HPLVFilterResult(
            price_percentile=price_pctile,
            volume_ratio=volume_ratio,
        )


# Module-level singleton
_filter: Optional[HighPositionLowVolumeFilter] = None


def get_hplv_filter() -> HighPositionLowVolumeFilter:
    """Get or create the singleton HighPositionLowVolumeFilter."""
    global _filter
    if _filter is None:
        _filter = HighPositionLowVolumeFilter()
    return _filter
