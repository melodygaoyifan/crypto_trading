"""
[SQUEEZE-DETECTOR] Short Squeeze Detector - CoinGlass Derivatives Data
=======================================================================
Authority Level: ADVISE (no veto, no trigger - feeds Authority Fusion)

Detects conditions where SHORT positions are at risk of being squeezed:
  1. Short liquidation spike: shorts being liquidated at >3x average rate
  2. OI collapse: rapid OI drop (>10% in 4h) = forced covering
  3. Funding rate flip: funding going positive -> shorts paying longs
  4. Price acceleration: rapid price rise with volume confirmation
  5. Liquidation imbalance: >60% of liquidations are shorts
  6. Taker buy dominance: aggressive buying pressure

Output:
  squeeze_score: 0.0-1.0 composite score
  squeeze_level: SAFE / WATCH / WARNING / CRITICAL
  recommended_action: NONE / REDUCE_30 / REDUCE_50 / FLATTEN

Data Source: market_data keys populated by main.py from CoinGlass feed
Frequency: Every 4H tick
Fail-safe: All errors -> neutral output (squeeze_score=0, level=SAFE)
"""

import logging
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class SqueezeDetectorAgent:
    """
    [SQUEEZE-DETECTOR] Detects short squeeze risk from derivatives data.

    Authority: ADVISE only - feeds into Authority Fusion as a risk signal.
    Uses market_data keys already populated by main.py from CoinGlass.
    """

    # Squeeze level thresholds
    LEVEL_WATCH = 0.3
    LEVEL_WARNING = 0.55
    LEVEL_CRITICAL = 0.75

    SUPPORTED_ASSETS = ("BTC", "ETH", "SOL")

    def __init__(
        self,
        short_liq_spike_ratio: float = 3.0,
        oi_collapse_threshold: float = 0.10,
        funding_flip_threshold: float = 0.0001,
        price_accel_threshold: float = 0.03,
        lookback_bars: int = 42,  # 7 days at 4H
    ):
        self._short_liq_spike_ratio = short_liq_spike_ratio
        self._oi_collapse_threshold = oi_collapse_threshold
        self._funding_flip_threshold = funding_flip_threshold
        self._price_accel_threshold = price_accel_threshold

        # History buffers per asset
        self._short_liq_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=lookback_bars)
        )
        self._oi_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=lookback_bars)
        )
        self._funding_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=lookback_bars)
        )
        self._price_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=lookback_bars)
        )

        # Cache
        self._last_result: Dict[str, Dict] = {}
        self._last_compute_time: float = 0
        self._errors: int = 0

        logger.info("[SQUEEZE-DETECTOR] Initialized")

    def generate_signal(self, asset: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        [SQUEEZE-DETECTOR] Compute squeeze risk from market_data.

        Returns neutral values on any error - never blocks tick loop.
        """
        neutral = {
            "squeeze_score": 0.0,
            "squeeze_level": "SAFE",
            "recommended_action": "NONE",
            "components": {},
            "confidence": 0.0,
            "source": "squeeze_detector",
        }

        if asset not in self.SUPPORTED_ASSETS:
            return neutral

        try:
            return self._compute(asset, market_data)
        except Exception as e:
            self._errors += 1
            logger.debug(f"[SQUEEZE-DETECTOR] {asset} error: {e}")
            cached = self._last_result.get(asset)
            return cached if cached else neutral

    def _compute(self, asset: str, md: Dict[str, Any]) -> Dict[str, Any]:
        """Compute squeeze score from 6 weighted components."""

        # Extract data from market_data (populated by main.py from CoinGlass)
        short_liq = float(md.get("short_liquidations_24h", 0))
        long_liq = float(md.get("long_liquidations_24h", 0))
        total_liq = float(md.get("total_liquidations_24h", 0))
        oi_usd = float(md.get("open_interest", 0))
        oi_change_1h = float(md.get("oi_change_1h_pct", 0))
        oi_change_24h = float(md.get("oi_change_24h_pct", 0))
        funding_rate = float(md.get("funding_rate", 0))
        price = float(md.get("price", md.get("close", 0)))
        volume_ratio = float(md.get("volume_ratio", 1.0))
        price_change_4h = float(md.get("price_change_4h_pct", 0))

        # Update history buffers
        if short_liq > 0:
            self._short_liq_history[asset].append(short_liq)
        if oi_usd > 0:
            self._oi_history[asset].append(oi_usd)
        if funding_rate != 0:
            self._funding_history[asset].append(funding_rate)
        if price > 0:
            self._price_history[asset].append(price)

        # Component scores (each 0.0-1.0)
        components = {}

        # 1. Short liquidation spike (weight=0.25)
        s1 = self._score_short_liq_spike(asset, short_liq)
        components["short_liq_spike"] = round(s1, 3)

        # 2. OI collapse (weight=0.20)
        s2 = self._score_oi_collapse(oi_change_1h, oi_change_24h)
        components["oi_collapse"] = round(s2, 3)

        # 3. Funding rate flip (weight=0.15)
        s3 = self._score_funding_flip(asset, funding_rate)
        components["funding_flip"] = round(s3, 3)

        # 4. Price acceleration (weight=0.15)
        s4 = self._score_price_accel(price_change_4h, volume_ratio)
        components["price_acceleration"] = round(s4, 3)

        # 5. Liquidation imbalance - short-heavy (weight=0.15)
        s5 = self._score_liq_imbalance(short_liq, long_liq, total_liq)
        components["liq_imbalance"] = round(s5, 3)

        # 6. Taker buy dominance (weight=0.10)
        s6 = self._score_taker_dominance(md)
        components["taker_buy_dominance"] = round(s6, 3)

        # Weighted composite
        weights = [0.25, 0.20, 0.15, 0.15, 0.15, 0.10]
        scores = [s1, s2, s3, s4, s5, s6]
        squeeze_score = float(np.clip(sum(w * s for w, s in zip(weights, scores)), 0.0, 1.0))

        # Determine level
        if squeeze_score >= self.LEVEL_CRITICAL:
            level = "CRITICAL"
            action = "FLATTEN"
        elif squeeze_score >= self.LEVEL_WARNING:
            level = "WARNING"
            action = "REDUCE_50"
        elif squeeze_score >= self.LEVEL_WATCH:
            level = "WATCH"
            action = "REDUCE_30"
        else:
            level = "SAFE"
            action = "NONE"

        # Confidence based on data availability
        data_points = sum([
            len(self._short_liq_history[asset]) >= 3,
            len(self._oi_history[asset]) >= 3,
            len(self._funding_history[asset]) >= 3,
            len(self._price_history[asset]) >= 3,
            total_liq > 0,
            funding_rate != 0,
        ])
        confidence = min(0.85, data_points * 0.15)

        result = {
            "squeeze_score": round(squeeze_score, 3),
            "squeeze_level": level,
            "recommended_action": action,
            "components": components,
            "confidence": round(confidence, 3),
            "source": "squeeze_detector",
        }

        self._last_result[asset] = result
        self._last_compute_time = time.time()
        return result

    def _score_short_liq_spike(self, asset: str, short_liq: float) -> float:
        """Short liquidations spiking = shorts being squeezed."""
        history = self._short_liq_history[asset]
        if len(history) < 3 or short_liq <= 0:
            return 0.0
        avg = float(np.mean(list(history)))
        if avg <= 0:
            return 0.0
        ratio = short_liq / avg
        if ratio > self._short_liq_spike_ratio:
            return min(1.0, (ratio - 1.0) / (self._short_liq_spike_ratio - 1.0))
        return 0.0

    def _score_oi_collapse(self, oi_change_1h: float, oi_change_24h: float) -> float:
        """Rapid OI drop = forced position closing (covering)."""
        score = 0.0
        # 1h OI drop > threshold (negative = dropping)
        if oi_change_1h < -self._oi_collapse_threshold:
            score = min(1.0, abs(oi_change_1h) / (self._oi_collapse_threshold * 2))
        # 24h OI drop amplifies
        if oi_change_24h < -0.05:
            score = min(1.0, score + 0.3)
        return score

    def _score_funding_flip(self, asset: str, funding_rate: float) -> float:
        """Funding going positive = shorts paying longs = squeeze pressure."""
        history = self._funding_history[asset]
        if len(history) < 2:
            return 0.0

        prev_rate = float(history[-2]) if len(history) >= 2 else 0.0

        # Flip from negative to positive = squeeze signal
        if prev_rate < 0 and funding_rate > self._funding_flip_threshold:
            return min(1.0, funding_rate / 0.001)  # Normalize: 0.1% = full signal

        # Already very positive = shorts under pressure
        if funding_rate > 0.0005:  # 0.05%/8h
            return min(0.7, funding_rate / 0.001)

        return 0.0

    def _score_price_accel(self, price_change_4h: float, volume_ratio: float) -> float:
        """Rapid price rise with volume = squeeze dynamics."""
        if price_change_4h <= self._price_accel_threshold:
            return 0.0

        # Price up > 3% in 4h
        price_score = min(1.0, (price_change_4h - self._price_accel_threshold) / 0.05)

        # Volume confirmation amplifies
        vol_mult = 1.0
        if volume_ratio > 2.0:
            vol_mult = 1.3
        elif volume_ratio > 1.5:
            vol_mult = 1.15

        return min(1.0, price_score * vol_mult)

    def _score_liq_imbalance(
        self, short_liq: float, long_liq: float, total_liq: float,
    ) -> float:
        """Short-heavy liquidation imbalance = shorts getting squeezed."""
        if total_liq <= 0:
            return 0.0
        short_pct = short_liq / total_liq
        if short_pct > 0.6:
            return min(1.0, (short_pct - 0.4) / 0.4)  # 60%->0.5, 80%->1.0
        return 0.0

    def _score_taker_dominance(self, md: Dict[str, Any]) -> float:
        """Taker buy/sell ratio from market_data if available."""
        taker_buy = float(md.get("taker_buy_volume", 0))
        taker_sell = float(md.get("taker_sell_volume", 0))
        if taker_buy <= 0 or taker_sell <= 0:
            return 0.0
        ratio = taker_buy / taker_sell
        if ratio > 1.5:
            return min(1.0, (ratio - 1.0) / 1.0)  # 1.5->0.5, 2.0->1.0
        return 0.0

    def get_status(self) -> Dict[str, Any]:
        """Return agent status for diagnostics."""
        return {
            "errors": self._errors,
            "last_compute_time": self._last_compute_time,
            "history_depths": {
                asset: len(self._short_liq_history[asset])
                for asset in self.SUPPORTED_ASSETS
                if asset in self._short_liq_history
            },
            "last_results": {
                asset: {
                    "squeeze_score": r.get("squeeze_score", 0),
                    "squeeze_level": r.get("squeeze_level", "SAFE"),
                }
                for asset, r in self._last_result.items()
            },
        }


# =============================================================================
# SINGLETON
# =============================================================================

_instance: Optional[SqueezeDetectorAgent] = None


def get_squeeze_detector() -> SqueezeDetectorAgent:
    """Get or create singleton."""
    global _instance
    if _instance is None:
        _instance = SqueezeDetectorAgent()
    return _instance


def reset_squeeze_detector():
    """Reset singleton."""
    global _instance
    _instance = None