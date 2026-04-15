"""
VPIN Calculator - Batch computation from Kraken REST trades.

Called once per 4H tick per asset. Fetches recent trades from Kraken REST API,
classifies buy/sell using tick rule, computes VPIN via volume bucketing.

This is a simplified but accurate VPIN implementation suitable for 4H frequency.
Full tick-stream VPIN (GPUVPINCalculator) can replace this for live deployment.
"""

import logging
import time
import numpy as np
from typing import Dict, Optional, List
from collections import deque

logger = logging.getLogger(__name__)


class BatchVPINCalculator:
    """
    Compute VPIN from a batch of recent trades.

    Algorithm:
    1. Classify each trade as buy/sell (tick rule)
    2. Bucket trades by volume ($10K buckets for $10K account)
    3. VPIN = mean(|buy_vol - sell_vol| / bucket_vol) over last N buckets
    """

    def __init__(
        self,
        bucket_size_usd: float = 10_000.0,
        n_buckets: int = 30,
    ):
        self.bucket_size_usd = bucket_size_usd
        self.n_buckets = n_buckets
        self._history: Dict[str, deque] = {}
        self._last_compute: Dict[str, float] = {}
        logger.info(
            f"BatchVPINCalculator initialized: "
            f"bucket=${bucket_size_usd:.0f}, n_buckets={n_buckets}"
        )

    def compute_from_trades(
        self,
        trades: List[Dict],
        asset: str,
    ) -> float:
        """
        Compute VPIN from a list of trade records.

        Args:
            trades: List of dicts with keys: price, volume (+ optional timestamp).
                    Sorted by timestamp ascending.
            asset: Asset name (for history tracking)

        Returns:
            VPIN value [0.0, 1.0], or -1.0 if insufficient data.
        """
        if not trades or len(trades) < 20:
            logger.debug(f"[VPIN] {asset}: insufficient trades ({len(trades) if trades else 0})")
            return -1.0

        classified = self._classify_trades(trades)
        buckets = self._volume_bucket(classified)

        if len(buckets) < 5:
            logger.debug(f"[VPIN] {asset}: insufficient buckets ({len(buckets)})")
            return -1.0

        recent = buckets[-self.n_buckets:]
        imbalances = []
        for b_buy, b_sell, b_total in recent:
            if b_total > 0:
                imbalances.append(abs(b_buy - b_sell) / b_total)

        if not imbalances:
            return -1.0

        vpin = float(np.mean(imbalances))
        vpin = max(0.0, min(1.0, vpin))

        if asset not in self._history:
            self._history[asset] = deque(maxlen=100)
        self._history[asset].append(vpin)
        self._last_compute[asset] = time.time()

        logger.info(
            f"[VPIN] {asset}: computed={vpin:.4f} "
            f"from {len(trades)} trades, {len(buckets)} buckets"
        )
        return vpin

    def _classify_trades(self, trades: List[Dict]) -> List[Dict]:
        """Classify trades as buy/sell using tick rule."""
        result = []
        prev_price = trades[0].get("price", 0)

        for t in trades:
            price = t.get("price", 0)
            volume_usd = t.get("volume", 0) * price

            if price > prev_price:
                result.append({"buy": volume_usd, "sell": 0.0})
            elif price < prev_price:
                result.append({"buy": 0.0, "sell": volume_usd})
            else:
                result.append({"buy": volume_usd * 0.5, "sell": volume_usd * 0.5})

            prev_price = price

        return result

    def _volume_bucket(self, classified: List[Dict]) -> List[tuple]:
        """Aggregate classified trades into volume buckets."""
        buckets = []
        cur_buy = 0.0
        cur_sell = 0.0
        cur_total = 0.0

        for t in classified:
            cur_buy += t["buy"]
            cur_sell += t["sell"]
            cur_total += t["buy"] + t["sell"]

            if cur_total >= self.bucket_size_usd:
                buckets.append((cur_buy, cur_sell, cur_total))
                cur_buy = 0.0
                cur_sell = 0.0
                cur_total = 0.0

        if cur_total > self.bucket_size_usd * 0.3:
            buckets.append((cur_buy, cur_sell, cur_total))

        return buckets

    def get_last_vpin(self, asset: str) -> Optional[float]:
        """Get most recent VPIN for asset, or None."""
        if asset in self._history and self._history[asset]:
            return self._history[asset][-1]
        return None

    # [DEAD-CLEANUP 2026-04-15] Removed get_vpin_zscore() — was a v3.1 engine
    # design field with 0 callers in production code (only archive/engine_legacy
    # used it). Current engines use raw `vpin` + `composite_toxicity` instead.
