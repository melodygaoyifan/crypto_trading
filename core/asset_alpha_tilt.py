"""
Per-Asset Alpha Tilt (SOTA G6-lite)

Adjusts per-asset exposure multiplier based on rolling 30-day Sortino ratio.
Assets with better recent risk-adjusted performance get more exposure;
underperformers get reduced exposure.

Multiplier range: 0.5 ~ 1.5 (never zeros out, never doubles).
Update frequency: every 6 ticks (~24h at 4H bars).
"""
import logging
import numpy as np
from typing import Dict, Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class AssetAlphaTilt:
    MIN_MULTIPLIER = 0.3   # [PRE-LAUNCH] was 0.5. More aggressive reduction for losers
    MAX_MULTIPLIER = 1.5

    TILT_TABLE = [
        # (sortino_threshold, multiplier) — [PRE-LAUNCH] steeper penalty curve
        (2.0,  1.5),
        (1.0,  1.3),
        (0.0,  1.0),
        (-0.3, 0.6),   # was -0.5 → 0.7. Faster reduction for underperformers
        (-999, 0.3),   # was 0.5. Severe underperformers get 70% size reduction
    ]

    SMOOTHING_ALPHA = 0.3
    MIN_TRADES_FOR_TILT = 5

    def __init__(self):
        self._current_multipliers: Dict[str, float] = {
            'BTC': 1.0, 'ETH': 1.0, 'SOL': 1.0
        }
        self._last_update: Optional[datetime] = None

    def update(self, asset_trade_pnls: Dict[str, list]) -> Dict[str, float]:
        """
        Recalculate tilt multipliers from recent trade PnLs.

        Args:
            asset_trade_pnls: {'BTC': [pnl1, ...], 'ETH': [...], 'SOL': [...]}

        Returns:
            {'BTC': 1.0, 'ETH': 0.7, 'SOL': 1.3}
        """
        new_mults = {}

        for asset in ['BTC', 'ETH', 'SOL']:
            pnls = asset_trade_pnls.get(asset, [])

            if len(pnls) < self.MIN_TRADES_FOR_TILT:
                new_mults[asset] = 1.0
                continue

            sortino = self._compute_sortino(pnls)
            raw_mult = self._sortino_to_multiplier(sortino)

            prev = self._current_multipliers.get(asset, 1.0)
            smoothed = self.SMOOTHING_ALPHA * raw_mult + (1 - self.SMOOTHING_ALPHA) * prev
            smoothed = np.clip(smoothed, self.MIN_MULTIPLIER, self.MAX_MULTIPLIER)

            new_mults[asset] = round(float(smoothed), 2)

            if abs(new_mults[asset] - prev) > 0.05:
                logger.info(
                    f"[ALPHA-TILT] {asset}: Sortino={sortino:.2f} -> "
                    f"mult {prev:.2f} -> {new_mults[asset]:.2f}"
                )

        self._current_multipliers = new_mults
        self._last_update = datetime.now(timezone.utc)
        return new_mults

    def get_multiplier(self, asset: str) -> float:
        """Get current multiplier for asset. Returns 1.0 if not initialized."""
        return self._current_multipliers.get(asset, 1.0)

    def _compute_sortino(self, pnls: list) -> float:
        """Sortino ratio: mean / downside_std."""
        arr = np.array(pnls)
        mean_ret = np.mean(arr)
        downside = arr[arr < 0]
        if len(downside) < 2:
            return 3.0 if mean_ret > 0 else 0.0
        downside_std = np.std(downside)
        if downside_std < 1e-10:
            return 3.0 if mean_ret > 0 else 0.0
        return float(mean_ret / downside_std)

    def _sortino_to_multiplier(self, sortino: float) -> float:
        for threshold, mult in self.TILT_TABLE:
            if sortino >= threshold:
                return mult
        return self.MIN_MULTIPLIER

    def to_dict(self) -> Dict:
        return {
            "multipliers": dict(self._current_multipliers),
            "last_update": self._last_update.isoformat() if self._last_update else None,
        }

    def from_dict(self, data: Dict):
        self._current_multipliers = data.get("multipliers", {'BTC': 1.0, 'ETH': 1.0, 'SOL': 1.0})
        if data.get("last_update"):
            self._last_update = datetime.fromisoformat(data["last_update"])