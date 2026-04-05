"""
[STABLECOIN-MONITOR] Stablecoin Flow Monitor - CoinGlass v3 API
================================================================
Authority Level: ADVISE (no veto, no trigger)

Tracks stablecoin (USDT) total supply changes as a macro liquidity signal:
  - Rising stablecoin supply = new capital entering crypto = headwind for shorts
  - Falling stablecoin supply = capital exiting crypto = tailwind for shorts

Output:
  supply_change_7d_pct: 7-day rolling % change in USDT market cap
  short_size_multiplier: 0.7-1.1 (>1.0 = favorable for shorts, <1.0 = reduce)

Data Source: CoinGlass v3 API (/api/stablecoin/charts)
Frequency: Every 4H tick (supply changes slowly, cached aggressively)
Fail-safe: All errors -> neutral output (multiplier=1.0)
"""

import asyncio
import logging
import os
import time
from collections import deque
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Safe import of shared HTTP helper
try:
    from data_mgmt.feeds._http import create_session
except ImportError:
    import aiohttp
    def create_session(**kwargs):
        return aiohttp.ClientSession(**kwargs)


class StablecoinFlowMonitor:
    """
    [STABLECOIN-MONITOR] Tracks USDT supply as macro liquidity proxy.

    Authority: ADVISE only - modulates short sizing, never vetoes.
    """

    BASE_URL = "https://open-api-v3.coinglass.com/api"
    CACHE_TTL = 3600 * 4  # 4 hours - stablecoin supply changes slowly

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.environ.get("COINGLASS_API_KEY", "")
        self._mock_mode = not bool(self._api_key)

        # Supply history for trend computation
        self._supply_history: deque = deque(maxlen=84)  # 14 days at 4H

        # Cache
        self._last_result: Optional[Dict[str, Any]] = None
        self._last_fetch_time: float = 0
        self._fetch_errors: int = 0

        if self._mock_mode:
            logger.info("[STABLECOIN-MONITOR] No API key -> mock mode")
        else:
            logger.info("[STABLECOIN-MONITOR] Initialized with CoinGlass API")

    def _headers(self) -> Dict[str, str]:
        return {
            "CG-API-KEY": self._api_key,
            "accept": "application/json",
        }

    async def generate_signal(self) -> Dict[str, Any]:
        """
        [STABLECOIN-MONITOR] Generate stablecoin flow signal.

        Returns neutral values on any error - never blocks tick loop.
        """
        neutral = {
            "supply_change_7d_pct": 0.0,
            "supply_change_24h_pct": 0.0,
            "total_supply_usd": 0.0,
            "short_size_multiplier": 1.0,
            "flow_direction": "NEUTRAL",
            "confidence": 0.0,
            "source": "stablecoin_monitor",
        }

        try:
            if self._mock_mode:
                return self._mock_signal()

            # Use cache if fresh
            if self._last_result and (time.time() - self._last_fetch_time) < self.CACHE_TTL:
                return self._last_result

            return await self._fetch_and_compute()

        except Exception as e:
            self._fetch_errors += 1
            logger.debug(f"[STABLECOIN-MONITOR] error: {e}")
            return self._last_result if self._last_result else neutral

    async def _fetch_and_compute(self) -> Dict[str, Any]:
        """Fetch USDT supply from CoinGlass v3 and compute flow signal."""
        async with create_session() as session:
            supply_data = await self._fetch_stablecoin_supply(session)

        if not supply_data:
            if self._last_result:
                return self._last_result
            return self._mock_signal()

        # Extract current and historical supply
        current_supply = supply_data.get("current", 0.0)
        history = supply_data.get("history", [])

        # Update internal history
        if current_supply > 0:
            self._supply_history.append({
                "time": time.time(),
                "supply": current_supply,
            })

        # Compute changes from API history or internal buffer
        change_7d = 0.0
        change_24h = 0.0

        if history and len(history) >= 2:
            # Use API-provided history
            current_val = history[-1].get("supply", current_supply)
            # 7-day lookback (~42 entries at 4H)
            idx_7d = max(0, len(history) - 42)
            val_7d = history[idx_7d].get("supply", current_val)
            if val_7d > 0:
                change_7d = (current_val - val_7d) / val_7d

            # 24h lookback (~6 entries at 4H)
            idx_24h = max(0, len(history) - 6)
            val_24h = history[idx_24h].get("supply", current_val)
            if val_24h > 0:
                change_24h = (current_val - val_24h) / val_24h

        elif len(self._supply_history) >= 6:
            # Fallback to internal buffer
            buf = list(self._supply_history)
            current_val = buf[-1]["supply"]
            if len(buf) >= 42:
                val_7d = buf[-42]["supply"]
                if val_7d > 0:
                    change_7d = (current_val - val_7d) / val_7d
            if len(buf) >= 6:
                val_24h = buf[-6]["supply"]
                if val_24h > 0:
                    change_24h = (current_val - val_24h) / val_24h

        # Compute short size multiplier
        multiplier = self._compute_multiplier(change_7d)

        # Flow direction label
        if change_7d < -0.005:
            direction = "OUTFLOW"     # Capital leaving = good for shorts
        elif change_7d > 0.005:
            direction = "INFLOW"      # Capital entering = bad for shorts
        else:
            direction = "NEUTRAL"

        # Confidence based on data availability
        confidence = 0.0
        if current_supply > 0:
            confidence = 0.4
        if len(self._supply_history) >= 6:
            confidence = 0.6
        if len(self._supply_history) >= 42:
            confidence = 0.75

        result = {
            "supply_change_7d_pct": round(change_7d * 100, 4),
            "supply_change_24h_pct": round(change_24h * 100, 4),
            "total_supply_usd": round(current_supply, 0),
            "short_size_multiplier": round(multiplier, 3),
            "flow_direction": direction,
            "confidence": round(confidence, 3),
            "source": "stablecoin_monitor",
        }

        self._last_result = result
        self._last_fetch_time = time.time()
        return result

    @staticmethod
    def _compute_multiplier(change_7d: float) -> float:
        """
        Map 7-day supply change to short size multiplier.

        Supply dropping (negative) = capital leaving = GOOD for shorts -> mult > 1.0
        Supply rising  (positive) = capital entering = BAD for shorts -> mult < 1.0

        Range: [0.7, 1.1]
        """
        # change_7d negative = outflow = favorable
        # change_7d positive = inflow = headwind
        # Scale: +-2% change maps to full range
        raw = -change_7d / 0.02  # Invert: outflow -> positive
        return float(np.clip(1.0 + raw * 0.1, 0.7, 1.1))

    async def _fetch_stablecoin_supply(self, session) -> Optional[Dict]:
        """
        GET /api/stablecoin/charts - USDT supply over time.

        CoinGlass v3 stablecoin endpoint returns supply history.
        """
        try:
            url = f"{self.BASE_URL}/stablecoin/charts"
            async with session.get(
                url,
                headers=self._headers(),
                params={"symbol": "USDT"},
                timeout=__import__("aiohttp").ClientTimeout(total=15),
            ) as resp:
                if resp.status != 200:
                    logger.debug(f"[STABLECOIN-MONITOR] API status {resp.status}")
                    return None
                data = await resp.json()
                if str(data.get("code")) != "0" or not data.get("data"):
                    return None

                raw = data["data"]
                # Parse supply history
                # Expected format: list of [timestamp, supply_value]
                # or dict with "dataMap" or similar structure
                if isinstance(raw, list):
                    history = []
                    for entry in raw:
                        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                            history.append({"time": entry[0], "supply": float(entry[1])})
                        elif isinstance(entry, dict):
                            supply = float(
                                entry.get("totalSupply",
                                    entry.get("marketCap",
                                        entry.get("supply", 0)))
                            )
                            if supply > 0:
                                history.append({
                                    "time": entry.get("createTime", entry.get("t", 0)),
                                    "supply": supply,
                                })
                    if history:
                        return {
                            "current": history[-1]["supply"],
                            "history": history,
                        }
                elif isinstance(raw, dict):
                    # Single data point
                    supply = float(
                        raw.get("totalSupply",
                            raw.get("marketCap",
                                raw.get("supply", 0)))
                    )
                    if supply > 0:
                        return {"current": supply, "history": []}
                return None

        except Exception as e:
            logger.debug(f"[STABLECOIN-MONITOR] fetch error: {e}")
            return None

    def _mock_signal(self) -> Dict[str, Any]:
        """[STABLECOIN-MONITOR] Mock = neutral, no signal."""
        return {
            "supply_change_7d_pct": 0.0,
            "supply_change_24h_pct": 0.0,
            "total_supply_usd": 0.0,
            "short_size_multiplier": 1.0,
            "flow_direction": "NEUTRAL",
            "confidence": 0.0,
            "source": "stablecoin_monitor_mock",
        }

    def get_status(self) -> Dict[str, Any]:
        """Return monitor status for diagnostics."""
        return {
            "mock_mode": self._mock_mode,
            "fetch_errors": self._fetch_errors,
            "history_depth": len(self._supply_history),
            "last_fetch_time": self._last_fetch_time,
            "last_result": self._last_result,
        }


# =============================================================================
# SINGLETON
# =============================================================================

_instance: Optional[StablecoinFlowMonitor] = None


def get_stablecoin_monitor(api_key: Optional[str] = None) -> StablecoinFlowMonitor:
    """Get or create singleton."""
    global _instance
    resolved_api_key = api_key if api_key is not None else os.environ.get("COINGLASS_API_KEY", "")
    if _instance is None:
        _instance = StablecoinFlowMonitor(api_key=resolved_api_key)
    elif str(getattr(_instance, "_api_key", "") or "") != str(resolved_api_key or ""):
        logger.info("StablecoinFlowMonitor API key changed; refreshing singleton instance")
        _instance = StablecoinFlowMonitor(api_key=resolved_api_key)
    return _instance


def reset_stablecoin_monitor():
    """Reset singleton."""
    global _instance
    _instance = None
