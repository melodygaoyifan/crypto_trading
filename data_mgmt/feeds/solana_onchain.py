"""
================================================================================
HMATS - Solana On-Chain Metrics Feed
================================================================================
Provides: TPS, slot time, network congestion, Jito MEV tip floor.

Data sources (all public, no API key needed):
  1. Solana RPC: getRecentPerformanceSamples -> TPS, slot time
  2. Jito Public API: tip_floor -> MEV pressure / congestion proxy

Rate limits: Solana RPC ~40 req/10s, Jito undocumented (we do 1 req/120s).
================================================================================
"""

import os
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Any, Dict
from datetime import datetime, timezone

logger = logging.getLogger("HMATS.SolanaOnChain")

# Attempt aiohttp import
try:
    import aiohttp
    from data_mgmt.feeds._http import create_session
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

DEFAULT_RPC = "https://api.mainnet-beta.solana.com"
JITO_TIP_URL = "https://bundles.jito.wtf/api/v1/bundles/tip_floor"
MIN_FETCH_INTERVAL = 120.0  # 2 minutes


@dataclass
class SolanaOnChainData:
    """SOL-specific on-chain metrics."""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Network health
    tps: float = 0.0                    # Transactions per second
    slot_time_ms: float = 400.0         # Average slot time (normal ~400ms)
    network_congested: bool = False     # True if TPS < 1500 or slot_time > 600ms

    # Jito MEV pressure
    jito_tip_50th_sol: float = 0.0      # 50th percentile tip in SOL
    jito_tip_95th_sol: float = 0.0      # 95th percentile tip in SOL
    jito_ema_sol: float = 0.0           # EMA of 50th percentile
    jito_pressure: float = 0.0          # 0-1 normalized congestion signal

    is_mock: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tps": self.tps,
            "slot_time_ms": self.slot_time_ms,
            "network_congested": self.network_congested,
            "jito_tip_50th_sol": self.jito_tip_50th_sol,
            "jito_tip_95th_sol": self.jito_tip_95th_sol,
            "jito_pressure": self.jito_pressure,
            "is_mock": self.is_mock,
        }


class SolanaOnChainFeed:
    """
    Fetches SOL on-chain data from public Solana RPC + Jito.

    Rate limited to MIN_FETCH_INTERVAL (120s).
    No API key needed - all public endpoints.
    """

    def __init__(self, rpc_url: str = ""):
        self._rpc_url = rpc_url or os.environ.get("SOLANA_RPC_URL", DEFAULT_RPC)
        self._last_fetch_time: float = 0.0
        self._data = SolanaOnChainData()
        self._fetch_count: int = 0
        self._error_count: int = 0

        logger.info(f"[SOL_ONCHAIN] RPC={self._rpc_url[:50]}...")

    @property
    def data(self) -> SolanaOnChainData:
        return self._data

    async def fetch(self) -> SolanaOnChainData:
        """Fetch SOL on-chain metrics. Rate-limited to 120s."""
        if not AIOHTTP_AVAILABLE:
            return self._data

        now = time.time()
        if now - self._last_fetch_time < MIN_FETCH_INTERVAL and not self._data.is_mock:
            return self._data

        data = SolanaOnChainData(timestamp=datetime.now(timezone.utc), is_mock=False)

        try:
            async with create_session(
                timeout=aiohttp.ClientTimeout(total=15)
            ) as session:
                # 1. Network performance (TPS, slot time)
                await self._fetch_performance(session, data)

                # 2. Jito MEV tip floor
                await self._fetch_jito(session, data)

        except Exception as e:
            logger.warning(f"[SOL_ONCHAIN] Session failed: {e}")
            data.is_mock = True
            self._error_count += 1

        self._data = data
        self._last_fetch_time = now
        self._fetch_count += 1
        return data

    async def _fetch_performance(self, session: aiohttp.ClientSession, data: SolanaOnChainData):
        """Fetch TPS and slot time from Solana RPC."""
        try:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getRecentPerformanceSamples",
                "params": [5],
            }
            async with session.post(
                self._rpc_url, json=payload,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    logger.warning(f"[SOL_ONCHAIN] RPC returned {resp.status}")
                    # [P101 2026-04-27] Mark mock so downstream is_mock
                    # checks distinguish "RPC down" from "RPC says quiet".
                    data.is_mock = True
                    return

                result = await resp.json()
                samples = result.get("result", [])
                if not samples:
                    logger.warning(
                        f"[SOL_ONCHAIN] RPC returned 200 but empty samples "
                        f"(result keys: {list(result.keys())}); marking mock "
                        f"so downstream skips bogus zero TPS reading."
                    )
                    data.is_mock = True
                    return

                total_txs = sum(s.get("numTransactions", 0) for s in samples)
                total_slots = sum(s.get("numSlots", 1) for s in samples)
                total_secs = sum(s.get("samplePeriodSecs", 60) for s in samples)

                data.tps = total_txs / max(total_secs, 1)
                data.slot_time_ms = (total_secs / max(total_slots, 1)) * 1000
                data.network_congested = data.tps < 1500 or data.slot_time_ms > 600

                logger.info(
                    f"[SOL_ONCHAIN] TPS={data.tps:.0f}, "
                    f"slot_time={data.slot_time_ms:.0f}ms, "
                    f"congested={data.network_congested}"
                )

        except Exception as e:
            logger.warning(f"[SOL_ONCHAIN] RPC performance failed: {e}")
            self._error_count += 1

    async def _fetch_jito(self, session: aiohttp.ClientSession, data: SolanaOnChainData):
        """Fetch Jito MEV tip floor."""
        try:
            async with session.get(
                JITO_TIP_URL,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    logger.warning(f"[SOL_ONCHAIN] Jito returned {resp.status}")
                    # [P101 2026-04-27] Don't mark whole-data as mock here
                    # (TPS may still be valid); just log so operator knows
                    # MEV pressure signal is missing this fetch.
                    return

                tip_data = await resp.json()

                # Response is a list with one entry (verified Feb 24)
                # Fields: landed_tips_{25,50,75,95,99}th_percentile (in SOL)
                #         ema_landed_tips_50th_percentile (in SOL)
                entry = {}
                if isinstance(tip_data, list) and tip_data:
                    entry = tip_data[0]
                elif isinstance(tip_data, dict):
                    entry = tip_data

                if not entry:
                    # [P101 2026-04-27] Empty Jito payload had silent
                    # zero-fallthrough — jito_pressure stayed at 0.0
                    # (dataclass default), indistinguishable from real
                    # "no MEV pressure". Surface so operator knows the
                    # feed returned 200 but no usable data.
                    logger.warning(
                        f"[SOL_ONCHAIN] Jito returned 200 but no usable "
                        f"tip_data ({type(tip_data).__name__}, "
                        f"len={len(tip_data) if hasattr(tip_data, '__len__') else 'N/A'}); "
                        f"jito_pressure stays at default."
                    )

                if entry:
                    data.jito_tip_50th_sol = float(entry.get("landed_tips_50th_percentile", 0))
                    data.jito_tip_95th_sol = float(entry.get("landed_tips_95th_percentile", 0))
                    data.jito_ema_sol = float(entry.get("ema_landed_tips_50th_percentile", 0))

                    # Normalize to 0-1 pressure signal
                    # Typical range: 1e-6 (calm) to 1e-3 (extreme) SOL
                    # Log-scale normalization:
                    #   < 1e-6 SOL -> 0.0 (calm)
                    #   1e-5 SOL -> 0.33
                    #   1e-4 SOL -> 0.67
                    #   > 1e-3 SOL -> 1.0 (extreme)
                    tip = max(data.jito_tip_50th_sol, 1e-9)
                    import math
                    log_tip = math.log10(tip)
                    # Map log10 from [-6, -3] to [0, 1]
                    data.jito_pressure = max(0.0, min(1.0, (log_tip + 6) / 3.0))

                    logger.info(
                        f"[SOL_ONCHAIN] Jito tip_50={data.jito_tip_50th_sol:.2e} SOL, "
                        f"tip_95={data.jito_tip_95th_sol:.2e} SOL, "
                        f"pressure={data.jito_pressure:.2f}"
                    )

        except Exception as e:
            logger.warning(f"[SOL_ONCHAIN] Jito fetch failed: {e}")
            self._error_count += 1

    def get_status(self) -> Dict[str, Any]:
        return {
            "fetch_count": self._fetch_count,
            "error_count": self._error_count,
            "has_data": not self._data.is_mock,
            "tps": self._data.tps,
            "congested": self._data.network_congested,
            "jito_pressure": self._data.jito_pressure,
        }


# Singleton
_instance: Optional[SolanaOnChainFeed] = None


def get_solana_onchain_feed(**kwargs) -> SolanaOnChainFeed:
    global _instance
    if _instance is None:
        _instance = SolanaOnChainFeed(**kwargs)
    return _instance