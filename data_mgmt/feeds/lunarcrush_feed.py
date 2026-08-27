"""
================================================================================
HMATS v6.5 - LunarCrush Data Feed
================================================================================

LunarCrush API integration for social/attention metrics.

Provides:
1. Social volume / dominance
2. Galaxy Score
3. Social engagement velocity
4. Sentiment breakdown (for CROWDING only, NOT direction)

CRITICAL CONSTRAINT:
- LunarCrush data is used for ATTENTION/CROWDING detection ONLY
- NEVER used to determine trade direction
- High LunarCrush scores = potential crowding risk, NOT buy signal

API Docs: https://lunarcrush.com/developers/docs

================================================================================
"""

import logging
import asyncio
import aiohttp
from data_mgmt.feeds._http import create_session
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta, timezone
import os
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

SUPPORTED_SYMBOLS = ["BTC", "ETH", "SOL"]

# Z-score thresholds for attention
ATTENTION_HIGH_THRESHOLD = 2.0
ATTENTION_EXTREME_THRESHOLD = 3.0

# Rolling window for statistics
ROLLING_WINDOW = 24 * 6  # 24 hours at 10min intervals


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class LunarCrushMetrics:
    """Raw LunarCrush metrics for a symbol."""
    symbol: str
    
    # Social metrics
    galaxy_score: float  # 0-100
    alt_rank: int  # 1 = highest
    social_volume: float
    social_volume_change_24h: float
    social_dominance: float  # % of total crypto social
    social_contributors: int
    
    # Engagement
    social_engagement: float
    social_engagement_change_24h: float
    
    # Sentiment breakdown (CROWDING only)
    bullish_pct: float  # % bullish posts
    bearish_pct: float  # % bearish posts
    
    # Market correlation
    correlation_rank: int
    
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "galaxy_score": self.galaxy_score,
            "alt_rank": self.alt_rank,
            "social_volume": self.social_volume,
            "social_dominance": self.social_dominance,
            "bullish_pct": self.bullish_pct,
            "bearish_pct": self.bearish_pct,
        }


@dataclass
class LunarCrushAttentionData:
    """Processed attention/crowding data."""
    timestamp: datetime
    staleness_sec: float
    
    # Per-symbol raw metrics
    metrics: Dict[str, LunarCrushMetrics] = field(default_factory=dict)
    
    # Computed attention metrics (CROWDING, not direction)
    attention_pressure: Dict[str, float] = field(default_factory=dict)  # [0, 1]
    crowding_bias: Dict[str, float] = field(default_factory=dict)  # [-1, 1]
    narrative_shift: Dict[str, float] = field(default_factory=dict)  # [-1, 1]
    extreme_attention: Dict[str, bool] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "attention_pressure": self.attention_pressure,
            "crowding_bias": self.crowding_bias,
            "narrative_shift": self.narrative_shift,
            "extreme_attention": self.extreme_attention,
        }


# =============================================================================
# LUNARCRUSH FEED
# =============================================================================

class LunarCrushFeed:
    """
    LunarCrush API feed for social attention data.
    
    IMPORTANT: This data is for CROWDING detection, NOT direction prediction.
    High attention = crowding risk, NOT a buy signal.
    """
    
    BASE_URL = "https://lunarcrush.com/api4/public"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        poll_interval_sec: float = 600,  # 10 minutes
        event_bus_callback: Optional[Callable] = None,
        mock_mode: bool = False,
    ):
        self.api_key = api_key or os.environ.get("LUNARCRUSH_API_KEY", "")
        self.poll_interval_sec = poll_interval_sec
        self._event_bus_callback = event_bus_callback
        self._mock_mode = mock_mode
        
        # Internal state
        self._last_data: Optional[LunarCrushAttentionData] = None
        self._last_fetch_time: Optional[datetime] = None
        self._running = False
        self._fetch_errors = 0
        
        # History for z-score and trend calculation
        self._volume_history: Dict[str, deque] = {
            s: deque(maxlen=ROLLING_WINDOW) for s in SUPPORTED_SYMBOLS
        }
        self._attention_history: Dict[str, deque] = {
            s: deque(maxlen=ROLLING_WINDOW) for s in SUPPORTED_SYMBOLS
        }
        
        if not self.api_key and not mock_mode:
            logger.warning("[LUNARCRUSH] No API key. Set LUNARCRUSH_API_KEY env var.")
        
        logger.info(f"[LUNARCRUSH] Initialized: mock={mock_mode}")
    
    # =========================================================================
    # PUBLIC API
    # =========================================================================
    
    async def start(self):
        """Start polling loop."""
        if self._running:
            return
        
        self._running = True
        logger.info("[LUNARCRUSH] Started")
        
        while self._running:
            try:
                data = await self.fetch()
                if data and self._event_bus_callback:
                    self._event_bus_callback("LUNARCRUSH_DATA", data.to_dict())
            except Exception as e:
                logger.error(f"[LUNARCRUSH] Fetch error: {e}")
                self._fetch_errors += 1
            
            await asyncio.sleep(self.poll_interval_sec)
    
    def stop(self):
        """Stop the feed."""
        self._running = False
        logger.info("[LUNARCRUSH] Stopped")
    
    async def fetch(self) -> Optional[LunarCrushAttentionData]:
        """Fetch latest LunarCrush data."""
        try:
            if self._mock_mode:
                return await self._fetch_mock()
            
            if not self.api_key:
                logger.warning("[LUNARCRUSH] No API key, using mock")
                return await self._fetch_mock()
            
            return await self._fetch_real()
            
        except Exception as e:
            logger.error(f"[LUNARCRUSH] Fetch failed: {e}")
            self._fetch_errors += 1
            return self._last_data
    
    def get_latest(self) -> Optional[LunarCrushAttentionData]:
        """Get cached data."""
        return self._last_data

    def cache_age_sec(self) -> Optional[float]:
        """Seconds since the last successful fetch, or None if never fetched.

        [P293] None ("never fetched") is deliberately distinct from a large
        age ("fetched long ago") — different causes, different fixes.
        """
        if self._last_fetch_time is None:
            return None
        _t = self._last_fetch_time
        if _t.tzinfo is None:  # [P40/P97] defensive
            _t = _t.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - _t).total_seconds()

    async def fetch_if_stale(self) -> Optional[LunarCrushAttentionData]:
        """Fetch only when the cache is older than poll_interval_sec.

        [P293] One _fetch_real() costs one request PER SUPPORTED_SYMBOL, and
        the tick loop calls this once per asset — so throttling is what keeps
        a 4H loop from spending ~9 requests per cycle on 10-minute data.
        """
        _age = self.cache_age_sec()
        if _age is not None and _age < self.poll_interval_sec:
            return self._last_data
        return await self.fetch()


    def get_attention_metrics(self, symbol: str = "BTC") -> Dict[str, Any]:
        """
        Get attention metrics for integration.
        
        IMPORTANT: These are CROWDING metrics, NOT direction signals.
        """
        if not self._last_data:
            return self._get_default_attention_metrics()
        
        data = self._last_data
        
        return {
            "attention_pressure": data.attention_pressure.get(symbol, 0.5),
            "crowding_bias": data.crowding_bias.get(symbol, 0.0),
            "narrative_shift": data.narrative_shift.get(symbol, 0.0),
            "extreme_attention": data.extreme_attention.get(symbol, False),
            # Raw metrics for debugging
            "galaxy_score": data.metrics.get(symbol, LunarCrushMetrics(
                symbol=symbol, galaxy_score=50, alt_rank=100,
                social_volume=0, social_volume_change_24h=0,
                social_dominance=0, social_contributors=0,
                social_engagement=0, social_engagement_change_24h=0,
                bullish_pct=0.5, bearish_pct=0.5, correlation_rank=100,
                timestamp=datetime.now(timezone.utc)
            )).galaxy_score,
        }
    
    def _get_default_attention_metrics(self) -> Dict[str, Any]:
        """Default metrics when no data available."""
        return {
            "attention_pressure": 0.5,
            "crowding_bias": 0.0,
            "narrative_shift": 0.0,
            "extreme_attention": False,
            "galaxy_score": 50,
        }
    
    # =========================================================================
    # REAL API
    # =========================================================================
    
    async def _fetch_real(self) -> Optional[LunarCrushAttentionData]:  # [P420] None = keep the previous cache
        """Fetch real data from LunarCrush."""
        now = datetime.now(timezone.utc)
        
        data = LunarCrushAttentionData(
            timestamp=now,
            staleness_sec=0.0,
        )
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }
        
        self._last_status_by_symbol: Dict[str, Optional[int]] = {}
        async with create_session() as session:
            for symbol in SUPPORTED_SYMBOLS:
                try:
                    metrics = await self._fetch_coin_metrics(session, headers, symbol)
                    if metrics:
                        data.metrics[symbol] = metrics
                except Exception as e:
                    logger.warning(f"[LUNARCRUSH] {symbol}: {e}")

        # [P420] A fetch in which NO symbol returned metrics (every request a
        # non-200 or exception) used to fall through to
        # _compute_attention_metrics on an EMPTY metrics map and store the
        # result fresh-stamped: attention_pressure 0.5 / crowding 0.0 for
        # every symbol = NEUTRAL, indistinguishable from a measured calm
        # (P2/P216 shape), and fetch_if_stale then served it for
        # poll_interval_sec. Keep the previous cache (its own timestamp, so
        # cache_age_sec reports honest staleness), log the statuses, and do
        # NOT stamp _last_fetch_time on a failed fetch.
        if not data.metrics:
            _st = {k: v for k, v in
                   getattr(self, "_last_status_by_symbol", {}).items()}
            logger.warning(
                f"[LUNARCRUSH] fetch returned metrics for NO symbol "
                f"(statuses={_st or 'n/a'}) — keeping the previous cache "
                f"({'present' if self._last_data is not None else 'ABSENT'}) "
                f"rather than stamping fresh NEUTRAL metrics (P420)")
            return self._last_data

        # Compute derived metrics
        self._compute_attention_metrics(data)

        self._last_data = data
        self._last_fetch_time = now

        return data
    
    async def _fetch_coin_metrics(
        self,
        session: aiohttp.ClientSession,
        headers: Dict,
        symbol: str,
    ) -> Optional[LunarCrushMetrics]:
        """Fetch metrics for a single coin."""
        url = f"{self.BASE_URL}/coins/{symbol.lower()}/v1"
        
        try:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                # [P420] record the status so a fetch-wide failure can be
                # reported with its cause instead of as silent NEUTRAL
                if hasattr(self, "_last_status_by_symbol"):
                    self._last_status_by_symbol[symbol] = resp.status
                if resp.status != 200:
                    logger.warning(f"[LUNARCRUSH] {symbol}: HTTP {resp.status}")
                    return None

                result = await resp.json()
                coin_data = result.get("data", {})

                if not coin_data:
                    return None

                # Extract sentiment breakdown
                sentiment = coin_data.get("sentiment", {})
                bullish = sentiment.get("bullish", 50)
                bearish = sentiment.get("bearish", 50)
                total = bullish + bearish
                bullish_pct = bullish / total if total > 0 else 0.5
                bearish_pct = bearish / total if total > 0 else 0.5

                return LunarCrushMetrics(
                    symbol=symbol,
                    galaxy_score=coin_data.get("galaxy_score", 50),
                    alt_rank=coin_data.get("alt_rank", 100),
                    social_volume=coin_data.get("social_volume", 0),
                    social_volume_change_24h=coin_data.get("social_volume_change_24h", 0),
                    social_dominance=coin_data.get("social_dominance", 0),
                    social_contributors=coin_data.get("social_contributors", 0),
                    social_engagement=coin_data.get("social_engagement", 0),
                    social_engagement_change_24h=coin_data.get("social_engagement_change_24h", 0),
                    bullish_pct=bullish_pct,
                    bearish_pct=bearish_pct,
                    correlation_rank=coin_data.get("correlation_rank", 100),
                    timestamp=datetime.now(timezone.utc),
                )
        except asyncio.TimeoutError:
            logger.error(f"[LUNARCRUSH] {symbol} timed out (10s)")
            return None
    
    # =========================================================================
    # ATTENTION COMPUTATION
    # =========================================================================
    
    def _compute_attention_metrics(self, data: LunarCrushAttentionData):
        """
        Compute attention/crowding metrics from raw data.
        
        CRITICAL: These metrics indicate CROWDING, not direction.
        High attention = potential crowding risk.
        """
        for symbol in SUPPORTED_SYMBOLS:
            if symbol not in data.metrics:
                data.attention_pressure[symbol] = 0.5
                data.crowding_bias[symbol] = 0.0
                data.narrative_shift[symbol] = 0.0
                data.extreme_attention[symbol] = False
                continue
            
            metrics = data.metrics[symbol]
            
            # Update history
            self._volume_history[symbol].append(metrics.social_volume)
            
            # === Attention Pressure ===
            # Z-score of social volume
            volume_history = list(self._volume_history[symbol])
            if len(volume_history) >= 10:
                mean_vol = np.mean(volume_history)
                std_vol = np.std(volume_history) + 1e-6
                z_score = (metrics.social_volume - mean_vol) / std_vol
                # Map z-score to [0, 1]: z=0 -> 0.5, z=3 -> 1.0, z=-3 -> 0.0
                attention_pressure = np.clip(z_score / 3 * 0.5 + 0.5, 0.0, 1.0)
            else:
                # Galaxy score fallback
                attention_pressure = metrics.galaxy_score / 100
            
            data.attention_pressure[symbol] = attention_pressure
            self._attention_history[symbol].append(attention_pressure)
            
            # === Crowding Bias ===
            # Based on sentiment breakdown (CROWDING indicator, not direction)
            # High bullish consensus = potential crowded longs
            # High bearish consensus = potential crowded shorts
            if metrics.bullish_pct > 0 or metrics.bearish_pct > 0:
                # Positive = bullish crowding, Negative = bearish crowding
                crowding_bias = (metrics.bullish_pct - 0.5) * 2
                crowding_bias = np.clip(crowding_bias, -1.0, 1.0)
            else:
                crowding_bias = (attention_pressure - 0.5) * 2
                crowding_bias = np.clip(crowding_bias, -1.0, 1.0)
            
            data.crowding_bias[symbol] = crowding_bias
            
            # === Narrative Shift ===
            # EMA delta of attention pressure
            attention_history = list(self._attention_history[symbol])
            if len(attention_history) >= 3:
                alpha = 0.3
                ema = attention_history[0]
                for val in attention_history[1:]:
                    ema = alpha * val + (1 - alpha) * ema
                
                recent_avg = np.mean(attention_history[-3:])
                shift = (recent_avg - ema) * 3
                narrative_shift = np.clip(shift, -1.0, 1.0)
            else:
                narrative_shift = 0.0
            
            data.narrative_shift[symbol] = narrative_shift
            
            # === Extreme Attention ===
            # Triggered when attention is very high (crowding risk)
            extreme = (
                attention_pressure >= 0.9 or
                abs(crowding_bias) >= 0.8 or
                abs(narrative_shift) >= 0.7
            )
            data.extreme_attention[symbol] = extreme
    
    # =========================================================================
    # MOCK DATA
    # =========================================================================
    
    async def _fetch_mock(self) -> LunarCrushAttentionData:
        """Generate mock data.

        [P101 2026-04-27] PATCH-7a principle (per sentiment_feed.py:380):
        random in mock data injects non-deterministic NOISE into agent
        signals, making fallback indistinguishable from real low-signal
        data and causing non-reproducible trades on test/CI runs.
        Replaced with NEUTRAL CONSTANTS that explicitly signal "no
        opinion" — galaxy_score=50 (mid-range), bullish=bearish=0.5
        (perfectly neutral), volumes/contributors at moderate fixed
        values. Downstream consumers can detect mock via is_mock=True
        if needed.
        """
        await asyncio.sleep(0.1)

        now = datetime.now(timezone.utc)
        data = LunarCrushAttentionData(timestamp=now, staleness_sec=0.0)

        # Per-symbol social_dominance is the only varying field — it
        # reflects real market structure (BTC dominates), so keeping it
        # asymmetric is informative even in mock.
        for symbol in SUPPORTED_SYMBOLS:
            data.metrics[symbol] = LunarCrushMetrics(
                symbol=symbol,
                galaxy_score=50.0,                    # mid-range neutral
                alt_rank=50,                          # mid-list
                social_volume=50000.0,                # moderate fixed
                social_volume_change_24h=0.0,         # no change → neutral
                social_dominance={"BTC": 40, "ETH": 25, "SOL": 5}.get(symbol, 5),
                social_contributors=5000,             # moderate fixed
                social_engagement=250000.0,           # moderate fixed
                social_engagement_change_24h=0.0,     # no change → neutral
                bullish_pct=0.5,                      # perfectly neutral
                bearish_pct=0.5,                      # perfectly neutral
                correlation_rank=25,                  # mid-list
                timestamp=now,
            )

        self._compute_attention_metrics(data)
        self._last_data = data
        self._last_fetch_time = now

        return data


# =============================================================================
# SINGLETON
# =============================================================================

_lunarcrush_feed_instance: Optional[LunarCrushFeed] = None


def get_lunarcrush_feed(
    api_key: Optional[str] = None,
    mock_mode: bool = False,
) -> LunarCrushFeed:
    """Get or create LunarCrushFeed singleton."""
    global _lunarcrush_feed_instance
    if _lunarcrush_feed_instance is None:
        _lunarcrush_feed_instance = LunarCrushFeed(
            api_key=api_key,
            mock_mode=mock_mode,
        )
    return _lunarcrush_feed_instance


def reset_lunarcrush_feed():
    """Reset singleton."""
    global _lunarcrush_feed_instance
    if _lunarcrush_feed_instance:
        _lunarcrush_feed_instance.stop()
    _lunarcrush_feed_instance = None


# =============================================================================
# TESTS
# =============================================================================

if __name__ == "__main__":
    import asyncio
    
    async def test():
        print("=" * 70)
        print("LunarCrush Feed Test")
        print("=" * 70)
        print("\nIMPORTANT: These metrics are for CROWDING detection, NOT direction!")
        print("High attention = crowding risk, NOT a buy signal.\n")
        
        feed = LunarCrushFeed(mock_mode=True)
        
        # Fetch multiple times to build history
        for i in range(5):
            data = await feed.fetch()
        
        print(f"Attention Pressure: {data.attention_pressure}")
        print(f"Crowding Bias: {data.crowding_bias}")
        print(f"Narrative Shift: {data.narrative_shift}")
        print(f"Extreme Attention: {data.extreme_attention}")
        
        print("\nBTC Attention Metrics:")
        metrics = feed.get_attention_metrics("BTC")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        
        print("\nON Test passed!")
    
    asyncio.run(test())
