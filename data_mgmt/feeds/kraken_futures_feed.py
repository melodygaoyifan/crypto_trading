"""
================================================================================
HMATS - Kraken Futures Derivatives Data Feed
================================================================================
Version: 1.0.0
Purpose: Derivatives data source (funding rate, OI, mark price).

Uses Kraken's PUBLIC Futures API:
  GET https://futures.kraken.com/derivatives/api/v3/tickers

Provides per-asset:
  - funding_rate (converted from per-hour to per-8h)
  - funding_rate_prediction
  - open_interest_usd
  - mark_price
  - premium (basis)
  - volume_24h

No API key required. Single call returns all assets.
================================================================================
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Any
from datetime import datetime, timezone

logger = logging.getLogger("HMATS.KrakenFutures")

import math

def _is_finite(v: float) -> bool:
    """Check if a float is finite (not NaN, not inf)."""
    return math.isfinite(v)

def _safe_float(v, default: float = 0.0) -> float:
    """Convert to float with fallback for None/invalid."""
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default

# Attempt aiohttp import
try:
    import aiohttp
    from data_mgmt.feeds._http import create_session, fetch_with_retry
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================

TICKERS_URL = "https://futures.kraken.com/derivatives/api/v3/tickers"

# Kraken Futures symbol mapping
# PF_ = multi-collateral perpetual (preferred)
SYMBOL_MAP = {
    "BTC": "PF_XBTUSD",
    "ETH": "PF_ETHUSD",
    "SOL": "PF_SOLUSD",
}

# Reverse map for lookup
REVERSE_MAP = {v: k for k, v in SYMBOL_MAP.items()}

# Kraken funding rate is PER HOUR. System standard is PER 8 HOURS.
FUNDING_RATE_MULTIPLIER = 8.0

# Fetch interval (seconds) - don't spam the API
MIN_FETCH_INTERVAL = 60.0  # 1 minute minimum between fetches


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class KrakenFuturesTicker:
    """Parsed ticker data for one perpetual contract."""
    symbol: str                          # "BTC", "ETH", "SOL"
    funding_rate_hourly: float           # Per-hour funding rate (raw from API)
    funding_rate_8h: float               # Per-8h (converted, for system use)
    funding_rate_prediction_hourly: float # Predicted next hour
    funding_rate_prediction_8h: float    # Predicted next 8h
    open_interest_usd: float             # Open interest in USD
    mark_price: float                    # Mark price
    premium: float                       # Basis/premium vs index
    volume_24h: float                    # 24h volume
    index_price: float                   # Spot index price
    bid: float
    ask: float
    last_price: float
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "funding_rate_8h": self.funding_rate_8h,
            "funding_rate_prediction_8h": self.funding_rate_prediction_8h,
            "open_interest_usd": self.open_interest_usd,
            "mark_price": self.mark_price,
            "premium": self.premium,
            "volume_24h": self.volume_24h,
            "index_price": self.index_price,
        }


@dataclass
class KrakenFuturesSnapshot:
    """All perpetual contract data from a single API call."""
    timestamp: datetime
    tickers: Dict[str, KrakenFuturesTicker] = field(default_factory=dict)
    fetch_latency_ms: float = 0.0
    error: Optional[str] = None

    def get(self, asset: str) -> Optional[KrakenFuturesTicker]:
        """Get ticker for an asset (e.g., 'BTC')."""
        return self.tickers.get(asset.upper())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "tickers": {k: v.to_dict() for k, v in self.tickers.items()},
            "fetch_latency_ms": self.fetch_latency_ms,
        }


# =============================================================================
# FEED IMPLEMENTATION
# =============================================================================

class KrakenFuturesFeed:
    """
    Fetches derivatives data from Kraken Futures public API.

    Usage:
        feed = KrakenFuturesFeed()
        snapshot = await feed.fetch()
        btc = snapshot.get("BTC")
        print(btc.funding_rate_8h, btc.open_interest_usd)
    """

    def __init__(self):
        self._last_fetch_time: float = 0.0
        self._last_snapshot: Optional[KrakenFuturesSnapshot] = None
        self._fetch_count: int = 0
        self._error_count: int = 0
        logger.info("[KRAKEN_FUTURES] Feed initialized")

    @property
    def last_snapshot(self) -> Optional[KrakenFuturesSnapshot]:
        return self._last_snapshot

    async def fetch(self) -> Optional[KrakenFuturesSnapshot]:
        """
        Fetch all perpetual tickers from Kraken Futures.
        Rate-limited to MIN_FETCH_INTERVAL.
        Returns cached snapshot if called too frequently.
        """
        if not AIOHTTP_AVAILABLE:
            logger.warning("[KRAKEN_FUTURES] aiohttp not available")
            return None

        # Rate limit
        now = time.time()
        if now - self._last_fetch_time < MIN_FETCH_INTERVAL and self._last_snapshot:
            return self._last_snapshot

        t0 = time.monotonic()
        snapshot = KrakenFuturesSnapshot(
            timestamp=datetime.now(timezone.utc),
        )

        try:
            async with create_session(
                timeout=aiohttp.ClientTimeout(total=15)
            ) as session:
                data = await fetch_with_retry(
                    session,
                    TICKERS_URL,
                    max_retries=2,
                    operation="kraken_futures_tickers",
                )

                if not data or "tickers" not in data:
                    snapshot.error = f"Invalid response: {type(data)}"
                    self._error_count += 1
                    logger.warning(f"[KRAKEN_FUTURES] {snapshot.error}")
                    return snapshot

                # Parse perpetual tickers for our assets
                for ticker in data["tickers"]:
                    symbol = ticker.get("symbol", "")
                    tag = ticker.get("tag", "")

                    # Only perpetual contracts we care about
                    if tag != "perpetual" or symbol not in REVERSE_MAP:
                        continue

                    asset = REVERSE_MAP[symbol]

                    # [FIX-39] Validate markPrice before any downstream division
                    try:
                        mark = float(ticker.get("markPrice", 0.0))
                    except (TypeError, ValueError):
                        logger.warning(f"[KRAKEN_FUTURES] {asset}: invalid markPrice={ticker.get('markPrice')}")
                        continue
                    if mark <= 0 or mark > 1_000_000 or not _is_finite(mark):
                        logger.warning(f"[KRAKEN_FUTURES] {asset}: markPrice out of bounds: {mark}")
                        continue  # Skip this asset entirely

                    # Kraken tickers return ABSOLUTE funding rate (USD per
                    # contract-unit per hour).  Convert to RELATIVE (fraction
                    # of mark price) before applying the 8h multiplier.
                    fr_abs = _safe_float(ticker.get("fundingRate", 0.0))
                    frp_abs = _safe_float(ticker.get("fundingRatePrediction", 0.0))
                    fr_hourly = fr_abs / mark  # mark > 0 guaranteed above
                    frp_hourly = frp_abs / mark

                    # [FIX-39] Bounds check: hourly funding rate > ±0.1% is suspicious
                    if abs(fr_hourly) > 0.001:
                        logger.warning(f"[KRAKEN_FUTURES] {asset}: funding_rate_hourly={fr_hourly:.6f} out of range, clamping")
                        fr_hourly = max(-0.001, min(0.001, fr_hourly))
                    if abs(frp_hourly) > 0.001:
                        frp_hourly = max(-0.001, min(0.001, frp_hourly))

                    # OI from API is in native units (BTC/ETH/SOL) -> USD
                    oi_native = _safe_float(ticker.get("openInterest", 0.0))
                    oi_usd = oi_native * mark
                    # [FIX-39] OI sanity check
                    if oi_usd > 1e12 or oi_usd < 0:
                        logger.warning(f"[KRAKEN_FUTURES] {asset}: OI_USD out of bounds: {oi_usd}")
                        oi_usd = 0.0

                    parsed = KrakenFuturesTicker(
                        symbol=asset,
                        funding_rate_hourly=fr_hourly,
                        funding_rate_8h=fr_hourly * FUNDING_RATE_MULTIPLIER,
                        funding_rate_prediction_hourly=frp_hourly,
                        funding_rate_prediction_8h=frp_hourly * FUNDING_RATE_MULTIPLIER,
                        open_interest_usd=oi_usd,
                        mark_price=mark,
                        premium=float(ticker.get("premium", 0.0)),
                        volume_24h=float(ticker.get("vol24h", 0.0)),
                        index_price=float(ticker.get("indexPrice", 0.0)),
                        bid=float(ticker.get("bid", 0.0)),
                        ask=float(ticker.get("ask", 0.0)),
                        last_price=float(ticker.get("last", 0.0)),
                        timestamp=datetime.now(timezone.utc),
                    )
                    snapshot.tickers[asset] = parsed

                snapshot.fetch_latency_ms = (time.monotonic() - t0) * 1000
                self._last_fetch_time = now
                self._last_snapshot = snapshot
                self._fetch_count += 1

                # Log summary
                for asset, t in snapshot.tickers.items():
                    logger.info(
                        f"[KRAKEN_FUTURES] {asset}: "
                        f"funding={t.funding_rate_8h:.6f}/8h "
                        f"(pred={t.funding_rate_prediction_8h:.6f}), "
                        f"OI=${t.open_interest_usd:,.0f}, "
                        f"premium={t.premium:.6f}, "
                        f"mark=${t.mark_price:,.2f}"
                    )

        except Exception as e:
            snapshot.error = str(e)
            self._error_count += 1
            logger.warning(f"[KRAKEN_FUTURES] Fetch failed: {e}")

        return snapshot

    def get_status(self) -> Dict[str, Any]:
        """Get feed status for monitoring."""
        return {
            "fetch_count": self._fetch_count,
            "error_count": self._error_count,
            "last_fetch": self._last_fetch_time,
            "has_data": self._last_snapshot is not None and len(self._last_snapshot.tickers) > 0,
            "assets": list(self._last_snapshot.tickers.keys()) if self._last_snapshot else [],
        }


# =============================================================================
# SINGLETON
# =============================================================================

_instance: Optional[KrakenFuturesFeed] = None

def get_kraken_futures_feed() -> KrakenFuturesFeed:
    """Get or create singleton feed instance."""
    global _instance
    if _instance is None:
        _instance = KrakenFuturesFeed()
    return _instance