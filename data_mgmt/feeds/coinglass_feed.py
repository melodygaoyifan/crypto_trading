"""
================================================================================
HMATS v6.5 - Coinglass Data Feed
================================================================================

Coinglass API integration for crypto derivatives data.

Provides:
1. Funding rates (perpetual swaps)
2. Open interest
3. Liquidation data
4. Long/Short ratio

CRITICAL: This is a P0 data source for crowding/squeeze detection.

API Docs: https://docs.coinglass.com/

================================================================================
"""

import logging
import asyncio
import aiohttp
from collections import deque
from data_mgmt.feeds._http import create_session
from dataclasses import dataclass, field
from typing import Dict, Deque, List, Optional, Any, Callable
from datetime import datetime, timedelta, timezone
from enum import Enum
import os
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Supported assets
SUPPORTED_SYMBOLS = ["BTC", "ETH", "SOL"]

# Funding rate normalization
FUNDING_NORM = 0.0003  # 0.03% = neutral, typical 8h funding

# OI change thresholds
OI_CHANGE_SIGNIFICANT = 0.05  # 5% change is significant


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class FundingData:
    """Funding rate data."""
    symbol: str
    rate: float  # Current funding rate
    predicted_rate: Optional[float]  # Predicted next rate
    interval_hours: int
    exchange: str
    timestamp: datetime


@dataclass
class OpenInterestData:
    """Open interest data."""
    symbol: str
    open_interest_usd: float
    open_interest_btc: float  # Normalized to BTC
    change_24h_pct: float
    change_1h_pct: float
    timestamp: datetime


@dataclass
class LiquidationData:
    """Liquidation data."""
    symbol: str
    long_liquidations_24h: float
    short_liquidations_24h: float
    total_liquidations_24h: float
    largest_single_liquidation: float
    timestamp: datetime


@dataclass
class LongShortRatio:
    """Long/Short position ratio."""
    symbol: str
    long_pct: float
    short_pct: float
    ratio: float  # long / short
    timestamp: datetime


@dataclass
class CoinglassCrowdData:
    """Aggregated Coinglass crowd/derivatives data."""
    timestamp: datetime
    staleness_sec: float
    
    # Per-symbol data
    funding: Dict[str, FundingData] = field(default_factory=dict)
    open_interest: Dict[str, OpenInterestData] = field(default_factory=dict)
    liquidations: Dict[str, LiquidationData] = field(default_factory=dict)
    long_short_ratio: Dict[str, LongShortRatio] = field(default_factory=dict)
    
    # Aggregated crowding metrics
    funding_bias: Dict[str, float] = field(default_factory=dict)  # [-1, 1]
    oi_trend: Dict[str, float] = field(default_factory=dict)  # [-1, 1]
    liquidation_imbalance: Dict[str, float] = field(default_factory=dict)  # [-1, 1]
    crowding_score: Dict[str, float] = field(default_factory=dict)  # [0, 1]

    # [P287] Age (seconds, at fetch time) of entries CARRIED FORWARD from the
    # previous cache because their endpoint family failed this fetch. Keyed
    # "family:SYMBOL" (e.g. "liquidations:BTC"). Absent key = fetched fresh.
    # Consumers that care about staleness read this; the carried entries also
    # keep their ORIGINAL per-entry timestamps.
    family_age_sec: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "funding_bias": self.funding_bias,
            "oi_trend": self.oi_trend,
            "liquidation_imbalance": self.liquidation_imbalance,
            "crowding_score": self.crowding_score,
            "family_age_sec": self.family_age_sec,
        }
    
    def get_crowd_context(self, symbol: str) -> Dict[str, float]:
        """Get crowd context for a specific symbol."""
        return {
            "funding_bias": self.funding_bias.get(symbol, 0.0),
            "oi_trend": self.oi_trend.get(symbol, 0.0),
            "liquidation_imbalance": self.liquidation_imbalance.get(symbol, 0.0),
            "crowding_score": self.crowding_score.get(symbol, 0.0),
        }


# =============================================================================
# COINGLASS FEED
# =============================================================================

class CoinglassFeed:
    """
    Coinglass API feed for derivatives data.

    Fetches funding rates, OI, liquidations for crowding detection.
    """

    BASE_URL = "https://open-api.coinglass.com/public/v2"
    V3_BASE_URL = "https://open-api-v3.coinglass.com/api"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        poll_interval_sec: float = 300,  # 5 minutes
        event_bus_callback: Optional[Callable] = None,
        mock_mode: bool = False,
    ):
        self.api_key = api_key or os.environ.get("COINGLASS_API_KEY", "")
        self.poll_interval_sec = poll_interval_sec
        self._event_bus_callback = event_bus_callback
        self._mock_mode = mock_mode
        
        # Internal state
        self._last_data: Optional[CoinglassCrowdData] = None
        self._last_fetch_time: Optional[datetime] = None
        self._running = False
        self._fetch_errors = 0
        
        # History for trend calculation - [FIX-38] bounded deques (was unbounded list)
        self._history_max_len = 24  # 2 hours at 5min intervals
        self._oi_history: Dict[str, Deque[float]] = {
            s: deque(maxlen=self._history_max_len) for s in SUPPORTED_SYMBOLS
        }
        self._funding_history: Dict[str, Deque[float]] = {
            s: deque(maxlen=self._history_max_len) for s in SUPPORTED_SYMBOLS
        }
        
        if not self.api_key and not mock_mode:
            logger.warning("[COINGLASS] No API key. Set COINGLASS_API_KEY env var.")
        
        logger.info(f"[COINGLASS] Initialized: mock={mock_mode}")
    
    # =========================================================================
    # PUBLIC API
    # =========================================================================
    
    async def start(self):
        """Start polling loop."""
        if self._running:
            return
        
        self._running = True
        logger.info("[COINGLASS] Started")
        
        while self._running:
            try:
                data = await self.fetch()
                if data and self._event_bus_callback:
                    self._event_bus_callback("COINGLASS_DATA", data.to_dict())
            except Exception as e:
                logger.error(f"[COINGLASS] Fetch error: {e}")
                self._fetch_errors += 1
            
            await asyncio.sleep(self.poll_interval_sec)
    
    def stop(self):
        """Stop the feed."""
        self._running = False
        logger.info("[COINGLASS] Stopped")
    
    async def fetch(self) -> Optional[CoinglassCrowdData]:
        """Fetch latest Coinglass data."""
        try:
            if self._mock_mode:
                return await self._fetch_mock()

            if not self.api_key:
                logger.warning("[COINGLASS] No API key, using mock")
                return await self._fetch_mock()

            return await self._fetch_real()

        except Exception as e:
            logger.error(f"[COINGLASS] Fetch failed: {e}")
            self._fetch_errors += 1
            # [P100 2026-04-27] Update staleness_sec on the stale fallback
            # so downstream data_health gates can detect aged data. Field
            # was previously frozen at 0.0 at fetch time, making stale
            # crowd data look fresh forever.
            return self._refreshed_staleness(self._last_data)

    async def fetch_if_stale(self) -> Optional[CoinglassCrowdData]:
        """[P293f] Fetch only when the cache is older than poll_interval_sec.

        MEASURED WASTE this removes: `fetch()` has no throttle and the tick
        loop calls it once per ASSET, while `_fetch_real()` already loops
        SUPPORTED_SYMBOLS = [BTC, ETH, SOL] internally across three endpoint
        families. So a 4H cycle spent roughly 3x the requests it needed —
        every symbol fetched three times, on a PAID plan — because the first
        asset's fetch already contained the other two assets' data.

        The APIs here expose no ETag / Last-Modified (probed 2026-08-17), so
        conditional requests are unavailable and client-side TTL is the only
        lever. 300s on a 14400s tick means all three assets read data less
        than a minute old while only one request set is spent.
        """
        from data_mgmt.feeds._http import cache_age_seconds
        _age = cache_age_seconds(self._last_fetch_time)
        if _age is not None and _age < float(self.poll_interval_sec or 0):
            return self._refreshed_staleness(self._last_data)
        return await self.fetch()

    def get_latest(self) -> Optional[CoinglassCrowdData]:
        """Get cached data with up-to-date staleness_sec."""
        # [P100 2026-04-27] Compute current staleness on every read so
        # consumers see actual age, not the 0.0 baked in at fetch time.
        return self._refreshed_staleness(self._last_data)

    def _refreshed_staleness(
        self, data: Optional[CoinglassCrowdData]
    ) -> Optional[CoinglassCrowdData]:
        """Update staleness_sec on cached data based on current time."""
        if data is None or data.timestamp is None:
            return data
        try:
            now = datetime.now(timezone.utc)
            data.staleness_sec = max(
                0.0, (now - data.timestamp).total_seconds()
            )
        except Exception as _e:  # noqa: silent-swallow
            # naive vs aware mismatch shouldn't happen (we control both
            # ends), but defend so the read path never crashes.
            logger.debug(f"[COINGLASS] staleness compute skipped: {_e}")
        return data
    
    def get_crowd_metrics(self, symbol: str = "BTC") -> Dict[str, Any]:
        """
        Get crowd metrics for profit_max_adapter integration.
        
        Returns standardized crowd context.
        """
        if not self._last_data:
            return self._get_default_crowd_metrics()
        
        data = self._last_data
        
        # Compute extreme crowding
        crowding = data.crowding_score.get(symbol, 0.0)
        funding_b = data.funding_bias.get(symbol, 0.0)
        extreme_crowding = crowding >= 0.8 or abs(funding_b) >= 0.7
        
        return {
            "funding_bias": data.funding_bias.get(symbol, 0.0),
            "oi_trend": data.oi_trend.get(symbol, 0.0),
            "liquidation_imbalance": data.liquidation_imbalance.get(symbol, 0.0),
            "crowding_score": crowding,
            "extreme_crowding": extreme_crowding,
            # [P287] Per-family carried-forward age (sec at fetch time);
            # 0.0 = fetched fresh this cycle. Lets consumers distinguish a
            # live reading from one surviving an endpoint-family outage.
            "funding_age_sec": data.family_age_sec.get(f"funding:{symbol}", 0.0),
            "oi_age_sec": data.family_age_sec.get(f"open_interest:{symbol}", 0.0),
            "liquidation_age_sec": data.family_age_sec.get(f"liquidations:{symbol}", 0.0),
        }
    
    def _get_default_crowd_metrics(self) -> Dict[str, Any]:
        """Default metrics when no data available."""
        return {
            "funding_bias": 0.0,
            "oi_trend": 0.0,
            "liquidation_imbalance": 0.0,
            "crowding_score": 0.0,
            "extreme_crowding": False,
        }
    
    # =========================================================================
    # REAL API
    # =========================================================================
    
    async def _fetch_real(self) -> CoinglassCrowdData:
        """Fetch real data from Coinglass.

        v2 API structure (verified Feb 2026):
        - /open_interest?symbol=X returns per-exchange list (23 items). Each has:
          openInterest (per exchange, USD), avgFundingRateBySymbol (cross-exchange),
          h4OIChangePercent. Must sum openInterest across exchanges. Requires symbol param.
        - /funding returns ALL symbols in one call. Funding rates nested in
          uMarginList[].rate (per exchange), not at top level.
        - /liquidation v2 endpoint returns empty (code=None).

        Strategy: Per-symbol /open_interest calls get OI (summed) + avgFundingRate.
        /funding supplements with per-exchange detail. Liquidation skipped.
        """
        now = datetime.now(timezone.utc)

        data = CoinglassCrowdData(
            timestamp=now,
            staleness_sec=0.0,
        )

        headers = {
            "coinglassSecret": self.api_key,
            "Accept": "application/json",
        }

        _dns_warned = getattr(self, '_dns_warning_shown', False)
        _errors_this_fetch = 0

        async with create_session() as session:
            # Per-symbol /open_interest calls -> OI (summed) + avgFundingRate
            for symbol in SUPPORTED_SYMBOLS:
                try:
                    oi_data, fr_data = await self._fetch_oi_and_funding(session, headers, symbol)
                    if oi_data:
                        data.open_interest[symbol] = oi_data
                    if fr_data:
                        data.funding[symbol] = fr_data
                except Exception as e:
                    _errors_this_fetch += 1
                    if not _dns_warned:
                        logger.warning(f"[COINGLASS] OI+Funding {symbol}: {e}")

            # Fetch per-exchange funding rates from /funding (supplements avgFR)
            try:
                fr_detailed = await self._fetch_funding_detailed(session, headers)
                for symbol in SUPPORTED_SYMBOLS:
                    if symbol in fr_detailed and symbol not in data.funding:
                        data.funding[symbol] = fr_detailed[symbol]
            except Exception as e:
                _errors_this_fetch += 1
                if not _dns_warned:
                    logger.warning(f"[COINGLASS] Funding detailed: {e}")

            # Liquidation via v3 API (v2 endpoint non-functional)
            try:
                await self._fetch_liquidation_v3(session, data)
            except Exception as e:
                _errors_this_fetch += 1
                if not _dns_warned:
                    logger.warning(f"[COINGLASS] Liquidation v3: {e}")

        if _errors_this_fetch >= 4 and not _dns_warned:
            logger.info(f"[COINGLASS] All {_errors_this_fetch} requests failed - suppressing repeat warnings")
            self._dns_warning_shown = True

        # [P265] A fetch that produced NO raw content at all must not
        # overwrite the cache. Every per-endpoint failure is caught INSIDE
        # this method, so on a full API outage no exception escaped to the
        # P100 stale-fallback in fetch(): a fresh-stamped CoinglassCrowdData
        # full of computed ZEROS (funding_bias/oi_trend/liquidation_imbalance
        # = 0.0, staleness 0.0) replaced the cached good data — the timestamp
        # LIED and even a staleness-aware consumer could not recover. Serve
        # the cache with honest staleness instead; "no data" and "calm
        # market" must never be byte-identical.
        _got_anything = bool(data.open_interest or data.funding
                             or data.liquidations)
        _prev = self._last_data
        if not _got_anything and _prev is not None:
            logger.warning(
                "[COINGLASS] fetch produced NO content (all endpoints "
                "failed) — keeping the previous cache with honest staleness "
                "rather than fabricating fresh zeros")
            return self._refreshed_staleness(_prev) or _prev

        # [P265] Re-arm the failure telemetry on recovery. The one-shot
        # latch used to stay set for the process lifetime, so after one bad
        # 5-minute cycle every subsequent OI/funding/liquidation failure
        # warning was suppressed forever.
        if _got_anything and getattr(self, '_dns_warning_shown', False):
            self._dns_warning_shown = False
            logger.info("[COINGLASS] feed recovered — failure warnings re-armed")

        # [P287] Per-FAMILY carry-forward. The P265 guard above covers only
        # the nothing-at-all outage: a PARTIAL fetch (e.g. OI+funding OK,
        # liquidation v3 down for every symbol) still fell through to
        # _compute_metrics with the failed family EMPTY, which wrote 0.0
        # into its derived metric and replaced the cache FRESH-stamped —
        # "endpoint down" and "calm market" were byte-identical. Carry the
        # previous cached entries forward WITH their original timestamps
        # (age stays honest, exposed via family_age_sec) instead of
        # fabricating neutral. Carried entries never re-enter the trend
        # history deques (a repeated stale value each cycle would collapse
        # the rolling stats — the same corruption one buffer over).
        _carried: set = set()
        if _prev is not None:
            # Typed as generic dicts: each tuple row pairs SAME-family maps,
            # but mypy sees the union across rows and rejects the carry
            # assignment; the family pairing is positional and test-pinned.
            _fam_rows: List[Any] = [
                ("funding", data.funding, _prev.funding),
                ("open_interest", data.open_interest, _prev.open_interest),
                ("liquidations", data.liquidations, _prev.liquidations)]
            for _fam_name, _cur_map, _prev_map in _fam_rows:
                for _sym in SUPPORTED_SYMBOLS:
                    if _sym not in _cur_map and _sym in _prev_map:
                        _cur_map[_sym] = _prev_map[_sym]
                        _carried.add((_fam_name, _sym))
                        _ts = getattr(_prev_map[_sym], "timestamp", None)
                        data.family_age_sec[f"{_fam_name}:{_sym}"] = (
                            max(0.0, (now - _ts).total_seconds())
                            if isinstance(_ts, datetime) else -1.0)
        if _carried:
            logger.warning(
                "[COINGLASS] partial fetch — carried previous data for "
                + ", ".join(sorted(f"{f}:{s}" for f, s in _carried))
                + " (original timestamps kept, ages in family_age_sec; "
                  "NOT fresh-stamped)")

        # Compute derived metrics
        self._compute_metrics(data, carried=_carried)

        self._last_data = data
        self._last_fetch_time = now

        return data

    async def _fetch_oi_and_funding(
        self,
        session: aiohttp.ClientSession,
        headers: Dict,
        symbol: str,
    ) -> tuple:
        """Fetch OI + avgFundingRate for one symbol.

        /open_interest?symbol=X returns per-exchange list. We sum OI across
        exchanges and extract the cross-exchange avgFundingRateBySymbol.

        Returns (OpenInterestData or None, FundingData or None).
        """
        url = f"{self.BASE_URL}/open_interest"
        now = datetime.now(timezone.utc)

        async with session.get(url, headers=headers, params={"symbol": symbol},
                               timeout=aiohttp.ClientTimeout(total=15)) as resp:
            if resp.status != 200:
                # [P38 2026-04-24] Make 429 visible — was silently dropped.
                if resp.status == 429:
                    from data_mgmt.feeds._http import parse_retry_after
                    _retry = parse_retry_after(resp.headers.get("Retry-After"))
                    logger.warning(
                        f"[COINGLASS] {symbol} OI rate-limited (429), "
                        f"Retry-After={_retry}s — data missing this poll cycle"
                    )
                return None, None

            result = await resp.json()
            if str(result.get("code")) != "0":
                logger.warning(
                    f"[COINGLASS] {symbol} OI: code={result.get('code')}, "
                    f"msg={result.get('msg', 'unknown')}"
                )
                return None, None

            items = result.get("data", [])
            if not isinstance(items, list) or not items:
                return None, None

            # [P265] The response's FIRST row is the exchange aggregate
            # (exchangeName="All", probe-verified 2026-08-14) — summing every
            # row therefore DOUBLE-counted total OI (~2x, small enough to
            # pass the $1T sanity bound, so it never surfaced). Prefer the
            # aggregate row; sum only the real exchange rows otherwise.
            _all_row = next(
                (i for i in items
                 if str(i.get("exchangeName", "")).strip().lower() == "all"),
                None)
            if _all_row is not None:
                total_oi = float(_all_row.get("openInterest", 0) or 0)
            else:
                total_oi = sum(float(item.get("openInterest", 0) or 0)
                               for item in items)

            # [FIX-38] Bounds check: OI must be non-negative and sane
            if not np.isfinite(total_oi) or total_oi < 0:
                logger.warning(f"[COINGLASS] {symbol} OI out of bounds: {total_oi}")
                total_oi = 0.0
            elif total_oi > 1e12:  # > $1T is clearly corrupted
                logger.warning(f"[COINGLASS] {symbol} OI suspiciously large: {total_oi}")
                total_oi = 0.0

            # Change fields are the same across all exchange rows for the
            # same symbol - take from first item.
            # [P265] FIELD MISLABELS, probe-verified against live values
            # (2026-08-14: h24Change == oichangePercent == -1.14 with the
            # 12h/4h/1h OI trajectory +0.5/-0.24/-0.04, while
            # h1VolChangePercent/h24VolChangePercent are VOLUME at
            # -0.49/+8.82): change_24h_pct used to carry h4OIChangePercent
            # (the 4-HOUR change — every consumer is calibrated to a 24h
            # scale: sentiment's /10.0 crowding clip, the |x|>8 trigger,
            # squeeze detector, smart beta, the short_bias whale proxy), and
            # change_1h_pct carried h1VolChangePercent — a VOLUME change,
            # not OI. Now: the real 24h and 1h OI changes.
            first = items[0]
            h24_change = float(first.get("h24Change",
                                         first.get("oichangePercent", 0)) or 0)
            h1_change = float(first.get("h1OIChangePercent", 0) or 0)

            # [FIX-38] Bounds check: percentage changes
            h24_change = np.clip(h24_change, -100.0, 100.0) if np.isfinite(h24_change) else 0.0
            h1_change = np.clip(h1_change, -100.0, 100.0) if np.isfinite(h1_change) else 0.0

            oi_data = OpenInterestData(
                symbol=symbol,
                open_interest_usd=total_oi,
                open_interest_btc=total_oi / 100000,
                change_24h_pct=h24_change,
                change_1h_pct=h1_change,
                timestamp=now,
            )

            # Funding rate (OI-weighted average across exchanges, pre-computed by API)
            fr_data = None
            avg_fr = first.get("avgFundingRateBySymbol")
            if avg_fr is not None:
                avg_fr_f = float(avg_fr)
                # [P287] An extreme print is CLAMPED to the ±1%/8h bound,
                # never zeroed: converting a squeeze-level funding rate to
                # 0.0 read as "no crowding" exactly when the crowding
                # detector matters most. kraken_futures_feed [FIX-39] clamps
                # the same bound — the two feeds now agree on out-of-range
                # semantics. Non-finite stays "no data" (fr_data None), not
                # a fabricated neutral.
                if not np.isfinite(avg_fr_f):
                    logger.warning(f"[COINGLASS] {symbol} funding rate non-finite: {avg_fr!r} — no funding datum this fetch")
                else:
                    if abs(avg_fr_f) > 0.01:
                        logger.warning(f"[COINGLASS] {symbol} funding rate {avg_fr_f:.6f} beyond ±1%/8h — clamping (extreme crowding print, NOT neutral)")
                        avg_fr_f = max(-0.01, min(0.01, avg_fr_f))
                    fr_data = FundingData(
                        symbol=symbol,
                        rate=avg_fr_f,
                        predicted_rate=None,
                        interval_hours=8,
                        exchange="aggregate",
                        timestamp=now,
                    )

        return oi_data, fr_data

    async def _fetch_funding_detailed(
        self,
        session: aiohttp.ClientSession,
        headers: Dict,
    ) -> Dict[str, FundingData]:
        """Fetch per-exchange funding rates from /funding endpoint.

        v2 /funding returns ALL symbols. Each item has uMarginList[] with
        per-exchange {rate, exchangeName, status}. We compute a simple
        average of active exchanges.
        """
        url = f"{self.BASE_URL}/funding"
        fr_map: Dict[str, FundingData] = {}
        now = datetime.now(timezone.utc)

        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            if resp.status != 200:
                if resp.status == 429:
                    from data_mgmt.feeds._http import parse_retry_after
                    _retry = parse_retry_after(resp.headers.get("Retry-After"))
                    logger.warning(
                        f"[COINGLASS] funding rate-limited (429), "
                        f"Retry-After={_retry}s — data missing this poll cycle"
                    )
                return fr_map

            result = await resp.json()
            if str(result.get("code")) != "0":
                logger.warning(
                    f"[COINGLASS] funding: code={result.get('code')}, "
                    f"msg={result.get('msg', 'unknown')}"
                )
                return fr_map

            items = result.get("data", [])
            if not isinstance(items, list):
                return fr_map

            for item in items:
                symbol = item.get("symbol", "")
                if symbol not in SUPPORTED_SYMBOLS:
                    continue

                # Extract rates from uMarginList (USDT-margined perps)
                u_margin = item.get("uMarginList", []) or []
                active_rates = [
                    x.get("rate", 0) for x in u_margin
                    if x.get("status") == 1 and x.get("rate") is not None
                ]

                # [FIX-38] Filter out non-finite rates; [P287] extreme
                # finite rates are CLAMPED to ±1%/8h, not dropped —
                # silently dropping them removed exactly the exchanges
                # showing squeeze-level crowding from the average.
                active_rates = [
                    max(-0.01, min(0.01, float(r))) for r in active_rates
                    if isinstance(r, (int, float)) and np.isfinite(r)
                ]

                if active_rates:
                    avg_rate = sum(active_rates) / len(active_rates)
                    fr_map[symbol] = FundingData(
                        symbol=symbol,
                        rate=avg_rate,
                        predicted_rate=None,
                        interval_hours=8,
                        exchange=f"avg_{len(active_rates)}_exchanges",
                        timestamp=now,
                    )

        return fr_map
    
    async def _fetch_liquidation_v3(
        self,
        session: aiohttp.ClientSession,
        data: CoinglassCrowdData,
    ) -> None:
        """Fetch liquidation data from v3 API.

        v3 /api/futures/liquidation/exchange-list?symbol=X&range=24h
        Returns per-exchange liquidation amounts (long/short/total in USD).
        Header: CG-API-KEY (not coinglassSecret).
        """
        v3_headers = {
            "CG-API-KEY": self.api_key,
            "accept": "application/json",
        }
        now = datetime.now(timezone.utc)

        for symbol in SUPPORTED_SYMBOLS:
            try:
                url = f"{self.V3_BASE_URL}/futures/liquidation/exchange-list"
                async with session.get(
                    url, headers=v3_headers,
                    params={"symbol": symbol, "range": "24h"},
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    if resp.status != 200:
                        if resp.status == 429:
                            from data_mgmt.feeds._http import parse_retry_after
                            _retry = parse_retry_after(resp.headers.get("Retry-After"))
                            logger.warning(
                                f"[COINGLASS] {symbol} liquidation rate-limited (429), "
                                f"Retry-After={_retry}s — data missing this cycle"
                            )
                        continue

                    result = await resp.json()
                    if str(result.get("code")) != "0" or not result.get("data"):
                        logger.warning(
                            f"[COINGLASS] liquidation: code={result.get('code')}, "
                            f"msg={result.get('msg', 'unknown')}"
                        )
                        continue

                    items = result["data"]
                    if not isinstance(items, list):
                        continue

                    # Sum across exchanges
                    total_long = 0.0
                    total_short = 0.0
                    largest = 0.0
                    for item in items:
                        long_usd = float(item.get("longLiquidationUsd", 0) or 0)
                        short_usd = float(item.get("shortLiquidationUsd", 0) or 0)
                        total_long += long_usd
                        total_short += short_usd
                        largest = max(largest, long_usd, short_usd)

                    total = total_long + total_short

                    # Bounds check
                    if not np.isfinite(total) or total < 0:
                        continue
                    if total > 1e12:  # > $1T clearly corrupted
                        logger.warning(f"[COINGLASS] {symbol} liquidation suspiciously large: {total}")
                        continue

                    data.liquidations[symbol] = LiquidationData(
                        symbol=symbol,
                        long_liquidations_24h=total_long,
                        short_liquidations_24h=total_short,
                        total_liquidations_24h=total,
                        largest_single_liquidation=largest,
                        timestamp=now,
                    )

            except Exception as e:
                # [P100 2026-04-27] Promoted DEBUG → WARNING. Liquidation
                # v3 per-symbol exceptions (auth fail, parse error, API
                # timeout) were invisible in production at INFO level.
                # Asymmetric vs other endpoints in this file which already
                # log at WARNING. Liquidation imbalance feeds the crowding
                # detector + position sizer; silent per-symbol miss leaks
                # stale data downstream.
                # [P287] Message states what actually happens: the symbol is
                # simply ABSENT from this fetch; the P287 carry-forward in
                # _fetch_real then keeps the previous cached row (if any)
                # with its honest age. The old text claimed "using stale
                # data" while the code fabricated a fresh zero.
                logger.warning(
                    f"[COINGLASS] Liquidation v3 {symbol} FAILED "
                    f"({type(e).__name__}: {e}); no {symbol} liquidation row "
                    f"this fetch — previous cached row (if any) carried "
                    f"forward with honest age."
                )

    # =========================================================================
    # METRICS COMPUTATION
    # =========================================================================
    
    def _compute_metrics(self, data: CoinglassCrowdData, carried=None):
        """Compute derived crowding metrics.

        [P287] `carried` is the set of (family, symbol) pairs whose entries
        were carried forward from the previous cache (endpoint family failed
        this fetch). Their metrics are still computed from the carried data,
        but they must NOT re-enter the trend history deques — appending the
        same stale value each 5-min cycle would collapse the rolling stats.
        """
        carried = carried or set()
        for symbol in SUPPORTED_SYMBOLS:
            # Funding bias
            if symbol in data.funding:
                rate = data.funding[symbol].rate
                # Normalize: positive funding = longs pay shorts = crowded longs
                funding_bias = np.clip(rate / FUNDING_NORM, -1.0, 1.0)
                data.funding_bias[symbol] = funding_bias

                # Update history (deque maxlen auto-caps)
                if ("funding", symbol) not in carried:
                    self._funding_history[symbol].append(rate)
            else:
                data.funding_bias[symbol] = 0.0

            # OI trend
            if symbol in data.open_interest:
                oi_change = data.open_interest[symbol].change_24h_pct / 100
                # Normalize: rising OI = more positions being opened
                data.oi_trend[symbol] = np.clip(oi_change / 0.1, -1.0, 1.0)

                # Update history (deque maxlen auto-caps)
                if ("open_interest", symbol) not in carried:
                    self._oi_history[symbol].append(data.open_interest[symbol].open_interest_usd)
            else:
                data.oi_trend[symbol] = 0.0
            
            # Liquidation imbalance
            if symbol in data.liquidations:
                liq = data.liquidations[symbol]
                total = liq.total_liquidations_24h
                if total > 0:
                    # Positive = more longs liquidated
                    imbalance = (liq.long_liquidations_24h - liq.short_liquidations_24h) / total
                    data.liquidation_imbalance[symbol] = np.clip(imbalance, -1.0, 1.0)
                else:
                    data.liquidation_imbalance[symbol] = 0.0
            else:
                data.liquidation_imbalance[symbol] = 0.0
            
            # Crowding score (composite)
            # High crowding = high funding + rising OI + one-sided liquidations
            funding_contrib = abs(data.funding_bias.get(symbol, 0))
            oi_contrib = max(0, data.oi_trend.get(symbol, 0))  # Only rising OI
            liq_contrib = abs(data.liquidation_imbalance.get(symbol, 0))
            
            crowding = (funding_contrib * 0.4 + oi_contrib * 0.3 + liq_contrib * 0.3)
            data.crowding_score[symbol] = np.clip(crowding, 0.0, 1.0)
    
    # =========================================================================
    # MOCK DATA
    # =========================================================================
    
    async def _fetch_mock(self) -> CoinglassCrowdData:
        """
        Neutral-safe fallback when Coinglass API is unavailable.

        [PATCH-7b] Replaced random.uniform with neutral constants.
        Root cause: random mock data injected noise into funding/OI/liquidation
        signals, causing non-deterministic crowd-metric decisions on feed failure.
        """
        await asyncio.sleep(0.1)

        now = datetime.now(timezone.utc)
        data = CoinglassCrowdData(timestamp=now, staleness_sec=0.0)

        for symbol in SUPPORTED_SYMBOLS:
            # Neutral funding (0.0 = no directional bias)
            data.funding[symbol] = FundingData(
                symbol=symbol,
                rate=0.0,
                predicted_rate=None,
                interval_hours=8,
                exchange="mock",
                timestamp=now,
            )

            # Neutral OI (base value, zero change = no signal)
            base_oi = {"BTC": 30e9, "ETH": 15e9, "SOL": 3e9}[symbol]
            data.open_interest[symbol] = OpenInterestData(
                symbol=symbol,
                open_interest_usd=base_oi,
                open_interest_btc=base_oi / 100000,
                change_24h_pct=0.0,
                change_1h_pct=0.0,
                timestamp=now,
            )

            # Neutral liquidations (balanced = no directional signal)
            data.liquidations[symbol] = LiquidationData(
                symbol=symbol,
                long_liquidations_24h=0.0,
                short_liquidations_24h=0.0,
                total_liquidations_24h=0.0,
                largest_single_liquidation=0.0,
                timestamp=now,
            )

        self._compute_metrics(data)
        self._last_data = data
        self._last_fetch_time = now

        return data


# =============================================================================
# SINGLETON
# =============================================================================

_coinglass_feed_instance: Optional[CoinglassFeed] = None


def get_coinglass_feed(
    api_key: Optional[str] = None,
    mock_mode: bool = False,
) -> CoinglassFeed:
    """Get or create CoinglassFeed singleton."""
    global _coinglass_feed_instance
    if _coinglass_feed_instance is None:
        _coinglass_feed_instance = CoinglassFeed(
            api_key=api_key,
            mock_mode=mock_mode,
        )
    return _coinglass_feed_instance


def reset_coinglass_feed():
    """Reset singleton."""
    global _coinglass_feed_instance
    if _coinglass_feed_instance:
        _coinglass_feed_instance.stop()
    _coinglass_feed_instance = None


# =============================================================================
# TESTS
# =============================================================================

if __name__ == "__main__":
    import asyncio
    
    async def test():
        print("=" * 70)
        print("Coinglass Feed Test")
        print("=" * 70)
        
        feed = CoinglassFeed(mock_mode=True)
        data = await feed.fetch()
        
        print(f"\nFunding Bias: {data.funding_bias}")
        print(f"OI Trend: {data.oi_trend}")
        print(f"Liquidation Imbalance: {data.liquidation_imbalance}")
        print(f"Crowding Score: {data.crowding_score}")
        
        print("\nBTC Crowd Metrics:")
        metrics = feed.get_crowd_metrics("BTC")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        
        print("\nON Test passed!")
    
    asyncio.run(test())
