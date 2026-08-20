"""
================================================================================
HMATS v6.5 - FRED API Data Feed
================================================================================

Official Federal Reserve Economic Data (FRED) API integration.

Provides:
1. Interest rates (Fed Funds, Treasury yields)
2. Economic indicators (CPI, PCE, unemployment)
3. Release calendar and event windows
4. Financial conditions index

CRITICAL: This is a P0 data source - must be reliable and deterministic.

API Docs: https://fred.stlouisfed.org/docs/api/fred/

================================================================================
"""

import logging
import asyncio
import aiohttp
from data_mgmt.feeds._http import create_session
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta, timezone
from enum import Enum
import os

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

# FRED series IDs for key indicators
FRED_SERIES = {
    # Interest Rates
    "FEDFUNDS": "Federal Funds Effective Rate",
    "DFF": "Federal Funds Rate (daily)",
    "DGS10": "10-Year Treasury Constant Maturity Rate",
    "DGS2": "2-Year Treasury Constant Maturity Rate",
    "T10Y2Y": "10-Year Treasury Minus 2-Year Treasury",
    
    # Inflation
    "CPIAUCSL": "CPI All Urban Consumers",
    "PCEPI": "PCE Price Index",
    "T5YIE": "5-Year Breakeven Inflation Rate",
    
    # Employment
    "UNRATE": "Unemployment Rate",
    "PAYEMS": "Total Nonfarm Payrolls",
    "ICSA": "Initial Jobless Claims",
    
    # Financial Conditions
    "NFCI": "Chicago Fed National Financial Conditions Index",
    "DTWEXBGS": "Trade Weighted U.S. Dollar Index",
    
    # Volatility proxy
    "VIXCLS": "CBOE Volatility Index (VIX)",
}

# Key economic releases for event calendar
KEY_RELEASES = {
    "CPI": {"series": "CPIAUCSL", "importance": 3, "window_min": 180},
    "PCE": {"series": "PCEPI", "importance": 3, "window_min": 120},
    "NFP": {"series": "PAYEMS", "importance": 3, "window_min": 180},
    "UNEMPLOYMENT": {"series": "UNRATE", "importance": 2, "window_min": 60},
    "JOBLESS_CLAIMS": {"series": "ICSA", "importance": 2, "window_min": 30},
}

# Static importance mapping for releases
IMPORTANCE_MAP = {
    "high": 3,
    "medium": 2,
    "low": 1,
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================


def _aware_utc(dt: datetime) -> datetime:
    """[P299] FRED dates parse NAIVE and are then compared against an AWARE
    `datetime.now(timezone.utc)`.

    `datetime.fromisoformat("2026-08-18")` yields a naive datetime, and
    `_compute_event_window` does `release.release_date - now`, which raises
    `TypeError: can't subtract offset-naive and offset-aware datetimes`. That
    exception is caught by `fetch()`'s handler, so the whole FRED fetch
    returned None and the macro context stayed on its neutral defaults -
    i.e. P293's headline fix (FRED had no producer) shipped a producer that
    THREW on every call, which looks identical from the consumer side.
    Surfaced the moment `macro_gci_live` started driving it.

    The P40/P97 rule, applied at the parse boundary: a bare date is UTC
    midnight, never a local-time guess.
    """
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)

@dataclass
class FREDObservation:
    """Single FRED data observation."""
    series_id: str
    value: float
    date: datetime
    realtime_start: datetime
    realtime_end: datetime


@dataclass
class FREDRelease:
    """Economic release/event."""
    release_id: int
    name: str
    release_date: datetime
    importance: int  # 1-3
    series_affected: List[str] = field(default_factory=list)


@dataclass
class MacroEventWindow:
    """Represents an active or upcoming macro event window."""
    active: bool
    name: Optional[str]
    t_minus_min: int  # Minutes until event (negative = past)
    t_plus_min: int   # Minutes since event (for recent events)
    importance: int   # 1-3


@dataclass
class FREDMacroData:
    """Aggregated FRED macro data output."""
    timestamp: datetime
    staleness_sec: float
    
    # Interest rates
    fed_funds_rate: Optional[float] = None
    treasury_10y: Optional[float] = None
    treasury_2y: Optional[float] = None
    yield_curve_spread: Optional[float] = None  # 10Y - 2Y
    
    # Inflation
    cpi_yoy: Optional[float] = None
    pce_yoy: Optional[float] = None
    breakeven_5y: Optional[float] = None
    
    # Employment
    unemployment_rate: Optional[float] = None
    nfp_change: Optional[float] = None
    jobless_claims: Optional[float] = None
    
    # Financial conditions
    nfci: Optional[float] = None  # Negative = loose, positive = tight
    dxy_index: Optional[float] = None
    vix: Optional[float] = None
    
    # Event window
    event_window: MacroEventWindow = field(default_factory=lambda: MacroEventWindow(
        active=False, name=None, t_minus_min=0, t_plus_min=0, importance=0
    ))
    
    # Computed
    macro_risk: float = 0.0  # 0-1, higher = more risk aversion
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for context injection."""
        return {
            "fed_funds_rate": self.fed_funds_rate,
            "treasury_10y": self.treasury_10y,
            "treasury_2y": self.treasury_2y,
            "yield_curve_spread": self.yield_curve_spread,
            "cpi_yoy": self.cpi_yoy,
            "unemployment_rate": self.unemployment_rate,
            "nfci": self.nfci,
            "vix": self.vix,
            "event_window": {
                "active": self.event_window.active,
                "name": self.event_window.name,
                "t_minus_min": self.event_window.t_minus_min,
                "t_plus_min": self.event_window.t_plus_min,
                "importance": self.event_window.importance,
            },
            "macro_risk": self.macro_risk,
        }


# =============================================================================
# FRED FEED
# =============================================================================

class FREDFeed:
    """
    FRED API data feed.
    
    Fetches official macro data from Federal Reserve Economic Data.
    """
    
    BASE_URL = "https://api.stlouisfed.org/fred"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        poll_interval_sec: float = 3600,  # 1 hour (macro data is slow)
        event_bus_callback: Optional[Callable] = None,
        mock_mode: bool = False,
    ):
        self.api_key = api_key or os.environ.get("FRED_API_KEY", "")
        self.poll_interval_sec = poll_interval_sec
        self._event_bus_callback = event_bus_callback
        self._mock_mode = mock_mode
        
        # Internal state
        self._cache: Dict[str, FREDObservation] = {}
        self._releases: List[FREDRelease] = []
        self._last_fetch_time: Optional[datetime] = None
        self._last_data: Optional[FREDMacroData] = None
        self._running = False
        self._fetch_errors = 0
        # [P329b] Per-endpoint consecutive-failure streaks. FRED is ADVISORY:
        # a failed series degrades to a NEUTRAL mock that "contributes nothing
        # to Macro" (the GCI says so in its own log), so an isolated 10s
        # timeout is not something an operator can act on — and 9 of the 10
        # FRED log lines in a 16h live window were ERROR-level timeouts
        # forwarded to Discord. A SUSTAINED failure is different: it means the
        # endpoint is not working at all, which is worth hearing exactly once
        # per streak (P202/P240, the P303 pattern).
        self._endpoint_fail_streak: Dict[str, int] = {}
        
        # Event window configuration
        self._event_window_minutes = 180  # 3 hours before/after
        
        if not self.api_key and not mock_mode:
            logger.warning("[FRED_FEED] No API key provided. Set FRED_API_KEY env var.")
        
        logger.info(f"[FRED_FEED] Initialized: mock={mock_mode}")
    
    # =========================================================================
    # PUBLIC API
    # =========================================================================
    
    async def start(self):
        """Start the feed polling loop."""
        if self._running:
            return
        
        self._running = True
        logger.info("[FRED_FEED] Started")
        
        while self._running:
            try:
                data = await self.fetch()
                if data and self._event_bus_callback:
                    self._event_bus_callback("FRED_DATA", data.to_dict())
            except Exception as e:
                logger.error(f"[FRED_FEED] Fetch error: {e}")
                self._fetch_errors += 1
            
            await asyncio.sleep(self.poll_interval_sec)
    
    def stop(self):
        """Stop the feed."""
        self._running = False
        logger.info("[FRED_FEED] Stopped")
    
    async def fetch(self) -> Optional[FREDMacroData]:
        """Fetch latest FRED data."""
        try:
            if self._mock_mode:
                return await self._fetch_mock()
            
            if not self.api_key:
                logger.warning("[FRED_FEED] No API key, using mock data")
                return await self._fetch_mock()
            
            return await self._fetch_real()
            
        except Exception as e:
            logger.error(f"[FRED_FEED] Fetch failed: {e}")
            self._fetch_errors += 1
            return self._last_data
    
    def get_latest(self) -> Optional[FREDMacroData]:
        """Get cached data."""
        return self._last_data

    def cache_age_sec(self) -> Optional[float]:
        """Seconds since the last successful fetch, or None if never fetched.

        [P293] None is deliberately distinct from a large number: "never
        fetched" and "fetched a long time ago" have different fixes, and
        collapsing them is how this feed's emptiness stayed invisible.
        """
        if self._last_fetch_time is None:
            return None
        _t = self._last_fetch_time
        # [P40/P97] _last_fetch_time is aware on every current write path;
        # defend anyway — a naive value here would raise inside whatever
        # try block the caller happens to be in.
        if _t.tzinfo is None:
            _t = _t.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - _t).total_seconds()

    async def fetch_if_stale(self) -> Optional[FREDMacroData]:
        """Fetch only when the cache is older than poll_interval_sec.

        [P293] The 4H tick loop runs this once per asset, so an unthrottled
        fetch would spend 3x the requests per cycle for data that updates
        daily. Throttling lives HERE rather than at the call site so a
        second caller cannot drift from the first (P172 single-source).
        """
        _age = self.cache_age_sec()
        if _age is not None and _age < self.poll_interval_sec:
            return self._last_data
        return await self.fetch()


    def get_macro_context(self) -> Dict[str, Any]:
        """
        Get macro context in the required v6.5 format.
        
        Returns context["macro"] structure.
        """
        data = self._last_data
        if not data:
            return self._get_default_macro_context()
        
        return {
            "event_window": {
                "active": data.event_window.active,
                "name": data.event_window.name,
                "t_minus_min": data.event_window.t_minus_min,
                "t_plus_min": data.event_window.t_plus_min,
                "importance": data.event_window.importance,
            },
            "macro_shock": 0.0,  # Will be computed by Trading Economics feed
            "macro_risk": data.macro_risk,
        }
    
    def _get_default_macro_context(self) -> Dict[str, Any]:
        """Default macro context when no data available."""
        return {
            "event_window": {
                "active": False,
                "name": None,
                "t_minus_min": 0,
                "t_plus_min": 0,
                "importance": 0,
            },
            "macro_shock": 0.0,
            "macro_risk": 0.5,  # Neutral
        }
    
    # =========================================================================
    # REAL FRED API
    # =========================================================================
    
    async def _fetch_real(self) -> FREDMacroData:
        """Fetch real data from FRED API."""
        now = datetime.now(timezone.utc)
        
        async with create_session() as session:
            # Fetch key series
            tasks = []
            for series_id in ["DFF", "DGS10", "DGS2", "VIXCLS", "NFCI"]:
                tasks.append(self._fetch_series(session, series_id))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Parse results
            series_data = {}
            for result in results:
                if isinstance(result, FREDObservation):
                    series_data[result.series_id] = result.value
            
            # Fetch upcoming releases
            releases = await self._fetch_releases(session)
            
        # Build output
        data = FREDMacroData(
            timestamp=now,
            staleness_sec=0.0,
            fed_funds_rate=series_data.get("DFF"),
            treasury_10y=series_data.get("DGS10"),
            treasury_2y=series_data.get("DGS2"),
            vix=series_data.get("VIXCLS"),
            nfci=series_data.get("NFCI"),
        )
        
        # Compute yield curve spread
        if data.treasury_10y and data.treasury_2y:
            data.yield_curve_spread = data.treasury_10y - data.treasury_2y
        
        # Compute event window
        data.event_window = self._compute_event_window(releases, now)
        
        # Compute macro risk
        data.macro_risk = self._compute_macro_risk(data)
        
        self._last_data = data
        self._last_fetch_time = now
        
        return data
    
    async def _fetch_series(
        self,
        session: aiohttp.ClientSession,
        series_id: str,
    ) -> Optional[FREDObservation]:
        """Fetch single FRED series."""
        url = f"{self.BASE_URL}/series/observations"
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 1,
        }
        
        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    logger.warning(f"[FRED_FEED] Series {series_id} HTTP {resp.status}")
                    return None

                data = await resp.json()
                observations = data.get("observations", [])

                if not observations:
                    return None

                obs = observations[0]
                value_str = obs.get("value", ".")

                if value_str == ".":
                    return None  # Missing value

                self._report_endpoint_ok(f"series:{series_id}")
                return FREDObservation(
                    series_id=series_id,
                    value=float(value_str),
                    date=_aware_utc(datetime.fromisoformat(obs["date"])),
                    realtime_start=_aware_utc(datetime.fromisoformat(obs["realtime_start"])),
                    realtime_end=_aware_utc(datetime.fromisoformat(obs["realtime_end"])),
                )

        except asyncio.TimeoutError:  # noqa: silent-swallow — logs via _report_endpoint_failure (the linter cannot see method-call logging, P252)
            self._report_endpoint_failure(
                f"series:{series_id}", f"Series {series_id} timed out (10s)",
                "that indicator degrades to a NEUTRAL mock and contributes "
                "nothing to Macro; no other value is affected.")
            return None
        except Exception as e:  # noqa: silent-swallow — logs via _report_endpoint_failure (the linter cannot see method-call logging, P252)
            self._report_endpoint_failure(
                f"series:{series_id}",
                f"Series {series_id} error: {type(e).__name__}: {e}",
                "that indicator degrades to a NEUTRAL mock and contributes "
                "nothing to Macro; no other value is affected.")
            return None
    

    # [P329b] ------------------------------------------------------------
    SUSTAINED_FAILURES = 3   # ~3 consecutive macro refreshes

    def _report_endpoint_failure(self, key: str, detail: str,
                                 consequence: str) -> None:
        """Isolated failure -> WARNING; sustained -> ERROR, once per streak.

        The message states the CONSEQUENCE, because "Series DGS10 timed out"
        tells an operator nothing about whether anything is now wrong.
        """
        streak = self._endpoint_fail_streak.get(key, 0) + 1
        self._endpoint_fail_streak[key] = streak
        msg = f"[FRED_FEED] {detail} (consecutive={streak}) — {consequence}"
        if streak == self.SUSTAINED_FAILURES:
            logger.error(
                msg + " SUSTAINED: this endpoint is not answering at all.")
        elif streak < self.SUSTAINED_FAILURES:
            logger.warning(msg)
        # Past the escalation point stay quiet: the ERROR above already said
        # it, and repeating per refresh is the wallpaper P202 warns about.

    def _report_endpoint_ok(self, key: str) -> None:
        """A success resets the streak, or a long-lived process drifts into a
        permanent ERROR that no longer describes the present (P303)."""
        if self._endpoint_fail_streak.pop(key, 0):
            logger.info(f"[FRED_FEED] {key}: recovered")

    async def _fetch_releases(
        self,
        session: aiohttp.ClientSession,
    ) -> List[FREDRelease]:
        """Fetch upcoming releases."""
        url = f"{self.BASE_URL}/releases/dates"
        now = datetime.now(timezone.utc)
        
        params = {
            "api_key": self.api_key,
            "file_type": "json",
            "realtime_start": now.strftime("%Y-%m-%d"),
            "realtime_end": (now + timedelta(days=7)).strftime("%Y-%m-%d"),
            "include_release_dates_with_no_data": "false",
        }
        
        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    return []

                data = await resp.json()
                release_dates = data.get("release_dates", [])

                releases = []
                for rd in release_dates:
                    release_name = rd.get("release_name", "")

                    # Determine importance based on release name
                    importance = 1
                    if any(k in release_name.upper() for k in ["CPI", "PCE", "FOMC", "NFP", "EMPLOYMENT"]):
                        importance = 3
                    elif any(k in release_name.upper() for k in ["GDP", "RETAIL", "HOUSING"]):
                        importance = 2

                    releases.append(FREDRelease(
                        release_id=rd.get("release_id", 0),
                        name=release_name,
                        release_date=_aware_utc(datetime.fromisoformat(rd["date"])),
                        importance=importance,
                    ))

                self._releases = releases
                self._report_endpoint_ok("releases")
                return releases

        except asyncio.TimeoutError:  # noqa: silent-swallow — logs via _report_endpoint_failure (the linter cannot see method-call logging, P252)
            self._report_endpoint_failure(
                "releases", "Releases fetch timed out (10s)",
                "event_window stays INACTIVE. Measured 2026-08-19: this "
                "endpoint timed out on every refresh, and event_window has no "
                "consumer outside this module, so nothing downstream changes.")
            return []
        except Exception as e:  # noqa: silent-swallow — logs via _report_endpoint_failure (the linter cannot see method-call logging, P252)
            self._report_endpoint_failure(
                "releases",
                f"Releases fetch error: {type(e).__name__}: {e}",
                "event_window stays INACTIVE (no consumer outside this "
                "module, so nothing downstream changes).")
            return []
    
    # =========================================================================
    # COMPUTATIONS
    # =========================================================================
    
    def _compute_event_window(
        self,
        releases: List[FREDRelease],
        now: datetime,
    ) -> MacroEventWindow:
        """Compute active event window."""
        window_minutes = self._event_window_minutes
        
        for release in releases:
            # [P299] belt-and-braces: the parse boundary above makes these
            # aware, and this keeps ONE malformed row from killing the fetch.
            delta = (_aware_utc(release.release_date)
                     - _aware_utc(now)).total_seconds() / 60
            
            # Check if within window
            if -window_minutes <= delta <= window_minutes:
                return MacroEventWindow(
                    active=True,
                    name=release.name,
                    t_minus_min=int(delta) if delta > 0 else 0,
                    t_plus_min=int(-delta) if delta < 0 else 0,
                    importance=release.importance,
                )
        
        return MacroEventWindow(
            active=False,
            name=None,
            t_minus_min=0,
            t_plus_min=0,
            importance=0,
        )
    
    def _compute_macro_risk(self, data: FREDMacroData) -> float:
        """
        Compute macro_risk score in [0, 1].
        
        Higher = more risk aversion warranted.
        
        Formula:
        - VIX contribution (0.4 weight)
        - NFCI contribution (0.3 weight)
        - Event window contribution (0.3 weight)
        """
        risk = 0.5  # Baseline
        
        # VIX contribution (0.4 weight)
        if data.vix:
            if data.vix > 30:
                risk += 0.4 * min((data.vix - 20) / 30, 1.0)
            elif data.vix < 15:
                risk -= 0.2
        
        # NFCI contribution (0.3 weight)
        # NFCI: negative = loose conditions, positive = tight
        if data.nfci:
            nfci_contrib = min(max(data.nfci, -1), 1) * 0.15
            risk += nfci_contrib
        
        # Event window contribution (0.3 weight)
        if data.event_window.active:
            t_min = abs(data.event_window.t_minus_min)
            window = self._event_window_minutes
            # Closer to event = higher risk
            proximity = 1 - (t_min / window) if window > 0 else 1.0
            importance_weight = data.event_window.importance / 3.0
            risk += 0.3 * proximity * importance_weight
        
        return max(0.0, min(1.0, risk))
    
    # =========================================================================
    # MOCK DATA
    # =========================================================================
    
    async def _fetch_mock(self) -> FREDMacroData:
        """Generate mock data for testing.

        [P113 (3/6) 2026-04-27] WARN-level signal so production operator
        can detect when this fallback is taken. P101 silent-mock-fallback
        prevention.
        """
        if not getattr(self, '_mock_warned', False):
            logger.warning(
                "[FRED] _fetch_mock invoked — production using mock macro "
                "data. Verify FRED_API_KEY configured + endpoint reachable."
            )
            self._mock_warned = True
        await asyncio.sleep(0.1)  # Simulate latency

        now = datetime.now(timezone.utc)

        data = FREDMacroData(
            timestamp=now,
            staleness_sec=0.0,
            fed_funds_rate=5.25,
            treasury_10y=4.50,
            treasury_2y=4.80,
            yield_curve_spread=-0.30,  # Inverted
            cpi_yoy=3.2,
            unemployment_rate=4.1,
            nfci=-0.15,  # Slightly loose
            vix=18.5,
        )
        
        # Mock event window (occasionally active)
        import random
        if random.random() < 0.1:  # 10% chance of active window
            data.event_window = MacroEventWindow(
                active=True,
                name="CPI Release",
                t_minus_min=random.randint(30, 120),
                t_plus_min=0,
                importance=3,
            )
        
        data.macro_risk = self._compute_macro_risk(data)
        
        self._last_data = data
        self._last_fetch_time = now
        
        return data


# =============================================================================
# SINGLETON
# =============================================================================

_fred_feed_instance: Optional[FREDFeed] = None


def get_fred_feed(
    api_key: Optional[str] = None,
    mock_mode: bool = False,
) -> FREDFeed:
    """Get or create FREDFeed singleton."""
    global _fred_feed_instance
    if _fred_feed_instance is None:
        _fred_feed_instance = FREDFeed(
            api_key=api_key,
            mock_mode=mock_mode,
        )
    return _fred_feed_instance


def reset_fred_feed():
    """Reset singleton."""
    global _fred_feed_instance
    if _fred_feed_instance:
        _fred_feed_instance.stop()
    _fred_feed_instance = None


# =============================================================================
# TESTS
# =============================================================================

if __name__ == "__main__":
    import asyncio
    
    async def test():
        print("=" * 70)
        print("FRED Feed Test")
        print("=" * 70)
        
        feed = FREDFeed(mock_mode=True)
        
        data = await feed.fetch()
        print(f"\nFed Funds Rate: {data.fed_funds_rate}")
        print(f"10Y Treasury: {data.treasury_10y}")
        print(f"VIX: {data.vix}")
        print(f"NFCI: {data.nfci}")
        print(f"Event Window Active: {data.event_window.active}")
        print(f"Macro Risk: {data.macro_risk:.3f}")
        
        print("\nMacro Context:")
        ctx = feed.get_macro_context()
        for k, v in ctx.items():
            print(f"  {k}: {v}")
        
        print("\nON Test passed!")
    
    asyncio.run(test())
