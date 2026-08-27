"""
GlobalContextInformer - Macro & Institutional Flow Monitor
===========================================================

Integrates global macro indicators and institutional flow data to provide
informational edge for the HMATS v3.1 trading system.

Data Sources:
- DXY (US Dollar Index) via yfinance
- US10Y (10-Year Treasury Yield) via yfinance
- S&P500 via yfinance
- Spot BTC/ETH ETF Flows (IBIT, FBTC, ETHA, etc.)

Logic:
- Calculate 30-day dynamic correlation coefficients
- Trigger Macro-Regime caution flags on DXY breakout (>1σ)
- Trigger risk-off signal on 3+ consecutive days of ETF outflows

Update Frequency: 1 hour (low-frequency macro loop)

Hardware Target: MSI Titan 18 HX (24-Core Ultra 9-285HX)
Author: Senior Quant Research & ML Systems Architect
Version: 3.1.1
"""

import os
import asyncio
import logging
import time
import json
import hashlib
import threading
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np

# Optional imports
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    yf = None
    YFINANCE_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    PANDAS_AVAILABLE = False

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    aiohttp = None
    AIOHTTP_AVAILABLE = False

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    stats = None
    SCIPY_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('GlobalContextInformer')


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS & CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Macro Tickers
MACRO_TICKERS = {
    'DXY': 'DX-Y.NYB',      # US Dollar Index
    'US10Y': '^TNX',        # 10-Year Treasury Yield
    'SPX': '^GSPC',         # S&P 500
    'VIX': '^VIX',          # Volatility Index
    'GOLD': 'GC=F',         # Gold Futures
    'HY_OAS': 'HYOAS-FRED',  # [P393] HY credit spread - FRED-only (no yf symbol)
}

# Spot Crypto ETFs
SPOT_ETF_TICKERS = {
    'BTC': [
        'IBIT',   # iShares Bitcoin Trust
        'FBTC',   # Fidelity Wise Origin Bitcoin
        'GBTC',   # Grayscale Bitcoin Trust
        'ARKB',   # ARK 21Shares Bitcoin ETF
        'BITB',   # Bitwise Bitcoin ETF
        'HODL',   # VanEck Bitcoin Trust
        'BRRR',   # Valkyrie Bitcoin Fund
        'BTCW',   # WisdomTree Bitcoin Fund
    ],
    'ETH': [
        'ETHA',   # iShares Ethereum Trust
        'FETH',   # Fidelity Ethereum Fund
        'ETHE',   # Grayscale Ethereum Trust
        'ETHW',   # Bitwise Ethereum ETF
    ],
}

# Correlation lookback
CORRELATION_LOOKBACK_DAYS = 30

# Signal thresholds
DXY_BREAKOUT_SIGMA = 1.0
ETF_OUTFLOW_CONSECUTIVE_DAYS = 3
MACRO_UPDATE_INTERVAL_SECONDS = 3600  # 1 hour


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

class MacroRegime(Enum):
    """Macro regime classification."""
    RISK_ON = auto()          # Favorable macro environment
    NEUTRAL = auto()          # Mixed signals
    RISK_OFF = auto()         # Unfavorable macro
    CRISIS = auto()           # Extreme risk-off
    DOLLAR_BREAKOUT = auto()  # DXY surge (crypto negative)


class ETFFlowSignal(Enum):
    """ETF flow signal classification."""
    STRONG_INFLOW = auto()    # Large institutional buying
    MODERATE_INFLOW = auto()  # Positive but muted
    NEUTRAL = auto()          # Mixed flows
    MODERATE_OUTFLOW = auto() # Negative flows
    STRONG_OUTFLOW = auto()   # Large redemptions


@dataclass
class MacroIndicator:
    """Single macro indicator state."""
    ticker: str
    name: str
    value: float
    prev_value: float
    change_pct: float
    zscore_30d: float
    timestamp: float
    
    @property
    def is_breakout_up(self) -> bool:
        return self.zscore_30d > DXY_BREAKOUT_SIGMA
    
    @property
    def is_breakout_down(self) -> bool:
        return self.zscore_30d < -DXY_BREAKOUT_SIGMA


@dataclass
class ETFFlowData:
    """ETF flow data for a single fund."""
    ticker: str
    asset: str  # BTC or ETH
    aum: float  # Assets Under Management (USD)
    daily_flow: float  # Daily net flow (USD)
    flow_7d_ma: float  # 7-day moving average
    flow_streak: int  # Consecutive days same direction (+/-)
    timestamp: float
    # [P420] the COMPLETED UTC day the flow describes (CoinGlass aggregate
    # path). The streak tracker counts a day exactly once by this key;
    # None = unknown day (per-ticker/mock paths), which keeps the legacy
    # advance-per-call behaviour.
    flow_day: Optional[str] = None


@dataclass
class MacroCorrelation:
    """Correlation between crypto and macro indicators."""
    btc_dxy_corr: float = 0.0
    btc_spx_corr: float = 0.0
    btc_us10y_corr: float = 0.0
    eth_dxy_corr: float = 0.0
    eth_spx_corr: float = 0.0
    sol_spx_corr: float = 0.0
    
    # Aggregated score
    macro_correlation_score: float = 0.0  # -1 to +1 (negative = crypto/macro inversely correlated)
    
    # Timestamp
    calculated_at: float = 0.0


@dataclass
class GlobalContextState:
    """Complete global context state."""
    
    # Macro indicators
    dxy: Optional[MacroIndicator] = None
    us10y: Optional[MacroIndicator] = None
    spx: Optional[MacroIndicator] = None
    vix: Optional[MacroIndicator] = None
    gold: Optional[MacroIndicator] = None
    hy_oas: Optional[MacroIndicator] = None  # [P393] credit spreads
    
    # ETF flows
    btc_etf_flows: List[ETFFlowData] = field(default_factory=list)
    eth_etf_flows: List[ETFFlowData] = field(default_factory=list)
    
    # Aggregated metrics
    btc_total_daily_flow: float = 0.0
    eth_total_daily_flow: float = 0.0
    btc_flow_streak: int = 0  # +N = N consecutive inflow days
    eth_flow_streak: int = 0
    
    # Correlations
    correlations: MacroCorrelation = field(default_factory=MacroCorrelation)
    
    # Regime flags
    macro_regime: MacroRegime = MacroRegime.NEUTRAL
    hy_stress_caution: bool = False  # [P393]
    btc_etf_signal: ETFFlowSignal = ETFFlowSignal.NEUTRAL
    eth_etf_signal: ETFFlowSignal = ETFFlowSignal.NEUTRAL
    
    # Caution flags
    dxy_breakout_caution: bool = False
    etf_outflow_caution: bool = False
    yield_spike_caution: bool = False
    
    # Composite score for DRL integration
    institutional_net_flow_7d: float = 0.0  # USD in millions
    macro_risk_score: float = 0.5  # 0 = risk-off, 1 = risk-on

    # [MACRO-FIX4] VIX gradient: rising vs stable distinction
    vix_gradient: float = 0.0   # change_pct of VIX (positive = rising fear)
    vix_rising: bool = False    # True when VIX rising meaningfully (>2% change)

    # Timing
    last_update: float = 0.0
    update_latency_ms: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# MACRO DATA FETCHER
# ═══════════════════════════════════════════════════════════════════════════════

class MacroDataFetcher:
    """
    Fetches macro indicator data via yfinance.
    Uses thread pool for parallel fetching.
    """
    
    def __init__(
        self,
        max_workers: int = 4,
        cache_ttl_seconds: int = 300,  # 5 min cache
    ):
        self.max_workers = max_workers
        self.cache_ttl = cache_ttl_seconds
        
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Historical data for z-score calculation
        self._history: Dict[str, deque] = {
            ticker: deque(maxlen=CORRELATION_LOOKBACK_DAYS)
            for ticker in MACRO_TICKERS.keys()
        }
        
        logger.info(f"MacroDataFetcher initialized (yfinance={YFINANCE_AVAILABLE})")
    
    def _fetch_ticker_sync(self, ticker: str, yf_symbol: str) -> Optional[MacroIndicator]:
        """Synchronous fetch for a single ticker."""
        if not YFINANCE_AVAILABLE:
            # [P293] yfinance is NOT in the engine image, so this branch has
            # been the ONLY branch in production and every macro indicator
            # came from _generate_mock_indicator. FRED serves the same
            # quantities on a key we already hold; try it before falling back
            # to the mock, so the macro regime can be computed from a real
            # observation for the first time.
            _fred_ind = self._fetch_ticker_fred(ticker)
            if _fred_ind is not None:
                return _fred_ind
            return self._generate_mock_indicator(ticker)

        try:
            # Check cache
            with self._lock:
                if ticker in self._cache:
                    data, cached_at = self._cache[ticker]
                    if time.time() - cached_at < self.cache_ttl:
                        return data
            
            # Fetch from yfinance
            yf_ticker = yf.Ticker(yf_symbol)
            hist = yf_ticker.history(period="35d")  # Extra days for z-score
            
            if hist.empty:
                logger.warning(f"No data for {ticker} ({yf_symbol})")
                return self._generate_mock_indicator(ticker)
            
            # Extract values
            closes = hist['Close'].values
            current = closes[-1]
            prev = closes[-2] if len(closes) > 1 else current
            change_pct = (current - prev) / prev * 100 if prev != 0 else 0
            
            # Z-score calculation
            if len(closes) >= CORRELATION_LOOKBACK_DAYS:
                mean_30d = np.mean(closes[-CORRELATION_LOOKBACK_DAYS:])
                std_30d = np.std(closes[-CORRELATION_LOOKBACK_DAYS:])
                zscore = (current - mean_30d) / std_30d if std_30d > 0 else 0
            else:
                zscore = 0
            
            indicator = MacroIndicator(
                ticker=ticker,
                name=yf_symbol,
                value=float(current),
                prev_value=float(prev),
                change_pct=float(change_pct),
                zscore_30d=float(zscore),
                timestamp=time.time()
            )
            
            # Update history
            with self._lock:
                self._history[ticker].append(current)
                self._cache[ticker] = (indicator, time.time())
            
            return indicator
            
        except Exception as e:
            logger.error(f"Error fetching {ticker}: {e}")
            return self._generate_mock_indicator(ticker)
    
    def _fetch_ticker_fred(self, ticker: str) -> Optional[MacroIndicator]:
        """[P293] Build one indicator from FRED daily observations.

        Returns None — never a placeholder — when the series is unmapped
        (GOLD), the key is missing, or the fetch fails, so the caller can
        fall back explicitly instead of consuming a fabricated level.

        Runs inside the existing ThreadPoolExecutor, so a blocking stdlib
        request is correct here and never touches the event loop (P253c).
        """
        try:
            from data_mgmt.feeds.fred_macro_series import fetch_daily_closes, is_proxy
        except Exception:  # noqa: silent-swallow
            return None

        try:
            with self._lock:
                cached = self._cache.get(ticker)
            if cached is not None:
                data, cached_at = cached
                if time.time() - cached_at < self.cache_ttl:
                    return data

            rows = fetch_daily_closes(ticker, lookback_days=60)
            if len(rows) < 2:
                return None

            closes = [v for _, v in rows]
            current = closes[-1]
            prev = closes[-2]
            change_pct = ((current - prev) / prev * 100) if prev else 0.0

            window = closes[-CORRELATION_LOOKBACK_DAYS:]
            if len(window) >= CORRELATION_LOOKBACK_DAYS:
                mean_30d = float(np.mean(window))
                std_30d = float(np.std(window))
                # A degenerate window has no scale — 0.0 here means "no
                # z-score", and the breakout tests correctly stay False.
                zscore = ((current - mean_30d) / std_30d) if std_30d > 1e-9 else 0.0
            else:
                zscore = 0.0

            indicator = MacroIndicator(
                ticker=ticker,
                name=f"FRED:{ticker}" + ("(proxy)" if is_proxy(ticker) else ""),
                value=float(current),
                prev_value=float(prev),
                change_pct=float(change_pct),
                zscore_30d=float(zscore),
                timestamp=time.time(),
            )

            with self._lock:
                self._history[ticker].append(current)
                self._cache[ticker] = (indicator, time.time())

            return indicator
        except Exception as e:
            logger.warning(f"[P293][GCI] FRED fetch failed for {ticker}: {e}")
            return None

    def _generate_mock_indicator(self, ticker: str) -> MacroIndicator:
        """
        Generate conservative mock data when yfinance unavailable.

        [MACRO-FIX5] Mock = "I don't know macro state" -> NEUTRAL baseline.
        All values fixed (zero change, zero zscore) - NO randomness.
        Previous version used np.random.uniform for zscore -> caused
        spurious regime transitions on data outages.
        """
        _now = time.time()
        if not hasattr(self, '_mock_warn_time') or _now - self._mock_warn_time > 3600:
            # [P308] The old text said "yfinance FAILED ... Check Yahoo
            # Finance API", which is misleading twice over now: P293 moved
            # this to FRED, and GOLD is UNMAPPED THERE BY DECISION (FRED's
            # daily gold series is discontinued), so it is not a fault and
            # there is no API to go and check. An instruction the operator
            # cannot act on is the P202 shape. The mock stays neutral by
            # construction (change_pct=0.0, zscore_30d=0.0 — MACRO-FIX5), and
            # P294 verified `_state.gold` is write-only, so the fabricated
            # LEVEL reaches no consumer.
            _known_unmapped = ticker.upper() in ("GOLD",)
            logger.warning(
                f"[GCI] {ticker}: no live series — serving a NEUTRAL mock "
                f"(change=0, z=0), so it contributes nothing to MacroRegime. "
                + ("This is EXPECTED and needs no action: GOLD is deliberately "
                   "unmapped in FRED (its daily series is discontinued) and "
                   "the value is not consumed (P293/P294/P308)."
                   if _known_unmapped else
                   "Unexpected for this ticker — check the FRED series "
                   "mapping in fred_macro_series.py (P293).")
            )
            self._mock_warn_time = _now
        # [MACRO-FIX5] Fixed values - same value/prev_value -> zero change, zero zscore
        mock_values = {
            'DXY': 104.5,
            'US10Y': 4.25,
            'SPX': 5850.0,
            'VIX': 15.0,      # Baseline calm - avoids false CRISIS
            'GOLD': 2050.0,
        }

        value = mock_values.get(ticker, 100.0)

        return MacroIndicator(
            ticker=ticker,
            name=f"MOCK_{ticker}",
            value=value,
            prev_value=value,       # [MACRO-FIX5] Same as value -> change=0
            change_pct=0.0,         # [MACRO-FIX5] Zero -> no regime signal
            zscore_30d=0.0,         # [MACRO-FIX5] Zero -> no breakout flags
            timestamp=time.time()
        )
    
    async def fetch_all_async(self) -> Dict[str, MacroIndicator]:
        """Fetch all macro indicators asynchronously."""
        loop = asyncio.get_event_loop()
        
        # Submit all fetches to thread pool
        futures = {
            ticker: loop.run_in_executor(
                self._executor,
                self._fetch_ticker_sync,
                ticker,
                symbol
            )
            for ticker, symbol in MACRO_TICKERS.items()
        }
        
        # Gather results
        results = {}
        for ticker, future in futures.items():
            try:
                result = await future
                if result:
                    results[ticker] = result
            except Exception as e:
                logger.error(f"Async fetch error for {ticker}: {e}")
        
        return results
    
    def fetch_all_sync(self) -> Dict[str, MacroIndicator]:
        """Synchronous fetch all macro indicators."""
        results = {}
        for ticker, symbol in MACRO_TICKERS.items():
            result = self._fetch_ticker_sync(ticker, symbol)
            if result:
                results[ticker] = result
        return results
    
    def get_history(self, ticker: str) -> np.ndarray:
        """Get historical values for correlation calculation."""
        with self._lock:
            return np.array(list(self._history.get(ticker, [])))


# ═══════════════════════════════════════════════════════════════════════════════
# ETF FLOW TRACKER
# ═══════════════════════════════════════════════════════════════════════════════

class ETFFlowTracker:
    """
    Tracks Spot BTC/ETH ETF flows for institutional signal.
    
    Data sources:
    - yfinance for AUM/volume proxy
    - Public APIs for flow data (simulated if unavailable)
    """
    
    def __init__(self):
        self._flow_history: Dict[str, deque] = {}
        self._streak_tracker: Dict[str, int] = {
            'BTC': 0,
            'ETH': 0,
        }
        self._lock = threading.Lock()
        
        # Initialize flow history
        for asset, etfs in SPOT_ETF_TICKERS.items():
            for etf in etfs:
                self._flow_history[etf] = deque(maxlen=30)
        
        logger.info("ETFFlowTracker initialized")
    
    def _fetch_etf_data_sync(self, ticker: str, asset: str) -> Optional[ETFFlowData]:
        """Fetch ETF flow data synchronously."""
        if not YFINANCE_AVAILABLE:
            # [P293] REFUSE rather than fabricate. yfinance is absent from
            # the engine image, so this branch is the only one that runs;
            # returning a mock flow would feed invented ETF numbers into
            # macro_regime the moment the GCI updater is driven. There is no
            # FRED substitute for fund flows — the real ETF data lives in
            # the P270 EtfFlowShadow (CoinGlass v4), which is a separate,
            # observation-only stream. Absence must stay absence (P2).
            _now = time.time()
            if _now - getattr(self, "_etf_refuse_warn_time", 0.0) > 3600:
                logger.warning(
                    "[P293][GCI] yfinance unavailable -> NO ETF flow reading "
                    "for %s. Flows are absent, NOT zero. Real ETF flow data "
                    "is in the P270 EtfFlowShadow stream.", asset
                )
                self._etf_refuse_warn_time = _now
            return None

        try:
            yf_ticker = yf.Ticker(ticker)
            info = yf_ticker.info
            hist = yf_ticker.history(period="10d")
            
            if hist.empty:
                return self._generate_mock_flow(ticker, asset)
            
            # Estimate AUM from market cap or use volume as proxy
            aum = info.get('totalAssets', 0) or info.get('marketCap', 0) or 1e9
            
            # Estimate daily flow from volume and price change
            # In production, would use actual flow APIs
            if len(hist) >= 2:
                volume = hist['Volume'].iloc[-1]
                price = hist['Close'].iloc[-1]
                prev_price = hist['Close'].iloc[-2]
                
                # Simplified flow estimation
                price_impact = (price - prev_price) / prev_price if prev_price > 0 else 0
                estimated_flow = volume * price * price_impact * 0.1  # Scaling factor
            else:
                estimated_flow = 0
            
            # Calculate 7-day MA
            with self._lock:
                self._flow_history[ticker].append(estimated_flow)
                flows = list(self._flow_history[ticker])
                flow_7d_ma = np.mean(flows[-7:]) if len(flows) >= 7 else estimated_flow
            
            # Track streak
            streak = self._calculate_streak(ticker, estimated_flow)
            
            return ETFFlowData(
                ticker=ticker,
                asset=asset,
                aum=float(aum),
                daily_flow=float(estimated_flow),
                flow_7d_ma=float(flow_7d_ma),
                flow_streak=streak,
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.warning(f"Error fetching ETF {ticker}: {e}")
            return self._generate_mock_flow(ticker, asset)
    
    def _generate_mock_flow(self, ticker: str, asset: str) -> ETFFlowData:
        """
        Generate conservative mock ETF flow data.

        [MACRO-FIX5] Zero flow, not random. Previous version used
        np.random.normal -> spurious flow_streak transitions.
        Mock = "no ETF data available" -> zero flow -> NEUTRAL signal.
        """
        base_aum = {'IBIT': 50e9, 'FBTC': 20e9, 'GBTC': 25e9}.get(ticker, 5e9)

        return ETFFlowData(
            ticker=ticker,
            asset=asset,
            aum=base_aum,
            daily_flow=0.0,       # [MACRO-FIX5] Zero -> NEUTRAL, not random
            flow_7d_ma=0.0,       # [MACRO-FIX5] Zero -> no trend signal
            flow_streak=0,        # [MACRO-FIX5] No streak
            timestamp=time.time()
        )
    
    def _calculate_streak(self, ticker: str, flow: float) -> int:
        """Calculate consecutive days of same-direction flow."""
        # Simplified streak tracking
        # [P293b] NOTE the else-branch: any ticker NOT in the BTC list is
        # treated as ETH. That is fine for the per-fund tickers this was
        # written for, and WRONG for anything else — so asset-keyed callers
        # must use _calculate_streak_for_asset below rather than inventing a
        # synthetic ticker and landing in the ETH bucket by default.
        asset = 'BTC' if ticker in SPOT_ETF_TICKERS.get('BTC', []) else 'ETH'
        return self._calculate_streak_for_asset(asset, flow)

    # [P420] streak persistence -------------------------------------------
    _STREAK_STATE_VERSION = "etf_streak_v1"

    def _streak_state_path(self) -> str:
        return os.path.join(os.environ.get("HMATS_DATA_DIR", "data"),
                            "etf_streak_state.json")

    def _restore_streak_state(self) -> None:
        """Lazy, once, and ONLY from the day-keyed path: the legacy
        per-call entry point never touches disk, so callers that construct a
        tracker without a data dir (tests, the per-ticker path) see exactly
        the pre-P420 behaviour. A missing/corrupt/version-mismatched file
        restores nothing (cold streak, logged) — never a fabricated one."""
        if getattr(self, "_streak_restored", False):
            return
        self._streak_restored = True
        self._streak_last_day: Dict[str, str] = {}
        p = self._streak_state_path()
        try:
            if not os.path.exists(p):
                return
            with open(p, encoding="utf-8") as fh:
                pay = json.load(fh)
            if pay.get("version") != self._STREAK_STATE_VERSION:
                logger.info("[P420][GCI] streak state version mismatch — "
                            "cold streaks")
                return
            for a, st in (pay.get("assets") or {}).items():
                if a not in self._streak_tracker:
                    continue
                self._streak_tracker[a] = int(st.get("streak", 0) or 0)
                d = st.get("last_day")
                if isinstance(d, str) and d:
                    self._streak_last_day[a] = d
            logger.info("[P420][GCI] restored ETF flow streaks %s",
                        {a: (self._streak_tracker[a],
                             self._streak_last_day.get(a))
                         for a in self._streak_last_day})
        except Exception as e:  # noqa: silent-swallow — logged; a bad file = cold streak
            logger.warning("[P420][GCI] streak state unreadable (%s: %s) — "
                           "cold streaks", type(e).__name__, e)

    def _persist_streak_state(self) -> None:
        """Atomic write (os.replace). Never raises: persistence failure must
        not break the macro update."""
        import tempfile
        p = self._streak_state_path()
        try:
            d = os.path.dirname(p) or "."
            os.makedirs(d, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump({
                        "version": self._STREAK_STATE_VERSION,
                        "saved_ts": time.time(),
                        "assets": {
                            a: {"streak": int(self._streak_tracker.get(a, 0)),
                                "last_day": self._streak_last_day.get(a)}
                            for a in self._streak_tracker},
                    }, fh)
                os.replace(tmp, p)
            finally:
                if os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except OSError:  # noqa: silent-swallow — tmp cleanup only
                        pass
        except Exception as e:  # noqa: silent-swallow — logged; persistence only
            logger.warning("[P420][GCI] streak persist failed (%s: %s) — the "
                           "next restart re-counts from a cold streak",
                           type(e).__name__, e)

    def _calculate_streak_for_asset(self, asset: str, flow: float,
                                    day_iso: Optional[str] = None) -> int:
        """[P293b] Streak keyed by ASSET, not by ticker.

        Same state and same rules as _calculate_streak — extracted so the
        CoinGlass aggregate path cannot be mis-bucketed by the ticker
        heuristic above (P172: one implementation, two entry points).

        [P420] `day_iso` = the COMPLETED UTC day the flow describes. The
        streak is "N consecutive DAYS", but this ran once per HOURLY GCI
        update with no day gate and RAM-only state — so one outflow day
        advanced the streak ~24 times, and every restart re-counted the
        same day. With a day key the streak advances exactly once per new
        day and the last counted day is persisted; a repeat of the same
        day (or a restart inside it) returns the standing streak untouched.
        `day_iso=None` keeps the legacy advance-per-call behaviour.
        """
        with self._lock:
            if day_iso is not None:
                self._restore_streak_state()
                if self._streak_last_day.get(asset) == day_iso:
                    return self._streak_tracker.get(asset, 0)
            current_streak = self._streak_tracker.get(asset, 0)
            
            if flow > 0:
                if current_streak >= 0:
                    self._streak_tracker[asset] = current_streak + 1
                else:
                    self._streak_tracker[asset] = 1
            elif flow < 0:
                if current_streak <= 0:
                    self._streak_tracker[asset] = current_streak - 1
                else:
                    self._streak_tracker[asset] = -1

            if day_iso is not None:
                self._streak_last_day[asset] = day_iso
                self._persist_streak_state()
            return self._streak_tracker[asset]
    
    def _fetch_aggregate_flow_coinglass(self, asset: str) -> Optional[ETFFlowData]:
        """[P293b] ONE aggregate ETF flow record per asset, from CoinGlass.

        yfinance is absent from the engine image, so the per-ticker path is
        unavailable and previously fell back to a MOCK flow — which would
        have injected invented fund flows into macro_regime the moment the
        GCI updater was driven. Real daily ETF flows already arrive through
        the P270 EtfFlowShadow (CoinGlass v4), so this REUSES that reader
        rather than adding a second implementation of the same endpoint
        (P172 single-source) — including its in-progress-day guard, which
        only serves the newest COMPLETED UTC day (P253c).

        Returns ONE record for the whole asset, deliberately: the caller
        sums the list, so emitting one aggregate per *ticker* would multiply
        the real flow by the number of tickers.

        Returns None — never a placeholder — when the flow is unavailable or
        stale, so absence stays absence (P2).
        """
        try:
            from defense.etf_flow_shadow import EtfFlowShadow
        except Exception:  # noqa: silent-swallow
            return None

        try:
            if getattr(self, "_cg_etf_reader", None) is None:
                self._cg_etf_reader = EtfFlowShadow(
                    data_dir=os.environ.get("HMATS_DATA_DIR", "data"))
            flow_usd, age_days, day_iso = \
                self._cg_etf_reader.latest_completed_flow(asset)
            if flow_usd is None:
                return None
            # A flow older than ~4 days is a reporting gap or an outage, not
            # a reading about today (weekends legitimately reach ~3 days).
            if age_days is not None and age_days > 4.0:
                logger.warning(
                    "[P293b][GCI] %s ETF flow is %.1f days old (%s) — treating "
                    "as ABSENT rather than current", asset, age_days, day_iso)
                return None
            return ETFFlowData(
                ticker=f"{asset}_AGGREGATE_COINGLASS",
                asset=asset,
                aum=0.0,          # not published by this endpoint; 0 = unknown
                daily_flow=float(flow_usd),
                flow_7d_ma=0.0,   # not computed here; the tracker owns streaks
                flow_streak=0,
                timestamp=time.time(),
                flow_day=str(day_iso) if day_iso else None,  # [P420]
            )
        except Exception as e:
            logger.warning(
                "[P293b][GCI] CoinGlass ETF aggregate failed for %s: %s: %s",
                asset, type(e).__name__, e)
            return None

    async def fetch_all_flows_async(self) -> Tuple[List[ETFFlowData], List[ETFFlowData]]:
        """Fetch all ETF flows asynchronously."""
        loop = asyncio.get_event_loop()

        btc_flows: List[ETFFlowData] = []
        eth_flows: List[ETFFlowData] = []

        # [P293b] Without yfinance the per-ticker loop below can only refuse,
        # so take the real aggregate from CoinGlass instead. Kept as an
        # early return so the per-ticker path is untouched wherever yfinance
        # IS available.
        if not YFINANCE_AVAILABLE:
            for _asset, _bucket in (("BTC", btc_flows), ("ETH", eth_flows)):
                _agg = await loop.run_in_executor(
                    None, self._fetch_aggregate_flow_coinglass, _asset)
                if _agg is not None:
                    _bucket.append(_agg)
                    # [P420] day-keyed: counts each completed day once
                    self._calculate_streak_for_asset(
                        _asset, _agg.daily_flow,
                        day_iso=getattr(_agg, "flow_day", None))
            logger.info(
                "[P293b][GCI] ETF flows via CoinGlass aggregate: "
                "BTC=%s ETH=%s",
                f"{btc_flows[0].daily_flow:+,.0f}" if btc_flows else "absent",
                f"{eth_flows[0].daily_flow:+,.0f}" if eth_flows else "absent",
            )
            return btc_flows, eth_flows

        # BTC ETFs
        for ticker in SPOT_ETF_TICKERS['BTC']:
            try:
                flow = await loop.run_in_executor(
                    None,
                    self._fetch_etf_data_sync,
                    ticker,
                    'BTC'
                )
                if flow:
                    btc_flows.append(flow)
            except Exception as e:
                logger.error(f"Error fetching BTC ETF {ticker}: {e}")
        
        # ETH ETFs
        for ticker in SPOT_ETF_TICKERS['ETH']:
            try:
                flow = await loop.run_in_executor(
                    None,
                    self._fetch_etf_data_sync,
                    ticker,
                    'ETH'
                )
                if flow:
                    eth_flows.append(flow)
            except Exception as e:
                logger.error(f"Error fetching ETH ETF {ticker}: {e}")
        
        return btc_flows, eth_flows
    
    def get_streak(self, asset: str) -> int:
        """Get current flow streak for an asset."""
        with self._lock:
            return self._streak_tracker.get(asset, 0)


# ═══════════════════════════════════════════════════════════════════════════════
# CORRELATION CALCULATOR
# ═══════════════════════════════════════════════════════════════════════════════

class CorrelationCalculator:
    """
    Calculates dynamic correlations between crypto and macro indicators.
    Uses 30-day rolling window.
    """
    
    def __init__(self, lookback_days: int = CORRELATION_LOOKBACK_DAYS):
        self.lookback = lookback_days
        
        # Price history for crypto
        self._crypto_history: Dict[str, deque] = {
            'BTC': deque(maxlen=lookback_days),
            'ETH': deque(maxlen=lookback_days),
            'SOL': deque(maxlen=lookback_days),
        }
        
        self._lock = threading.Lock()
        logger.info(f"CorrelationCalculator initialized (lookback={lookback_days})")
    
    def update_crypto_price(self, asset: str, price: float):
        """Update crypto price history."""
        with self._lock:
            if asset in self._crypto_history:
                self._crypto_history[asset].append(price)
    
    def calculate_correlation(
        self,
        series1: np.ndarray,
        series2: np.ndarray
    ) -> float:
        """Calculate Pearson correlation between two series."""
        if len(series1) < 5 or len(series2) < 5:
            return 0.0
        
        # Align lengths
        min_len = min(len(series1), len(series2))
        s1 = series1[-min_len:]
        s2 = series2[-min_len:]
        
        # Calculate correlation
        if SCIPY_AVAILABLE:
            corr, _ = stats.pearsonr(s1, s2)
            return float(corr) if not np.isnan(corr) else 0.0
        else:
            # Manual calculation
            if np.std(s1) == 0 or np.std(s2) == 0:
                return 0.0
            return float(np.corrcoef(s1, s2)[0, 1])
    
    def calculate_all_correlations(
        self,
        macro_history: Dict[str, np.ndarray]
    ) -> MacroCorrelation:
        """Calculate all crypto-macro correlations."""
        
        with self._lock:
            btc_prices = np.array(list(self._crypto_history['BTC']))
            eth_prices = np.array(list(self._crypto_history['ETH']))
            sol_prices = np.array(list(self._crypto_history['SOL']))
        
        # Get macro data
        dxy_data = macro_history.get('DXY', np.array([]))
        spx_data = macro_history.get('SPX', np.array([]))
        us10y_data = macro_history.get('US10Y', np.array([]))
        
        # Calculate correlations
        correlations = MacroCorrelation(
            btc_dxy_corr=self.calculate_correlation(btc_prices, dxy_data),
            btc_spx_corr=self.calculate_correlation(btc_prices, spx_data),
            btc_us10y_corr=self.calculate_correlation(btc_prices, us10y_data),
            eth_dxy_corr=self.calculate_correlation(eth_prices, dxy_data),
            eth_spx_corr=self.calculate_correlation(eth_prices, spx_data),
            sol_spx_corr=self.calculate_correlation(sol_prices, spx_data),
            calculated_at=time.time()
        )
        
        # Calculate composite score
        # Positive correlation with SPX is good, negative with DXY is good
        correlations.macro_correlation_score = (
            correlations.btc_spx_corr * 0.3 +
            correlations.eth_spx_corr * 0.2 +
            (-correlations.btc_dxy_corr) * 0.3 +  # Invert DXY correlation
            (-correlations.btc_us10y_corr) * 0.2   # Invert yield correlation
        )
        
        return correlations


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL CONTEXT INFORMER (MAIN CLASS)
# ═══════════════════════════════════════════════════════════════════════════════

class GlobalContextInformer:
    """
    Main class for global macro and institutional flow monitoring.
    
    Integrates:
    - Macro indicators (DXY, US10Y, SPX, VIX, GOLD)
    - ETF flows (BTC: IBIT, FBTC, etc. | ETH: ETHA, FETH, etc.)
    - Dynamic correlations (30-day rolling)
    
    Update frequency: 1 hour (configurable)
    """
    
    def __init__(
        self,
        update_interval_seconds: int = MACRO_UPDATE_INTERVAL_SECONDS,
        enable_auto_update: bool = True
    ):
        self.update_interval = update_interval_seconds
        self.enable_auto_update = enable_auto_update
        
        # Components
        self.macro_fetcher = MacroDataFetcher()
        self.etf_tracker = ETFFlowTracker()
        self.correlation_calculator = CorrelationCalculator()
        
        # State
        self._state = GlobalContextState()
        self._state_lock = threading.Lock()
        
        # Background task control
        self._running = False
        self._update_task: Optional[asyncio.Task] = None
        
        # Callbacks for state updates
        self._callbacks: List[Callable[[GlobalContextState], None]] = []
        
        logger.info(f"GlobalContextInformer initialized (interval={update_interval_seconds}s)")
    
    def register_callback(self, callback: Callable[[GlobalContextState], None]):
        """Register callback for state updates."""
        self._callbacks.append(callback)
    
    def _notify_callbacks(self):
        """Notify all registered callbacks."""
        state = self.get_state()
        for callback in self._callbacks:
            try:
                callback(state)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    async def update_if_stale(self) -> Optional[GlobalContextState]:
        """[P293] Update only when the state is older than update_interval.

        `update_async()` has NO caller anywhere in the non-test tree — the
        live log has never once emitted "GlobalContext updated" — so every
        consumer of `get_macro_signal()` has read the initial default state
        for the life of the system. This is the throttled entry point a
        caller can safely put on a per-asset tick.

        Never raises: a macro refresh failing must not disturb a tick.
        """
        try:
            _last = float(getattr(self._state, "last_update", 0.0) or 0.0)
            _interval = float(getattr(self, "update_interval", 3600.0) or 3600.0)
            if _last > 0.0 and (time.time() - _last) < _interval:
                return self._state
            return await self.update_async()
        except Exception as e:
            logger.warning(
                f"[P293][GCI] update_if_stale failed: {type(e).__name__}: {e}"
            )
            return None

    async def update_async(self) -> GlobalContextState:
        """Perform full async update of global context."""
        start_time = time.perf_counter()
        
        try:
            # Fetch macro indicators
            macro_data = await self.macro_fetcher.fetch_all_async()
            
            # Fetch ETF flows
            btc_flows, eth_flows = await self.etf_tracker.fetch_all_flows_async()
            
            # Calculate correlations
            macro_history = {
                ticker: self.macro_fetcher.get_history(ticker)
                for ticker in MACRO_TICKERS.keys()
            }
            correlations = self.correlation_calculator.calculate_all_correlations(macro_history)
            
            # Update state
            with self._state_lock:
                self._state.dxy = macro_data.get('DXY')
                self._state.us10y = macro_data.get('US10Y')
                self._state.spx = macro_data.get('SPX')
                self._state.vix = macro_data.get('VIX')
                self._state.hy_oas = macro_data.get('HY_OAS')  # [P393]
                # [MACRO-FIX4] VIX gradient from change_pct
                if self._state.vix:
                    self._state.vix_gradient = self._state.vix.change_pct
                    self._state.vix_rising = self._state.vix.change_pct > 2.0  # >2% = meaningful rise
                else:
                    self._state.vix_gradient = 0.0
                    self._state.vix_rising = False
                self._state.gold = macro_data.get('GOLD')
                
                self._state.btc_etf_flows = btc_flows
                self._state.eth_etf_flows = eth_flows
                
                # Aggregate ETF metrics
                self._state.btc_total_daily_flow = sum(f.daily_flow for f in btc_flows)
                self._state.eth_total_daily_flow = sum(f.daily_flow for f in eth_flows)
                self._state.btc_flow_streak = self.etf_tracker.get_streak('BTC')
                self._state.eth_flow_streak = self.etf_tracker.get_streak('ETH')
                
                self._state.correlations = correlations
                
                # Calculate regime and signals
                self._update_regime_signals()
                
                # Timing
                self._state.last_update = time.time()
                self._state.update_latency_ms = (time.perf_counter() - start_time) * 1000
            
            # Notify callbacks
            self._notify_callbacks()
            
            logger.info(
                f"GlobalContext updated: DXY={self._state.dxy.value if self._state.dxy else 'N/A':.2f}, "
                f"BTC_Flow=${self._state.btc_total_daily_flow/1e6:.1f}M, "
                f"Regime={self._state.macro_regime.name}"
            )
            
            return self.get_state()
            
        except Exception as e:
            logger.error(f"Update error: {e}")
            raise
    
    def _update_regime_signals(self):
        """Update regime and signal classifications."""
        
        # === DXY Breakout Check ===
        if self._state.dxy:
            self._state.dxy_breakout_caution = self._state.dxy.is_breakout_up
        
        # === ETF Outflow Check ===
        btc_outflow = self._state.btc_flow_streak <= -ETF_OUTFLOW_CONSECUTIVE_DAYS
        eth_outflow = self._state.eth_flow_streak <= -ETF_OUTFLOW_CONSECUTIVE_DAYS
        self._state.etf_outflow_caution = btc_outflow or eth_outflow
        
        # === US10Y Yield Spike Check ===
        if self._state.us10y:
            self._state.yield_spike_caution = self._state.us10y.zscore_30d > 1.5
        # [P393] HY credit-spread stress: the one macro factor measured
        # significant with a stable sign in BOTH eras on all three assets
        # (P392/P393 labs; widening spreads = crypto down, same day). A
        # 30d z > 1.5 counts as a caution exactly like a yield spike; an
        # absent series contributes nothing (P2 - never a fabricated calm).
        self._state.hy_stress_caution = bool(
            self._state.hy_oas and self._state.hy_oas.zscore_30d > 1.5)
        
        # === ETF Flow Signal Classification ===
        btc_flow = self._state.btc_total_daily_flow
        if btc_flow > 500e6:
            self._state.btc_etf_signal = ETFFlowSignal.STRONG_INFLOW
        elif btc_flow > 100e6:
            self._state.btc_etf_signal = ETFFlowSignal.MODERATE_INFLOW
        elif btc_flow < -500e6:
            self._state.btc_etf_signal = ETFFlowSignal.STRONG_OUTFLOW
        elif btc_flow < -100e6:
            self._state.btc_etf_signal = ETFFlowSignal.MODERATE_OUTFLOW
        else:
            self._state.btc_etf_signal = ETFFlowSignal.NEUTRAL
        
        # === Macro Regime Classification ===
        caution_count = sum([
            self._state.dxy_breakout_caution,
            self._state.etf_outflow_caution,
            self._state.yield_spike_caution,
            self._state.hy_stress_caution,   # [P393] credit-spread stress
        ])
        
        vix_value = self._state.vix.value if self._state.vix else 15.0
        
        if caution_count >= 2 or vix_value > 30:
            self._state.macro_regime = MacroRegime.CRISIS
        elif self._state.dxy_breakout_caution:
            self._state.macro_regime = MacroRegime.DOLLAR_BREAKOUT
        elif caution_count >= 1:
            self._state.macro_regime = MacroRegime.RISK_OFF
        elif self._state.btc_etf_signal in [ETFFlowSignal.STRONG_INFLOW, ETFFlowSignal.MODERATE_INFLOW]:
            self._state.macro_regime = MacroRegime.RISK_ON
        else:
            self._state.macro_regime = MacroRegime.NEUTRAL
        
        # === Composite Metrics for DRL ===
        # Institutional net flow (7-day, in millions)
        btc_7d_ma = sum(f.flow_7d_ma for f in self._state.btc_etf_flows)
        eth_7d_ma = sum(f.flow_7d_ma for f in self._state.eth_etf_flows)
        self._state.institutional_net_flow_7d = (btc_7d_ma + eth_7d_ma) / 1e6
        
        # Macro risk score (0 = risk-off, 1 = risk-on)
        risk_score = 0.5
        if self._state.macro_regime == MacroRegime.RISK_ON:
            risk_score = 0.8
        elif self._state.macro_regime == MacroRegime.RISK_OFF:
            risk_score = 0.3
        elif self._state.macro_regime in [MacroRegime.CRISIS, MacroRegime.DOLLAR_BREAKOUT]:
            risk_score = 0.1
        self._state.macro_risk_score = risk_score
    
    def get_state(self) -> GlobalContextState:
        """Get current state (thread-safe copy)."""
        with self._state_lock:
            # Return a copy to prevent race conditions
            import copy
            return copy.deepcopy(self._state)
    
    def get_drl_features(self) -> Dict[str, float]:
        """Get features formatted for DRL state vector augmentation."""
        with self._state_lock:
            state = self._state
            macro_correlation_score = state.correlations.macro_correlation_score
            institutional_net_flow_7d = state.institutional_net_flow_7d
            macro_risk_score = state.macro_risk_score
            dxy_zscore = state.dxy.zscore_30d if state.dxy else 0.0
            vix_level = state.vix.value / 100 if state.vix else 0.15  # Normalized
            btc_etf_flow_signal = state.btc_etf_signal
            eth_etf_flow_signal = state.eth_etf_signal
            macro_regime = state.macro_regime
            dxy_breakout_caution = state.dxy_breakout_caution

        return {
            'macro_correlation_score': macro_correlation_score,
            'institutional_net_flow_7d': institutional_net_flow_7d,
            'macro_risk_score': macro_risk_score,
            'dxy_zscore': dxy_zscore,
            'vix_level': vix_level,
            'btc_etf_flow_signal': self._etf_signal_to_float(btc_etf_flow_signal),
            'eth_etf_flow_signal': self._etf_signal_to_float(eth_etf_flow_signal),
            'regime_is_risk_on': 1.0 if macro_regime == MacroRegime.RISK_ON else 0.0,
            'regime_is_crisis': 1.0 if macro_regime == MacroRegime.CRISIS else 0.0,
            'dxy_breakout': 1.0 if dxy_breakout_caution else 0.0,
        }
    
    def get_macro_signal(self) -> Dict[str, Any]:
        """
        Get macro signal for Authority Fusion integration.

        [P393] ADVERSE-RESTRICTS orientation (replaces [MACRO-FIX1]).

        MACRO-FIX1 inverted this map for a SHORT-BIASED system (CRISIS=1.5
        so "shorts should have NO macro restriction"). That premise died
        with the P140 era: the live books are structurally LONG-biased
        (ETH/SOL books never short). And the P393 audit measured that the
        sole consumer (authority_fusion LAYER 6) acts ONLY on cap < 1.0 -
        caps above 1.0 have NO reader anywhere - so under MACRO-FIX1 the
        fetched macro classification could only ever penalize the book in
        BENIGN regimes (RISK_ON 0.7 cut the long book ~50-58% via the
        conviction channel, observed live at conviction 0.42-0.52 on
        RISK_ON days) while providing ZERO de-risking in CRISIS.

        The map now encodes the one macro fact the P392/P393 labs
        validated (risk-off asymmetry: adverse macro = crypto down,
        contemporaneous): ADVERSE regimes carry cap < 1.0 and restrict the
        LONG side (shorts get relief in the consumer, never a boost);
        favorable/neutral regimes restrict NOTHING. The cap can never
        exceed 1.0 - macro de-risks, it never sizes UP (P293d de-risk-only
        principle).

        Macro NEVER vetoes (v6.5 principle preserved).
        """
        with self._state_lock:
            state = self._state
            macro_regime = state.macro_regime
            macro_risk_score = state.macro_risk_score
            dxy_breakout_caution = state.dxy_breakout_caution
            etf_outflow_caution = state.etf_outflow_caution
            yield_spike_caution = state.yield_spike_caution
            btc_flow_streak = state.btc_flow_streak
            eth_flow_streak = state.eth_flow_streak
            btc_etf_signal = state.btc_etf_signal
            eth_etf_signal = state.eth_etf_signal
            vix_value = state.vix.value if state.vix else 15.0
            vix_gradient = state.vix_gradient
            vix_rising = state.vix_rising
            dxy_zscore = state.dxy.zscore_30d if state.dxy else None

        leverage_cap_map = {
            MacroRegime.CRISIS: 0.4,
            MacroRegime.DOLLAR_BREAKOUT: 0.8,
            MacroRegime.RISK_OFF: 0.7,
            MacroRegime.NEUTRAL: 1.0,
            MacroRegime.RISK_ON: 1.0,
        }
        leverage_cap = leverage_cap_map.get(macro_regime, 1.0)

        short_opportunity = 1.0 - macro_risk_score

        caution_flags = []
        if dxy_breakout_caution:
            caution_flags.append("DXY_BREAKOUT_SHORT_FAVORABLE")
        if etf_outflow_caution:
            caution_flags.append("ETF_OUTFLOW_SHORT_FAVORABLE")
        if yield_spike_caution:
            caution_flags.append("YIELD_SPIKE_AMBIGUOUS")

        if yield_spike_caution and macro_regime == MacroRegime.RISK_ON:
            leverage_cap = min(leverage_cap, 0.6)

        etf_short_boost = 0.0
        if btc_flow_streak <= -3:
            # [P393] sustained ETF OUTFLOWS are crypto-adverse for a LONG
            # book: DE-RISK (was: a short-bias-era cap RAISE toward 1.5,
            # which the consumer could never read anyway - caps > 1.0 have
            # no reader). The streak magnitude is still exported.
            etf_short_boost = min(0.15, abs(btc_flow_streak) * 0.05)
            leverage_cap = min(leverage_cap, 0.8)

        vix_urgency = 0.0
        if vix_value > 25:
            vix_urgency = min(1.0, (vix_value - 25) / 20)
            if vix_rising:
                vix_urgency = min(1.0, vix_urgency * 1.3)
        elif vix_value < 12:
            vix_urgency = -0.3

        return {
            'leverage_cap': round(leverage_cap, 2),
            'risk_appetite': round(short_opportunity, 3),
            'short_opportunity': round(short_opportunity, 3),
            'regime': macro_regime.name,
            'caution_flags': caution_flags,
            'btc_etf_signal': btc_etf_signal.name,
            'eth_etf_signal': eth_etf_signal.name,
            'vix': vix_value,
            'vix_urgency': round(vix_urgency, 3),
            'vix_gradient': round(vix_gradient, 3),
            'vix_rising': vix_rising,
            'dxy_zscore': dxy_zscore,
            'etf_short_boost': round(etf_short_boost, 3),
            'btc_flow_streak': btc_flow_streak,
            'eth_flow_streak': eth_flow_streak,
        }
    
    def _etf_signal_to_float(self, signal: ETFFlowSignal) -> float:
        """Convert ETF signal to float for DRL."""
        mapping = {
            ETFFlowSignal.STRONG_INFLOW: 1.0,
            ETFFlowSignal.MODERATE_INFLOW: 0.5,
            ETFFlowSignal.NEUTRAL: 0.0,
            ETFFlowSignal.MODERATE_OUTFLOW: -0.5,
            ETFFlowSignal.STRONG_OUTFLOW: -1.0,
        }
        return mapping.get(signal, 0.0)
    
    async def _auto_update_loop(self):
        """Background auto-update loop."""
        while self._running:
            try:
                await self.update_async()
            except Exception as e:
                logger.error(f"Auto-update error: {e}")
            
            await asyncio.sleep(self.update_interval)
    
    def start(self):
        """Start background update loop."""
        if self._running:
            return
        
        self._running = True
        
        if self.enable_auto_update:
            loop = asyncio.get_event_loop()
            self._update_task = loop.create_task(self._auto_update_loop())
        
        logger.info("GlobalContextInformer started")
    
    def stop(self):
        """Stop background update loop."""
        self._running = False
        
        if self._update_task:
            self._update_task.cancel()
            self._update_task = None
        
        logger.info("GlobalContextInformer stopped")
    
    def update_crypto_price(self, asset: str, price: float):
        """Update crypto price for correlation calculation."""
        self.correlation_calculator.update_crypto_price(asset, price)


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Main class
    'GlobalContextInformer',
    
    # Data structures
    'GlobalContextState',
    'MacroIndicator',
    'ETFFlowData',
    'MacroCorrelation',
    
    # Enums
    'MacroRegime',
    'ETFFlowSignal',
    
    # Components
    'MacroDataFetcher',
    'ETFFlowTracker',
    'CorrelationCalculator',
    
    # Constants
    'MACRO_TICKERS',
    'SPOT_ETF_TICKERS',
    'CORRELATION_LOOKBACK_DAYS',
]


# ═══════════════════════════════════════════════════════════════════════════════
# TESTING
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    async def main():
        print("=" * 70)
        print("GlobalContextInformer Test")
        print("=" * 70)
        
        informer = GlobalContextInformer(
            update_interval_seconds=3600,
            enable_auto_update=False
        )
        
        # Single update
        state = await informer.update_async()
        
        print(f"\nMacro Indicators:")
        print(f"  DXY: {state.dxy.value:.2f} (z={state.dxy.zscore_30d:.2f})" if state.dxy else "  DXY: N/A")
        print(f"  US10Y: {state.us10y.value:.2f}%" if state.us10y else "  US10Y: N/A")
        print(f"  SPX: {state.spx.value:.0f}" if state.spx else "  SPX: N/A")
        print(f"  VIX: {state.vix.value:.1f}" if state.vix else "  VIX: N/A")
        
        print(f"\nETF Flows:")
        print(f"  BTC Total: ${state.btc_total_daily_flow/1e6:.1f}M (streak={state.btc_flow_streak})")
        print(f"  ETH Total: ${state.eth_total_daily_flow/1e6:.1f}M (streak={state.eth_flow_streak})")
        
        print(f"\nRegime Signals:")
        print(f"  Macro Regime: {state.macro_regime.name}")
        print(f"  BTC ETF Signal: {state.btc_etf_signal.name}")
        print(f"  DXY Breakout Caution: {state.dxy_breakout_caution}")
        print(f"  ETF Outflow Caution: {state.etf_outflow_caution}")
        
        print(f"\nDRL Features:")
        features = informer.get_drl_features()
        for k, v in features.items():
            print(f"  {k}: {v:.4f}")
        
        print(f"\n✓ Test completed in {state.update_latency_ms:.1f}ms")
    
    asyncio.run(main())
