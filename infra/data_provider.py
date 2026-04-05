"""
HMATS v3.2 - Real Market Data Provider
Production-grade market data fetching with fail-safe mechanisms.

P0-1 Fix: Replace hardcoded market data with real provider.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class DataSource(Enum):
    """Market data sources."""
    KRAKEN = "kraken"
    BINANCE = "binance"
    CCXT = "ccxt"
    MOCK = "mock"  # Only for testing


@dataclass
class MarketDataConfig:
    """Configuration for market data provider."""
    # Source configuration
    primary_source: DataSource = DataSource.CCXT
    fallback_source: Optional[DataSource] = DataSource.MOCK
    
    # Safety flags
    ENABLE_LIVE_DATA: bool = False  # Default FALSE for safety
    ALLOW_MOCK_IN_PRODUCTION: bool = False
    
    # Data requirements
    required_assets: List[str] = field(default_factory=lambda: ["BTC", "ETH", "SOL"])
    ohlcv_timeframe: str = "4h"
    ohlcv_history_bars: int = 200  # For indicator calculation
    
    # Staleness thresholds
    max_data_age_seconds: float = 30.0  # Data must be < 30s old
    max_price_deviation_pct: float = 10.0  # Alert if price deviates > 10%
    
    # Retry configuration
    max_retries: int = 3
    retry_delay_seconds: float = 1.0


@dataclass
class OHLCVBar:
    """Single OHLCV bar."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class AssetMarketData:
    """Market data for a single asset."""
    asset: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    returns_4h: float = 0.0
    volatility_4h: float = 0.02
    ohlcv_history: Optional[List[OHLCVBar]] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format."""
        result = {
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "returns_4h": self.returns_4h,
            "volatility_4h": self.volatility_4h,
        }
        if self.ohlcv_history:
            import pandas as pd
            result["ohlcv_history"] = pd.DataFrame([
                {
                    "timestamp": bar.timestamp,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume
                }
                for bar in self.ohlcv_history
            ])
        return result


@dataclass
class MarketDataResult:
    """Result of market data fetch."""
    success: bool
    data: Optional[Dict] = None
    error: Optional[str] = None
    source: DataSource = DataSource.MOCK
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    data_age_seconds: float = 0.0
    is_mock: bool = False
    
    def to_canonical(self) -> Dict:
        """Convert to canonical market data format."""
        if not self.success or not self.data:
            return {}
        
        result = dict(self.data)
        result["timestamp"] = self.timestamp
        result["data_age_seconds"] = self.data_age_seconds
        result["_source"] = self.source.value
        result["_is_mock"] = self.is_mock
        return result


class MarketDataProvider:
    """
    Production market data provider with fail-safe mechanisms.
    
    Features:
    - Multiple data sources (CCXT, Kraken, Binance)
    - Automatic fallback on failure
    - Data validation and staleness checks
    - Explicit mock mode (never silent)
    """
    
    def __init__(self, config: Optional[MarketDataConfig] = None):
        self.config = config or MarketDataConfig()
        self._ccxt_exchange = None
        self._requests_session = None
        self._last_prices: Dict[str, float] = {}
        self._mock_warning_logged = False
        
        # Initialize CCXT if available and enabled
        if self.config.ENABLE_LIVE_DATA:
            self._init_ccxt()
        else:
            logger.warning(
                "[MarketDataProvider] LIVE DATA DISABLED - using mock data. "
                "Set ENABLE_LIVE_DATA=True for production."
            )
    
    def _init_ccxt(self):
        """Initialize CCXT exchange connection."""
        try:
            import ccxt
            self._ccxt_exchange = ccxt.kraken({
                'enableRateLimit': True,
                'timeout': 30000,
            })
            logger.info("[MarketDataProvider] CCXT/Kraken initialized successfully")
        except ImportError:
            logger.error("[MarketDataProvider] ccxt not installed: pip install ccxt")
            self._ccxt_exchange = None
        except Exception as e:
            logger.error(f"[MarketDataProvider] CCXT init failed: {e}")
            self._ccxt_exchange = None

    def _get_requests_session(self):
        """Get shared requests session for direct REST fallbacks."""
        if self._requests_session is not None:
            return self._requests_session
        try:
            import requests
        except ImportError:
            return None
        self._requests_session = requests.Session()
        self._requests_session.headers.update({"User-Agent": "HMATS/MarketDataProvider"})
        return self._requests_session

    def close(self):
        """Close any persistent network clients."""
        if self._ccxt_exchange is not None and hasattr(self._ccxt_exchange, "close"):
            try:
                self._ccxt_exchange.close()
            except Exception:
                pass
        if self._requests_session is not None:
            try:
                self._requests_session.close()
            except Exception:
                pass
            self._requests_session = None
    
    def fetch(self) -> MarketDataResult:
        """
        Fetch market data from configured sources.
        
        Returns:
            MarketDataResult with success/failure and data
        """
        # Check if live data is enabled
        if not self.config.ENABLE_LIVE_DATA:
            return self._get_mock_data(reason="ENABLE_LIVE_DATA=False")
        
        # Try primary source
        result = self._fetch_from_source(self.config.primary_source)
        if result.success:
            return result
        
        # Try fallback
        if self.config.fallback_source:
            logger.warning(
                f"[MarketDataProvider] Primary source failed, trying fallback: "
                f"{self.config.fallback_source.value}"
            )
            result = self._fetch_from_source(self.config.fallback_source)
            if result.success:
                return result
        
        # All sources failed - return mock with explicit warning
        logger.error("[MarketDataProvider] ALL SOURCES FAILED - using mock as last resort")
        return self._get_mock_data(reason="All sources failed")
    
    def _fetch_from_source(self, source: DataSource) -> MarketDataResult:
        """Fetch from a specific source."""
        try:
            if source == DataSource.CCXT:
                return self._fetch_ccxt()
            elif source == DataSource.KRAKEN:
                return self._fetch_kraken_direct()
            elif source == DataSource.BINANCE:
                return self._fetch_binance_direct()
            elif source == DataSource.MOCK:
                return self._get_mock_data(reason="Mock source selected")
            else:
                return MarketDataResult(
                    success=False,
                    error=f"Unknown source: {source}"
                )
        except Exception as e:
            logger.error(f"[MarketDataProvider] {source.value} fetch failed: {e}")
            return MarketDataResult(success=False, error=str(e))
    
    def _fetch_ccxt(self) -> MarketDataResult:
        """Fetch via CCXT library."""
        if not self._ccxt_exchange:
            return MarketDataResult(
                success=False,
                error="CCXT not initialized"
            )
        
        try:
            fetch_start = time.time()
            data = {}
            
            # Symbol mapping for Kraken
            symbols = {
                "BTC": "BTC/USD",
                "ETH": "ETH/USD",
                "SOL": "SOL/USD"
            }
            
            for asset, symbol in symbols.items():
                try:
                    # Fetch ticker for current price
                    ticker = self._ccxt_exchange.fetch_ticker(symbol)
                    
                    # Fetch OHLCV history
                    ohlcv = self._ccxt_exchange.fetch_ohlcv(
                        symbol,
                        timeframe=self.config.ohlcv_timeframe,
                        limit=self.config.ohlcv_history_bars
                    )
                    
                    # Current bar
                    current = ohlcv[-1] if ohlcv else None
                    prev = ohlcv[-2] if len(ohlcv) >= 2 else None
                    
                    # Calculate returns
                    returns_4h = 0.0
                    if current and prev and prev[4] > 0:  # prev close > 0
                        returns_4h = (current[4] - prev[4]) / prev[4]
                    
                    # Calculate volatility (std of returns over last 20 bars)
                    volatility_4h = 0.02  # default
                    if len(ohlcv) >= 20:
                        closes = [bar[4] for bar in ohlcv[-20:]]
                        returns = [(closes[i] - closes[i-1]) / closes[i-1] 
                                   for i in range(1, len(closes)) if closes[i-1] > 0]
                        if returns:
                            import statistics
                            volatility_4h = statistics.stdev(returns) if len(returns) > 1 else 0.02
                    
                    # Build OHLCV history
                    history = [
                        OHLCVBar(
                            timestamp=datetime.fromtimestamp(bar[0] / 1000, tz=timezone.utc),
                            open=bar[1],
                            high=bar[2],
                            low=bar[3],
                            close=bar[4],
                            volume=bar[5]
                        )
                        for bar in ohlcv
                    ]
                    
                    asset_data = AssetMarketData(
                        asset=asset,
                        open=ticker.get('open', current[1] if current else 0),
                        high=ticker.get('high', current[2] if current else 0),
                        low=ticker.get('low', current[3] if current else 0),
                        close=ticker.get('last', current[4] if current else 0),
                        volume=ticker.get('baseVolume', current[5] if current else 0),
                        returns_4h=returns_4h,
                        volatility_4h=volatility_4h,
                        ohlcv_history=history
                    )
                    
                    data[asset] = asset_data.to_dict()
                    self._last_prices[asset] = asset_data.close
                    
                except Exception as e:
                    logger.error(f"[MarketDataProvider] Failed to fetch {asset}: {e}")
                    return MarketDataResult(
                        success=False,
                        error=f"Failed to fetch {asset}: {e}"
                    )
            
            data_age = time.time() - fetch_start
            
            # Validate completeness
            missing = [a for a in self.config.required_assets if a not in data]
            if missing:
                return MarketDataResult(
                    success=False,
                    error=f"Missing required assets: {missing}"
                )
            
            logger.info(
                f"[MarketDataProvider] CCXT fetch successful: "
                f"BTC=${data['BTC']['close']:.0f}, "
                f"ETH=${data['ETH']['close']:.0f}, "
                f"SOL=${data['SOL']['close']:.2f}, "
                f"age={data_age:.2f}s"
            )
            
            return MarketDataResult(
                success=True,
                data=data,
                source=DataSource.CCXT,
                data_age_seconds=data_age,
                is_mock=False
            )
            
        except Exception as e:
            logger.error(f"[MarketDataProvider] CCXT fetch error: {e}")
            return MarketDataResult(success=False, error=str(e))
    
    def _fetch_kraken_direct(self) -> MarketDataResult:
        """Direct Kraken REST API fetch (fallback if CCXT fails)."""
        try:
            session = self._get_requests_session()
            if session is None:
                logger.error("[MarketDataProvider] requests not installed")
                return MarketDataResult(success=False, error="requests not installed")
            
            fetch_start = time.time()
            data = {}
            
            pairs = {
                "BTC": "XXBTZUSD",
                "ETH": "XETHZUSD",
                "SOL": "SOLUSD"
            }
            
            for asset, pair in pairs.items():
                # Ticker
                ticker_url = f"https://api.kraken.com/0/public/Ticker?pair={pair}"
                ticker_resp = session.get(ticker_url, timeout=10)
                ticker_resp.raise_for_status()
                ticker_data = ticker_resp.json()
                
                if ticker_data.get("error"):
                    raise Exception(f"Kraken error: {ticker_data['error']}")
                
                result_key = list(ticker_data["result"].keys())[0]
                ticker = ticker_data["result"][result_key]
                
                # OHLCV
                ohlcv_url = f"https://api.kraken.com/0/public/OHLC?pair={pair}&interval=240"
                ohlcv_resp = session.get(ohlcv_url, timeout=10)
                ohlcv_resp.raise_for_status()
                ohlcv_data = ohlcv_resp.json()
                
                if ohlcv_data.get("error"):
                    raise Exception(f"Kraken OHLCV error: {ohlcv_data['error']}")
                
                result_key = list(ohlcv_data["result"].keys())[0]
                if result_key == "last":
                    result_key = list(ohlcv_data["result"].keys())[1]
                ohlcv = ohlcv_data["result"][result_key]
                
                # Parse OHLCV
                current = ohlcv[-1] if ohlcv else None
                prev = ohlcv[-2] if len(ohlcv) >= 2 else None
                
                returns_4h = 0.0
                if current and prev:
                    prev_close = float(prev[4])
                    curr_close = float(current[4])
                    if prev_close > 0:
                        returns_4h = (curr_close - prev_close) / prev_close
                
                # Build history
                history = [
                    OHLCVBar(
                        timestamp=datetime.fromtimestamp(int(bar[0]), tz=timezone.utc),
                        open=float(bar[1]),
                        high=float(bar[2]),
                        low=float(bar[3]),
                        close=float(bar[4]),
                        volume=float(bar[6])
                    )
                    for bar in ohlcv[-self.config.ohlcv_history_bars:]
                ]
                
                asset_data = AssetMarketData(
                    asset=asset,
                    open=float(ticker['o']),
                    high=float(ticker['h'][0]),
                    low=float(ticker['l'][0]),
                    close=float(ticker['c'][0]),
                    volume=float(ticker['v'][1]),
                    returns_4h=returns_4h,
                    ohlcv_history=history
                )
                
                data[asset] = asset_data.to_dict()
                self._last_prices[asset] = asset_data.close
            
            data_age = time.time() - fetch_start
            
            logger.info(
                f"[MarketDataProvider] Kraken direct fetch successful: "
                f"BTC=${data['BTC']['close']:.0f}, "
                f"ETH=${data['ETH']['close']:.0f}, "
                f"SOL=${data['SOL']['close']:.2f}"
            )
            
            return MarketDataResult(
                success=True,
                data=data,
                source=DataSource.KRAKEN,
                data_age_seconds=data_age,
                is_mock=False
            )
            
        except Exception as e:
            logger.error(f"[MarketDataProvider] Kraken direct fetch error: {e}")
            return MarketDataResult(success=False, error=str(e))
    
    def _fetch_binance_direct(self) -> MarketDataResult:
        """Direct Binance REST API fetch."""
        try:
            session = self._get_requests_session()
            if session is None:
                logger.error("[MarketDataProvider] requests not installed")
                return MarketDataResult(success=False, error="requests not installed")
            
            fetch_start = time.time()
            data = {}
            
            symbols = {
                "BTC": "BTCUSDT",
                "ETH": "ETHUSDT",
                "SOL": "SOLUSDT"
            }
            
            for asset, symbol in symbols.items():
                # Ticker
                ticker_url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
                ticker_resp = session.get(ticker_url, timeout=10)
                ticker_resp.raise_for_status()
                ticker = ticker_resp.json()
                
                # Klines (OHLCV)
                klines_url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=4h&limit={self.config.ohlcv_history_bars}"
                klines_resp = session.get(klines_url, timeout=10)
                klines_resp.raise_for_status()
                klines = klines_resp.json()
                
                current = klines[-1] if klines else None
                prev = klines[-2] if len(klines) >= 2 else None
                
                returns_4h = 0.0
                if current and prev:
                    prev_close = float(prev[4])
                    curr_close = float(current[4])
                    if prev_close > 0:
                        returns_4h = (curr_close - prev_close) / prev_close
                
                history = [
                    OHLCVBar(
                        timestamp=datetime.fromtimestamp(int(bar[0]) / 1000, tz=timezone.utc),
                        open=float(bar[1]),
                        high=float(bar[2]),
                        low=float(bar[3]),
                        close=float(bar[4]),
                        volume=float(bar[5])
                    )
                    for bar in klines
                ]
                
                asset_data = AssetMarketData(
                    asset=asset,
                    open=float(ticker['openPrice']),
                    high=float(ticker['highPrice']),
                    low=float(ticker['lowPrice']),
                    close=float(ticker['lastPrice']),
                    volume=float(ticker['volume']),
                    returns_4h=returns_4h,
                    ohlcv_history=history
                )
                
                data[asset] = asset_data.to_dict()
                self._last_prices[asset] = asset_data.close
            
            data_age = time.time() - fetch_start
            
            logger.info(
                f"[MarketDataProvider] Binance fetch successful: "
                f"BTC=${data['BTC']['close']:.0f}, "
                f"ETH=${data['ETH']['close']:.0f}, "
                f"SOL=${data['SOL']['close']:.2f}"
            )
            
            return MarketDataResult(
                success=True,
                data=data,
                source=DataSource.BINANCE,
                data_age_seconds=data_age,
                is_mock=False
            )
            
        except Exception as e:
            logger.error(f"[MarketDataProvider] Binance fetch error: {e}")
            return MarketDataResult(success=False, error=str(e))
    
    def _get_mock_data(self, reason: str = "") -> MarketDataResult:
        """
        Generate mock data with EXPLICIT warning.
        Never silent - always logged.
        """
        if not self._mock_warning_logged:
            logger.warning(
                f"[MarketDataProvider] [WARN]️ USING MOCK DATA - NOT FOR PRODUCTION [WARN]️\n"
                f"  Reason: {reason}\n"
                f"  This data is FAKE and should NOT be used for real trading decisions."
            )
            self._mock_warning_logged = True
        
        # Use last known prices if available, otherwise defaults
        btc_price = self._last_prices.get("BTC", 95000.0)
        eth_price = self._last_prices.get("ETH", 3200.0)
        sol_price = self._last_prices.get("SOL", 180.0)
        
        data = {
            "BTC": {
                "open": btc_price * 0.995,
                "high": btc_price * 1.01,
                "low": btc_price * 0.99,
                "close": btc_price,
                "volume": 1000.0,
                "returns_4h": 0.005,
                "volatility_4h": 0.02,
            },
            "ETH": {
                "open": eth_price * 0.995,
                "high": eth_price * 1.01,
                "low": eth_price * 0.99,
                "close": eth_price,
                "volume": 50000.0,
                "returns_4h": 0.003,
                "volatility_4h": 0.025,
            },
            "SOL": {
                "open": sol_price * 0.995,
                "high": sol_price * 1.02,
                "low": sol_price * 0.98,
                "close": sol_price,
                "volume": 500000.0,
                "returns_4h": 0.008,
                "volatility_4h": 0.04,
            },
        }
        
        return MarketDataResult(
            success=True,
            data=data,
            source=DataSource.MOCK,
            data_age_seconds=0.0,
            is_mock=True  # EXPLICITLY marked as mock
        )


# =============================================================================
# Singleton instance
# =============================================================================

_market_data_provider: Optional[MarketDataProvider] = None


def get_market_data_provider(config: Optional[MarketDataConfig] = None) -> MarketDataProvider:
    """Get or create market data provider singleton."""
    global _market_data_provider
    if _market_data_provider is None:
        _market_data_provider = MarketDataProvider(config)
    return _market_data_provider


def reset_market_data_provider() -> None:
    """Close and reset singleton provider."""
    global _market_data_provider
    if _market_data_provider is not None:
        _market_data_provider.close()
    _market_data_provider = None


def fetch_market_data(config: Optional[MarketDataConfig] = None) -> MarketDataResult:
    """Convenience function to fetch market data."""
    provider = get_market_data_provider(config)
    return provider.fetch()


# =============================================================================
# Fail-safe validation
# =============================================================================

def validate_market_data_for_trading(result: MarketDataResult) -> Tuple[bool, str]:
    """
    Validate market data is suitable for trading decisions.
    
    Returns:
        (is_valid, error_message)
    """
    if not result.success:
        return False, f"Market data fetch failed: {result.error}"
    
    if result.is_mock:
        return False, "Mock data detected - cannot use for trading"
    
    if result.data_age_seconds > 30.0:
        return False, f"Data too stale: {result.data_age_seconds:.1f}s > 30s"
    
    # Check required assets
    required = ["BTC", "ETH", "SOL"]
    for asset in required:
        if asset not in result.data:
            return False, f"Missing required asset: {asset}"
        
        asset_data = result.data[asset]
        if not isinstance(asset_data.get("close"), (int, float)):
            return False, f"Invalid close price for {asset}"
        
        if asset_data.get("close", 0) <= 0:
            return False, f"Non-positive close price for {asset}"
    
    return True, ""


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Market Data Provider Test")
    print("=" * 60)
    
    # Test 1: Mock mode (default)
    print("\n[Test 1] Default mode (mock)...")
    config = MarketDataConfig(ENABLE_LIVE_DATA=False)
    provider = MarketDataProvider(config)
    result = provider.fetch()
    print(f"  Success: {result.success}")
    print(f"  Source: {result.source.value}")
    print(f"  Is Mock: {result.is_mock}")
    
    is_valid, error = validate_market_data_for_trading(result)
    print(f"  Valid for trading: {is_valid}")
    if not is_valid:
        print(f"  Reason: {error}")
    
    # Test 2: Live mode
    print("\n[Test 2] Live mode (CCXT)...")
    config = MarketDataConfig(ENABLE_LIVE_DATA=True)
    reset_market_data_provider()
    result = fetch_market_data(config)
    print(f"  Success: {result.success}")
    print(f"  Source: {result.source.value}")
    print(f"  Is Mock: {result.is_mock}")
    
    if result.success and result.data:
        for asset in ["BTC", "ETH", "SOL"]:
            if asset in result.data:
                print(f"  {asset}: ${result.data[asset]['close']:.2f}")
    
    is_valid, error = validate_market_data_for_trading(result)
    print(f"  Valid for trading: {is_valid}")
    if not is_valid:
        print(f"  Reason: {error}")
    
    print("\n" + "=" * 60)
