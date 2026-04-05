"""
================================================================================
HMATS v6.1.0 - Kraken REST Client
================================================================================
Purpose: Unified REST API client for Kraken using ccxt

Implements critical functions that were previously placeholders:
- fetch_orderbook: Get L2 orderbook snapshot
- cancel_all_orders: Emergency order cancellation
- get_balance: Account balance query
- fetch_ticker: Current price data

Usage:
    from infra.kraken_rest_client import get_kraken_rest_client
    
    client = get_kraken_rest_client(api_key, api_secret)
    
    # Sync methods
    orderbook = client.fetch_orderbook("BTC/USD")
    client.cancel_all_orders()
    
    # Async methods
    orderbook = await client.async_fetch_orderbook("BTC/USD")

================================================================================
"""

import logging
import random
import time
import threading
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Retry configuration for transient errors
MAX_RETRIES = 3
BACKOFF_BASE_MS = 200     # 200ms, 400ms, 800ms
BACKOFF_JITTER = 0.1      # ±10% jitter

# Check for ccxt
CCXT_AVAILABLE = False
try:
    import ccxt
    import ccxt.async_support as ccxt_async
    CCXT_AVAILABLE = True
except ImportError:
    logger.warning("ccxt not available: pip install ccxt")


@dataclass
class OrderbookSnapshot:
    """Orderbook snapshot data."""
    symbol: str
    bids: List[Tuple[float, float]]  # (price, size)
    asks: List[Tuple[float, float]]
    timestamp: int  # ms
    datetime_str: str
    
    def mid_price(self) -> float:
        if self.bids and self.asks:
            return (self.bids[0][0] + self.asks[0][0]) / 2
        return 0.0
    
    def spread_bps(self) -> float:
        if self.bids and self.asks:
            mid = self.mid_price()
            if mid > 0:
                return (self.asks[0][0] - self.bids[0][0]) / mid * 10000
        return 0.0
    
    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "bids": self.bids[:10],  # Top 10
            "asks": self.asks[:10],
            "timestamp": self.timestamp,
            "mid_price": self.mid_price(),
            "spread_bps": self.spread_bps(),
        }


@dataclass 
class TickerData:
    """Ticker data."""
    symbol: str
    bid: float
    ask: float
    last: float
    volume_24h: float
    timestamp: int
    
    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "bid": self.bid,
            "ask": self.ask,
            "last": self.last,
            "volume_24h": self.volume_24h,
            "timestamp": self.timestamp,
        }


@dataclass
class BalanceData:
    """Account balance data."""
    balances: Dict[str, float] = field(default_factory=dict)
    timestamp: int = 0
    
    def get(self, asset: str) -> float:
        return self.balances.get(asset, 0.0)
    
    def total_usd(self, prices: Dict[str, float]) -> float:
        total = self.balances.get("USD", 0.0)
        for asset, amount in self.balances.items():
            if asset != "USD" and asset in prices:
                total += amount * prices[asset]
        return total


class KrakenRESTClient:
    """
    Kraken REST API client using ccxt.
    
    Thread-safe, with rate limiting handled by ccxt.
    """
    
    # Symbol mapping
    SYMBOL_MAP = {
        "BTC": "BTC/USD",
        "ETH": "ETH/USD", 
        "SOL": "SOL/USD",
        "XBTUSD": "BTC/USD",
        "ETHUSD": "ETH/USD",
        "SOLUSD": "SOL/USD",
    }
    
    def __init__(
        self,
        api_key: str = None,
        api_secret: str = None,
        sandbox: bool = False,
    ):
        self._api_key = api_key
        self._api_secret = api_secret
        self._sandbox = sandbox
        
        self._exchange: Optional[ccxt.kraken] = None
        self._async_exchange = None
        self._lock = threading.Lock()
        
        self._initialized = False
        self._last_error: Optional[str] = None
        
        # Stats
        self._request_count = 0
        self._error_count = 0
        
        self._init_exchange()
    
    def _init_exchange(self):
        """Initialize ccxt exchange."""
        if not CCXT_AVAILABLE:
            self._last_error = "ccxt not installed"
            logger.error("ccxt not available - install with: pip install ccxt")
            return
        
        try:
            config = {
                'enableRateLimit': True,
                'rateLimit': 1000,  # 1 request per second
                'options': {
                    'defaultType': 'spot',
                    'adjustForTimeDifference': True,
                }
            }
            
            if self._api_key and self._api_secret:
                config['apiKey'] = self._api_key
                config['secret'] = self._api_secret
            
            self._exchange = ccxt.kraken(config)
            
            if self._sandbox:
                self._exchange.set_sandbox_mode(True)
            
            self._initialized = True
            logger.info("[KrakenREST] Client initialized")
            
        except Exception as e:
            self._last_error = str(e)
            logger.error(f"[KrakenREST] Init failed: {e}")
    
    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol to ccxt format."""
        symbol = symbol.upper().replace("-", "/")
        return self.SYMBOL_MAP.get(symbol, symbol)

    def _retry_call(self, func, *args, operation: str = "API call", **kwargs):
        """
        Execute a ccxt call with retry + exponential backoff.

        Acquires self._lock per-attempt and releases during backoff sleep
        so other threads (e.g., different assets) are not blocked during retries.

        Retries on RateLimitExceeded and NetworkError only.
        AuthenticationError and other errors are NOT retried.
        """
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                with self._lock:
                    self._request_count += 1
                    return func(*args, **kwargs)

            except ccxt.AuthenticationError:
                raise  # Never retry auth errors

            except ccxt.InsufficientFunds:
                raise  # Never retry fund errors

            except (ccxt.RateLimitExceeded, ccxt.NetworkError) as e:
                last_error = e
                with self._lock:
                    self._error_count += 1
                backoff_ms = BACKOFF_BASE_MS * (2 ** attempt)
                jitter = backoff_ms * BACKOFF_JITTER * (2 * random.random() - 1)
                sleep_sec = (backoff_ms + jitter) / 1000.0
                logger.warning(
                    f"[KrakenREST] {operation} attempt {attempt + 1}/{MAX_RETRIES} "
                    f"failed ({type(e).__name__}), retrying in {sleep_sec:.1f}s"
                )
                time.sleep(sleep_sec)  # Sleep WITHOUT lock held

            except Exception as e:
                with self._lock:
                    self._error_count += 1
                logger.error(f"[KrakenREST] {operation} failed: {e}")
                return None

        # All retries exhausted
        logger.error(f"[KrakenREST] {operation} failed after {MAX_RETRIES} retries: {last_error}")
        return None

    # =========================================================================
    # SYNC METHODS
    # =========================================================================
    
    def fetch_orderbook(
        self,
        symbol: str,
        limit: int = 100,
    ) -> Optional[OrderbookSnapshot]:
        """
        Fetch orderbook snapshot with retry.

        Args:
            symbol: Trading pair (e.g., "BTC/USD", "BTC", "XBTUSD")
            limit: Depth limit

        Returns:
            OrderbookSnapshot or None on error
        """
        if not self._initialized:
            logger.warning("[KrakenREST] Not initialized")
            return None

        symbol = self._normalize_symbol(symbol)

        orderbook = self._retry_call(
            self._exchange.fetch_order_book, symbol, limit,
            operation=f"fetch_orderbook({symbol})"
        )
        if orderbook is None:
            return None

        return OrderbookSnapshot(
            symbol=symbol,
            bids=[(float(b[0]), float(b[1])) for b in orderbook['bids']],
            asks=[(float(a[0]), float(a[1])) for a in orderbook['asks']],
            timestamp=orderbook.get('timestamp', int(time.time() * 1000)),
            datetime_str=orderbook.get('datetime', datetime.now(timezone.utc).isoformat()),
        )
    
    def fetch_ticker(self, symbol: str) -> Optional[TickerData]:
        """
        Fetch ticker data with retry.

        Args:
            symbol: Trading pair

        Returns:
            TickerData or None on error
        """
        if not self._initialized:
            return None

        symbol = self._normalize_symbol(symbol)

        ticker = self._retry_call(
            self._exchange.fetch_ticker, symbol,
            operation=f"fetch_ticker({symbol})"
        )
        if ticker is None:
            return None

        return TickerData(
            symbol=symbol,
            bid=float(ticker.get('bid', 0)),
            ask=float(ticker.get('ask', 0)),
            last=float(ticker.get('last', 0)),
            volume_24h=float(ticker.get('quoteVolume', 0)),
            timestamp=ticker.get('timestamp', int(time.time() * 1000)),
        )
    
    def fetch_balance(self) -> Optional[BalanceData]:
        """
        Fetch account balance with retry.

        Requires API credentials.

        Returns:
            BalanceData or None on error
        """
        if not self._initialized:
            return None

        if not self._api_key:
            logger.warning("[KrakenREST] No API key for balance fetch")
            return None

        try:
            balance = self._retry_call(
                self._exchange.fetch_balance,
                operation="fetch_balance"
            )
            if balance is None:
                return None

            # Extract free balances
            balances = {}
            for asset, data in balance.get('free', {}).items():
                if data and float(data) > 0:
                    balances[asset] = float(data)

            return BalanceData(
                balances=balances,
                timestamp=int(time.time() * 1000),
            )

        except ccxt.AuthenticationError as e:
            with self._lock:
                self._error_count += 1
            logger.error(f"[KrakenREST] Auth error: {e}")
            return None
    
    def cancel_all_orders(self, symbol: str = None) -> bool:
        """
        Cancel all open orders.
        
        Args:
            symbol: Optional symbol to cancel orders for.
                   If None, cancels for all configured symbols.
        
        Returns:
            True if successful, False otherwise
        """
        if not self._initialized:
            logger.warning("[KrakenREST] Not initialized")
            return False
        
        if not self._api_key:
            logger.warning("[KrakenREST] No API key for cancel")
            return False
        
        symbols_to_cancel = [self._normalize_symbol(symbol)] if symbol else [
            "BTC/USD", "ETH/USD", "SOL/USD"
        ]
        
        with self._lock:
            all_success = True
            
            for sym in symbols_to_cancel:
                try:
                    self._request_count += 1
                    
                    # Fetch open orders first
                    open_orders = self._exchange.fetch_open_orders(sym)
                    
                    if not open_orders:
                        logger.info(f"[KrakenREST] No open orders for {sym}")
                        continue
                    
                    # Cancel each order
                    for order in open_orders:
                        try:
                            self._exchange.cancel_order(order['id'], sym)
                            logger.info(f"[KrakenREST] Cancelled order {order['id']}")
                        except Exception as e:
                            logger.warning(f"[KrakenREST] Failed to cancel {order['id']}: {e}")
                            all_success = False
                    
                except ccxt.OrderNotFound:
                    # No orders to cancel
                    continue
                    
                except Exception as e:
                    self._error_count += 1
                    logger.error(f"[KrakenREST] Cancel failed for {sym}: {e}")
                    all_success = False
            
            return all_success

    def cancel_all_orders_after(self, timeout_sec: int) -> bool:
        """
        Set Kraken server-side dead-man's switch.

        Kraken will cancel ALL open orders after `timeout_sec` seconds
        unless refreshed by another call. Send timeout=0 to disable.

        This is critical for crash safety: if the client dies, the
        server-side timer ensures orders are cleaned up automatically.

        Args:
            timeout_sec: Seconds until auto-cancel. 0 = disable timer.
                        Max 86400 (24h).

        Returns:
            True if successful, False otherwise
        """
        if not self._initialized:
            logger.warning("[KrakenREST] Not initialized for dead-man switch")
            return False

        if not self._api_key:
            logger.warning("[KrakenREST] No API key for dead-man switch")
            return False

        with self._lock:
            try:
                self._request_count += 1
                timeout_ms = timeout_sec * 1000
                result = self._exchange.cancel_all_orders_after(timeout_ms)
                if timeout_sec > 0:
                    logger.debug(f"[KrakenREST] Dead-man switch set: {timeout_sec}s")
                else:
                    logger.info("[KrakenREST] Dead-man switch disabled (timeout=0)")
                return True
            except Exception as e:
                self._error_count += 1
                logger.error(f"[KrakenREST] Dead-man switch failed: {e}")
                return False

    def place_order(
        self,
        symbol: str,
        side: str,  # "buy" or "sell"
        order_type: str,  # "market" or "limit"
        amount: float,
        price: float = None,
    ) -> Optional[Dict]:
        """
        Place an order.
        
        Args:
            symbol: Trading pair
            side: "buy" or "sell"
            order_type: "market" or "limit"
            amount: Order size
            price: Limit price (required for limit orders)
            
        Returns:
            Order info dict or None on error
        """
        if not self._initialized:
            return None
        
        if not self._api_key:
            logger.warning("[KrakenREST] No API key for order")
            return None
        
        symbol = self._normalize_symbol(symbol)
        
        with self._lock:
            try:
                self._request_count += 1
                
                if order_type == "market":
                    order = self._exchange.create_market_order(
                        symbol, side, amount
                    )
                else:
                    if price is None:
                        logger.error("[KrakenREST] Limit order requires price")
                        return None
                    order = self._exchange.create_limit_order(
                        symbol, side, amount, price
                    )
                
                logger.info(f"[KrakenREST] Order placed: {order.get('id')}")
                return order
                
            except ccxt.InsufficientFunds as e:
                self._error_count += 1
                logger.error(f"[KrakenREST] Insufficient funds: {e}")
                return None
                
            except Exception as e:
                self._error_count += 1
                logger.error(f"[KrakenREST] Order failed: {e}")
                return None
    
    # =========================================================================
    # STATUS
    # =========================================================================
    
    def is_initialized(self) -> bool:
        """Check if client is initialized."""
        return self._initialized
    
    def get_stats(self) -> Dict:
        """Get client statistics."""
        return {
            "initialized": self._initialized,
            "request_count": self._request_count,
            "error_count": self._error_count,
            "last_error": self._last_error,
            "has_credentials": bool(self._api_key),
            "sandbox_mode": self._sandbox,
        }
    
    def test_connection(self) -> bool:
        """Test connection to Kraken."""
        try:
            ticker = self.fetch_ticker("BTC/USD")
            return ticker is not None and ticker.last > 0
        except Exception as e:
            logger.error(f"[KrakenREST] Connection test failed: {e}")
            return False


# =============================================================================
# ASYNC CLIENT
# =============================================================================

class KrakenRESTClientAsync:
    """
    Async version of Kraken REST client.
    
    For use in async contexts (e.g., WebSocket handlers).
    """
    
    SYMBOL_MAP = KrakenRESTClient.SYMBOL_MAP
    
    def __init__(
        self,
        api_key: str = None,
        api_secret: str = None,
        sandbox: bool = False,
    ):
        self._api_key = api_key
        self._api_secret = api_secret
        self._sandbox = sandbox
        
        self._exchange = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize async exchange."""
        if not CCXT_AVAILABLE:
            logger.error("ccxt not available")
            return False
        
        try:
            config = {
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',
                }
            }
            
            if self._api_key:
                config['apiKey'] = self._api_key
                config['secret'] = self._api_secret
            
            self._exchange = ccxt_async.kraken(config)
            
            if self._sandbox:
                self._exchange.set_sandbox_mode(True)
            
            self._initialized = True
            logger.info("[KrakenRESTAsync] Initialized")
            return True
            
        except Exception as e:
            logger.error(f"[KrakenRESTAsync] Init failed: {e}")
            return False
    
    async def close(self):
        """Close async exchange."""
        if self._exchange:
            await self._exchange.close()
    
    def _normalize_symbol(self, symbol: str) -> str:
        symbol = symbol.upper().replace("-", "/")
        return self.SYMBOL_MAP.get(symbol, symbol)
    
    async def fetch_orderbook(
        self,
        symbol: str,
        limit: int = 100,
    ) -> Optional[OrderbookSnapshot]:
        """Async fetch orderbook."""
        if not self._initialized:
            return None
        
        symbol = self._normalize_symbol(symbol)
        
        try:
            orderbook = await self._exchange.fetch_order_book(symbol, limit)
            
            return OrderbookSnapshot(
                symbol=symbol,
                bids=[(float(b[0]), float(b[1])) for b in orderbook['bids']],
                asks=[(float(a[0]), float(a[1])) for a in orderbook['asks']],
                timestamp=orderbook.get('timestamp', int(time.time() * 1000)),
                datetime_str=orderbook.get('datetime', datetime.now(timezone.utc).isoformat()),
            )
            
        except Exception as e:
            logger.error(f"[KrakenRESTAsync] Orderbook fetch failed: {e}")
            return None
    
    async def cancel_all_orders(self, symbol: str = None) -> bool:
        """Async cancel all orders."""
        if not self._initialized or not self._api_key:
            return False
        
        symbols = [self._normalize_symbol(symbol)] if symbol else [
            "BTC/USD", "ETH/USD", "SOL/USD"
        ]
        
        all_success = True
        
        for sym in symbols:
            try:
                open_orders = await self._exchange.fetch_open_orders(sym)
                
                for order in open_orders:
                    try:
                        await self._exchange.cancel_order(order['id'], sym)
                    except Exception as e:
                        logger.warning(f"[KrakenRESTAsync] Cancel failed: {e}")
                        all_success = False
                        
            except Exception as e:
                logger.error(f"[KrakenRESTAsync] Cancel all failed for {sym}: {e}")
                all_success = False
        
        return all_success


# =============================================================================
# SINGLETON
# =============================================================================

_kraken_client: Optional[KrakenRESTClient] = None
_kraken_lock = threading.Lock()


def get_kraken_rest_client(
    api_key: str = None,
    api_secret: str = None,
    sandbox: bool = False,
) -> KrakenRESTClient:
    """Get singleton Kraken REST client."""
    global _kraken_client
    
    if _kraken_client is None:
        with _kraken_lock:
            if _kraken_client is None:
                _kraken_client = KrakenRESTClient(
                    api_key=api_key,
                    api_secret=api_secret,
                    sandbox=sandbox,
                )
    
    return _kraken_client


def reset_kraken_rest_client():
    """Reset singleton (for testing)."""
    global _kraken_client
    with _kraken_lock:
        _kraken_client = None


if __name__ == "__main__":
    # Test
    client = get_kraken_rest_client()
    
    if client.is_initialized():
        print("Stats:", client.get_stats())
        
        # Test connection
        if client.test_connection():
            print("Connection OK")
            
            # Fetch orderbook
            ob = client.fetch_orderbook("BTC/USD")
            if ob:
                print(f"BTC/USD: mid={ob.mid_price():.2f}, spread={ob.spread_bps():.2f}bps")
            
            # Fetch ticker
            ticker = client.fetch_ticker("ETH/USD")
            if ticker:
                print(f"ETH/USD: last={ticker.last:.2f}")
        else:
            print("Connection failed")
    else:
        print("Client not initialized")
