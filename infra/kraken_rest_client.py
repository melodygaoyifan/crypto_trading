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

import hashlib
import json
import logging
import os
import random
import time
import threading
import traceback
from pathlib import Path
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
        # Last dead-man-switch error text — read by DeadManSwitch.refresh()
        # so consecutive-failure WARNING/CRITICAL log carries the cause.
        self._last_dead_man_error: Optional[str] = None

        # Monotonic nonce ratchet, persisted to disk per API-key fingerprint.
        # Kraken's last-seen nonce per key survives across our process
        # restarts; a fresh in-memory ratchet at process start can land
        # below Kraken's tracker even with load_time_difference(). The
        # persistent state stores the highest nonce we've successfully used
        # so the next process boots above it. The state file is keyed by
        # an HMAC of the API key (not the key itself) so multiple keys can
        # coexist and a key rotation isn't poisoned by old state.
        self._nonce_lock = threading.Lock()
        self._nonce_state_path = Path(
            os.environ.get("HMATS_KRAKEN_NONCE_STATE")
            or (Path(__file__).resolve().parents[1] / "data" / "kraken_nonce_state.json")
        )
        self._nonce_key_fingerprint = self._compute_key_fingerprint(api_key)
        self._nonce_ratchet_ms = self._load_persisted_nonce()
        self._nonce_dirty = False  # set when ratchet advances; flushed periodically

        self._init_exchange()

    @staticmethod
    def _compute_key_fingerprint(api_key: Optional[str]) -> str:
        """Stable fingerprint for the API key (NOT the key itself).

        Used to key persisted nonce state so a key rotation starts fresh.
        """
        if not api_key:
            return "no_key"
        return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]

    def _load_persisted_nonce(self) -> int:
        """Load the highest nonce we've previously issued for this key.

        Returns 0 if no state file or no entry for this key fingerprint.
        Caller will then fall back to clock-time on first nonce request.
        """
        try:
            if not self._nonce_state_path.exists():
                return 0
            with open(self._nonce_state_path, "r") as f:
                state = json.load(f) or {}
            entry = state.get(self._nonce_key_fingerprint, {})
            persisted = int(entry.get("highest_nonce_ms", 0))
            if persisted > 0:
                logger.info(
                    f"[KrakenREST] Loaded persisted nonce ratchet for "
                    f"key fp={self._nonce_key_fingerprint}: {persisted}"
                )
            return persisted
        except Exception as e:
            logger.warning(f"[KrakenREST] Could not load nonce state: {e}")
            return 0

    def _persist_nonce_locked(self) -> None:
        """Atomic-write the current ratchet to the state file.

        Caller MUST hold self._nonce_lock. Writes the WHOLE state map so
        multi-key state survives. Best-effort: failures are logged but
        don't break the request path.
        """
        try:
            self._nonce_state_path.parent.mkdir(parents=True, exist_ok=True)
            existing: Dict[str, Any] = {}
            if self._nonce_state_path.exists():
                try:
                    with open(self._nonce_state_path, "r") as f:
                        existing = json.load(f) or {}
                except Exception:
                    existing = {}
            existing[self._nonce_key_fingerprint] = {
                "highest_nonce_ms": int(self._nonce_ratchet_ms),
                "updated_at": int(time.time()),
            }
            tmp = self._nonce_state_path.with_suffix(".json.tmp")
            with open(tmp, "w") as f:
                json.dump(existing, f, separators=(",", ":"))
            os.replace(tmp, self._nonce_state_path)
            self._nonce_dirty = False
        except Exception as e:
            logger.warning(f"[KrakenREST] Could not persist nonce state: {e}")

    def _monotonic_nonce(self) -> int:
        """Strictly-monotonic nonce in milliseconds.

        Returns max(local_ms, persisted_high+1). Thread-safe so concurrent
        REST calls (e.g. dead-man heartbeat thread + main async path) don't
        emit duplicate nonces. Marks state dirty for periodic flush; we don't
        flush every call to avoid I/O on the hot path.
        """
        with self._nonce_lock:
            now_ms = int(time.time() * 1000)
            if self._nonce_ratchet_ms < now_ms:
                self._nonce_ratchet_ms = now_ms
            else:
                self._nonce_ratchet_ms += 1
            self._nonce_dirty = True
            # Periodic flush — every ~30s of issued nonces. Cheap I/O,
            # bounds worst-case loss to one window if the process dies.
            if self._nonce_ratchet_ms % 30000 < 5:  # opportunistic
                self._persist_nonce_locked()
            return self._nonce_ratchet_ms

    def _bump_nonce_past_error(self, seconds: int = 60) -> int:
        """Jump the nonce ratchet forward by `seconds` after a nonce error.

        Used by retry paths in cancel_all_orders_after / fetch_balance when
        Kraken returns 'Invalid nonce'. A single `load_time_difference()`
        only corrects clock skew (typically < 1s); if the previous process
        left Kraken with a higher last-seen nonce, we need to jump past it.

        Persists immediately so a crash mid-bump doesn't lose ground.

        Returns:
            The new ratchet value (in ms).
        """
        with self._nonce_lock:
            self._nonce_ratchet_ms = max(
                self._nonce_ratchet_ms, int(time.time() * 1000)
            ) + (seconds * 1000)
            self._nonce_dirty = True
            self._persist_nonce_locked()  # bumps are rare; flush every time
            return self._nonce_ratchet_ms
    
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

            # Replace ccxt's default nonce method with our ratchet so every
            # path going through this exchange (fetch_balance, place_order,
            # cancel_all_orders_after, etc.) gets a strictly-monotonic nonce.
            # ccxt expects the method to be bound to the exchange; bind via
            # closure so it can call back into our ratchet.
            self._exchange.nonce = self._monotonic_nonce  # type: ignore[assignment]

            # [NONCE-FIX] Pre-warm time difference so first authenticated call
            # uses a Kraken-aligned nonce. Without this, the very first auth call
            # (e.g. dead-man switch on startup) sends a local-time-only nonce that
            # may lag behind what Kraken last saw from a previous container.
            if self._api_key and self._api_secret:
                try:
                    self._exchange.load_time_difference()
                    logger.info(
                        f"[KrakenREST] Time difference pre-loaded: "
                        f"{self._exchange.options.get('timeDifference', 0)}ms; "
                        f"nonce_ratchet={self._nonce_ratchet_ms}"
                    )
                except Exception as _td_err:
                    logger.warning(f"[KrakenREST] load_time_difference failed: {_td_err}")

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
        _nonce_retried = False
        for attempt in range(MAX_RETRIES):
            try:
                with self._lock:
                    self._request_count += 1
                    return func(*args, **kwargs)

            except ccxt.AuthenticationError as e:
                # [NONCE-FIX 2026-04-15] Kraken's "EAPI:Invalid nonce" is an
                # AuthenticationError in ccxt. Reload time difference and retry
                # ONCE — covers the case where a previous container left a
                # higher last-seen nonce. Real auth failures (bad keys) re-raise
                # because load_time_difference will succeed but next call still
                # fails with auth-related error other than Invalid nonce.
                if not _nonce_retried and "Invalid nonce" in str(e):
                    _nonce_retried = True
                    try:
                        with self._lock:
                            self._exchange.load_time_difference()
                        logger.warning(
                            f"[KrakenREST] {operation}: nonce mismatch, "
                            f"reloaded timeDifference, retrying once"
                        )
                        continue
                    except Exception as _td_err:
                        logger.error(f"[KrakenREST] reload time_difference failed: {_td_err}")
                raise  # Never retry other auth errors

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
            timeout_ms = timeout_sec * 1000
            # 3 attempts: original + nonce ratchet bump + clock skew reload.
            # Order matters: bump ratchet first (likely cause), then reload
            # clock as fallback. Production logs on 2026-04-25 showed nonce
            # errors that single load_time_difference() couldn't fix.
            attempt_errors: List[str] = []
            for attempt in range(3):
                try:
                    self._request_count += 1
                    self._exchange.cancel_all_orders_after(timeout_ms)
                    if timeout_sec > 0:
                        logger.debug(f"[KrakenREST] Dead-man switch set: {timeout_sec}s")
                    else:
                        logger.info("[KrakenREST] Dead-man switch disabled (timeout=0)")
                    self._last_dead_man_error = None
                    return True
                except Exception as e:
                    self._error_count += 1
                    _err_str = str(e)
                    # Build a verbose, non-truncating error description. ccxt's
                    # default str(e) on auth errors is just the URL — actual
                    # Kraken response (EAPI:..., {"error":[...]}, etc.) lives in
                    # e.args. Concatenate every arg so the response body shows.
                    _arg_parts: List[str] = []
                    for _a in (getattr(e, "args", ()) or ()):
                        try:
                            _s = _a if isinstance(_a, str) else repr(_a)
                        except Exception:
                            _s = "<unrepr>"
                        if _s and _s not in _arg_parts:
                            _arg_parts.append(_s)
                    _cause_chain = ""
                    _ce = getattr(e, "__cause__", None) or getattr(e, "__context__", None)
                    if _ce is not None and _ce is not e:
                        _cause_chain = f" | caused_by={type(_ce).__name__}: {_ce}"
                    _verbose = (
                        f"{type(e).__name__}: str={_err_str!r} "
                        f"args=[{' || '.join(_arg_parts) if _arg_parts else '<empty>'}]"
                        f"{_cause_chain}"
                    )
                    attempt_errors.append(f"attempt{attempt + 1}={_verbose}")
                    # Per-attempt visibility — without this the operator only
                    # sees the final summary and can't tell whether attempts
                    # 1+2 had the same root cause as attempt 3.
                    logger.warning(
                        f"[KrakenREST] Dead-man attempt {attempt + 1}/3 failed: {_verbose}"
                    )
                    if "Invalid nonce" in _err_str:
                        if attempt == 0:
                            new_ratchet = self._bump_nonce_past_error(60)
                            logger.warning(
                                f"[KrakenREST] Invalid nonce on dead-man (attempt 1); "
                                f"bumped ratchet by 60s to {new_ratchet}, retrying"
                            )
                            continue
                        if attempt == 1:
                            try:
                                self._exchange.load_time_difference()
                                new_ratchet = self._bump_nonce_past_error(120)
                                logger.warning(
                                    f"[KrakenREST] Invalid nonce persisted (attempt 2); "
                                    f"reloaded timeDifference="
                                    f"{self._exchange.options.get('timeDifference', 0)}ms "
                                    f"and bumped ratchet to {new_ratchet}, retrying"
                                )
                                continue
                            except Exception as _td_err:
                                logger.error(
                                    f"[KrakenREST] load_time_difference failed: "
                                    f"{type(_td_err).__name__}: {_td_err!r}"
                                )
                    # Propagate the failure cause so DeadManSwitch.refresh()
                    # can surface it in the WARNING/CRITICAL log instead of
                    # generic "Refresh FAILED". Carry every attempt's error
                    # so the operator sees the full sequence in one line.
                    self._last_dead_man_error = " ; ".join(attempt_errors)
                    logger.error(
                        f"[KrakenREST] Dead-man switch failed after {attempt + 1} attempt(s): "
                        f"{self._last_dead_man_error}\n"
                        f"Final traceback:\n{traceback.format_exc()}"
                    )
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
