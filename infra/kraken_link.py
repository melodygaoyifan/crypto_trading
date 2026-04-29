"""
KrakenDefensiveLink - Hardened Kraken V2 WebSocket Wrapper
==========================================================

Production-grade defensive layer for Kraken WebSocket V2 with:

1. CRC32 Checksum Validator
   - Real-time orderbook integrity verification
   - Kraken-spec decimal stripping
   - Auto-recovery on mismatch

2. Heartbeat Watchdog
   - Per-asset last_message tracking
   - 5-second silence detection for high-vol assets
   - Active reconnect trigger

3. Self-Healing Architecture
   - Full Reset: Clear Book -> Close WS -> Reconnect -> REST Snapshot
   - Hybrid Gateway: WS (Hot) + REST (Escape Channel)
   - Cancel-on-Disconnect enforcement

4. Error Classification
   - Retriable: Network timeouts, rate limits
   - Fatal: Auth failure, invalid API keys
   - Conditional: Market-specific errors

Hardware Target: MSI Titan 18 HX (24-core, 128GB RAM)
Author: Senior Lead Quant Architect & Defensive Systems Engineer
Version: 3.2.0
"""

import os
import sys
import json
import time
import zlib
import asyncio
import logging
import hashlib
import threading
import traceback
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Tuple, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN
from collections import deque, OrderedDict
from abc import ABC, abstractmethod
import multiprocessing as mp
from multiprocessing import shared_memory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger('KrakenDefensiveLink')


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS & CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Kraken WebSocket V2 Endpoints
KRAKEN_WS_PUBLIC = "wss://ws.kraken.com/v2"
KRAKEN_WS_PRIVATE = "wss://ws-auth.kraken.com/v2"
KRAKEN_REST_BASE = "https://api.kraken.com"

# Watchdog Configuration
WATCHDOG_INTERVAL_MS = 1000  # Check every 1 second
HIGH_VOL_SILENCE_THRESHOLD_MS = 5000  # 5 seconds for SOL
NORMAL_SILENCE_THRESHOLD_MS = 15000  # 15 seconds for BTC/ETH
HIGH_VOLATILITY_ASSETS = {'SOL', 'SOLUSD', 'SOLUSDT', 'SOL/USD', 'SOL/USDT', 'XSOLUSD'}

# Checksum Configuration
MAX_CHECKSUM_FAILURES = 3  # Trigger reset after 3 consecutive failures
CHECKSUM_DEPTH = 10  # Top 10 bids and asks for checksum

# Reconnect Configuration
MAX_RECONNECT_ATTEMPTS = 5
RECONNECT_BACKOFF_BASE = 1.0  # seconds
RECONNECT_BACKOFF_MAX = 30.0  # seconds

# Heartbeat
HEARTBEAT_INTERVAL_S = 30
PING_TIMEOUT_S = 10


# ═══════════════════════════════════════════════════════════════════════════════
# ERROR CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

class ErrorCategory(Enum):
    """Classification of errors for handling strategy."""
    RETRIABLE = auto()      # Network timeouts, rate limits - retry with backoff
    FATAL = auto()          # Auth failure, invalid API keys - halt system
    CONDITIONAL = auto()    # Market-specific, may require manual intervention
    CHECKSUM = auto()       # Data integrity failure - trigger reset
    TIMEOUT = auto()        # Connection timeout - reconnect
    UNKNOWN = auto()        # Unclassified - log and continue


@dataclass
class ClassifiedError:
    """Error with classification and metadata."""
    category: ErrorCategory
    code: str
    message: str
    timestamp: float
    retriable: bool
    retry_after_ms: int = 0
    requires_reset: bool = False
    context: Dict[str, Any] = field(default_factory=dict)


class ErrorClassifier:
    """Classifies Kraken errors into handling categories."""
    
    # Kraken error codes mapping
    RETRIABLE_CODES = {
        'EGeneral:Temporary lockout',
        'EService:Unavailable',
        'EService:Busy',
        'EGeneral:Timeout',
        'EAPI:Rate limit exceeded',
    }
    
    FATAL_CODES = {
        'EAPI:Invalid key',
        'EAPI:Invalid signature',
        'EGeneral:Permission denied',
        'EAccount:Disabled',
    }
    
    CONDITIONAL_CODES = {
        'EOrder:Insufficient funds',
        'EOrder:Invalid price',
        'EOrder:Market in post_only mode',
        'EOrder:Invalid order',
        'EGeneral:Invalid arguments',
    }
    
    @classmethod
    def classify(cls, error: str, code: str = None) -> ClassifiedError:
        """Classify an error and return handling strategy."""
        error_str = str(error)
        code = code or error_str
        
        # Check fatal first (highest priority)
        for fatal in cls.FATAL_CODES:
            if fatal in error_str:
                return ClassifiedError(
                    category=ErrorCategory.FATAL,
                    code=code,
                    message=error_str,
                    timestamp=time.time(),
                    retriable=False,
                    requires_reset=False
                )
        
        # Check retriable
        for retriable in cls.RETRIABLE_CODES:
            if retriable in error_str:
                return ClassifiedError(
                    category=ErrorCategory.RETRIABLE,
                    code=code,
                    message=error_str,
                    timestamp=time.time(),
                    retriable=True,
                    retry_after_ms=1000  # Default 1s backoff
                )
        
        # Check conditional
        for conditional in cls.CONDITIONAL_CODES:
            if conditional in error_str:
                return ClassifiedError(
                    category=ErrorCategory.CONDITIONAL,
                    code=code,
                    message=error_str,
                    timestamp=time.time(),
                    retriable='post_only' in error_str.lower(),
                    retry_after_ms=100 if 'post_only' in error_str.lower() else 0
                )
        
        # Unknown error
        return ClassifiedError(
            category=ErrorCategory.UNKNOWN,
            code=code,
            message=error_str,
            timestamp=time.time(),
            retriable=False
        )


# ═══════════════════════════════════════════════════════════════════════════════
# ORDERBOOK DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class OrderbookLevel:
    """Single orderbook price level."""
    price: Decimal
    qty: Decimal
    timestamp: float = 0.0


@dataclass
class LocalOrderbook:
    """Local orderbook state with checksum support."""
    symbol: str
    bids: OrderedDict  # price -> OrderbookLevel (descending)
    asks: OrderedDict  # price -> OrderbookLevel (ascending)
    sequence: int = 0
    last_update: float = 0.0
    checksum: int = 0
    checksum_valid: bool = True
    consecutive_failures: int = 0
    
    def __post_init__(self):
        if self.bids is None:
            self.bids = OrderedDict()
        if self.asks is None:
            self.asks = OrderedDict()


# ═══════════════════════════════════════════════════════════════════════════════
# CRC32 CHECKSUM VALIDATOR
# ═══════════════════════════════════════════════════════════════════════════════

class OrderbookChecksumValidator:
    """
    Validates Kraken orderbook integrity using CRC32 checksum.
    
    Kraken Checksum Spec:
    1. Take top 10 bid and ask levels
    2. For each level: strip trailing zeros and decimal point from price and qty
    3. Concatenate: ask1_price + ask1_qty + ask2_price + ... + bid1_price + bid1_qty + ...
    4. Calculate CRC32 checksum
    """
    
    @staticmethod
    def strip_decimals(value: Decimal) -> str:
        """
        Strip trailing zeros and decimal point per Kraken spec.
        
        Examples:
        - "0.00100000" -> "001"
        - "1234.50000" -> "12345"
        - "1000.00000" -> "1000"
        """
        # Convert to string and remove decimal point
        str_val = str(value)
        
        # Handle scientific notation
        if 'E' in str_val or 'e' in str_val:
            str_val = f"{value:f}"
        
        # Remove decimal point
        str_val = str_val.replace('.', '')
        
        # Strip trailing zeros
        str_val = str_val.rstrip('0')
        
        # If all zeros were stripped, keep at least one digit
        if not str_val or str_val == '-':
            str_val = '0'
        
        return str_val
    
    @classmethod
    def calculate_checksum(cls, book: LocalOrderbook, depth: int = CHECKSUM_DEPTH) -> int:
        """
        Calculate CRC32 checksum for orderbook.
        
        Args:
            book: Local orderbook state
            depth: Number of levels to include (default 10)
            
        Returns:
            CRC32 checksum as unsigned 32-bit integer
        """
        parts = []
        
        # Get top N asks (ascending by price)
        asks = list(book.asks.items())[:depth]
        for price, level in asks:
            parts.append(cls.strip_decimals(price))
            parts.append(cls.strip_decimals(level.qty))
        
        # Get top N bids (descending by price)
        bids = list(book.bids.items())[:depth]
        for price, level in bids:
            parts.append(cls.strip_decimals(price))
            parts.append(cls.strip_decimals(level.qty))
        
        # Concatenate and calculate CRC32
        checksum_string = ''.join(parts)
        checksum = zlib.crc32(checksum_string.encode('utf-8')) & 0xFFFFFFFF
        
        return checksum
    
    @classmethod
    def validate(cls, book: LocalOrderbook, expected_checksum: int) -> Tuple[bool, int]:
        """
        Validate orderbook against expected checksum.
        
        Returns:
            (is_valid, calculated_checksum)
        """
        calculated = cls.calculate_checksum(book)
        is_valid = calculated == expected_checksum
        return is_valid, calculated


# ═══════════════════════════════════════════════════════════════════════════════
# WATCHDOG MONITOR
# ═══════════════════════════════════════════════════════════════════════════════

class HeartbeatWatchdog:
    """
    Monitors WebSocket activity and triggers reconnect on silence.
    
    Features:
    - Per-asset last_message tracking
    - Different thresholds for high-vol vs normal assets
    - Active reconnect trigger
    """
    
    def __init__(
        self,
        high_vol_threshold_ms: int = HIGH_VOL_SILENCE_THRESHOLD_MS,
        normal_threshold_ms: int = NORMAL_SILENCE_THRESHOLD_MS,
        high_vol_assets: Set[str] = None
    ):
        self.high_vol_threshold = high_vol_threshold_ms
        self.normal_threshold = normal_threshold_ms
        self.high_vol_assets = high_vol_assets or HIGH_VOLATILITY_ASSETS
        
        # Last message timestamps per asset
        self._last_message: Dict[str, float] = {}
        
        # Global last activity
        self._last_activity: float = time.time()
        
        # Callback for reconnect
        self._reconnect_callback: Optional[Callable] = None
        
        # Lock for thread safety
        self._lock = threading.Lock()
        
        # Running state
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        
        logger.info(f"HeartbeatWatchdog initialized (high_vol={high_vol_threshold_ms}ms, normal={normal_threshold_ms}ms)")
    
    def record_message(self, asset: str = None):
        """Record a message received for an asset."""
        now = time.time()
        with self._lock:
            self._last_activity = now
            if asset:
                self._last_message[asset] = now
    
    def set_reconnect_callback(self, callback: Callable):
        """Set callback to trigger on silence detection."""
        self._reconnect_callback = callback
    
    def start(self):
        """Start the watchdog monitor."""
        if self._running:
            return
        
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("HeartbeatWatchdog started")
    
    def stop(self):
        """Stop the watchdog monitor."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2)
        logger.info("HeartbeatWatchdog stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                self._check_activity()
                time.sleep(WATCHDOG_INTERVAL_MS / 1000)
            except Exception as e:
                logger.error(f"Watchdog error: {e}")
    
    def _check_activity(self):
        """Check for silence on monitored assets."""
        now = time.time()
        
        with self._lock:
            # Check high-vol assets
            for asset in self.high_vol_assets:
                if asset in self._last_message:
                    silence_ms = (now - self._last_message[asset]) * 1000
                    if silence_ms > self.high_vol_threshold:
                        logger.warning(f"Silence detected on {asset}: {silence_ms:.0f}ms")
                        self._trigger_reconnect(asset, silence_ms)
                        return
            
            # Check global activity
            global_silence_ms = (now - self._last_activity) * 1000
            if global_silence_ms > self.normal_threshold:
                logger.warning(f"Global silence detected: {global_silence_ms:.0f}ms")
                self._trigger_reconnect('GLOBAL', global_silence_ms)
    
    def _trigger_reconnect(self, source: str, silence_ms: float):
        """Trigger reconnect callback."""
        logger.warning(f"Triggering reconnect (source={source}, silence={silence_ms:.0f}ms)")
        if self._reconnect_callback:
            try:
                self._reconnect_callback(source)
            except Exception as e:
                logger.error(f"Reconnect callback failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# CONNECTION STATE
# ═══════════════════════════════════════════════════════════════════════════════

class ConnectionState(Enum):
    """WebSocket connection state."""
    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    AUTHENTICATED = auto()
    SUBSCRIBING = auto()
    READY = auto()
    RECONNECTING = auto()
    RESETTING = auto()
    ERROR = auto()


@dataclass
class ConnectionMetrics:
    """Metrics for connection health monitoring."""
    connected_at: float = 0.0
    last_message_at: float = 0.0
    messages_received: int = 0
    reconnect_count: int = 0
    checksum_failures: int = 0
    last_error: Optional[ClassifiedError] = None
    uptime_seconds: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# KRAKEN DEFENSIVE LINK
# ═══════════════════════════════════════════════════════════════════════════════

class KrakenDefensiveLink:
    """
    Hardened Kraken WebSocket V2 wrapper with defensive features.
    
    Features:
    - CRC32 checksum validation for orderbook integrity
    - Heartbeat watchdog with auto-reconnect
    - Self-healing on data corruption
    - Cancel-on-disconnect enforcement
    - Error classification and handling
    - Hybrid gateway (WS + REST escape channel)
    """
    
    def __init__(
        self,
        api_key: str = None,
        api_secret: str = None,
        symbols: List[str] = None,
        enable_checksum: bool = True,
        enable_watchdog: bool = True,
        cancel_on_disconnect: bool = True
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        # [P133 2026-04-29] SOL on USDT (USD pair dead on Kraken).
        self.symbols = symbols or ['BTC/USD', 'ETH/USD', 'SOL/USDT']
        self.enable_checksum = enable_checksum
        self.enable_watchdog = enable_watchdog
        self.cancel_on_disconnect = cancel_on_disconnect
        
        # Connection state
        self._state = ConnectionState.DISCONNECTED
        self._metrics = ConnectionMetrics()
        
        # WebSocket
        self._ws = None
        # Avoid touching event-loop state in __init__ (can run on threads with no loop).
        self._ws_lock = None
        
        # Orderbooks
        self._orderbooks: Dict[str, LocalOrderbook] = {}
        
        # Checksum validator
        self._checksum_validator = OrderbookChecksumValidator()
        
        # Watchdog
        self._watchdog = HeartbeatWatchdog() if enable_watchdog else None
        if self._watchdog:
            self._watchdog.set_reconnect_callback(self._on_watchdog_trigger)
        
        # Callbacks
        self._on_orderbook_update: Optional[Callable] = None
        self._on_trade: Optional[Callable] = None
        self._on_error: Optional[Callable] = None
        self._on_reset: Optional[Callable] = None
        self._on_disconnect: Optional[Callable] = None  # v6.3.1: Disconnect event callback
        self._on_recovered: Optional[Callable] = None   # v6.3.1: Recovery event callback
        
        # v6.3.1: Disconnect state tracking
        self._disconnect_triggered = False
        
        # Reconnect state
        self._reconnect_attempts = 0
        self._reconnect_task: Optional[asyncio.Task] = None
        
        # REST client for escape channel
        self._rest_session = None
        
        logger.info(f"KrakenDefensiveLink initialized (symbols={symbols}, checksum={enable_checksum})")
    
    # ───────────────────────────────────────────────────────────────────────────
    # PUBLIC INTERFACE
    # ───────────────────────────────────────────────────────────────────────────
    
    async def connect(self):
        """Establish WebSocket connection."""
        if self._state in [ConnectionState.CONNECTED, ConnectionState.READY]:
            logger.warning("Already connected")
            return
        
        self._state = ConnectionState.CONNECTING
        
        try:
            # Import websockets here to allow graceful degradation
            try:
                import websockets
                self._ws = await websockets.connect(
                    KRAKEN_WS_PUBLIC,
                    ping_interval=HEARTBEAT_INTERVAL_S,
                    ping_timeout=PING_TIMEOUT_S,
                    close_timeout=5
                )
                self._state = ConnectionState.CONNECTED
                self._metrics.connected_at = time.time()
                self._metrics.reconnect_count = 0
                
                # Start watchdog
                if self._watchdog:
                    self._watchdog.start()
                
                logger.info("WebSocket connected")
                
            except ImportError:
                logger.warning("websockets not installed - using mock connection")
                self._state = ConnectionState.CONNECTED
                self._metrics.connected_at = time.time()
                
        except Exception as e:
            self._state = ConnectionState.ERROR
            error = ErrorClassifier.classify(str(e))
            self._metrics.last_error = error
            logger.error(f"Connection failed: {e}")
            raise
    
    async def subscribe_orderbook(self, symbol: str, depth: int = 25):
        """Subscribe to orderbook updates with checksum validation."""
        if self._state != ConnectionState.CONNECTED and self._state != ConnectionState.READY:
            raise RuntimeError(f"Cannot subscribe in state {self._state}")
        
        # Initialize local orderbook
        self._orderbooks[symbol] = LocalOrderbook(
            symbol=symbol,
            bids=OrderedDict(),
            asks=OrderedDict()
        )
        
        # Send subscription
        sub_msg = {
            "method": "subscribe",
            "params": {
                "channel": "book",
                "symbol": [symbol],
                "depth": depth,
                "snapshot": True
            }
        }
        
        if self._ws:
            await self._ws.send(json.dumps(sub_msg))
            logger.info(f"Subscribed to orderbook: {symbol}")
        else:
            logger.info(f"Mock subscription to orderbook: {symbol}")
    
    async def disconnect(self):
        """Gracefully disconnect."""
        self._state = ConnectionState.DISCONNECTED
        
        if self._watchdog:
            self._watchdog.stop()
        
        if self._ws:
            await self._ws.close()
            self._ws = None
        
        # v6.3.1: Trigger disconnect event for cancel-on-disconnect
        self._trigger_disconnect_event("graceful_disconnect")
        
        logger.info("Disconnected")
    
    def get_orderbook(self, symbol: str) -> Optional[LocalOrderbook]:
        """Get current local orderbook state."""
        return self._orderbooks.get(symbol)
    
    def get_metrics(self) -> ConnectionMetrics:
        """Get connection metrics."""
        if self._metrics.connected_at > 0:
            self._metrics.uptime_seconds = time.time() - self._metrics.connected_at
        return self._metrics
    
    # ───────────────────────────────────────────────────────────────────────────
    # CALLBACK REGISTRATION
    # ───────────────────────────────────────────────────────────────────────────
    
    def on_orderbook_update(self, callback: Callable):
        """Register orderbook update callback."""
        self._on_orderbook_update = callback
    
    def on_trade(self, callback: Callable):
        """Register trade callback."""
        self._on_trade = callback
    
    def on_error(self, callback: Callable):
        """Register error callback."""
        self._on_error = callback
    
    def on_reset(self, callback: Callable):
        """Register reset callback."""
        self._on_reset = callback
    
    def on_disconnect(self, callback: Callable):
        """
        Register disconnect callback (v6.3.1).
        
        Called when connection is lost due to:
        - Explicit disconnect() call
        - WebSocket close/error
        - Watchdog silence timeout
        - Reconnect max attempts exceeded
        - Full reset triggered
        
        Callback receives: reason (str)
        """
        self._on_disconnect = callback
    
    def on_recovered(self, callback: Callable):
        """
        Register recovery callback (v6.3.1).
        
        Called when connection is restored after a disconnect event.
        Used to clear ExecutionManager disconnect latch.
        
        Callback receives: no arguments
        """
        self._on_recovered = callback
    
    # ───────────────────────────────────────────────────────────────────────────
    # v6.3.1: DISCONNECT EVENT HANDLING
    # These methods ensure all disconnect scenarios trigger the callback
    # ───────────────────────────────────────────────────────────────────────────
    
    def _trigger_disconnect_event(self, reason: str):
        """
        Trigger disconnect event to external handlers (v6.3.1).
        
        This is the SINGLE point that ensures cancel-on-disconnect is triggered
        for ALL disconnect scenarios, not just explicit errors.
        
        Args:
            reason: Human-readable reason for disconnect
        """
        # Only trigger if cancel_on_disconnect is enabled
        if not self.cancel_on_disconnect:
            logger.debug(f"Disconnect event ignored (cancel_on_disconnect=False): {reason}")
            return
        
        # Avoid duplicate triggers
        if self._disconnect_triggered:
            logger.debug(f"Disconnect event already triggered, skipping: {reason}")
            return
        
        self._disconnect_triggered = True
        logger.warning(f"[DISCONNECT EVENT] Triggered: {reason}")
        
        # Call registered callback
        if self._on_disconnect:
            try:
                self._on_disconnect(reason)
                logger.info(f"[DISCONNECT EVENT] Callback executed successfully")
            except Exception as e:
                logger.error(f"[DISCONNECT EVENT] Callback failed: {e}")
        
        # Also trigger on_error for backward compatibility
        if self._on_error:
            try:
                # P86: was `KrakenLinkError(...)` — class undefined. NameError
                # was caught by surrounding except, so the error callback
                # silently never fired since extraction. The on_error contract
                # passes ClassifiedError objects (see _classifier.classify usage
                # at line 153 + 892); use the existing ClassifiedError dataclass
                # consistently. Same shape as P85's ShadowLedgerWriter
                # frozen_allocations cascade — reader-side trusted a writer-
                # side contract that didn't exist.
                error = ClassifiedError(
                    category=ErrorCategory.FATAL,
                    code="DISCONNECT_EVENT",
                    message=f"disconnect:{reason}",
                    timestamp=time.time(),
                    retriable=False,
                    requires_reset=True,
                )
                self._on_error(error)
            except Exception as e:
                logger.error(f"[DISCONNECT EVENT] Error callback failed: {e}")
    
    def _trigger_recovered_event(self):
        """
        Trigger recovery event when connection is restored (v6.3.1).
        
        This clears the disconnect latch and allows trading to resume.
        """
        if not self._disconnect_triggered:
            # No disconnect was triggered, nothing to recover
            return
        
        self._disconnect_triggered = False
        logger.info("[RECOVERED] Connection restored, clearing disconnect state")
        
        if self._on_recovered:
            try:
                self._on_recovered()
                logger.info("[RECOVERED] Recovery callback executed")
            except Exception as e:
                logger.error(f"[RECOVERED] Recovery callback failed: {e}")
    
    # ───────────────────────────────────────────────────────────────────────────
    # MESSAGE PROCESSING
    # ───────────────────────────────────────────────────────────────────────────
    
    async def _process_message(self, message: str):
        """Process incoming WebSocket message."""
        try:
            data = json.loads(message)
            self._metrics.messages_received += 1
            self._metrics.last_message_at = time.time()
            
            if self._watchdog:
                self._watchdog.record_message()
            
            # Handle different message types
            if 'channel' in data:
                channel = data.get('channel')
                
                if channel == 'book':
                    await self._handle_orderbook_message(data)
                elif channel == 'trade':
                    await self._handle_trade_message(data)
                elif channel == 'heartbeat':
                    pass  # Heartbeat acknowledged
                    
            elif 'error' in data:
                await self._handle_error_message(data)
                
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON: {e}")
        except Exception as e:
            logger.error(f"Message processing error: {e}")
    
    async def _handle_orderbook_message(self, data: Dict):
        """Handle orderbook update with checksum validation."""
        try:
            msg_type = data.get('type')
            symbol = data.get('data', [{}])[0].get('symbol', '')
            
            if symbol not in self._orderbooks:
                return
            
            book = self._orderbooks[symbol]
            
            if msg_type == 'snapshot':
                await self._apply_snapshot(book, data)
            elif msg_type == 'update':
                await self._apply_update(book, data)
            
            # Validate checksum if enabled
            if self.enable_checksum and 'checksum' in data.get('data', [{}])[0]:
                expected_checksum = data['data'][0]['checksum']
                is_valid, calculated = self._checksum_validator.validate(book, expected_checksum)
                
                if not is_valid:
                    book.consecutive_failures += 1
                    book.checksum_valid = False
                    self._metrics.checksum_failures += 1
                    
                    logger.warning(
                        f"Checksum mismatch for {symbol}: "
                        f"expected={expected_checksum}, calculated={calculated}, "
                        f"failures={book.consecutive_failures}"
                    )
                    
                    if book.consecutive_failures >= MAX_CHECKSUM_FAILURES:
                        await self._trigger_full_reset(symbol, "checksum_failure")
                else:
                    book.consecutive_failures = 0
                    book.checksum_valid = True
                    book.checksum = expected_checksum
            
            # Notify callback
            if self._on_orderbook_update:
                self._on_orderbook_update(symbol, book)
                
        except Exception as e:
            logger.error(f"Orderbook handling error: {e}")
    
    async def _apply_snapshot(self, book: LocalOrderbook, data: Dict):
        """Apply orderbook snapshot."""
        snapshot = data.get('data', [{}])[0]
        
        # Clear existing
        book.bids.clear()
        book.asks.clear()
        
        # Apply bids (descending order)
        for level in snapshot.get('bids', []):
            price = Decimal(str(level['price']))
            qty = Decimal(str(level['qty']))
            book.bids[price] = OrderbookLevel(price=price, qty=qty, timestamp=time.time())
        
        # Apply asks (ascending order)
        for level in snapshot.get('asks', []):
            price = Decimal(str(level['price']))
            qty = Decimal(str(level['qty']))
            book.asks[price] = OrderbookLevel(price=price, qty=qty, timestamp=time.time())
        
        # Sort bids descending, asks ascending
        book.bids = OrderedDict(sorted(book.bids.items(), reverse=True))
        book.asks = OrderedDict(sorted(book.asks.items()))
        
        book.last_update = time.time()
        logger.debug(f"Snapshot applied for {book.symbol}: {len(book.bids)} bids, {len(book.asks)} asks")
    
    async def _apply_update(self, book: LocalOrderbook, data: Dict):
        """Apply incremental orderbook update."""
        update = data.get('data', [{}])[0]
        
        # Apply bid updates
        for level in update.get('bids', []):
            price = Decimal(str(level['price']))
            qty = Decimal(str(level['qty']))
            
            if qty == 0:
                book.bids.pop(price, None)
            else:
                book.bids[price] = OrderbookLevel(price=price, qty=qty, timestamp=time.time())
        
        # Apply ask updates
        for level in update.get('asks', []):
            price = Decimal(str(level['price']))
            qty = Decimal(str(level['qty']))
            
            if qty == 0:
                book.asks.pop(price, None)
            else:
                book.asks[price] = OrderbookLevel(price=price, qty=qty, timestamp=time.time())
        
        # Re-sort
        book.bids = OrderedDict(sorted(book.bids.items(), reverse=True))
        book.asks = OrderedDict(sorted(book.asks.items()))
        
        book.last_update = time.time()
    
    async def _handle_trade_message(self, data: Dict):
        """Handle trade message."""
        if self._on_trade:
            self._on_trade(data)
    
    async def _handle_error_message(self, data: Dict):
        """Handle error message."""
        error_msg = data.get('error', 'Unknown error')
        error = ErrorClassifier.classify(error_msg)
        self._metrics.last_error = error
        
        logger.error(f"WebSocket error: {error.category.name} - {error_msg}")
        
        if self._on_error:
            self._on_error(error)
        
        if error.category == ErrorCategory.FATAL:
            await self.disconnect()
    
    # ───────────────────────────────────────────────────────────────────────────
    # SELF-HEALING
    # ───────────────────────────────────────────────────────────────────────────
    
    async def _trigger_full_reset(self, symbol: str, reason: str):
        """
        Trigger full orderbook reset.
        
        Steps:
        1. Clear local orderbook
        2. Close WebSocket
        3. Reconnect
        4. Fetch REST snapshot
        5. Re-subscribe
        """
        logger.warning(f"Triggering full reset for {symbol}: {reason}")
        self._state = ConnectionState.RESETTING
        
        # v6.3.1: Trigger disconnect event BEFORE reset starts
        # This ensures cancel-on-disconnect even during planned resets
        self._trigger_disconnect_event(f"full_reset:{symbol}:{reason}")
        
        # Notify callback
        if self._on_reset:
            self._on_reset(symbol, reason)
        
        try:
            # 1. Clear local orderbook
            if symbol in self._orderbooks:
                book = self._orderbooks[symbol]
                book.bids.clear()
                book.asks.clear()
                book.consecutive_failures = 0
            
            # 2. Close WebSocket
            if self._ws:
                await self._ws.close()
                self._ws = None
            
            # 3. Wait before reconnect
            await asyncio.sleep(1)
            
            # 4. Reconnect
            await self._reconnect()
            
            # 5. Fetch REST snapshot (if available)
            await self._fetch_rest_snapshot(symbol)
            
            # 6. Re-subscribe
            await self.subscribe_orderbook(symbol)
            
            self._state = ConnectionState.READY
            logger.info(f"Full reset completed for {symbol}")
            
        except Exception as e:
            logger.error(f"Full reset failed: {e}")
            self._state = ConnectionState.ERROR
    
    async def _reconnect(self):
        """Reconnect with exponential backoff."""
        self._state = ConnectionState.RECONNECTING
        
        for attempt in range(MAX_RECONNECT_ATTEMPTS):
            try:
                backoff = min(
                    RECONNECT_BACKOFF_BASE * (2 ** attempt),
                    RECONNECT_BACKOFF_MAX
                )
                
                logger.info(f"Reconnect attempt {attempt + 1}/{MAX_RECONNECT_ATTEMPTS} in {backoff}s")
                await asyncio.sleep(backoff)
                
                await self.connect()
                self._reconnect_attempts = attempt + 1
                self._metrics.reconnect_count += 1
                
                # v6.3.1: Trigger recovery event on successful reconnect
                self._trigger_recovered_event()
                
                return
                
            except Exception as e:
                logger.error(f"Reconnect attempt {attempt + 1} failed: {e}")
        
        # v6.3.1: Trigger disconnect event when max attempts exceeded
        self._trigger_disconnect_event("max_reconnect_attempts_exceeded")
        
        logger.error("Max reconnect attempts exceeded")
        self._state = ConnectionState.ERROR
        raise RuntimeError("Failed to reconnect after max attempts")
    
    async def _fetch_rest_snapshot(self, symbol: str):
        """Fetch orderbook snapshot via REST API."""
        try:
            # Use KrakenRESTClient for actual API call
            from infra.kraken_rest_client import get_kraken_rest_client
            
            client = get_kraken_rest_client()
            if client.is_initialized():
                orderbook = client.fetch_orderbook(symbol)
                if orderbook:
                    logger.info(f"Fetched REST snapshot for {symbol}: mid={orderbook.mid_price():.2f}")
                    # P86: was `self._process_orderbook_snapshot(symbol, orderbook.to_dict())`
                    # — method never defined on this class. AttributeError on
                    # every full-reset path. Agent audit confirmed "data not
                    # used post-fetch anyway" — the fetch + mid log was the
                    # actual purpose. Removed the dangling call rather than
                    # define a no-op (P85 discipline rule 2: don't leave
                    # broken contracts in the codebase).
                else:
                    logger.warning(f"Empty orderbook for {symbol}")
            else:
                logger.warning("KrakenRESTClient not initialized, skipping snapshot")
            
        except ImportError:
            logger.warning("KrakenRESTClient not available")
        except Exception as e:
            logger.error(f"REST snapshot fetch failed: {e}")
    
    def _on_watchdog_trigger(self, source: str):
        """Handle watchdog reconnect trigger."""
        logger.warning(f"Watchdog triggered reconnect from {source}")
        
        # v6.3.1: Trigger disconnect event BEFORE attempting reconnect
        # This ensures orders are cancelled immediately on heartbeat timeout
        self._trigger_disconnect_event(f"watchdog_silence:{source}")

        # Schedule reconnect in the event loop
        # [P37 2026-04-24] Was discarded create_task() — reconnect coroutine
        # could be GC'd before completing, leaving the link dead silently.
        # Now tracked as instance attr; if a previous reconnect is still
        # running, skip (don't spawn a duplicate).
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                _prev = getattr(self, "_reconnect_task", None)
                if _prev is not None and not _prev.done():
                    logger.debug("[KRAKEN_LINK] reconnect already in progress, skipping duplicate")
                else:
                    self._reconnect_task = asyncio.create_task(
                        self._reconnect(), name="kraken_link_reconnect"
                    )
        except RuntimeError:
            logger.error("No event loop available for reconnect")
    
    # ───────────────────────────────────────────────────────────────────────────
    # REST ESCAPE CHANNEL
    # ───────────────────────────────────────────────────────────────────────────
    
    async def emergency_cancel_all(self) -> bool:
        """
        Emergency cancel all orders via REST escape channel.
        
        Used when WS is unreliable but we need to exit positions.
        """
        logger.warning("Executing emergency cancel all via REST")
        
        try:
            # Use KrakenRESTClient for actual API call
            from infra.kraken_rest_client import get_kraken_rest_client
            
            client = get_kraken_rest_client()
            if client.is_initialized():
                success = client.cancel_all_orders()
                if success:
                    logger.info("Emergency cancel all: SUCCESS")
                else:
                    logger.warning("Emergency cancel all: PARTIAL or FAILED")
                return success
            else:
                logger.error("KrakenRESTClient not initialized for emergency cancel")
                return False
            
        except ImportError:
            logger.error("KrakenRESTClient not available for emergency cancel")
            return False
        except Exception as e:
            logger.error(f"Emergency cancel failed: {e}")
            return False


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED MEMORY STATE (FOR MULTI-PROCESS)
# ═══════════════════════════════════════════════════════════════════════════════

class SharedConnectionState:
    """
    Shared connection state for multi-process architecture.
    
    Uses multiprocessing.shared_memory for cross-process visibility.
    """
    
    BUFFER_SIZE = 1024  # bytes
    
    def __init__(self, name: str = "kraken_defensive_state", create: bool = True):
        self.name = name
        self._shm: Optional[shared_memory.SharedMemory] = None
        
        try:
            if create:
                self._shm = shared_memory.SharedMemory(
                    name=name,
                    create=True,
                    size=self.BUFFER_SIZE
                )
                self._initialize()
            else:
                self._shm = shared_memory.SharedMemory(name=name)
                
            logger.info(f"SharedConnectionState initialized: {name}")
            
        except FileExistsError:
            # Attach to existing
            self._shm = shared_memory.SharedMemory(name=name)
    
    def _initialize(self):
        """Initialize shared memory with default state."""
        import struct
        
        # Layout: state(1) + connected_at(8) + last_message(8) + msg_count(8) + reconnects(4)
        initial = struct.pack('<BddqI', 0, 0.0, 0.0, 0, 0)
        self._shm.buf[:len(initial)] = initial
    
    def update_state(self, state: ConnectionState):
        """Update connection state."""
        if self._shm:
            self._shm.buf[0] = state.value
    
    def get_state(self) -> ConnectionState:
        """Get current connection state."""
        if self._shm:
            return ConnectionState(self._shm.buf[0])
        return ConnectionState.DISCONNECTED
    
    def unlink(self):
        """Remove shared memory."""
        if self._shm:
            try:
                self._shm.close()
                self._shm.unlink()
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Main class
    'KrakenDefensiveLink',
    
    # Validators
    'OrderbookChecksumValidator',
    
    # Watchdog
    'HeartbeatWatchdog',
    
    # Error handling
    'ErrorClassifier',
    'ErrorCategory',
    'ClassifiedError',
    
    # Data structures
    'LocalOrderbook',
    'OrderbookLevel',
    'ConnectionState',
    'ConnectionMetrics',
    
    # Shared state
    'SharedConnectionState',
]


# ═══════════════════════════════════════════════════════════════════════════════
# TESTING
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("KrakenDefensiveLink Test Suite")
    print("=" * 70)
    
    # Test 1: Checksum Validator
    print("\n[1] CRC32 Checksum Validator")
    print("-" * 50)
    
    # Create mock orderbook
    book = LocalOrderbook(
        symbol="BTC/USD",
        bids=OrderedDict(),
        asks=OrderedDict()
    )
    
    # Add levels (simulating Kraken data)
    book.asks[Decimal("97000.00")] = OrderbookLevel(Decimal("97000.00"), Decimal("1.50000000"))
    book.asks[Decimal("97001.00")] = OrderbookLevel(Decimal("97001.00"), Decimal("0.75000000"))
    book.bids[Decimal("96999.00")] = OrderbookLevel(Decimal("96999.00"), Decimal("2.00000000"))
    book.bids[Decimal("96998.00")] = OrderbookLevel(Decimal("96998.00"), Decimal("1.25000000"))
    
    checksum = OrderbookChecksumValidator.calculate_checksum(book, depth=2)
    print(f"Calculated checksum: {checksum}")
    
    # Test strip_decimals
    print(f"Strip '0.00100000' -> '{OrderbookChecksumValidator.strip_decimals(Decimal('0.00100000'))}'")
    print(f"Strip '1234.50000' -> '{OrderbookChecksumValidator.strip_decimals(Decimal('1234.50000'))}'")
    
    # Test 2: Error Classification
    print("\n[2] Error Classification")
    print("-" * 50)
    
    errors = [
        "EAPI:Rate limit exceeded",
        "EAPI:Invalid key",
        "EOrder:Market in post_only mode",
        "Unknown error occurred"
    ]
    
    for err in errors:
        classified = ErrorClassifier.classify(err)
        print(f"'{err}' -> {classified.category.name} (retriable={classified.retriable})")
    
    # Test 3: Watchdog
    print("\n[3] Heartbeat Watchdog")
    print("-" * 50)
    
    watchdog = HeartbeatWatchdog(high_vol_threshold_ms=100, normal_threshold_ms=500)
    
    reconnect_triggered = []
    def mock_reconnect(source):
        reconnect_triggered.append(source)
    
    watchdog.set_reconnect_callback(mock_reconnect)
    watchdog.record_message('BTC/USD')
    watchdog.record_message('SOL/USD')
    
    print(f"Recorded messages for BTC/USD and SOL/USD")
    print(f"High-vol assets: {watchdog.high_vol_assets}")
    
    # Test 4: Connection State
    print("\n[4] Connection State Machine")
    print("-" * 50)
    
    for state in ConnectionState:
        print(f"  {state.name} = {state.value}")
    
    # Test 5: Defensive Link (mock)
    print("\n[5] KrakenDefensiveLink Initialization")
    print("-" * 50)
    
    link = KrakenDefensiveLink(
        symbols=['BTC/USD', 'ETH/USD', 'SOL/USD'],
        enable_checksum=True,
        enable_watchdog=True,
        cancel_on_disconnect=True
    )
    
    print(f"Symbols: {link.symbols}")
    print(f"State: {link._state.name}")
    print(f"Checksum enabled: {link.enable_checksum}")
    print(f"Watchdog enabled: {link.enable_watchdog}")
    
    # Test 6: Shared State
    print("\n[6] Shared Connection State")
    print("-" * 50)
    
    try:
        shared = SharedConnectionState(name="test_kraken_state", create=True)
        shared.update_state(ConnectionState.CONNECTED)
        state = shared.get_state()
        print(f"Shared state: {state.name}")
        shared.unlink()
        print("Shared memory cleaned up")
    except Exception as e:
        print(f"Shared memory test skipped: {e}")
    
    print("\n" + "=" * 70)
    print("✓ All KrakenDefensiveLink tests completed!")
    print("=" * 70)


# =============================================================================
# SINGLETON ACCESSOR
# =============================================================================

_kraken_link_instance: Optional[KrakenDefensiveLink] = None

def get_kraken_link(
    symbols: List[str] = None,
    **kwargs
) -> KrakenDefensiveLink:
    """Get or create singleton KrakenDefensiveLink instance."""
    global _kraken_link_instance
    if _kraken_link_instance is None:
        _kraken_link_instance = KrakenDefensiveLink(
            symbols=symbols or ['BTC/USD', 'ETH/USD', 'SOL/USDT'],  # [P133]
            **kwargs
        )
    return _kraken_link_instance


def reset_kraken_link():
    """Reset singleton (for testing).

    P86: was `_kraken_link_instance.close()` — method undefined on
    KrakenDefensiveLink. The class has `disconnect()` (async) at line 600
    but no sync `close()`. AttributeError on every reset call (test cleanup
    path). Defensive getattr+best-effort cancel pattern: try the async
    disconnect via run_until_complete if no loop is running, else just
    drop the reference and let GC + WebSocket internals close on their
    own. Either way the singleton reset succeeds (the goal of this fn).
    """
    global _kraken_link_instance
    if _kraken_link_instance is not None:
        # P85 discipline rule 2: defensive getattr — don't trust the
        # specific cleanup method name, prefer disconnect → close → no-op.
        _cleanup = getattr(_kraken_link_instance, 'disconnect', None) \
            or getattr(_kraken_link_instance, 'close', None)
        if _cleanup is not None:
            try:
                import asyncio
                if asyncio.iscoroutinefunction(_cleanup):
                    try:
                        # If a loop is already running (test harness), just
                        # schedule and move on — don't block.
                        asyncio.get_event_loop().create_task(_cleanup())
                    except RuntimeError:
                        # No running loop — synchronous cleanup is OK.
                        asyncio.run(_cleanup())
                else:
                    _cleanup()
            except Exception as e:
                logger.warning(
                    f"[KRAKEN_LINK] reset_kraken_link cleanup raised "
                    f"{type(e).__name__}: {e}; dropping reference anyway"
                )
    _kraken_link_instance = None


# Alias for backward compatibility
KrakenLink = KrakenDefensiveLink
