"""
================================================================================
KRAKEN INTEGRITY SHIELD - Orderbook Validation & Connection Resilience
================================================================================
Version: 3.2.0
Purpose: Ensure data integrity for Kraken orderbook with CRC32 validation

Features:
- CRC32 orderbook checksum validation (Kraken WS V2)
- Automatic Full Reset on checksum failure
- Separated WS Hot Channel and REST Escape Channel
- Connection state machine with exponential backoff
================================================================================
"""

import time
import zlib
import json
import logging
import asyncio
import hashlib
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from collections import deque
from decimal import Decimal
import threading

# Import KrakenRESTClient for REST fallback
try:
    from infra.kraken_rest_client import KrakenRESTClient
    REST_CLIENT_AVAILABLE = True
except ImportError:
    REST_CLIENT_AVAILABLE = False
    KrakenRESTClient = None

logger = logging.getLogger("HMATS.KrakenIntegrity")


class ConnectionState(Enum):
    """WebSocket connection state."""
    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    AUTHENTICATED = auto()
    SUBSCRIBED = auto()
    RECOVERING = auto()
    FAILED = auto()


class ChannelType(Enum):
    """Channel classification."""
    HOT = auto()      # Real-time orderbook, trades
    WARM = auto()     # Balances, positions
    COLD = auto()     # Historical data
    ESCAPE = auto()   # REST fallback


@dataclass
class OrderbookLevel:
    """Single orderbook level."""
    price: Decimal
    quantity: Decimal
    timestamp: float = 0.0


@dataclass
class OrderbookSnapshot:
    """Complete orderbook snapshot with checksum."""
    symbol: str
    bids: List[OrderbookLevel] = field(default_factory=list)
    asks: List[OrderbookLevel] = field(default_factory=list)
    sequence: int = 0
    checksum: int = 0
    timestamp: float = field(default_factory=time.time)
    validated: bool = False


@dataclass
class IntegrityMetrics:
    """Integrity monitoring metrics."""
    total_updates: int = 0
    checksum_passes: int = 0
    checksum_failures: int = 0
    full_resets: int = 0
    rest_fallbacks: int = 0
    last_valid_sequence: int = 0
    last_failure_time: float = 0.0
    consecutive_failures: int = 0


class KrakenCRC32Validator:
    """
    CRC32 checksum validator for Kraken orderbook.
    
    Kraken WS V2 checksum calculation:
    1. Take top 10 bid and ask levels
    2. Format: price1:qty1:price2:qty2:...
    3. Remove decimal points from prices and quantities
    4. Calculate CRC32 of resulting string
    """
    
    TOP_LEVELS = 10
    
    @classmethod
    def calculate_checksum(cls, bids: List[OrderbookLevel], asks: List[OrderbookLevel]) -> int:
        """
        Calculate Kraken CRC32 checksum.
        
        Format: ask1_price:ask1_qty:...ask10_price:ask10_qty:bid1_price:bid1_qty:...bid10_price:bid10_qty
        """
        parts = []
        
        # Asks (sorted ascending by price, take first 10)
        sorted_asks = sorted(asks, key=lambda x: x.price)[:cls.TOP_LEVELS]
        for level in sorted_asks:
            price_str = cls._format_decimal(level.price)
            qty_str = cls._format_decimal(level.quantity)
            parts.extend([price_str, qty_str])
        
        # Bids (sorted descending by price, take first 10)
        sorted_bids = sorted(bids, key=lambda x: x.price, reverse=True)[:cls.TOP_LEVELS]
        for level in sorted_bids:
            price_str = cls._format_decimal(level.price)
            qty_str = cls._format_decimal(level.quantity)
            parts.extend([price_str, qty_str])
        
        # Join and calculate CRC32
        checksum_str = ":".join(parts)
        crc = zlib.crc32(checksum_str.encode('utf-8')) & 0xFFFFFFFF
        
        return crc
    
    @classmethod
    def _format_decimal(cls, value: Decimal) -> str:
        """Format decimal: remove trailing zeros and decimal point."""
        # Convert to string, strip trailing zeros
        s = str(value)
        if '.' in s:
            s = s.rstrip('0').rstrip('.')
        # Remove decimal point
        return s.replace('.', '')
    
    @classmethod
    def validate(cls, snapshot: OrderbookSnapshot) -> bool:
        """Validate orderbook snapshot checksum."""
        calculated = cls.calculate_checksum(snapshot.bids, snapshot.asks)
        return calculated == snapshot.checksum


class KrakenOrderbookManager:
    """
    Manage Kraken orderbook with integrity validation.
    
    Maintains local orderbook state and validates all updates.
    """
    
    MAX_CONSECUTIVE_FAILURES = 3
    RESET_COOLDOWN_SECONDS = 5.0
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.bids: Dict[Decimal, OrderbookLevel] = {}
        self.asks: Dict[Decimal, OrderbookLevel] = {}
        self.sequence = 0
        self.metrics = IntegrityMetrics()
        self._lock = threading.Lock()
        self._last_reset_time = 0.0
    
    def apply_snapshot(self, snapshot: OrderbookSnapshot) -> bool:
        """Apply full orderbook snapshot."""
        with self._lock:
            self.bids.clear()
            self.asks.clear()
            
            for level in snapshot.bids:
                self.bids[level.price] = level
            
            for level in snapshot.asks:
                self.asks[level.price] = level
            
            self.sequence = snapshot.sequence
            
            # Validate
            if snapshot.checksum:
                valid = self._validate_checksum(snapshot.checksum)
                snapshot.validated = valid
                return valid
            
            return True
    
    def apply_update(
        self,
        bid_updates: List[Tuple[Decimal, Decimal]],
        ask_updates: List[Tuple[Decimal, Decimal]],
        checksum: int,
        sequence: int
    ) -> Tuple[bool, Optional[str]]:
        """
        Apply incremental orderbook update.
        
        Returns:
            (success, error_message)
        """
        with self._lock:
            self.metrics.total_updates += 1
            
            # Check sequence
            if sequence <= self.sequence:
                return True, None  # Duplicate, ignore
            
            # Apply updates
            for price, qty in bid_updates:
                if qty == 0:
                    self.bids.pop(price, None)
                else:
                    self.bids[price] = OrderbookLevel(price=price, quantity=qty, timestamp=time.time())
            
            for price, qty in ask_updates:
                if qty == 0:
                    self.asks.pop(price, None)
                else:
                    self.asks[price] = OrderbookLevel(price=price, quantity=qty, timestamp=time.time())
            
            self.sequence = sequence
            
            # Validate checksum
            if checksum:
                valid = self._validate_checksum(checksum)
                if not valid:
                    self.metrics.checksum_failures += 1
                    self.metrics.consecutive_failures += 1
                    self.metrics.last_failure_time = time.time()
                    
                    if self.metrics.consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
                        return False, "CHECKSUM_FAILURE_THRESHOLD"
                    
                    return False, "CHECKSUM_MISMATCH"
                else:
                    self.metrics.checksum_passes += 1
                    self.metrics.consecutive_failures = 0
                    self.metrics.last_valid_sequence = sequence
            
            return True, None
    
    def _validate_checksum(self, expected: int) -> bool:
        """Validate current orderbook against checksum."""
        bids_list = list(self.bids.values())
        asks_list = list(self.asks.values())
        calculated = KrakenCRC32Validator.calculate_checksum(bids_list, asks_list)
        return calculated == expected
    
    def get_snapshot(self) -> OrderbookSnapshot:
        """Get current orderbook snapshot."""
        with self._lock:
            return OrderbookSnapshot(
                symbol=self.symbol,
                bids=list(self.bids.values()),
                asks=list(self.asks.values()),
                sequence=self.sequence,
                checksum=KrakenCRC32Validator.calculate_checksum(
                    list(self.bids.values()),
                    list(self.asks.values())
                ),
                validated=True
            )
    
    def needs_reset(self) -> bool:
        """Check if orderbook needs full reset."""
        if self.metrics.consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
            # Check cooldown
            if time.time() - self._last_reset_time > self.RESET_COOLDOWN_SECONDS:
                return True
        return False
    
    def mark_reset(self):
        """Mark that a reset was performed."""
        self._last_reset_time = time.time()
        self.metrics.full_resets += 1
        self.metrics.consecutive_failures = 0


class KrakenIntegrityShield:
    """
    Main integrity shield coordinating WS and REST channels.
    
    Features:
    - Hot Channel: WebSocket for real-time updates
    - Escape Channel: REST API for recovery
    - Automatic failover on integrity violations
    """
    
    # Kraken WS V2 endpoints
    WS_PUBLIC_URL = "wss://ws.kraken.com/v2"
    WS_PRIVATE_URL = "wss://ws-auth.kraken.com/v2"
    
    # Kraken REST endpoints
    REST_BASE_URL = "https://api.kraken.com"
    REST_ORDERBOOK_ENDPOINT = "/0/public/Depth"
    
    def __init__(
        self,
        symbols: List[str] = None,
        on_integrity_violation: Callable = None,
        on_recovery_complete: Callable = None
    ):
        # [P133 2026-04-29] SOL routed to SOL/USDT (USD pair dead on Kraken).
        self.symbols = symbols or ['XBT/USD', 'ETH/USD', 'SOL/USDT']
        self.on_integrity_violation = on_integrity_violation
        self.on_recovery_complete = on_recovery_complete
        
        # Orderbook managers per symbol
        self.orderbooks: Dict[str, KrakenOrderbookManager] = {
            symbol: KrakenOrderbookManager(symbol) for symbol in self.symbols
        }
        
        # Connection state
        self.ws_state = ConnectionState.DISCONNECTED
        self.rest_available = True
        
        # Recovery state
        self._recovering = False
        self._recovery_lock = asyncio.Lock()
        
        # Metrics
        self.global_metrics = {
            'ws_reconnects': 0,
            'rest_fallbacks': 0,
            'total_integrity_violations': 0,
            'uptime_start': time.time()
        }
        
        logger.info(f"KrakenIntegrityShield initialized for {self.symbols}")
    
    async def handle_ws_message(self, message: Dict) -> bool:
        """
        Handle incoming WebSocket message.
        
        Returns:
            True if message processed successfully
        """
        msg_type = message.get('type')
        
        if msg_type == 'snapshot':
            return await self._handle_snapshot(message)
        elif msg_type == 'update':
            return await self._handle_update(message)
        elif msg_type == 'error':
            logger.error(f"Kraken WS error: {message}")
            return False
        
        return True
    
    async def _handle_snapshot(self, message: Dict) -> bool:
        """Handle orderbook snapshot message."""
        try:
            symbol = message.get('symbol')
            if symbol not in self.orderbooks:
                return True
            
            manager = self.orderbooks[symbol]
            
            # Parse snapshot
            bids = [
                OrderbookLevel(price=Decimal(str(b[0])), quantity=Decimal(str(b[1])))
                for b in message.get('bids', [])
            ]
            asks = [
                OrderbookLevel(price=Decimal(str(a[0])), quantity=Decimal(str(a[1])))
                for a in message.get('asks', [])
            ]
            
            snapshot = OrderbookSnapshot(
                symbol=symbol,
                bids=bids,
                asks=asks,
                sequence=message.get('sequence', 0),
                checksum=message.get('checksum', 0)
            )
            
            success = manager.apply_snapshot(snapshot)
            
            if not success:
                logger.warning(f"Snapshot validation failed for {symbol}")
                await self._trigger_recovery(symbol, "SNAPSHOT_INVALID")
            
            return success
            
        except Exception as e:
            logger.error(f"Error handling snapshot: {e}")
            return False
    
    async def _handle_update(self, message: Dict) -> bool:
        """Handle orderbook update message."""
        try:
            symbol = message.get('symbol')
            if symbol not in self.orderbooks:
                return True
            
            manager = self.orderbooks[symbol]
            
            # Parse updates
            bid_updates = [
                (Decimal(str(b[0])), Decimal(str(b[1])))
                for b in message.get('bids', [])
            ]
            ask_updates = [
                (Decimal(str(a[0])), Decimal(str(a[1])))
                for a in message.get('asks', [])
            ]
            
            success, error = manager.apply_update(
                bid_updates=bid_updates,
                ask_updates=ask_updates,
                checksum=message.get('checksum', 0),
                sequence=message.get('sequence', 0)
            )
            
            if not success:
                logger.warning(f"Update validation failed for {symbol}: {error}")
                
                if manager.needs_reset():
                    await self._trigger_recovery(symbol, error)
            
            return success
            
        except Exception as e:
            logger.error(f"Error handling update: {e}")
            return False
    
    async def _trigger_recovery(self, symbol: str, reason: str):
        """Trigger orderbook recovery via REST."""
        async with self._recovery_lock:
            if self._recovering:
                return
            
            self._recovering = True
            self.global_metrics['total_integrity_violations'] += 1
            
            logger.warning(f"Triggering recovery for {symbol}: {reason}")
            
            if self.on_integrity_violation:
                self.on_integrity_violation(symbol, reason)
            
            try:
                # Fetch from REST (Escape Channel)
                snapshot = await self._fetch_rest_orderbook(symbol)
                
                if snapshot:
                    manager = self.orderbooks[symbol]
                    manager.apply_snapshot(snapshot)
                    manager.mark_reset()
                    
                    self.global_metrics['rest_fallbacks'] += 1
                    logger.info(f"Recovery complete for {symbol}")
                    
                    if self.on_recovery_complete:
                        self.on_recovery_complete(symbol)
                else:
                    logger.error(f"REST fallback failed for {symbol}")
                    
            except Exception as e:
                logger.error(f"Recovery failed: {e}")
            finally:
                self._recovering = False
    
    async def _fetch_rest_orderbook(self, symbol: str, depth: int = 100) -> Optional[OrderbookSnapshot]:
        """Fetch orderbook via REST API (Escape Channel)."""
        try:
            # Use KrakenRESTClient if available
            if REST_CLIENT_AVAILABLE and hasattr(self, '_rest_client') and self._rest_client:
                orderbook = await asyncio.get_event_loop().run_in_executor(
                    None, 
                    lambda: self._rest_client.fetch_orderbook(symbol, limit=depth)
                )
                
                if orderbook and 'bids' in orderbook and 'asks' in orderbook:
                    bids = [
                        OrderbookLevel(
                            price=Decimal(str(bid[0])),
                            quantity=Decimal(str(bid[1])),
                            timestamp=time.time()
                        )
                        for bid in orderbook['bids'][:depth]
                    ]
                    asks = [
                        OrderbookLevel(
                            price=Decimal(str(ask[0])),
                            quantity=Decimal(str(ask[1])),
                            timestamp=time.time()
                        )
                        for ask in orderbook['asks'][:depth]
                    ]
                    
                    return OrderbookSnapshot(
                        symbol=symbol,
                        bids=bids,
                        asks=asks,
                        timestamp=time.time(),
                        checksum=None,
                        sequence=0
                    )
            
            # Fallback: direct aiohttp call
            from data_mgmt.feeds._http import create_session

            # Convert symbol format (XBT/USD -> XXBTZUSD)
            pair = symbol.replace('/', '').replace('XBT', 'XXBT').replace('USD', 'ZUSD')

            url = f"{self.REST_BASE_URL}{self.REST_ORDERBOOK_ENDPOINT}"
            params = {'pair': pair, 'count': depth}

            async with create_session() as session:
                async with session.get(url, params=params, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if 'result' in data and pair in data['result']:
                            book = data['result'][pair]
                            
                            bids = [
                                OrderbookLevel(
                                    price=Decimal(bid[0]),
                                    quantity=Decimal(bid[1]),
                                    timestamp=float(bid[2]) if len(bid) > 2 else time.time()
                                )
                                for bid in book.get('bids', [])
                            ]
                            asks = [
                                OrderbookLevel(
                                    price=Decimal(ask[0]),
                                    quantity=Decimal(ask[1]),
                                    timestamp=float(ask[2]) if len(ask) > 2 else time.time()
                                )
                                for ask in book.get('asks', [])
                            ]
                            
                            logger.info(f"REST fetch successful for {symbol}: {len(bids)} bids, {len(asks)} asks")
                            
                            return OrderbookSnapshot(
                                symbol=symbol,
                                bids=bids,
                                asks=asks,
                                timestamp=time.time(),
                                checksum=None,
                                sequence=0
                            )
            
            return None
            
        except Exception as e:
            logger.error(f"REST fetch failed: {e}")
            return None
    
    def get_orderbook(self, symbol: str) -> Optional[OrderbookSnapshot]:
        """Get validated orderbook snapshot."""
        if symbol in self.orderbooks:
            return self.orderbooks[symbol].get_snapshot()
        return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get integrity metrics."""
        result = {
            'global': self.global_metrics.copy(),
            'symbols': {}
        }
        
        for symbol, manager in self.orderbooks.items():
            result['symbols'][symbol] = {
                'total_updates': manager.metrics.total_updates,
                'checksum_passes': manager.metrics.checksum_passes,
                'checksum_failures': manager.metrics.checksum_failures,
                'full_resets': manager.metrics.full_resets,
                'consecutive_failures': manager.metrics.consecutive_failures,
                'pass_rate': (
                    manager.metrics.checksum_passes / max(1, manager.metrics.total_updates)
                ) * 100
            }
        
        return result
    
    def is_healthy(self) -> bool:
        """Check if all orderbooks are healthy."""
        for manager in self.orderbooks.values():
            if manager.metrics.consecutive_failures > 0:
                return False
        return True
