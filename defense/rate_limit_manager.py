"""
================================================================================
RATE LIMIT MANAGER - Kraken Counter-Decay Token Bucket
================================================================================
Version: 3.2.0
Purpose: Prevent API rate limit violations with Kraken-specific token bucket

Features:
- Counter-decay token bucket algorithm
- Per-endpoint cost tracking
- Shared memory for multi-process coordination
- Automatic backoff on 429 responses
================================================================================
"""

import time
import ctypes
import logging
import threading
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from multiprocessing import shared_memory
import struct

logger = logging.getLogger("HMATS.RateLimitManager")


class KrakenEndpoint(Enum):
    """Kraken API endpoints with cost."""
    # Public endpoints (1 counter)
    TIME = ("Time", 1, "public")
    SYSTEM_STATUS = ("SystemStatus", 1, "public")
    ASSETS = ("Assets", 1, "public")
    ASSET_PAIRS = ("AssetPairs", 1, "public")
    TICKER = ("Ticker", 1, "public")
    OHLC = ("OHLC", 1, "public")
    DEPTH = ("Depth", 1, "public")
    TRADES = ("Trades", 1, "public")
    SPREAD = ("Spread", 1, "public")
    
    # Private endpoints (varying cost)
    BALANCE = ("Balance", 1, "private")
    TRADE_BALANCE = ("TradeBalance", 1, "private")
    OPEN_ORDERS = ("OpenOrders", 1, "private")
    CLOSED_ORDERS = ("ClosedOrders", 1, "private")
    QUERY_ORDERS = ("QueryOrders", 1, "private")
    TRADES_HISTORY = ("TradesHistory", 1, "private")
    QUERY_TRADES = ("QueryTrades", 2, "private")  # Higher cost
    OPEN_POSITIONS = ("OpenPositions", 1, "private")
    LEDGERS = ("Ledgers", 2, "private")  # Higher cost
    QUERY_LEDGERS = ("QueryLedgers", 2, "private")  # Higher cost
    TRADE_VOLUME = ("TradeVolume", 1, "private")
    
    # Trading endpoints (highest cost)
    ADD_ORDER = ("AddOrder", 0, "trading")  # 0 = no cost initially, +1 on cancel
    CANCEL_ORDER = ("CancelOrder", 0, "trading")
    CANCEL_ALL = ("CancelAll", 0, "trading")
    CANCEL_ALL_AFTER = ("CancelAllOrdersAfter", 0, "trading")
    
    def __init__(self, name: str, cost: int, category: str):
        self._name = name
        self.cost = cost
        self.category = category


@dataclass
class RateLimitConfig:
    """Kraken rate limit configuration."""
    # Counter limits by tier (Intermediate tier shown)
    public_max_counter: int = 15
    private_max_counter: int = 20
    
    # Decay rates (counter reduction per second)
    public_decay_rate: float = 0.33    # ~3 seconds to decay 1 counter
    private_decay_rate: float = 0.5    # 2 seconds to decay 1 counter
    
    # Trading limits (separate from counter)
    orders_per_second: float = 1.0     # Max orders/sec
    max_open_orders: int = 500         # Max simultaneous open orders
    
    # Safety margins
    counter_safety_margin: float = 0.8  # Use only 80% of limit
    backoff_base_ms: int = 100          # Base backoff on limit hit
    backoff_max_ms: int = 5000          # Max backoff


@dataclass
class TokenBucketState(ctypes.Structure):
    """Shared memory structure for token bucket."""
    _fields_ = [
        ('tokens', ctypes.c_double),
        ('last_update', ctypes.c_double),
        ('capacity', ctypes.c_double),
        ('decay_rate', ctypes.c_double),
        ('lock_flag', ctypes.c_int),
    ]


class KrakenCounterBucket:
    """
    Kraken-specific counter-decay token bucket.
    
    Kraken uses a counter that:
    - Increases by endpoint cost on each call
    - Decays over time (counter -= decay_rate × elapsed)
    - Rejects requests if counter > max
    """
    
    def __init__(
        self,
        name: str,
        max_counter: int,
        decay_rate: float,
        safety_margin: float = 0.8
    ):
        self.name = name
        self.max_counter = max_counter
        self.decay_rate = decay_rate
        self.safety_margin = safety_margin
        self.effective_max = max_counter * safety_margin
        
        self.current_counter = 0.0
        self.last_update = time.time()
        
        self._lock = threading.Lock()
        
        # Statistics
        self.total_requests = 0
        self.total_cost = 0
        self.blocked_requests = 0
    
    def _decay(self):
        """Apply decay to counter."""
        now = time.time()
        elapsed = now - self.last_update
        
        if elapsed > 0:
            decay_amount = elapsed * self.decay_rate
            self.current_counter = max(0, self.current_counter - decay_amount)
            self.last_update = now
    
    def can_request(self, cost: int = 1) -> Tuple[bool, float]:
        """
        Check if request can be made.
        
        Returns:
            (can_proceed, wait_time_seconds)
        """
        with self._lock:
            self._decay()
            
            if self.current_counter + cost <= self.effective_max:
                return True, 0.0
            
            # Calculate wait time
            excess = (self.current_counter + cost) - self.effective_max
            wait_time = excess / self.decay_rate
            
            return False, wait_time
    
    def consume(self, cost: int = 1) -> bool:
        """
        Consume counter for request.
        
        Returns:
            True if consumed, False if would exceed limit
        """
        with self._lock:
            self._decay()
            
            if self.current_counter + cost > self.effective_max:
                self.blocked_requests += 1
                return False
            
            self.current_counter += cost
            self.total_requests += 1
            self.total_cost += cost
            
            return True
    
    def force_consume(self, cost: int = 1):
        """Force consume even if over limit (for tracking)."""
        with self._lock:
            self._decay()
            self.current_counter += cost
            self.total_requests += 1
            self.total_cost += cost
    
    def get_status(self) -> Dict:
        """Get bucket status."""
        with self._lock:
            self._decay()
            return {
                'name': self.name,
                'current_counter': self.current_counter,
                'max_counter': self.max_counter,
                'effective_max': self.effective_max,
                'decay_rate': self.decay_rate,
                'utilization': self.current_counter / self.effective_max,
                'total_requests': self.total_requests,
                'blocked_requests': self.blocked_requests
            }


class KrakenRateLimitManager:
    """
    Complete rate limit manager for Kraken API.
    
    Manages separate buckets for public, private, and trading endpoints.
    """
    
    def __init__(self, config: RateLimitConfig = None):
        self.config = config or RateLimitConfig()
        
        # Create buckets for each category
        self.buckets = {
            'public': KrakenCounterBucket(
                name='public',
                max_counter=self.config.public_max_counter,
                decay_rate=self.config.public_decay_rate,
                safety_margin=self.config.counter_safety_margin
            ),
            'private': KrakenCounterBucket(
                name='private',
                max_counter=self.config.private_max_counter,
                decay_rate=self.config.private_decay_rate,
                safety_margin=self.config.counter_safety_margin
            ),
            'trading': KrakenCounterBucket(
                name='trading',
                max_counter=100,  # Separate limit for trading
                decay_rate=1.0,   # Faster decay
                safety_margin=0.9
            )
        }
        
        # Backoff state
        self.current_backoff_ms = 0
        self.last_429_time = 0.0
        self.consecutive_429s = 0
        
        # Order tracking
        self.pending_orders = set()
        
        self._lock = threading.Lock()
        
        logger.info("KrakenRateLimitManager initialized")
    
    def acquire(self, endpoint: KrakenEndpoint, block: bool = True, timeout: float = 30.0) -> bool:
        """
        Acquire rate limit token for endpoint.
        
        Args:
            endpoint: Kraken endpoint to call
            block: Whether to wait if limit reached
            timeout: Max wait time in seconds
            
        Returns:
            True if acquired, False if timeout or not blocking
        """
        bucket = self.buckets[endpoint.category]
        cost = endpoint.cost
        
        start_time = time.time()
        
        while True:
            can_proceed, wait_time = bucket.can_request(cost)
            
            if can_proceed:
                bucket.consume(cost)
                return True
            
            if not block:
                return False
            
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                logger.warning(f"Rate limit timeout for {endpoint.name}")
                return False
            
            # Wait with backoff
            actual_wait = min(wait_time, timeout - elapsed, self.current_backoff_ms / 1000)
            if actual_wait > 0:
                time.sleep(actual_wait)
    
    def report_429(self, endpoint: KrakenEndpoint):
        """Report a 429 response from Kraken."""
        with self._lock:
            self.last_429_time = time.time()
            self.consecutive_429s += 1
            
            # Exponential backoff
            self.current_backoff_ms = min(
                self.config.backoff_base_ms * (2 ** self.consecutive_429s),
                self.config.backoff_max_ms
            )
            
            # Force bump the counter
            bucket = self.buckets[endpoint.category]
            bucket.force_consume(5)  # Penalty for 429
            
            logger.warning(f"429 received for {endpoint.name}, backoff: {self.current_backoff_ms}ms")
    
    def report_success(self, endpoint: KrakenEndpoint):
        """Report successful request (reset backoff)."""
        with self._lock:
            if time.time() - self.last_429_time > 10:  # 10 seconds of success
                self.consecutive_429s = 0
                self.current_backoff_ms = 0
    
    def register_order(self, order_id: str) -> bool:
        """Register pending order."""
        with self._lock:
            if len(self.pending_orders) >= self.config.max_open_orders:
                logger.error(f"Max open orders reached: {self.config.max_open_orders}")
                return False
            
            self.pending_orders.add(order_id)
            return True
    
    def unregister_order(self, order_id: str):
        """Unregister order (filled, cancelled, etc.)."""
        with self._lock:
            self.pending_orders.discard(order_id)
    
    def get_status(self) -> Dict:
        """Get complete rate limit status."""
        status = {
            'buckets': {name: bucket.get_status() for name, bucket in self.buckets.items()},
            'backoff_ms': self.current_backoff_ms,
            'consecutive_429s': self.consecutive_429s,
            'pending_orders': len(self.pending_orders),
            'max_open_orders': self.config.max_open_orders
        }
        return status
    
    def can_proceed(self) -> bool:
        """Check if any trading request can proceed without blocking.

        Used by P0SafetyIntegrator for pre-trade rate limit check.
        Returns True if the trading bucket has capacity.
        """
        bucket = self.buckets.get('trading', self.buckets.get('private'))
        if bucket is None:
            return True
        ok, _ = bucket.can_request(cost=1)
        return ok

    def estimate_wait_time(self, endpoint: KrakenEndpoint) -> float:
        """Estimate wait time for endpoint."""
        bucket = self.buckets[endpoint.category]
        can_proceed, wait_time = bucket.can_request(endpoint.cost)
        return wait_time if not can_proceed else 0.0


class SharedRateLimitManager:
    """
    Shared memory rate limit manager for multi-process coordination.
    
    Uses shared memory to coordinate rate limits across all processes.
    """
    
    SHM_NAME_PREFIX = "hmats_ratelimit_"
    
    def __init__(self, is_primary: bool = False):
        self.is_primary = is_primary
        self._shm_buckets: Dict[str, shared_memory.SharedMemory] = {}
        self._bucket_states: Dict[str, TokenBucketState] = {}
        
        self._lock = threading.Lock()
    
    def initialize(self, categories: List[str] = None):
        """Initialize shared memory segments."""
        categories = categories or ['public', 'private', 'trading']
        
        for category in categories:
            shm_name = f"{self.SHM_NAME_PREFIX}{category}"
            size = ctypes.sizeof(TokenBucketState)
            
            try:
                if self.is_primary:
                    # Try to clean up existing
                    try:
                        existing = shared_memory.SharedMemory(name=shm_name)
                        existing.close()
                        existing.unlink()
                    except FileNotFoundError:
                        pass
                    
                    # Create new
                    shm = shared_memory.SharedMemory(name=shm_name, create=True, size=size)
                    state = TokenBucketState.from_buffer(shm.buf)
                    
                    # Initialize state
                    state.tokens = 0.0
                    state.last_update = time.time()
                    state.lock_flag = 0
                    
                else:
                    # Attach to existing
                    shm = shared_memory.SharedMemory(name=shm_name)
                    state = TokenBucketState.from_buffer(shm.buf)
                
                self._shm_buckets[category] = shm
                self._bucket_states[category] = state
                
                logger.info(f"{'Created' if self.is_primary else 'Attached to'} "
                           f"shared rate limit: {category}")
                
            except Exception as e:
                logger.error(f"Failed to initialize shared rate limit {category}: {e}")
    
    def cleanup(self):
        """Clean up shared memory."""
        for category, shm in self._shm_buckets.items():
            try:
                shm.close()
                if self.is_primary:
                    shm.unlink()
            except Exception as e:
                logger.error(f"Failed to cleanup {category}: {e}")
