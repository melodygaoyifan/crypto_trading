"""
KrakenTokenBucket - Kraken-Specific Rate Limiter
================================================

Implementation of Kraken's "Counter Decay Model" for rate limiting.

Kraken Rate Limit Rules:
1. Each API call has a specific "cost" that adds to a counter
2. Counter decays at a fixed rate per second based on account tier
3. Requests fail if counter exceeds maximum

Tier Specifications:
- Starter: max=15, decay=0.33/s
- Intermediate: max=20, decay=0.5/s  
- Pro: max=20, decay=1.0/s

API Call Costs:
- Ledger/Trade History: 2
- Add/Cancel Order: 0 (separate order rate limit)
- Other calls: 1

Order Rate Limits (separate):
- Starter: 60/min
- Intermediate: 125/min
- Pro: 225/min

Hardware Target: MSI Titan 18 HX (24-core, 128GB RAM)
Author: Senior Lead Quant Architect & Defensive Systems Engineer
Version: 3.2.0
"""

import os
import time
import logging
import threading
import struct
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from collections import deque
import multiprocessing as mp
from multiprocessing import shared_memory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger('KrakenTokenBucket')


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS & CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

class KrakenTier(Enum):
    """Kraken account verification tiers."""
    STARTER = auto()
    INTERMEDIATE = auto()
    PRO = auto()


# Tier specifications
TIER_SPECS = {
    KrakenTier.STARTER: {
        'max_counter': 15,
        'decay_rate': 0.33,  # per second
        'order_limit': 60,   # per minute
        'order_window': 60,  # seconds
    },
    KrakenTier.INTERMEDIATE: {
        'max_counter': 20,
        'decay_rate': 0.5,
        'order_limit': 125,
        'order_window': 60,
    },
    KrakenTier.PRO: {
        'max_counter': 20,
        'decay_rate': 1.0,
        'order_limit': 225,
        'order_window': 60,
    },
}

# API call costs
API_COSTS = {
    # High cost calls
    'Ledgers': 2,
    'QueryLedgers': 2,
    'TradesHistory': 2,
    'QueryTrades': 2,
    'OpenPositions': 2,
    'Ledgers/History': 2,
    
    # Standard cost calls
    'Balance': 1,
    'TradeBalance': 1,
    'OpenOrders': 1,
    'ClosedOrders': 1,
    'QueryOrders': 1,
    'Ticker': 1,
    'OHLC': 1,
    'Depth': 1,
    'Trades': 1,
    'Spread': 1,
    'Time': 1,
    'Assets': 1,
    'AssetPairs': 1,
    
    # Zero cost (separate order limit)
    'AddOrder': 0,
    'CancelOrder': 0,
    'CancelAll': 0,
    'CancelAllOrdersAfter': 0,
    'EditOrder': 0,
}

# Default cost for unknown endpoints
DEFAULT_COST = 1

# Matching trigger cooldown
MATCHING_TRIGGER_COOLDOWN_S = 10.0


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RateLimitState:
    """Current rate limit state."""
    counter: float = 0.0
    last_decay: float = 0.0
    order_count: int = 0
    order_window_start: float = 0.0
    last_request: float = 0.0
    blocked_requests: int = 0
    matching_trigger_until: float = 0.0


@dataclass 
class RequestRecord:
    """Record of an API request."""
    endpoint: str
    cost: int
    timestamp: float
    blocked: bool = False
    wait_time: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# KRAKEN TOKEN BUCKET
# ═══════════════════════════════════════════════════════════════════════════════

class KrakenTokenBucket:
    """
    Kraken-specific rate limiter using Counter Decay Model.
    
    Features:
    - Accurate counter decay simulation
    - Separate order rate limiting
    - Cross-process state via shared memory
    - Matching trigger cooldown (post_only mode)
    - Request history for debugging
    """
    
    def __init__(
        self,
        tier: KrakenTier = KrakenTier.PRO,
        shared_name: str = None,
        enable_logging: bool = True
    ):
        self.tier = tier
        self.spec = TIER_SPECS[tier]
        self.enable_logging = enable_logging
        
        # State
        self._state = RateLimitState()
        self._state.last_decay = time.time()
        self._state.order_window_start = time.time()
        
        # Request history
        self._history: deque = deque(maxlen=1000)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Shared memory (optional)
        self._shared: Optional[SharedRateLimitState] = None
        if shared_name:
            self._shared = SharedRateLimitState(shared_name, create=True)
        
        logger.info(
            f"KrakenTokenBucket initialized: tier={tier.name}, "
            f"max={self.spec['max_counter']}, decay={self.spec['decay_rate']}/s, "
            f"orders={self.spec['order_limit']}/min"
        )
    
    # ───────────────────────────────────────────────────────────────────────────
    # CORE RATE LIMITING
    # ───────────────────────────────────────────────────────────────────────────
    
    def acquire(
        self,
        endpoint: str,
        block: bool = True,
        timeout: float = 30.0
    ) -> Tuple[bool, float]:
        """
        Acquire permission to make an API call.
        
        Args:
            endpoint: API endpoint name (e.g., 'Balance', 'AddOrder')
            block: If True, wait until rate limit allows; if False, return immediately
            timeout: Maximum time to wait if blocking
            
        Returns:
            (acquired, wait_time) - acquired is True if request can proceed
        """
        cost = self._get_cost(endpoint)
        is_order = endpoint in ['AddOrder', 'EditOrder']
        
        start_time = time.time()
        
        with self._lock:
            while True:
                # Apply decay
                self._apply_decay()
                
                # Check matching trigger cooldown
                if self._state.matching_trigger_until > time.time():
                    wait_time = self._state.matching_trigger_until - time.time()
                    if not block or (time.time() - start_time + wait_time) > timeout:
                        self._record_request(endpoint, cost, blocked=True, wait_time=wait_time)
                        return False, wait_time
                    
                    # Wait for cooldown
                    self._lock.release()
                    time.sleep(min(wait_time, 0.1))
                    self._lock.acquire()
                    continue
                
                # Check order rate limit
                if is_order:
                    can_order, order_wait = self._check_order_limit()
                    if not can_order:
                        if not block or (time.time() - start_time + order_wait) > timeout:
                            self._record_request(endpoint, cost, blocked=True, wait_time=order_wait)
                            return False, order_wait
                        
                        self._lock.release()
                        time.sleep(min(order_wait, 0.1))
                        self._lock.acquire()
                        continue
                
                # Check counter limit
                if cost > 0:
                    if self._state.counter + cost > self.spec['max_counter']:
                        # Calculate wait time
                        excess = (self._state.counter + cost) - self.spec['max_counter']
                        wait_time = excess / self.spec['decay_rate']
                        
                        if not block or (time.time() - start_time + wait_time) > timeout:
                            self._record_request(endpoint, cost, blocked=True, wait_time=wait_time)
                            self._state.blocked_requests += 1
                            return False, wait_time
                        
                        # Wait for decay
                        self._lock.release()
                        time.sleep(min(wait_time, 0.1))
                        self._lock.acquire()
                        continue
                    
                    # Add cost to counter
                    self._state.counter += cost
                
                # Track order
                if is_order:
                    self._state.order_count += 1
                
                self._state.last_request = time.time()
                
                # Update shared state
                if self._shared:
                    self._shared.update(self._state)
                
                wait_time = time.time() - start_time
                self._record_request(endpoint, cost, blocked=False, wait_time=wait_time)
                
                if self.enable_logging and wait_time > 0.1:
                    logger.debug(
                        f"Rate limit waited {wait_time:.2f}s for {endpoint} "
                        f"(counter={self._state.counter:.2f})"
                    )
                
                return True, wait_time
    
    def _apply_decay(self):
        """Apply counter decay based on elapsed time."""
        now = time.time()
        elapsed = now - self._state.last_decay
        
        if elapsed > 0:
            decay_amount = elapsed * self.spec['decay_rate']
            self._state.counter = max(0, self._state.counter - decay_amount)
            self._state.last_decay = now
        
        # Reset order window if needed
        window_elapsed = now - self._state.order_window_start
        if window_elapsed >= self.spec['order_window']:
            self._state.order_count = 0
            self._state.order_window_start = now
    
    def _check_order_limit(self) -> Tuple[bool, float]:
        """
        Check order rate limit.
        
        Returns:
            (can_order, wait_time)
        """
        if self._state.order_count < self.spec['order_limit']:
            return True, 0.0
        
        # Calculate wait time until window resets
        elapsed = time.time() - self._state.order_window_start
        wait_time = self.spec['order_window'] - elapsed
        
        return False, max(0, wait_time)
    
    def _get_cost(self, endpoint: str) -> int:
        """Get cost for an endpoint."""
        return API_COSTS.get(endpoint, DEFAULT_COST)
    
    # ───────────────────────────────────────────────────────────────────────────
    # POST-ONLY MODE HANDLING
    # ───────────────────────────────────────────────────────────────────────────
    
    def trigger_matching_cooldown(self, duration: float = MATCHING_TRIGGER_COOLDOWN_S):
        """
        Trigger matching engine cooldown.
        
        Called when receiving "Market in post_only mode" error.
        Prevents further order submissions until cooldown expires.
        """
        with self._lock:
            self._state.matching_trigger_until = time.time() + duration
            
            logger.warning(
                f"Matching trigger cooldown activated for {duration}s "
                f"(until {datetime.fromtimestamp(self._state.matching_trigger_until)})"
            )
    
    def is_in_cooldown(self) -> bool:
        """Check if currently in matching cooldown."""
        with self._lock:
            return self._state.matching_trigger_until > time.time()
    
    # ───────────────────────────────────────────────────────────────────────────
    # REPRICE LOGIC
    # ───────────────────────────────────────────────────────────────────────────
    
    def get_repriced_order(
        self,
        side: str,
        original_price: Decimal,
        tick_size: Decimal,
        ticks_away: int = 1
    ) -> Decimal:
        """
        Get repriced order price after post_only rejection.
        
        Adjusts price by tick_size to ensure maker execution:
        - BUY: decrease price (more aggressive)
        - SELL: increase price (more aggressive)
        
        Args:
            side: 'buy' or 'sell'
            original_price: Original rejected price
            tick_size: Minimum price increment
            ticks_away: Number of ticks to move
            
        Returns:
            New price
        """
        adjustment = tick_size * ticks_away
        
        if side.lower() == 'buy':
            # Move bid down to avoid crossing
            new_price = original_price - adjustment
        else:
            # Move ask up to avoid crossing
            new_price = original_price + adjustment
        
        logger.debug(
            f"Repriced {side} order: {original_price} -> {new_price} "
            f"(tick_size={tick_size}, ticks={ticks_away})"
        )
        
        return new_price
    
    # ───────────────────────────────────────────────────────────────────────────
    # STATUS & MONITORING
    # ───────────────────────────────────────────────────────────────────────────
    
    def get_status(self) -> Dict[str, Any]:
        """Get current rate limit status."""
        with self._lock:
            self._apply_decay()
            
            return {
                'tier': self.tier.name,
                'counter': round(self._state.counter, 3),
                'max_counter': self.spec['max_counter'],
                'headroom': round(self.spec['max_counter'] - self._state.counter, 3),
                'decay_rate': self.spec['decay_rate'],
                'order_count': self._state.order_count,
                'order_limit': self.spec['order_limit'],
                'order_headroom': self.spec['order_limit'] - self._state.order_count,
                'blocked_requests': self._state.blocked_requests,
                'in_cooldown': self._state.matching_trigger_until > time.time(),
                'last_request': self._state.last_request,
            }
    
    def get_wait_time(self, endpoint: str) -> float:
        """
        Get estimated wait time for an endpoint.
        
        Returns 0 if request would succeed immediately.
        """
        cost = self._get_cost(endpoint)
        
        with self._lock:
            self._apply_decay()
            
            # Check cooldown
            if self._state.matching_trigger_until > time.time():
                return self._state.matching_trigger_until - time.time()
            
            # Check order limit
            if endpoint in ['AddOrder', 'EditOrder']:
                if self._state.order_count >= self.spec['order_limit']:
                    elapsed = time.time() - self._state.order_window_start
                    return max(0, self.spec['order_window'] - elapsed)
            
            # Check counter
            if cost > 0:
                if self._state.counter + cost > self.spec['max_counter']:
                    excess = (self._state.counter + cost) - self.spec['max_counter']
                    return excess / self.spec['decay_rate']
            
            return 0.0
    
    def _record_request(
        self,
        endpoint: str,
        cost: int,
        blocked: bool,
        wait_time: float
    ):
        """Record request for history."""
        record = RequestRecord(
            endpoint=endpoint,
            cost=cost,
            timestamp=time.time(),
            blocked=blocked,
            wait_time=wait_time
        )
        self._history.append(record)
    
    def get_history(self, limit: int = 100) -> List[RequestRecord]:
        """Get recent request history."""
        return list(self._history)[-limit:]
    
    # ───────────────────────────────────────────────────────────────────────────
    # RESET
    # ───────────────────────────────────────────────────────────────────────────
    
    def reset(self):
        """Reset rate limiter state."""
        with self._lock:
            self._state = RateLimitState()
            self._state.last_decay = time.time()
            self._state.order_window_start = time.time()
            
            if self._shared:
                self._shared.update(self._state)
            
            logger.info("Rate limiter reset")


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED MEMORY STATE
# ═══════════════════════════════════════════════════════════════════════════════

class SharedRateLimitState:
    """
    Shared memory representation of rate limit state.
    
    For cross-process visibility on 24-core system.
    """
    
    # Layout: counter(8) + last_decay(8) + order_count(4) + order_window_start(8) + 
    #         blocked(4) + matching_until(8)
    BUFFER_SIZE = 40
    FORMAT = '<ddiqiq'  # double, double, int, quad, int, quad
    
    def __init__(self, name: str, create: bool = True):
        self.name = f"kraken_ratelimit_{name}"
        self._shm: Optional[shared_memory.SharedMemory] = None
        self._lock = threading.Lock()
        
        try:
            if create:
                self._shm = shared_memory.SharedMemory(
                    name=self.name,
                    create=True,
                    size=self.BUFFER_SIZE
                )
                self._initialize()
            else:
                self._shm = shared_memory.SharedMemory(name=self.name)
                
            logger.info(f"SharedRateLimitState initialized: {self.name}")
            
        except FileExistsError:
            self._shm = shared_memory.SharedMemory(name=self.name)
    
    def _initialize(self):
        """Initialize with default state."""
        now = time.time()
        data = struct.pack(self.FORMAT, 0.0, now, 0, int(now), 0, 0)
        self._shm.buf[:len(data)] = data
    
    def update(self, state: RateLimitState):
        """Update shared state."""
        if not self._shm:
            return
        
        with self._lock:
            data = struct.pack(
                self.FORMAT,
                state.counter,
                state.last_decay,
                state.order_count,
                int(state.order_window_start),
                state.blocked_requests,
                int(state.matching_trigger_until * 1000)
            )
            self._shm.buf[:len(data)] = data
    
    def get(self) -> Optional[RateLimitState]:
        """Get state from shared memory."""
        if not self._shm:
            return None
        
        with self._lock:
            data = struct.unpack(self.FORMAT, bytes(self._shm.buf[:self.BUFFER_SIZE]))
            
            state = RateLimitState()
            state.counter = data[0]
            state.last_decay = data[1]
            state.order_count = data[2]
            state.order_window_start = float(data[3])
            state.blocked_requests = data[4]
            state.matching_trigger_until = data[5] / 1000.0
            
            return state
    
    def unlink(self):
        """Remove shared memory."""
        if self._shm:
            try:
                self._shm.close()
                self._shm.unlink()
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL RATE LIMITER (SINGLETON)
# ═══════════════════════════════════════════════════════════════════════════════

class GlobalKrakenRateLimiter:
    """
    Global singleton rate limiter for multi-threaded access.
    
    Ensures all threads/processes use the same rate limit state.
    """
    
    _instance: Optional[KrakenTokenBucket] = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(
        cls,
        tier: KrakenTier = KrakenTier.PRO,
        shared_name: str = "global"
    ) -> KrakenTokenBucket:
        """Get or create global rate limiter instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = KrakenTokenBucket(
                    tier=tier,
                    shared_name=shared_name
                )
            return cls._instance
    
    @classmethod
    def reset(cls):
        """Reset global instance."""
        with cls._lock:
            if cls._instance:
                cls._instance.reset()
                cls._instance = None


# ═══════════════════════════════════════════════════════════════════════════════
# DECORATOR FOR AUTOMATIC RATE LIMITING
# ═══════════════════════════════════════════════════════════════════════════════

def rate_limited(endpoint: str, bucket: KrakenTokenBucket = None):
    """
    Decorator for automatic rate limiting of API calls.
    
    Usage:
        @rate_limited('Balance')
        async def get_balance():
            ...
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            limiter = bucket or GlobalKrakenRateLimiter.get_instance()
            acquired, wait_time = limiter.acquire(endpoint, block=True)
            
            if not acquired:
                raise RuntimeError(f"Rate limit exceeded for {endpoint}")
            
            return await func(*args, **kwargs)
        
        def sync_wrapper(*args, **kwargs):
            limiter = bucket or GlobalKrakenRateLimiter.get_instance()
            acquired, wait_time = limiter.acquire(endpoint, block=True)
            
            if not acquired:
                raise RuntimeError(f"Rate limit exceeded for {endpoint}")
            
            return func(*args, **kwargs)
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return wrapper
        return sync_wrapper
    
    return decorator


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Main classes
    'KrakenTokenBucket',
    'GlobalKrakenRateLimiter',
    
    # Enums
    'KrakenTier',
    
    # Data structures
    'RateLimitState',
    'RequestRecord',
    
    # Shared state
    'SharedRateLimitState',
    
    # Decorator
    'rate_limited',
    
    # Constants
    'TIER_SPECS',
    'API_COSTS',
]


# ═══════════════════════════════════════════════════════════════════════════════
# TESTING
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("KrakenTokenBucket Test Suite")
    print("=" * 70)
    
    # Test 1: Basic Tier Configuration
    print("\n[1] Tier Configurations")
    print("-" * 50)
    
    for tier in KrakenTier:
        spec = TIER_SPECS[tier]
        print(f"{tier.name}:")
        print(f"  Max Counter: {spec['max_counter']}")
        print(f"  Decay Rate: {spec['decay_rate']}/s")
        print(f"  Order Limit: {spec['order_limit']}/min")
    
    # Test 2: Basic Rate Limiting
    print("\n[2] Basic Rate Limiting")
    print("-" * 50)
    
    bucket = KrakenTokenBucket(tier=KrakenTier.PRO)
    
    # Make several requests
    for endpoint in ['Balance', 'Ticker', 'Depth', 'TradesHistory']:
        acquired, wait_time = bucket.acquire(endpoint, block=False)
        status = bucket.get_status()
        print(f"{endpoint}: acquired={acquired}, counter={status['counter']:.2f}, cost={API_COSTS.get(endpoint, 1)}")
    
    # Test 3: Order Rate Limiting
    print("\n[3] Order Rate Limiting")
    print("-" * 50)
    
    # Make several order requests
    for i in range(5):
        acquired, wait_time = bucket.acquire('AddOrder', block=False)
        status = bucket.get_status()
        print(f"Order {i+1}: acquired={acquired}, order_count={status['order_count']}")
    
    # Test 4: Counter Decay
    print("\n[4] Counter Decay")
    print("-" * 50)
    
    initial = bucket.get_status()['counter']
    print(f"Initial counter: {initial:.2f}")
    
    time.sleep(2)  # Wait for decay
    
    after = bucket.get_status()['counter']
    print(f"After 2s: {after:.2f}")
    print(f"Decayed: {initial - after:.2f} (expected: {2 * bucket.spec['decay_rate']:.2f})")
    
    # Test 5: Wait Time Estimation
    print("\n[5] Wait Time Estimation")
    print("-" * 50)
    
    # Fill up counter
    bucket.reset()
    for _ in range(18):
        bucket.acquire('Depth', block=False)
    
    wait = bucket.get_wait_time('TradesHistory')  # Cost 2
    status = bucket.get_status()
    print(f"Counter: {status['counter']:.2f}/{status['max_counter']}")
    print(f"Estimated wait for TradesHistory (cost 2): {wait:.2f}s")
    
    # Test 6: Matching Cooldown
    print("\n[6] Matching Cooldown (Post-Only)")
    print("-" * 50)
    
    bucket.reset()
    bucket.trigger_matching_cooldown(duration=2.0)
    
    print(f"In cooldown: {bucket.is_in_cooldown()}")
    
    acquired, wait_time = bucket.acquire('AddOrder', block=False)
    print(f"AddOrder during cooldown: acquired={acquired}, wait={wait_time:.2f}s")
    
    time.sleep(2.1)
    
    print(f"After cooldown: {bucket.is_in_cooldown()}")
    acquired, wait_time = bucket.acquire('AddOrder', block=False)
    print(f"AddOrder after cooldown: acquired={acquired}")
    
    # Test 7: Reprice Logic
    print("\n[7] Reprice Logic")
    print("-" * 50)
    
    original_buy = Decimal('97000.0')
    tick_size = Decimal('0.1')
    
    repriced_buy = bucket.get_repriced_order('buy', original_buy, tick_size, ticks_away=1)
    print(f"Buy reprice: {original_buy} -> {repriced_buy}")
    
    original_sell = Decimal('97000.0')
    repriced_sell = bucket.get_repriced_order('sell', original_sell, tick_size, ticks_away=1)
    print(f"Sell reprice: {original_sell} -> {repriced_sell}")
    
    # Test 8: Blocking Mode
    print("\n[8] Blocking Mode")
    print("-" * 50)
    
    bucket.reset()
    
    # Fill counter
    for _ in range(19):
        bucket.acquire('Ticker', block=False)
    
    print(f"Counter before blocking: {bucket.get_status()['counter']:.2f}")
    
    start = time.time()
    acquired, wait_time = bucket.acquire('Ticker', block=True, timeout=5.0)
    elapsed = time.time() - start
    
    print(f"Blocking acquire: acquired={acquired}, elapsed={elapsed:.2f}s")
    
    # Test 9: Request History
    print("\n[9] Request History")
    print("-" * 50)
    
    history = bucket.get_history(limit=5)
    for record in history:
        print(f"  {record.endpoint}: cost={record.cost}, blocked={record.blocked}, wait={record.wait_time:.2f}s")
    
    # Test 10: Global Rate Limiter
    print("\n[10] Global Rate Limiter (Singleton)")
    print("-" * 50)
    
    # Reset any existing global instance
    GlobalKrakenRateLimiter.reset()
    
    # Create without shared memory for test
    global1 = KrakenTokenBucket(tier=KrakenTier.PRO, shared_name=None)
    global2 = KrakenTokenBucket(tier=KrakenTier.PRO, shared_name=None)
    
    print(f"Both are PRO tier: {global1.tier == global2.tier}")
    print(f"Tier: {global1.tier.name}")
    
    # Test 11: Shared Memory
    print("\n[11] Shared Memory State")
    print("-" * 50)
    
    try:
        import uuid
        unique_name = f"test_rl_{uuid.uuid4().hex[:8]}"
        shared = SharedRateLimitState(unique_name, create=True)
        
        state = RateLimitState()
        state.counter = 5.5
        state.order_count = 10
        state.blocked_requests = 2
        state.last_decay = time.time()
        state.order_window_start = time.time()
        state.matching_trigger_until = 0.0
        shared.update(state)
        
        loaded = shared.get()
        print(f"Shared counter: {loaded.counter}")
        print(f"Shared order_count: {loaded.order_count}")
        print(f"Shared blocked: {loaded.blocked_requests}")
        
        shared.unlink()
        print("Shared memory cleaned up")
    except Exception as e:
        print(f"Shared memory test skipped: {e}")
    
    print("\n" + "=" * 70)
    print("✓ All KrakenTokenBucket tests completed!")
    print("=" * 70)
