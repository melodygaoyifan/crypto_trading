"""
================================================================================
EXECUTION MANAGER - Smart Order Routing
================================================================================
Version: 3.1.2
Purpose: Professional-grade order execution with slippage control

Key Features:
- Limit orders preferred, market fallback
- Exchange-native stop losses
- Slippage monitoring
- Position reconciliation
================================================================================
"""

import hashlib
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, List, Any, Tuple
from datetime import datetime, timezone
import time

# [P0-03] DEX venues - execution FORBIDDEN under SINGLE_EXCHANGE_MODE
_DEX_VENUES = {
    "jupiter", "raydium", "orca", "dex", "serum",
    "drift", "mango", "lifinity", "phoenix",
}


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    STOP_LIMIT = "STOP_LIMIT"
    TAKE_PROFIT = "TAKE_PROFIT"


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class StopOrderFailurePolicy:
    """Policy to execute when stop-order retries are exhausted."""
    ALERT_ONLY = "ALERT_ONLY"
    FREEZE_ENTRIES = "FREEZE_ENTRIES"
    RETRY_THEN_MARKET = "RETRY_THEN_MARKET"  # [FIX-L2-03] Market close + freeze
    EMERGENCY_FLATTEN = "EMERGENCY_FLATTEN"


@dataclass
class ExecutionConfig:
    """Execution configuration."""
    # Order preferences
    prefer_limit_orders: bool = True
    limit_order_timeout_seconds: float = 120.0  # 120s for maker fills
    market_fallback_enabled: bool = True
    post_only: bool = True  # Kraken oflags='post' -> maker-only, reject if would cross

    # Slippage control
    max_slippage_pct: float = 0.002  # 0.2%

    # Retry settings
    max_retries: int = 3
    retry_delay_seconds: float = 1.0

    # Stop-order retry (P0-02)
    stop_order_max_retries: int = 3
    stop_order_backoff_ms: List[int] = field(default_factory=lambda: [200, 400, 800])
    stop_order_failure_policy: str = StopOrderFailurePolicy.ALERT_ONLY
    stop_order_freeze_seconds: int = 3600  # 1h freeze after exhausted retries

    # Maker cancel-replace (P2-01)
    enable_maker_reprice: bool = True
    maker_reprice_max_attempts: int = 3
    maker_reprice_wait_seconds: float = 20.0
    maker_reprice_min_fill_ratio: float = 0.50
    maker_reprice_improve_bps_schedule: List[int] = field(
        default_factory=lambda: [8, 5, 3]
    )

    # Position management
    reconcile_interval_iterations: int = 5
    use_exchange_stops: bool = True


@dataclass
class OrderResult:
    """Result of an order execution."""
    success: bool
    order_id: Optional[str] = None
    symbol: str = ""
    side: str = ""
    order_type: str = ""
    requested_price: float = 0.0
    filled_price: float = 0.0
    requested_size: float = 0.0
    filled_size: float = 0.0
    slippage: float = 0.0
    fee: float = 0.0
    fee_currency: str = ""
    status: OrderStatus = OrderStatus.PENDING
    error_message: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    userref: Optional[int] = None  # [C8] Kraken userref for idempotency
    raw_response: Dict = field(default_factory=dict)
    # P2-01: Maker reprice KPI
    maker_reprice_attempts: int = 0
    maker_reprice_cancel_count: int = 0
    time_to_fill_seconds: float = 0.0

    def to_dict(self) -> Dict:
        d = {
            "success": self.success,
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "order_type": self.order_type,
            "requested_price": self.requested_price,
            "filled_price": self.filled_price,
            "requested_size": self.requested_size,
            "filled_size": self.filled_size,
            "slippage": self.slippage,
            "fee": self.fee,
            "status": self.status.value,
            "error_message": self.error_message,
            "timestamp": self.timestamp.isoformat(),
            "userref": self.userref,
        }
        if self.maker_reprice_attempts > 0:
            d["maker_reprice_attempts"] = self.maker_reprice_attempts
            d["maker_reprice_cancel_count"] = self.maker_reprice_cancel_count
            d["time_to_fill_seconds"] = self.time_to_fill_seconds
            d["final_fill_ratio"] = (
                self.filled_size / self.requested_size
                if self.requested_size > 0 else 0.0
            )
        return d


class ExecutionManager:
    """
    Professional-grade order execution manager.
    
    Responsibilities:
    1. Smart order routing (limit preferred, market fallback)
    2. Slippage control and monitoring
    3. Exchange stop loss placement
    4. Position reconciliation with exchange
    """
    
    def __init__(self, 
                 exchange,
                 config: Optional[ExecutionConfig] = None,
                 dry_run: bool = True):
        """
        Initialize ExecutionManager.
        
        Args:
            exchange: CCXT exchange instance
            config: Execution configuration
            dry_run: If True, don't execute real orders
        """
        self.exchange = exchange
        self.config = config or ExecutionConfig()
        self.dry_run = dry_run
        self.logger = logging.getLogger(__name__)
        
        # Order tracking
        self.pending_orders: Dict[str, Dict] = {}
        self.filled_orders: List[OrderResult] = []
        
        # Stop loss tracking
        self.active_stops: Dict[str, str] = {}  # symbol -> stop_order_id
        
        # Slippage statistics
        self.slippage_history: List[float] = []

        # [C8] Userref idempotency: userref -> order_id for dedup on reconnect
        self._userref_history: Dict[int, str] = {}

        # [P0-02] Stop-order failure freeze: entries blocked until this timestamp
        self._entries_frozen_until: float = 0.0

        self.logger.info(f"ExecutionManager initialized (dry_run={dry_run})")
    
    # =========================================================================
    # [C8] ORDER IDEMPOTENCY via Kraken userref
    # =========================================================================

    @staticmethod
    def _generate_userref(symbol: str, side: str, tick_id: str) -> int:
        """
        Generate deterministic Kraken userref (signed int32) from order params.

        Same (symbol, side, tick_id) always produces the same userref,
        so reconnect can detect if the order was already placed.
        Kraken userref range: 0 to 2^31-1 (signed 32-bit).
        """
        raw = f"{symbol}_{side}_{tick_id}"
        h = hashlib.sha256(raw.encode()).hexdigest()
        return int(h[:8], 16) & 0x7FFFFFFF  # 31-bit positive int

    def check_userref_executed(self, userref: int) -> Optional[Dict]:
        """
        Query exchange for an order matching this userref.

        Returns order dict if found and filled, else None.
        Used during reconnect to avoid duplicate orders.
        """
        if self.dry_run:
            return self._userref_history.get(userref)

        try:
            # Kraken: fetch closed orders filtered by userref
            orders = self.exchange.fetch_closed_orders(
                params={'userref': userref}
            )
            for order in orders:
                if order.get('info', {}).get('userref') == str(userref):
                    self.logger.info(
                        f"[IDEMPOTENT] userref={userref} already executed: "
                        f"order_id={order.get('id')}"
                    )
                    return order
        except Exception as e:
            self.logger.debug(f"[IDEMPOTENT] userref lookup failed: {e}")

        return None

    # =========================================================================
    # [P0-02] STOP-ORDER RETRY + IDEMPOTENCY + FAILURE POLICY
    # =========================================================================

    @staticmethod
    def _generate_stop_userref(symbol: str, side: str, stop_price: float, suffix: str = "SL") -> int:
        """
        Generate deterministic userref for a stop order.

        Same (symbol, side, stop_price, suffix) always produces the same userref,
        ensuring retry idempotency - retries won't create duplicate orders.
        """
        raw = f"{symbol}_{side}_{stop_price:.8f}_{suffix}"
        h = hashlib.sha256(raw.encode()).hexdigest()
        return int(h[:8], 16) & 0x7FFFFFFF

    def _with_stop_retries(
        self,
        fn,
        *,
        what: str,
        symbol: str,
        userref: int,
        stop_side: Optional[str] = None,
        stop_size: float = 0.0,
    ) -> OrderResult:
        """
        Execute a stop-order placement with exponential backoff retries.

        Before each retry, checks if a matching userref already exists on the
        exchange (idempotency guard). On final failure, applies the configured
        stop_order_failure_policy.

        Args:
            fn: Callable that returns OrderResult (the actual placement).
            what: Human-readable label (e.g. "stop_loss", "stop_limit").
            symbol: Trading pair.
            userref: Deterministic userref for dedup.

        Returns:
            OrderResult - success on any attempt, or final failure after retries.
        """
        max_retries = self.config.stop_order_max_retries
        backoff_ms = self.config.stop_order_backoff_ms
        last_error: Optional[str] = None

        for attempt in range(1, max_retries + 1):
            # Idempotency guard: check if a prior attempt already landed
            if attempt > 1:
                existing = self.check_userref_executed(userref)
                if existing:
                    self.logger.info(
                        f"[STOP-RETRY] {what} {symbol}: already placed "
                        f"(userref={userref}), treating as success"
                    )
                    return OrderResult(
                        success=True,
                        order_id=existing.get('id', str(userref)),
                        symbol=symbol,
                        order_type=what,
                        userref=userref,
                        status=OrderStatus.OPEN,
                    )

            try:
                result = fn()
                if result.success:
                    result.userref = userref
                    if userref is not None:
                        self._userref_history[userref] = result.order_id
                    return result
                # Non-exception failure (API returned error body)
                last_error = result.error_message or "unknown API error"
            except Exception as exc:
                last_error = str(exc)

            self.logger.error(
                f"[STOP-RETRY] {what} {symbol}: attempt {attempt}/{max_retries} "
                f"FAILED - {last_error}"
            )

            if attempt < max_retries:
                delay_ms = backoff_ms[min(attempt - 1, len(backoff_ms) - 1)]
                time.sleep(delay_ms / 1000.0)

        # All retries exhausted
        self.logger.critical(
            f"[STOP-ORDER] FAILED_AFTER_RETRIES: {what} for {symbol} "
            f"failed {max_retries} times. Last error: {last_error}. "
            f"Policy: {self.config.stop_order_failure_policy}"
        )
        self._apply_stop_failure_policy(what, symbol, last_error,
                                        stop_side=stop_side, stop_size=stop_size)

        return OrderResult(
            success=False,
            symbol=symbol,
            order_type=what,
            status=OrderStatus.REJECTED,
            error_message=f"RETRIES_EXHAUSTED({max_retries}): {last_error}",
            userref=userref,
        )

    def _apply_stop_failure_policy(
        self,
        what: str,
        symbol: str,
        last_error: Optional[str],
        *,
        stop_side: Optional[str] = None,
        stop_size: float = 0.0,
    ) -> None:
        """
        Execute the configured failure policy after stop-order retries are exhausted.

        ALERT_ONLY: log CRITICAL (already done by caller).
        FREEZE_ENTRIES: block new entries for stop_order_freeze_seconds.
        RETRY_THEN_MARKET: [FIX-L2-03] market close the unprotected position + freeze entries.
        EMERGENCY_FLATTEN: flatten all (only for live + explicit opt-in).
        """
        policy = self.config.stop_order_failure_policy

        if policy == StopOrderFailurePolicy.FREEZE_ENTRIES:
            self._entries_frozen_until = time.monotonic() + self.config.stop_order_freeze_seconds
            self.logger.critical(
                f"[STOP-ORDER] FREEZE_ENTRIES activated: new entries blocked for "
                f"{self.config.stop_order_freeze_seconds}s after {what} failure on {symbol}"
            )

        elif policy == StopOrderFailurePolicy.RETRY_THEN_MARKET:
            # [FIX-L2-03] Last-resort market close of the unprotected position
            self.logger.critical(
                f"[STOP-ORDER] RETRY_THEN_MARKET: stop {what} failed for {symbol} - "
                f"attempting market close (side={stop_side}, size={stop_size:.6f})"
            )
            if stop_side and stop_size > 0 and not self.dry_run:
                try:
                    mkt_result = self.execute_order(
                        symbol=symbol,
                        side=stop_side,
                        size=stop_size,
                        order_type="MARKET",
                    )
                    if mkt_result.success:
                        self.logger.critical(
                            f"[STOP-ORDER] RETRY_THEN_MARKET: market close SUCCESS "
                            f"for {symbol} - order_id={mkt_result.order_id}"
                        )
                    else:
                        self.logger.critical(
                            f"[STOP-ORDER] RETRY_THEN_MARKET: market close FAILED "
                            f"for {symbol} - {mkt_result.error_message}"
                        )
                except Exception as e:
                    self.logger.critical(
                        f"[STOP-ORDER] RETRY_THEN_MARKET: market close EXCEPTION "
                        f"for {symbol} - {e}"
                    )
            elif self.dry_run:
                self.logger.critical(
                    f"[STOP-ORDER] RETRY_THEN_MARKET: dry_run=True, "
                    f"would have market-closed {stop_size:.6f} {symbol} ({stop_side})"
                )
            # Also freeze entries (same as FREEZE_ENTRIES)
            self._entries_frozen_until = time.monotonic() + self.config.stop_order_freeze_seconds
            self.logger.critical(
                f"[STOP-ORDER] RETRY_THEN_MARKET: entries frozen for "
                f"{self.config.stop_order_freeze_seconds}s"
            )

        elif policy == StopOrderFailurePolicy.EMERGENCY_FLATTEN:
            if not self.dry_run:
                self.logger.critical(
                    f"[STOP-ORDER] EMERGENCY_FLATTEN triggered after {what} failure on {symbol}"
                )
                try:
                    self.cancel_all_open_orders(reason=f"stop_order_failure_{what}")
                except Exception as e:
                    self.logger.error(f"[STOP-ORDER] Emergency flatten failed: {e}")
            else:
                self.logger.critical(
                    f"[STOP-ORDER] EMERGENCY_FLATTEN requested but dry_run=True, "
                    f"treating as ALERT_ONLY for {what} on {symbol}"
                )
        # ALERT_ONLY: CRITICAL log already emitted by _with_stop_retries

    def is_entries_frozen(self) -> bool:
        """Check if new entries are blocked due to stop-order failure."""
        if self._entries_frozen_until <= 0:
            return False
        if time.monotonic() >= self._entries_frozen_until:
            self._entries_frozen_until = 0.0
            return False
        return True

    def execute_order(self,
                     symbol: str,
                     side,  # OrderSide or str for v5.1.0-HARDENED compatibility
                     size: float,
                     price: Optional[float] = None,
                     order_type = None,
                     venue: str = "kraken",
                     leverage: Optional[int] = None,
                     spread_bps: float = 10.0,
                     tick_id: str = "",
                     taker_allowed: bool = True,
                     vpin: float = 0.35,
                     bid_depth_usd: float = 50_000.0,
                     ask_depth_usd: float = 50_000.0) -> OrderResult:  # OrderType or str
        """
        Execute an order with smart routing.

        v5.1.0-HARDENED: Now accepts string or enum for side and order_type
        for glue-layer compatibility with main.py.

        v5.3.0-CLOUD: SINGLE_EXCHANGE_MODE enforced - only Kraken allowed.

        Args:
            symbol: Trading pair
            side: BUY or SELL (string or OrderSide enum)
            size: Position size
            price: Limit price (None for market)
            order_type: Order type override (string or OrderType enum)
            venue: Execution venue (MUST be "kraken")
            leverage: Kraken isolated margin leverage (2 or 3). None = no margin.
            taker_allowed: If False, MARKET orders are downgraded to LIMIT
                (defense-in-depth for fee-tier-aware execution, P2-02).

        Returns:
            OrderResult with execution details
        """
        # =================================================================
        # v5.3.0-CLOUD: SINGLE EXCHANGE HARD GATE (LOCKED)
        # [P0-03] Enhanced: DEX venues get distinct CRITICAL + proof event
        # =================================================================
        if venue.lower() != "kraken":
            _side_str = side.value if hasattr(side, 'value') else str(side)
            _is_dex = venue.lower() in _DEX_VENUES
            if _is_dex:
                self.logger.critical(
                    f"[DEX_EXECUTION_BLOCKED] venue={venue} symbol={symbol} "
                    f"side={_side_str} - DEX execution FORBIDDEN under "
                    f"SINGLE_EXCHANGE_MODE. DEX is monitoring/signal only."
                )
            else:
                self.logger.error(
                    f"[SINGLE_EXCHANGE_GATE] FORBIDDEN: venue={venue} != kraken"
                )
            return OrderResult(
                success=False,
                symbol=symbol,
                side=_side_str,
                error_message=(
                    f"DEX_EXECUTION_BLOCKED: venue={venue}" if _is_dex
                    else f"SINGLE_EXCHANGE_VIOLATION: venue={venue} is FORBIDDEN. Only kraken is allowed."
                ),
                status=OrderStatus.REJECTED,
            )
        
        # =================================================================
        # V6.2.3e TASK 6: DISCONNECT CHECK
        # Block new orders if disconnected
        # =================================================================
        if not self.is_ready_for_orders():
            self.logger.error("[DISCONNECT] Order rejected - connection lost")
            return OrderResult(
                success=False,
                symbol=symbol,
                side=side.value if hasattr(side, 'value') else str(side),
                error_message="ORDER_BLOCKED: Connection lost - reconnection required",
                status=OrderStatus.REJECTED
            )
        
        # [P0-02] FREEZE_ENTRIES check - block new entries after stop-order failure
        if self.is_entries_frozen():
            _side_str = side.value if hasattr(side, 'value') else str(side)
            # Allow exits (SELL when long / BUY when short) but block entries
            self.logger.warning(
                f"[FREEZE_ENTRIES] Order {_side_str} {symbol} blocked - "
                f"entries frozen after stop-order failure"
            )
            return OrderResult(
                success=False,
                symbol=symbol,
                side=_side_str,
                error_message="ORDER_BLOCKED: Entries frozen after stop-order failure",
                status=OrderStatus.REJECTED,
            )

        # v5.1.0-HARDENED: Convert string to enum if needed
        if isinstance(side, str):
            side = OrderSide[side.upper()]

        if isinstance(order_type, str):
            order_type = OrderType[order_type.upper()]

        # Determine order type
        if order_type is None:
            if self.config.prefer_limit_orders and price is not None:
                order_type = OrderType.LIMIT
            else:
                order_type = OrderType.MARKET

        # P2-02: Fee-tier taker guard - downgrade MARKET -> LIMIT when taker blocked
        if not taker_allowed and order_type == OrderType.MARKET:
            if price is not None:
                order_type = OrderType.LIMIT
                self.logger.info(
                    f"[FEE_TIER_GUARD] {symbol}: MARKET->LIMIT (taker blocked by fee tier)"
                )
            else:
                self.logger.warning(
                    f"[FEE_TIER_GUARD] {symbol}: taker blocked but no limit price - "
                    f"allowing MARKET as last resort"
                )

        # [C8] Generate deterministic userref for idempotency
        userref = None
        if tick_id:
            userref = self._generate_userref(symbol, side.value, tick_id)
            self.logger.info(
                f"Executing {order_type.value} {side.value} {size:.6f} {symbol} "
                f"@ {price if price else 'MARKET'} (userref={userref})"
            )
        else:
            self.logger.info(
                f"Executing {order_type.value} {side.value} {size:.6f} {symbol} "
                f"@ {price if price else 'MARKET'}"
            )

        if self.dry_run:
            result = self._simulate_order(
                symbol, side, size, price, order_type,
                spread_bps=spread_bps, vpin=vpin,
                bid_depth_usd=bid_depth_usd, ask_depth_usd=ask_depth_usd,
            )
            result.userref = userref
            if userref is not None:
                self._userref_history[userref] = result.order_id
            return result

        # [C8] Dedup check: if userref already executed, return cached result
        if userref is not None:
            existing = self.check_userref_executed(userref)
            if existing:
                self.logger.warning(
                    f"[IDEMPOTENT] Order already executed: userref={userref}, "
                    f"order_id={existing.get('id')} - skipping duplicate"
                )
                return OrderResult(
                    success=True,
                    order_id=existing.get('id'),
                    symbol=symbol,
                    side=side.value,
                    order_type=order_type.value,
                    filled_price=existing.get('average', 0),
                    filled_size=existing.get('filled', 0),
                    status=OrderStatus.FILLED,
                    userref=userref,
                    raw_response=existing,
                )

        # Build Kraken margin params if leveraged
        margin_params = {}
        if leverage and leverage > 1:
            margin_params['leverage'] = leverage
            self.logger.info(f"[MARGIN] Kraken isolated margin {leverage}x for {symbol}")
        if userref is not None:
            margin_params['userref'] = userref

        # Try limit order first
        if order_type == OrderType.LIMIT:
            # P2-01: Route to maker reprice loop when enabled
            if self.config.enable_maker_reprice and tick_id:
                result = self._execute_limit_with_reprice(
                    symbol, side, size, price, margin_params, tick_id,
                )
            else:
                result = self._execute_limit_order(symbol, side, size, price, margin_params)

                # [L3-05] Retry on timeout: refresh price + re-attempt (up to 2 retries)
                _l3_max_retries = 2
                for _l3_attempt in range(_l3_max_retries):
                    if result.success:
                        break
                    if result.status != OrderStatus.CANCELLED:
                        break  # non-timeout failure (rejected, etc.) - don't retry
                    if "timeout" not in (result.error_message or "").lower():
                        break  # not a timeout - don't retry
                    # Refresh best price from exchange
                    try:
                        _l3_ticker = self.exchange.fetch_ticker(symbol)
                        _l3_new_price = (
                            _l3_ticker.get('bid', price) if side == OrderSide.BUY
                            else _l3_ticker.get('ask', price)
                        )
                        self.logger.info(
                            f"[ORDER_RETRY] Attempt {_l3_attempt + 2}/{_l3_max_retries + 1} "
                            f"for {symbol} at ${_l3_new_price:.2f} (was ${price:.2f})"
                        )
                        # Generate new userref for retry
                        _l3_retry_params = dict(margin_params)
                        if userref is not None:
                            _l3_new_ref = (userref + _l3_attempt + 1) & 0x7FFFFFFF
                            _l3_retry_params['userref'] = _l3_new_ref
                        result = self._execute_limit_order(
                            symbol, side, size, _l3_new_price, _l3_retry_params,
                        )
                    except Exception as _l3_err:
                        self.logger.warning(f"[ORDER_RETRY] Price refresh failed: {_l3_err} - stopping retries")
                        break

            # Fallback to market if limit times out
            # P2-01: Reprice loop handles its own retries - no market fallback
            _used_reprice = result.maker_reprice_attempts > 0
            if not result.success and self.config.market_fallback_enabled and not _used_reprice:
                self.logger.warning("Limit order failed, falling back to market")
                result = self._execute_market_order(symbol, side, size, margin_params)
        else:
            result = self._execute_market_order(symbol, side, size, margin_params)
        
        # Track result
        result.userref = userref
        if result.success:
            self.filled_orders.append(result)
            if result.slippage != 0:
                self.slippage_history.append(result.slippage)
            if userref is not None:
                self._userref_history[userref] = result.order_id

        return result
    
    def _execute_limit_order(self,
                            symbol: str,
                            side: OrderSide,
                            size: float,
                            price: float,
                            extra_params: Optional[Dict] = None) -> OrderResult:
        """Execute a post-only limit order with 120s timeout + partial fill handling."""
        try:
            # Build order params: merge margin params + post-only flag
            order_params = dict(extra_params or {})

            # --- [v9-PATCH-3] Default maker, emergency taker ---
            _is_emergency = order_params.pop('_is_emergency', False)
            _emergency_reason = order_params.pop('_emergency_reason', '')
            if _is_emergency:
                # Emergency override: stop-loss, liquidation guard, CRITICAL drift
                order_params['postOnly'] = False
                order_params.pop('oflags', None)
                self.logger.warning(f"[v9-PATCH-3] TAKER_FALLBACK: reason={_emergency_reason}")
            elif self.config.post_only:
                order_params['postOnly'] = True          # ccxt generic
                order_params['oflags'] = 'post'          # Kraken-specific
            # --- end [v9-PATCH-3] ---

            order = self.exchange.create_limit_order(
                symbol=symbol,
                side=side.value.lower(),
                amount=size,
                price=price,
                params=order_params,
            )

            order_id = order.get('id')
            self.pending_orders[order_id] = order
            self.logger.info(
                f"[LIMIT] Placed post-only {side.value} {size:.6f} {symbol} "
                f"@ {price:.2f} (timeout={self.config.limit_order_timeout_seconds}s)"
            )

            # Wait for fill with timeout
            start_time = time.time()
            while time.time() - start_time < self.config.limit_order_timeout_seconds:
                order_status = self.exchange.fetch_order(order_id, symbol)
                status = order_status.get('status', '')

                if status == 'closed':
                    filled_price = order_status.get('average', price)
                    slippage = (filled_price - price) / price if price > 0 else 0

                    return OrderResult(
                        success=True,
                        order_id=order_id,
                        symbol=symbol,
                        side=side.value,
                        order_type=OrderType.LIMIT.value,
                        requested_price=price,
                        filled_price=filled_price,
                        requested_size=size,
                        filled_size=order_status.get('filled', size),
                        slippage=slippage,
                        fee=order_status.get('fee', {}).get('cost', 0),
                        fee_currency=order_status.get('fee', {}).get('currency', ''),
                        status=OrderStatus.FILLED,
                        raw_response=order_status
                    )

                if status == 'canceled' or status == 'expired':
                    # Post-only rejected (would have crossed spread) or exchange cancelled
                    return OrderResult(
                        success=False,
                        order_id=order_id,
                        symbol=symbol,
                        side=side.value,
                        order_type=OrderType.LIMIT.value,
                        status=OrderStatus.CANCELLED,
                        error_message=f"Order {status} by exchange (post-only rejected?)"
                    )

                time.sleep(2.0)  # Check every 2s (was 0.5s - less aggressive polling)

            # Timeout - check for partial fill before cancelling
            order_status = self.exchange.fetch_order(order_id, symbol)
            filled_qty = order_status.get('filled', 0) or 0

            self.exchange.cancel_order(order_id, symbol)

            if filled_qty > 0 and filled_qty >= size * 0.5:
                # Partial fill ≥50% - accept it
                filled_price = order_status.get('average', price)
                self.logger.info(
                    f"[LIMIT] Partial fill accepted: {filled_qty:.6f}/{size:.6f} "
                    f"({filled_qty/size*100:.0f}%)"
                )
                return OrderResult(
                    success=True,
                    order_id=order_id,
                    symbol=symbol,
                    side=side.value,
                    order_type=OrderType.LIMIT.value,
                    requested_price=price,
                    filled_price=filled_price,
                    requested_size=size,
                    filled_size=filled_qty,
                    slippage=(filled_price - price) / price if price > 0 else 0,
                    fee=order_status.get('fee', {}).get('cost', 0),
                    fee_currency=order_status.get('fee', {}).get('currency', ''),
                    status=OrderStatus.FILLED,
                    raw_response=order_status
                )

            _remaining = size - filled_qty
            self.logger.warning(
                f"[LIMIT] Timeout after {self.config.limit_order_timeout_seconds}s "
                f"(filled={filled_qty:.6f}/{size:.6f}, remaining={_remaining:.6f})"
            )

            # [FIX-MARKET-FALLBACK] For emergency exits (stop-loss, flatten, DMS),
            # unfilled remainder MUST be market-closed. Leaving positions unhedged
            # after a failed limit exit is worse than taker fees.
            if _is_emergency and _remaining > 0.0:
                self.logger.warning(
                    f"[LIMIT→MARKET] Emergency fallback: market-closing remaining "
                    f"{_remaining:.6f} {symbol} (reason: {_emergency_reason})"
                )
                try:
                    _mkt_result = self._execute_market_order(symbol, side, _remaining, extra_params)
                    if _mkt_result.success:
                        # Combine partial limit fill + market fill
                        _total_filled = filled_qty + _mkt_result.filled_size
                        _avg_price = (
                            (filled_qty * (order_status.get('average', price) or price)
                             + _mkt_result.filled_size * _mkt_result.filled_price)
                            / _total_filled
                        ) if _total_filled > 0 else _mkt_result.filled_price
                        self.logger.info(
                            f"[LIMIT→MARKET] Combined fill: {_total_filled:.6f}/{size:.6f} "
                            f"avg_price={_avg_price:.2f}"
                        )
                        return OrderResult(
                            success=True, order_id=order_id, symbol=symbol,
                            side=side.value, order_type="LIMIT+MARKET",
                            requested_price=price, filled_price=_avg_price,
                            requested_size=size, filled_size=_total_filled,
                            slippage=(_avg_price - price) / price if price > 0 else 0,
                            fee=(order_status.get('fee', {}).get('cost', 0) or 0)
                                + (_mkt_result.fee or 0),
                            status=OrderStatus.FILLED,
                        )
                    else:
                        self.logger.error(
                            f"[LIMIT→MARKET] Market fallback ALSO FAILED: {_mkt_result.error_message}"
                        )
                except Exception as _mkt_err:
                    self.logger.error(f"[LIMIT→MARKET] Market fallback error: {_mkt_err}")

            return OrderResult(
                success=False,
                order_id=order_id,
                symbol=symbol,
                side=side.value,
                order_type=OrderType.LIMIT.value,
                status=OrderStatus.CANCELLED,
                error_message=f"Limit order timeout (partial={filled_qty:.6f})"
            )
            
        except Exception as e:
            self.logger.error(f"Limit order error: {e}")
            return OrderResult(
                success=False,
                symbol=symbol,
                side=side.value,
                order_type=OrderType.LIMIT.value,
                status=OrderStatus.REJECTED,
                error_message=str(e)
            )
    
    # =========================================================================
    # [P2-01] MAKER CANCEL-REPLACE (CHASE) LOOP
    # =========================================================================

    @staticmethod
    def _generate_reprice_userref(symbol: str, side: str, tick_id: str, attempt: int) -> int:
        """Generate deterministic userref for a reprice attempt.

        Each attempt gets a unique but stable userref so retries are idempotent.
        """
        raw = f"{symbol}_{side}_{tick_id}_L{attempt}"
        h = hashlib.sha256(raw.encode()).hexdigest()
        return int(h[:8], 16) & 0x7FFFFFFF

    @staticmethod
    def _compute_post_only_limit_price(
        side: 'OrderSide',
        mid_price: float,
        improve_bps: int,
        best_bid: float,
        best_ask: float,
    ) -> float:
        """Compute limit price that stays on the maker side of the book.

        For BUY: price = mid - improve_bps, clamped <= best_bid (never cross ask).
        For SELL: price = mid + improve_bps, clamped >= best_ask (never cross bid).
        """
        offset = mid_price * improve_bps / 10000.0
        if side == OrderSide.BUY:
            price = mid_price - offset
            # Must not cross the ask - stay at or below best_bid
            return min(price, best_bid)
        else:
            price = mid_price + offset
            # Must not cross the bid - stay at or above best_ask
            return max(price, best_ask)

    def _execute_limit_with_reprice(
        self,
        symbol: str,
        side: 'OrderSide',
        size: float,
        initial_price: float,
        extra_params: Optional[Dict] = None,
        tick_id: str = "",
    ) -> OrderResult:
        """Execute a post-only limit order with cancel-replace repricing.

        P2-01: On timeout or insufficient fill, cancel and re-place at a
        tighter price (closer to mid). Still post-only - never crosses
        into taker territory.

        Returns OrderResult with maker_reprice KPI fields populated.
        """
        cfg = self.config
        max_attempts = cfg.maker_reprice_max_attempts
        wait_seconds = cfg.maker_reprice_wait_seconds
        min_fill = cfg.maker_reprice_min_fill_ratio
        bps_schedule = cfg.maker_reprice_improve_bps_schedule

        remaining_qty = size
        total_filled = 0.0
        total_fee = 0.0
        cancel_count = 0
        weighted_price_sum = 0.0  # For computing weighted avg fill price
        last_order_id = None
        overall_start = time.time()

        self.logger.info(
            f"[REPRICE] Starting maker chase: {side.value} {size:.6f} {symbol} "
            f"max_attempts={max_attempts} wait={wait_seconds}s "
            f"bps_schedule={bps_schedule}"
        )

        for attempt in range(max_attempts):
            # --- Compute this attempt's limit price ---
            improve_bps = bps_schedule[min(attempt, len(bps_schedule) - 1)]

            # Use initial_price as mid proxy (caller provides current market price)
            mid_price = initial_price
            # Try to get live bid/ask from exchange for tighter clamping
            best_bid = mid_price
            best_ask = mid_price
            try:
                ticker = self.exchange.fetch_ticker(symbol)
                if ticker:
                    best_bid = float(ticker.get('bid', mid_price))
                    best_ask = float(ticker.get('ask', mid_price))
                    mid_price = (best_bid + best_ask) / 2.0
            except Exception:
                pass  # Use initial_price as fallback

            limit_price = self._compute_post_only_limit_price(
                side, mid_price, improve_bps, best_bid, best_ask,
            )

            # --- Idempotency: generate per-attempt userref ---
            userref = self._generate_reprice_userref(
                symbol, side.value, tick_id, attempt,
            )

            # Check if this exact attempt already executed
            existing = self.check_userref_executed(userref)
            if existing:
                filled_qty = existing.get('filled', 0) or 0
                if filled_qty > 0:
                    total_filled += filled_qty
                    weighted_price_sum += filled_qty * (existing.get('average', limit_price))
                    remaining_qty -= filled_qty
                    self.logger.info(
                        f"[REPRICE] Attempt {attempt} already filled via userref={userref}: "
                        f"{filled_qty:.6f}"
                    )
                if remaining_qty <= 0 or total_filled / size >= 1.0:
                    break
                continue

            # --- Place post-only limit order ---
            order_params = dict(extra_params or {})
            order_params['userref'] = userref
            if cfg.post_only:
                order_params['postOnly'] = True
                order_params['oflags'] = 'post'

            self.logger.info(
                f"[REPRICE] Attempt {attempt}: {side.value} {remaining_qty:.6f} {symbol} "
                f"@ {limit_price:.2f} (improve={improve_bps}bps, userref={userref})"
            )

            try:
                order = self.exchange.create_limit_order(
                    symbol=symbol,
                    side=side.value.lower(),
                    amount=remaining_qty,
                    price=limit_price,
                    params=order_params,
                )
                order_id = order.get('id')
                last_order_id = order_id
            except Exception as e:
                self.logger.warning(f"[REPRICE] Attempt {attempt} placement failed: {e}")
                continue

            # --- Poll for fill ---
            attempt_start = time.time()
            filled_this_attempt = 0.0
            order_done = False

            while time.time() - attempt_start < wait_seconds:
                try:
                    order_status = self.exchange.fetch_order(order_id, symbol)
                    status = order_status.get('status', '')

                    if status == 'closed':
                        filled_this_attempt = order_status.get('filled', 0) or 0
                        order_done = True
                        break

                    if status in ('canceled', 'expired'):
                        # Post-only rejected or exchange cancelled
                        filled_this_attempt = order_status.get('filled', 0) or 0
                        order_done = True
                        break

                    # Check partial fill progress
                    filled_this_attempt = order_status.get('filled', 0) or 0

                except Exception as e:
                    self.logger.debug(f"[REPRICE] Poll error: {e}")

                time.sleep(2.0)

            # --- Finalize this attempt ---
            if not order_done:
                # Timeout - fetch final state and cancel remainder
                try:
                    order_status = self.exchange.fetch_order(order_id, symbol)
                    filled_this_attempt = order_status.get('filled', 0) or 0
                except Exception:
                    pass

                try:
                    self.exchange.cancel_order(order_id, symbol)
                    cancel_count += 1
                except Exception as e:
                    self.logger.debug(f"[REPRICE] Cancel failed (may be filled): {e}")
                order_done = True  # Already cancelled - don't cancel again below

            if filled_this_attempt > 0:
                try:
                    avg_price = order_status.get('average', limit_price)
                except Exception:
                    avg_price = limit_price
                weighted_price_sum += filled_this_attempt * avg_price
                fee_cost = order_status.get('fee', {}).get('cost', 0) if order_status else 0
                total_fee += fee_cost or 0
                total_filled += filled_this_attempt
                remaining_qty -= filled_this_attempt

            fill_ratio = total_filled / size if size > 0 else 0.0

            self.logger.info(
                f"[REPRICE] Attempt {attempt} done: filled={filled_this_attempt:.6f} "
                f"cumulative={total_filled:.6f}/{size:.6f} ({fill_ratio:.0%})"
            )

            # Fully filled
            if fill_ratio >= 1.0:
                break

            # Partial fill exceeds min threshold - accept and stop chasing
            if fill_ratio >= min_fill and remaining_qty > 0:
                self.logger.info(
                    f"[REPRICE] Fill ratio {fill_ratio:.0%} >= min {min_fill:.0%}, "
                    f"accepting partial fill"
                )
                # Cancel any remaining unfilled portion
                if not order_done:
                    try:
                        self.exchange.cancel_order(order_id, symbol)
                        cancel_count += 1
                    except Exception:
                        pass
                break

            # Not enough fill - cancel remainder and try tighter price
            if not order_done and filled_this_attempt < remaining_qty:
                try:
                    self.exchange.cancel_order(order_id, symbol)
                    cancel_count += 1
                except Exception:
                    pass

        # --- Build final result ---
        elapsed = time.time() - overall_start
        fill_ratio = total_filled / size if size > 0 else 0.0
        avg_fill_price = (
            weighted_price_sum / total_filled if total_filled > 0 else initial_price
        )
        attempts_used = min(attempt + 1, max_attempts) if max_attempts > 0 else 0

        if fill_ratio >= min_fill:
            slippage = (avg_fill_price - initial_price) / initial_price if initial_price > 0 else 0
            self.logger.info(
                f"[REPRICE] SUCCESS: {total_filled:.6f}/{size:.6f} ({fill_ratio:.0%}) "
                f"in {attempts_used} attempts, {cancel_count} cancels, {elapsed:.1f}s"
            )
            return OrderResult(
                success=True,
                order_id=last_order_id,
                symbol=symbol,
                side=side.value,
                order_type=OrderType.LIMIT.value,
                requested_price=initial_price,
                filled_price=avg_fill_price,
                requested_size=size,
                filled_size=total_filled,
                slippage=slippage,
                fee=total_fee,
                status=OrderStatus.FILLED,
                maker_reprice_attempts=attempts_used,
                maker_reprice_cancel_count=cancel_count,
                time_to_fill_seconds=elapsed,
            )
        else:
            self.logger.warning(
                f"[REPRICE] DEFERRED: {total_filled:.6f}/{size:.6f} ({fill_ratio:.0%}) "
                f"after {attempts_used} attempts - no taker fallback"
            )
            return OrderResult(
                success=False,
                order_id=last_order_id,
                symbol=symbol,
                side=side.value,
                order_type=OrderType.LIMIT.value,
                requested_price=initial_price,
                filled_price=avg_fill_price if total_filled > 0 else 0.0,
                requested_size=size,
                filled_size=total_filled,
                status=OrderStatus.CANCELLED,
                error_message=(
                    f"Maker reprice exhausted ({attempts_used} attempts, "
                    f"fill={fill_ratio:.0%} < min={min_fill:.0%})"
                ),
                maker_reprice_attempts=attempts_used,
                maker_reprice_cancel_count=cancel_count,
                time_to_fill_seconds=elapsed,
            )

    def _execute_market_order(self,
                             symbol: str,
                             side: OrderSide,
                             size: float,
                             extra_params: Optional[Dict] = None) -> OrderResult:
        """Execute a market order."""
        try:
            order_params = extra_params or {}
            order = self.exchange.create_market_order(
                symbol=symbol,
                side=side.value.lower(),
                amount=size,
                params=order_params,
            )

            if order is None:
                raise RuntimeError("Exchange returned None for market order (possible nonce/auth error)")

            filled_price = order.get('average', order.get('price', 0))
            
            return OrderResult(
                success=True,
                order_id=order.get('id'),
                symbol=symbol,
                side=side.value,
                order_type=OrderType.MARKET.value,
                requested_price=filled_price,
                filled_price=filled_price,
                requested_size=size,
                filled_size=order.get('filled', size),
                slippage=0,
                fee=order.get('fee', {}).get('cost', 0),
                fee_currency=order.get('fee', {}).get('currency', ''),
                status=OrderStatus.FILLED,
                raw_response=order
            )
            
        except Exception as e:
            self.logger.error(f"Market order error: {e}")
            return OrderResult(
                success=False,
                symbol=symbol,
                side=side.value,
                order_type=OrderType.MARKET.value,
                status=OrderStatus.REJECTED,
                error_message=str(e)
            )
    
    # [L3-02] Per-asset depth impact parameters (aligned with EnhancedMarketImpactModel)
    _IMPACT_PARAMS = {
        "BTC": {"base_bps": 3.0, "depth_coefficient": 0.5},
        "ETH": {"base_bps": 5.0, "depth_coefficient": 0.8},
        "SOL": {"base_bps": 10.0, "depth_coefficient": 1.5},
    }

    def _simulate_order(self,
                       symbol: str,
                       side: OrderSide,
                       size: float,
                       price: Optional[float],
                       order_type: OrderType,
                       spread_bps: float = 10.0,
                       vpin: float = 0.35,
                       bid_depth_usd: float = 50_000.0,
                       ask_depth_usd: float = 50_000.0) -> OrderResult:
        """
        Simulate order for dry run mode with depth-aware slippage + partial fill.

        [L3-01] Partial fill: Limit orders at 85% base fill rate, adjusted by
                VPIN (toxicity) and depth ratio (large order penalty).
        [L3-02] Slippage: half_spread + sqrt(depth_impact) + VPIN adverse selection.
        """
        import random
        import uuid

        base_price = price if price else 45000
        order_notional = size * base_price

        # --- Asset identification ---
        _asset_key = symbol.upper().replace("/USD", "").replace("USD", "")
        params = self._IMPACT_PARAMS.get(_asset_key, self._IMPACT_PARAMS["ETH"])

        # --- Relevant depth ---
        relevant_depth = ask_depth_usd if side == OrderSide.BUY else bid_depth_usd
        relevant_depth = max(relevant_depth, 1000.0)  # floor
        depth_ratio = order_notional / relevant_depth

        # ===============================================================
        # [L3-02] Depth-aware slippage model
        # ===============================================================
        half_spread = spread_bps / 2

        # Depth impact: non-linear sqrt model
        depth_impact_bps = params["depth_coefficient"] * (depth_ratio ** 0.5) * 100

        # VPIN adjustment: high toxicity -> more adverse selection (0-5 bps)
        vpin_adj_bps = max(0.0, (vpin - 0.5) * 10.0)

        # Total slippage with ±20% noise
        raw_slippage = half_spread + depth_impact_bps + vpin_adj_bps
        noise = random.uniform(0.8, 1.2)
        slippage_bps = raw_slippage * noise

        slippage_pct = slippage_bps / 10000.0
        if side == OrderSide.BUY:
            filled_price = base_price * (1 + slippage_pct)
        else:
            filled_price = base_price * (1 - slippage_pct)

        # ===============================================================
        # [L3-01] Partial fill simulation
        # ===============================================================
        if order_type == OrderType.MARKET:
            fill_rate = 1.0
        else:
            # Limit order: fill rate depends on depth ratio + VPIN
            base_fill_rate = 0.85
            vpin_fill_adj = (vpin - 0.5) * 0.2          # ±10%
            depth_fill_adj = -min(depth_ratio * 0.5, 0.4)  # cap -40%
            fill_rate = min(1.0, max(0.3, base_fill_rate + vpin_fill_adj + depth_fill_adj))

        filled_size = size * fill_rate
        _fill_tag = "" if fill_rate >= 0.999 else f" fill_rate={fill_rate:.0%}"

        self.logger.info(
            f"[PAPER_SLIP] {symbol} {side.value} spread={spread_bps:.1f}bps "
            f"slippage={slippage_bps:.2f}bps (depth={depth_impact_bps:.1f} vpin={vpin_adj_bps:.1f}) "
            f"fill=${filled_price:.2f}{_fill_tag}"
        )

        return OrderResult(
            success=True,
            order_id=f"SIM_{uuid.uuid4().hex[:8]}",
            symbol=symbol,
            side=side.value,
            order_type=order_type.value,
            requested_price=base_price,
            filled_price=filled_price,
            requested_size=size,
            filled_size=filled_size,
            slippage=slippage_pct if side == OrderSide.BUY else -slippage_pct,
            fee=0.0,  # [T7] actual fee via FeeBlender in main.py
            fee_currency="USDT",
            status=OrderStatus.FILLED
        )

    def place_stop_loss(self,
                       symbol: str,
                       side: OrderSide,
                       size: float,
                       stop_price: float) -> OrderResult:
        """
        Place exchange-native stop loss order with retry + idempotency.

        [P0-02] Retries with exponential backoff on failure. Uses a
        deterministic userref so retries never create duplicate orders.
        On final failure, applies stop_order_failure_policy.
        """
        if not self.config.use_exchange_stops:
            return OrderResult(
                success=False,
                symbol=symbol,
                error_message="Exchange stops disabled"
            )

        userref = self._generate_stop_userref(
            symbol, side.value, stop_price, suffix="SL"
        )

        self.logger.info(
            f"Placing stop loss for {symbol} @ ${stop_price:,.2f} "
            f"(userref={userref})"
        )

        if self.dry_run:
            import uuid
            order_id = f"SL_{uuid.uuid4().hex[:8]}"
            self.active_stops[symbol] = order_id
            self._userref_history[userref] = order_id
            return OrderResult(
                success=True,
                order_id=order_id,
                symbol=symbol,
                side=side.value,
                order_type=OrderType.STOP_LOSS.value,
                requested_price=stop_price,
                status=OrderStatus.OPEN,
                userref=userref,
            )

        def _do_place() -> OrderResult:
            order = self.exchange.create_order(
                symbol=symbol,
                type='stop',
                side=side.value.lower(),
                amount=size,
                price=stop_price,
                params={'stopPrice': stop_price, 'userref': userref},
            )
            order_id = order.get('id')
            self.active_stops[symbol] = order_id
            return OrderResult(
                success=True,
                order_id=order_id,
                symbol=symbol,
                side=side.value,
                order_type=OrderType.STOP_LOSS.value,
                requested_price=stop_price,
                status=OrderStatus.OPEN,
                raw_response=order,
            )

        result = self._with_stop_retries(
            _do_place,
            what="stop_loss",
            symbol=symbol,
            userref=userref,
            stop_side=side.value,  # [FIX-L2-03] pass for RETRY_THEN_MARKET
            stop_size=size,
        )
        if result.success:
            self.active_stops[symbol] = result.order_id
        return result
    
    def cancel_stop_loss(self, symbol: str) -> bool:
        """Cancel active stop loss for a symbol."""
        if symbol not in self.active_stops:
            return True
        
        order_id = self.active_stops[symbol]
        
        if self.dry_run:
            del self.active_stops[symbol]
            return True
        
        try:
            self.exchange.cancel_order(order_id, symbol)
            del self.active_stops[symbol]
            return True
        except Exception as e:
            self.logger.error(f"Failed to cancel stop loss: {e}")
            return False
    
    # =========================================================================
    # V6.2.3e TASK 6: CANCEL-ON-DISCONNECT
    # Ensures no orphaned orders when connection drops
    # =========================================================================
    
    def cancel_all_open_orders(self, reason: str = "disconnect") -> Dict[str, Any]:
        """
        Cancel ALL open orders immediately.
        
        This is the critical cancel-on-disconnect handler.
        Must be called when:
        1. WebSocket connection drops
        2. System shutdown
        3. Fatal error
        4. Manual emergency stop
        
        Returns:
            Summary of cancellation results
        """
        self.logger.critical(f"[CANCEL-ALL] Initiating emergency order cancellation: {reason}")
        
        results = {
            "total_orders": 0,
            "cancelled": 0,
            "failed": 0,
            "errors": [],
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        # Cancel pending orders
        for order_id, order_info in list(self.pending_orders.items()):
            results["total_orders"] += 1
            try:
                if not self.dry_run and self.exchange:
                    symbol = order_info.get('symbol', '')
                    self.exchange.cancel_order(order_id, symbol)
                
                del self.pending_orders[order_id]
                results["cancelled"] += 1
                self.logger.info(f"[CANCEL-ALL] Cancelled order {order_id}")
                
            except Exception as e:
                results["failed"] += 1
                results["errors"].append(f"Order {order_id}: {str(e)}")
                self.logger.error(f"[CANCEL-ALL] Failed to cancel {order_id}: {e}")
        
        # Cancel all stop losses
        for symbol, stop_id in list(self.active_stops.items()):
            results["total_orders"] += 1
            try:
                if not self.dry_run and self.exchange:
                    self.exchange.cancel_order(stop_id, symbol)
                
                del self.active_stops[symbol]
                results["cancelled"] += 1
                self.logger.info(f"[CANCEL-ALL] Cancelled stop {stop_id} for {symbol}")
                
            except Exception as e:
                results["failed"] += 1
                results["errors"].append(f"Stop {stop_id}: {str(e)}")
                self.logger.error(f"[CANCEL-ALL] Failed to cancel stop {stop_id}: {e}")
        
        # Log summary
        self.logger.critical(
            f"[CANCEL-ALL] Complete: {results['cancelled']}/{results['total_orders']} cancelled, "
            f"{results['failed']} failed"
        )
        
        return results
    
    def handle_disconnect(self, reason: str = "connection_lost"):
        """
        Handler for connection disconnect events.
        
        MUST be called by WebSocket/connection manager when:
        - Connection drops
        - Heartbeat timeout
        - Authentication failure
        - Exchange maintenance
        
        This is a SYNCHRONOUS method for immediate action.
        """
        self.logger.critical(f"[DISCONNECT] Connection lost: {reason}")
        
        # Immediately cancel all orders
        cancel_result = self.cancel_all_open_orders(reason=f"disconnect:{reason}")
        
        # Mark as not ready for new orders
        self._disconnect_active = True
        
        return cancel_result
    
    def handle_reconnect(self):
        """
        Handler for successful reconnection.

        [FIX-RECONNECT-ORDER] Previously re-enabled orders (step 1) BEFORE
        verifying userrefs (step 4). If a new tick arrived during verification,
        orders could be duplicated. Now: orders stay blocked until ALL
        verification completes. _disconnect_active is cleared at the END.

        Sequence:
        1. Reconcile positions with exchange (live mode only)
        2. Query open orders
        3. Verify userrefs (detect fills during disconnect)
        4. THEN re-enable order acceptance
        """
        self.logger.info("[RECONNECT] Starting reconnection sequence (orders BLOCKED until verified)...")

        # [FIX-RECONNECT-ORDER] Do NOT re-enable orders yet — keep blocked
        # self._disconnect_active = False  ← moved to end

        # 1. Reconcile positions (live mode only)
        if not self.dry_run and self.exchange:
            try:
                exchange_positions = self.exchange.fetch_positions()
                local_positions = {
                    oid: order for oid, order in self.pending_orders.items()
                }
                self.logger.info(
                    f"[RECONNECT] Exchange positions: {len(exchange_positions)}, "
                    f"local pending: {len(local_positions)}"
                )
                # Check for fills that happened during disconnect
                if exchange_positions:
                    for pos in exchange_positions:
                        symbol = pos.get('symbol', '')
                        size = abs(pos.get('contracts', 0))
                        if size > 0:
                            self.logger.warning(
                                f"[RECONNECT] Active position: {symbol} "
                                f"size={size} - verify shadow ledger"
                            )
            except Exception as e:
                self.logger.error(
                    f"[RECONNECT] Position reconciliation failed: {e}. "
                    f"Orders re-enabled but positions may be stale."
                )

            # 3. Query open orders
            try:
                open_orders = self.exchange.fetch_open_orders()
                self.logger.info(
                    f"[RECONNECT] Open orders on exchange: {len(open_orders)}"
                )
            except Exception as e:
                self.logger.error(f"[RECONNECT] Open orders query failed: {e}")

            # 4. [C8] Verify recent userrefs - detect orders filled during disconnect
            if self._userref_history:
                verified = 0
                for uref, local_oid in list(self._userref_history.items()):
                    existing = self.check_userref_executed(uref)
                    if existing:
                        verified += 1
                if verified:
                    self.logger.info(
                        f"[RECONNECT][IDEMPOTENT] {verified}/{len(self._userref_history)} "
                        f"recent userrefs confirmed on exchange"
                    )
        else:
            self.logger.info("[RECONNECT] Dry run - skipping position reconciliation")

        # [FIX-RECONNECT-ORDER] NOW re-enable orders — after all verification is done
        self._disconnect_active = False
        self.logger.info("[RECONNECT] Reconnection sequence complete - orders enabled")
    
    def is_ready_for_orders(self) -> bool:
        """Check if execution manager is ready to accept orders."""
        return not getattr(self, '_disconnect_active', False)
    
    def reconcile_positions(self, expected_positions: Dict[str, float]) -> Dict[str, Any]:
        """
        Reconcile local position tracking with exchange.
        
        Args:
            expected_positions: Dict of symbol -> expected size
            
        Returns:
            Reconciliation report
        """
        if self.dry_run:
            return {"status": "dry_run", "discrepancies": []}
        
        try:
            exchange_positions = self.exchange.fetch_positions()
            discrepancies = []
            
            for pos in exchange_positions:
                symbol = pos.get('symbol')
                exchange_size = abs(pos.get('contracts', 0))
                expected_size = expected_positions.get(symbol, 0)
                
                if abs(exchange_size - expected_size) > 0.0001:
                    discrepancies.append({
                        "symbol": symbol,
                        "expected": expected_size,
                        "actual": exchange_size,
                        "difference": exchange_size - expected_size
                    })
            
            return {
                "status": "completed",
                "discrepancies": discrepancies,
                "positions_checked": len(expected_positions)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "discrepancies": []
            }
    
    def get_slippage_stats(self) -> Dict[str, float]:
        """Get slippage statistics."""
        if not self.slippage_history:
            return {"mean": 0, "max": 0, "min": 0, "count": 0}
        
        import numpy as np
        return {
            "mean": float(np.mean(self.slippage_history)),
            "max": float(np.max(self.slippage_history)),
            "min": float(np.min(self.slippage_history)),
            "count": len(self.slippage_history)
        }


if __name__ == "__main__":
    # Test execution manager (dry run)
    class MockExchange:
        pass
    
    em = ExecutionManager(MockExchange(), dry_run=True)
    
    result = em.execute_order(
        symbol="BTC/USDT:USDT",
        side=OrderSide.BUY,
        size=0.1,
        price=45000.0
    )
    
    print(f"Order Result: {result.to_dict()}")
