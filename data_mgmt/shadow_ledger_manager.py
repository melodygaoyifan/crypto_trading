"""
ShadowLedgerManager - Real-Time Local Balance Tracking
======================================================

Financial Fortress implementation for HMATS v3.2 with:

1. Shadow Balance Tracking
   - Local mirror of exchange balances
   - Immediate freeze on order intent (before API confirmation)
   - Prevents over-drafting due to WS latency

2. Precision Manager
   - Strict Decimal quantizing using Kraken's lot_decimals/pair_decimals
   - Prohibits float-based order sizes
   - Prevents "Invalid arguments" errors

3. Dust Awareness
   - Tracks sub-tradeable dust balances
   - Includes in net equity calculation
   - Excludes from executable strategy logic

4. Atomic Operations
   - Thread-safe freeze/unfreeze
   - Shared memory for cross-process visibility
   - Transaction logging for audit

Hardware Target: MSI Titan 18 HX (24-core, 128GB RAM)
Author: Senior Lead Quant Architect & Defensive Systems Engineer
Version: 3.2.0
"""

import os
import time
import json
import logging
import threading
import struct
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN, ROUND_UP, InvalidOperation
from collections import defaultdict
from contextlib import contextmanager
import multiprocessing as mp
from multiprocessing import shared_memory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger('ShadowLedgerManager')


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS & CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Kraken pair specifications (subset - production would fetch from API)
KRAKEN_PAIR_SPECS = {
    'XXBTZUSD': {'base': 'XBT', 'quote': 'USD', 'lot_decimals': 8, 'pair_decimals': 1, 'min_order': Decimal('0.0001')},
    'XETHZUSD': {'base': 'ETH', 'quote': 'USD', 'lot_decimals': 8, 'pair_decimals': 2, 'min_order': Decimal('0.001')},
    'SOLUSD': {'base': 'SOL', 'quote': 'USD', 'lot_decimals': 8, 'pair_decimals': 4, 'min_order': Decimal('0.01')},
    'BTC/USD': {'base': 'XBT', 'quote': 'USD', 'lot_decimals': 8, 'pair_decimals': 1, 'min_order': Decimal('0.0001')},
    'ETH/USD': {'base': 'ETH', 'quote': 'USD', 'lot_decimals': 8, 'pair_decimals': 2, 'min_order': Decimal('0.001')},
    'SOL/USD': {'base': 'SOL', 'quote': 'USD', 'lot_decimals': 8, 'pair_decimals': 4, 'min_order': Decimal('0.01')},
    # [P133 2026-04-29] SOL/USDT added — primary spot pair for SOL going
    # forward (USD pair dead on Kraken). Legacy SOL/USD kept for any
    # in-flight positions opened on the old pair. min_order=0.02 per
    # Kraken's market constraints (verified via exchange.market()).
    'SOL/USDT': {'base': 'SOL', 'quote': 'USDT', 'lot_decimals': 8, 'pair_decimals': 4, 'min_order': Decimal('0.02')},
    'SOLUSDT': {'base': 'SOL', 'quote': 'USDT', 'lot_decimals': 8, 'pair_decimals': 4, 'min_order': Decimal('0.02')},
}

# Dust thresholds (USD equivalent)
DUST_THRESHOLD_USD = Decimal('1.00')

# Freeze timeout (auto-unfreeze if order not confirmed)
FREEZE_TIMEOUT_S = 60.0


# ═══════════════════════════════════════════════════════════════════════════════
# PRECISION MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class PrecisionManager:
    """
    High-precision Decimal handler for Kraken trading.
    
    Ensures all order sizes and prices conform to Kraken's precision requirements.
    Prohibits float-based calculations to prevent rounding errors.
    """
    
    def __init__(self, pair_specs: Dict[str, Dict] = None):
        self.pair_specs = pair_specs or KRAKEN_PAIR_SPECS
        
        logger.info(f"PrecisionManager initialized with {len(self.pair_specs)} pair specs")
    
    def get_pair_spec(self, pair: str) -> Dict:
        """Get pair specification."""
        # Normalize pair name
        normalized = self._normalize_pair(pair)
        
        if normalized not in self.pair_specs:
            raise ValueError(f"Unknown pair: {pair}")
        
        return self.pair_specs[normalized]
    
    def _normalize_pair(self, pair: str) -> str:
        """Normalize pair name to match specs."""
        # Try direct match first
        if pair in self.pair_specs:
            return pair
        
        # Try without slash
        no_slash = pair.replace('/', '')
        if no_slash in self.pair_specs:
            return no_slash
        
        # Try with slash
        for key in self.pair_specs:
            if key.replace('/', '') == no_slash:
                return key
        
        return pair
    
    def quantize_size(self, pair: str, size: Decimal) -> Decimal:
        """
        Quantize order size to pair's lot_decimals.
        
        Always rounds DOWN to avoid exceeding available balance.
        """
        if isinstance(size, float):
            raise TypeError("Float sizes are prohibited. Use Decimal.")
        
        spec = self.get_pair_spec(pair)
        lot_decimals = spec['lot_decimals']
        
        # Create quantize format
        quantize_str = '1.' + '0' * lot_decimals
        quantizer = Decimal(quantize_str)
        
        return size.quantize(quantizer, rounding=ROUND_DOWN)
    
    def quantize_price(self, pair: str, price: Decimal) -> Decimal:
        """
        Quantize price to pair's pair_decimals.
        
        Rounds DOWN for buy orders, UP for sell orders (caller decides).
        """
        if isinstance(price, float):
            raise TypeError("Float prices are prohibited. Use Decimal.")
        
        spec = self.get_pair_spec(pair)
        pair_decimals = spec['pair_decimals']
        
        quantize_str = '1.' + '0' * pair_decimals
        quantizer = Decimal(quantize_str)
        
        return price.quantize(quantizer, rounding=ROUND_DOWN)
    
    def get_min_order_size(self, pair: str) -> Decimal:
        """Get minimum order size for pair."""
        spec = self.get_pair_spec(pair)
        return spec['min_order']
    
    def validate_order(self, pair: str, size: Decimal, price: Decimal) -> Tuple[bool, str]:
        """
        Validate order parameters.
        
        Returns: (is_valid, error_message)
        """
        try:
            spec = self.get_pair_spec(pair)
            
            # Check types
            if isinstance(size, float) or isinstance(price, float):
                return False, "Float values prohibited"
            
            # Check minimum size
            if size < spec['min_order']:
                return False, f"Size {size} below minimum {spec['min_order']}"
            
            # Check precision
            quantized_size = self.quantize_size(pair, size)
            if quantized_size != size:
                return False, f"Size precision exceeds lot_decimals ({spec['lot_decimals']})"
            
            quantized_price = self.quantize_price(pair, price)
            if quantized_price != price:
                return False, f"Price precision exceeds pair_decimals ({spec['pair_decimals']})"
            
            return True, ""
            
        except ValueError as e:
            return False, str(e)
    
    def safe_decimal(self, value: Any) -> Decimal:
        """
        Convert value to Decimal safely.
        
        Raises TypeError for float inputs.
        """
        if isinstance(value, float):
            raise TypeError(f"Float conversion prohibited. Got: {value}")
        
        if isinstance(value, Decimal):
            return value
        
        return Decimal(str(value))


# ═══════════════════════════════════════════════════════════════════════════════
# BALANCE DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

class FreezeReason(Enum):
    """Reason for balance freeze."""
    ORDER_INTENT = auto()       # Pending order submission
    ORDER_PENDING = auto()      # Order submitted, awaiting fill
    WITHDRAWAL = auto()         # Pending withdrawal
    MARGIN_REQUIREMENT = auto() # Margin collateral
    MANUAL = auto()             # Manual hold


@dataclass
class BalanceFreeze:
    """Record of a balance freeze."""
    freeze_id: str
    asset: str
    amount: Decimal
    reason: FreezeReason
    order_id: Optional[str]
    created_at: float
    expires_at: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AssetBalance:
    """Balance for a single asset."""
    asset: str
    total: Decimal = Decimal('0')
    available: Decimal = Decimal('0')
    frozen: Decimal = Decimal('0')
    dust: Decimal = Decimal('0')
    last_update: float = 0.0
    
    # Active freezes
    freezes: Dict[str, BalanceFreeze] = field(default_factory=dict)
    
    def recalculate(self):
        """Recalculate available balance from total and freezes."""
        self.frozen = sum(f.amount for f in self.freezes.values())
        self.available = self.total - self.frozen - self.dust


@dataclass
class LedgerTransaction:
    """Transaction record for audit trail."""
    tx_id: str
    timestamp: float
    tx_type: str  # 'freeze', 'unfreeze', 'update', 'trade'
    asset: str
    amount: Decimal
    balance_before: Decimal
    balance_after: Decimal
    order_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# SHADOW LEDGER MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class ShadowLedgerManager:
    """
    Real-time local balance tracking with atomic freezing/unfreezing.
    
    The "Financial Fortress" - ensures capital preservation by:
    1. Freezing funds on order intent (before API confirmation)
    2. Preventing over-drafting due to WS latency
    3. Tracking dust balances separately
    4. Maintaining complete audit trail
    """
    
    def __init__(
        self,
        precision_manager: PrecisionManager = None,
        freeze_timeout_s: float = FREEZE_TIMEOUT_S,
        dust_threshold_usd: Decimal = DUST_THRESHOLD_USD
    ):
        self.precision = precision_manager or PrecisionManager()
        self.freeze_timeout = freeze_timeout_s
        self.dust_threshold = dust_threshold_usd
        
        # Balances per asset
        self._balances: Dict[str, AssetBalance] = {}
        
        # Transaction log (ring buffer for memory efficiency)
        self._transactions: List[LedgerTransaction] = []
        self._max_transactions = 10000
        
        # Freeze counter for unique IDs
        self._freeze_counter = 0
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Callbacks
        self._on_balance_change: Optional[callable] = None
        self._on_freeze_expire: Optional[callable] = None
        
        # Cleanup thread
        self._cleanup_running = False
        self._cleanup_thread: Optional[threading.Thread] = None
        
        logger.info("ShadowLedgerManager initialized")
    
    # ───────────────────────────────────────────────────────────────────────────
    # BALANCE OPERATIONS
    # ───────────────────────────────────────────────────────────────────────────
    
    def update_balance(
        self,
        asset: str,
        total: Decimal,
        source: str = "exchange"
    ) -> AssetBalance:
        """
        Update total balance from exchange sync.
        
        Args:
            asset: Asset symbol (e.g., 'XBT', 'ETH', 'USD')
            total: Total balance from exchange
            source: Source of update for logging
            
        Returns:
            Updated AssetBalance
        """
        total = self.precision.safe_decimal(total)
        
        with self._lock:
            if asset not in self._balances:
                self._balances[asset] = AssetBalance(asset=asset)
            
            balance = self._balances[asset]
            old_total = balance.total
            
            balance.total = total
            balance.last_update = time.time()
            balance.recalculate()
            
            # Log transaction
            self._log_transaction(
                tx_type='update',
                asset=asset,
                amount=total - old_total,
                balance_before=old_total,
                balance_after=total,
                metadata={'source': source}
            )
            
            # Notify callback
            if self._on_balance_change:
                self._on_balance_change(asset, balance)
            
            logger.debug(f"Balance updated: {asset} = {total} (available={balance.available})")
            return balance
    
    def get_balance(self, asset: str) -> Optional[AssetBalance]:
        """Get current balance for asset."""
        with self._lock:
            return self._balances.get(asset)
    
    def get_available(self, asset: str) -> Decimal:
        """Get available (unfrozen) balance for asset."""
        with self._lock:
            balance = self._balances.get(asset)
            if balance:
                return balance.available
            return Decimal('0')
    
    def get_all_balances(self) -> Dict[str, AssetBalance]:
        """Get all balances."""
        with self._lock:
            return dict(self._balances)
    
    # ───────────────────────────────────────────────────────────────────────────
    # FREEZE OPERATIONS
    # ───────────────────────────────────────────────────────────────────────────
    
    def freeze(
        self,
        asset: str,
        amount: Decimal,
        reason: FreezeReason = FreezeReason.ORDER_INTENT,
        order_id: str = None,
        timeout_s: float = None
    ) -> Optional[str]:
        """
        Freeze balance for pending operation.
        
        Called BEFORE submitting order to exchange to prevent over-drafting.
        
        Args:
            asset: Asset to freeze
            amount: Amount to freeze
            reason: Reason for freeze
            order_id: Associated order ID
            timeout_s: Custom timeout (default: self.freeze_timeout)
            
        Returns:
            freeze_id if successful, None if insufficient balance
        """
        amount = self.precision.safe_decimal(amount)
        timeout = timeout_s or self.freeze_timeout
        
        with self._lock:
            balance = self._balances.get(asset)
            if not balance:
                logger.warning(f"Cannot freeze: no balance for {asset}")
                return None
            
            if amount > balance.available:
                logger.warning(
                    f"Cannot freeze {amount} {asset}: only {balance.available} available"
                )
                return None
            
            # Generate freeze ID
            self._freeze_counter += 1
            freeze_id = f"FRZ_{asset}_{self._freeze_counter}_{int(time.time()*1000)}"
            
            # Create freeze record
            freeze = BalanceFreeze(
                freeze_id=freeze_id,
                asset=asset,
                amount=amount,
                reason=reason,
                order_id=order_id,
                created_at=time.time(),
                expires_at=time.time() + timeout
            )
            
            # Apply freeze
            balance.freezes[freeze_id] = freeze
            balance.recalculate()
            
            # Log transaction
            self._log_transaction(
                tx_type='freeze',
                asset=asset,
                amount=amount,
                balance_before=balance.available + amount,
                balance_after=balance.available,
                order_id=order_id,
                metadata={'freeze_id': freeze_id, 'reason': reason.name}
            )
            
            logger.info(
                f"Frozen {amount} {asset} (id={freeze_id}, reason={reason.name}, "
                f"available={balance.available})"
            )
            
            return freeze_id
    
    def unfreeze(self, freeze_id: str) -> bool:
        """
        Unfreeze a specific freeze.
        
        Called when order is filled, cancelled, or rejected.
        
        Args:
            freeze_id: ID from freeze() call
            
        Returns:
            True if successfully unfrozen
        """
        with self._lock:
            for asset, balance in self._balances.items():
                if freeze_id in balance.freezes:
                    freeze = balance.freezes.pop(freeze_id)
                    old_available = balance.available
                    balance.recalculate()
                    
                    # Log transaction
                    self._log_transaction(
                        tx_type='unfreeze',
                        asset=asset,
                        amount=freeze.amount,
                        balance_before=old_available,
                        balance_after=balance.available,
                        order_id=freeze.order_id,
                        metadata={'freeze_id': freeze_id}
                    )
                    
                    logger.info(
                        f"Unfrozen {freeze.amount} {asset} (id={freeze_id}, "
                        f"available={balance.available})"
                    )
                    return True
            
            logger.warning(f"Freeze not found: {freeze_id}")
            return False
    
    def unfreeze_by_order(self, order_id: str) -> int:
        """
        Unfreeze all freezes associated with an order.
        
        Args:
            order_id: Order ID to unfreeze
            
        Returns:
            Number of freezes unfrozen
        """
        unfrozen_count = 0
        
        with self._lock:
            for asset, balance in self._balances.items():
                to_remove = [
                    fid for fid, freeze in balance.freezes.items()
                    if freeze.order_id == order_id
                ]
                
                for freeze_id in to_remove:
                    freeze = balance.freezes.pop(freeze_id)
                    unfrozen_count += 1
                    
                    self._log_transaction(
                        tx_type='unfreeze',
                        asset=asset,
                        amount=freeze.amount,
                        balance_before=balance.available,
                        balance_after=balance.available + freeze.amount,
                        order_id=order_id,
                        metadata={'freeze_id': freeze_id, 'reason': 'order_complete'}
                    )
                
                balance.recalculate()
        
        if unfrozen_count > 0:
            logger.info(f"Unfrozen {unfrozen_count} freeze(s) for order {order_id}")
        
        return unfrozen_count
    
    @contextmanager
    def atomic_freeze(
        self,
        asset: str,
        amount: Decimal,
        reason: FreezeReason = FreezeReason.ORDER_INTENT
    ):
        """
        Context manager for atomic freeze/unfreeze.
        
        Usage:
            with ledger.atomic_freeze('USD', Decimal('1000')) as freeze_id:
                # Submit order...
                if order_failed:
                    raise Exception("Order failed")
            # Automatically unfrozen on exit or exception
        """
        freeze_id = self.freeze(asset, amount, reason)
        if freeze_id is None:
            raise ValueError(f"Insufficient balance to freeze {amount} {asset}")
        
        try:
            yield freeze_id
        finally:
            self.unfreeze(freeze_id)
    
    # ───────────────────────────────────────────────────────────────────────────
    # DUST MANAGEMENT
    # ───────────────────────────────────────────────────────────────────────────
    
    def update_dust(self, asset: str, dust_amount: Decimal, price_usd: Decimal = None):
        """
        Update dust balance for an asset.
        
        Dust = balance that is below minimum tradeable amount.
        Included in equity but excluded from trading logic.
        """
        dust_amount = self.precision.safe_decimal(dust_amount)
        
        with self._lock:
            if asset not in self._balances:
                self._balances[asset] = AssetBalance(asset=asset)
            
            balance = self._balances[asset]
            balance.dust = dust_amount
            balance.recalculate()
            
            logger.debug(f"Dust updated: {asset} = {dust_amount}")
    
    def calculate_dust(self, asset: str, min_order_size: Decimal) -> Decimal:
        """
        Calculate dust portion of balance.
        
        Returns amount below minimum order size.
        """
        with self._lock:
            balance = self._balances.get(asset)
            if not balance:
                return Decimal('0')
            
            remainder = balance.available % min_order_size
            return remainder if remainder > 0 else Decimal('0')
    
    # ───────────────────────────────────────────────────────────────────────────
    # EQUITY CALCULATIONS
    # ───────────────────────────────────────────────────────────────────────────
    
    def get_net_equity_usd(self, prices: Dict[str, Decimal]) -> Decimal:
        """
        Calculate total net equity in USD.
        
        Includes dust balances for accurate accounting.
        
        Args:
            prices: Current prices in USD (e.g., {'XBT': Decimal('97000')})
            
        Returns:
            Total equity in USD
        """
        total_usd = Decimal('0')
        
        with self._lock:
            for asset, balance in self._balances.items():
                if asset in ['USD', 'ZUSD']:
                    total_usd += balance.total
                elif asset in prices:
                    total_usd += balance.total * prices[asset]
        
        return total_usd
    
    def get_tradeable_equity_usd(self, prices: Dict[str, Decimal]) -> Decimal:
        """
        Calculate tradeable equity (excludes frozen and dust).
        
        Args:
            prices: Current prices in USD
            
        Returns:
            Tradeable equity in USD
        """
        total_usd = Decimal('0')
        
        with self._lock:
            for asset, balance in self._balances.items():
                if asset in ['USD', 'ZUSD']:
                    total_usd += balance.available
                elif asset in prices:
                    total_usd += balance.available * prices[asset]
        
        return total_usd
    
    # ───────────────────────────────────────────────────────────────────────────
    # VALIDATION
    # ───────────────────────────────────────────────────────────────────────────
    
    def can_trade(
        self,
        pair: str,
        side: str,
        size: Decimal,
        price: Decimal
    ) -> Tuple[bool, str]:
        """
        Check if trade is possible with current available balance.
        
        Args:
            pair: Trading pair (e.g., 'BTC/USD')
            side: 'buy' or 'sell'
            size: Order size
            price: Order price
            
        Returns:
            (can_trade, reason)
        """
        try:
            spec = self.precision.get_pair_spec(pair)
            base_asset = spec['base']
            quote_asset = spec['quote']
            
            # Validate order parameters
            valid, error = self.precision.validate_order(pair, size, price)
            if not valid:
                return False, error
            
            with self._lock:
                if side.lower() == 'buy':
                    # Need quote currency
                    required = size * price
                    available = self.get_available(quote_asset)
                    
                    if required > available:
                        return False, f"Insufficient {quote_asset}: need {required}, have {available}"
                        
                else:  # sell
                    # Need base currency
                    available = self.get_available(base_asset)
                    
                    if size > available:
                        return False, f"Insufficient {base_asset}: need {size}, have {available}"
            
            return True, ""
            
        except Exception as e:
            return False, str(e)
    
    # ───────────────────────────────────────────────────────────────────────────
    # TRANSACTION LOGGING
    # ───────────────────────────────────────────────────────────────────────────
    
    def _log_transaction(
        self,
        tx_type: str,
        asset: str,
        amount: Decimal,
        balance_before: Decimal,
        balance_after: Decimal,
        order_id: str = None,
        metadata: Dict = None
    ):
        """Log a transaction for audit trail."""
        tx = LedgerTransaction(
            tx_id=f"TX_{int(time.time()*1000000)}",
            timestamp=time.time(),
            tx_type=tx_type,
            asset=asset,
            amount=amount,
            balance_before=balance_before,
            balance_after=balance_after,
            order_id=order_id,
            metadata=metadata or {}
        )
        
        self._transactions.append(tx)
        
        # Trim if exceeds max
        if len(self._transactions) > self._max_transactions:
            self._transactions = self._transactions[-self._max_transactions:]
    
    def get_transactions(
        self,
        asset: str = None,
        tx_type: str = None,
        since: float = None,
        limit: int = 100
    ) -> List[LedgerTransaction]:
        """Get transaction history with filters."""
        with self._lock:
            filtered = self._transactions
            
            if asset:
                filtered = [t for t in filtered if t.asset == asset]
            
            if tx_type:
                filtered = [t for t in filtered if t.tx_type == tx_type]
            
            if since:
                filtered = [t for t in filtered if t.timestamp >= since]
            
            return filtered[-limit:]
    
    # ───────────────────────────────────────────────────────────────────────────
    # CLEANUP
    # ───────────────────────────────────────────────────────────────────────────
    
    def start_cleanup_thread(self, interval_s: float = 10.0):
        """Start background thread to cleanup expired freezes."""
        if self._cleanup_running:
            return
        
        self._cleanup_running = True
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            args=(interval_s,),
            daemon=True
        )
        self._cleanup_thread.start()
        logger.info("Cleanup thread started")
    
    def stop_cleanup_thread(self):
        """Stop cleanup thread."""
        self._cleanup_running = False
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=2)
        logger.info("Cleanup thread stopped")
    
    def _cleanup_loop(self, interval_s: float):
        """Cleanup expired freezes periodically."""
        while self._cleanup_running:
            try:
                self._cleanup_expired_freezes()
                time.sleep(interval_s)
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
    
    def _cleanup_expired_freezes(self):
        """Remove expired freezes."""
        now = time.time()
        
        with self._lock:
            for asset, balance in self._balances.items():
                expired = [
                    fid for fid, freeze in balance.freezes.items()
                    if freeze.expires_at < now
                ]
                
                for freeze_id in expired:
                    freeze = balance.freezes.pop(freeze_id)
                    
                    logger.warning(
                        f"Freeze expired: {freeze_id} ({freeze.amount} {asset})"
                    )
                    
                    self._log_transaction(
                        tx_type='unfreeze',
                        asset=asset,
                        amount=freeze.amount,
                        balance_before=balance.available,
                        balance_after=balance.available + freeze.amount,
                        order_id=freeze.order_id,
                        metadata={'freeze_id': freeze_id, 'reason': 'expired'}
                    )
                    
                    # Notify callback
                    if self._on_freeze_expire:
                        self._on_freeze_expire(freeze)
                
                balance.recalculate()
    
    # ───────────────────────────────────────────────────────────────────────────
    # CALLBACKS
    # ───────────────────────────────────────────────────────────────────────────
    
    def on_balance_change(self, callback: callable):
        """Register balance change callback."""
        self._on_balance_change = callback
    
    def on_freeze_expire(self, callback: callable):
        """Register freeze expiration callback."""
        self._on_freeze_expire = callback
    
    # ───────────────────────────────────────────────────────────────────────────
    # SERIALIZATION
    # ───────────────────────────────────────────────────────────────────────────
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize ledger state to dictionary."""
        with self._lock:
            return {
                'balances': {
                    asset: {
                        'total': str(bal.total),
                        'available': str(bal.available),
                        'frozen': str(bal.frozen),
                        'dust': str(bal.dust),
                        'last_update': bal.last_update,
                        'freezes': {
                            fid: {
                                'amount': str(f.amount),
                                'reason': f.reason.name,
                                'order_id': f.order_id,
                                'created_at': f.created_at,
                                'expires_at': f.expires_at
                            }
                            for fid, f in bal.freezes.items()
                        }
                    }
                    for asset, bal in self._balances.items()
                },
                'transaction_count': len(self._transactions),
                'freeze_counter': self._freeze_counter
            }
    
    def from_dict(self, data: Dict[str, Any]):
        """Restore ledger state from dictionary."""
        with self._lock:
            self._freeze_counter = data.get('freeze_counter', 0)
            
            for asset, bal_data in data.get('balances', {}).items():
                balance = AssetBalance(asset=asset)
                balance.total = Decimal(bal_data['total'])
                balance.dust = Decimal(bal_data.get('dust', '0'))
                balance.last_update = bal_data.get('last_update', 0)
                
                for fid, freeze_data in bal_data.get('freezes', {}).items():
                    freeze = BalanceFreeze(
                        freeze_id=fid,
                        asset=asset,
                        amount=Decimal(freeze_data['amount']),
                        reason=FreezeReason[freeze_data['reason']],
                        order_id=freeze_data.get('order_id'),
                        created_at=freeze_data.get('created_at', 0),
                        expires_at=freeze_data.get('expires_at', 0)
                    )
                    balance.freezes[fid] = freeze
                
                balance.recalculate()
                self._balances[asset] = balance
    
    # ───────────────────────────────────────────────────────────────────────────
    # v6.2.0: GOVERNOR VETO LOGGING
    # ───────────────────────────────────────────────────────────────────────────
    
    def log_veto(
        self,
        source: str,
        reason: str,
        details: Dict[str, Any] = None,
    ):
        """
        Log a governor veto event.
        
        Used by:
        - OpportunityBudgetGovernor
        - RegimeTransitionBuffer
        - CascadeExhaustionGovernor
        
        Args:
            source: Veto source (e.g., "OpportunityBudgetGovernor")
            reason: Veto reason code (e.g., "OPPORTUNITY_BUDGET_EXCEEDED")
            details: Additional details about the veto
        """
        veto_record = {
            'timestamp': time.time(),
            'source': source,
            'reason': reason,
            'details': details or {},
        }
        
        # Log as special transaction
        self._log_transaction(
            tx_type='veto',
            asset='SYSTEM',
            amount=Decimal('0'),
            balance_before=Decimal('0'),
            balance_after=Decimal('0'),
            metadata=veto_record,
        )
        
        logger.warning(f"[VETO] {source}: {reason}")


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON INSTANCE (v6.2.0)
# ═══════════════════════════════════════════════════════════════════════════════

_shadow_ledger_instance: Optional[ShadowLedgerManager] = None
_shadow_ledger_lock = threading.Lock()


def get_shadow_ledger() -> ShadowLedgerManager:
    """
    Get singleton ShadowLedgerManager instance.
    
    Used by governors for veto logging:
    - OpportunityBudgetGovernor
    - RegimeTransitionBuffer
    - CascadeExhaustionGovernor
    """
    global _shadow_ledger_instance
    if _shadow_ledger_instance is None:
        with _shadow_ledger_lock:
            if _shadow_ledger_instance is None:
                _shadow_ledger_instance = ShadowLedgerManager()
                logger.info("[ShadowLedger] Singleton instance created")
    return _shadow_ledger_instance


def reset_shadow_ledger():
    """Reset singleton instance (for testing)."""
    global _shadow_ledger_instance
    with _shadow_ledger_lock:
        _shadow_ledger_instance = None


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED MEMORY LEDGER STATE
# ═══════════════════════════════════════════════════════════════════════════════

class SharedLedgerState:
    """
    Shared memory representation of critical ledger state.
    
    For cross-process visibility on 24-core system.
    """
    
    # Layout: [asset_count(4)] + [per_asset: name(8) + total(8) + available(8) + frozen(8)]
    ASSET_SIZE = 32  # bytes per asset
    MAX_ASSETS = 10
    BUFFER_SIZE = 4 + (ASSET_SIZE * MAX_ASSETS)
    
    def __init__(self, name: str = "shadow_ledger_state", create: bool = True):
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
                
            logger.info(f"SharedLedgerState initialized: {name}")
            
        except FileExistsError:
            self._shm = shared_memory.SharedMemory(name=name)
    
    def _initialize(self):
        """Initialize with zeros."""
        for i in range(self.BUFFER_SIZE):
            self._shm.buf[i] = 0
    
    def update_balance(self, asset: str, total: float, available: float, frozen: float):
        """Update balance in shared memory."""
        if not self._shm:
            return
        
        # Find or create slot for asset
        asset_bytes = asset.encode('utf-8')[:8].ljust(8, b'\x00')
        
        # Simple linear search (fine for small number of assets)
        count = struct.unpack('<I', bytes(self._shm.buf[0:4]))[0]
        
        slot_found = False
        for i in range(min(count, self.MAX_ASSETS)):
            offset = 4 + (i * self.ASSET_SIZE)
            existing = bytes(self._shm.buf[offset:offset+8])
            
            if existing == asset_bytes:
                # Update existing
                data = struct.pack('<8sddd', asset_bytes, total, available, frozen)
                for j, b in enumerate(data):
                    self._shm.buf[offset + j] = b
                slot_found = True
                break
        
        if not slot_found and count < self.MAX_ASSETS:
            # Add new
            offset = 4 + (count * self.ASSET_SIZE)
            data = struct.pack('<8sddd', asset_bytes, total, available, frozen)
            for j, b in enumerate(data):
                self._shm.buf[offset + j] = b
            
            # Update count
            new_count = struct.pack('<I', count + 1)
            for j, b in enumerate(new_count):
                self._shm.buf[j] = b
    
    def get_balance(self, asset: str) -> Optional[Tuple[float, float, float]]:
        """Get balance from shared memory."""
        if not self._shm:
            return None
        
        asset_bytes = asset.encode('utf-8')[:8].ljust(8, b'\x00')
        count = struct.unpack('<I', bytes(self._shm.buf[0:4]))[0]
        
        for i in range(min(count, self.MAX_ASSETS)):
            offset = 4 + (i * self.ASSET_SIZE)
            existing = bytes(self._shm.buf[offset:offset+8])
            
            if existing == asset_bytes:
                data = struct.unpack('<8sddd', bytes(self._shm.buf[offset:offset+self.ASSET_SIZE]))
                return (data[1], data[2], data[3])  # total, available, frozen
        
        return None
    
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
    # Main classes
    'ShadowLedgerManager',
    'PrecisionManager',
    
    # Data structures
    'AssetBalance',
    'BalanceFreeze',
    'FreezeReason',
    'LedgerTransaction',
    
    # Shared state
    'SharedLedgerState',
    
    # Constants
    'KRAKEN_PAIR_SPECS',
]


# ═══════════════════════════════════════════════════════════════════════════════
# TESTING
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("ShadowLedgerManager Test Suite")
    print("=" * 70)
    
    # Test 1: Precision Manager
    print("\n[1] Precision Manager")
    print("-" * 50)
    
    precision = PrecisionManager()
    
    # Test quantization
    size = Decimal('0.123456789')
    quantized = precision.quantize_size('BTC/USD', size)
    print(f"Quantize size {size} -> {quantized}")
    
    price = Decimal('97123.456')
    quantized_price = precision.quantize_price('BTC/USD', price)
    print(f"Quantize price {price} -> {quantized_price}")
    
    # Test validation
    valid, error = precision.validate_order('BTC/USD', Decimal('0.001'), Decimal('97000.0'))
    print(f"Validate order: valid={valid}, error='{error}'")
    
    # Test float rejection
    try:
        precision.safe_decimal(0.001)
    except TypeError as e:
        print(f"Float rejection: {e}")
    
    # Test 2: Shadow Ledger
    print("\n[2] Shadow Ledger Balance Updates")
    print("-" * 50)
    
    ledger = ShadowLedgerManager(precision_manager=precision)
    
    # Update balances
    ledger.update_balance('USD', Decimal('50000.00'))
    ledger.update_balance('XBT', Decimal('0.5'))
    ledger.update_balance('ETH', Decimal('5.0'))
    
    for asset in ['USD', 'XBT', 'ETH']:
        bal = ledger.get_balance(asset)
        print(f"{asset}: total={bal.total}, available={bal.available}")
    
    # Test 3: Freeze/Unfreeze
    print("\n[3] Freeze/Unfreeze Operations")
    print("-" * 50)
    
    # Freeze USD for order
    freeze_id = ledger.freeze('USD', Decimal('10000.00'), FreezeReason.ORDER_INTENT, order_id='ORD-001')
    print(f"Frozen $10,000: freeze_id={freeze_id}")
    
    bal = ledger.get_balance('USD')
    print(f"USD after freeze: total={bal.total}, available={bal.available}, frozen={bal.frozen}")
    
    # Unfreeze
    ledger.unfreeze(freeze_id)
    bal = ledger.get_balance('USD')
    print(f"USD after unfreeze: total={bal.total}, available={bal.available}")
    
    # Test 4: Atomic Freeze Context Manager
    print("\n[4] Atomic Freeze Context Manager")
    print("-" * 50)
    
    try:
        with ledger.atomic_freeze('USD', Decimal('5000.00')) as fid:
            bal = ledger.get_balance('USD')
            print(f"Inside context: available={bal.available}")
            # Simulating order submission...
        
        bal = ledger.get_balance('USD')
        print(f"After context: available={bal.available}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 5: Trade Validation
    print("\n[5] Trade Validation")
    print("-" * 50)
    
    can_trade, reason = ledger.can_trade('BTC/USD', 'buy', Decimal('0.01'), Decimal('97000.0'))
    print(f"Can buy 0.01 BTC @ $97000: {can_trade} - {reason}")
    
    can_trade, reason = ledger.can_trade('BTC/USD', 'buy', Decimal('1.0'), Decimal('97000.0'))
    print(f"Can buy 1.0 BTC @ $97000: {can_trade} - {reason}")
    
    can_trade, reason = ledger.can_trade('BTC/USD', 'sell', Decimal('0.1'), Decimal('97000.0'))
    print(f"Can sell 0.1 BTC: {can_trade} - {reason}")
    
    # Test 6: Equity Calculation
    print("\n[6] Equity Calculation")
    print("-" * 50)
    
    prices = {
        'XBT': Decimal('97000'),
        'ETH': Decimal('3200')
    }
    
    net_equity = ledger.get_net_equity_usd(prices)
    tradeable_equity = ledger.get_tradeable_equity_usd(prices)
    
    print(f"Net equity: ${net_equity:,.2f}")
    print(f"Tradeable equity: ${tradeable_equity:,.2f}")
    
    # Test 7: Transaction History
    print("\n[7] Transaction History")
    print("-" * 50)
    
    transactions = ledger.get_transactions(limit=5)
    for tx in transactions:
        print(f"  {tx.tx_type}: {tx.amount} {tx.asset} ({tx.balance_before} -> {tx.balance_after})")
    
    # Test 8: Serialization
    print("\n[8] Serialization")
    print("-" * 50)
    
    state = ledger.to_dict()
    print(f"Serialized state: {len(state['balances'])} assets, {state['transaction_count']} transactions")
    
    # Test 9: Shared Memory
    print("\n[9] Shared Ledger State")
    print("-" * 50)
    
    try:
        shared = SharedLedgerState(name="test_ledger_state", create=True)
        shared.update_balance('USD', 50000.0, 45000.0, 5000.0)
        shared.update_balance('XBT', 0.5, 0.5, 0.0)
        
        usd_bal = shared.get_balance('USD')
        print(f"Shared USD: total={usd_bal[0]}, available={usd_bal[1]}, frozen={usd_bal[2]}")
        
        shared.unlink()
        print("Shared memory cleaned up")
    except Exception as e:
        print(f"Shared memory test skipped: {e}")
    
    print("\n" + "=" * 70)
    print("✓ All ShadowLedgerManager tests completed!")
    print("=" * 70)
