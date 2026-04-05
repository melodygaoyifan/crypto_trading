"""
================================================================================
SHADOW LEDGER - Local Balance Tracking & Position Management
================================================================================
Version: 3.2.0
Purpose: Maintain authoritative local state for balance and position tracking

Features:
- Decimal precision management (Kraken lot sizes)
- Local balance freezing for pending orders
- Dust balance identification and isolation
- Position reconciliation with exchange
- Double-entry bookkeeping for audit trail
================================================================================
"""

import time
import logging
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from collections import defaultdict
import threading
import json

logger = logging.getLogger("HMATS.ShadowLedger")


class BalanceType(Enum):
    """Balance classification."""
    AVAILABLE = auto()   # Free to trade
    FROZEN = auto()      # Reserved for pending orders
    DUST = auto()        # Below minimum tradeable
    TOTAL = auto()       # Sum of all


class EntryType(Enum):
    """Ledger entry type."""
    CREDIT = auto()      # Increase balance
    DEBIT = auto()       # Decrease balance
    FREEZE = auto()      # Move to frozen
    UNFREEZE = auto()    # Release from frozen
    DUST_OUT = auto()    # Move to dust
    RECONCILE = auto()   # Exchange sync adjustment


@dataclass
class AssetPrecision:
    """Asset precision configuration."""
    asset: str
    lot_size: Decimal           # Minimum order size
    lot_decimals: int           # Decimal places for quantity
    price_decimals: int         # Decimal places for price
    min_order_value: Decimal    # Minimum order value in USD
    dust_threshold: Decimal     # Below this = dust
    
    @classmethod
    def kraken_defaults(cls) -> Dict[str, 'AssetPrecision']:
        """Get Kraken default precision settings."""
        return {
            'XBT': cls(
                asset='XBT',
                lot_size=Decimal('0.0001'),
                lot_decimals=8,
                price_decimals=1,
                min_order_value=Decimal('5.00'),
                dust_threshold=Decimal('0.00001')
            ),
            'ETH': cls(
                asset='ETH',
                lot_size=Decimal('0.001'),
                lot_decimals=8,
                price_decimals=2,
                min_order_value=Decimal('5.00'),
                dust_threshold=Decimal('0.0001')
            ),
            'SOL': cls(
                asset='SOL',
                lot_size=Decimal('0.01'),
                lot_decimals=8,
                price_decimals=4,
                min_order_value=Decimal('5.00'),
                dust_threshold=Decimal('0.001')
            ),
            'USD': cls(
                asset='USD',
                lot_size=Decimal('0.01'),
                lot_decimals=4,
                price_decimals=4,
                min_order_value=Decimal('5.00'),
                dust_threshold=Decimal('0.01')
            )
        }


@dataclass
class LedgerEntry:
    """Single ledger entry for audit trail."""
    entry_id: str
    timestamp: float
    asset: str
    entry_type: EntryType
    amount: Decimal
    balance_before: Decimal
    balance_after: Decimal
    reference: str = ""       # Order ID, trade ID, etc.
    notes: str = ""


@dataclass
class AssetBalance:
    """Balance for a single asset."""
    asset: str
    available: Decimal = Decimal('0')
    frozen: Decimal = Decimal('0')
    dust: Decimal = Decimal('0')
    
    @property
    def total(self) -> Decimal:
        return self.available + self.frozen + self.dust
    
    def to_dict(self) -> Dict:
        return {
            'asset': self.asset,
            'available': str(self.available),
            'frozen': str(self.frozen),
            'dust': str(self.dust),
            'total': str(self.total)
        }


@dataclass
class FrozenAllocation:
    """Frozen balance allocation for a pending order."""
    order_id: str
    asset: str
    amount: Decimal
    timestamp: float
    expires_at: float = 0.0   # 0 = no expiry


class ShadowLedger:
    """
    Local balance and position tracker.
    
    Maintains authoritative view of account state independent of
    exchange API, with full audit trail.
    """
    
    FREEZE_EXPIRY_SECONDS = 300  # 5 minutes default
    
    def __init__(self, precision_config: Dict[str, AssetPrecision] = None):
        self.precision = precision_config or AssetPrecision.kraken_defaults()
        
        # Balances by asset
        self.balances: Dict[str, AssetBalance] = defaultdict(
            lambda: AssetBalance(asset='UNKNOWN')
        )
        
        # Frozen allocations by order_id
        self.frozen_allocations: Dict[str, FrozenAllocation] = {}
        
        # Audit trail
        self.ledger_entries: List[LedgerEntry] = []
        self._entry_counter = 0
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Reconciliation state
        self.last_reconcile_time = 0.0
        self.reconcile_discrepancies: List[Dict] = []
        
        logger.info("ShadowLedger initialized")
    
    def _generate_entry_id(self) -> str:
        """Generate unique entry ID."""
        self._entry_counter += 1
        return f"LE_{int(time.time())}_{self._entry_counter:06d}"
    
    def _record_entry(
        self,
        asset: str,
        entry_type: EntryType,
        amount: Decimal,
        balance_before: Decimal,
        balance_after: Decimal,
        reference: str = "",
        notes: str = ""
    ):
        """Record ledger entry for audit trail."""
        entry = LedgerEntry(
            entry_id=self._generate_entry_id(),
            timestamp=time.time(),
            asset=asset,
            entry_type=entry_type,
            amount=amount,
            balance_before=balance_before,
            balance_after=balance_after,
            reference=reference,
            notes=notes
        )
        self.ledger_entries.append(entry)
        
        # Keep only last 10000 entries
        if len(self.ledger_entries) > 10000:
            self.ledger_entries = self.ledger_entries[-10000:]
    
    def set_balance(self, asset: str, amount: Decimal, source: str = "INIT"):
        """Set initial balance for an asset."""
        with self._lock:
            amount = self._normalize_amount(asset, amount)
            
            if asset not in self.balances:
                self.balances[asset] = AssetBalance(asset=asset)
            
            old_balance = self.balances[asset].available
            self.balances[asset].available = amount
            
            self._record_entry(
                asset=asset,
                entry_type=EntryType.CREDIT if amount > old_balance else EntryType.DEBIT,
                amount=abs(amount - old_balance),
                balance_before=old_balance,
                balance_after=amount,
                reference=source,
                notes="Balance set"
            )
            
            # Check for dust
            self._check_dust(asset)
            
            logger.debug(f"Set {asset} balance: {amount}")
    
    def credit(self, asset: str, amount: Decimal, reference: str = "") -> bool:
        """Credit (add) amount to available balance."""
        with self._lock:
            amount = self._normalize_amount(asset, amount)
            
            if amount <= 0:
                logger.warning(f"Invalid credit amount: {amount}")
                return False
            
            if asset not in self.balances:
                self.balances[asset] = AssetBalance(asset=asset)
            
            old_balance = self.balances[asset].available
            self.balances[asset].available += amount
            
            self._record_entry(
                asset=asset,
                entry_type=EntryType.CREDIT,
                amount=amount,
                balance_before=old_balance,
                balance_after=self.balances[asset].available,
                reference=reference
            )
            
            logger.debug(f"Credit {asset}: +{amount} (now: {self.balances[asset].available})")
            return True
    
    def debit(self, asset: str, amount: Decimal, reference: str = "") -> bool:
        """Debit (subtract) amount from available balance."""
        with self._lock:
            amount = self._normalize_amount(asset, amount)
            
            if amount <= 0:
                logger.warning(f"Invalid debit amount: {amount}")
                return False
            
            if asset not in self.balances:
                logger.error(f"Cannot debit unknown asset: {asset}")
                return False
            
            if self.balances[asset].available < amount:
                logger.error(f"Insufficient balance for {asset}: have {self.balances[asset].available}, need {amount}")
                return False
            
            old_balance = self.balances[asset].available
            self.balances[asset].available -= amount
            
            self._record_entry(
                asset=asset,
                entry_type=EntryType.DEBIT,
                amount=amount,
                balance_before=old_balance,
                balance_after=self.balances[asset].available,
                reference=reference
            )
            
            # Check for dust
            self._check_dust(asset)
            
            logger.debug(f"Debit {asset}: -{amount} (now: {self.balances[asset].available})")
            return True
    
    def freeze(
        self,
        asset: str,
        amount: Decimal,
        order_id: str,
        expires_in: float = None
    ) -> bool:
        """Freeze amount for pending order."""
        with self._lock:
            amount = self._normalize_amount(asset, amount)
            
            if asset not in self.balances:
                logger.error(f"Cannot freeze unknown asset: {asset}")
                return False
            
            if self.balances[asset].available < amount:
                logger.error(f"Insufficient available for freeze: {self.balances[asset].available} < {amount}")
                return False
            
            # Move from available to frozen
            old_available = self.balances[asset].available
            self.balances[asset].available -= amount
            self.balances[asset].frozen += amount
            
            # Record allocation
            expires_at = 0.0
            if expires_in:
                expires_at = time.time() + expires_in
            
            self.frozen_allocations[order_id] = FrozenAllocation(
                order_id=order_id,
                asset=asset,
                amount=amount,
                timestamp=time.time(),
                expires_at=expires_at
            )
            
            self._record_entry(
                asset=asset,
                entry_type=EntryType.FREEZE,
                amount=amount,
                balance_before=old_available,
                balance_after=self.balances[asset].available,
                reference=order_id,
                notes=f"Frozen for order"
            )
            
            logger.debug(f"Freeze {asset}: {amount} for order {order_id}")
            return True
    
    def unfreeze(self, order_id: str, consumed: Decimal = None) -> bool:
        """
        Unfreeze amount when order completes or cancels.
        
        Args:
            order_id: Order ID
            consumed: Amount actually consumed (filled). Rest returns to available.
        """
        with self._lock:
            if order_id not in self.frozen_allocations:
                logger.warning(f"No frozen allocation for order: {order_id}")
                return False
            
            allocation = self.frozen_allocations.pop(order_id)
            asset = allocation.asset
            frozen_amount = allocation.amount
            
            consumed = consumed or Decimal('0')
            consumed = self._normalize_amount(asset, consumed)
            returned = frozen_amount - consumed
            
            # Update balances
            self.balances[asset].frozen -= frozen_amount
            self.balances[asset].available += returned
            
            self._record_entry(
                asset=asset,
                entry_type=EntryType.UNFREEZE,
                amount=returned,
                balance_before=self.balances[asset].available - returned,
                balance_after=self.balances[asset].available,
                reference=order_id,
                notes=f"Consumed: {consumed}, Returned: {returned}"
            )
            
            logger.debug(f"Unfreeze {asset}: returned {returned} for order {order_id}")
            return True
    
    def cleanup_expired_freezes(self) -> List[str]:
        """Release expired frozen allocations."""
        with self._lock:
            now = time.time()
            expired = []
            
            for order_id, allocation in list(self.frozen_allocations.items()):
                if allocation.expires_at > 0 and now > allocation.expires_at:
                    self.unfreeze(order_id)
                    expired.append(order_id)
                    logger.warning(f"Auto-unfroze expired allocation: {order_id}")
            
            return expired
    
    def _check_dust(self, asset: str):
        """Check if available balance is dust and move it."""
        if asset not in self.precision:
            return
        
        balance = self.balances[asset]
        threshold = self.precision[asset].dust_threshold
        
        if balance.available > 0 and balance.available < threshold:
            dust_amount = balance.available
            balance.dust += dust_amount
            balance.available = Decimal('0')
            
            self._record_entry(
                asset=asset,
                entry_type=EntryType.DUST_OUT,
                amount=dust_amount,
                balance_before=dust_amount,
                balance_after=Decimal('0'),
                notes="Moved to dust"
            )
            
            logger.debug(f"Moved {dust_amount} {asset} to dust")
    
    def _normalize_amount(self, asset: str, amount: Decimal) -> Decimal:
        """Normalize amount to asset precision."""
        if asset not in self.precision:
            return amount
        
        decimals = self.precision[asset].lot_decimals
        return amount.quantize(Decimal(10) ** -decimals, rounding=ROUND_DOWN)
    
    def round_to_lot_size(self, asset: str, amount: Decimal) -> Decimal:
        """Round amount to valid lot size."""
        if asset not in self.precision:
            return amount
        
        lot_size = self.precision[asset].lot_size
        lots = (amount / lot_size).quantize(Decimal('1'), rounding=ROUND_DOWN)
        return lots * lot_size
    
    def is_dust(self, asset: str, amount: Decimal) -> bool:
        """Check if amount is below dust threshold."""
        if asset not in self.precision:
            return False
        return amount < self.precision[asset].dust_threshold
    
    def get_balance(self, asset: str, balance_type: BalanceType = BalanceType.AVAILABLE) -> Decimal:
        """Get balance for asset."""
        with self._lock:
            if asset not in self.balances:
                return Decimal('0')
            
            balance = self.balances[asset]
            
            if balance_type == BalanceType.AVAILABLE:
                return balance.available
            elif balance_type == BalanceType.FROZEN:
                return balance.frozen
            elif balance_type == BalanceType.DUST:
                return balance.dust
            elif balance_type == BalanceType.TOTAL:
                return balance.total
            
            return Decimal('0')
    
    def get_all_balances(self) -> Dict[str, Dict]:
        """Get all balances as dict."""
        with self._lock:
            return {asset: balance.to_dict() for asset, balance in self.balances.items()}
    
    def reconcile(self, exchange_balances: Dict[str, Decimal]) -> Dict[str, Any]:
        """
        Reconcile local state with exchange balances.
        
        Returns discrepancies found.
        """
        with self._lock:
            self.last_reconcile_time = time.time()
            discrepancies = []
            
            for asset, exchange_balance in exchange_balances.items():
                local_total = self.get_balance(asset, BalanceType.TOTAL)
                exchange_balance = self._normalize_amount(asset, exchange_balance)
                
                diff = exchange_balance - local_total
                
                if abs(diff) > self.precision.get(asset, AssetPrecision.kraken_defaults()['USD']).dust_threshold:
                    discrepancy = {
                        'asset': asset,
                        'local': str(local_total),
                        'exchange': str(exchange_balance),
                        'difference': str(diff),
                        'timestamp': time.time()
                    }
                    discrepancies.append(discrepancy)
                    
                    # Auto-correct
                    old_balance = self.balances[asset].available
                    self.balances[asset].available += diff
                    
                    self._record_entry(
                        asset=asset,
                        entry_type=EntryType.RECONCILE,
                        amount=abs(diff),
                        balance_before=old_balance,
                        balance_after=self.balances[asset].available,
                        notes=f"Reconciliation adjustment: {diff}"
                    )
                    
                    logger.warning(f"Reconciled {asset}: adjusted by {diff}")
            
            self.reconcile_discrepancies.extend(discrepancies)
            
            # Keep only last 100 discrepancies
            if len(self.reconcile_discrepancies) > 100:
                self.reconcile_discrepancies = self.reconcile_discrepancies[-100:]
            
            return {
                'timestamp': self.last_reconcile_time,
                'discrepancies': discrepancies,
                'balances': self.get_all_balances()
            }
    
    def can_trade(self, asset: str, amount: Decimal, price: Decimal = None) -> Tuple[bool, str]:
        """
        Check if trade is possible.
        
        Returns:
            (can_trade, reason)
        """
        with self._lock:
            if asset not in self.precision:
                return False, "Unknown asset"
            
            precision = self.precision[asset]
            amount = self._normalize_amount(asset, amount)
            
            # Check lot size
            if amount < precision.lot_size:
                return False, f"Below lot size: {precision.lot_size}"
            
            # Check dust
            if amount < precision.dust_threshold:
                return False, f"Amount is dust: < {precision.dust_threshold}"
            
            # Check min order value
            if price:
                value = amount * price
                if value < precision.min_order_value:
                    return False, f"Below min order value: ${precision.min_order_value}"
            
            # Check available balance
            available = self.get_balance(asset, BalanceType.AVAILABLE)
            if amount > available:
                return False, f"Insufficient balance: {available} < {amount}"
            
            return True, "OK"
    
    def export_state(self) -> Dict:
        """Export full ledger state for persistence."""
        with self._lock:
            return {
                'timestamp': time.time(),
                'balances': self.get_all_balances(),
                'frozen_allocations': {
                    order_id: {
                        'order_id': alloc.order_id,
                        'asset': alloc.asset,
                        'amount': str(alloc.amount),
                        'timestamp': alloc.timestamp,
                        'expires_at': alloc.expires_at
                    }
                    for order_id, alloc in self.frozen_allocations.items()
                },
                'recent_entries': [
                    {
                        'entry_id': e.entry_id,
                        'timestamp': e.timestamp,
                        'asset': e.asset,
                        'type': e.entry_type.name,
                        'amount': str(e.amount),
                        'reference': e.reference
                    }
                    for e in self.ledger_entries[-100:]
                ]
            }
    
    def import_state(self, state: Dict) -> bool:
        """Import ledger state from persistence."""
        with self._lock:
            try:
                # Import balances
                for asset, balance_data in state.get('balances', {}).items():
                    self.balances[asset] = AssetBalance(
                        asset=asset,
                        available=Decimal(balance_data['available']),
                        frozen=Decimal(balance_data['frozen']),
                        dust=Decimal(balance_data['dust'])
                    )
                
                # Import frozen allocations
                for order_id, alloc_data in state.get('frozen_allocations', {}).items():
                    self.frozen_allocations[order_id] = FrozenAllocation(
                        order_id=alloc_data['order_id'],
                        asset=alloc_data['asset'],
                        amount=Decimal(alloc_data['amount']),
                        timestamp=alloc_data['timestamp'],
                        expires_at=alloc_data['expires_at']
                    )
                
                logger.info("Imported ledger state successfully")
                return True
                
            except Exception as e:
                logger.error(f"Failed to import state: {e}")
                return False
