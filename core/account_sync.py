"""
================================================================================
HMATS Account Sync - Real-Time Account Equity Provider
================================================================================
Version: 5.4.1-P0FIX
Purpose: Provide REAL, FRESH account equity for unit conversion and leverage checks

CRITICAL CONSTRAINTS:
    1. Account equity MUST come from exchange (Kraken ONLY)
    2. Equity MUST be refreshed at least every tick
    3. FORBIDDEN: hard-coded equity, config defaults, stale cache
    4. If equity unavailable -> FAIL-CLOSED (no trading)

This module is the SINGLE SOURCE OF TRUTH for account equity.
================================================================================
"""

import logging
import time
import asyncio
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum, auto

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Maximum age for cached equity before it's considered stale
MAX_EQUITY_AGE_SECONDS = 60.0  # Must refresh at least every minute

# Kraken-only validation
ALLOWED_EXCHANGES = {"kraken"}


# =============================================================================
# EQUITY STATUS
# =============================================================================

class EquityStatus(Enum):
    """Status of account equity data."""
    VALID = auto()          # Fresh, verified equity
    STALE = auto()          # Equity older than MAX_EQUITY_AGE_SECONDS
    UNAVAILABLE = auto()    # Cannot fetch equity
    UNINITIALIZED = auto()  # Never fetched


@dataclass
class AccountState:
    """Current account state from exchange."""
    equity: float = 0.0
    available_balance: float = 0.0
    used_margin: float = 0.0
    unrealized_pnl: float = 0.0
    
    # Metadata
    timestamp: float = 0.0
    exchange: str = ""
    status: EquityStatus = EquityStatus.UNINITIALIZED
    
    # Positions (for leverage calculation)
    total_position_notional: float = 0.0
    position_count: int = 0
    
    def is_valid(self) -> bool:
        """Check if equity is valid for trading decisions."""
        if self.status != EquityStatus.VALID:
            return False
        
        age = time.time() - self.timestamp
        if age > MAX_EQUITY_AGE_SECONDS:
            return False
        
        if self.equity <= 0:
            return False
        
        return True
    
    def age_seconds(self) -> float:
        """Get age of this equity snapshot."""
        return time.time() - self.timestamp


# =============================================================================
# ACCOUNT SYNC MANAGER
# =============================================================================

class AccountSyncManager:
    """
    Manages real-time account equity synchronization with exchange.
    
    FAIL-CLOSED SEMANTICS:
    - If equity cannot be fetched -> no trading allowed
    - If equity is stale -> no trading allowed
    - If exchange is not Kraken -> no trading allowed
    
    Usage:
        sync = AccountSyncManager(kraken_client)
        await sync.refresh()
        
        equity = sync.get_equity()  # Raises if unavailable
        # or
        equity, valid = sync.get_equity_safe()  # Returns (0, False) if unavailable
    """
    
    def __init__(
        self, 
        exchange_client = None,
        exchange_name: str = "kraken",
        dry_run: bool = True
    ):
        """
        Initialize account sync manager.
        
        Args:
            exchange_client: Kraken exchange client (ccxt or custom)
            exchange_name: Must be "kraken"
            dry_run: If True, use simulated equity
        """
        # Validate exchange
        if exchange_name.lower() not in ALLOWED_EXCHANGES:
            raise ValueError(
                f"FAIL-CLOSED: Exchange '{exchange_name}' not allowed. "
                f"Only {ALLOWED_EXCHANGES} permitted."
            )
        
        self.exchange_client = exchange_client
        self.exchange_name = exchange_name.lower()
        self.dry_run = dry_run
        
        # Current state
        self._state = AccountState(
            status=EquityStatus.UNINITIALIZED,
            exchange=self.exchange_name
        )
        
        # Dry run simulated equity - [CFG-4] was $100K, actual account is $10K
        self._dry_run_equity = 10000.0
        self._dry_run_pnl = 0.0
        
        # Statistics
        self._refresh_count = 0
        self._failure_count = 0
        self._last_error: Optional[str] = None
        
        logger.info(
            f"[ACCOUNT_SYNC] Initialized: exchange={exchange_name}, dry_run={dry_run}"
        )
    
    async def refresh(self) -> Tuple[bool, str]:
        """
        Refresh account equity from exchange.
        
        Returns:
            (success, error_message)
        """
        self._refresh_count += 1
        
        try:
            if self.dry_run:
                return await self._refresh_dry_run()
            else:
                return await self._refresh_live()
                
        except Exception as e:
            self._failure_count += 1
            self._last_error = str(e)
            self._state.status = EquityStatus.UNAVAILABLE
            
            logger.error(f"[ACCOUNT_SYNC] Refresh failed: {e}")
            return False, str(e)
    
    async def _refresh_dry_run(self) -> Tuple[bool, str]:
        """Refresh with simulated equity for dry run."""
        self._state = AccountState(
            equity=self._dry_run_equity + self._dry_run_pnl,
            available_balance=self._dry_run_equity + self._dry_run_pnl,
            used_margin=0.0,
            unrealized_pnl=self._dry_run_pnl,
            timestamp=time.time(),
            exchange=self.exchange_name,
            status=EquityStatus.VALID,
            total_position_notional=0.0,
            position_count=0,
        )
        
        logger.debug(f"[ACCOUNT_SYNC] Dry run equity: ${self._state.equity:,.2f}")
        return True, ""
    
    async def _refresh_live(self) -> Tuple[bool, str]:
        """Refresh from live Kraken API."""
        if self.exchange_client is None:
            self._state.status = EquityStatus.UNAVAILABLE
            return False, "No exchange client configured"

        # [NONCE-FIX 2026-04-15] Retry once on Invalid nonce after reloading
        # Kraken time difference. Without this, a single nonce mismatch flips
        # status to UNAVAILABLE and blocks all execution for MAX_EQUITY_AGE
        # seconds. ccxt's nonce is local-time-based; if the previous container
        # left Kraken with a higher last-seen nonce, our first call lags.
        # [G2-RATELIMIT 2026-04-15] Also retry on Kraken 429/RateLimitExceeded
        # with exponential backoff (1s, 2s, 4s) to avoid IP-level ban.
        balance = None
        _max_attempts = 4  # 1 nonce retry + up to 3 rate-limit retries
        for _attempt in range(_max_attempts):
            try:
                balance = await asyncio.wait_for(
                    asyncio.to_thread(self.exchange_client.fetch_balance),
                    timeout=15.0,
                )
                break
            except Exception as _e:
                _err_str = str(_e)
                # Nonce: reload time + retry once
                if _attempt == 0 and "Invalid nonce" in _err_str:
                    try:
                        await asyncio.to_thread(self.exchange_client.load_time_difference)
                        logger.warning(
                            "[ACCOUNT_SYNC] Nonce mismatch; reloaded timeDifference, retrying"
                        )
                        continue
                    except Exception:
                        pass
                # Rate limit: exponential backoff (1, 2, 4 seconds)
                _is_rate_limited = (
                    "RateLimitExceeded" in type(_e).__name__
                    or "EAPI:Rate limit exceeded" in _err_str
                    or "429" in _err_str
                    or "Too Many Requests" in _err_str
                )
                if _is_rate_limited and _attempt < _max_attempts - 1:
                    _backoff = 2 ** _attempt
                    logger.warning(
                        f"[ACCOUNT_SYNC] Kraken rate limit; backoff {_backoff}s "
                        f"(attempt {_attempt+1}/{_max_attempts-1})"
                    )
                    await asyncio.sleep(_backoff)
                    continue
                self._state.status = EquityStatus.UNAVAILABLE
                self._failure_count += 1
                self._last_error = _err_str
                logger.error(f"[ACCOUNT_SYNC] fetch_balance failed: {_e}")
                return False, _err_str

        try:

            # Kraken returns balance in 'total' and 'free'
            # [SPOT-EQUITY-FIX 2026-04-15] HMATS uses Kraken SPOT (not futures).
            # For spot, "positions" ARE the crypto holdings — they're part of
            # the account balance, not separate. The previous code only summed
            # USD/USDT and treated crypto holdings as zero, causing equity to
            # collapse by ~25% the moment we bought BTC/ETH/SOL (the USD
            # decreased, the crypto balance increased, but equity ignored the
            # crypto half). This made existence_fuse + drawdown logic see
            # phantom losses on every buy.
            #
            # Fix: sum USD + USDT + (each crypto × ticker.last) for total equity.
            usd_total = float(balance.get('total', {}).get('USD', 0.0) or 0.0)
            usdt_total = float(balance.get('total', {}).get('USDT', 0.0) or 0.0)
            usd_free = float(balance.get('free', {}).get('USD', 0.0) or 0.0)
            usdt_free = float(balance.get('free', {}).get('USDT', 0.0) or 0.0)

            # Sum stable balances first
            equity = usd_total + usdt_total
            available = usd_free + usdt_free

            # Add USD value of crypto holdings (spot positions)
            crypto_value_usd = 0.0
            crypto_assets = ('BTC', 'XBT', 'ETH', 'SOL')
            kraken_symbol_map = {'BTC': 'BTC/USD', 'XBT': 'BTC/USD', 'ETH': 'ETH/USD', 'SOL': 'SOL/USD'}
            for sym in crypto_assets:
                _amt = float(balance.get('total', {}).get(sym, 0.0) or 0.0)
                if _amt > 1e-8:
                    try:
                        _ticker_sym = kraken_symbol_map[sym]
                        _ticker = await asyncio.wait_for(
                            asyncio.to_thread(self.exchange_client.fetch_ticker, _ticker_sym),
                            timeout=10.0,
                        )
                        _price = float(_ticker.get('last', 0.0) or 0.0)
                        if _price > 0:
                            _val = _amt * _price
                            crypto_value_usd += _val
                            logger.debug(
                                f"[ACCOUNT_SYNC] {sym}: {_amt:.6f} × ${_price:.2f} = ${_val:.2f}"
                            )
                    except Exception as _ce:
                        logger.warning(f"[ACCOUNT_SYNC] {sym} valuation failed: {_ce}")
            equity += crypto_value_usd
            # available stays as USD/USDT only (can't trade crypto holdings as collateral atomically)
            
            # Fetch positions for margin calculation
            positions = []
            try:
                positions = await asyncio.wait_for(
                    asyncio.to_thread(self.exchange_client.fetch_positions),
                    timeout=15.0,
                )
            except (asyncio.TimeoutError, Exception):
                pass  # Positions optional for spot
            
            total_notional = 0.0
            unrealized_pnl = 0.0
            for pos in positions:
                notional = abs(pos.get('notional', 0) or pos.get('contracts', 0) * pos.get('markPrice', 0))
                total_notional += notional
                unrealized_pnl += pos.get('unrealizedPnl', 0)
            
            self._state = AccountState(
                equity=equity,
                available_balance=available,
                used_margin=equity - available,
                unrealized_pnl=unrealized_pnl,
                timestamp=time.time(),
                exchange=self.exchange_name,
                status=EquityStatus.VALID,
                total_position_notional=total_notional,
                position_count=len(positions),
            )
            
            logger.info(
                f"[ACCOUNT_SYNC] Live equity: ${equity:,.2f}, "
                f"positions: {len(positions)}, notional: ${total_notional:,.2f}"
            )
            return True, ""
            
        except Exception as e:
            self._state.status = EquityStatus.UNAVAILABLE
            return False, str(e)
    
    def get_equity(self) -> float:
        """
        Get current account equity.
        
        FAIL-CLOSED: Raises exception if equity unavailable or stale.
        
        Returns:
            Account equity in USD
            
        Raises:
            RuntimeError: If equity is unavailable, stale, or invalid
        """
        if not self._state.is_valid():
            status = self._state.status.name
            age = self._state.age_seconds()
            raise RuntimeError(
                f"FAIL-CLOSED: Account equity unavailable. "
                f"Status={status}, Age={age:.1f}s, Equity={self._state.equity}"
            )
        
        return self._state.equity
    
    def get_equity_safe(self) -> Tuple[float, bool]:
        """
        Get equity with validity flag (for graceful handling).
        
        Returns:
            (equity, is_valid) tuple
        """
        return self._state.equity, self._state.is_valid()
    
    def get_state(self) -> AccountState:
        """Get full account state."""
        return self._state
    
    def get_total_position_notional(self) -> float:
        """Get total position notional for leverage calculation."""
        return self._state.total_position_notional
    
    def update_dry_run_pnl(self, pnl_change: float):
        """Update simulated P&L for dry run mode."""
        if self.dry_run:
            self._dry_run_pnl += pnl_change
    
    def set_dry_run_equity(self, equity: float):
        """Set simulated equity for dry run mode."""
        if self.dry_run:
            self._dry_run_equity = equity
    
    def get_stats(self) -> Dict[str, Any]:
        """Get sync statistics."""
        return {
            "refresh_count": self._refresh_count,
            "failure_count": self._failure_count,
            "success_rate": (
                (self._refresh_count - self._failure_count) / max(self._refresh_count, 1)
            ),
            "last_error": self._last_error,
            "current_status": self._state.status.name,
            "current_equity": self._state.equity,
            "equity_age_seconds": self._state.age_seconds(),
        }


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_account_sync: Optional[AccountSyncManager] = None


def get_account_sync(
    exchange_client = None,
    exchange_name: str = "kraken",
    dry_run: bool = True
) -> AccountSyncManager:
    """Get or create account sync singleton."""
    global _account_sync
    if _account_sync is None:
        _account_sync = AccountSyncManager(
            exchange_client=exchange_client,
            exchange_name=exchange_name,
            dry_run=dry_run
        )
    elif (
        getattr(_account_sync, "exchange_client", None) is not exchange_client
        or getattr(_account_sync, "exchange_name", "").lower() != exchange_name.lower()
        or bool(getattr(_account_sync, "dry_run", True)) != bool(dry_run)
    ):
        logger.info("[AccountSync] constructor inputs changed; refreshing singleton instance")
        _account_sync = AccountSyncManager(
            exchange_client=exchange_client,
            exchange_name=exchange_name,
            dry_run=dry_run
        )
    return _account_sync


def reset_account_sync():
    """Reset account sync singleton."""
    global _account_sync
    _account_sync = None


# =============================================================================
# VALIDATION UTILITIES
# =============================================================================

def validate_exchange_kraken_only(exchange_name: str, context: str = "") -> bool:
    """
    Validate that exchange is Kraken only.
    
    FAIL-CLOSED: Raises if not Kraken.
    """
    if exchange_name.lower() not in ALLOWED_EXCHANGES:
        raise RuntimeError(
            f"FAIL-CLOSED [{context}]: Exchange '{exchange_name}' forbidden. "
            f"Only Kraken is allowed in PAPER/LIVE modes."
        )
    return True


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    async def test():
        print("=" * 60)
        print("Account Sync Self-Test")
        print("=" * 60)
        
        # Test 1: Dry run mode
        print("\n[Test 1] Dry run equity")
        sync = AccountSyncManager(dry_run=True)
        success, error = await sync.refresh()
        assert success, f"Refresh failed: {error}"
        
        equity = sync.get_equity()
        print(f"  Equity: ${equity:,.2f}")
        assert equity == 10000.0, f"Expected $10k, got ${equity}"  # [CFG-4]
        
        # Test 2: Non-Kraken rejection
        print("\n[Test 2] Non-Kraken rejection")
        try:
            AccountSyncManager(exchange_name="binance")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            print(f"  ✓ Correctly rejected: {e}")
        
        # Test 3: Stale equity detection
        print("\n[Test 3] Stale equity detection")
        sync._state.timestamp = time.time() - 120  # 2 minutes old
        try:
            sync.get_equity()
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            print(f"  ✓ Correctly rejected stale: {e}")
        
        # Test 4: Safe getter
        print("\n[Test 4] Safe equity getter")
        equity, valid = sync.get_equity_safe()
        print(f"  Equity: ${equity:,.2f}, Valid: {valid}")
        assert not valid, "Should be invalid (stale)"
        
        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)
    
    asyncio.run(test())
