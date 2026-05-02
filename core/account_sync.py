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

# Maximum age for cached equity before it's considered stale.
# 2026-05-02: bumped 60s → 120s. Symptom: production saw "Status=VALID,
# Age=66.3s, Equity=9562.39" FAIL-CLOSED rejections — equity was known
# and recently fetched, but refresh + downstream work between refresh
# and get_equity() can routinely consume 30-50s during Kraken
# congestion (each fetch_balance/fetch_ticker has a 15s timeout, and
# crypto valuation iterates per non-zero balance asset). 60s left no
# headroom; 120s gives a 2× safety margin while still being tighter
# than the 4H tick cycle.
MAX_EQUITY_AGE_SECONDS = 120.0

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
            # [P69 2026-04-26] Same fresh-state preservation as the
            # exhausted-retry branch below. If the exchange client is
            # transiently None (e.g. mid-reinit), don't invalidate
            # fresh cached equity — the staleness threshold will
            # invalidate it later if it actually goes stale.
            _was_valid = self._state.status == EquityStatus.VALID
            _age = (
                time.time() - self._state.timestamp
                if self._state.timestamp
                else float("inf")
            )
            if _was_valid and _age < MAX_EQUITY_AGE_SECONDS:
                logger.warning(
                    f"[ACCOUNT_SYNC] No exchange client but prior equity is "
                    f"fresh (age={_age:.1f}s); keeping VALID state."
                )
                return False, "transient_keep_valid: no exchange client"
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
        _max_attempts = 5  # 2 nonce retries + up to 3 rate-limit retries
        for _attempt in range(_max_attempts):
            try:
                balance = await asyncio.wait_for(
                    asyncio.to_thread(self.exchange_client.fetch_balance),
                    timeout=15.0,
                )
                break
            except Exception as _e:
                _err_str = str(_e)
                # Nonce: try ratchet bump first, then time-difference reload.
                # Production logs on 2026-04-25 showed back-to-back fetch_balance
                # nonce errors after the 1-attempt retry, meaning load_time_difference
                # alone wasn't enough — the highest nonce Kraken has seen for this
                # key was beyond clock skew. Bump the wrapper's nonce ratchet by
                # a fixed offset so the next call clears the tracker.
                if "Invalid nonce" in _err_str and _attempt < 2:
                    # Find the wrapper that owns the nonce ratchet (KrakenRESTClient).
                    _bump = getattr(self.exchange_client, "_bump_nonce_past_error", None)
                    if _bump is None:
                        # exchange_client may BE the inner ccxt; reach for the wrapper
                        # via its parent attribute set in main.py:4281 if available.
                        _bump = getattr(getattr(self, "_rest_wrapper", None), "_bump_nonce_past_error", None)
                    try:
                        if _bump is not None:
                            _new = _bump(60 if _attempt == 0 else 120)
                            logger.warning(
                                f"[ACCOUNT_SYNC] Nonce mismatch (attempt {_attempt+1}); "
                                f"bumped ratchet to {_new}, retrying"
                            )
                        else:
                            await asyncio.to_thread(self.exchange_client.load_time_difference)
                            logger.warning(
                                "[ACCOUNT_SYNC] Nonce mismatch; reloaded timeDifference (no ratchet available), retrying"
                            )
                        continue
                    except Exception as _bump_err:
                        logger.warning(
                            f"[ACCOUNT_SYNC] Nonce ratchet bump failed: {_bump_err}; "
                            "falling through to fail path"
                        )
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
                # [P69 2026-04-26] Preserve fresh cached state across a
                # transient fetch_balance failure. Pre-P69 ANY exhausted
                # retry loop unconditionally flipped status to UNAVAILABLE
                # — even when the prior equity snapshot was 5-30s old and
                # would have happily satisfied get_equity()'s staleness
                # check. Production symptom: operator saw
                # `Status=UNAVAILABLE, Age=23.7s, Equity=$8682.32` blocking
                # ALL execution via [P0_FAIL_CLOSED] when the cached value
                # was perfectly usable. Same shape as the parse-exception
                # branch's existing guard at line 361 — applying here too.
                # The staleness threshold (MAX_EQUITY_AGE_SECONDS=60s)
                # remains the authoritative validity gate.
                _was_valid = self._state.status == EquityStatus.VALID
                _age = (
                    time.time() - self._state.timestamp
                    if self._state.timestamp
                    else float("inf")
                )
                self._failure_count += 1
                self._last_error = _err_str
                if _was_valid and _age < MAX_EQUITY_AGE_SECONDS:
                    logger.warning(
                        f"[ACCOUNT_SYNC] fetch_balance failed but prior equity "
                        f"is fresh (age={_age:.1f}s < {MAX_EQUITY_AGE_SECONDS:.0f}s); "
                        f"keeping VALID state. Last error: {type(_e).__name__}: "
                        f"{_err_str[:200]}"
                    )
                    return False, f"transient_keep_valid: {_err_str}"
                self._state.status = EquityStatus.UNAVAILABLE
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
            # [P133 2026-04-29] SOL switched to SOL/USDT for valuation —
            # SOL/USD is dead on Kraken (0/1 bars, $0 volume in 24h).
            # USDT ≈ USD peg (<0.5% deviation) so notional math unchanged.
            crypto_value_usd = 0.0
            crypto_assets = ('BTC', 'XBT', 'ETH', 'SOL')
            kraken_symbol_map = {'BTC': 'BTC/USD', 'XBT': 'BTC/USD', 'ETH': 'ETH/USD', 'SOL': 'SOL/USDT'}
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
            except (asyncio.TimeoutError, Exception) as _pos_err:
                # Positions optional for spot, but a silent swallow hid Kraken
                # API errors (P25/P64 silent-failure pattern). Surface at WARNING
                # so operator sees fetch_positions failures instead of a
                # mysterious notional=0.0 in leverage calculations.
                logger.warning(
                    f"[ACCOUNT_SYNC] fetch_positions failed: "
                    f"{type(_pos_err).__name__}: {_pos_err}; "
                    f"using empty positions list (notional=0.0)"
                )
            
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
            # Preserve recently-valid state across a transient parse/ticker
            # exception. The previous behavior wholesale flipped status to
            # UNAVAILABLE on any exception, which fail-closes thesis_budget
            # at the call site (main.py:11017) even when the previous equity
            # snapshot is fresher than the staleness threshold. Only flip to
            # UNAVAILABLE if the prior state is genuinely stale.
            _was_valid = self._state.status == EquityStatus.VALID
            _age = time.time() - self._state.timestamp if self._state.timestamp else float("inf")
            if _was_valid and _age < MAX_EQUITY_AGE_SECONDS:
                logger.warning(
                    f"[ACCOUNT_SYNC] Parse exception but prior equity is fresh "
                    f"(age={_age:.1f}s < {MAX_EQUITY_AGE_SECONDS:.0f}s); "
                    f"keeping VALID state. Exception: {type(e).__name__}: {e}"
                )
                return False, f"transient_keep_valid: {type(e).__name__}: {e}"
            logger.exception(f"[ACCOUNT_SYNC] update() failed and prior state stale (age={_age:.1f}s)")
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
        # Soft-staleness signal: log when equity is between 75% and 100% of
        # MAX_EQUITY_AGE_SECONDS so operator sees the gap before it FAIL-CLOSES.
        # If this WARN appears regularly, the refresh→get_equity work is
        # consuming too much wall time and either MAX_EQUITY_AGE should be
        # raised again or the path between them needs trimming.
        _age = self._state.age_seconds()
        if _age > MAX_EQUITY_AGE_SECONDS * 0.75:
            logger.warning(
                f"[ACCOUNT_SYNC] equity age {_age:.1f}s approaching staleness "
                f"limit {MAX_EQUITY_AGE_SECONDS:.0f}s; refresh→get_equity path "
                f"is slow."
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
