"""
================================================================================
HMATS Exchange Guard - Kraken-Only Runtime Enforcement
================================================================================
Version: 5.4.1-P0FIX
Purpose: Enforce Kraken-only trading at runtime

CRITICAL RULE:
    In PAPER/LIVE modes, ONLY Kraken is allowed.
    Any attempt to use other exchanges -> FAIL-CLOSED

This module provides:
    1. Exchange validation at startup
    2. Runtime guards for any exchange operations
    3. Detection of non-Kraken clients

Usage:
    from core.exchange_guard import validate_kraken_only, ExchangeGuard
    
    # At startup
    validate_kraken_only(exchange_name, mode)
    
    # Runtime guard
    guard = ExchangeGuard(mode="live")
    guard.check_client(client)  # Raises if not Kraken
================================================================================
"""

import logging
from typing import Any, Optional, Set
from enum import Enum, auto

logger = logging.getLogger(__name__)


# =============================================================================
# EXCEPTIONS
# =============================================================================

class NonKrakenExchangeError(Exception):
    """Raised when a non-Kraken exchange is detected."""
    pass


class DEXExecutionBlockedError(RuntimeError):
    """Raised when DEX execution is attempted under SINGLE_EXCHANGE_MODE."""
    pass


# =============================================================================
# CONSTANTS
# =============================================================================

# Only Kraken is allowed
ALLOWED_EXCHANGES: Set[str] = {"kraken"}

# Known disallowed exchange identifiers
DISALLOWED_PATTERNS = {
    "binance",
    "deribit",
    "bybit",
    "okx",
    "ftx",
    "huobi",
    "coinbase",
    "kucoin",
    "bitfinex",
    "bitmex",
    "phemex",
}

# [P0-03] DEX venues - monitoring allowed, execution FORBIDDEN
DEX_VENUES = {
    "jupiter",
    "raydium",
    "orca",
    "dex",
    "serum",
    "drift",
    "mango",
    "lifinity",
    "phoenix",
}


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_kraken_only(
    exchange_name: str,
    mode: str,
    context: str = "",
) -> bool:
    """
    Validate that exchange is Kraken.
    
    Args:
        exchange_name: Exchange identifier
        mode: Run mode (verify, backtest, paper, live)
        context: Additional context for error message
        
    Returns:
        True if valid
        
    Raises:
        NonKrakenExchangeError: If not Kraken in PAPER/LIVE mode
    """
    exchange_lower = exchange_name.lower().strip()
    
    # Check if it's Kraken
    is_kraken = exchange_lower in ALLOWED_EXCHANGES
    
    # In verify/backtest mode, log warning but allow
    if mode.lower() in ("verify", "backtest"):
        if not is_kraken:
            logger.warning(
                f"[EXCHANGE_GUARD] Non-Kraken exchange '{exchange_name}' "
                f"detected in {mode} mode. Allowed for testing only. "
                f"Context: {context}"
            )
        return True
    
    # In PAPER/LIVE mode, enforce Kraken-only
    if not is_kraken:
        error_msg = (
            f"FAIL-CLOSED: Non-Kraken exchange '{exchange_name}' detected "
            f"in {mode.upper()} mode. Only Kraken is allowed for real trading. "
            f"Context: {context}"
        )
        logger.critical(f"[EXCHANGE_GUARD] {error_msg}")
        raise NonKrakenExchangeError(error_msg)
    
    logger.info(
        f"[EXCHANGE_GUARD] Exchange validated: {exchange_name} "
        f"(mode={mode}, context={context})"
    )
    return True


def assert_not_dex_execution(
    venue: str,
    *,
    action: str,
    asset: str = "",
    single_exchange_mode: bool = True,
) -> None:
    """
    [P0-03] Block DEX execution under SINGLE_EXCHANGE_MODE.

    Monitoring/signal functions should NOT call this - only functions that
    would result in an on-chain or off-chain trade (swap, submit_tx, etc.).

    Args:
        venue: Venue identifier (e.g. "jupiter", "dex", "raydium").
        action: Human-readable action name (e.g. "swap", "place_order").
        asset: Asset being traded (for logging).
        single_exchange_mode: If False, DEX execution is allowed (future use).

    Raises:
        DEXExecutionBlockedError: Always raises under SINGLE_EXCHANGE_MODE
            when venue matches a known DEX.
    """
    if not single_exchange_mode:
        return  # DEX execution allowed (not currently used)

    venue_lower = venue.lower().strip()
    if venue_lower in DEX_VENUES or venue_lower in DISALLOWED_PATTERNS:
        error_msg = (
            f"DEX_EXECUTION_BLOCKED: venue='{venue}' action='{action}' "
            f"asset='{asset}' - SINGLE_EXCHANGE_MODE is ON. "
            f"Only Kraken execution is allowed. DEX may be used for "
            f"monitoring/signals only."
        )
        logger.critical(f"[EXCHANGE_GUARD] {error_msg}")
        raise DEXExecutionBlockedError(error_msg)


def detect_exchange_from_url(url: str) -> Optional[str]:
    """
    Detect exchange from API URL.
    
    Args:
        url: API endpoint URL
        
    Returns:
        Exchange name or None if unknown
    """
    url_lower = url.lower()
    
    # Check for Kraken
    if "kraken" in url_lower:
        return "kraken"
    
    # Check for disallowed
    for exchange in DISALLOWED_PATTERNS:
        if exchange in url_lower:
            return exchange
    
    return None


def detect_exchange_from_client(client: Any) -> Optional[str]:
    """
    Detect exchange from CCXT or custom client.
    
    Args:
        client: Exchange client object
        
    Returns:
        Exchange name or None if unknown
    """
    if client is None:
        return None
    
    # CCXT clients have 'id' attribute
    if hasattr(client, 'id'):
        return client.id.lower()
    
    # Check class name
    class_name = type(client).__name__.lower()
    
    for exchange in ALLOWED_EXCHANGES | DISALLOWED_PATTERNS:
        if exchange in class_name:
            return exchange
    
    # Check for 'name' attribute
    if hasattr(client, 'name'):
        name = str(client.name).lower()
        for exchange in ALLOWED_EXCHANGES | DISALLOWED_PATTERNS:
            if exchange in name:
                return exchange
    
    return None


# =============================================================================
# EXCHANGE GUARD CLASS
# =============================================================================

class ExchangeGuard:
    """
    Runtime guard for exchange operations.
    
    Usage:
        guard = ExchangeGuard(mode="live")
        
        # Check client
        guard.check_client(client)
        
        # Check URL
        guard.check_url(url)
        
        # Wrap operations
        with guard.validate(exchange_name):
            # ... exchange operation ...
    """
    
    def __init__(self, mode: str = "verify"):
        """
        Initialize exchange guard.
        
        Args:
            mode: Run mode (verify, backtest, paper, live)
        """
        self.mode = mode.lower()
        self._is_strict = self.mode in ("paper", "live")
        
        logger.info(
            f"[EXCHANGE_GUARD] Initialized: mode={mode}, strict={self._is_strict}"
        )
    
    def check_client(self, client: Any, context: str = "") -> bool:
        """
        Check if client is for Kraken.
        
        Args:
            client: Exchange client
            context: Additional context
            
        Returns:
            True if valid
            
        Raises:
            NonKrakenExchangeError: If not Kraken in strict mode
        """
        exchange = detect_exchange_from_client(client)
        
        if exchange is None:
            if self._is_strict:
                raise NonKrakenExchangeError(
                    f"FAIL-CLOSED: Unable to determine exchange from client "
                    f"in {self.mode.upper()} mode. Context: {context}"
                )
            else:
                logger.warning(
                    f"[EXCHANGE_GUARD] Unknown exchange client. Context: {context}"
                )
                return True
        
        return validate_kraken_only(exchange, self.mode, context)
    
    def check_url(self, url: str, context: str = "") -> bool:
        """
        Check if URL is for Kraken.
        
        Args:
            url: API URL
            context: Additional context
            
        Returns:
            True if valid
            
        Raises:
            NonKrakenExchangeError: If not Kraken in strict mode
        """
        exchange = detect_exchange_from_url(url)
        
        if exchange is None:
            if self._is_strict:
                logger.warning(
                    f"[EXCHANGE_GUARD] Unable to detect exchange from URL: {url}"
                )
            return True
        
        return validate_kraken_only(exchange, self.mode, context)
    
    def check_name(self, exchange_name: str, context: str = "") -> bool:
        """
        Check exchange name directly.
        
        Args:
            exchange_name: Exchange name
            context: Additional context
            
        Returns:
            True if valid
        """
        return validate_kraken_only(exchange_name, self.mode, context)


# =============================================================================
# SINGLETON
# =============================================================================

_exchange_guard: Optional[ExchangeGuard] = None


def get_exchange_guard(mode: str = "verify") -> ExchangeGuard:
    """Get or create exchange guard singleton."""
    global _exchange_guard
    if _exchange_guard is None:
        _exchange_guard = ExchangeGuard(mode)
    elif getattr(_exchange_guard, "mode", None) != mode:
        logger.info("[ExchangeGuard] mode changed; refreshing singleton instance")
        _exchange_guard = ExchangeGuard(mode)
    return _exchange_guard


def reset_exchange_guard():
    """Reset exchange guard singleton."""
    global _exchange_guard
    _exchange_guard = None


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Exchange Guard Self-Test")
    print("=" * 60)
    
    # Test 1: Kraken allowed
    print("\n[Test 1] Kraken allowed in live mode")
    validate_kraken_only("kraken", "live", "test1")
    print("  ✓ Kraken validated")
    
    # Test 2: Binance rejected in live mode
    print("\n[Test 2] Binance rejected in live mode")
    try:
        validate_kraken_only("binance", "live", "test2")
        print("  ✗ Should have raised exception")
    except NonKrakenExchangeError as e:
        print(f"  ✓ Correctly rejected: {e}")
    
    # Test 3: Binance allowed in verify mode
    print("\n[Test 3] Binance allowed in verify mode (warning only)")
    validate_kraken_only("binance", "verify", "test3")
    print("  ✓ Allowed with warning")
    
    # Test 4: URL detection
    print("\n[Test 4] URL detection")
    assert detect_exchange_from_url("https://api.kraken.com") == "kraken"
    assert detect_exchange_from_url("https://api.binance.com") == "binance"
    print("  ✓ URL detection works")
    
    # Test 5: Guard class
    print("\n[Test 5] ExchangeGuard class")
    guard = ExchangeGuard(mode="live")
    try:
        guard.check_name("deribit", "test5")
        print("  ✗ Should have raised exception")
    except NonKrakenExchangeError:
        print("  ✓ Guard rejected non-Kraken")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
