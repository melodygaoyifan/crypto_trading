"""
================================================================================
HMATS v5.3.0 - Exchange Module (SINGLE_EXCHANGE_MODE)
================================================================================

LOCKED CONFIGURATION:
    ACTIVE_EXCHANGE: kraken
    EXECUTION_VENUE: kraken ONLY
    
FROZEN (moved to legacy/data_only_exchanges/):
    - binance (data bootstrap only - NOT for execution)
    - deribit (DVOL data only - NOT for execution)

DO NOT:
    - Add exchange routing/switching
    - Add multi-venue order splitting
    - Add cross-exchange arbitrage
    - Import execution code from binance/deribit
    - Route trades to DEX (Jupiter/Raydium/Orca) - monitoring only [P0-03]
================================================================================
"""

import logging

logger = logging.getLogger(__name__)

# ============================================================================
# SINGLE EXCHANGE MODE ENFORCEMENT (LOCKED)
# ============================================================================

SINGLE_EXCHANGE_MODE = True
ACTIVE_EXCHANGE = "kraken"
FROZEN_EXCHANGES = ["binance", "deribit"]  # Moved to legacy/data_only_exchanges/

# ============================================================================
# KRAKEN EXPORTS (CANONICAL)
# ============================================================================

try:
    from .kraken import (
        KrakenDefensiveLink,
        KrakenTokenBucket,
        get_kraken_link,
        reset_kraken_link,
    )
    KRAKEN_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Kraken module import failed: {e}")
    KrakenDefensiveLink = None
    KrakenTokenBucket = None
    KRAKEN_AVAILABLE = False

# ============================================================================
# STARTUP BANNER
# ============================================================================

def print_exchange_banner():
    """Print single exchange mode banner at startup."""
    banner = """
============================================================
[LOCK] SINGLE_EXCHANGE_MODE=ON
   Active Exchange: KRAKEN
   Frozen Exchanges: binance, deribit (moved to legacy/)
   Execution Venue: KRAKEN ONLY
============================================================
"""
    print(banner)
    logger.info("[EXCHANGE] SINGLE_EXCHANGE_MODE=ON Active=KRAKEN")

# Auto-print on import
print_exchange_banner()

# ============================================================================
# FORBIDDEN: Data-only exchanges are in legacy/data_only_exchanges/
# DO NOT import them here for execution
# ============================================================================

__all__ = [
    "SINGLE_EXCHANGE_MODE",
    "ACTIVE_EXCHANGE",
    "FROZEN_EXCHANGES",
    "KRAKEN_AVAILABLE",
    "KrakenDefensiveLink",
    "KrakenTokenBucket",
    "get_kraken_link",
    "reset_kraken_link",
    "print_exchange_banner",
]
