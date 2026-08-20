"""
================================================================================
HMATS - Exchange Module
================================================================================

[P338] THIS HEADER AND ITS BANNER WERE FALSIFIED ON 2026-06-13 AND STOOD.

    They declared "EXECUTION_VENUE: kraken ONLY" and "DO NOT: Add exchange
    routing/switching" -- printed to stdout on every import of the very
    package that contains `coinbase_adapter.py`, `coinbase_sleeve.py`,
    `routing.py` and `cutover.py`. Since the Phase B cutover the Coinbase US
    perp sleeve has been the SOLE directional venue and Kraken has been
    structurally flat (P152). Same class as the main.py header P239 corrected
    ("SINGLE EXCHANGE MODE (LOCKED): Kraken ONLY", falsified for two months)
    -- a mitigation applied to one instance of a class is not applied to the
    class (P171/P226).

EXECUTION VENUES (current):
    - Coinbase Derivatives Exchange US perps -- the SOLE directional driver
      (exchange/coinbase_sleeve.py, driven from main.py run_live).
    - Kraken spot/margin -- structurally flat since 2026-06-13; P152 skips
      every NEW entry for a Coinbase-routed asset, so this path can only ever
      unwind legacy spot, of which there is none.

DATA-ONLY (never execution):
    - binance (bootstrap/flow data), deribit (DVOL/options data)

STILL PROHIBITED, and actually enforced:
    - Routing trades to a DEX (Jupiter/Raydium/Orca) -- monitoring only.
      The enforcement lives in `core/exchange_guard.assert_not_dex_execution`,
      which does NOT read the module constants below.
================================================================================
"""

import logging

logger = logging.getLogger(__name__)

# ============================================================================
# SINGLE EXCHANGE MODE ENFORCEMENT (LOCKED)
# ============================================================================

# [P338] These three have ZERO readers anywhere in the tree (verified by
# grep across every production package): the real DEX prohibition is
# enforced by core/exchange_guard.py, which never consults them. They are
# KEPT rather than deleted because removing a module-level export is a
# contract change, and annotated so the next reader does not mistake them
# for live configuration. ACTIVE_EXCHANGE in particular is NOT the venue
# that executes -- see the header.
SINGLE_EXCHANGE_MODE = True          # unread; see core/exchange_guard.py
ACTIVE_EXCHANGE = "kraken"           # unread; NOT the directional venue
FROZEN_EXCHANGES = ["binance", "deribit"]  # unread; data-only exchanges

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
[EXCHANGE] Directional venue: COINBASE US PERPS (sole driver)
   Kraken: structurally flat since 2026-06-13 (P152) - unwind only
   Data-only: binance, deribit
   DEX execution: BLOCKED (core/exchange_guard.py)
============================================================
"""
    print(banner)
    logger.info("[EXCHANGE] directional=COINBASE_PERPS kraken=FLAT_UNWIND_ONLY "
                "dex=BLOCKED")

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
