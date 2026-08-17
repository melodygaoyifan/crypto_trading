"""
HMATS v5.1 Phase 2 - Symbol Mapping (Kraken + Coinbase)
=========================================================

Single source of truth for HMATS-canonical asset → venue-specific symbol.

HMATS canonical:  BTC / ETH / SOL  (uppercase 3-letter ticker)
Kraken Futures:   PF_XBTUSD / PF_ETHUSD / PF_SOLUSD
Kraken Spot:      XBT/USD or BTC/USDT, ETH/USDT, SOL/USDT
Coinbase Perp:    BTC-PERP / ETH-PERP / SOL-PERP (per V14 evidence)
Coinbase Spot:    BTC-USD / ETH-USD / SOL-USD

Iron Laws:
  4. fail-closed: unknown symbol → KeyError with documented suggestion.
"""

from __future__ import annotations

from typing import Dict


# Phase 2 dual-venue support: HMATS canonical → venue-specific
SYMBOL_MAP: Dict[str, Dict[str, Dict[str, str]]] = {
    "kraken": {
        "perp": {
            "BTC": "PF_XBTUSD",
            "ETH": "PF_ETHUSD",
            "SOL": "PF_SOLUSD",
        },
        # [P253d] Aligned to core/execution_service._CANONICAL_SPOT_SYMBOL,
        # which is the evidence-backed truth: BTC/ETH trade USD pairs (the
        # account is USD-denominated), while SOL is DELIBERATELY USDT —
        # P133/P135/P137 moved it after Kraken's SOL/USD pair went dead
        # (OnMaintenance). This map previously said USDT for all three, and
        # the first P253d alignment wrongly said USD for all three — the
        # per-asset test now compares THIS map against the canonical dict so
        # the two cannot drift apart in either direction again. The only
        # consumer of this spot map is exchange/kraken_adapter.py (a
        # placeholder with no production caller), so no live order changes.
        "spot": {
            "BTC": "BTC/USD",
            "ETH": "ETH/USD",
            "SOL": "SOL/USDT",
        },
    },
    "coinbase": {
        # US Perpetual-Style Futures (Coinbase Derivatives Exchange / CFTC FCM).
        # Confirmed via live account probe 2026-06-13 — display names "BTC/ETH/SOL
        # PERP", 5yr-dated (20DEC30) perpetual-style contracts on the -CDE venue.
        # WARNING: the -PERP-INTX products (BTC-PERP-INTX, ...) are Coinbase
        # International = US-RESTRICTED; do NOT use them for a US account.
        # The 20DEC30 expiry tag is the current perpetual-style contract; re-run
        # scripts/coinbase_probe.py if Coinbase rolls it.
        "perp": {
            "BTC": "BIP-20DEC30-CDE",
            "ETH": "ETP-20DEC30-CDE",
            "SOL": "SLP-20DEC30-CDE",
            # [P292] The five P262-certified breadth assets. Product ids and
            # contract specs were read from the venue by the P291-C read-only
            # probe (2026-08-17) and already sit in the adapter's fallback
            # tables; the probe re-read BTC/ETH/SOL in the same pass and
            # reproduced the three rows above exactly, which is the control
            # that makes these five trustworthy. Full raw output lives in
            # tests/test_p291_breadth_readiness.py's module docstring.
            #
            # ADDING THESE CHANGES NOTHING LIVE. A perp entry only lets an
            # asset be RESOLVED to a product id; it cannot make one trade.
            # Three independent locks still stand, in the order the runtime
            # hits them (all pinned in tests/test_p292_xrp_readiness.py):
            #   1. `config.assets` — the sleeve driver loops over exactly this
            #      list (main.py) and the sleeve's `_pid_to_asset` is built
            #      from it, so an absent asset is never even considered;
            #   2. no `coinbase_target_fraction_by_asset` /
            #      `coinbase_max_contracts_by_asset` entry -> unsizable;
            #   3. `data/coinbase_routing_state.json` `coinbase_assets`
            #      -> `_coinbase_routed()` is False.
            # Widening is a config flip (+ the P197 one-asset-first rule) on a
            # PASS of the ~2026-09-15 breadth forward read — not a code change,
            # which is the whole point of landing this now.
            "XRP": "XPP-20DEC30-CDE",
            "ADA": "ADP-20DEC30-CDE",
            "LTC": "LCP-20DEC30-CDE",
            "DOGE": "DOP-20DEC30-CDE",
            "BNB": "BNB-20DEC30-CDE",
        },
        # [P292] Deliberately NOT extended: the breadth widening is a PERP
        # decision on the CDE sleeve. Spot symbols for these assets are
        # unverified (no probe covered them) and unused — an unverified entry
        # is exactly the fabricated-unit hazard P265h exists to prevent.
        "spot": {
            "BTC": "BTC-USD",
            "ETH": "ETH-USD",
            "SOL": "SOL-USD",
        },
    },
}


def to_venue_symbol(asset: str, venue: str, market: str = "perp") -> str:
    """Convert HMATS canonical asset → venue-specific symbol.

    Args:
        asset: 'BTC' / 'ETH' / 'SOL'
        venue: 'kraken' / 'coinbase'
        market: 'perp' / 'spot'

    Raises:
        KeyError if asset/venue/market combo is not mapped.
    """
    asset = asset.upper()
    venue = venue.lower()
    market = market.lower()
    if venue not in SYMBOL_MAP:
        raise KeyError(f"unknown venue: {venue} (known: {list(SYMBOL_MAP.keys())})")
    venue_map = SYMBOL_MAP[venue]
    if market not in venue_map:
        raise KeyError(f"unknown market: {market} for venue {venue}")
    market_map = venue_map[market]
    if asset not in market_map:
        raise KeyError(
            f"unknown asset {asset} for {venue}/{market} "
            f"(known: {list(market_map.keys())})"
        )
    return market_map[asset]


def from_venue_symbol(venue_symbol: str, venue: str, market: str = "perp") -> str:
    """Inverse of to_venue_symbol."""
    venue = venue.lower()
    market = market.lower()
    if venue not in SYMBOL_MAP or market not in SYMBOL_MAP[venue]:
        raise KeyError(f"unknown venue/market: {venue}/{market}")
    for asset, vsym in SYMBOL_MAP[venue][market].items():
        if vsym == venue_symbol:
            return asset
    raise KeyError(f"unknown {venue}/{market} symbol: {venue_symbol}")


def supported_assets(venue: str, market: str = "perp") -> list:
    """Return list of HMATS-canonical assets supported by venue/market."""
    venue = venue.lower()
    market = market.lower()
    if venue not in SYMBOL_MAP or market not in SYMBOL_MAP[venue]:
        return []
    return list(SYMBOL_MAP[venue][market].keys())
