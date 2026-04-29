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
        "spot": {
            "BTC": "BTC/USDT",
            "ETH": "ETH/USDT",
            "SOL": "SOL/USDT",
        },
    },
    "coinbase": {
        "perp": {
            "BTC": "BTC-PERP",
            "ETH": "ETH-PERP",
            "SOL": "SOL-PERP",
        },
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
