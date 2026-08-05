"""
HMATS v5.1 Phase 2 - ExchangeAdapter abstraction
==================================================

Abstract base class for venue-specific order placement, balance/position
queries, and funding-rate access. Decouples HMATS from any single exchange.

Concrete implementations:
  exchange.kraken_adapter.KrakenAdapter   — wraps existing CCXT Kraken
                                             integration (live)
  exchange.coinbase_adapter.CoinbaseAdapter — Coinbase Advanced Trade
                                              perp adapter (skeleton)

Phase 2 cutover protocol (dual-venue, [PARAMETER 3] = dual_venue):
  Week 1-2: Coinbase shadow (read-only data, no trades). Compares funding
            + spread + depth between venues. Validates symbol mapping.
  Week 3:   50% capital migrated. Both adapters active. Per-asset routing
            governed by exchange.routing.RoutingPolicy.
  Week 4:   100% migrated. Kraken adapter standby for rollback.

Iron Laws:
  4. fail-closed: every adapter method may raise; callers must catch.
     Each method documents its exception types.
  5. DRL ACTIVE during cutover: adapter swap does NOT touch DRL inference;
     DRL authority level untouched.
  8. DRL ACTIVE during cutover (Iron Law 8 per v5.1 prompt).
     ⚠️ [P155, 2026-08-04] This said "enforced at
     exchange.cutover.cutover_invariants(). Continuous check." Neither half
     was true: this module never imported cutover, and cutover_invariants()
     has no production caller at all. See exchange/cutover.py's header for
     what is actually wired.
  10. zero pre-existing-runtime side-effect for the abstraction itself —
      only concrete implementations interact with venue.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class OrderRequest:
    symbol: str
    side: str            # "BUY" or "SELL"
    size: float          # base-asset units
    order_type: str = "LIMIT"   # "LIMIT" / "MARKET" / "POST" / "STOP"
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"
    client_order_id: Optional[str] = None
    reduce_only: bool = False
    post_only: bool = True   # Iron Law 9: maker-first default
    leverage: Optional[int] = None


@dataclass
class OrderResult:
    success: bool
    venue: str
    order_id: Optional[str] = None
    client_order_id: Optional[str] = None
    filled_size: float = 0.0
    filled_price: Optional[float] = None
    status: str = "UNKNOWN"
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None


@dataclass
class FundingRateData:
    symbol: str
    funding_rate: float          # per-period rate as decimal
    funding_period_hours: float  # 1 (Kraken hourly) or 8 (Coinbase 8h)
    funding_rate_8h: float       # normalized to 8h equivalent
    next_funding_ts: Optional[float] = None
    venue: str = ""


class ExchangeAdapter(ABC):
    """Abstract venue adapter. All HMATS execution + market-data paths
    eventually call through one of these."""

    @property
    @abstractmethod
    def venue(self) -> str:
        """e.g. 'kraken' or 'coinbase'."""

    @abstractmethod
    def is_connected(self) -> bool:
        """True if the adapter has a live API session."""

    @abstractmethod
    async def place_order(self, request: OrderRequest) -> OrderResult: ...

    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> bool: ...

    @abstractmethod
    async def fetch_balance(self) -> Dict[str, float]:
        """Returns {asset: free_balance}. Includes USD/USDC quote."""

    @abstractmethod
    async def fetch_positions(self) -> List[Dict[str, Any]]:
        """Returns list of open position dicts."""

    @abstractmethod
    async def fetch_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]: ...

    @abstractmethod
    async def fetch_funding_rate(self, symbol: str) -> FundingRateData: ...

    @abstractmethod
    async def fetch_orderbook(
        self, symbol: str, depth: int = 25
    ) -> Dict[str, List[List[float]]]:
        """Returns {'bids': [[px, qty], ...], 'asks': [[px, qty], ...]}."""

    # ----- helpers (default impls; override only if venue differs) ---------

    def to_venue_symbol(self, hmats_asset: str) -> str:
        """Convert HMATS canonical asset (BTC/ETH/SOL) to venue-specific
        symbol. Default raises; concrete adapters override."""
        raise NotImplementedError(
            f"{self.venue} adapter must implement to_venue_symbol({hmats_asset})"
        )

    def from_venue_symbol(self, venue_symbol: str) -> str:
        """Inverse of to_venue_symbol."""
        raise NotImplementedError
