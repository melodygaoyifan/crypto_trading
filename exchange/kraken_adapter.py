"""
HMATS v5.1 Phase 2 - Kraken adapter (wraps existing live integration)
======================================================================

Wraps the pre-existing CCXT Kraken integration so it conforms to
exchange.adapter.ExchangeAdapter. This is a thin compatibility shell —
the live engine continues to use the existing execution_manager + ccxt
client paths through Phase 2 dual-venue. After Phase 2 cutover completes,
this adapter is the canonical Kraken touchpoint.

Iron Laws:
  4. fail-closed: every method catches ccxt errors → returns failure
     OrderResult / empty list / RuntimeError per docstring.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from exchange.adapter import (
    ExchangeAdapter,
    FundingRateData,
    OrderRequest,
    OrderResult,
)
from exchange.symbol_mapping import to_venue_symbol, from_venue_symbol

logger = logging.getLogger(__name__)


class KrakenAdapter(ExchangeAdapter):
    """Adapter that wraps the existing Kraken execution path.

    Constructor accepts an optional pre-wired ccxt client; if None, a
    new client is lazily constructed from env vars at first use.
    """

    def __init__(self, ccxt_client: Optional[Any] = None) -> None:
        self._client = ccxt_client
        self._client_init_failed: Optional[str] = None

    @property
    def venue(self) -> str:
        return "kraken"

    def is_connected(self) -> bool:
        return self._client is not None and self._client_init_failed is None

    def _ensure_client(self) -> bool:
        if self._client is not None:
            return True
        if self._client_init_failed is not None:
            return False
        try:
            import ccxt  # type: ignore
            self._client = ccxt.kraken()
            return True
        except Exception as e:
            self._client_init_failed = f"{type(e).__name__}:{e}"
            logger.warning(f"[KRAKEN_ADAPTER] ccxt client init failed: {self._client_init_failed}")
            return False

    def to_venue_symbol(self, hmats_asset: str, market: str = "perp") -> str:
        return to_venue_symbol(hmats_asset, "kraken", market)

    def from_venue_symbol(self, venue_symbol: str, market: str = "perp") -> str:
        return from_venue_symbol(venue_symbol, "kraken", market)

    async def place_order(self, request: OrderRequest) -> OrderResult:
        if not self._ensure_client():
            return OrderResult(
                success=False, venue="kraken",
                error_code="CLIENT_INIT_FAILED",
                error_message=self._client_init_failed,
            )
        # In production, defer to existing execution_manager.execute_order;
        # this method is the v5.1 abstraction surface that Phase 2 cutover
        # gates use. Concrete wiring lands in Phase 2 step 2.5.
        return OrderResult(
            success=False, venue="kraken",
            error_code="DELEGATED",
            error_message=("KrakenAdapter is the v5.1 abstraction surface; "
                           "live order placement still flows through "
                           "execution_manager.execute_order during dual-venue. "
                           "Phase 2 step 2.5 cuts the abstraction over."),
        )

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        # Same delegation pattern as place_order during dual-venue.
        return False

    async def fetch_balance(self) -> Dict[str, float]:
        if not self._ensure_client():
            return {}
        try:
            bal = await self._client.fetch_balance()  # type: ignore[attr-defined]
            return {k: float(v) for k, v in bal.get("free", {}).items()}
        except Exception as e:
            logger.warning(f"[KRAKEN_ADAPTER] fetch_balance failed: {type(e).__name__}: {e}")
            return {}

    async def fetch_positions(self) -> List[Dict[str, Any]]:
        if not self._ensure_client():
            return []
        try:
            return await self._client.fetch_positions()  # type: ignore[attr-defined]
        except Exception as e:
            logger.warning(f"[KRAKEN_ADAPTER] fetch_positions failed: {type(e).__name__}: {e}")
            return []

    async def fetch_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        if not self._ensure_client():
            return []
        try:
            return await self._client.fetch_open_orders(symbol)  # type: ignore[attr-defined]
        except Exception as e:
            logger.warning(f"[KRAKEN_ADAPTER] fetch_open_orders failed: {type(e).__name__}: {e}")
            return []

    async def fetch_funding_rate(self, symbol: str) -> FundingRateData:
        # Existing implementation lives in data_mgmt/feeds/kraken_futures_feed.py.
        # This abstraction caller should normally use the feed directly during
        # dual-venue. After Phase 2 cutover the feed and adapter merge.
        raise RuntimeError(
            "use data_mgmt/feeds/kraken_futures_feed.py during dual-venue; "
            "this method is for post-cutover unification"
        )

    async def fetch_orderbook(
        self, symbol: str, depth: int = 25
    ) -> Dict[str, List[List[float]]]:
        if not self._ensure_client():
            return {"bids": [], "asks": []}
        try:
            ob = await self._client.fetch_order_book(symbol, depth)  # type: ignore[attr-defined]
            return {"bids": ob.get("bids", []), "asks": ob.get("asks", [])}
        except Exception as e:
            logger.warning(f"[KRAKEN_ADAPTER] fetch_orderbook failed: {type(e).__name__}: {e}")
            return {"bids": [], "asks": []}
