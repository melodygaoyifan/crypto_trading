"""
HMATS v5.1 Phase 2 - Coinbase Advanced Trade Adapter (SKELETON)
================================================================

Coinbase Advanced Trade perp adapter. Skeleton implementation — actual
HTTP calls are stubs that fail-closed when not configured. Operator
provides API credentials via env vars:
  COINBASE_API_KEY
  COINBASE_API_SECRET
  COINBASE_PASSPHRASE   (if applicable to the auth flow used)

Per V14 GREEN: Coinbase Crypto Perp = 0bps maker, 3bps taker (promotional).

Iron Laws:
  4. fail-closed: every method raises NotImplementedError or returns failure
     until operator wires real API client.
  9. maker-first: post_only defaults True in OrderRequest.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from exchange.adapter import (
    ExchangeAdapter,
    FundingRateData,
    OrderRequest,
    OrderResult,
)
from exchange.symbol_mapping import to_venue_symbol, from_venue_symbol

logger = logging.getLogger(__name__)


class CoinbaseAdapter(ExchangeAdapter):
    """Coinbase Advanced Trade adapter (perpetuals).

    SKELETON: HTTP/WS clients not yet wired. Methods raise or return
    explicit "not_configured" failures. Operator wires concrete client
    via Phase 2 cutover step 2.4.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        passphrase: Optional[str] = None,
        paper: bool = True,
    ) -> None:
        self._api_key = api_key or os.environ.get("COINBASE_API_KEY")
        self._api_secret = api_secret or os.environ.get("COINBASE_API_SECRET")
        self._passphrase = passphrase or os.environ.get("COINBASE_PASSPHRASE")
        self._paper = paper
        self._session: Any = None  # populated when concrete HTTP client wires

    @property
    def venue(self) -> str:
        return "coinbase"

    def is_connected(self) -> bool:
        # Skeleton: returns False until operator wires the real client.
        return self._session is not None

    def to_venue_symbol(self, hmats_asset: str, market: str = "perp") -> str:
        return to_venue_symbol(hmats_asset, "coinbase", market)

    def from_venue_symbol(self, venue_symbol: str, market: str = "perp") -> str:
        return from_venue_symbol(venue_symbol, "coinbase", market)

    async def place_order(self, request: OrderRequest) -> OrderResult:
        if not self.is_connected():
            return OrderResult(
                success=False, venue="coinbase",
                error_code="NOT_CONFIGURED",
                error_message=("CoinbaseAdapter skeleton: HTTP client not wired. "
                               "Operator: implement Phase 2 step 2.4 client init."),
            )
        # Real implementation calls Advanced Trade /api/v3/brokerage/orders
        raise NotImplementedError("CoinbaseAdapter.place_order skeleton")

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        if not self.is_connected():
            return False
        raise NotImplementedError("CoinbaseAdapter.cancel_order skeleton")

    async def fetch_balance(self) -> Dict[str, float]:
        if not self.is_connected():
            logger.warning("[COINBASE] fetch_balance: not configured; returning {}")
            return {}
        raise NotImplementedError

    async def fetch_positions(self) -> List[Dict[str, Any]]:
        if not self.is_connected():
            return []
        raise NotImplementedError

    async def fetch_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        if not self.is_connected():
            return []
        raise NotImplementedError

    async def fetch_funding_rate(self, symbol: str) -> FundingRateData:
        if not self.is_connected():
            # Fail-closed: caller catches and falls back to Kraken funding feed
            raise RuntimeError("coinbase_funding_not_configured")
        raise NotImplementedError

    async def fetch_orderbook(
        self, symbol: str, depth: int = 25
    ) -> Dict[str, List[List[float]]]:
        if not self.is_connected():
            return {"bids": [], "asks": []}
        raise NotImplementedError
