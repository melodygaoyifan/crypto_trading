"""
HMATS v5.1 Phase 2 - Coinbase US Perpetual-Style Futures Adapter
================================================================

Real implementation over the official `coinbase-advanced-py` SDK (RESTClient),
which handles CDP (ES256 JWT) auth + endpoint signing. Targets Coinbase US
Perpetual-Style Futures (CFTC) via the Advanced Trade API.

STILL PENDING before live use (see docs/COINBASE_MIGRATION_PREP.md):
  * exact perp `product_id`s — run scripts/coinbase_probe.py, then finalize
    exchange/symbol_mapping.py (coinbase/perp). Until then to_venue_symbol
    returns the placeholder `BTC-PERP`.
  * USDC funded into the perps portfolio.
  * credentials: .coinbase_key.json (CDP key file) or COINBASE_API_KEY/SECRET.

Fail-closed (Iron Law 4): with no client/creds, place/cancel return failure,
balance/positions/orderbook return empty, funding raises. Maker-first (Iron
Law 9): OrderRequest.post_only defaults True.

NOTE: the SDK is synchronous; these async methods call it inline. For shadow +
low-frequency 4H trading that is acceptable; a later pass can offload to a
thread executor if needed.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Dict, List, Optional

from exchange.adapter import (
    ExchangeAdapter,
    FundingRateData,
    OrderRequest,
    OrderResult,
)
from exchange.symbol_mapping import to_venue_symbol, from_venue_symbol

logger = logging.getLogger(__name__)

_DEFAULT_KEY_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".coinbase_key.json"
)


def _attr(obj: Any, key: str, default: Any = None) -> Any:
    """Read `key` from an SDK response that may be an object or a dict."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class CoinbaseAdapter(ExchangeAdapter):
    """Coinbase US Perpetual-Style Futures adapter (Advanced Trade)."""

    def __init__(
        self,
        rest_client: Optional[Any] = None,
        key_file: Optional[str] = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        paper: bool = True,
    ) -> None:
        self._client = rest_client
        self._key_file = key_file
        self._api_key = api_key or os.environ.get("COINBASE_API_KEY")
        self._api_secret = api_secret or os.environ.get("COINBASE_API_SECRET")
        self._paper = paper
        self._init_failed: Optional[str] = None
        self._contract_size_cache: Dict[str, float] = {}
        self._price_increment_cache: Dict[str, float] = {}

    # CDE nano contract sizes (probe 2026-06-13); fallback if get_product fails.
    _CONTRACT_SIZE_FALLBACK = {
        "BIP-20DEC30-CDE": 0.01,   # BTC
        "ETP-20DEC30-CDE": 0.1,    # ETH
        "SLP-20DEC30-CDE": 5.0,    # SOL
    }
    # CDE price tick size (probe 2026-06-13); limit prices must be a multiple.
    _PRICE_INCREMENT_FALLBACK = {
        "BIP-20DEC30-CDE": 5.0,    # BTC
        "ETP-20DEC30-CDE": 0.5,    # ETH
        "SLP-20DEC30-CDE": 0.01,   # SOL
    }

    def _price_increment(self, product_id: str) -> Optional[float]:
        """Tick size for limit prices. Cached; fallback table if fetch fails."""
        if product_id in self._price_increment_cache:
            return self._price_increment_cache[product_id]
        inc = None
        try:
            p = self._client.get_product(product_id=product_id)
            raw = _attr(p, "price_increment")
            if raw is not None:
                inc = float(raw)
        except Exception:
            pass
        if not inc or inc <= 0:
            inc = self._PRICE_INCREMENT_FALLBACK.get(product_id)
        if inc:
            self._price_increment_cache[product_id] = inc
        return inc

    def _round_to_tick(self, product_id: str, price: float) -> float:
        inc = self._price_increment(product_id)
        if not inc or inc <= 0:
            return price
        return round(round(price / inc) * inc, 8)

    def _contract_size(self, product_id: str) -> Optional[float]:
        """Base-asset units per contract for a CDE futures product (e.g. BTC
        0.01). Cached. Returns None for non-contract products (spot)."""
        if product_id in self._contract_size_cache:
            return self._contract_size_cache[product_id]
        cs = None
        try:
            p = self._client.get_product(product_id=product_id)
            fpd = _attr(p, "future_product_details") or {}
            raw = _attr(fpd, "contract_size")
            if raw is not None:
                cs = float(raw)
        except Exception as e:
            logger.warning(f"[COINBASE] contract_size fetch failed for "
                           f"{product_id}: {type(e).__name__}: {e}")
        if not cs or cs <= 0:
            cs = self._CONTRACT_SIZE_FALLBACK.get(product_id)
        if cs:
            self._contract_size_cache[product_id] = cs
        return cs

    @property
    def venue(self) -> str:
        return "coinbase"

    # ----- client lifecycle ------------------------------------------------

    def _ensure_client(self) -> bool:
        if self._client is not None:
            return True
        if self._init_failed is not None:
            return False
        try:
            from coinbase.rest import RESTClient  # type: ignore
        except Exception as e:
            self._init_failed = f"sdk_missing:{type(e).__name__}"
            logger.warning("[COINBASE] coinbase-advanced-py not installed; adapter disabled")
            return False
        kf = self._key_file or (os.environ.get("COINBASE_KEY_FILE"))
        if not kf and os.path.exists(_DEFAULT_KEY_FILE):
            kf = _DEFAULT_KEY_FILE
        try:
            if kf and os.path.exists(kf):
                self._client = RESTClient(key_file=kf)
            elif self._api_key and self._api_secret:
                self._client = RESTClient(api_key=self._api_key, api_secret=self._api_secret)
            else:
                self._init_failed = "no_credentials"
                return False
            return True
        except Exception as e:
            self._init_failed = f"{type(e).__name__}:{e}"
            logger.warning(f"[COINBASE] client init failed: {self._init_failed}")
            return False

    def is_connected(self) -> bool:
        return self._ensure_client()

    def to_venue_symbol(self, hmats_asset: str, market: str = "perp") -> str:
        return to_venue_symbol(hmats_asset, "coinbase", market)

    def from_venue_symbol(self, venue_symbol: str, market: str = "perp") -> str:
        return from_venue_symbol(venue_symbol, "coinbase", market)

    # ----- orders ----------------------------------------------------------

    async def place_order(self, request: OrderRequest) -> OrderResult:
        if not self._ensure_client():
            return OrderResult(
                success=False, venue="coinbase",
                error_code="NOT_CONFIGURED",
                error_message=("CoinbaseAdapter not configured: "
                               f"{self._init_failed or 'no client'}. "
                               "Provide .coinbase_key.json or COINBASE_API_KEY/SECRET."),
            )
        coid = request.client_order_id or uuid.uuid4().hex
        side = request.side.upper()
        product_id = request.symbol
        lev = str(int(request.leverage)) if request.leverage and request.leverage > 1 else None
        # CDE futures trade in WHOLE contracts (base_increment=1), each worth
        # `contract_size` base units (BTC 0.01 / ETH 0.1 / SOL 5.0). Convert the
        # HMATS base-asset exposure -> integer contract count.
        size_str = str(request.size)
        cs = self._contract_size(product_id)
        if cs and cs > 0:
            contracts = int(round(request.size / cs))
            if contracts < 1:
                return OrderResult(
                    success=False, venue="coinbase", client_order_id=coid,
                    error_code="BELOW_MIN_CONTRACT",
                    error_message=(f"size {request.size} on {product_id} rounds to "
                                   f"0 contracts (1 contract = {cs} base units)"))
            size_str = str(contracts)
        # NOTE: CDE rejects `reduce_only` ("unknown field" — confirmed via live
        # test 2026-06-13). Futures positions net naturally, so a close is just
        # an opposite-side order of the position size; reduce_only is not sent.
        try:
            if request.order_type.upper() == "MARKET":
                resp = self._client.market_order(
                    client_order_id=coid, product_id=product_id, side=side,
                    base_size=size_str, leverage=lev,
                )
            else:  # LIMIT / POST — maker-first
                if request.price is None:
                    return OrderResult(success=False, venue="coinbase",
                                       error_code="NO_PRICE",
                                       error_message="LIMIT order requires price")
                # round to the product tick (CDE rejects off-tick prices).
                # Returns the value unchanged when no tick is known.
                px = self._round_to_tick(product_id, request.price)
                resp = self._client.limit_order_gtc(
                    client_order_id=coid, product_id=product_id, side=side,
                    base_size=size_str, limit_price=str(px),
                    post_only=bool(request.post_only), leverage=lev,
                )
            return self._parse_order_response(resp, coid)
        except Exception as e:
            return OrderResult(success=False, venue="coinbase",
                               error_code="SDK_CALL_FAILED",
                               error_message=f"{type(e).__name__}: {e}")

    def _parse_order_response(self, resp: Any, coid: str) -> OrderResult:
        success = bool(_attr(resp, "success", False))
        sr = _attr(resp, "success_response")
        er = _attr(resp, "error_response")
        order_id = _attr(sr, "order_id") if sr else _attr(resp, "order_id")
        if success:
            return OrderResult(
                success=True, venue="coinbase", order_id=order_id,
                client_order_id=coid, status="SUBMITTED",
                raw=_attr(resp, "__dict__") or None,
            )
        return OrderResult(
            success=False, venue="coinbase", client_order_id=coid,
            order_id=order_id, status="REJECTED",
            error_code=str(_attr(er, "error", "ORDER_REJECTED")),
            error_message=str(_attr(er, "message", er or resp)),
        )

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        if not self._ensure_client():
            return False
        try:
            resp = self._client.cancel_orders(order_ids=[order_id])
            results = _attr(resp, "results", []) or []
            for r in results:
                if str(_attr(r, "order_id")) == str(order_id):
                    return bool(_attr(r, "success", False))
            return bool(results)
        except Exception as e:
            logger.warning(f"[COINBASE] cancel_order failed: {type(e).__name__}: {e}")
            return False

    async def fetch_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        if not self._ensure_client():
            return []
        try:
            kwargs: Dict[str, Any] = {"order_status": ["OPEN"]}
            if symbol:
                kwargs["product_ids"] = [symbol]
            resp = self._client.list_orders(**kwargs)
            orders = _attr(resp, "orders", []) or []
            return [o if isinstance(o, dict) else _attr(o, "__dict__", {}) for o in orders]
        except Exception as e:
            logger.warning(f"[COINBASE] fetch_open_orders failed: {type(e).__name__}: {e}")
            return []

    # ----- account / positions --------------------------------------------

    async def fetch_balance(self) -> Dict[str, float]:
        if not self._ensure_client():
            return {}
        try:
            resp = self._client.get_accounts()
            out: Dict[str, float] = {}
            for ac in (_attr(resp, "accounts", []) or []):
                cur = _attr(ac, "currency")
                avail = _attr(ac, "available_balance") or {}
                val = _attr(avail, "value", "0")
                if cur is not None:
                    try:
                        out[str(cur)] = float(val or 0)
                    except (TypeError, ValueError):
                        pass
            return out
        except Exception as e:
            logger.warning(f"[COINBASE] fetch_balance failed: {type(e).__name__}: {e}")
            return {}

    async def fetch_positions(self) -> List[Dict[str, Any]]:
        if not self._ensure_client():
            return []
        # CDE (US FCM) uses list_futures_positions; list_perps_positions (INTX)
        # needs a portfolio_uuid and fails for CDE — try futures first.
        for method in ("list_futures_positions", "list_perps_positions"):
            fn = getattr(self._client, method, None)
            if fn is None:
                continue
            try:
                resp = fn()
                positions = (_attr(resp, "positions", None)
                             or _attr(resp, "perps_positions", None)
                             or _attr(resp, "futures_positions", None) or [])
                return [p if isinstance(p, dict) else _attr(p, "__dict__", {}) for p in positions]
            except Exception as e:
                logger.warning(f"[COINBASE] {method} failed: {type(e).__name__}: {e}")
        return []

    # ----- market data -----------------------------------------------------

    async def fetch_funding_rate(self, symbol: str) -> FundingRateData:
        if not self._ensure_client():
            raise RuntimeError("coinbase_funding_not_configured")
        try:
            prod = self._client.get_product(product_id=symbol)
            fpd = _attr(prod, "future_product_details") or _attr(prod, "perpetual_details") or {}
            rate = _attr(fpd, "funding_rate")
            if rate is None:
                raise RuntimeError(f"no funding_rate in product {symbol}")
            # CDE exposes the interval as a duration string, e.g. "3600s".
            # (Confirmed via live probe 2026-06-13.)
            period = 1.0
            interval = _attr(fpd, "funding_interval")
            if interval is not None:
                try:
                    period = float(str(interval).strip().rstrip("s")) / 3600.0
                except (TypeError, ValueError):
                    period = float(_attr(fpd, "funding_interval_hours", 1.0) or 1.0)
            rate = float(rate)
            return FundingRateData(
                symbol=symbol, funding_rate=rate,
                funding_period_hours=period or 1.0,
                funding_rate_8h=rate * (8.0 / period if period else 1.0),
                next_funding_ts=_attr(fpd, "funding_time"), venue="coinbase",
            )
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"coinbase_funding_fetch_failed: {type(e).__name__}: {e}")

    async def fetch_orderbook(
        self, symbol: str, depth: int = 25
    ) -> Dict[str, List[List[float]]]:
        if not self._ensure_client():
            return {"bids": [], "asks": []}
        try:
            resp = self._client.get_product_book(product_id=symbol, limit=depth)
            book = _attr(resp, "pricebook") or resp
            def _levels(side: str):
                out = []
                for lvl in (_attr(book, side, []) or []):
                    px = _attr(lvl, "price"); sz = _attr(lvl, "size")
                    if px is not None and sz is not None:
                        out.append([float(px), float(sz)])
                return out
            return {"bids": _levels("bids"), "asks": _levels("asks")}
        except Exception as e:
            logger.warning(f"[COINBASE] fetch_orderbook failed: {type(e).__name__}: {e}")
            return {"bids": [], "asks": []}
