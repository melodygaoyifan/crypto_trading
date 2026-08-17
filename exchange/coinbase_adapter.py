"""
HMATS v5.1 Phase 2 - Coinbase US Perpetual-Style Futures Adapter
================================================================

Real implementation over the official `coinbase-advanced-py` SDK (RESTClient),
which handles CDP (ES256 JWT) auth + endpoint signing. Targets Coinbase US
Perpetual-Style Futures (CFTC) via the Advanced Trade API.

[STATUS 2026-08-07 — LIVE.] The "STILL PENDING before live use" list that used
to sit here was completed in June 2026 and the header was never updated (stale
docs cost diagnosis time — P155 layer 1). Resolved state: perp product_ids are
finalized in exchange/symbol_mapping.py (BTC=BIP-20DEC30-CDE etc., nano
contracts 0.01/0.1/5.0); the Default portfolio's ~$4K USDC cross-collateralizes
the perps (P153); the trade key lives at /opt/hmats/data/.coinbase_key.json
(`COINBASE_KEY_FILE`). This adapter places real orders every 4H tick via the
sleeve (market/limit/STOP branches — the STOP branch is [P197]).

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


def _plain(obj: Any, _depth: int = 0) -> Any:
    """[P197-fix] Recursively convert an SDK response into plain dicts/lists.

    The SDK returns nested model objects (`Order.order_configuration` is an
    `OrderConfiguration`, not a dict). A one-level `__dict__` conversion leaves
    those nested objects intact, so a caller doing `isinstance(cfg, dict)` gets
    False and concludes the field is absent — which is how a stop-limit that was
    demonstrably resting at the venue read as "no stop found".

    Depth-bounded so a self-referential SDK object cannot spin.
    """
    if _depth > 6:
        return obj
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {k: _plain(v, _depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v, _depth + 1) for v in obj]
    inner = getattr(obj, "__dict__", None)
    if isinstance(inner, dict):
        return {k: _plain(v, _depth + 1) for k, v in inner.items()}
    return obj


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
            elif request.order_type.upper() in ("STOP", "STOP_LIMIT"):
                # [P197] Protective stop. Before this branch existed,
                # order_type="STOP" fell through to the LIMIT path below and
                # silently placed a plain GTC limit at `price`, ignoring
                # `stop_price` entirely — an order that looks like protection in
                # the code and is not one at the venue. OrderRequest has carried
                # an unused `stop_price` field since exchange/adapter.py:48.
                if request.stop_price is None:
                    return OrderResult(
                        success=False, venue="coinbase", client_order_id=coid,
                        error_code="NO_STOP_PRICE",
                        error_message="STOP order requires stop_price")
                stop_px = self._round_to_tick(product_id, request.stop_price)
                # Limit price sits THROUGH the stop so the order actually fills
                # once triggered; a limit exactly at the stop can be left behind
                # by a fast move. Caller may override via request.price.
                _lim = (request.price if request.price is not None
                        else request.stop_price * (0.995 if side == "SELL" else 1.005))
                lim_px = self._round_to_tick(product_id, _lim)
                # SELL stop protects a LONG -> triggers on the way DOWN.
                # BUY  stop protects a SHORT -> triggers on the way UP.
                direction = ("STOP_DIRECTION_STOP_DOWN" if side == "SELL"
                             else "STOP_DIRECTION_STOP_UP")
                # getattr, not attribute access: `self._client` is Optional, and
                # every direct `self._client.x` in this file is an existing
                # baselined union-attr finding. New code must not add more.
                fn = getattr(self._client,
                             "stop_limit_order_gtc_sell" if side == "SELL"
                             else "stop_limit_order_gtc_buy")
                resp = fn(
                    client_order_id=coid, product_id=product_id,
                    base_size=size_str, limit_price=str(lim_px),
                    stop_price=str(stop_px), stop_direction=direction,
                    leverage=lev,
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
            # [P287] UNCONFIRMED = FAILED. The old fallback `return
            # bool(results)` reported success whenever the batch response
            # carried ANY row — including a row for a different order, a row
            # whose order_id key was absent/renamed by an SDK bump, or a row
            # that explicitly said success: False. Every P265 failed-cancel
            # refusal (stop replacement, the maker CANCEL_FAILED no-cross
            # guard, the flat-path orphan cancel) keys off this boolean, so a
            # false True converted each refusal into the exact double-stop /
            # double-order it exists to prevent. Only an explicitly matched
            # row may certify the cancel.
            logger.warning(
                f"[COINBASE] cancel_order {order_id}: response carried no "
                f"matching result row ({len(results)} row(s)) — treating the "
                f"cancel as FAILED (unconfirmed != cancelled, P287). "
                f"rows={results!r}"[:500])
            return False
        except Exception as e:
            logger.warning(f"[COINBASE] cancel_order failed: {type(e).__name__}: {e}")
            return False

    async def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """[P290] Read-only single-order lookup for the fill-quality ledger.

        Returns the order as a plain dict, or None on ANY failure with a
        warning. Deliberately the OPPOSITE fail semantics of
        fetch_open_orders' post-P287 raise: that one protects order-path
        GUARDS (an unreadable book must refuse actions); this one feeds only
        an observation ledger, and a read failure must never ripple into the
        order path — the recorder writes status="unresolved" instead
        (absence stays absent, P2; never a fabricated fill).
        """
        client = self._client if self._ensure_client() else None
        if client is None:
            return None
        try:
            resp = client.get_order(order_id=order_id)
            order = _attr(resp, "order") or resp
            plain = _plain(order)
            return plain if isinstance(plain, dict) else None
        except Exception as e:
            logger.warning(f"[COINBASE] get_order {order_id} failed: "
                           f"{type(e).__name__}: {e} — fill quality will "
                           f"record UNRESOLVED")
            return None

    async def fetch_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """List OPEN orders. RAISES on any listing failure (P287).

        [P287] This used to swallow every error into `[]`, which made "could
        not read the book" byte-identical to "the book is clean" (the
        P159/P171 shape, on the live order path). Four sleeve guards fail-OPEN
        on that conflation: `ensure_protective_stop` sees `resting=[]` and
        places a SECOND stop beside the invisible real one (two plain stops on
        a no-reduce_only venue — the second OPENS the opposite side when
        touched); the maker poll reads `still_open=False` and logs "filled at
        0bps" for an order that is still resting; and both stale-order sweeps
        report a clean book they never saw. The sleeve's callers were all
        written expecting a raise (their try/excepts were dead code against
        the swallowing version). NOT-CONFIGURED still returns [] — no
        credentials means nothing of ours can be resting, and is_ready() gates
        every sleeve path before this call anyway.
        """
        if not self._ensure_client():
            return []
        try:
            kwargs: Dict[str, Any] = {"order_status": ["OPEN"]}
            if symbol:
                kwargs["product_ids"] = [symbol]
            resp = self._client.list_orders(**kwargs)
            orders = _attr(resp, "orders", []) or []
            # [P197-fix] DEEP-convert to plain dicts. The old one-level
            # `o.__dict__` left NESTED SDK objects intact: an Order's
            # `order_configuration` stays an `OrderConfiguration`, not a dict.
            # Callers that reasonably did `isinstance(cfg, dict)` then saw False
            # and concluded "no stop here" — for a stop that was demonstrably
            # resting at the venue. Normalising at the boundary is the fix;
            # every consumer downstream gets ordinary dicts.
            return [_plain(o) for o in orders]
        except Exception as e:
            logger.warning(f"[COINBASE] fetch_open_orders failed: {type(e).__name__}: {e}")
            raise

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
