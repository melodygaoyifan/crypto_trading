"""
HMATS v5.1 Phase 2 - CoinbaseAdapter logic tests (mock SDK client).

Verifies the real SDK-backed adapter maps OrderRequest -> SDK calls correctly
and parses responses, WITHOUT live credentials. A fake RESTClient records
calls and returns canned Advanced-Trade-shaped responses.
"""
import asyncio

from exchange.adapter import OrderRequest
from exchange.coinbase_adapter import CoinbaseAdapter


class FakeClient:
    def __init__(self, order_ok=True):
        self.calls = []
        self._order_ok = order_ok

    def limit_order_gtc(self, **kw):
        self.calls.append(("limit", kw))
        return self._order_resp("CB-LIM-1")

    def market_order(self, **kw):
        self.calls.append(("market", kw))
        return self._order_resp("CB-MKT-1")

    def _order_resp(self, oid):
        if self._order_ok:
            return {"success": True, "success_response": {"order_id": oid}}
        return {"success": False,
                "error_response": {"error": "INSUFFICIENT_FUND", "message": "no usdc"}}

    def cancel_orders(self, order_ids):
        self.calls.append(("cancel", order_ids))
        return {"results": [{"order_id": order_ids[0], "success": True}]}

    def get_accounts(self, **kw):
        return {"accounts": [
            {"currency": "USDC", "available_balance": {"value": "1234.5"}},
            {"currency": "BTC", "available_balance": {"value": "0"}},
        ]}

    def list_orders(self, **kw):
        self.calls.append(("list_orders", kw))
        return {"orders": [{"order_id": "O1", "product_id": "BTC-PERP"}]}

    def get_product_book(self, product_id, limit=None, **kw):
        return {"pricebook": {"bids": [{"price": "100", "size": "1"}],
                              "asks": [{"price": "101", "size": "2"}]}}


def _adapter(ok=True):
    return CoinbaseAdapter(rest_client=FakeClient(ok))


def test_connected_when_client_injected():
    assert _adapter().is_connected() is True


def test_place_limit_order_maps_args_and_parses_success():
    a = _adapter()
    req = OrderRequest(symbol="BTC-PERP", side="BUY", size=0.01,
                       order_type="LIMIT", price=63000, post_only=True, leverage=2)
    res = asyncio.run(a.place_order(req))
    assert res.success and res.venue == "coinbase" and res.order_id == "CB-LIM-1"
    kind, kw = a._client.calls[0]
    assert kind == "limit"
    assert kw["product_id"] == "BTC-PERP" and kw["side"] == "BUY"
    assert kw["base_size"] == "0.01" and kw["limit_price"] == "63000"
    assert kw["post_only"] is True and kw["leverage"] == "2"
    assert "reduce_only" not in kw  # only sent when True


def test_place_market_order_maps_args():
    a = _adapter()
    req = OrderRequest(symbol="ETH-PERP", side="SELL", size=0.1,
                       order_type="MARKET", leverage=3, reduce_only=True)
    res = asyncio.run(a.place_order(req))
    assert res.success and res.order_id == "CB-MKT-1"
    kind, kw = a._client.calls[0]
    assert kind == "market" and kw["base_size"] == "0.1" and kw["leverage"] == "3"
    assert kw.get("reduce_only") is True


def test_leverage_omitted_when_spot_like():
    a = _adapter()
    req = OrderRequest(symbol="BTC-PERP", side="BUY", size=0.01,
                       order_type="LIMIT", price=1, leverage=1)
    asyncio.run(a.place_order(req))
    _, kw = a._client.calls[0]
    assert kw["leverage"] is None  # leverage<=1 -> None


def test_limit_without_price_rejected():
    a = _adapter()
    req = OrderRequest(symbol="BTC-PERP", side="BUY", size=0.01,
                       order_type="LIMIT", price=None)
    res = asyncio.run(a.place_order(req))
    assert res.success is False and res.error_code == "NO_PRICE"


def test_order_rejection_parsed():
    a = _adapter(ok=False)
    req = OrderRequest(symbol="BTC-PERP", side="BUY", size=0.01,
                       order_type="MARKET")
    res = asyncio.run(a.place_order(req))
    assert res.success is False and res.status == "REJECTED"
    assert res.error_code == "INSUFFICIENT_FUND"


def test_fetch_balance_parses_usdc():
    bal = asyncio.run(_adapter().fetch_balance())
    assert bal["USDC"] == 1234.5 and bal["BTC"] == 0.0


def test_cancel_order_parses_result():
    assert asyncio.run(_adapter().cancel_order("O1", "BTC-PERP")) is True


def test_fetch_orderbook_parses_levels():
    ob = asyncio.run(_adapter().fetch_orderbook("BTC-PERP", depth=5))
    assert ob["bids"] == [[100.0, 1.0]] and ob["asks"] == [[101.0, 2.0]]


def test_open_orders_filters_by_symbol():
    a = _adapter()
    asyncio.run(a.fetch_open_orders("BTC-PERP"))
    kind, kw = a._client.calls[0]
    assert kind == "list_orders"
    assert kw["product_ids"] == ["BTC-PERP"] and kw["order_status"] == ["OPEN"]
