"""[P383] Urgent sleeve exits are MARKET orders, not 0.2%-through GTC limits.

Defect: `CoinbaseSleeve.execute_target(..., urgent=True)` — the FORCE_FLAT kill
switch and the FastRiskTick EXIT_ONLY/REDUCE_50 watchdog — skipped the maker
ladder (P270) but still placed the cross as a GTC LIMIT 0.2% through the
decision mid. In a fast market (the very condition the watchdog fires in) a
limit through a stale mid can be LEFT RESTING UNFILLED, while execute_target
has already SWEPT the protective stop. An emergency exit that can rest
unfilled beside a swept stop is not an exit; P382's stop follow-up bounds
the stranded window, it does not remove it.

Pinned here:
  1. urgent -> the adapter receives a MARKET OrderRequest with NO price, the
     right side and size (= contracts * contract_size), post_only False, and
     the order type is the ONE named decision `URGENT_ORDER_TYPE`.
  2. non-urgent is byte-identical: LIMIT, price = mid * 1.002 (BUY) /
     mid * 0.998 (SELL), exact multiplier pinned.
  3. urgent still REFUSES on a stale reconcile, an unverifiable resting-order
     sweep, and an unknown contract size — no guard was loosened.
  4. the fill-quality row for an urgent leg carries liquidity="market_urgent".
  5. an unreadable decision mid blocks a non-urgent LIMIT (P253, unchanged)
     and does NOT block the urgent MARKET; the ledger then records mid=None.
  6. the adapter's MARKET branch sends base_size = size/contract_size
     CONTRACTS through the SDK's `market_order` (the method that exists on
     coinbase-advanced-py 1.8.3), parses the response through
     `_parse_order_response`, and never sends reduce_only.

Falsification (run by hand, recorded in the P383 entry): make the urgent
branch fall back to the LIMIT request -> tests 1/4/5 go red; restore.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
import types

import pytest

from exchange.adapter import OrderRequest
from exchange.coinbase_adapter import CoinbaseAdapter
from exchange.coinbase_sleeve import CoinbaseSleeve

PID = "SLP-20DEC30-CDE"
CS = 5.0          # SOL nano contract = 5 SOL
MID = 100.0


# ---------------------------------------------------------------------------
# sleeve harness (the P290 shape: object.__new__ + stubbed collaborators)
# ---------------------------------------------------------------------------

class _FakeAdapter:
    def __init__(self, mid="100.0", order_payload=None, contract_size=CS,
                 list_raises=False):
        self._cs = contract_size
        self.placed = []
        self.cancelled = []
        self._open = []
        self.list_raises = list_raises
        self.order_payload = order_payload
        self.product = {"mid_market_price": mid}
        self._client = types.SimpleNamespace(
            get_product=lambda product_id: self.product,
            get_product_book=lambda product_id, limit: {
                "pricebook": {"bids": [{"price": "99.9"}],
                              "asks": [{"price": "100.1"}]}},
            get_best_bid_ask=lambda product_ids: {"pricebooks": [{
                "bids": [{"price": "99.9"}], "asks": [{"price": "100.1"}]}]},
        )

    def is_connected(self):
        return True

    def to_venue_symbol(self, asset, market="perp"):
        return PID

    def _contract_size(self, pid):
        return self._cs

    async def fetch_open_orders(self, symbol=None):
        if self.list_raises:
            raise RuntimeError("venue 502")
        return list(self._open)

    async def cancel_order(self, oid, pid):
        self.cancelled.append(oid)
        self._open = [o for o in self._open if o.get("order_id") != oid]
        return True

    async def get_order(self, order_id):
        return dict(self.order_payload or {})

    async def place_order(self, req):
        self.placed.append(req)
        oid = f"oid-{len(self.placed)}"
        return types.SimpleNamespace(success=True, order_id=oid,
                                     error_code=None, error_message=None)


def _sleeve(adapter, tmp_path, monkeypatch, cur=1.0, maker_first=False,
            reconcile_ok=True, filled_after=0.0):
    monkeypatch.setenv("HMATS_DATA_DIR", str(tmp_path))
    s = object.__new__(CoinbaseSleeve)
    s._adapter = adapter
    s._maker_first = maker_first
    s._maker_wait_sec = 0.01
    s._reconcile_ok = reconcile_ok
    s._halted = False
    s._halt_reason = ""
    s._max_contracts_per_asset = 5
    s._max_net_exposure = None
    s._max_asset_exposure = {}
    state = {"cur": cur}
    s.signed_contracts = lambda asset: state["cur"]  # type: ignore[assignment]
    s.is_ready = lambda: True  # type: ignore[assignment]

    def _reconcile():
        if adapter.placed:
            state["cur"] = filled_after
        return {}
    s.reconcile_positions = _reconcile  # type: ignore[assignment]
    s.can_trade = lambda a, d: (True, "ok")  # type: ignore[assignment]
    return s


def _rows(tmp_path):
    p = tmp_path / "fill_quality.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()
            if l.strip()]


FILLED = {"status": "FILLED", "average_filled_price": "99.8",
          "filled_size": "1", "total_fees": "0.42"}


class TestUrgentIsMarket:

    def test_the_decision_is_one_named_constant(self):
        assert CoinbaseSleeve.URGENT_ORDER_TYPE == "MARKET"

    def test_urgent_flatten_of_a_long_sends_a_market_sell_with_no_price(
            self, tmp_path, monkeypatch):
        ad = _FakeAdapter()
        s = _sleeve(ad, tmp_path, monkeypatch, cur=1.0)
        res = asyncio.run(s.execute_target("SOL", 0, urgent=True))
        assert res["status"] == "OK", res
        assert len(ad.placed) == 1
        req = ad.placed[0]
        assert req.order_type == CoinbaseSleeve.URGENT_ORDER_TYPE == "MARKET"
        assert req.price is None, "a MARKET order carries no limit price"
        assert req.side == "SELL"
        assert req.size == pytest.approx(1 * CS)   # base units; adapter -> contracts
        assert req.post_only is False

    def test_urgent_flatten_of_a_short_sends_a_market_buy(
            self, tmp_path, monkeypatch):
        ad = _FakeAdapter()
        s = _sleeve(ad, tmp_path, monkeypatch, cur=-2.0)
        res = asyncio.run(s.execute_target("SOL", 0, urgent=True))
        assert res["status"] == "OK", res
        req = ad.placed[0]
        assert (req.order_type, req.side, req.price) == ("MARKET", "BUY", None)
        assert req.size == pytest.approx(2 * CS)

    def test_urgent_never_touches_the_maker_ladder_even_when_armed(
            self, tmp_path, monkeypatch):
        ad = _FakeAdapter()
        s = _sleeve(ad, tmp_path, monkeypatch, cur=1.0, maker_first=True)
        res = asyncio.run(s.execute_target("SOL", 0, urgent=True))
        assert res["status"] == "OK", res
        assert [p.order_type for p in ad.placed] == ["MARKET"]
        assert all(not p.post_only for p in ad.placed)

    def test_urgent_fill_quality_row_is_market_urgent(
            self, tmp_path, monkeypatch):
        ad = _FakeAdapter(order_payload=FILLED)
        s = _sleeve(ad, tmp_path, monkeypatch, cur=1.0)
        asyncio.run(s.execute_target("SOL", 0, urgent=True))
        rows = _rows(tmp_path)
        assert len(rows) == 1
        r = rows[0]
        assert r["liquidity"] == "market_urgent"
        assert r["urgent"] is True
        assert r["decision_mid"] == pytest.approx(MID)
        assert r["status"] == "filled"


class TestNonUrgentIsByteIdentical:
    """The 0.2%-through GTC limit is what non-urgent callers must still get."""

    @pytest.mark.parametrize("cur,target,side,mult", [
        (0.0, 1, "BUY", 1.002),
        (1.0, 0, "SELL", 0.998),
        (0.0, -1, "SELL", 0.998),
        (-1.0, 0, "BUY", 1.002),
    ])
    def test_limit_0p2pct_through_the_mid(self, tmp_path, monkeypatch,
                                          cur, target, side, mult):
        ad = _FakeAdapter()
        s = _sleeve(ad, tmp_path, monkeypatch, cur=cur, filled_after=target)
        res = asyncio.run(s.execute_target("SOL", target))
        assert res["status"] == "OK", res
        req = ad.placed[0]
        assert req.order_type == "LIMIT"
        assert req.side == side
        assert req.price == pytest.approx(MID * mult, rel=0, abs=1e-12)
        assert req.post_only is False

    def test_non_urgent_fill_quality_row_is_unchanged(
            self, tmp_path, monkeypatch):
        ad = _FakeAdapter(order_payload=FILLED)
        s = _sleeve(ad, tmp_path, monkeypatch, cur=1.0)
        asyncio.run(s.execute_target("SOL", 0))
        r = _rows(tmp_path)[0]
        assert r["liquidity"] == "direct"   # maker_first off, not urgent
        assert r["urgent"] is False

    def test_the_constant_is_not_the_non_urgent_type(self):
        """If someone 'simplifies' by routing every cross through the
        constant, this is the pin that says the two paths are different."""
        src = inspect.getsource(CoinbaseSleeve.execute_target)
        assert re.search(r"if urgent:\s*\n(?:.*\n)*?\s*order_type=self\.URGENT_ORDER_TYPE",
                         src), "the urgent branch no longer uses URGENT_ORDER_TYPE"
        assert "px = mid * (1.002 if side == \"BUY\" else 0.998)" in src, (
            "the non-urgent 0.2%-through limit price was changed")


class TestUrgentStillRefuses:
    """Every refusal applies to urgent too — no guard was loosened."""

    def test_stale_reconcile(self, tmp_path, monkeypatch):
        ad = _FakeAdapter()
        s = _sleeve(ad, tmp_path, monkeypatch, cur=1.0, reconcile_ok=False)
        res = asyncio.run(s.execute_target("SOL", 0, urgent=True))
        assert res["status"] == "SKIPPED_STALE"
        assert ad.placed == []

    def test_unverifiable_resting_order_sweep(self, tmp_path, monkeypatch):
        ad = _FakeAdapter(list_raises=True)
        s = _sleeve(ad, tmp_path, monkeypatch, cur=1.0)
        res = asyncio.run(s.execute_target("SOL", 0, urgent=True))
        assert res["status"] == "SKIPPED_STALE"
        assert res["reason"] == "resting_orders_unverifiable"
        assert ad.placed == []

    def test_unknown_contract_size(self, tmp_path, monkeypatch):
        ad = _FakeAdapter(contract_size=None)
        s = _sleeve(ad, tmp_path, monkeypatch, cur=1.0)
        res = asyncio.run(s.execute_target("SOL", 0, urgent=True))
        assert res["status"] == "ERROR"
        assert res["reason"].startswith("no_contract_size")
        assert ad.placed == []

    def test_can_trade_block_still_binds(self, tmp_path, monkeypatch):
        ad = _FakeAdapter()
        s = _sleeve(ad, tmp_path, monkeypatch, cur=1.0)
        s.can_trade = lambda a, d: (False, "halted")  # type: ignore[assignment]
        res = asyncio.run(s.execute_target("SOL", 0, urgent=True))
        assert res["status"] == "BLOCKED"
        assert ad.placed == []


class TestUnreadableMid:
    """P253's priceless-order guard protects a LIMIT. A MARKET has no price to
    be ~0, so an unreadable mid must not block the emergency exit — but the
    ledger must record the mid as ABSENT, never a fabricated 0.0."""

    @pytest.mark.parametrize("bad_mid", ["0", None, "nan"])
    def test_non_urgent_still_refuses(self, tmp_path, monkeypatch, bad_mid):
        ad = _FakeAdapter(mid=bad_mid)
        s = _sleeve(ad, tmp_path, monkeypatch, cur=1.0)
        res = asyncio.run(s.execute_target("SOL", 0))
        assert res["status"] == "ERROR"
        assert res["reason"].startswith("no_price")
        assert ad.placed == []

    @pytest.mark.parametrize("bad_mid", ["0", None, "nan"])
    def test_urgent_market_proceeds_and_records_mid_absent(
            self, tmp_path, monkeypatch, bad_mid):
        ad = _FakeAdapter(mid=bad_mid, order_payload=FILLED)
        s = _sleeve(ad, tmp_path, monkeypatch, cur=1.0)
        res = asyncio.run(s.execute_target("SOL", 0, urgent=True))
        assert res["status"] == "OK", res
        assert ad.placed[0].order_type == "MARKET"
        r = _rows(tmp_path)[0]
        assert r["liquidity"] == "market_urgent"
        assert r["decision_mid"] is None
        assert r["realized_slippage_bps"] is None  # no mid -> no slippage claim


# ---------------------------------------------------------------------------
# adapter: the MARKET branch (reuses the protective-stop test's fake-client
# pattern — records which SDK method was hit and with what)
# ---------------------------------------------------------------------------

class _FakeClient:
    def __init__(self):
        self.calls = []

    def market_order(self, **kw):
        self.calls.append(("market", kw))
        return {"success": True, "success_response": {"order_id": "M1"}}

    def limit_order_gtc(self, **kw):
        self.calls.append(("limit", kw))
        return {"success": True, "success_response": {"order_id": "L1"}}


def _adapter(contract_size=CS):
    a = object.__new__(CoinbaseAdapter)
    a._client = _FakeClient()
    a._contract_size_cache = {PID: contract_size}
    a._price_increment_cache = {PID: 0.01}
    a._init_failed = None
    return a


def _place(adapter, **kw):
    return asyncio.run(adapter.place_order(OrderRequest(**kw)))


class TestAdapterMarketBranch:

    def test_sends_contracts_not_base_units_through_market_order(self):
        a = _adapter()
        res = _place(a, symbol=PID, side="SELL", size=2 * CS,
                     order_type="MARKET", price=None, post_only=False)
        assert res.success and res.order_id == "M1"
        assert [c[0] for c in a._client.calls] == ["market"]
        kw = a._client.calls[0][1]
        assert kw["base_size"] == "2", "base_size must be the CONTRACT count (P195)"
        assert kw["side"] == "SELL"
        assert kw["product_id"] == PID
        assert kw["client_order_id"]
        assert "limit_price" not in kw and "price" not in kw
        assert "reduce_only" not in kw, "CDE rejects reduce_only; never sent"
        assert "quote_size" not in kw, "perps size by base (contracts), never quote"

    def test_buy_side_is_passed_through(self):
        a = _adapter()
        _place(a, symbol=PID, side="BUY", size=1 * CS,
               order_type="MARKET", price=None, post_only=False)
        assert a._client.calls[0][1]["side"] == "BUY"

    def test_sub_contract_size_is_refused_before_the_sdk(self):
        a = _adapter()
        res = _place(a, symbol=PID, side="SELL", size=0.4 * CS,
                     order_type="MARKET", price=None, post_only=False)
        assert not res.success and res.error_code == "BELOW_MIN_CONTRACT"
        assert a._client.calls == []

    def test_market_does_not_need_a_price_but_limit_still_does(self):
        a = _adapter()
        res = _place(a, symbol=PID, side="SELL", size=CS,
                     order_type="LIMIT", price=None, post_only=False)
        assert not res.success and res.error_code == "NO_PRICE"
        assert a._client.calls == []

    def test_the_sdk_method_name_exists_on_the_installed_client(self):
        """coinbase-advanced-py 1.8.3 RESTClient: `market_order(client_order_id,
        product_id, side, quote_size=None, base_size=None, ..., leverage=None)`.
        A renamed SDK method would make the urgent exit raise SDK_CALL_FAILED
        on every call — the emergency path failing loudly, but failing."""
        try:
            from coinbase.rest import RESTClient
        except Exception:
            pytest.skip("coinbase-advanced-py not installed here")
        assert hasattr(RESTClient, "market_order")
        sig = inspect.signature(RESTClient.market_order)
        for p in ("client_order_id", "product_id", "side", "base_size",
                  "leverage"):
            assert p in sig.parameters, f"market_order lost kwarg {p}"

    def test_market_branch_is_routed_to_parse_order_response(self):
        """A failure response must come back as a real OrderResult failure,
        i.e. the MARKET branch shares the parser with every other branch."""
        a = _adapter()

        def _fail(**kw):
            a._client.calls.append(("market", kw))
            return {"success": False,
                    "error_response": {"error": "INSUFFICIENT_FUND",
                                       "message": "no buying power"}}
        a._client.market_order = _fail
        res = _place(a, symbol=PID, side="SELL", size=CS,
                     order_type="MARKET", price=None, post_only=False)
        assert res.success is False
        assert res.error_code not in (None, "", "SDK_CALL_FAILED")
