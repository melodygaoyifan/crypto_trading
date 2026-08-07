"""[P197] Server-side protective stop on the Coinbase perp sleeve.

Before this, the sleeve had NO venue-resting protection: every exit was a
client-side API call on the 4H tick, so a dead process left BTC/ETH/SOL perp
exposure with nothing at the venue to close it. Preview-verified 2026-08-07 that
CDE accepts stop-limits on all three contracts (errs: [], order_margin_total = 0,
i.e. the venue treats them as position-REDUCING).

TWO HAZARDS THESE TESTS EXIST TO PIN
------------------------------------
1. **The adapter had no STOP branch.** `order_type="STOP"` fell through the
   `else` and silently placed a plain GTC limit at `price`, ignoring
   `stop_price` — an order that reads as protection in the code and is not one
   at the venue. `OrderRequest.stop_price` had been carried unused since
   exchange/adapter.py:48.
2. **CDE rejects `reduce_only`** (coinbase_adapter.py:206). A resting stop is
   therefore a PLAIN order: if the position it guards is gone and the stop is
   still live, triggering it OPENS an opposite position. So the stop must be
   reconciled every tick and cancelled the moment the asset is flat.

The feature is OFF by default (`protective_stop_pct <= 0`), because enabling it
places REAL resting orders on a live account — P141: activation is a deliberate,
operator-watched step.
"""

import asyncio
import types

import pytest

from exchange.adapter import OrderRequest
from exchange.coinbase_adapter import CoinbaseAdapter
from exchange.coinbase_sleeve import CoinbaseSleeve

PID = "SLP-20DEC30-CDE"


# ---------------------------------------------------------------------------
# adapter: the STOP branch
# ---------------------------------------------------------------------------

class _FakeClient:
    """Records which SDK order method was called, and with what."""

    def __init__(self):
        self.calls = []

    def stop_limit_order_gtc_sell(self, **kw):
        self.calls.append(("stop_sell", kw))
        return {"success": True, "success_response": {"order_id": "S1"}}

    def stop_limit_order_gtc_buy(self, **kw):
        self.calls.append(("stop_buy", kw))
        return {"success": True, "success_response": {"order_id": "S2"}}

    def limit_order_gtc(self, **kw):
        self.calls.append(("limit", kw))
        return {"success": True, "success_response": {"order_id": "L1"}}

    def market_order(self, **kw):
        self.calls.append(("market", kw))
        return {"success": True, "success_response": {"order_id": "M1"}}


def _adapter(tick=0.01, contract_size=5.0):
    a = object.__new__(CoinbaseAdapter)
    a._client = _FakeClient()
    a._contract_size_cache = {PID: contract_size}
    a._price_increment_cache = {PID: tick}
    a._init_failed = None
    return a


def _place(adapter, **kw):
    return asyncio.run(adapter.place_order(OrderRequest(**kw)))


class TestAdapterStopBranch:

    def test_stop_order_calls_the_stop_endpoint_not_a_plain_limit(self):
        """The exact bug: order_type='STOP' used to become a GTC limit."""
        a = _adapter()
        res = _place(a, symbol=PID, side="SELL", size=5.0,
                     order_type="STOP", stop_price=60.0)
        assert res.success
        kinds = [c[0] for c in a._client.calls]
        assert kinds == ["stop_sell"], (
            f"expected the stop endpoint, got {kinds}. order_type='STOP' falling "
            f"through to limit_order_gtc is the silent-downgrade bug: the code "
            f"reads as protection, the venue holds an ordinary limit."
        )

    def test_the_stop_price_actually_reaches_the_venue(self):
        a = _adapter()
        _place(a, symbol=PID, side="SELL", size=5.0, order_type="STOP",
               stop_price=60.0)
        kw = a._client.calls[0][1]
        assert kw["stop_price"] == "60.0", kw

    def test_sell_stop_triggers_downward_and_buy_stop_upward(self):
        """A SELL stop protects a LONG (fires as price falls), and vice versa.

        Inverting this gives an order that can only trigger in the direction that
        was never a risk.
        """
        a = _adapter()
        _place(a, symbol=PID, side="SELL", size=5.0, order_type="STOP", stop_price=60.0)
        assert a._client.calls[0][1]["stop_direction"] == "STOP_DIRECTION_STOP_DOWN"

        b = _adapter()
        _place(b, symbol=PID, side="BUY", size=5.0, order_type="STOP", stop_price=90.0)
        assert b._client.calls[0][1]["stop_direction"] == "STOP_DIRECTION_STOP_UP"

    def test_limit_price_crosses_through_the_stop_so_it_can_fill(self):
        """A limit exactly at the stop can be left behind by a fast move."""
        a = _adapter()
        _place(a, symbol=PID, side="SELL", size=5.0, order_type="STOP", stop_price=60.0)
        kw = a._client.calls[0][1]
        assert float(kw["limit_price"]) < 60.0

        b = _adapter()
        _place(b, symbol=PID, side="BUY", size=5.0, order_type="STOP", stop_price=90.0)
        assert float(b._client.calls[0][1]["limit_price"]) > 90.0

    def test_prices_are_rounded_to_a_MULTIPLE_of_the_tick(self):
        """CDE rejects off-tick prices. BTC's tick is 5 — rounding to whole
        numbers is NOT the same as rounding to a multiple of 5, which is what
        made the capability probe report a false negative."""
        a = _adapter(tick=5.0, contract_size=0.01)
        _place(a, symbol=PID, side="SELL", size=0.01, order_type="STOP",
               stop_price=57816.0)
        kw = a._client.calls[0][1]
        assert float(kw["stop_price"]) % 5 == 0, kw["stop_price"]
        assert float(kw["limit_price"]) % 5 == 0, kw["limit_price"]

    def test_size_is_converted_to_whole_contracts(self):
        """base_size is in CONTRACTS at the venue (base_increment = 1)."""
        a = _adapter(contract_size=5.0)
        _place(a, symbol=PID, side="SELL", size=5.0, order_type="STOP", stop_price=60.0)
        assert a._client.calls[0][1]["base_size"] == "1"

    def test_a_stop_without_a_stop_price_is_refused_not_downgraded(self):
        a = _adapter()
        res = _place(a, symbol=PID, side="SELL", size=5.0, order_type="STOP")
        assert not res.success
        assert res.error_code == "NO_STOP_PRICE"
        assert a._client.calls == [], "refused order still hit the venue"

    def test_plain_limit_and_market_paths_are_untouched(self):
        a = _adapter()
        _place(a, symbol=PID, side="BUY", size=5.0, order_type="LIMIT", price=72.0)
        _place(a, symbol=PID, side="BUY", size=5.0, order_type="MARKET")
        assert [c[0] for c in a._client.calls] == ["limit", "market"]


# ---------------------------------------------------------------------------
# sleeve: reconcile-to-desired-state
# ---------------------------------------------------------------------------

class _FakeAdapter:
    def __init__(self, open_orders=None, contract_size=5.0, place_ok=True):
        self._open = list(open_orders or [])
        self._cs = contract_size
        self._place_ok = place_ok
        self.placed = []
        self.cancelled = []
        self._client = types.SimpleNamespace(
            get_product=lambda product_id: {"mid_market_price": "72.0"})

    def is_connected(self): return True
    def to_venue_symbol(self, asset, market="perp"): return PID
    def _contract_size(self, pid): return self._cs

    async def fetch_open_orders(self, symbol=None): return list(self._open)

    async def cancel_order(self, order_id, symbol):
        self.cancelled.append(order_id)
        self._open = [o for o in self._open if o.get("order_id") != order_id]
        return True

    async def place_order(self, req):
        self.placed.append(req)
        return types.SimpleNamespace(success=self._place_ok, error_code="X",
                                     error_message="boom")


def _sleeve(adapter, signed, pct=0.10, assets=None, entry=72.0, reconcile_ok=True):
    s = object.__new__(CoinbaseSleeve)
    s._adapter = adapter
    s._assets = ("SOL",)
    s._protective_stop_pct = pct
    s._protective_stop_assets = tuple(assets) if assets else None
    s._reconcile_ok = reconcile_ok
    s._last_positions = ({} if signed == 0 else
                         {"SOL": {"product_id": PID, "signed_contracts": signed,
                                  "entry_vwap": entry}})
    return s


def _stop_order(side="SELL", contracts=1.0, oid="O1"):
    return {"order_id": oid, "side": side,
            "order_configuration": {"stop_limit_stop_limit_gtc":
                                    {"base_size": str(contracts)}}}


def _limit_order(oid="L9"):
    return {"order_id": oid, "side": "BUY",
            "order_configuration": {"limit_limit_gtc": {"base_size": "1"}}}


def _run(s): return asyncio.run(s.ensure_protective_stop("SOL"))


class TestSleeveStopReconcile:

    def test_disabled_by_default_places_nothing(self):
        a = _FakeAdapter()
        assert _run(_sleeve(a, signed=1, pct=0.0))["status"] == "DISABLED"
        assert a.placed == [] and a.cancelled == []

    def test_asset_allowlist_lets_you_roll_out_one_asset_first(self):
        """P141: activation is deliberate. SOL-only must leave BTC alone."""
        a = _FakeAdapter()
        s = _sleeve(a, signed=1, assets=["BTC"])
        assert _run(s)["status"] == "DISABLED"
        assert a.placed == []

    def test_a_stale_snapshot_touches_nothing(self):
        """Cancelling against last-known state could remove a live stop we
        cannot currently see. Same rule manage_to_signal follows."""
        a = _FakeAdapter(open_orders=[_stop_order()])
        s = _sleeve(a, signed=1, reconcile_ok=False)
        assert _run(s)["status"] == "SKIPPED_STALE"
        assert a.placed == [] and a.cancelled == []

    def test_long_with_no_stop_places_a_sell_stop_below_entry(self):
        a = _FakeAdapter()
        res = _run(_sleeve(a, signed=1, entry=72.0, pct=0.10))
        assert res["status"] == "PLACED"
        assert len(a.placed) == 1
        req = a.placed[0]
        assert req.side == "SELL" and req.order_type == "STOP"
        assert req.stop_price == pytest.approx(64.8)  # 72 * 0.90

    def test_short_with_no_stop_places_a_buy_stop_above_entry(self):
        a = _FakeAdapter()
        res = _run(_sleeve(a, signed=-1, entry=72.0, pct=0.10))
        assert res["status"] == "PLACED"
        req = a.placed[0]
        assert req.side == "BUY"
        assert req.stop_price == pytest.approx(79.2)  # 72 * 1.10

    def test_the_stop_is_anchored_to_entry_not_to_the_mark(self):
        """Anchoring to the mark would silently make this a TRAILING stop that
        ratchets every tick and re-places orders forever."""
        a = _FakeAdapter()                      # get_product mid = 72.0
        s = _sleeve(a, signed=1, entry=100.0, pct=0.10)
        assert s.desired_stop_price("SOL") == pytest.approx(90.0)  # not 64.8

    def test_entry_price_is_read_from_the_key_the_venue_actually_returns(self):
        """[P197] CDE returns `avg_entry_price`; `entry_vwap` is never present.

        reconcile_positions read `entry_vwap`, so the entry was silently None for
        every position since the sleeve was written. Nothing consumed it, so it
        was invisible — the protective stop is its first consumer, and without
        this it anchors to the MARK instead of to entry, without saying so.
        Textbook P2 reader/writer key mismatch.
        """
        s = object.__new__(CoinbaseSleeve)
        s._adapter = _FakeAdapter()
        s._assets = ("SOL",)
        s._pid_to_asset = {PID: "SOL"}
        s._last_positions = {}
        s._reconcile_ok = False
        s._adapter._client = types.SimpleNamespace(
            list_futures_positions=lambda: {"positions": [{
                "product_id": PID, "side": "LONG", "number_of_contracts": "1",
                "avg_entry_price": "64395", "current_price": "64200",
                "unrealized_pnl": "-1.7",
            }]})
        out = s.reconcile_positions()
        assert out["SOL"]["entry_vwap"] == pytest.approx(64395.0), (
            "entry price lost — reconcile is reading a key the venue does not send"
        )

    def test_a_correct_resting_stop_is_left_alone(self):
        a = _FakeAdapter(open_orders=[_stop_order("SELL", 1.0)])
        res = _run(_sleeve(a, signed=1))
        assert res["status"] == "OK_EXISTS"
        assert a.placed == [] and a.cancelled == []

    def test_a_wrong_side_stop_is_replaced(self):
        """Left over from before a flip: a BUY stop does not protect a long."""
        a = _FakeAdapter(open_orders=[_stop_order("BUY", 1.0)])
        res = _run(_sleeve(a, signed=1))
        assert res["status"] == "PLACED"
        assert a.cancelled == ["O1"]
        assert a.placed[0].side == "SELL"

    def test_a_wrong_size_stop_is_replaced(self):
        a = _FakeAdapter(open_orders=[_stop_order("SELL", 1.0)])
        res = _run(_sleeve(a, signed=2))
        assert res["status"] == "PLACED"
        assert a.cancelled == ["O1"]

    # --- the reduce_only hazard -------------------------------------------

    def test_flat_cancels_an_orphan_stop(self):
        """THE most important case. CDE rejects reduce_only, so a stop left
        resting on a flat asset is a plain order that OPENS a position when it
        triggers."""
        a = _FakeAdapter(open_orders=[_stop_order("SELL", 1.0)])
        res = _run(_sleeve(a, signed=0))
        assert res["status"] == "FLAT_CANCELLED"
        assert a.cancelled == ["O1"]
        assert a.placed == [], "placed a stop on a flat asset"

    def test_flat_with_no_stop_is_a_quiet_noop(self):
        a = _FakeAdapter()
        assert _run(_sleeve(a, signed=0))["status"] == "FLAT_NONE"
        assert a.cancelled == [] and a.placed == []

    def test_non_stop_orders_are_not_mistaken_for_stops(self):
        """A resting entry limit must not be counted as protection."""
        a = _FakeAdapter(open_orders=[_limit_order()])
        res = _run(_sleeve(a, signed=1))
        assert res["status"] == "PLACED", "an entry limit was mistaken for a stop"

    def test_a_failed_placement_is_reported_not_swallowed(self):
        a = _FakeAdapter(place_ok=False)
        res = _run(_sleeve(a, signed=1))
        assert res["status"] == "FAILED"
        assert "X" in res["reason"]

    def test_it_never_raises_into_the_tick(self):
        class _Boom(_FakeAdapter):
            async def fetch_open_orders(self, symbol=None):
                raise RuntimeError("venue down")
        res = _run(_sleeve(_Boom(), signed=1))
        assert res["status"] == "ERROR"


class _SdkOrderConfiguration:
    """Mimics the SDK's OrderConfiguration: NOT a dict, data in __dict__.

    This is the exact shape `list_orders` returns. Every fake in this file
    originally used a plain dict, which is why the bug below shipped.
    """

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class TestVenueObjectsAreNotDicts:
    """[P197-fix] The reconciler could not recognise its own live stop.

    `_is_stop_order` required `order_configuration` to be a dict. The SDK returns
    an `OrderConfiguration` object, and `fetch_open_orders` only did a one-level
    `__dict__` conversion, so the nested object survived. Verified against the
    real venue: a stop-limit resting on SOL (stop 80.82, STOP_DIRECTION_STOP_UP)
    read as `_is_stop_order() -> False`.

    Two consequences, both live-order defects rather than cosmetic:
      * the next tick would not see the existing stop and would place a SECOND
        one — stacking a new real order every 4H;
      * on going flat, the orphan-cancel branch iterates the (empty) stop list
        and cancels NOTHING, leaving exactly the orphan the whole design exists
        to prevent, on a venue with no reduce_only.

    Tests must use the object shape, not a convenient dict.
    """

    def _order(self):
        return {"order_id": "REAL1", "side": "BUY",
                "order_configuration": _SdkOrderConfiguration(
                    stop_limit_stop_limit_gtc={
                        "base_size": "1", "limit_price": "81.22",
                        "stop_price": "80.82",
                        "stop_direction": "STOP_DIRECTION_STOP_UP"})}

    def test_a_stop_whose_config_is_an_object_is_still_recognised(self):
        assert CoinbaseSleeve._is_stop_order(self._order()) is True, (
            "the live stop was not recognised as a stop — the reconciler would "
            "place a second one next tick and fail to cancel it when flat"
        )

    def test_an_object_shaped_stop_is_left_alone_not_duplicated(self):
        a = _FakeAdapter(open_orders=[self._order()])
        res = _run(_sleeve(a, signed=-1))
        assert res["status"] == "OK_EXISTS", res
        assert a.placed == [], "placed a duplicate stop on top of the live one"

    def test_an_object_shaped_orphan_is_cancelled_when_flat(self):
        a = _FakeAdapter(open_orders=[self._order()])
        res = _run(_sleeve(a, signed=0))
        assert res["status"] == "FLAT_CANCELLED"
        assert a.cancelled == ["REAL1"], (
            "orphan stop survived going flat — on CDE (no reduce_only) it would "
            "OPEN a position when triggered"
        )

    def test_the_adapter_normalises_nested_objects_to_plain_dicts(self):
        """Fix at the boundary, so every consumer gets ordinary dicts."""
        from exchange.coinbase_adapter import _plain
        out = _plain(self._order())
        assert isinstance(out["order_configuration"], dict)
        assert out["order_configuration"]["stop_limit_stop_limit_gtc"]["stop_price"] == "80.82"

    def test_plain_is_depth_bounded_against_self_reference(self):
        from exchange.coinbase_adapter import _plain
        a = _SdkOrderConfiguration()
        a.self_ref = a          # cyclic
        _plain(a)               # must terminate, not recurse forever


class TestTheTickSummaryCannotContradictItself:
    """[P197] The stop-summary block was first written between
    `if _m_summary:` and its `else:`, which silently re-bound the else to the
    stop condition. With stops disabled (the default) the engine then logged

        [COINBASE-MANAGE] tick summary: BTC=NOOP, ETH=NOOP, SOL=NOOP
        [COINBASE-MANAGE] NO routed assets managed this tick — the order path is inert

    one line apart. The log contradicted itself about the single thing this
    block exists to report, and it shipped to production before the live output
    made it obvious. The guard is now an explicit `if not _m_summary:` rather
    than a dangling else, so inserting code between them cannot re-bind it.
    """

    @staticmethod
    def _main_src():
        from pathlib import Path
        return (Path(__file__).resolve().parents[1] / "main.py").read_text(
            encoding="utf-8", errors="replace")

    def test_the_inert_warning_is_guarded_explicitly_not_by_a_dangling_else(self):
        src = self._main_src()
        marker = '"[COINBASE-MANAGE] NO routed assets managed this tick "'
        assert marker in src, "the inert-path warning vanished"
        window = src[max(0, src.index(marker) - 400):src.index(marker)]
        assert "if not _m_summary:" in window, (
            "the 'NO routed assets managed' warning is no longer guarded by an "
            "explicit `if not _m_summary:`. If it has been turned back into an "
            "`else:`, any block inserted above it re-binds the condition — which "
            "is exactly how it came to fire on ticks where all three assets were "
            "managed."
        )

    def test_both_summaries_are_still_emitted(self):
        src = self._main_src()
        assert '"[COINBASE-MANAGE] tick summary: "' in src
        assert '"[COINBASE-STOP] tick summary: "' in src


class TestIntentBeatsTheSnapshotWhenFlattening:
    """[P207] A real orphan, observed live 2026-08-07.

    P206 activation flattened ETH and SOL (the alpha gate refused both).
    `execute_target` places a marketable LIMIT, which had not FILLED when
    `ensure_protective_stop` ran, so `reconcile_positions` still reported the
    old position and a protective stop was PLACED on an asset that went flat
    seconds later. Venue state immediately after:

        POSITIONS:   BTC LONG 1        (ETH and SOL flat)
        OPEN ORDERS: BTC SELL stop     (correct)
                     SOL BUY  stop     <- ORPHAN on a flat asset
                     ETH SELL stop     <- ORPHAN on a flat asset

    CDE rejects reduce_only, so those are PLAIN orders: touching 80.82 would
    have OPENED a SOL long. The next reconcile was 4 hours away.

    Reconciling to the venue is right in the steady state and wrong in the
    instant between "accepted" and "filled". When the caller knows the intent,
    intent wins.
    """

    def test_intended_flat_cancels_even_while_the_snapshot_still_shows_a_position(self):
        a = _FakeAdapter(open_orders=[_stop_order("SELL", 1.0)])
        s = _sleeve(a, signed=1)          # snapshot: still long (fill pending)
        res = asyncio.run(s.ensure_protective_stop("SOL", intended_target=0))
        assert res["status"] == "FLAT_CANCELLED", res
        assert a.cancelled == ["O1"]
        assert a.placed == [], "re-placed a stop on an asset being flattened"

    def test_intended_flat_places_nothing_when_no_stop_rests(self):
        a = _FakeAdapter()
        s = _sleeve(a, signed=1)
        res = asyncio.run(s.ensure_protective_stop("SOL", intended_target=0))
        assert res["status"] == "FLAT_NONE"
        assert a.placed == []

    def test_a_nonzero_intent_still_uses_the_snapshot(self):
        """The override is narrow: only 'intended flat' bypasses reconcile."""
        a = _FakeAdapter()
        s = _sleeve(a, signed=1)
        res = asyncio.run(s.ensure_protective_stop("SOL", intended_target=1))
        assert res["status"] == "PLACED"

    def test_omitting_the_argument_preserves_the_old_behaviour(self):
        a = _FakeAdapter()
        s = _sleeve(a, signed=1)
        assert asyncio.run(s.ensure_protective_stop("SOL"))["status"] == "PLACED"
