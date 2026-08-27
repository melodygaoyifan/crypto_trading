"""[P420] The maker ladder's DONE branch sizes the cross off the post-only's
OWN filled_size, never off a position snapshot that LAGS the fill.

Measured: list_futures_positions lags fills (P382 4/4 taker legs read the
PRE-trade size; 2026-08-27 10:08 `SELL 1ct MARKET URGENT -> now=2.0ct`).
The old DONE branch did `reconcile; delta = target - cur; cross delta`, so a
FULL maker fill read as "nothing filled" and the remainder crossed at taker:
a DOUBLE position, ungated.

  * full fill + lagged snapshot -> NO cross;
  * partial fill -> cross the remainder only;
  * unreadable fill state + snapshot unmoved after the bounded wait ->
    NO cross, distinct reason `maker_fill_unresolved_no_cross`;
  * unreadable + snapshot moved -> the snapshot is the evidence;
  * venue-cancelled with filled 0 -> full remainder crossed, and the log
    says "venue-cancelled ... filled 0" rather than "filled" [task 7];
  * a CONFIRMED-cancelled 1ct order with unreadable fill state crosses
    without waiting (a partial on 1ct is impossible; the cancel
    confirmation rules out a full fill) -- the recorded deviation.
"""
from __future__ import annotations

import asyncio
import logging
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from exchange.coinbase_sleeve import CoinbaseSleeve  # noqa: E402

PID = "SLP-20DEC30-CDE"


@pytest.fixture(autouse=True)
def _private_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HMATS_DATA_DIR", str(tmp_path))


@pytest.fixture(autouse=True)
def _instant_clock(monkeypatch):
    """Fake monotonic + instant asyncio.sleep so the maker window and the
    P420 bounded wait cost no wall time."""
    clock = {"t": 0.0}

    class _Asyncio:
        def __getattr__(self, name):
            if name == "sleep":
                async def _sleep(secs):
                    clock["t"] += float(secs)
                return _sleep
            return getattr(asyncio, name)

    import time as _rt
    facade = types.SimpleNamespace(monotonic=lambda: clock["t"],
                                   time=_rt.time, sleep=_rt.sleep)
    monkeypatch.setattr("exchange.coinbase_sleeve.asyncio", _Asyncio())
    monkeypatch.setattr("exchange.coinbase_sleeve.time", facade)
    return clock


class _FakeAdapter:
    """post-only leaves the book at the first poll (`leaves_book=True`) or
    rests until the timeout cancel. get_order is scripted."""

    def __init__(self, get_order=None, leaves_book=True, cancel_ok=True,
                 get_order_raises=False):
        self._cs = 5.0
        self.get_order_payload = get_order
        self.get_order_raises = get_order_raises
        self.leaves_book = leaves_book
        self.cancel_ok = cancel_ok
        self.placed = []
        self.cancelled = []
        self._open = []
        self.get_order_calls = 0
        self._client = types.SimpleNamespace(
            get_product=lambda product_id: {"mid_market_price": "100.0"},
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
        if self.leaves_book:
            self._open = []
        return list(self._open)

    async def cancel_order(self, oid, pid):
        if not self.cancel_ok:
            return False
        self.cancelled.append(oid)
        self._open = [o for o in self._open if o.get("order_id") != oid]
        return True

    async def get_order(self, order_id):
        self.get_order_calls += 1
        if self.get_order_raises:
            raise ConnectionError("get_order down")
        return None if self.get_order_payload is None else dict(
            self.get_order_payload)

    async def place_order(self, req):
        self.placed.append(req)
        oid = f"oid-{len(self.placed)}"
        if getattr(req, "post_only", False):
            self._open.append({"order_id": oid, "side": req.side,
                               "order_configuration": {
                                   "limit_limit_gtc": {"base_size": "1"}}})
        return types.SimpleNamespace(success=True, order_id=oid,
                                     error_code=None, error_message=None)


def _sleeve(adapter, cur=0.0, moves_to=None, lag=99, wait=20.0):
    """`moves_to`: the snapshot value the position eventually shows;
    `lag`: how many reconcile calls AFTER the post-only until it shows."""
    s = object.__new__(CoinbaseSleeve)
    s._adapter = adapter
    s._maker_first = True
    s._maker_wait_sec = wait
    s._maker_reprice = False
    s._reconcile_ok = True
    s._halted = False
    s._halt_reason = ""
    s._max_contracts_per_asset = 10
    s._max_net_exposure = None
    s._max_asset_exposure = {}
    state = {"cur": float(cur), "reconciles": 0}
    s.signed_contracts = lambda asset: state["cur"]
    s.is_ready = lambda: True

    def _reconcile():
        if moves_to is not None and any(
                getattr(p, "post_only", False) for p in adapter.placed):
            state["reconciles"] += 1
            if state["reconciles"] > lag:
                state["cur"] = float(moves_to)
        return {}
    s.reconcile_positions = _reconcile
    s.can_trade = lambda a, d: (True, "ok")

    async def _noop(*a, **k):
        return 0
    s._cancel_resting_orders = _noop
    s._cancel_stale_entry_orders = _noop
    return s, state


def _crosses(ad):
    return [p for p in ad.placed if not getattr(p, "post_only", False)]


def _run(coro):
    return asyncio.run(coro)


FILLED_1 = {"status": "FILLED", "filled_size": "1",
            "average_filled_price": "99.9"}


# ---------------------------------------------------------------------------
# the double-fill class
# ---------------------------------------------------------------------------

class TestFullFill:
    def test_full_fill_with_lagged_snapshot_places_no_cross(self):
        ad = _FakeAdapter(get_order=FILLED_1)
        s, st = _sleeve(ad, cur=0.0, moves_to=1.0, lag=99)  # never visible
        res = _run(s.execute_target("SOL", 1))
        assert res["status"] == "OK" and res.get("maker") is True
        assert res.get("maker_filled") == 1.0
        assert _crosses(ad) == [], "DOUBLE position: full maker fill + taker cross"

    def test_full_fill_on_a_3ct_order(self):
        ad = _FakeAdapter(get_order={"status": "FILLED", "filled_size": "3"})
        s, _ = _sleeve(ad, cur=0.0, moves_to=3.0, lag=99)
        res = _run(s.execute_target("SOL", 3))
        assert res["status"] == "OK" and _crosses(ad) == []

    def test_filled_and_snapshot_visible_is_still_ok(self):
        ad = _FakeAdapter(get_order=FILLED_1)
        s, _ = _sleeve(ad, cur=0.0, moves_to=1.0, lag=0)
        res = _run(s.execute_target("SOL", 1))
        assert res["status"] == "OK" and _crosses(ad) == []


class TestPartialFill:
    def test_partial_fill_crosses_only_the_remainder(self):
        ad = _FakeAdapter(get_order={"status": "CANCELLED",
                                     "filled_size": "1"})
        s, _ = _sleeve(ad, cur=0.0)
        res = _run(s.execute_target("SOL", 3))
        assert res["status"] == "OK"
        cr = _crosses(ad)
        assert len(cr) == 1 and cr[0].size == pytest.approx(2 * ad._cs)
        assert cr[0].side == "BUY"

    def test_remainder_never_exceeds_the_snapshots_outstanding(self):
        # get_order says 1 filled of 3, but the snapshot already shows the
        # book AT target: cross nothing (under-target is the safe direction)
        ad = _FakeAdapter(get_order={"status": "CANCELLED",
                                     "filled_size": "1"})
        s, _ = _sleeve(ad, cur=0.0, moves_to=3.0, lag=0)
        res = _run(s.execute_target("SOL", 3))
        assert res["status"] == "OK" and _crosses(ad) == []

    def test_partial_on_a_short(self):
        ad = _FakeAdapter(get_order={"status": "CANCELLED",
                                     "filled_size": "2"})
        s, _ = _sleeve(ad, cur=0.0)
        res = _run(s.execute_target("SOL", -3))
        cr = _crosses(ad)
        assert res["status"] == "OK" and len(cr) == 1
        assert cr[0].side == "SELL" and cr[0].size == pytest.approx(ad._cs)


# ---------------------------------------------------------------------------
# unreadable fill state
# ---------------------------------------------------------------------------

class TestUnreadableFillState:
    def test_unreadable_and_unmoved_refuses_with_a_distinct_reason(self):
        ad = _FakeAdapter(get_order=None)          # get_order -> None
        s, st = _sleeve(ad, cur=0.0)               # snapshot never moves
        res = _run(s.execute_target("SOL", 1))
        assert res["status"] == "FAILED"
        assert res["reason"] == "maker_fill_unresolved_no_cross"
        assert _crosses(ad) == []
        # the bounded wait actually re-reconciled: 1 pre + 1 post + 3 waits
        assert st["reconciles"] == 0  # moves_to None -> counter unused
        assert ad.get_order_calls == 1

    def test_unreadable_and_unmoved_reconciles_a_bounded_number_of_times(
            self):
        ad = _FakeAdapter(get_order=None)
        s, st = _sleeve(ad, cur=0.0, moves_to=1.0, lag=99)
        _run(s.execute_target("SOL", 1))
        # post-maker reconcile + MAKER_FILL_WAIT_TRIES re-reconciles
        assert st["reconciles"] == 1 + CoinbaseSleeve.MAKER_FILL_WAIT_TRIES

    def test_unreadable_but_moved_is_a_maker_fill(self):
        ad = _FakeAdapter(get_order=None)
        s, _ = _sleeve(ad, cur=0.0, moves_to=1.0, lag=2)  # shows on the 3rd
        res = _run(s.execute_target("SOL", 1))
        assert res["status"] == "OK" and res.get("maker") is True
        assert _crosses(ad) == []

    def test_get_order_raising_is_unreadable_not_zero(self):
        ad = _FakeAdapter(get_order_raises=True)
        s, _ = _sleeve(ad, cur=0.0)
        res = _run(s.execute_target("SOL", 1))
        assert res["status"] == "FAILED"
        assert res["reason"] == "maker_fill_unresolved_no_cross"
        assert _crosses(ad) == []

    def test_wait_parameters_are_the_documented_bound(self):
        assert CoinbaseSleeve.MAKER_FILL_WAIT_TRIES == 3
        assert CoinbaseSleeve.MAKER_FILL_WAIT_SEC == 2.0


# ---------------------------------------------------------------------------
# venue-cancelled with filled 0 -> the full remainder is crossed (and said)
# ---------------------------------------------------------------------------

class TestVenueCancelled:
    def test_filled_zero_with_terminal_status_crosses_the_full_remainder(
            self, caplog):
        ad = _FakeAdapter(get_order={"status": "CANCELLED",
                                     "filled_size": "0"})
        s, _ = _sleeve(ad, cur=0.0)
        with caplog.at_level(logging.INFO, logger="exchange.coinbase_sleeve"):
            res = _run(s.execute_target("SOL", 1))
        assert res["status"] == "OK" and res.get("maker") is None
        cr = _crosses(ad)
        assert len(cr) == 1 and cr[0].size == pytest.approx(ad._cs)
        # [task 7] the log no longer claims a fill
        msgs = [r.getMessage() for r in caplog.records]
        assert any("venue-cancelled" in m and "filled 0/1" in m
                   for m in msgs), msgs
        assert not any("filled at 0bps maker" in m for m in msgs)

    def test_a_real_fill_is_logged_as_filled_n_of_m(self, caplog):
        ad = _FakeAdapter(get_order={"status": "FILLED", "filled_size": "2"})
        s, _ = _sleeve(ad, cur=0.0)
        with caplog.at_level(logging.INFO, logger="exchange.coinbase_sleeve"):
            _run(s.execute_target("SOL", 2))
        msgs = [r.getMessage() for r in caplog.records]
        assert any("filled 2/2" in m and "maker" in m for m in msgs), msgs


# ---------------------------------------------------------------------------
# timeout + CONFIRMED cancel (our own hand) with unreadable fill state
# ---------------------------------------------------------------------------

class TestConfirmedCancelUnreadable:
    def test_1ct_cancelled_unreadable_crosses_once_without_waiting(self):
        # The recorded deviation from "unreadable -> never cross": a 1ct
        # post-only CONFIRMED cancelled cannot have partially filled, and
        # the cancel confirmation rules out a full fill. Refusing here
        # would be an under-target tick for nothing.
        ad = _FakeAdapter(get_order=None, leaves_book=False)
        s, st = _sleeve(ad, cur=0.0, moves_to=1.0, lag=99, wait=10.0)
        res = _run(s.execute_target("SOL", 1))
        assert ad.cancelled == ["oid-1"]
        assert res["status"] == "OK"
        assert len(_crosses(ad)) == 1
        # post-maker reconcile + the cross's own post-place reconcile; NO
        # bounded-wait re-reconciles in between
        assert st["reconciles"] == 2, "waited for a partial that cannot exist"

    def test_multi_ct_cancelled_unreadable_waits_then_crosses_the_snapshot(
            self):
        ad = _FakeAdapter(get_order=None, leaves_book=False)
        s, st = _sleeve(ad, cur=0.0, moves_to=1.0, lag=1, wait=10.0)
        res = _run(s.execute_target("SOL", 3))
        assert res["status"] == "OK"
        cr = _crosses(ad)
        # the snapshot showed 1 filled before the cancel -> cross 2
        assert len(cr) == 1 and cr[0].size == pytest.approx(2 * ad._cs)

    def test_cancelled_with_readable_zero_crosses_the_full_n(self):
        ad = _FakeAdapter(get_order={"status": "CANCELLED",
                                     "filled_size": "0"}, leaves_book=False)
        s, _ = _sleeve(ad, cur=0.0, wait=10.0)
        res = _run(s.execute_target("SOL", 2))
        assert res["status"] == "OK"
        cr = _crosses(ad)
        assert len(cr) == 1 and cr[0].size == pytest.approx(2 * ad._cs)


# ---------------------------------------------------------------------------
# the parser: absence is not zero
# ---------------------------------------------------------------------------

class TestParseFilledContracts:
    P = staticmethod(CoinbaseSleeve._parse_filled_contracts)

    def test_table(self):
        assert self.P(None) is None
        assert self.P({}) is None                                   # no field
        assert self.P({"filled_size": None}) is None
        assert self.P({"filled_size": ""}) is None
        assert self.P({"filled_size": "abc"}) is None
        assert self.P({"filled_size": "-1"}) is None
        assert self.P({"filled_size": "0", "status": "FILLED"}) is None  # inconsistent
        assert self.P({"filled_size": "0", "status": "CANCELLED"}) == 0.0
        assert self.P({"filled_size": "0", "status": "EXPIRED"}) == 0.0
        assert self.P({"filled_size": "2", "status": "FILLED"}) == 2.0
        assert self.P({"filled_size": 1.0}) == 1.0

    def test_stash_shape_is_getattr_defended(self):
        s = object.__new__(CoinbaseSleeve)
        s._note_maker_filled("SOL", 1.0)
        assert s._maker_last_filled["SOL"] == {"filled": 1.0,
                                               "cancelled": False}
        s._note_maker_filled("SOL", None, cancelled=True)
        assert s._maker_last_filled["SOL"]["cancelled"] is True


class TestNoRegression:
    def test_urgent_never_enters_the_ladder(self):
        ad = _FakeAdapter(get_order=FILLED_1)
        s, _ = _sleeve(ad, cur=1.0)
        res = _run(s.execute_target("SOL", 0, urgent=True))
        assert res["status"] == "OK"
        assert not [p for p in ad.placed if getattr(p, "post_only", False)]
        # (the direct leg's P290 fill-quality read is the only get_order)
        assert ad.get_order_calls == 1
