"""[P265] Sleeve order-lifecycle fixes.

Four defects on the only live order path, all in the P207-observed non-fill
window or its neighbors:

  1. `execute_target`'s NOOP/BLOCKED early returns preceded the only cancel
     sweep — an unfilled marketable entry limit rested through every hold tick
     and could fill days later against a dead signal, INCLUDING UNDER HALT.
  2. `ensure_protective_stop` honored `intended_target` only when ==0: in the
     flip window a fresh stop was anchored to the SNAPSHOT (old side) — when
     the flip filled, that stop firing would DOUBLE the new position (no
     reduce_only on CDE, no can_trade gate on venue-triggered fills).
  3. The stop clear-and-replace loop discarded cancel results — a failed
     cancel yielded TWO plain stops, the second of which OPENS an opposite
     position when it fires.
  4. `_matches` checked side+size only — a config pct change never re-priced
     resting stops; a mark-anchored fallback stop was never re-anchored; any
     same-shaped stop at any trigger was silently adopted as "the protection".
"""

import asyncio
import types

import pytest

from exchange.coinbase_sleeve import CoinbaseSleeve

PID = "SLP-20DEC30-CDE"


class _FakeAdapter:
    def __init__(self, open_orders=None, contract_size=5.0, place_ok=True,
                 cancel_ok=True):
        self._open = list(open_orders or [])
        self._cs = contract_size
        self._place_ok = place_ok
        self._cancel_ok = cancel_ok
        self.placed = []
        self.cancelled = []
        self._client = types.SimpleNamespace(
            get_product=lambda product_id: {"mid_market_price": "72.0"})

    def is_connected(self): return True
    def to_venue_symbol(self, asset, market="perp"): return PID
    def _contract_size(self, pid): return self._cs

    async def fetch_open_orders(self, symbol=None): return list(self._open)

    async def cancel_order(self, order_id, symbol):
        if not self._cancel_ok:
            return False
        self.cancelled.append(order_id)
        self._open = [o for o in self._open if o.get("order_id") != order_id]
        return True

    async def place_order(self, req):
        self.placed.append(req)
        return types.SimpleNamespace(success=self._place_ok, error_code="X",
                                     error_message="boom")


def _sleeve(adapter, signed, pct=0.10, entry=72.0, reconcile_ok=True,
            halted=False):
    s = object.__new__(CoinbaseSleeve)
    s._adapter = adapter
    s._assets = ("SOL",)
    s._protective_stop_pct = pct
    s._protective_stop_assets = None
    s._reconcile_ok = reconcile_ok
    s._halted = halted
    s._halt_reason = "sleeve drawdown 16% >= 15%" if halted else ""
    s._max_contracts_per_asset = 1
    s._max_net_exposure = None
    s._max_asset_exposure = {}
    s._flip_persist_ticks = 0
    s._flip_pending = {}
    s._last_positions = ({} if signed == 0 else
                         {"SOL": {"product_id": PID, "signed_contracts": signed,
                                  "entry_vwap": entry, "current_price": entry}})
    s.reconcile_positions = lambda: s._last_positions  # type: ignore[assignment]
    return s


def _entry_limit(oid="L9", side="BUY"):
    return {"order_id": oid, "side": side,
            "order_configuration": {"limit_limit_gtc": {"base_size": "1"}}}


def _stop(side="SELL", contracts=1.0, oid="S1", stop_price="64.8"):
    return {"order_id": oid, "side": side,
            "order_configuration": {"stop_limit_stop_limit_gtc":
                                    {"base_size": str(contracts),
                                     "stop_price": str(stop_price)}}}


# ---------------------------------------------------------------------------
# 1. stale entry limits are swept on no-order ticks
# ---------------------------------------------------------------------------

class TestStaleEntrySweep:
    def test_noop_tick_cancels_a_resting_entry_limit(self):
        # Tick N: entry limit placed, unfilled. Tick N+1: signal decayed to
        # hold -> target 0, cur 0 -> NOOP. The limit used to survive forever.
        a = _FakeAdapter(open_orders=[_entry_limit()])
        s = _sleeve(a, signed=0)
        r = asyncio.run(s.execute_target("SOL", 0))
        assert r["status"] == "NOOP"
        assert "L9" in a.cancelled, (
            "the unfilled entry limit survived the NOOP tick — it can fill "
            "days later against a dead signal, with no stop resting (P265)")

    def test_blocked_tick_cancels_a_resting_entry_limit(self):
        # A limit placed just before the halt tripped must not survive the
        # halt: BLOCKED used to return before any cancel, so the pre-halt
        # order could open a position UNDER halt.
        a = _FakeAdapter(open_orders=[_entry_limit()])
        s = _sleeve(a, signed=0, halted=True)
        r = asyncio.run(s.execute_target("SOL", 1))
        assert r["status"] == "BLOCKED"
        assert "L9" in a.cancelled, (
            "the pre-halt entry limit survived the halt — the halt's 'no new "
            "risk' guarantee is bypassed by an order it never saw (P265)")

    def test_the_sweep_never_touches_the_protective_stop(self):
        # The stop is the ONE resting order that should survive a hold tick.
        a = _FakeAdapter(open_orders=[_stop(), _entry_limit()])
        s = _sleeve(a, signed=1)
        asyncio.run(s.execute_target("SOL", 1))  # cur==target -> NOOP
        assert "L9" in a.cancelled
        assert "S1" not in a.cancelled, (
            "the NOOP sweep cancelled the protective stop — that churns the "
            "venue every tick and resets the anchor")

    def test_clean_noop_makes_no_cancels(self):
        a = _FakeAdapter()
        s = _sleeve(a, signed=0)
        r = asyncio.run(s.execute_target("SOL", 0))
        assert r["status"] == "NOOP"
        assert a.cancelled == []

    def test_sweep_failure_does_not_break_the_noop(self):
        class _Boom(_FakeAdapter):
            async def fetch_open_orders(self, symbol=None):
                raise ConnectionError("down")
        s = _sleeve(_Boom(), signed=0)
        r = asyncio.run(s.execute_target("SOL", 0))
        assert r["status"] == "NOOP"


# ---------------------------------------------------------------------------
# 2. flip in flight places no stop
# ---------------------------------------------------------------------------

class TestFlipInTransition:
    def test_flip_window_places_no_stop_for_either_side(self):
        # Snapshot +1 (old long), intent -1 (flip sent, unfilled). A SELL stop
        # (snapshot side) doubles the short once the flip fills; a BUY stop
        # (intent side) doubles the long if the flip strands. Place nothing.
        a = _FakeAdapter()
        s = _sleeve(a, signed=1)
        r = asyncio.run(s.ensure_protective_stop("SOL", intended_target=-1))
        assert r["status"] == "FLIP_IN_TRANSITION"
        assert a.placed == [], (
            "a stop was placed during a flip in flight — wrong-side the "
            "moment the flip fills; firing it breaches the contract cap "
            "venue-side with no can_trade gate (P265)")

    def test_flatten_intent_still_cancels_orphans(self):
        # The P207 carve-out must survive: intended 0 treats the asset flat.
        a = _FakeAdapter(open_orders=[_stop()])
        s = _sleeve(a, signed=1)
        r = asyncio.run(s.ensure_protective_stop("SOL", intended_target=0))
        assert r["status"] == "FLAT_CANCELLED"
        assert "S1" in a.cancelled

    def test_agreeing_intent_reconciles_normally(self):
        a = _FakeAdapter()
        s = _sleeve(a, signed=1)
        r = asyncio.run(s.ensure_protective_stop("SOL", intended_target=1))
        assert r["status"] == "PLACED"
        assert a.placed[0].side == "SELL"

    def test_no_intent_keeps_snapshot_authority(self):
        a = _FakeAdapter()
        s = _sleeve(a, signed=1)
        r = asyncio.run(s.ensure_protective_stop("SOL"))
        assert r["status"] == "PLACED"


# ---------------------------------------------------------------------------
# 3. failed cancel refuses replacement
# ---------------------------------------------------------------------------

class TestCancelFailureRefusesReplacement:
    def test_failed_cancel_means_no_second_stop(self):
        a = _FakeAdapter(open_orders=[_stop(side="BUY")], cancel_ok=False)
        s = _sleeve(a, signed=1)  # long: needs SELL stop, BUY stop is wrong
        r = asyncio.run(s.ensure_protective_stop("SOL"))
        assert r["status"] == "CANCEL_FAILED"
        assert a.placed == [], (
            "a replacement stop was placed while the old one may still be "
            "live — two plain stops: the second OPENS an opposite position "
            "when it fires (P265)")

    def test_successful_cancel_still_replaces(self):
        a = _FakeAdapter(open_orders=[_stop(side="BUY")])
        s = _sleeve(a, signed=1)
        r = asyncio.run(s.ensure_protective_stop("SOL"))
        assert r["status"] == "PLACED"
        assert a.cancelled == ["S1"]


# ---------------------------------------------------------------------------
# 4. the match includes the price
# ---------------------------------------------------------------------------

class TestStopPriceMatch:
    def test_a_stop_at_the_wrong_price_is_replaced(self):
        # e.g. placed at 15% before the config moved to 10%, or anchored to
        # the mark while entry_vwap was momentarily absent.
        a = _FakeAdapter(open_orders=[_stop(stop_price="61.2")])  # 15% stop
        s = _sleeve(a, signed=1)  # desired: 72 * 0.90 = 64.8
        r = asyncio.run(s.ensure_protective_stop("SOL"))
        assert r["status"] == "PLACED", (
            "a stop 5% from the desired trigger was certified OK_EXISTS — "
            "config changes never re-price resting stops (P265)")
        assert a.cancelled == ["S1"]

    def test_a_stop_at_the_right_price_is_left_alone(self):
        a = _FakeAdapter(open_orders=[_stop(stop_price="64.8")])
        s = _sleeve(a, signed=1)
        r = asyncio.run(s.ensure_protective_stop("SOL"))
        assert r["status"] == "OK_EXISTS"
        assert a.placed == [] and a.cancelled == []

    def test_tick_rounding_jitter_does_not_churn(self):
        # Within 0.5% tolerance (venue rounds to tick multiples).
        a = _FakeAdapter(open_orders=[_stop(stop_price="64.82")])
        s = _sleeve(a, signed=1)
        r = asyncio.run(s.ensure_protective_stop("SOL"))
        assert r["status"] == "OK_EXISTS"

    def test_a_priceless_stop_record_is_replaced(self):
        # No stop_price in the config: cannot verify it protects anything.
        a = _FakeAdapter(open_orders=[_stop(stop_price="")])
        s = _sleeve(a, signed=1)
        r = asyncio.run(s.ensure_protective_stop("SOL"))
        assert r["status"] == "PLACED"
