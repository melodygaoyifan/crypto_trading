"""[P287] Exchange-layer fixes from the 2026-08-16 eight-fork read-through.

Findings covered (each verified at source before fixing):
  1. adapter.fetch_open_orders RAISES on failure (was: swallow into [] —
     "could not read the book" byte-identical to "book is clean", fail-opening
     four sleeve order-lifecycle guards into the P265 double-order class)
  2. adapter.cancel_order: UNCONFIRMED = FAILED (was: `return bool(results)`
     certified a failed cancel as success on any unmatched result row)
  3. maker no-order_id branch: sweep + verify before any cross (was: "DONE"
     -> blind cross beside the untracked resting post-only)
  4. flat-path orphan-stop cancel failures: distinct FLAT_CANCEL_FAILED
     status (was: silent, byte-identical to "no orphans existed")
  5. equity staleness bound (P156 rule applied to _last_equity_usd): dd hold,
     fixed-book sizing fallback, age exposed via snapshot()
  6. sizing hysteresis at the floor() contract boundary (pure helper)
  7. mark-anchored stop reuses its anchor within 2% (was: >0.5% drift
     cancel+replaced the stop every tick)
  8. withdrawal-direction note at the drawdown halt (comment only)
"""

import asyncio
import inspect
import time
import types

import pytest

from exchange.coinbase_adapter import CoinbaseAdapter
from exchange.coinbase_sleeve import CoinbaseSleeve

PID = "SLP-FAKE"


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------

class _FakeAdapter:
    """Scriptable adapter surface for the sleeve-side tests."""

    def __init__(self, open_orders=None, contract_size=0.1, cancel_ok=True,
                 fetch_raises=False, maker_fills=False, maker_no_oid=False,
                 fetch_raises_after_place=False, mid="100.0"):
        self._open = list(open_orders or [])
        self._cs = contract_size
        self.cancel_ok = cancel_ok
        self.fetch_raises = fetch_raises
        self.fetch_raises_after_place = fetch_raises_after_place
        self.maker_fills = maker_fills
        self.maker_no_oid = maker_no_oid
        self.placed = []
        self.cancelled = []
        self.product = {"mid_market_price": mid}
        self._client = types.SimpleNamespace(
            get_product=lambda product_id: self.product,
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
        if self.fetch_raises:
            raise ConnectionError("listing down")
        if self.fetch_raises_after_place and any(
                getattr(p, "post_only", False) for p in self.placed):
            raise ConnectionError("listing down mid-poll")
        return list(self._open)

    async def cancel_order(self, oid, pid):
        if not self.cancel_ok:
            return False
        self.cancelled.append(oid)
        self._open = [o for o in self._open if o.get("order_id") != oid]
        return True

    async def place_order(self, req):
        self.placed.append(req)
        untracked = bool(getattr(req, "post_only", False) and self.maker_no_oid)
        oid = None if untracked else f"oid-{len(self.placed)}"
        if getattr(req, "post_only", False) and not self.maker_fills:
            row = {"side": req.side,
                   "order_configuration": {"limit_limit_gtc":
                                           {"base_size": "1"}}}
            if oid:
                row["order_id"] = oid
            self._open.append(row)
        return types.SimpleNamespace(success=True, order_id=oid,
                                     error_code=None, error_message=None)


def _exec_sleeve(adapter, cur=0.0, maker_first=False, wait=5.0,
                 filled_after_maker=None):
    s = object.__new__(CoinbaseSleeve)
    s._adapter = adapter
    s._maker_first = maker_first
    s._maker_wait_sec = wait
    s._reconcile_ok = True
    s._halted = False
    s._halt_reason = ""
    s._max_contracts_per_asset = 5
    s._max_net_exposure = None
    s._max_asset_exposure = {}
    state = {"cur": cur}
    s.signed_contracts = lambda asset: state["cur"]  # type: ignore[assignment]
    s.is_ready = lambda: True  # type: ignore[assignment]

    def _reconcile():
        if filled_after_maker is not None and adapter.placed and any(
                getattr(p, "post_only", False) for p in adapter.placed):
            state["cur"] = filled_after_maker
        return {}
    s.reconcile_positions = _reconcile  # type: ignore[assignment]
    s.can_trade = lambda a, d: (True, "ok")  # type: ignore[assignment]
    return s, state


def _stop_sleeve(adapter, signed, pct=0.10, entry=72.0):
    s = object.__new__(CoinbaseSleeve)
    s._adapter = adapter
    s._assets = ("SOL",)
    s._protective_stop_pct = pct
    s._protective_stop_assets = None
    s._reconcile_ok = True
    s._last_positions = ({} if signed == 0 else
                         {"SOL": {"product_id": PID,
                                  "signed_contracts": signed,
                                  "entry_vwap": entry}})
    return s


def _stop_order(oid="S1", side="SELL", contracts=1.0, stop_price="64.8"):
    return {"order_id": oid, "side": side,
            "order_configuration": {"stop_limit_stop_limit_gtc":
                                    {"base_size": str(contracts),
                                     "stop_price": str(stop_price)}}}


# ---------------------------------------------------------------------------
# 1. the adapter raises; [] is reserved for "genuinely empty"
# ---------------------------------------------------------------------------

class TestAdapterFetchOpenOrdersRaises:
    def _adapter(self, client):
        a = CoinbaseAdapter(rest_client=client, paper=True)
        return a

    def test_a_listing_failure_raises_instead_of_reading_as_clean(self):
        client = types.SimpleNamespace(
            list_orders=lambda **kw: (_ for _ in ()).throw(
                RuntimeError("rate limited")))
        a = self._adapter(client)
        with pytest.raises(RuntimeError):
            _run(a.fetch_open_orders(PID))

    def test_a_successful_empty_listing_still_returns_a_list(self):
        client = types.SimpleNamespace(
            list_orders=lambda **kw: types.SimpleNamespace(orders=[]))
        a = self._adapter(client)
        assert _run(a.fetch_open_orders(PID)) == []

    def test_not_configured_returns_empty_not_raise(self):
        # no creds = nothing of ours can be resting; is_ready() gates every
        # sleeve path before this call anyway
        a = CoinbaseAdapter(rest_client=None, paper=True)
        a._init_failed = "no_credentials"
        assert _run(a.fetch_open_orders(PID)) == []


class TestAdapterCancelUnconfirmedIsFailed:
    def _adapter(self, results):
        client = types.SimpleNamespace(
            cancel_orders=lambda order_ids: types.SimpleNamespace(
                results=results))
        return CoinbaseAdapter(rest_client=client, paper=True)

    def test_a_row_for_a_different_order_does_not_certify_the_cancel(self):
        # the old `return bool(results)` fallback: ANY non-empty row list
        # read as success — including a row for another order entirely
        a = self._adapter([{"order_id": "OTHER", "success": True}])
        assert _run(a.cancel_order("MINE", PID)) is False

    def test_a_matching_failed_row_is_failed(self):
        a = self._adapter([{"order_id": "MINE", "success": False}])
        assert _run(a.cancel_order("MINE", PID)) is False

    def test_a_matching_success_row_is_success(self):
        a = self._adapter([{"order_id": "MINE", "success": True}])
        assert _run(a.cancel_order("MINE", PID)) is True

    def test_an_empty_result_list_is_failed(self):
        a = self._adapter([])
        assert _run(a.cancel_order("MINE", PID)) is False


# ---------------------------------------------------------------------------
# 1b. sleeve callers fail SAFE on the raise
# ---------------------------------------------------------------------------

class TestSleeveRefusesOnUnreadableBook:
    def test_execute_target_places_nothing_when_the_book_is_unreadable(self):
        ad = _FakeAdapter(fetch_raises=True)
        s, _ = _exec_sleeve(ad, cur=0.0)
        res = _run(s.execute_target("SOL", 1))
        assert res["status"] == "SKIPPED_STALE"
        assert res["reason"] == "resting_orders_unverifiable"
        assert ad.placed == [], (
            "an order was placed beside a book we could not read — the P265 "
            "double-order class re-opened (P287)")

    def test_execute_target_refuses_when_a_sweep_cancel_is_unconfirmed(self):
        # a resting order we could not kill + a new order = two live orders
        ad = _FakeAdapter(open_orders=[{"order_id": "L1", "side": "BUY",
                                        "order_configuration":
                                        {"limit_limit_gtc":
                                         {"base_size": "1"}}}],
                          cancel_ok=False)
        s, _ = _exec_sleeve(ad, cur=0.0)
        res = _run(s.execute_target("SOL", 1))
        assert res["status"] == "SKIPPED_STALE"
        assert ad.placed == []

    def test_the_noop_sweep_reports_unknown_not_clean(self):
        ad = _FakeAdapter(fetch_raises=True)
        s, _ = _exec_sleeve(ad, cur=0.0)
        out = _run(s._cancel_stale_entry_orders(PID, "SOL"))
        assert out is None, (
            "'could not read the book' returned the same value as 'book is "
            "clean' — the P159/P171 conflation on the live order path")

    def test_maker_poll_listing_failure_never_reads_as_filled(self):
        # OLD behavior with the swallowing adapter: outage -> [] ->
        # still_open=False -> "filled at 0bps maker" -> cross the FULL
        # remainder beside the still-resting post-only. NEW: unknown ->
        # keep waiting -> timeout -> cancel -> cross exactly once.
        ad = _FakeAdapter(maker_fills=False, fetch_raises_after_place=True)
        s, _ = _exec_sleeve(ad, cur=0.0, maker_first=True, wait=5.0)
        res = _run(s.execute_target("SOL", 1))
        assert res["status"] == "OK"
        post_onlys = [p for p in ad.placed if getattr(p, "post_only", False)]
        crosses = [p for p in ad.placed if not getattr(p, "post_only", False)]
        assert len(post_onlys) == 1 and len(crosses) == 1
        assert len(ad.cancelled) == 1, (
            "the unfilled post-only was presumed filled instead of being "
            "cancelled — a listing outage read as a fill (P287)")

    def test_ensure_protective_stop_refuses_not_clears_on_unreadable_book(self):
        ad = _FakeAdapter(fetch_raises=True)
        s = _stop_sleeve(ad, signed=1)
        res = _run(s.ensure_protective_stop("SOL"))
        assert res["status"] == "ORDERS_UNREADABLE"
        assert ad.placed == [] and ad.cancelled == [], (
            "clear-and-place ran against an unreadable book — that places a "
            "second stop beside the invisible real one (P287)")


# ---------------------------------------------------------------------------
# 3. untracked maker order: sweep + verify before any cross
# ---------------------------------------------------------------------------

class TestUntrackedMakerOrder:
    def test_untracked_gone_with_unmoved_snapshot_refuses_the_cross(self):
        # [P420] Re-pointed to the DECIDED value (P237 pattern), not
        # weakened. The pre-P420 premise was "filled=False here means not
        # our fill; the venue simply no longer shows it -> cross once" --
        # that assumed the position snapshot shows a fill INSTANTLY. It
        # does not (P382 4/4; 2026-08-27 SELL 1ct -> now=2.0ct): an
        # untracked post-only that LEFT the book with an unmoved snapshot
        # is exactly the ambiguous case where a cross can double the
        # position. Under-target for one tick is the safe direction.
        ad = _FakeAdapter(maker_fills=True, maker_no_oid=True)
        s, _ = _exec_sleeve(ad, cur=0.0, maker_first=True, wait=5.0)
        s.MAKER_FILL_WAIT_SEC = 0.0  # bounded wait, no real sleeping
        res = _run(s.execute_target("SOL", 1))
        assert res["status"] == "FAILED"
        assert res["reason"] == "maker_fill_unresolved_no_cross"
        crosses = [p for p in ad.placed if not getattr(p, "post_only", False)]
        assert crosses == [], "crossed beside a possibly-filled untracked order"

    def test_untracked_gone_with_moved_snapshot_is_a_maker_fill(self):
        # [P420] the companion: the snapshot DID move to target -> the
        # untracked post-only filled -> OK as maker, and no cross.
        ad = _FakeAdapter(maker_fills=True, maker_no_oid=True)
        s, _ = _exec_sleeve(ad, cur=0.0, maker_first=True, wait=5.0,
                            filled_after_maker=1.0)
        res = _run(s.execute_target("SOL", 1))
        assert res["status"] == "OK" and res.get("maker") is True
        crosses = [p for p in ad.placed if not getattr(p, "post_only", False)]
        assert crosses == []

    def test_untracked_still_resting_refuses_the_cross(self):
        # the untracked order rests (no order_id, so the sweep cannot even
        # address it) — the residual check must refuse the cross
        ad = _FakeAdapter(maker_fills=False, maker_no_oid=True)
        s, _ = _exec_sleeve(ad, cur=0.0, maker_first=True, wait=5.0)
        res = _run(s.execute_target("SOL", 1))
        assert res["status"] == "FAILED"
        assert res["reason"] == "maker_untracked_no_cross"
        crosses = [p for p in ad.placed if not getattr(p, "post_only", False)]
        assert crosses == [], (
            "a cross was placed beside the untracked resting post-only — "
            "when it fills, the position overshoots the target (P287)")


# ---------------------------------------------------------------------------
# 4. flat-path orphan cancel failures are loud and distinct
# ---------------------------------------------------------------------------

class TestFlatCancelFailedIsDistinct:
    def test_a_failed_orphan_cancel_is_not_flat_none(self):
        ad = _FakeAdapter(open_orders=[_stop_order()], cancel_ok=False)
        s = _stop_sleeve(ad, signed=0)
        res = _run(s.ensure_protective_stop("SOL"))
        assert res["status"] == "FLAT_CANCEL_FAILED", (
            "a failed orphan cancel reported the same status as 'no orphans "
            "existed' while a plain stop that OPENS a position stayed live")
        assert res["failed_cancels"] == 1

    def test_a_successful_orphan_cancel_is_still_flat_cancelled(self):
        ad = _FakeAdapter(open_orders=[_stop_order()])
        s = _stop_sleeve(ad, signed=0)
        res = _run(s.ensure_protective_stop("SOL"))
        assert res["status"] == "FLAT_CANCELLED"
        assert "S1" in ad.cancelled


# ---------------------------------------------------------------------------
# 5. equity staleness bound
# ---------------------------------------------------------------------------

def _risk_sleeve(eq=3000.0, ts_age_sec=None, start=4000.0, last_dd=0.07):
    s = object.__new__(CoinbaseSleeve)
    s._adapter = None
    s.sleeve_equity_usd = lambda: eq  # type: ignore[assignment]
    s._last_equity_ts = (time.time() - ts_age_sec) if ts_age_sec else None
    s._sleeve_start_equity = start
    s._last_dd_pct = last_dd
    s._halted = False
    s._halt_reason = ""
    s._max_sleeve_drawdown_pct = 0.15
    s._persist_state = lambda: None  # type: ignore[assignment]
    return s


class TestEquityStaleness:
    def test_stale_equity_holds_the_last_drawdown_and_cannot_trip_the_halt(self):
        # (4000-3000)/4000 = 25% would trip the 15% halt — but the equity is
        # 9h stale, so the reading is a frozen number, not a measurement
        s = _risk_sleeve(eq=3000.0, ts_age_sec=9 * 3600)
        out = s.update_risk()
        assert out["degraded"] is True
        assert out["drawdown_pct"] == pytest.approx(0.07)
        assert s._halted is False, (
            "the halt fired on a stale equity reading — staleness must HOLD "
            "the last drawdown, never recompute (P287/P156)")

    def test_fresh_equity_still_evaluates_and_trips(self):
        s = _risk_sleeve(eq=3000.0, ts_age_sec=60.0)
        out = s.update_risk()
        assert out["drawdown_pct"] == pytest.approx(0.25)
        assert s._halted is True

    def test_no_stamp_is_not_stale(self):
        # tests (and any caller that fakes equity without faking the clock)
        # legitimately never stamp the ts; production pairs a missing stamp
        # with equity<=0, which is already refused everywhere
        s = _risk_sleeve(eq=3000.0, ts_age_sec=None)
        out = s.update_risk()
        assert out["drawdown_pct"] == pytest.approx(0.25)

    def test_age_accessor_and_primary_read_stamp(self):
        s = object.__new__(CoinbaseSleeve)
        assert s.sleeve_equity_age_sec() is None
        s._last_equity_ts = time.time() - 120.0
        age = s.sleeve_equity_age_sec()
        assert age is not None and 100.0 < age < 300.0
        # and the PRIMARY equity path is what stamps it
        src = inspect.getsource(CoinbaseSleeve.sleeve_equity_usd)
        assert "_last_equity_ts = time.time()" in src

    def test_stale_equity_falls_back_to_the_fixed_book_for_sizing(self):
        s = object.__new__(CoinbaseSleeve)
        s._target_fraction_by_asset = {"ETH": 0.15}
        s._max_contracts_by_asset = {"ETH": 3}
        s.sleeve_equity_usd = lambda: 10900.0  # type: ignore[assignment]
        s._notional_usd = lambda asset, n: 188.0 * n  # type: ignore[assignment]
        s.signed_contracts = lambda asset: 0.0  # type: ignore[assignment]
        s._last_equity_ts = time.time() - 9 * 3600  # STALE
        assert s.target_for("ETH", 0.9) == 3, (
            "equity-scaled sizing ran on 9h-stale equity — must fall back to "
            "the FIXED (smaller, known-safe) book (P287)")
        s._last_equity_ts = time.time()  # fresh again
        assert s.target_for("ETH", 0.9) == 8

    def test_snapshot_exposes_the_equity_age(self):
        s = object.__new__(CoinbaseSleeve)
        s.reconcile_positions = lambda: {}  # type: ignore[assignment]
        s.buying_power_usd = lambda: 0.0  # type: ignore[assignment]
        s.update_risk = lambda: {}  # type: ignore[assignment]
        s._last_equity_ts = time.time() - 50.0
        snap = s.snapshot()
        assert "equity_age_sec" in snap
        assert 0.0 < snap["equity_age_sec"] < 300.0


# ---------------------------------------------------------------------------
# 6. sizing hysteresis at the contract boundary
# ---------------------------------------------------------------------------

class TestBoundaryHysteresis:
    D = staticmethod(CoinbaseSleeve._dampen_boundary_step)

    def test_entries_from_flat_are_never_dampened(self):
        assert self.D(0.0, 3, 0.15, 3760.0, 188.0) == 3

    def test_flattens_are_never_dampened(self):
        assert self.D(3.0, 0, 0.15, 3760.0, 188.0) == 0

    def test_flips_are_never_dampened(self):
        # flips belong to flip-persistence, not to sizing hysteresis
        assert self.D(3.0, -3, 0.15, 3760.0, 188.0) == -3

    def test_reduces_are_never_dampened(self):
        # P195: de-risking is always free — damping a reduce would hold a
        # LARGER position than target
        assert self.D(4.0, 3, 0.15, 3760.0, 188.0) == 3

    def test_two_contract_adds_are_never_dampened(self):
        assert self.D(2.0, 4, 0.15, 6000.0, 188.0) == 4

    def test_a_boundary_add_without_margin_is_held(self):
        # fr*eq/one_ct = 0.15*3785/188 = 3.020 -> floor 3; under eq*0.97 it
        # is 2.93 -> floor 2 < 3, so the +1 add is boundary noise: hold 2
        assert self.D(2.0, 3, 0.15, 3785.0, 188.0) == 2

    def test_a_boundary_add_with_real_margin_is_emitted(self):
        # fr*eq/one_ct = 0.15*3900/188 = 3.11; under 0.97 still 3.02 -> 3
        assert self.D(2.0, 3, 0.15, 3900.0, 188.0) == 3

    def test_short_side_mirrors(self):
        assert self.D(-2.0, -3, 0.15, 3785.0, 188.0) == -2
        assert self.D(-2.0, -3, 0.15, 3900.0, 188.0) == -3

    def test_target_for_wires_the_damping(self):
        s = object.__new__(CoinbaseSleeve)
        s._target_fraction_by_asset = {"ETH": 0.15}
        s.sleeve_equity_usd = lambda: 3785.0  # type: ignore[assignment]
        s._notional_usd = lambda asset, n: 188.0 * n  # type: ignore[assignment]
        s.signed_contracts = lambda asset: 2.0  # type: ignore[assignment]
        assert s.target_for("ETH", 0.9) == 2, (
            "the boundary +1 add was emitted without the 3% margin — equity "
            "hovering at a floor boundary churns a 1-ct round trip per tick")
        s.sleeve_equity_usd = lambda: 3900.0  # type: ignore[assignment]
        assert s.target_for("ETH", 0.9) == 3

    def test_cap_never_undercuts_the_dampened_target(self):
        # raw >= dampened in every branch: the cap (_sized_contracts) must
        # never block the target target_for just emitted
        s = object.__new__(CoinbaseSleeve)
        s._target_fraction_by_asset = {"ETH": 0.15}
        s.sleeve_equity_usd = lambda: 3785.0  # type: ignore[assignment]
        s._notional_usd = lambda asset, n: 188.0 * n  # type: ignore[assignment]
        s.signed_contracts = lambda asset: 2.0  # type: ignore[assignment]
        assert s._sized_contracts("ETH") >= abs(s.target_for("ETH", 0.9))


# ---------------------------------------------------------------------------
# 7. mark-anchored stop hysteresis
# ---------------------------------------------------------------------------

class TestMarkAnchorHysteresis:
    def _sleeve(self, mid="100.0"):
        ad = _FakeAdapter(mid=mid)
        s = _stop_sleeve(ad, signed=1, pct=0.10, entry=None)
        return s, ad

    def test_small_drift_reuses_the_anchor(self):
        s, ad = self._sleeve("100.0")
        assert s.desired_stop_price("SOL") == pytest.approx(90.0)
        ad.product = {"mid_market_price": "101.0"}  # 1% drift
        assert s.desired_stop_price("SOL") == pytest.approx(90.0), (
            "a 1% mark drift re-anchored the stop — with the 0.5% price "
            "match that is a cancel+replace every tick, each replacement a "
            "brief unprotected window (P287)")

    def test_a_real_move_reanchors(self):
        s, ad = self._sleeve("100.0")
        assert s.desired_stop_price("SOL") == pytest.approx(90.0)
        ad.product = {"mid_market_price": "103.0"}  # 3% > 2% band
        assert s.desired_stop_price("SOL") == pytest.approx(92.7)

    def test_entry_path_is_untouched_and_clears_the_mark_anchor(self):
        s, ad = self._sleeve("100.0")
        assert s.desired_stop_price("SOL") == pytest.approx(90.0)  # mark
        s._last_positions["SOL"]["entry_vwap"] = 80.0
        assert s.desired_stop_price("SOL") == pytest.approx(72.0)  # entry
        assert "SOL" not in getattr(s, "_stop_mark_anchor", {})

    def test_flat_clears_the_anchor(self):
        s, ad = self._sleeve("100.0")
        assert s.desired_stop_price("SOL") == pytest.approx(90.0)
        s._last_positions = {}
        assert s.desired_stop_price("SOL") is None
        assert "SOL" not in getattr(s, "_stop_mark_anchor", {})

    def test_side_change_reanchors(self):
        s, ad = self._sleeve("100.0")
        assert s.desired_stop_price("SOL") == pytest.approx(90.0)
        s._last_positions["SOL"]["signed_contracts"] = -1
        # short side: anchor must not be the stored long anchor's side
        assert s.desired_stop_price("SOL") == pytest.approx(110.0)


# ---------------------------------------------------------------------------
# 8. withdrawal direction is recorded at the halt site
# ---------------------------------------------------------------------------

class TestWithdrawalNote:
    def test_the_halt_site_records_the_withdrawal_direction(self):
        src = inspect.getsource(CoinbaseSleeve.update_risk)
        assert "withdrawal" in src.lower() and "reset_halt" in src, (
            "the withdrawal note is gone — a >15% withdrawal trips the "
            "sticky halt and the operator procedure must be recorded where "
            "the halt lives (P287; P274 documented deposits only)")
