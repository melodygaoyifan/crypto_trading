"""[P382] The protective stop must never be sized LARGER than the position it
guards, and an intent/snapshot disagreement must be re-checked within the
30s loop rather than left for the next 4H tick.

Observed live 2026-08-22 (venue-verified): every taker-cross leg that day
read its own position back as the PRE-trade size (`execute_target ETH SELL
2ct -> now=5.0ct`), so `ensure_protective_stop` — which sizes from that
snapshot — placed a 5ct SELL stop on a 3ct long (2h28m until the next tick
re-sized it), and later a 2ct stop on a 1ct SOL long. CDE rejects
reduce_only: a touch closes the long and OPENS the difference as a short,
with no gate. P207 covered intent==0 and P265 covered sign flips; the
same-sign size mismatch fell through.

Rules pinned here:
  * same sign, different size -> the stop is sized by the SMALLER of the two
    (oversized opens the opposite side; undersized under-protects for ~34s)
    and a follow-up is requested;
  * entry from flat with a snapshot still at zero -> one fresh reconcile,
    then NO stop this pass (a stop sized to the intent is the P207 orphan if
    the entry strands) and a follow-up is requested;
  * the follow-up settles the moment the venue agrees with the intent, and
    gives up (sizing to the venue, loudly) after a bounded wait;
  * the P207 / P265 carve-outs behave exactly as before.
"""
import asyncio
import types

import pytest

from exchange.coinbase_sleeve import CoinbaseSleeve

PID = "SLP-20DEC30-CDE"


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
                                     error_message="boom",
                                     order_id="S1")


def _sleeve(adapter, signed, pct=0.10, entry=72.0, reconcile_ok=True,
            venue_sequence=None):
    """`venue_sequence`: successive signed sizes `reconcile_positions()` will
    reveal (simulating the venue's lagging position read). When exhausted the
    last value repeats."""
    s = object.__new__(CoinbaseSleeve)
    s._adapter = adapter
    s._assets = ("SOL",)
    s._protective_stop_pct = pct
    s._protective_stop_assets = None
    s._reconcile_ok = reconcile_ok
    s._last_positions = ({} if signed == 0 else
                         {"SOL": {"product_id": PID, "signed_contracts": signed,
                                  "entry_vwap": entry}})
    seq = list(venue_sequence or [])

    def _reconcile():
        if seq:
            v = seq.pop(0)
            s._last_positions = ({} if v == 0 else
                                 {"SOL": {"product_id": PID,
                                          "signed_contracts": v,
                                          "entry_vwap": entry}})
        s._reconcile_ok = reconcile_ok
        return dict(s._last_positions)
    s.reconcile_positions = _reconcile
    return s


def _placed_size(adapter):
    return [float(r.size) for r in adapter.placed]


class TestSameSignSizeMismatch:
    def test_the_live_incident_5ct_stop_on_a_3ct_long_is_sized_to_3(self):
        a = _FakeAdapter()
        s = _sleeve(a, signed=5)                         # stale: 5 (real: 3)
        res = asyncio.run(s.ensure_protective_stop("SOL", intended_target=3))
        assert res["status"] == "PLACED", res
        assert res["contracts"] == 3
        assert _placed_size(a) == [3 * 5.0], (
            "a 5ct SELL stop on a 3ct long: a touch closes 3 and OPENS a 2ct "
            "short (no reduce_only) — the P382 incident")
        assert "SOL" in s.stop_followup_pending()

    def test_an_add_in_flight_is_sized_to_the_snapshot_not_the_intent(self):
        # intent 5, snapshot 3 (the BUY has not filled): a 5ct stop would be
        # oversized if the add strands — size to the snapshot
        a = _FakeAdapter()
        s = _sleeve(a, signed=3)
        res = asyncio.run(s.ensure_protective_stop("SOL", intended_target=5))
        assert res["status"] == "PLACED"
        assert res["contracts"] == 3
        assert _placed_size(a) == [3 * 5.0]

    def test_short_side_keeps_the_snapshot_sign(self):
        a = _FakeAdapter()
        s = _sleeve(a, signed=-4)
        res = asyncio.run(s.ensure_protective_stop("SOL", intended_target=-2))
        assert res["status"] == "PLACED"
        assert res["contracts"] == -2
        assert res["side"] == "BUY"

    def test_the_stop_is_never_sized_larger_than_either_number(self):
        for snap, intent in ((5, 3), (3, 5), (2, 1), (1, 2), (-4, -2), (-2, -4)):
            a = _FakeAdapter()
            s = _sleeve(a, signed=snap)
            res = asyncio.run(
                s.ensure_protective_stop("SOL", intended_target=intent))
            assert res["status"] == "PLACED", (snap, intent, res)
            assert abs(res["contracts"]) == min(abs(snap), abs(intent)), \
                (snap, intent, res)

    def test_agreeing_sizes_behave_exactly_as_before(self):
        a = _FakeAdapter()
        s = _sleeve(a, signed=3)
        res = asyncio.run(s.ensure_protective_stop("SOL", intended_target=3))
        assert res["status"] == "PLACED" and res["contracts"] == 3
        assert "SOL" not in s.stop_followup_pending()

    def test_no_intent_still_uses_the_snapshot(self):
        a = _FakeAdapter()
        s = _sleeve(a, signed=5)
        res = asyncio.run(s.ensure_protective_stop("SOL"))
        assert res["status"] == "PLACED" and res["contracts"] == 5


class TestEntryInTransition:
    def test_entry_from_flat_with_a_zero_snapshot_places_nothing_and_requests_followup(self):
        a = _FakeAdapter()
        s = _sleeve(a, signed=0, venue_sequence=[0])   # fresh reconcile: still 0
        res = asyncio.run(s.ensure_protective_stop("SOL", intended_target=2))
        assert res["status"] == "ENTRY_IN_TRANSITION", res
        assert a.placed == [], "a stop sized to the intent is the P207 orphan"
        assert s.stop_followup_pending() == {"SOL": 2.0}

    def test_a_fresh_reconcile_that_sees_the_fill_places_the_stop_now(self):
        a = _FakeAdapter()
        s = _sleeve(a, signed=0, venue_sequence=[2])   # fresh reconcile: 2
        res = asyncio.run(s.ensure_protective_stop("SOL", intended_target=2))
        assert res["status"] == "PLACED" and res["contracts"] == 2

    def test_a_fresh_reconcile_showing_the_opposite_side_is_the_flip_window(self):
        a = _FakeAdapter()
        s = _sleeve(a, signed=0, venue_sequence=[-1])
        res = asyncio.run(s.ensure_protective_stop("SOL", intended_target=2))
        assert res["status"] == "FLIP_IN_TRANSITION"
        assert a.placed == []


class TestFollowup:
    def test_followup_settles_once_the_venue_agrees(self):
        a = _FakeAdapter()
        # tick: stale 5, intent 3 -> stop sized 3, follow-up requested
        s = _sleeve(a, signed=5, venue_sequence=[5, 3])
        asyncio.run(s.ensure_protective_stop("SOL", intended_target=3))
        assert "SOL" in s.stop_followup_pending()
        # the previously placed 3ct stop now rests at the venue
        a._open = [{"order_id": "S1", "side": "SELL",
                    "order_configuration": {"stop_limit_stop_limit_gtc":
                                            {"base_size": "3", "stop_price": "64.8"}}}]
        # 30s loop: first pass venue still 5 -> PENDING; second pass 3 -> settled
        r1 = asyncio.run(s.followup_protective_stop("SOL"))
        assert r1["status"] == "PENDING", r1
        r2 = asyncio.run(s.followup_protective_stop("SOL"))
        assert r2["status"] == "OK_EXISTS", r2
        assert "SOL" not in s.stop_followup_pending()

    def test_followup_after_a_stranded_flatten_restores_the_stop(self):
        # watchdog exit swept the stop; the exit did NOT fill (venue still 2)
        a = _FakeAdapter()
        s = _sleeve(a, signed=2, venue_sequence=[2, 2])
        s.request_stop_followup("SOL", 0.0)
        r1 = asyncio.run(s.followup_protective_stop("SOL"))
        assert r1["status"] == "PENDING"          # venue disagrees with intent 0
        # ...bounded: force the wait past the max and it sizes to the VENUE
        s._stop_followup["SOL"] = (0.0, 0.0)      # requested "long ago"
        r2 = asyncio.run(s.followup_protective_stop("SOL"))
        assert r2["status"] == "PLACED" and r2["contracts"] == 2, r2
        assert "SOL" not in s.stop_followup_pending()

    def test_followup_after_a_filled_flatten_cancels_the_orphan(self):
        a = _FakeAdapter(open_orders=[{
            "order_id": "S9", "side": "SELL",
            "order_configuration": {"stop_limit_stop_limit_gtc":
                                    {"base_size": "2", "stop_price": "64.8"}}}])
        s = _sleeve(a, signed=2, venue_sequence=[0])
        s.request_stop_followup("SOL", 0.0)
        r = asyncio.run(s.followup_protective_stop("SOL"))
        assert r["status"] == "FLAT_CANCELLED"
        assert a.cancelled == ["S9"]

    def test_followup_is_a_noop_when_nothing_is_pending(self):
        a = _FakeAdapter()
        s = _sleeve(a, signed=2)
        assert asyncio.run(s.followup_protective_stop("SOL")) is None
        assert a.placed == [] and a.cancelled == []

    def test_followup_refuses_on_a_stale_reconcile(self):
        a = _FakeAdapter()
        s = _sleeve(a, signed=5, reconcile_ok=False)
        s.request_stop_followup("SOL", 3.0)
        r = asyncio.run(s.followup_protective_stop("SOL"))
        assert r["status"] == "SKIPPED_STALE"
        assert "SOL" in s.stop_followup_pending()   # still pending, retried

    def test_request_never_raises_even_on_a_bare_sleeve(self):
        s = object.__new__(CoinbaseSleeve)
        s.request_stop_followup("SOL", 1.0)      # no _stop_followup attr yet
        assert s.stop_followup_pending() == {"SOL": 1.0}

    def test_max_wait_is_minutes_not_hours(self):
        assert 60.0 <= CoinbaseSleeve.STOP_FOLLOWUP_MAX_SEC <= 1800.0


class TestTheCarveOutsStillHold:
    def test_p207_intended_flat_still_cancels_and_requests_followup(self):
        a = _FakeAdapter(open_orders=[{
            "order_id": "O1", "side": "SELL",
            "order_configuration": {"stop_limit_stop_limit_gtc":
                                    {"base_size": "1", "stop_price": "64.8"}}}])
        s = _sleeve(a, signed=1)
        res = asyncio.run(s.ensure_protective_stop("SOL", intended_target=0))
        assert res["status"] == "FLAT_CANCELLED"
        assert a.cancelled == ["O1"] and a.placed == []
        assert s.stop_followup_pending() == {"SOL": 0.0}

    def test_p265_flip_in_flight_still_places_nothing(self):
        a = _FakeAdapter()
        s = _sleeve(a, signed=1)
        res = asyncio.run(s.ensure_protective_stop("SOL", intended_target=-1))
        assert res["status"] == "FLIP_IN_TRANSITION"
        assert a.placed == []
        assert s.stop_followup_pending() == {"SOL": -1.0}


class TestManageReportsTheTargetItDroveTo:
    """[P382] The driver's stop reconcile must receive the target manage_to_signal
    actually drove to (post-conviction, post-boundary-damping). Recomputing
    `target_for(asset, dir)` without conviction gave a raw 6 against a dampened
    4 on the first live tick after the fix — read as a fill lag, arming a
    follow-up that could only give up 10 minutes later."""

    def test_manage_result_carries_the_sent_target(self):
        s = _sleeve(_FakeAdapter(), signed=1)
        s._flip_persist_ticks = 1
        s._flip_pending = {}
        s.target_for = lambda asset, direction, threshold=0.15, conviction=1.0: 2
        calls = []

        async def _exec(asset, target, **kw):
            calls.append(target)
            return {"status": "OK", "asset": asset}
        s.execute_target = _exec
        res = asyncio.run(s.manage_to_signal("SOL", 1.0, conviction=0.5))
        assert calls == [2]
        assert res["target"] == 2

    def test_an_existing_target_key_is_not_overwritten(self):
        s = _sleeve(_FakeAdapter(), signed=1)
        s._flip_persist_ticks = 1
        s._flip_pending = {}
        s.target_for = lambda *a, **k: 3

        async def _exec(asset, target, **kw):
            return {"status": "OK", "target": 99}
        s.execute_target = _exec
        assert asyncio.run(s.manage_to_signal("SOL", 1.0))["target"] == 99
