"""[P291] Maker-first reprice ladder — one re-post at the half-way mark.

Both live sleeve orders on 2026-08-16 posted at the touch, sat the full 45s
unfilled and crossed as taker. A passive order that is no longer at the FRONT
of the book will not fill, so waiting the window out buys nothing but
staleness. The ladder cancels and re-posts ONCE at the new touch, out of the
SAME time budget.

Load-bearing properties pinned here (each its own test):

  1. ONE AT A TIME. At most two post-only orders are ever PLACED per
     execute_target call and never two live: the cancel must be CONFIRMED
     before the re-post, and a FAILED cancel refuses BOTH the re-post and the
     cross (P287/P265 double-order class).
  2. NO MOVE -> NO REPRICE. A reprice that fires while we are still at the
     front of the book is pure churn — two venue calls buying nothing.
  3. BUDGET SPLIT, NEVER EXTENDED. The deadline is computed once; a longer
     wait means a staler signal (P265 dead-signal-fill class).
  4. urgent=True still skips the ladder entirely (FORCE_FLAT and the
     fast-risk watchdog need immediacy, not fee savings).
  5. A fill on the REPRICED order is still liquidity="maker" in the P290
     ledger, with the decision context of the price we actually posted at.
  6. NEVER RE-POST BESIDE AN UNKNOWN PARTIAL. If the cancelled order's fill
     state is unreadable or shows a partial, hand back to the caller whose
     venue reconcile sizes the remainder (anti-P139; filled_size is never
     arithmetic'd here, P219).
  7. FLAG-OFF IS BYTE-IDENTICAL to the P270 single-post ladder — including
     taking no extra book read.
"""
import asyncio
import json
import time as _real_time
import types
from pathlib import Path

import pytest

from exchange.coinbase_sleeve import CoinbaseSleeve

PID = "SLP-20DEC30-CDE"


# ---------------------------------------------------------------------------
# deterministic clock: asyncio.sleep advances it, time.monotonic reads it
# ---------------------------------------------------------------------------

class _Clock:
    def __init__(self):
        self.t = 0.0

    def monotonic(self):
        return self.t


def _install_clock(monkeypatch):
    """Fake monotonic + immediate sleep that ADVANCES the clock by the slept
    amount, so the poll loop's timing is exact and the suite is instant."""
    clock = _Clock()

    class _Asyncio:
        def __getattr__(self, name):
            if name == "sleep":
                async def _sleep(secs):
                    clock.t += float(secs)
                    return None
                return _sleep
            return getattr(asyncio, name)

    # proxy the real time module, overriding ONLY monotonic (the recorder
    # uses time.time() for its ts, which must stay real)
    _time_facade = types.SimpleNamespace(
        monotonic=clock.monotonic, time=_real_time.time,
        sleep=_real_time.sleep)
    monkeypatch.setattr("exchange.coinbase_sleeve.asyncio", _Asyncio())
    monkeypatch.setattr("exchange.coinbase_sleeve.time", _time_facade)
    return clock


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------

class _FakeAdapter:
    """Scriptable adapter. `books` is a list of (bid, ask) served one per
    get_best_bid_ask call, the last repeating — so a 'moved' touch is scripted
    rather than guessed."""

    def __init__(self, books=((99.9, 100.1),), cancel_ok=True,
                 fill_after_polls=None, place_fails_on=(), place_raises_on=(),
                 no_oid_on=(), get_order_payload=None, get_order_none=False,
                 book_raises_on=(), mid="100.0"):
        self._cs = 5.0
        self.books = [tuple(b) for b in books]
        self.book_reads = 0
        self.book_raises_on = set(book_raises_on)
        self.cancel_ok = cancel_ok
        self.fill_after_polls = fill_after_polls
        self.place_fails_on = set(place_fails_on)
        self.place_raises_on = set(place_raises_on)
        self.no_oid_on = set(no_oid_on)
        self.get_order_payload = get_order_payload
        self.get_order_none = get_order_none
        self.placed = []
        self.cancelled = []
        self.calls = []          # ordered call log: ("place"|"cancel", arg)
        self.polls = 0
        self.max_concurrent_post_only = 0
        self._open = []
        self.product = {"mid_market_price": mid}

        def _bba(product_ids):
            i = self.book_reads
            self.book_reads += 1
            if i in self.book_raises_on:
                raise ConnectionError("book read down")
            bid, ask = self.books[min(i, len(self.books) - 1)]
            return {"pricebooks": [{"bids": [{"price": str(bid)}],
                                    "asks": [{"price": str(ask)}]}]}

        self._client = types.SimpleNamespace(
            get_product=lambda product_id: self.product,
            get_product_book=lambda product_id, limit: {
                "pricebook": {"bids": [{"price": "99.9"}],
                              "asks": [{"price": "100.1"}]}},
            get_best_bid_ask=_bba,
        )

    def is_connected(self):
        return True

    def to_venue_symbol(self, asset, market="perp"):
        return PID

    def _contract_size(self, pid):
        return self._cs

    async def fetch_open_orders(self, symbol=None):
        self.polls += 1
        if (self.fill_after_polls is not None
                and self.polls > self.fill_after_polls):
            self._open = []          # the resting post-only left the book
        return list(self._open)

    async def cancel_order(self, oid, pid):
        self.calls.append(("cancel", oid))
        if not self.cancel_ok:
            return False
        self.cancelled.append(oid)
        self._open = [o for o in self._open if o.get("order_id") != oid]
        return True

    async def get_order(self, order_id):
        if self.get_order_none:
            return None
        return dict(self.get_order_payload or {"status": "FILLED",
                                               "filled_size": "0",
                                               "average_filled_price": "100.0"})

    async def place_order(self, req):
        n = len(self.placed)          # 0-based index of THIS placement
        self.placed.append(req)
        self.calls.append(("place", getattr(req, "price", None)))
        if n in self.place_raises_on:
            raise ConnectionError("placement transport error")
        if n in self.place_fails_on:
            return types.SimpleNamespace(success=False, order_id=None,
                                         error_code="POST_ONLY_WOULD_CROSS",
                                         error_message="x")
        oid = None if n in self.no_oid_on else f"oid-{n + 1}"
        if getattr(req, "post_only", False):
            if oid:
                self._open.append({"order_id": oid, "side": req.side,
                                   "order_configuration": {
                                       "limit_limit_gtc": {"base_size": "1"}}})
            self.max_concurrent_post_only = max(
                self.max_concurrent_post_only, len(self._open))
        return types.SimpleNamespace(success=True, order_id=oid,
                                     error_code=None, error_message=None)

    # convenience
    def post_only_placements(self):
        return [p for p in self.placed if getattr(p, "post_only", False)]


def _sleeve(adapter, tmp_path, monkeypatch, cur=0.0, wait=20.0,
            reprice=True, filled_after=None):
    monkeypatch.setenv("HMATS_DATA_DIR", str(tmp_path))
    s = object.__new__(CoinbaseSleeve)
    s._adapter = adapter
    s._maker_first = True
    s._maker_wait_sec = wait
    s._maker_reprice = reprice
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
        if filled_after is not None and adapter.placed:
            state["cur"] = filled_after
        return {}
    s.reconcile_positions = _reconcile  # type: ignore[assignment]
    s.can_trade = lambda a, d: (True, "ok")  # type: ignore[assignment]
    return s, state


def _ledger(tmp_path):
    p = Path(tmp_path) / "fill_quality.jsonl"
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()
            if x.strip()]


def _attempt(s, adapter, side="BUY"):
    return asyncio.run(s._maker_first_attempt(
        PID, "SOL", side, 5.0, intended_contracts=1))


# ---------------------------------------------------------------------------
# 1. the happy paths: fill before the reprice, and fill after it
# ---------------------------------------------------------------------------

class TestFillPaths:
    def test_fill_on_the_first_post_never_reprices(self, tmp_path,
                                                   monkeypatch):
        _install_clock(monkeypatch)
        # wait=20 -> reprice at t=10; the post leaves the book at poll 1 (t=5)
        ad = _FakeAdapter(books=[(99.9, 100.1)], fill_after_polls=0)
        s, _ = _sleeve(ad, tmp_path, monkeypatch, wait=20.0)
        assert _attempt(s, ad) == "DONE"
        assert len(ad.post_only_placements()) == 1
        assert ad.cancelled == []                      # nothing to cancel
        assert ad.book_reads == 1                      # no reprice book read
        rows = _ledger(tmp_path)
        assert len(rows) == 1 and rows[0]["liquidity"] == "maker"

    def test_fill_after_reprice_is_still_maker_at_the_new_touch(
            self, tmp_path, monkeypatch):
        _install_clock(monkeypatch)
        # touch moves UP (someone outbids our 99.9): reprice to 99.95, and the
        # repriced order fills.  polls: 1(t5) 2(t10 -> reprice) 3(t15 -> gone)
        ad = _FakeAdapter(books=[(99.9, 100.1), (99.95, 100.1)],
                          fill_after_polls=2)
        s, _ = _sleeve(ad, tmp_path, monkeypatch, wait=20.0)
        assert _attempt(s, ad) == "DONE"
        posts = ad.post_only_placements()
        assert len(posts) == 2, "expected exactly one reprice"
        assert posts[0].price == pytest.approx(99.9)
        assert posts[1].price == pytest.approx(99.95)
        # the cancel of the first order PRECEDES the second placement
        assert ad.calls.index(("cancel", "oid-1")) < ad.calls.index(
            ("place", pytest.approx(99.95)))
        assert ad.max_concurrent_post_only == 1, "two post-onlys were live"
        rows = _ledger(tmp_path)
        assert len(rows) == 1
        assert rows[0]["liquidity"] == "maker"
        assert rows[0]["order_id"] == "oid-2"
        # decision context is the REPRICED touch, not the stale one
        assert rows[0]["decision_bid"] == pytest.approx(99.95)


# ---------------------------------------------------------------------------
# 2. no move -> no reprice (churn pin)
# ---------------------------------------------------------------------------

class TestNoMoveNoReprice:
    def test_static_touch_never_repriced(self, tmp_path, monkeypatch):
        _install_clock(monkeypatch)
        ad = _FakeAdapter(books=[(99.9, 100.1)])          # never moves
        s, _ = _sleeve(ad, tmp_path, monkeypatch, wait=20.0)
        assert _attempt(s, ad) == "DONE"                  # timeout + cancel
        assert len(ad.post_only_placements()) == 1, "repriced without a move"
        # exactly ONE cancel — the timeout one, not a midway churn cancel
        assert ad.cancelled == ["oid-1"]

    def test_sell_side_move_test_is_directional(self, tmp_path, monkeypatch):
        # our resting ASK is 100.1; a HIGHER ask (100.2) means we are still
        # the front of the queue -> no reprice.  Only a LOWER ask displaces us.
        _install_clock(monkeypatch)
        ad = _FakeAdapter(books=[(99.9, 100.1), (99.9, 100.2)])
        s, _ = _sleeve(ad, tmp_path, monkeypatch, wait=20.0)
        assert _attempt(s, ad, side="SELL") == "DONE"
        assert len(ad.post_only_placements()) == 1

        ad2 = _FakeAdapter(books=[(99.9, 100.1), (99.9, 100.05)])
        s2, _ = _sleeve(ad2, tmp_path, monkeypatch, wait=20.0)
        _attempt(s2, ad2, side="SELL")
        assert len(ad2.post_only_placements()) == 2      # displaced -> reprice


# ---------------------------------------------------------------------------
# 3. one at a time: a failed cancel refuses BOTH the re-post and the cross
# ---------------------------------------------------------------------------

class TestFailedCancelRefuses:
    def test_reprice_cancel_failure_returns_cancel_failed(self, tmp_path,
                                                          monkeypatch):
        _install_clock(monkeypatch)
        ad = _FakeAdapter(books=[(99.9, 100.1), (99.95, 100.1)],
                          cancel_ok=False)
        s, _ = _sleeve(ad, tmp_path, monkeypatch, wait=20.0)
        assert _attempt(s, ad) == "CANCEL_FAILED"
        assert len(ad.post_only_placements()) == 1, "re-posted beside a live order"

    def test_execute_target_refuses_the_cross_on_that_verdict(self, tmp_path,
                                                              monkeypatch):
        _install_clock(monkeypatch)
        ad = _FakeAdapter(books=[(99.9, 100.1), (99.95, 100.1)],
                          cancel_ok=False)
        s, _ = _sleeve(ad, tmp_path, monkeypatch, wait=20.0)
        res = asyncio.run(s.execute_target("SOL", 1))
        assert res["status"] == "FAILED"
        assert res["reason"] == "maker_cancel_failed_no_cross"
        assert not [p for p in ad.placed
                    if not getattr(p, "post_only", False)], "crossed anyway"

    def test_reprice_placement_raise_refuses_the_cross(self, tmp_path,
                                                       monkeypatch):
        # the re-post raised: an order MAY rest under an id we never learned
        _install_clock(monkeypatch)
        ad = _FakeAdapter(books=[(99.9, 100.1), (99.95, 100.1)],
                          place_raises_on=(1,))
        s, _ = _sleeve(ad, tmp_path, monkeypatch, wait=20.0)
        res = asyncio.run(s.execute_target("SOL", 1))
        assert res["status"] == "FAILED"
        assert res["reason"] == "maker_reprice_unknown_no_cross"
        assert not [p for p in ad.placed
                    if not getattr(p, "post_only", False)], "crossed anyway"


# ---------------------------------------------------------------------------
# 4. never re-post beside an unknown / partial fill
# ---------------------------------------------------------------------------

class TestNeverRepostBesideAPartial:
    def test_partial_fill_hands_back_instead_of_reposting(self, tmp_path,
                                                          monkeypatch):
        _install_clock(monkeypatch)
        ad = _FakeAdapter(books=[(99.9, 100.1), (99.95, 100.1)],
                          get_order_payload={"status": "CANCELLED",
                                             "filled_size": "0.5",
                                             "average_filled_price": "99.9"})
        s, _ = _sleeve(ad, tmp_path, monkeypatch, wait=20.0)
        assert _attempt(s, ad) == "DONE"
        assert len(ad.post_only_placements()) == 1
        rows = _ledger(tmp_path)
        assert len(rows) == 1 and rows[0]["liquidity"] == "maker"

    def test_unreadable_fill_state_hands_back(self, tmp_path, monkeypatch):
        # get_order -> None (P290 read failure). Absence is not zero (P2):
        # re-posting full size beside an unknown partial would overshoot.
        _install_clock(monkeypatch)
        ad = _FakeAdapter(books=[(99.9, 100.1), (99.95, 100.1)],
                          get_order_none=True)
        s, _ = _sleeve(ad, tmp_path, monkeypatch, wait=20.0)
        assert _attempt(s, ad) == "DONE"
        assert len(ad.post_only_placements()) == 1


# ---------------------------------------------------------------------------
# 5. re-post rejected / untracked
# ---------------------------------------------------------------------------

class TestRepostOutcomes:
    def test_rejected_repost_falls_back_to_the_cross(self, tmp_path,
                                                     monkeypatch):
        # the cancel was CONFIRMED, so nothing rests -> crossing is safe
        _install_clock(monkeypatch)
        ad = _FakeAdapter(books=[(99.9, 100.1), (99.95, 100.1)],
                          place_fails_on=(1,))
        s, _ = _sleeve(ad, tmp_path, monkeypatch, wait=20.0)
        assert _attempt(s, ad) == "FALLBACK"
        assert ad.cancelled == ["oid-1"]

    def test_untracked_repost_returns_done_untracked(self, tmp_path,
                                                     monkeypatch):
        _install_clock(monkeypatch)
        ad = _FakeAdapter(books=[(99.9, 100.1), (99.95, 100.1)],
                          no_oid_on=(1,))
        s, _ = _sleeve(ad, tmp_path, monkeypatch, wait=20.0)
        assert _attempt(s, ad) == "DONE_UNTRACKED"


# ---------------------------------------------------------------------------
# 6. a book-read failure does not consume the opportunity
# ---------------------------------------------------------------------------

class TestBookReadFailure:
    def test_failed_reprice_read_holds_the_post_and_retries(self, tmp_path,
                                                            monkeypatch):
        _install_clock(monkeypatch)
        # book reads: 0 = initial post, 1 = reprice window (RAISES),
        # 2 = next poll (moved) -> the reprice still happens
        ad = _FakeAdapter(books=[(99.9, 100.1), (99.9, 100.1), (99.95, 100.1)],
                          book_raises_on=(1,))
        s, _ = _sleeve(ad, tmp_path, monkeypatch, wait=30.0)
        _attempt(s, ad)
        assert len(ad.post_only_placements()) == 2, (
            "a transient book-read failure consumed the one reprice")


# ---------------------------------------------------------------------------
# 7. budget, urgency, and flag-off byte-identity
# ---------------------------------------------------------------------------

class TestBudgetUrgencyAndFlagOff:
    def test_window_is_split_not_extended(self, tmp_path, monkeypatch):
        clock = _install_clock(monkeypatch)
        ad = _FakeAdapter(books=[(99.9, 100.1), (99.95, 100.1)])
        s, _ = _sleeve(ad, tmp_path, monkeypatch, wait=20.0)
        _attempt(s, ad)
        # bounded by the budget plus at most the final sleep quantum
        assert clock.t <= 20.0 + 5.0 + 1e-9, (
            f"the reprice extended the window to {clock.t}s")

    def test_urgent_skips_the_ladder_entirely(self, tmp_path, monkeypatch):
        _install_clock(monkeypatch)
        ad = _FakeAdapter(books=[(99.9, 100.1), (99.95, 100.1)])
        s, _ = _sleeve(ad, tmp_path, monkeypatch, wait=20.0)
        res = asyncio.run(s.execute_target("SOL", 1, urgent=True))
        assert res["status"] == "OK"
        assert ad.post_only_placements() == [], "posted passively when urgent"

    def test_flag_off_is_byte_identical_to_the_single_post_ladder(
            self, tmp_path, monkeypatch):
        _install_clock(monkeypatch)
        ad = _FakeAdapter(books=[(99.9, 100.1), (99.95, 100.1)])
        s, _ = _sleeve(ad, tmp_path, monkeypatch, wait=20.0, reprice=False)
        assert _attempt(s, ad) == "DONE"                 # timeout + cancel
        assert len(ad.post_only_placements()) == 1
        assert ad.cancelled == ["oid-1"]
        assert ad.book_reads == 1, "flag-off took an extra book read"

    def test_default_is_off_on_the_real_constructor(self):
        import inspect
        sig = inspect.signature(CoinbaseSleeve.__init__)
        assert "maker_reprice" in sig.parameters
        assert sig.parameters["maker_reprice"].default is False

    def test_a_sleeve_without_the_attribute_still_runs_the_old_ladder(
            self, tmp_path, monkeypatch):
        # [P85] The ctor is not the only way a sleeve exists — operator
        # scripts and every test fixture build one via object.__new__. An
        # undefended read raises AttributeError, which _maker_first_attempt's
        # outer handler swallows into "FALLBACK": that instance would cross
        # at TAKER on every order while the logs said "maker attempt error".
        # Missing attribute must mean OFF, not degraded-to-taker.
        _install_clock(monkeypatch)
        ad = _FakeAdapter(books=[(99.9, 100.1), (99.95, 100.1)])
        s, _ = _sleeve(ad, tmp_path, monkeypatch, wait=20.0)
        del s._maker_reprice
        assert _attempt(s, ad) == "DONE"                 # NOT "FALLBACK"
        assert len(ad.post_only_placements()) == 1
        assert ad.cancelled == ["oid-1"]
