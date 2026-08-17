"""[P290] Sleeve fill-quality logging — realized-fill cost for CDE.

P289 found no realized-fill cost measurement existed for the live venue
(fill_quality.jsonl empty; P169's [FILL_VS_MID] lives on the dormant Kraken
path), so the CDE spread table is certified by order-book probes only. This
suite pins the new instrument's load-bearing properties:

  1. SIGN CONVENTION: realized_slippage_bps is positive whenever the fill
     was worse than the decision mid FOR OUR SIDE — a BUY above mid and a
     SELL below mid must BOTH read positive (= paid). A flipped sign would
     make bad fills look like rebates and eventually re-derive the spread
     constants in the wrong direction.
  2. ABSENCE STAYS ABSENT (P2): an unreadable order records
     status="unresolved" with NO fill/slippage fields — never a fabricated
     measurement.
  3. IRON LAW 7: a raising recorder must not change execute_target's return
     value — the ledger is observation, the order path is money.
  4. The reader refuses below min-n (P199/P278: missing data is never a
     verdict) and its local CDE table cannot drift from constitution's
     (P192 two-file guard — the script is stdlib-only by design).
"""
import asyncio
import importlib.util
import json
import types
from pathlib import Path

import pytest

from exchange.coinbase_sleeve import CoinbaseSleeve

PID = "SLP-20DEC30-CDE"
REPO = Path(__file__).resolve().parent.parent


class _FastAsyncio:
    """asyncio facade whose sleep returns immediately (the maker poll loop
    sleeps 5s per pass; real sleeping would make this suite minutes long)."""

    def __getattr__(self, name):
        if name == "sleep":
            async def _sleep(_secs):
                return None
            return _sleep
        return getattr(asyncio, name)


class _FakeAdapter:
    def __init__(self, mid="100.0", order_payload=None, maker_fills=False,
                 get_order_none=False):
        self._cs = 5.0
        self.placed = []
        self.cancelled = []
        self._open = []
        self.maker_fills = maker_fills
        self.get_order_none = get_order_none
        self.order_payload = order_payload
        self.get_order_calls = []
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
        return list(self._open)

    async def cancel_order(self, oid, pid):
        self.cancelled.append(oid)
        self._open = [o for o in self._open if o.get("order_id") != oid]
        return True

    async def get_order(self, order_id):
        self.get_order_calls.append(order_id)
        if self.get_order_none:
            return None
        return dict(self.order_payload or {})

    async def place_order(self, req):
        self.placed.append(req)
        oid = f"oid-{len(self.placed)}"
        if getattr(req, "post_only", False) and not self.maker_fills:
            self._open.append({"order_id": oid, "side": req.side,
                               "order_configuration": {
                                   "limit_limit_gtc": {"base_size": "1"}}})
        return types.SimpleNamespace(success=True, order_id=oid,
                                     error_code=None, error_message=None)


def _sleeve(adapter, tmp_path, monkeypatch, cur=0.0, maker_first=False,
            wait=0.01, filled_after=None):
    monkeypatch.setenv("HMATS_DATA_DIR", str(tmp_path))
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
        if filled_after is not None and adapter.placed:
            state["cur"] = filled_after
        return {}
    s.reconcile_positions = _reconcile  # type: ignore[assignment]
    s.can_trade = lambda a, d: (True, "ok")  # type: ignore[assignment]
    return s, state


def _rows(tmp_path):
    p = tmp_path / "fill_quality.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()
            if l.strip()]


FILLED_BUY_ABOVE_MID = {"status": "FILLED", "average_filled_price": "100.2",
                        "filled_size": "1", "total_fees": "0.42"}
FILLED_SELL_BELOW_MID = {"status": "FILLED", "average_filled_price": "99.8",
                         "filled_size": "1"}


class TestSignConvention:
    def test_buy_filled_above_mid_reads_positive(self, tmp_path, monkeypatch):
        ad = _FakeAdapter(order_payload=FILLED_BUY_ABOVE_MID)
        s, _ = _sleeve(ad, tmp_path, monkeypatch, cur=0.0)
        r = asyncio.run(s.execute_target("SOL", 1))
        assert r["status"] == "OK"
        rows = _rows(tmp_path)
        assert len(rows) == 1
        rec = rows[0]
        assert rec["status"] == "filled"
        assert rec["liquidity"] == "direct"
        assert rec["side"] == "BUY"
        # fill 100.2 vs mid 100.0 on a BUY -> paid 20bps -> POSITIVE
        assert rec["realized_slippage_bps"] == pytest.approx(20.0, abs=0.1)
        assert rec["fees_usd"] == pytest.approx(0.42)

    def test_sell_filled_below_mid_also_reads_positive(self, tmp_path,
                                                       monkeypatch):
        ad = _FakeAdapter(order_payload=FILLED_SELL_BELOW_MID)
        s, _ = _sleeve(ad, tmp_path, monkeypatch, cur=0.0)
        r = asyncio.run(s.execute_target("SOL", -1))
        assert r["status"] == "OK"
        rec = _rows(tmp_path)[0]
        assert rec["side"] == "SELL"
        # fill 99.8 vs mid 100.0 on a SELL -> paid 20bps -> POSITIVE
        assert rec["realized_slippage_bps"] == pytest.approx(20.0, abs=0.1)


class TestLedgerShape:
    def test_record_carries_ts_iso_and_decision_context(self, tmp_path,
                                                        monkeypatch):
        ad = _FakeAdapter(order_payload=FILLED_BUY_ABOVE_MID)
        s, _ = _sleeve(ad, tmp_path, monkeypatch)
        asyncio.run(s.execute_target("SOL", 1))
        rec = _rows(tmp_path)[0]
        assert isinstance(rec["ts"], float)
        assert "T" in rec["iso"]  # ISO datetime
        assert rec["decision_mid"] == pytest.approx(100.0)
        assert rec["decision_bid"] == pytest.approx(99.9)
        assert rec["decision_ask"] == pytest.approx(100.1)
        # spread (100.1-99.9)/100 = 20bps
        assert rec["decision_spread_bps"] == pytest.approx(20.0, abs=0.1)
        assert rec["order_id"] == "oid-1"
        assert rec["contracts"] == 1
        # filled_size recorded VERBATIM, no unit guessing (P219)
        assert rec["filled_size_raw"] == "1"

    def test_maker_fill_records_maker_liquidity(self, tmp_path, monkeypatch):
        # maker path: post-only accepted, poll finds it gone, get_order says
        # FILLED at the bid (99.9) -> maker EARNED half-spread -> negative.
        monkeypatch.setattr("exchange.coinbase_sleeve.asyncio",
                            _FastAsyncio())
        ad = _FakeAdapter(maker_fills=True,
                          order_payload={"status": "FILLED",
                                         "average_filled_price": "99.9",
                                         "filled_size": "1"})
        s, _ = _sleeve(ad, tmp_path, monkeypatch, cur=0.0, maker_first=True,
                       wait=0.5, filled_after=1.0)
        r = asyncio.run(s.execute_target("SOL", 1))
        assert r["status"] == "OK"
        assert r.get("maker") is True
        recs = _rows(tmp_path)
        assert len(recs) == 1
        rec = recs[0]
        assert rec["liquidity"] == "maker"
        # BUY at 99.9 vs mid 100.0 -> earned 10bps -> NEGATIVE
        assert rec["realized_slippage_bps"] == pytest.approx(-10.0, abs=0.1)

    def test_taker_cross_after_maker_timeout_records_taker_cross(
            self, tmp_path, monkeypatch):
        monkeypatch.setattr("exchange.coinbase_sleeve.asyncio",
                            _FastAsyncio())
        # post-only rests (maker_fills=False), timeout cancels (no partial:
        # payload has filled_size 0), then the cross fills.
        ad = _FakeAdapter(maker_fills=False,
                          order_payload={"status": "FILLED",
                                         "average_filled_price": "100.2",
                                         "filled_size": "1"})
        # cancel-partial hook reads filled_size from get_order; make the
        # FIRST lookup (the cancelled post-only) show zero fill:
        payloads = [{"status": "CANCELLED", "filled_size": "0"},
                    {"status": "FILLED", "average_filled_price": "100.2",
                     "filled_size": "1"}]

        async def _get_order(order_id):
            ad.get_order_calls.append(order_id)
            return dict(payloads[min(len(ad.get_order_calls) - 1,
                                     len(payloads) - 1)])
        ad.get_order = _get_order  # type: ignore[assignment]
        s, _ = _sleeve(ad, tmp_path, monkeypatch, cur=0.0, maker_first=True,
                       wait=0.01)
        r = asyncio.run(s.execute_target("SOL", 1))
        assert r["status"] == "OK"
        recs = _rows(tmp_path)
        assert len(recs) == 1  # cancelled-unfilled post-only records nothing
        assert recs[0]["liquidity"] == "taker_cross"
        assert recs[0]["realized_slippage_bps"] == pytest.approx(20.0,
                                                                 abs=0.1)


class TestAbsenceStaysAbsent:
    def test_unreadable_order_records_unresolved_with_no_fill_fields(
            self, tmp_path, monkeypatch):
        ad = _FakeAdapter(get_order_none=True)
        s, _ = _sleeve(ad, tmp_path, monkeypatch)
        r = asyncio.run(s.execute_target("SOL", 1))
        assert r["status"] == "OK"
        rec = _rows(tmp_path)[0]
        assert rec["status"] == "unresolved"
        assert rec["fill_avg_price"] is None
        assert rec["realized_slippage_bps"] is None
        assert rec["order_id"] == "oid-1"  # the id IS recorded for follow-up


class TestIronLaw7:
    def test_raising_recorder_does_not_change_the_order_path(
            self, tmp_path, monkeypatch):
        ad = _FakeAdapter(order_payload=FILLED_BUY_ABOVE_MID)
        s, _ = _sleeve(ad, tmp_path, monkeypatch)
        baseline = asyncio.run(s.execute_target("SOL", 1))

        ad2 = _FakeAdapter(order_payload=FILLED_BUY_ABOVE_MID)
        s2, _ = _sleeve(ad2, tmp_path, monkeypatch)

        def _boom(*a, **k):
            raise RuntimeError("ledger disk full")
        s2._record_fill_quality = _boom  # type: ignore[assignment]
        r2 = asyncio.run(s2.execute_target("SOL", 1))
        assert r2["status"] == baseline["status"] == "OK"
        assert r2["contracts"] == baseline["contracts"]
        assert len(ad2.placed) == len(ad.placed) == 1

    def test_book_read_failure_degrades_to_none_context(self, tmp_path,
                                                        monkeypatch):
        ad = _FakeAdapter(order_payload=FILLED_BUY_ABOVE_MID)

        def _bad_book(product_id, limit):
            raise ConnectionError("book down")
        ad._client.get_product_book = _bad_book
        s, _ = _sleeve(ad, tmp_path, monkeypatch)
        r = asyncio.run(s.execute_target("SOL", 1))
        assert r["status"] == "OK"
        rec = _rows(tmp_path)[0]
        assert rec["decision_bid"] is None
        assert rec["decision_ask"] is None
        assert rec["decision_spread_bps"] is None
        # the fill measurement itself is still honest
        assert rec["realized_slippage_bps"] == pytest.approx(20.0, abs=0.1)


def _load_reader():
    spec = importlib.util.spec_from_file_location(
        "fill_quality_review_under_test",
        REPO / "scripts" / "fill_quality_review.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestReader:
    def _ledger(self, tmp_path, n_filled, asset="SOL", slip=1.0,
                liquidity="taker_cross"):
        p = tmp_path / "fill_quality.jsonl"
        with open(p, "a", encoding="utf-8") as f:
            for i in range(n_filled):
                f.write(json.dumps({
                    "ts": 1.0 + i, "iso": "2026-08-17T00:00:00+00:00",
                    "asset": asset, "order_id": f"o{i}", "side": "BUY",
                    "contracts": 1, "liquidity": liquidity, "urgent": False,
                    "decision_mid": 100.0, "decision_bid": 99.9,
                    "decision_ask": 100.1, "decision_spread_bps": 20.0,
                    "status": "filled", "fill_avg_price": 100.0 + slip / 100,
                    "realized_slippage_bps": slip, "fees_usd": None,
                    "filled_size_raw": "1"}) + "\n")
        return p

    def test_refuses_below_min_n(self, tmp_path, monkeypatch, capsys):
        p = self._ledger(tmp_path, 5)
        mod = _load_reader()
        monkeypatch.setattr("sys.argv",
                            ["x", "--path", str(p), "--min-n", "20"])
        assert mod.main() == 2
        out = capsys.readouterr().out
        assert "REFUSING VERDICT" in out

    def test_missing_file_refuses_distinctly(self, tmp_path, monkeypatch,
                                             capsys):
        mod = _load_reader()
        monkeypatch.setattr("sys.argv",
                            ["x", "--path", str(tmp_path / "absent.jsonl")])
        assert mod.main() == 2
        assert "no ledger" in capsys.readouterr().out

    def test_aggregates_and_states_the_rederivation_rule(
            self, tmp_path, monkeypatch, capsys):
        p = self._ledger(tmp_path, 25, slip=1.5)
        mod = _load_reader()
        monkeypatch.setattr("sys.argv",
                            ["x", "--path", str(p), "--min-n", "20"])
        assert mod.main() == 0
        out = capsys.readouterr().out
        assert "SOL" in out and "25" in out
        # realized 1.5bps < charged 4.0 at n>=20 -> eligibility NAMED, with
        # the never-automatic caveat
        assert "re-derivation ELIGIBLE" in out
        assert "never automatic" in out
        assert "P-entry" in out

    def test_cde_table_cannot_drift_from_constitution(self):
        # [P192] The reader restates the table (stdlib-only script); the two
        # copies must stay equal or the report compares against fiction.
        from defense.constitution import FrictionComponents
        mod = _load_reader()
        assert mod.CDE_SPREAD_BPS == FrictionComponents().CDE_SPREAD_BPS
