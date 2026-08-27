"""[P420] Two order-path gaps found in the 2026-08-27 read-through.

  [task 3] Stale NON-STOP orders were swept only inside execute_target's
  NOOP/BLOCKED branches. Every path returning BEFORE execute_target
  (FLIP_DEFERRED, RESIZE_DEFERRED, the driver's HOLD/cooldown/blocked
  branches, the stop follow-up's give-up path) left an unfilled limit from
  a previous tick resting indefinitely (P265 dead-signal-fill class).
  `sweep_stale_entries(asset)` is the PUBLIC wrapper: resolves the pid,
  refuses on a stale reconcile, never raises, never touches a stop.

  [task 6] A venue position on a product not in SYMBOL_MAP was `continue`d
  out of reconcile_positions -- invisible to the net-exposure cap. It is
  now counted as UNPRICED (priced_ok=False -> the cap fails OPEN, P208)
  and warned once per pid per process. No mapping is fabricated.
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
STOP = {"order_id": "stop-1", "side": "SELL",
        "order_configuration": {"stop_limit_stop_limit_gtc": {
            "base_size": "1", "stop_price": "80"}}}
ENTRY = {"order_id": "entry-1", "side": "BUY",
         "order_configuration": {"limit_limit_gtc": {"base_size": "1"}}}


@pytest.fixture(autouse=True)
def _private_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HMATS_DATA_DIR", str(tmp_path))


class _FakeAdapter:
    def __init__(self, open_orders=(), fetch_raises=False, cancel_ok=True):
        self._open = list(open_orders)
        self.fetch_raises = fetch_raises
        self.cancel_ok = cancel_ok
        self.cancelled = []

    def is_connected(self):
        return True

    def to_venue_symbol(self, asset, market="perp"):
        return PID

    async def fetch_open_orders(self, symbol=None):
        if self.fetch_raises:
            raise ConnectionError("listing down")
        return list(self._open)

    async def cancel_order(self, oid, pid):
        if not self.cancel_ok:
            return False
        self.cancelled.append(oid)
        self._open = [o for o in self._open if o.get("order_id") != oid]
        return True


def _sleeve(adapter, reconcile_ok=True):
    s = object.__new__(CoinbaseSleeve)
    s._adapter = adapter
    s._reconcile_ok = reconcile_ok
    s.is_ready = lambda: True
    return s


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# [task 3a] the public wrapper
# ---------------------------------------------------------------------------

class TestSweepStaleEntries:
    def test_a_resting_entry_is_cancelled(self):
        ad = _FakeAdapter(open_orders=[ENTRY])
        s = _sleeve(ad)
        assert _run(s.sweep_stale_entries("SOL")) == 1
        assert ad.cancelled == ["entry-1"]

    def test_a_stop_is_never_cancelled(self):
        ad = _FakeAdapter(open_orders=[STOP, ENTRY])
        s = _sleeve(ad)
        assert _run(s.sweep_stale_entries("SOL")) == 1
        assert ad.cancelled == ["entry-1"], "the protective stop was swept"

    def test_stale_reconcile_refuses_and_cancels_nothing(self):
        ad = _FakeAdapter(open_orders=[ENTRY])
        s = _sleeve(ad, reconcile_ok=False)
        assert _run(s.sweep_stale_entries("SOL")) is None
        assert ad.cancelled == []

    def test_missing_reconcile_flag_reads_as_stale(self):
        ad = _FakeAdapter(open_orders=[ENTRY])
        s = _sleeve(ad)
        del s._reconcile_ok
        assert _run(s.sweep_stale_entries("SOL")) is None
        assert ad.cancelled == []

    def test_no_adapter_returns_none_never_raises(self):
        s = object.__new__(CoinbaseSleeve)
        s._reconcile_ok = True
        assert _run(s.sweep_stale_entries("SOL")) is None

    def test_unreadable_book_returns_none(self):
        ad = _FakeAdapter(open_orders=[ENTRY], fetch_raises=True)
        s = _sleeve(ad)
        assert _run(s.sweep_stale_entries("SOL")) is None
        assert ad.cancelled == []

    def test_a_raising_pid_lookup_is_caught(self):
        ad = _FakeAdapter(open_orders=[ENTRY])
        ad.to_venue_symbol = lambda a, m="perp": (_ for _ in ()).throw(
            KeyError(a))
        s = _sleeve(ad)
        assert _run(s.sweep_stale_entries("XYZ")) is None

    def test_clean_book_returns_zero(self):
        s = _sleeve(_FakeAdapter())
        assert _run(s.sweep_stale_entries("SOL")) == 0


# ---------------------------------------------------------------------------
# [task 3c] the stop follow-up's give-up path sweeps BEFORE sizing the stop
# ---------------------------------------------------------------------------

class TestFollowupGiveUpSweeps:
    def test_sweep_precedes_the_snapshot_sized_stop(self, monkeypatch):
        s = _sleeve(_FakeAdapter())
        s._stop_followup = {"SOL": (0.0, 0.0)}  # intended flat, since epoch
        s._stop_enabled_for = lambda a: True
        s.reconcile_positions = lambda: {}
        s.signed_contracts = lambda a: 1.0        # venue still shows 1ct
        order = []

        async def _sweep(asset):
            order.append(("sweep", asset))
            return 1
        s.sweep_stale_entries = _sweep

        async def _ensure(asset, intended_target=None):
            order.append(("stop", asset, intended_target))
            return {"status": "PLACED", "asset": asset}
        s.ensure_protective_stop = _ensure
        res = _run(s.followup_protective_stop("SOL"))
        assert res["status"] == "PLACED"
        assert order == [("sweep", "SOL"), ("stop", "SOL", None)], order


# ---------------------------------------------------------------------------
# [task 6] unmapped venue positions are UNPRICED, not invisible
# ---------------------------------------------------------------------------

def _pos(pid, side, n):
    return {"product_id": pid, "side": side, "number_of_contracts": str(n),
            "avg_entry_price": "100", "current_price": "100",
            "unrealized_pnl": "0"}


def _reconcile_sleeve(positions):
    s = object.__new__(CoinbaseSleeve)
    s._adapter = types.SimpleNamespace(
        is_connected=lambda: True,
        _client=types.SimpleNamespace(
            list_futures_positions=lambda: {"positions": positions}))
    s._pid_to_asset = {PID: "SOL"}
    s._last_positions = {}
    s._last_equity_usd = 4000.0
    s._last_equity_ts = None
    s._reconcile_ok = False
    return s


class TestUnmappedPositions:
    def test_unmapped_position_is_recorded_and_the_mapped_one_kept(self):
        s = _reconcile_sleeve([_pos(PID, "LONG", 1),
                               _pos("XYZ-20DEC30-CDE", "SHORT", 2)])
        out = s.reconcile_positions()
        assert s._reconcile_ok is True
        assert out["SOL"]["signed_contracts"] == 1
        assert s._unmapped_positions == {"XYZ-20DEC30-CDE": 2.0}
        assert "XYZ-20DEC30-CDE" not in out, "a mapping was fabricated"

    def test_exposure_reads_unpriced_so_the_cap_fails_open(self):
        s = _reconcile_sleeve([_pos("XYZ-20DEC30-CDE", "SHORT", 2)])
        s.reconcile_positions()
        s._equity_is_stale = lambda: False
        s.sleeve_equity_age_sec = lambda: 0.0
        exp = s.sleeve_exposure()
        assert exp["priced_ok"] is False
        assert exp["unpriced_unmapped"] == {"XYZ-20DEC30-CDE": 2.0}

    def test_clean_book_is_still_priced(self):
        s = _reconcile_sleeve([_pos(PID, "LONG", 1)])
        s.reconcile_positions()
        s._equity_is_stale = lambda: False
        s.sleeve_equity_age_sec = lambda: 0.0
        s._notional_usd = lambda a, c: 100.0 * c
        exp = s.sleeve_exposure()
        assert exp["priced_ok"] is True and exp["unpriced_unmapped"] == {}

    def test_warned_once_per_pid_per_process(self, caplog):
        s = _reconcile_sleeve([_pos("XYZ-20DEC30-CDE", "SHORT", 2)])
        with caplog.at_level(logging.WARNING,
                             logger="exchange.coinbase_sleeve"):
            s.reconcile_positions()
            s.reconcile_positions()
        hits = [r for r in caplog.records
                if "UNMAPPED product" in r.getMessage()]
        assert len(hits) == 1

    def test_zero_contract_unmapped_row_is_ignored(self, caplog):
        s = _reconcile_sleeve([_pos("XYZ-20DEC30-CDE", "", 0)])
        with caplog.at_level(logging.WARNING,
                             logger="exchange.coinbase_sleeve"):
            s.reconcile_positions()
        assert s._unmapped_positions == {}
        assert not [r for r in caplog.records
                    if "UNMAPPED product" in r.getMessage()]

    def test_unmapped_clears_when_the_venue_no_longer_shows_it(self):
        s = _reconcile_sleeve([_pos("XYZ-20DEC30-CDE", "SHORT", 2)])
        s.reconcile_positions()
        assert s._unmapped_positions
        s._adapter._client.list_futures_positions = lambda: {"positions": []}
        s.reconcile_positions()
        assert s._unmapped_positions == {}

    def test_net_cap_fails_open_not_blind_with_an_unmapped_position(self):
        # with the unmapped SHORT counted as unpriced, an increasing order
        # against the net budget is ALLOWED with a warning (fail OPEN) --
        # never silently certified as within budget
        s = _reconcile_sleeve([_pos(PID, "LONG", 1),
                               _pos("XYZ-20DEC30-CDE", "SHORT", 40)])
        s.reconcile_positions()
        s._equity_is_stale = lambda: False
        s.sleeve_equity_age_sec = lambda: 0.0
        s._notional_usd = lambda a, c: 1500.0 * c   # 1ct = 37.5% of equity
        s._halted = False
        s._max_contracts_per_asset = 10
        s._max_net_exposure = 0.50
        s._max_asset_exposure = {}
        s._sized_contracts = lambda a: None
        ok, reason = s.can_trade("SOL", +1)
        assert ok is True, reason
