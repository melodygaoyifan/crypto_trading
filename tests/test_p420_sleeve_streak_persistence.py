"""[P420] The P198 flip streak and the P416 resize streak PERSIST.

Both were RAM-only. "A restart only delays a flip/resize" was true of one
restart; the engine restarted 7 times in 4h on 2026-08-27 and BTC's resize
sat at "1/2" on three consecutive boot ticks -- an UNBOUNDED deferral. The
flip streak is worse: a wanted reversal held through every restart.

Pinned here:
  * a streak advance is written to disk IMMEDIATELY (the process may die
    before the 4H save), with a timestamp per entry;
  * a fresh entry survives a restart and completes on the next agreeing
    tick (2/2) -- the whole point;
  * an entry older than STREAK_MAX_AGE_SEC (two 4H ticks) is dropped on
    restore AND on read: that stale is not "consecutive";
  * a streak the live process BROKE (a same-direction tick) reaches disk
    too, or a restart resurrects it into a non-consecutive 2/2;
  * corrupt / malformed state -> cold start, never a raise;
  * [task 4] a FLIP_DEFERRED tick resets the resize streak (a resize
    proposal, a flip-deferred tick, then the same resize is NOT two
    consecutive proposals);
  * [task 3b] both deferred returns run the stale-entry sweep.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from exchange.coinbase_sleeve import CoinbaseSleeve  # noqa: E402


@pytest.fixture(autouse=True)
def _private_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HMATS_DATA_DIR", str(tmp_path))
    return tmp_path


def _state_file(tmp_path):
    return Path(tmp_path) / "coinbase_sleeve_state.json"


def _sleeve(cur=2.0, flip_ticks=2, resize_ticks=2, restore=False):
    """Live-shaped sleeve: manage_to_signal's own logic, stubbed IO.
    `restore=True` mimics a fresh process reading the state file."""
    s = CoinbaseSleeve.__new__(CoinbaseSleeve)
    s._flip_persist_ticks = flip_ticks
    s._flip_pending = {}
    s._resize_persist_ticks = resize_ticks
    s._resize_pending = {}
    s._reconcile_ok = True
    s._cur = float(cur)
    s._executed = []
    s._swept = []
    s.reconcile_positions = lambda: None
    s.signed_contracts = lambda asset: s._cur

    def _target_for(asset, d, threshold=0.15, conviction=1.0):
        if abs(d) < threshold:
            return 0
        sized = max(1, int(round(2 * conviction)))
        return sized if d > 0 else -sized
    s.target_for = _target_for

    async def _exec(asset, target):
        s._executed.append(target)
        return {"status": "OK", "asset": asset}
    s.execute_target = _exec

    async def _sweep(asset):
        s._swept.append(asset)
        return 0
    s.sweep_stale_entries = _sweep
    if restore:
        s._restore_state()
    return s


def _manage(s, direction, conviction=1.0, asset="BTC"):
    return asyncio.run(s.manage_to_signal(asset, direction,
                                          conviction=conviction))


# ---------------------------------------------------------------------------
# round trip through disk
# ---------------------------------------------------------------------------

class TestFlipStreakPersists:
    def test_advance_is_written_immediately_with_a_timestamp(self, tmp_path):
        s = _sleeve(cur=+1.0)
        r = _manage(s, -0.8)
        assert r["status"] == "FLIP_DEFERRED" and r["streak"] == 1
        f = _state_file(tmp_path)
        assert f.exists(), "the streak advance did not reach disk"
        st = json.loads(f.read_text(encoding="utf-8"))
        e = st["flip_pending"]["BTC"]
        assert e["sign"] == -1 and e["streak"] == 1
        assert abs(time.time() - float(e["ts"])) < 60

    def test_fresh_entry_survives_a_restart_and_completes_2_of_2(self,
                                                               tmp_path):
        s1 = _sleeve(cur=+1.0)
        assert _manage(s1, -0.8)["status"] == "FLIP_DEFERRED"
        assert s1._executed == []
        # --- process dies; a fresh sleeve restores from disk ---
        s2 = _sleeve(cur=+1.0, restore=True)
        assert s2._flip_pending["BTC"][:2] == (-1, 1)
        r = _manage(s2, -0.8)
        assert r["status"] == "OK", r
        assert s2._executed == [-2], "the flip did not execute on tick 2/2"

    def test_the_unbounded_deferral_is_gone(self, tmp_path):
        # the live incident's shape: restart between every tick
        s = _sleeve(cur=+1.0)
        _manage(s, -0.8)
        for _ in range(5):
            s = _sleeve(cur=+1.0, restore=True)
            r = _manage(s, -0.8)
            if r["status"] == "OK":
                break
        assert r["status"] == "OK", "a restart between every tick held the flip forever"

    def test_a_broken_streak_reaches_disk_too(self, tmp_path):
        # streak 1 -> same-direction tick breaks it -> restart -> opposing
        # tick must read as 1, not 2 (a resurrected streak = a flip on
        # NON-consecutive ticks)
        s1 = _sleeve(cur=+1.0)
        _manage(s1, -0.8)
        _manage(s1, +0.8)  # same direction: streak broken (resize target 2 == cur? no: 2 vs 1)
        st = json.loads(_state_file(tmp_path).read_text(encoding="utf-8"))
        assert "BTC" not in st["flip_pending"]
        s2 = _sleeve(cur=+1.0, restore=True)
        r = _manage(s2, -0.8)
        assert r["status"] == "FLIP_DEFERRED" and r["streak"] == 1


class TestResizeStreakPersists:
    def test_resize_survives_a_restart_and_completes(self, tmp_path):
        s1 = _sleeve(cur=2.0)
        r = _manage(s1, +1.0, conviction=0.58)   # sized 1ct vs cur 2ct
        assert r["status"] == "RESIZE_DEFERRED" and r["streak"] == 1
        st = json.loads(_state_file(tmp_path).read_text(encoding="utf-8"))
        assert st["resize_pending"]["BTC"]["target"] == 1
        s2 = _sleeve(cur=2.0, restore=True)
        r = _manage(s2, +1.0, conviction=0.58)
        assert r["status"] == "OK" and s2._executed == [1]

    def test_a_changed_proposal_after_restart_restarts_at_1(self, tmp_path):
        s1 = _sleeve(cur=2.0)
        _manage(s1, +1.0, conviction=0.58)       # proposes 1
        s2 = _sleeve(cur=2.0, restore=True)
        r = _manage(s2, +1.0, conviction=1.5)    # proposes 3: not the same
        assert r["status"] == "RESIZE_DEFERRED" and r["streak"] == 1


# ---------------------------------------------------------------------------
# staleness: two 4H ticks is the bound, on restore AND on read
# ---------------------------------------------------------------------------

class TestStaleness:
    def _write(self, tmp_path, ts, key="flip_pending",
               entry=None):
        entry = entry or {"sign": -1, "streak": 1, "ts": ts}
        _state_file(tmp_path).write_text(json.dumps({
            "base_version": CoinbaseSleeve._BASE_VERSION,
            "sleeve_start_equity": 4000.0, "halted": False,
            "halt_reason": "", key: {"BTC": entry}}), encoding="utf-8")

    def test_an_entry_older_than_8h_is_dropped_on_restore(self, tmp_path):
        self._write(tmp_path, ts=time.time() - 9 * 3600)
        s = _sleeve(cur=+1.0, restore=True)
        assert s._flip_pending == {}
        r = _manage(s, -0.8)
        assert r["status"] == "FLIP_DEFERRED" and r["streak"] == 1

    def test_a_7h_old_entry_survives_restore(self, tmp_path):
        self._write(tmp_path, ts=time.time() - 7 * 3600)
        s = _sleeve(cur=+1.0, restore=True)
        assert s._flip_pending["BTC"][:2] == (-1, 1)

    def test_bound_is_two_4h_ticks(self):
        assert CoinbaseSleeve.STREAK_MAX_AGE_SEC == 8 * 3600.0

    def test_in_memory_entry_past_the_bound_restarts_at_1(self):
        s = _sleeve(cur=+1.0)
        s._flip_pending["BTC"] = (-1, 1, time.time() - 9 * 3600)
        r = _manage(s, -0.8)
        assert r["status"] == "FLIP_DEFERRED" and r["streak"] == 1, (
            "a gap of 9h between advances was counted as consecutive")

    def test_resize_entry_past_the_bound_restarts_at_1(self):
        s = _sleeve(cur=2.0)
        s._resize_pending["BTC"] = (1, 1, time.time() - 9 * 3600)
        r = _manage(s, +1.0, conviction=0.58)
        assert r["status"] == "RESIZE_DEFERRED" and r["streak"] == 1

    def test_restore_survives_a_base_version_mismatch(self, tmp_path):
        # streaks are independent of the equity formula; a version-bumped
        # file still carries a real half-finished streak
        _state_file(tmp_path).write_text(json.dumps({
            "base_version": "some_old_version",
            "flip_pending": {"BTC": {"sign": -1, "streak": 1,
                                     "ts": time.time()}}}), encoding="utf-8")
        s = _sleeve(cur=+1.0, restore=True)
        assert s._flip_pending["BTC"][:2] == (-1, 1)


# ---------------------------------------------------------------------------
# corrupt / malformed state -> cold start, never a raise
# ---------------------------------------------------------------------------

class TestCorruptState:
    def test_garbage_file_is_a_cold_start(self, tmp_path):
        _state_file(tmp_path).write_text("{not json", encoding="utf-8")
        s = _sleeve(cur=+1.0, restore=True)
        assert s._flip_pending == {} and s._resize_pending == {}
        assert _manage(s, -0.8)["streak"] == 1

    def test_malformed_entries_are_dropped_individually(self, tmp_path):
        _state_file(tmp_path).write_text(json.dumps({
            "base_version": CoinbaseSleeve._BASE_VERSION,
            "flip_pending": {"BTC": {"sign": -1},              # no ts/streak
                             "ETH": {"sign": 1, "streak": "x", "ts": 1.0},
                             "SOL": {"sign": 1, "streak": 1,
                                     "ts": time.time()}},
            "resize_pending": "not-a-dict"}), encoding="utf-8")
        s = _sleeve(cur=+1.0, restore=True)
        assert set(s._flip_pending) == {"SOL"}
        assert s._resize_pending == {}

    def test_legacy_2_tuple_in_memory_still_works(self):
        # a pre-P420 in-memory entry (sign, streak) ages from first sight
        s = _sleeve(cur=+1.0)
        s._flip_pending["BTC"] = (-1, 1)
        r = _manage(s, -0.8)
        assert r["status"] == "OK" and s._executed == [-2]

    def test_persist_failure_never_reaches_the_order_path(self, tmp_path,
                                                          monkeypatch):
        s = _sleeve(cur=+1.0)
        monkeypatch.setattr(s, "_persist_state",
                            lambda: (_ for _ in ()).throw(OSError("disk")))
        r = _manage(s, -0.8)
        assert r["status"] == "FLIP_DEFERRED"


# ---------------------------------------------------------------------------
# [task 4] FLIP_DEFERRED resets the resize streak
# ---------------------------------------------------------------------------

class TestFlipDeferredResetsResize:
    def test_resize_flip_resize_is_not_consecutive(self):
        s = _sleeve(cur=2.0, flip_ticks=2, resize_ticks=2)
        assert _manage(s, +1.0, conviction=0.58)["status"] == "RESIZE_DEFERRED"
        assert _manage(s, -1.0)["status"] == "FLIP_DEFERRED"
        r = _manage(s, +1.0, conviction=0.58)
        assert r["status"] == "RESIZE_DEFERRED" and r["streak"] == 1, (
            "a flip-deferred tick between two resize proposals counted as "
            "consecutive")
        assert s._executed == []

    def test_the_pop_reaches_disk(self, tmp_path):
        s = _sleeve(cur=2.0, flip_ticks=2, resize_ticks=2)
        _manage(s, +1.0, conviction=0.58)
        _manage(s, -1.0)
        st = json.loads(_state_file(tmp_path).read_text(encoding="utf-8"))
        assert "BTC" not in st["resize_pending"]
        assert st["flip_pending"]["BTC"]["streak"] == 1


# ---------------------------------------------------------------------------
# [task 3b] both deferred returns sweep stale entry orders
# ---------------------------------------------------------------------------

class TestDeferredTicksSweep:
    def test_flip_deferred_sweeps(self):
        s = _sleeve(cur=+1.0)
        _manage(s, -0.8)
        assert s._swept == ["BTC"]

    def test_resize_deferred_sweeps(self):
        s = _sleeve(cur=2.0)
        _manage(s, +1.0, conviction=0.58)
        assert s._swept == ["BTC"]

    def test_a_fixture_without_an_adapter_does_not_raise(self):
        # the REAL sweep on a sleeve with no adapter (operator scripts,
        # fixtures): returns None quietly, never raises into the tick
        s = _sleeve(cur=+1.0)
        del s.sweep_stale_entries  # use the real method
        r = _manage(s, -0.8)
        assert r["status"] == "FLIP_DEFERRED"


class TestNoRegression:
    def test_p416_oscillation_still_produces_zero_position_changes(self):
        s = _sleeve(cur=2.0)
        for conv in (1.09, 0.58, 1.09, 0.58):
            _manage(s, +1.0, conviction=conv)
        # conv 1.09 sizes to 2 == cur (a NOOP at the venue); the 1ct
        # proposal must never execute
        assert all(t == 2 for t in s._executed), s._executed

    def test_no_streak_change_means_no_write(self, tmp_path):
        # a plain same-direction tick with nothing pending must not touch
        # the state file every 4H
        s = _sleeve(cur=2.0)
        _manage(s, +1.0, conviction=1.0)  # target 2 == cur -> execute_target
        assert not _state_file(tmp_path).exists()
