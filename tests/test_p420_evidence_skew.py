"""[P420] The skew seat's evidence layer: sign convention stated correctly,
live z-window == calibration, raw inputs recorded per tick, the runtime series
banked, and a stale persisted hold REPLAYED rather than seated blindly.

Every assertion here is about the deploy side of the P420 read-through:
  1. the docstring said `call - put` (the field is `put - call`) — the SIGN is
     unchanged (live == validated), only the words were wrong;
  2. the live z used 29 obs / min 3 where the calibration uses 30 excl. current
     / min 8 — a train/serve skew on the live decider's own window;
  3. nothing recorded the raw z the seat decided on, so the forward ledger
     could never be re-scored against a re-fetched calibration series;
  4. a persisted deadband hold was restored with no age bound.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import random
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

import defense.skew_flow_signal as sfs  # noqa: E402
from defense.skew_flow_signal import (  # noqa: E402
    SkewFlowSignal, STRATEGY_LABEL, band_direction, last_diag, replay_hold,
    zscore_trailing, HOLD_MAX_AGE_DAYS)


def _rows(v25, v10=None, end=None):
    """Daily by-tenor rows ending today (ISO dates), oldest first."""
    end = end or dt.datetime.now(dt.timezone.utc)
    out = []
    for i, v in enumerate(v25):
        d = (end - dt.timedelta(days=len(v25) - 1 - i))
        r = {"tenor": 30, "skew_25d": v, "date": d.isoformat(),
             "call_25d_iv": 30.0 + i * 0.01, "put_25d_iv": 30.0 + v + i * 0.01,
             "atm_iv": 40.0}
        if v10 is not None:
            r["skew_10d"] = v10[i]
        out.append(r)
    return out


def _sig(tmp_path, rows, key="test"):
    s = SkewFlowSignal(data_dir=str(tmp_path))
    s._key = key
    s._fetch_trailing = lambda a: rows          # type: ignore[assignment]
    return s


# =============================================================================
# 1. Sign convention — the words, not the sign
# =============================================================================

class TestSignConvention:
    def test_docstring_states_put_minus_call_and_not_contrarian(self):
        doc = sfs.__doc__
        assert "put_25d_iv - call_25d_iv" in doc
        assert "NOT the contrarian" in doc
        assert "flip the sign" in doc and "DO NOT" in doc
        # the old, wrong sentence must be gone
        assert "skew_25d = call_iv - put_iv" not in doc

    def test_the_label_is_kept_and_named_a_misnomer(self):
        """Renaming breaks the P317/P407c vocabulary pins; the note must
        live at the label's definition."""
        assert STRATEGY_LABEL == "skew_contra"
        src = (REPO / "defense" / "skew_flow_signal.py").read_text(encoding="utf-8")
        i = src.index('STRATEGY_LABEL = "skew_contra"')
        assert "MISNOMER" in src[i - 600:i]

    def test_mapping_is_unchanged_z_below_band_is_long(self):
        """live == validated: the calibration's `contra = -z; +1 iff > band`."""
        assert band_direction(-1.5, 0.0) == 1.0
        assert band_direction(+1.5, 0.0) == -1.0
        assert band_direction(0.3, -1.0) == -1.0   # hold inside the band
        assert band_direction(-1.0, 0.0) == 0.0    # strictly beyond, not at


# =============================================================================
# 2. z-window == calibration, bar for bar
# =============================================================================

class TestZWindowMatchesCalibration:
    def test_trailing_30_excluding_current_min_8(self):
        # 8 trailing obs -> a z; 7 -> 0.0 (the calibration's _Z_MIN)
        base = [1.0, -1.0] * 4
        assert zscore_trailing(base + [5.0]) != 0.0          # 8 trailing
        assert zscore_trailing(base[:-1] + [5.0]) == 0.0     # 7 trailing
        # the current value is NOT in its own window: a constant 30-window
        # with a spike at the end gives sd == 0 -> 0.0, whereas including the
        # spike would give a finite z
        assert zscore_trailing([2.0] * 30 + [9.0]) == 0.0

    def test_equals_the_producers_zseries_on_a_random_series(self):
        """The calibration's `_zseries` is the reference; the live function
        must reproduce it at every index (P164/P214 parity class)."""
        prod = pytest.importorskip("training.skew_seat_calibration")
        rng = random.Random(418)
        sig = [rng.gauss(0, 3) for _ in range(120)]
        ref = prod._zseries(sig)
        live = [zscore_trailing(sig[:i + 1]) for i in range(len(sig))]
        for i, (a, b) in enumerate(zip(ref, live)):
            assert abs(a - b) < 1e-12, (i, a, b)

    def test_replay_equals_the_producers_position_walk(self):
        """replay_hold must land where `_positions_from_z` lands after every
        bar but the last — the state a continuous process carries INTO it."""
        prod = pytest.importorskip("training.skew_seat_calibration")
        rng = random.Random(7)
        sig = [rng.gauss(0, 3) for _ in range(120)]
        pos = prod._positions_from_z(prod._zseries(sig))
        assert replay_hold(sig) == pos[-2]

    def test_the_seat_uses_the_aligned_window(self, tmp_path):
        """A value that clears the band under 30-excl-current but NOT under
        the old 29-window arithmetic would be invisible to a source pin; the
        seat is driven end-to-end instead."""
        vals = [(-1.0) ** i for i in range(30)] + [-8.0]
        s = _sig(tmp_path, _rows(vals))
        d, fresh = s.seat_direction("BTC")
        assert fresh and d == 1.0
        diag = last_diag("BTC")
        assert diag["n_rows"] == 31
        assert diag["z25"] == pytest.approx(zscore_trailing(vals), abs=1e-3)


# =============================================================================
# 3. raw inputs recorded per tick, into the skewetf row
# =============================================================================

class TestDiagnosticsReachTheLedger:
    def test_last_diag_carries_z25_z10_z_and_the_tenor30_value(self, tmp_path):
        v25 = [(-1.0) ** i for i in range(30)] + [0.0]
        v10 = [(-1.0) ** i for i in range(30)] + [-8.0]
        s = _sig(tmp_path, _rows(v25, v10))
        s.seat_direction("ETH")
        d = last_diag("ETH")
        assert d["skew_25d"] == 0.0 and d["skew_10d"] == -8.0
        assert d["z25"] == pytest.approx(0.0, abs=1e-9)
        assert d["z10"] < -1.0
        assert d["z"] == pytest.approx((d["z25"] + d["z10"]) / 2.0, abs=1e-3)
        assert d["hold_source"] in ("persisted", "replayed")

    def test_absent_diag_is_none_not_zero(self):
        assert last_diag("NOPE") is None

    def test_combo_row_carries_the_skew_diag(self, tmp_path):
        from defense.skew_etf_combo_shadow import SkewEtfComboShadow
        v25 = [(-1.0) ** i for i in range(30)] + [-8.0]
        s = _sig(tmp_path, _rows(v25))
        s.seat_direction("BTC")
        sh = SkewEtfComboShadow(data_dir=str(tmp_path))
        sh.record_tick("BTC", 1.0, True, 0.0, False)   # no explicit diag
        rows = [json.loads(l) for l in
                (tmp_path / "strategy_shadow" / "skewetf_BTC.jsonl")
                .read_text(encoding="utf-8").splitlines()]
        assert rows and all("skew" in r["diagnostics"] for r in rows)
        sk = rows[0]["diagnostics"]["skew"]
        assert sk["skew_25d"] == -8.0 and sk["z25"] < -1.0 and "z" in sk

    def test_explicit_diag_wins_and_absent_stays_absent(self, tmp_path):
        from defense.skew_etf_combo_shadow import SkewEtfComboShadow
        sh = SkewEtfComboShadow(data_dir=str(tmp_path))
        sh.record_tick("SOL", 0.0, False, 0.0, False)  # no diag anywhere for SOL
        rows = [json.loads(l) for l in
                (tmp_path / "strategy_shadow" / "skewetf_SOL.jsonl")
                .read_text(encoding="utf-8").splitlines()]
        assert all("skew" not in r["diagnostics"] for r in rows), (
            "a missing diag must not be fabricated (P2)")
        sh.record_tick("SOL", 0.0, False, 0.0, False,
                       skew_diag={"z": 9.9, "z25": 9.9, "z10": None,
                                  "skew_25d": 1.0, "skew_10d": None,
                                  "n_rows": 1, "hold_source": "x"})
        rows = [json.loads(l) for l in
                (tmp_path / "strategy_shadow" / "skewetf_SOL.jsonl")
                .read_text(encoding="utf-8").splitlines()]
        assert rows[-1]["diagnostics"]["skew"]["z"] == 9.9


# =============================================================================
# 4. the runtime-series archive (task 2d)
# =============================================================================

class TestRuntimeArchive:
    def _archive(self, tmp_path, asset="BTC"):
        p = tmp_path / f"laevitas_apiv2_skew_{asset}.jsonl"
        if not p.exists():
            return []
        return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]

    def test_two_overlapping_fetches_leave_one_row_per_date(self, tmp_path):
        end = dt.datetime(2026, 8, 27, tzinfo=dt.timezone.utc)
        first = _rows([float(i) for i in range(40)], end=end - dt.timedelta(days=5))
        s = _sig(tmp_path, first)
        s.seat_direction("BTC")
        n1 = len(self._archive(tmp_path))
        assert n1 == 40
        second = _rows([float(i) + 100 for i in range(40)], end=end)  # 35 overlap
        s2 = _sig(tmp_path, second)
        s2.seat_direction("BTC")
        rows = self._archive(tmp_path)
        dates = [r["date"] for r in rows]
        assert len(dates) == len(set(dates)) == 45, "one row per date after the merge"
        assert dates == sorted(dates)
        # a re-fetched date carries the NEWER row (P266: new wins)
        overlap_date = second[0]["date"][:10]
        got = next(r for r in rows if r["date"] == overlap_date)
        assert got["skew_25d"] == 100.0
        assert got["tenor"] == 30 and "call_25d_iv" in got and "atm_iv" in got

    def test_a_failed_fetch_writes_nothing(self, tmp_path):
        s = _sig(tmp_path, None)          # fetch failure
        s.seat_direction("BTC")
        assert self._archive(tmp_path) == []
        s_nokey = SkewFlowSignal(data_dir=str(tmp_path))
        s_nokey._key = ""                 # mock/absent -> never banked
        s_nokey.seat_direction("BTC")
        assert self._archive(tmp_path) == []

    def test_absent_fields_are_none_never_fabricated(self, tmp_path):
        rows = _rows([float(i) for i in range(20)])
        for r in rows:
            r.pop("atm_iv"); r.pop("call_25d_iv")
        _sig(tmp_path, rows).seat_direction("ETH")
        got = self._archive(tmp_path, "ETH")
        assert got and all(r["atm_iv"] is None and r["call_25d_iv"] is None
                           and r["skew_10d"] is None for r in got)

    def test_the_docstring_names_the_archive_as_the_recalibration_path(self):
        assert "laevitas_apiv2_skew_" in sfs.__doc__
        assert "runtime-parity recalibration" in sfs.__doc__


# =============================================================================
# 5. stale persisted hold -> REPLAY, not a blind restore
# =============================================================================

def _persist(tmp_path, hold, age_days):
    (tmp_path / "skew_seat_state.json").write_text(json.dumps({
        "hold": hold, "saved_at": time.time() - age_days * 86400.0}),
        encoding="utf-8")


class TestHoldRestore:
    # a window whose continuous walk ends SHORT: flat noise then a spike UP
    # (puts rich) five bars from the end, then in-band values -> hold -1
    V_SHORT = [(-1.0) ** i for i in range(30)] + [8.0, 0.2, -0.1, 0.1, 0.0]

    def test_a_three_day_old_plus_one_is_not_seated_blindly(self, tmp_path):
        _persist(tmp_path, {"BTC": 1.0}, age_days=3.0)
        s = _sig(tmp_path, _rows(self.V_SHORT))
        assert s._hold == {}, "a stale hold must not be restored as-is"
        assert "BTC" in s._replay_pending
        d, fresh = s.seat_direction("BTC")
        assert fresh and d == -1.0, (
            "the replayed hold (-1, the spike UP) must win over the stale "
            "persisted +1")
        assert last_diag("BTC")["hold_source"] == "replayed"

    def test_replay_reproduces_a_continuous_process(self, tmp_path):
        """Continuous: one process walks the whole window bar by bar.
        Restart: a fresh object with a stale state file sees the same window
        once. Both must hold the same thing."""
        vals = self.V_SHORT
        # continuous process: feed growing prefixes through separate ticks
        cont = _sig(tmp_path / "cont", _rows(vals[:31]))
        (tmp_path / "cont").mkdir(exist_ok=True)
        for k in range(31, len(vals) + 1):
            cont._fetch_trailing = (lambda a, k=k: _rows(vals[:k]))  # type: ignore[assignment]
            cont._cache.clear()
            cont.seat_direction("BTC")
        continuous = cont._hold.get("BTC", 0.0)
        # restarted process with a 3-day-old, WRONG persisted hold
        (tmp_path / "rs").mkdir(exist_ok=True)
        _persist(tmp_path / "rs", {"BTC": -continuous or 1.0}, age_days=3.0)
        rs = _sig(tmp_path / "rs", _rows(vals))
        rs.seat_direction("BTC")
        assert rs._hold["BTC"] == continuous == -1.0

    def test_fresh_state_is_restored_and_absent_state_starts_flat(self, tmp_path):
        _persist(tmp_path, {"BTC": 1.0}, age_days=0.2)   # < HOLD_MAX_AGE_DAYS
        s = _sig(tmp_path, _rows([(-1.0) ** i for i in range(30)] + [0.1]))
        assert s._hold == {"BTC": 1.0} and not s._replay_pending
        assert s.seat_direction("BTC") == (1.0, True)     # in-band -> held +1
        # absent file: cold start at 0.0 -> in-band -> 0.0 -> no seat
        s2 = _sig(tmp_path / "empty", _rows([(-1.0) ** i for i in range(30)] + [0.1]))
        assert s2.seat_direction("BTC") == (0.0, True)

    def test_stale_hold_with_no_rows_starts_flat_not_fresh(self, tmp_path):
        _persist(tmp_path, {"BTC": 1.0}, age_days=3.0)
        s = _sig(tmp_path, None)
        assert s.seat_direction("BTC") == (0.0, False)   # no seat, incumbent stands

    def test_the_replayed_hold_is_persisted_with_a_fresh_stamp(self, tmp_path):
        _persist(tmp_path, {"BTC": 1.0}, age_days=3.0)
        s = _sig(tmp_path, _rows(self.V_SHORT))
        s.seat_direction("BTC")
        st = json.loads((tmp_path / "skew_seat_state.json").read_text(encoding="utf-8"))
        assert st["hold"]["BTC"] == -1.0
        assert time.time() - st["saved_at"] < HOLD_MAX_AGE_DAYS * 86400
