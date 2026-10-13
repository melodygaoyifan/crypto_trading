"""[P421e] Two tooling gaps found while verifying the P421 batch, closed.

1. The September countdown had no notion of a read having been TAKEN: after
   the 19 reads were executed (2026-10-12) it kept printing "19 READ(S) ARE
   DUE" every Monday — an alarm that fires on the healthy state is wallpaper
   (P202). `reads_taken.json` now records a read, written only by a scorer
   run whose window covered the candidate's full accrual (the P421 trap: a
   30d-from-today read scored n=0..noise while the full-span read decided).

2. The seat controller could not score the regimebook candidate while the
   live book was flat: agent_ic_review's `quant` series was a constant
   (n=609, IC undefined), so it REFUSED — correctly, but with a message that
   read as missing data, and with no route to the one verdict it was built
   to reach (flat wins by comparison, P271). It can now be fed the pooled
   regimebook ledger scored at its two horizons (--shadow-ic-report), and a
   flat decider names itself in the refusal.
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from datetime import date
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sc = _load("september_check_p421e", "scripts/september_check.py")
seat = _load("seat_check_p421e", "scripts/seat_check.py")


# --------------------------------------------------------------------------
# fixtures: synthetic scorer reports in the scorer's REAL output shape
# --------------------------------------------------------------------------
def _pooled_rec(strategy, ic, t, n=1500, horizons=("1", "4"), n_directional=1100):
    return {
        "strategy": strategy, "asset": "POOLED", "n_records": n + 20,
        "n_per_horizon": {h: n for h in horizons},
        "ic_per_horizon": {h: ic for h in horizons},
        "n_directional": n_directional,
        "verdict": "HOLD" if ic <= 0 else "PROMOTE",
        "per_horizon": {h: {"ic": ic, "n": n, "t_stat": t} for h in horizons},
    }


def _report(recs, window_days=65):
    return {"generated_at": "2026-10-13T00:00:00+00:00", "window_days": window_days,
            "horizons_bars": [4, 12, 24], "n_records": len(recs), "per_strategy": recs}


def _write(path: Path, obj) -> Path:
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


@pytest.fixture
def reports(tmp_path):
    pooled = _write(tmp_path / "pooled.json", _report([
        _pooled_rec("regimebook", -0.13, 2.6, horizons=("4", "12", "24")),
        _pooled_rec("liquidation_squeeze", 0.03, 0.9, horizons=("4", "12", "24")),
        _pooled_rec("liquidation_exhaustion", -0.03, 0.9, horizons=("4", "12", "24")),
    ]))
    per_asset = _write(tmp_path / "perasset.json", _report([
        {"strategy": "regimebook_adj", "asset": "BTC", "verdict": "HOLD",
         "ic_per_horizon": {"4": -0.1}, "n_per_horizon": {"4": 300}},
    ]))
    return per_asset, pooled


# --------------------------------------------------------------------------
# 1. reads-taken ledger
# --------------------------------------------------------------------------
class TestReadsTaken:
    def test_window_shorter_than_the_ledger_span_records_nothing(self, tmp_path, reports):
        per_asset, pooled = reports
        ledger = tmp_path / "reads.json"
        today = date(2026, 10, 13)
        # regimebook's ledger started 2026-08-10 -> span 64d; a 30d window
        # did NOT sit that exam and must not silence its alarm (the P421 trap).
        rec, skipped = sc.record_reads_taken(30, per_asset, pooled, today=today, path=ledger)
        assert "regimebook" not in rec and "regimebook" in skipped
        assert sc.load_reads_taken(ledger).get("regimebook") is None

    def test_full_span_window_records_with_the_scorers_verdict(self, tmp_path, reports):
        per_asset, pooled = reports
        ledger = tmp_path / "reads.json"
        rec, _ = sc.record_reads_taken(70, per_asset, pooled, today=date(2026, 10, 13), path=ledger)
        assert "regimebook" in rec and "derivflow" in rec
        got = sc.load_reads_taken(ledger)
        assert got["regimebook"]["read_on"] == "2026-10-13"
        assert got["regimebook"]["verdicts"]["regimebook"]["verdict"] == "HOLD"
        assert got["regimebook"]["verdicts"]["regimebook"]["scope"] == "POOLED"
        # a two-strategy candidate carries BOTH verdicts, not a prefix lump
        assert set(got["derivflow"]["verdicts"]) == {"liquidation_squeeze", "liquidation_exhaustion"}
        # a candidate with no POOLED row falls back to the per-asset report
        assert got["regimebook_adj"]["verdicts"]["regimebook_adj"]["scope"] == "per-asset"

    def test_unstarted_clock_is_never_recorded(self, tmp_path, reports):
        per_asset, pooled = reports
        rec, skipped = sc.record_reads_taken(400, per_asset, pooled,
                                             today=date(2026, 10, 13), path=tmp_path / "r.json")
        assert "regimebook_breadth" in skipped and "regimebook_breadth" not in rec

    def test_convsize_is_recorded_as_not_judged_here(self, tmp_path, reports):
        per_asset, pooled = reports
        sc.record_reads_taken(70, per_asset, pooled, today=date(2026, 10, 13),
                              path=tmp_path / "r.json")
        got = sc.load_reads_taken(tmp_path / "r.json")["convsize"]
        assert got["verdicts"] == {} and "not judged" in got["note"]

    def test_verdict_strategies_cover_every_candidate_both_ways(self):
        assert set(sc.VERDICT_STRATEGIES) == set(sc.CANDIDATES)

    def test_unreadable_ledger_reads_as_empty_so_alarms_still_shout(self, tmp_path):
        bad = tmp_path / "reads.json"
        bad.write_text("{not json", encoding="utf-8")
        assert sc.load_reads_taken(bad) == {}
        assert sc.load_reads_taken(tmp_path / "missing.json") == {}


class TestCountdown:
    def test_taken_read_is_not_due_and_prints_its_verdict(self, tmp_path, monkeypatch, capsys):
        ledger = tmp_path / "reads.json"
        ledger.write_text(json.dumps({"regimebook": {
            "read_on": "2026-10-12", "window_days": 65,
            "verdicts": {"regimebook": {"scope": "POOLED", "verdict": "HOLD",
                                        "ic": {"4": -0.13, "12": -0.34, "24": -0.46}}},
            "next_read": None}}), encoding="utf-8")
        monkeypatch.setattr(sc, "READS_TAKEN_PATH", ledger)
        due = sc.countdown(return_due=True, today_override=date(2026, 10, 13))
        names = {d[0] for d in due}
        assert "regimebook" not in names
        assert "regimebook_adj" in names  # untaken + overdue still shouts
        out = capsys.readouterr().out
        assert "TAKEN" in out and "regimebook HOLD ic -0.13/-0.34/-0.46" in out

    def test_empty_ledger_leaves_every_overdue_read_due(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sc, "READS_TAKEN_PATH", tmp_path / "none.json")
        due = sc.countdown(return_due=True, today_override=date(2026, 10, 13))
        assert "regimebook" in {d[0] for d in due}

    def test_a_read_taken_after_the_reference_date_still_counts_as_due(self, tmp_path, monkeypatch):
        ledger = tmp_path / "reads.json"
        ledger.write_text(json.dumps({"regimebook": {"read_on": "2026-10-13", "verdicts": {}}}),
                          encoding="utf-8")
        monkeypatch.setattr(sc, "READS_TAKEN_PATH", ledger)
        due = sc.countdown(return_due=True, today_override=date(2026, 9, 10))
        assert "regimebook" in {d[0] for d in due}

    def test_next_read_date_is_surfaced(self, tmp_path, monkeypatch, capsys):
        ledger = tmp_path / "reads.json"
        ledger.write_text(json.dumps({"calbasis": {
            "read_on": "2026-10-12", "verdicts": {"calbasis": {"scope": "POOLED", "verdict": "HOLD"}},
            "next_read": "2026-11-10 (scripts/calbasis_check.py)"}}), encoding="utf-8")
        monkeypatch.setattr(sc, "READS_TAKEN_PATH", ledger)
        sc.countdown(today_override=date(2026, 10, 13))
        assert "next read 2026-11-10" in capsys.readouterr().out

    def test_main_runs_the_seat_read_after_the_tripwire(self):
        src = inspect.getsource(sc.main)
        assert "run_seat_read(" in src
        assert src.index("run_tripwire()") < src.index("run_seat_read(")


# --------------------------------------------------------------------------
# 2. seat controller: the book's own ledger as evidence
# --------------------------------------------------------------------------
class TestSeatFromShadowReport:
    def _ledger_dir(self, tmp_path, available=True):
        d = tmp_path / "ledgers"
        d.mkdir()
        for a in ("BTC", "ETH", "SOL"):
            (d / f"regimebook_{a}.jsonl").write_text(
                json.dumps({"ts": 1.0, "asset": a, "direction": 1.0, "available": available}) + "\n",
                encoding="utf-8")
        return d

    def test_negative_book_reaches_the_flat_verdict(self, tmp_path, capsys):
        rep = _write(tmp_path / "seat.json", _report([_pooled_rec("regimebook", -0.13, 2.6)]))
        rc = seat.main(["--shadow-ic-report", str(rep), "--incumbent", "regimebook",
                        "--ledger-dir", str(self._ledger_dir(tmp_path))])
        out = capsys.readouterr().out
        assert rc == 3, out
        assert "flat" in out.lower()

    def test_positive_book_keeps_the_seat(self, tmp_path):
        rep = _write(tmp_path / "seat.json", _report([_pooled_rec("regimebook", 0.08, 2.4)]))
        rc = seat.main(["--shadow-ic-report", str(rep), "--incumbent", "regimebook",
                        "--ledger-dir", str(self._ledger_dir(tmp_path))])
        assert rc == 0

    def test_report_lacking_a_horizon_is_a_refusal_not_a_verdict(self, tmp_path, capsys):
        rep = _write(tmp_path / "seat.json",
                     _report([_pooled_rec("regimebook", -0.13, 2.6, horizons=("4", "12", "24"))]))
        rc = seat.main(["--shadow-ic-report", str(rep), "--incumbent", "regimebook",
                        "--ledger-dir", str(self._ledger_dir(tmp_path))])
        out = capsys.readouterr()
        assert rc == 2, out.out
        assert "lacks the 1-bar" in out.out

    def test_report_without_a_pooled_regimebook_record_refuses(self, tmp_path, capsys):
        rep = _write(tmp_path / "seat.json", _report([_pooled_rec("etfflow", 0.1, 2.0)]))
        rc = seat.main(["--shadow-ic-report", str(rep), "--incumbent", "regimebook",
                        "--ledger-dir", str(self._ledger_dir(tmp_path))])
        assert rc == 2
        assert "no POOLED" in capsys.readouterr().err

    def test_ledger_dir_argument_is_honoured_for_availability(self, tmp_path, capsys):
        rep = _write(tmp_path / "seat.json", _report([_pooled_rec("regimebook", -0.13, 2.6)]))
        rc = seat.main(["--shadow-ic-report", str(rep), "--incumbent", "regimebook",
                        "--ledger-dir", str(self._ledger_dir(tmp_path, available=False))])
        out = capsys.readouterr().out
        assert rc == 2 and "UNAVAILABLE" in out

    def test_in_market_rate_is_carried_from_the_record(self, tmp_path):
        rep = _write(tmp_path / "seat.json",
                     _report([_pooled_rec("regimebook", -0.1, 2.0, n=1500, n_directional=0)]))
        got = seat.regimebook_from_shadow_report(rep)
        assert got["in_market_rate"] == 0.0 and got["n"] == 1500


class TestFlatDeciderNamesItself:
    def test_quant_with_rows_but_no_ic_says_no_directional_claim(self, tmp_path, capsys):
        rep = _write(tmp_path / "agent_ic.json", {"agents": {
            "quant": {"horizons": {"1": {"n": 609, "ic": None, "verdict": "INSUFFICIENT"},
                                   "4": {"n": 609, "ic": None, "verdict": "INSUFFICIENT"}}},
        }})
        rc = seat.main(["--ic-report", str(rep), "--incumbent", "regimebook",
                        "--ledger-dir", str(tmp_path)])
        out = capsys.readouterr().out
        assert rc == 2
        assert "no directional claim" in out and "n=609" in out
        assert "--shadow-ic-report" in out
