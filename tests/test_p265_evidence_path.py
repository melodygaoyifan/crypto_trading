"""[P265] Evidence-path fixes — the streams the September promotion reads
depend on.

  * RegimeBookShadow's funding z had NO staleness bound: once >=30 days of
    history persisted, a sustained fapi outage froze the z and the BTC
    funding legs (the roster's ONLY uncertified component, P262) kept
    trading it into the forward ledger, unmarked and warn-once-silent.
  * The reviewer tools (agent_ic_review / slope_calibrator /
    trend_regime_review) priced the newest forward returns against Kraken's
    IN-PROGRESS candle while a comment claimed the opposite (P253c fixed
    only the regime-book harness).
  * The P237 tripwire's "4 consecutive weekly reports" was actually "the
    last 4 report DAYS" — four ad-hoc calibrator runs in one debugging week
    could demand the trend-injection removal.
  * EA-4a POPPED exit_trigger_tag before the C11 attribution's
    actual_exit_type read — the exit-type column was the constant
    "FULL_EXIT" whenever ea_tracker existed (production).
  * The heartbeat's equity (and data/equity_history.jsonl) measured the
    structurally-flat Kraken book only, gated on the audit_manager logging
    object — a Sharpe from that file was the Sharpe of a constant.
"""

import json
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
MAIN_SRC = (REPO / "main.py").read_text(encoding="utf-8-sig")
EXEC_SRC = (REPO / "core" / "execution_service.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. funding-z staleness bound
# ---------------------------------------------------------------------------

def _harness(tmp_path, last_day_offset: int):
    from defense.regime_book_shadow import RegimeBookShadow
    h = RegimeBookShadow(data_dir=str(tmp_path))
    today = datetime.now(timezone.utc).date()
    hist = {}
    for i in range(40):
        d = today - timedelta(days=last_day_offset + (39 - i))
        # strong positive tail so the z clears the funding_short bar
        hist[d.isoformat()] = 0.0001 * (1 + (i / 10.0 if i > 35 else 0))
    h._fund_hist["BTC"] = hist
    return h


def _bear_closes():
    # close < SMA200 -> "bear" (the funding_short leg's regime)
    return [100.0 - i * 0.1 for i in range(300)]


class TestFundingStalenessBound:
    def test_frozen_history_goes_flat_with_reason(self, tmp_path):
        h = _harness(tmp_path, last_day_offset=10)  # newest day: 10d ago
        rec = h.record_tick("BTC", _bear_closes(), price=70.0)
        assert rec is not None
        assert rec["funding_z"] is None, (
            "a 10-day-frozen funding history still produced a z — the "
            "funding legs keep trading a frozen input into the forward "
            "ledger (P265; the September read would judge the outage, not "
            "the strategy)")
        assert rec["funding_age_days"] == 10
        assert not rec["leg"].startswith("funding"), rec["leg"]

    def test_fresh_history_still_feeds_the_legs(self, tmp_path):
        h = _harness(tmp_path, last_day_offset=1)  # newest day: yesterday
        rec = h.record_tick("BTC", _bear_closes(), price=70.0)
        assert rec is not None
        assert rec["funding_z"] is not None, (
            "the staleness bound over-reached: yesterday's completed day is "
            "the NORMAL cadence and must feed the legs")
        assert rec["funding_age_days"] == 1

    def test_age_is_recorded_on_every_row(self, tmp_path):
        h = _harness(tmp_path, last_day_offset=1)
        rec = h.record_tick("BTC", _bear_closes(), price=70.0)
        assert "funding_age_days" in rec, (
            "stale-z rows are not filterable post-hoc without the age field")

    def test_stale_warning_rearms_per_transition(self, tmp_path):
        h = _harness(tmp_path, last_day_offset=10)
        h.record_tick("BTC", _bear_closes(), price=70.0)
        assert h._funding_stale.get("BTC") is True
        # recovery: rebuild fresh history
        today = datetime.now(timezone.utc).date()
        h._fund_hist["BTC"] = {
            (today - timedelta(days=40 - i)).isoformat(): 0.0001
            for i in range(39)}
        h.record_tick("BTC", _bear_closes(), price=70.0)
        assert h._funding_stale.get("BTC") is False, (
            "recovery did not clear the stale latch — the next outage would "
            "be silent (the once-per-process trap this replaces)")


# ---------------------------------------------------------------------------
# 2. in-progress candle dropped by the reviewer family
# ---------------------------------------------------------------------------

class TestPartialCandleDropped:
    def test_agent_ic_review_drops_the_last_row(self):
        src = (REPO / "analytics" / "ic" / "agent_ic_review.py").read_text(
            encoding="utf-8")
        assert "rows = rows[:-1]" in src, (
            "fetch_closes serves Kraken's in-progress candle again — the "
            "newest forward returns are priced on a provisional close")

    def test_trend_regime_review_drops_the_last_row(self):
        src = (REPO / "scripts" / "trend_regime_review.py").read_text(
            encoding="utf-8")
        assert "rows = rows[:-1]" in src

    def test_slope_calibrator_inherits_via_the_shared_fetcher(self):
        src = (REPO / "analytics" / "calibration" /
               "slope_calibrator.py").read_text(encoding="utf-8")
        assert "fetch_closes" in src and "agent_ic_review" in src, (
            "slope_calibrator no longer shares agent_ic_review's fetcher — "
            "the P172 one-resolver discipline broke and the candle fix does "
            "not reach it")


# ---------------------------------------------------------------------------
# 3. tripwire weekly spacing
# ---------------------------------------------------------------------------

def _write_report(d: Path, day: str, tradeable: bool):
    rep = {
        "generated": f"{day}T06:20:00+00:00",
        "assets": {a: {"4h": {"vs_threshold":
                              "TRADEABLE" if tradeable else "below threshold"}}
                   for a in ("BTC", "ETH", "SOL")},
    }
    (d / f"slope_{day}.json").write_text(json.dumps(rep), encoding="utf-8")


def _run_tripwire(monkeypatch, reports_dir: Path, today: str) -> int:
    import analytics.calibration.tripwire_check as tw
    monkeypatch.setattr(sys, "argv", [
        "tripwire_check", "--reports-dir", str(reports_dir),
        "--today", today])
    return tw.main()


class TestTripwireWeeklySpacing:
    def test_four_same_week_reports_do_not_fire(self, monkeypatch, tmp_path):
        for i in range(4):
            _write_report(tmp_path,
                          (date(2026, 9, 7) + timedelta(days=i)).isoformat(),
                          tradeable=False)
        rc = _run_tripwire(monkeypatch, tmp_path, "2026-09-11")
        assert rc != 3, (
            "four GATE-CLOSED reports from ONE debugging week fired the "
            "tripwire — the P237 criterion is four consecutive WEEKLY "
            "reports (P265)")

    def test_four_weekly_reports_do_fire(self, monkeypatch, tmp_path):
        for i in range(4):
            _write_report(tmp_path,
                          (date(2026, 8, 17) + timedelta(days=7 * i)).isoformat(),
                          tradeable=False)
        rc = _run_tripwire(monkeypatch, tmp_path, "2026-09-08")
        assert rc == 3, (
            "four genuinely weekly GATE-CLOSED reports past the date gate "
            "did NOT fire — the spacing filter over-reached")

    def test_tradeable_weeks_do_not_fire(self, monkeypatch, tmp_path):
        for i in range(4):
            _write_report(tmp_path,
                          (date(2026, 8, 17) + timedelta(days=7 * i)).isoformat(),
                          tradeable=True)
        rc = _run_tripwire(monkeypatch, tmp_path, "2026-09-08")
        assert rc != 3


# ---------------------------------------------------------------------------
# 4. exit_trigger_tag ordering
# ---------------------------------------------------------------------------

class TestExitTriggerTagOrdering:
    def test_ea4a_reads_and_the_clear_follows_the_c11_reader(self):
        ea4a = EXEC_SRC.index("[EA-4a] Exit Alpha Tracker -record full exit")
        ea4a_seg = EXEC_SRC[ea4a:ea4a + 1500]
        assert 'exit_trigger_tag.get(asset, "UNKNOWN")' in ea4a_seg, (
            "EA-4a pops the tag again — C11's actual_exit_type reads the "
            "default 'FULL_EXIT' on every production exit (P265)")
        c11_read = EXEC_SRC.index(
            'actual_exit_type=ctx.exit_trigger_tag.get(asset, "FULL_EXIT")')
        # the clearing pop must exist AFTER the C11 reader (the FIX-H3 pop
        # near the top of the file is a different, pre-entry site)
        clear = EXEC_SRC.find(
            "ctx.exit_trigger_tag.pop(asset, None)", c11_read)
        assert clear > c11_read, (
            "the tag is cleared before the C11 attribution reader — the "
            "exit-type column is a constant again")


# ---------------------------------------------------------------------------
# 5. heartbeat equity
# ---------------------------------------------------------------------------

class TestHeartbeatEquity:
    def test_equity_read_is_outside_the_audit_manager_gate(self):
        anchor = MAIN_SRC.index("[P265] The equity read lives OUTSIDE the audit_manager gate")
        # [P287] window widened 2500 -> 6000: the per-half equity_valid
        # comment block sits between the read and the gate now. The
        # invariant being pinned (read BEFORE the audit_manager gate) is
        # unchanged and still asserted below.
        seg = MAIN_SRC[anchor:anchor + 6000]
        read_idx = seg.index("get_equity_safe")
        gate_idx = seg.index("if self.audit_manager:")
        assert read_idx < gate_idx, (
            "the equity read moved back inside the audit_manager gate — a "
            "logging object gates the equity-history data feed again")

    def test_history_records_carry_both_denominations(self):
        anchor = MAIN_SRC.index("[EQUITY-LOG] Append equity snapshot")
        seg = MAIN_SRC[anchor:anchor + 1500]
        assert '"kraken_equity"' in seg and '"sleeve_equity"' in seg, (
            "equity_history.jsonl records the unlabeled single figure again "
            "— downstream cannot tell the constant Kraken book from the "
            "moving sleeve")

    def test_the_combined_figure_is_the_sum(self):
        anchor = MAIN_SRC.index("_hb_equity = _hb_kraken_eq + _hb_sleeve_eq")
        assert anchor > 0
