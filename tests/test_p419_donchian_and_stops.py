"""[P419] The K-line adjustment + the stop-loss verdict, both by measurement.

ETH's trend leg switched SMA200 -> DONCHIAN-100 by the pre-committed chassis
verdict (donchian_switch_lab_p419: ETH +2.161 vs +1.870 net over 6.6y, 2/3
eras, 3x the SMA leg in the MOST RECENT era; per-RT median 375.5bps). BTC and
SOL keep SMA200 (era_wins 1/3 each -- the P243/P244 era-stability rule, even
though SOL's full-window total was higher). seat_alpha's ETH rows are the
DONCHIAN book's own measured table (P320: a seat asserts the edge of the rule
it runs).

The stop sweep (stop_sweep_p419) plus the external literature both said: the
10% venue stop is INSURANCE, not the dip-seller the operator feared -- 6y net
+5.8pp summed, 0.2-0.5 fires/yr, zero live fires, no era harmed; TIGHTER
stops are the measured mistake (5%/8% whipsaw -3.5/-4.5pp on ETH/SOL). The
decided value stays 0.10.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


class TestDonchianSwitch:
    def test_the_switch_set_is_the_decided_value(self):
        from defense.regime_book_shadow import DONCHIAN_TREND_ASSETS
        assert DONCHIAN_TREND_ASSETS == frozenset({"ETH"}), (
            "ETH is the ONE asset that cleared the pre-committed rule "
            "(2/3 eras + full window). BTC/SOL keep SMA200 (era_wins 1/3) -- "
            "widening or reverting this set is a measured live-book change "
            "needing its own P-entry")

    def test_eth_book_version_marks_the_switch(self):
        from defense.regime_book_shadow import BOOKS_VERSION
        assert BOOKS_VERSION["ETH"] == "v2_donchian_trend", (
            "the ledger's book_version is the evidence-discontinuity marker "
            "-- regimebook_ETH rows before/after the switch record different "
            "books and must not be scored as one series")

    def test_donchian_target_bull_breakout_holds(self):
        from defense.regime_book_shadow import donchian_trend_target
        closes = list(np.linspace(1000, 2000, 300))   # steady uptrend
        assert donchian_trend_target(closes) == (1.0, "donchian_hold")

    def test_donchian_target_breakdown_goes_flat(self):
        from defense.regime_book_shadow import donchian_trend_target
        up = list(np.linspace(1000, 2000, 250))
        down = list(np.linspace(2000, 800, 150))      # close below 100-bar low
        assert donchian_trend_target(up + down) == (0.0, "donchian_flat")

    def test_short_history_falls_back_never_reads_flat(self):
        from defense.regime_book_shadow import donchian_trend_target
        assert donchian_trend_target([100.0] * 50) is None, (
            "a too-short window must fall back to the incumbent book -- "
            "absence is not an opinion (P2)")

    def test_record_tick_wiring_donchian_overrides_the_book(self, tmp_path):
        """The one case where the v1 book and donchian DISAGREE: the
        SMA-disagreement series (long decline + late spike above SMA200) is
        FLAT under ETH's old trend book and a 100-bar-high BREAKOUT under
        donchian. The deployed row must say LONG on the donchian leg -- the
        pin that goes red if the record_tick override is unwired (a switch
        that exists only in a constant is decoration, P170)."""
        from defense.regime_book_shadow import RegimeBookShadow
        x = np.linspace(0, 1, 600)
        closes = list(200 - 100 * x)
        closes[-1] = closes[-200] + 10
        h = RegimeBookShadow(data_dir=str(tmp_path))
        rec = h.record_tick("ETH", closes, price=1900.0)
        assert rec is not None and rec["direction"] == 1.0
        assert rec["leg"] == "donchian_hold"
        assert rec["book_version"] == "v2_donchian_trend"

    def test_labels_are_the_canonical_state_machine(self):
        """The live leg and the forward donchian ledger must share ONE label
        implementation (P172) -- drift here forward-tests a different rule."""
        import inspect
        from defense import regime_book_shadow as rbs
        src = inspect.getsource(rbs.donchian_trend_target)
        assert "trend_rule_shadow" in src and "donchian_labels" in src


class TestSeatAlphaEthDonchian:
    def test_eth_rows_are_the_producer_values(self):
        from core.seat_alpha import (REGIMEBOOK_ALPHA_BY_ERA,
                                     REGIMEBOOK_ALPHA_BPS_PER_ROUND_TRIP)
        assert REGIMEBOOK_ALPHA_BY_ERA["ETH"] == {
            "pre_design": 674.4, "design": 291.6, "validation": 375.5}
        assert REGIMEBOOK_ALPHA_BPS_PER_ROUND_TRIP["ETH"] == 375.5

    def test_series_map_names_eth_donchian(self):
        from core.seat_alpha import REGIMEBOOK_SERIES_BY_ASSET
        assert REGIMEBOOK_SERIES_BY_ASSET["ETH"] == "donchian"
        for a in ("BTC", "SOL", "XRP", "BNB"):
            assert REGIMEBOOK_SERIES_BY_ASSET[a] == "book", (
                f"{a} was NOT switched -- era_wins 1/3 (P243/P244); moving "
                f"it here without a new measured verdict is drift")

    def test_resolve_seat_edge_asserts_the_donchian_median(self):
        from core.seat_alpha import resolve_seat_edge
        v = resolve_seat_edge("ETH", "regimebook", 1.0, 30.0,
                              True, True, 2252.0)
        assert v == 375.5

    def test_calibrator_verify_uses_the_series_map(self):
        src = (REPO / "training" / "seat_alpha_calibration.py").read_text(
            encoding="utf-8-sig")
        assert "REGIMEBOOK_SERIES_BY_ASSET" in src
        assert 'choices=("book", "trend", "donchian")' in src


class TestStopDecidedValue:
    def test_stop_pct_stays_at_the_measured_value(self):
        live = json.loads((REPO / "configs" / "live_high_risk.json"
                           ).read_text(encoding="utf-8"))
        assert live.get("coinbase_protective_stop_pct") == 0.10, (
            "P419 decided value, validated by the 6y sweep "
            "(stop_sweep_p419: +5.8pp net summed, 0.2-0.5 fires/yr, no era "
            "harmed) AND the external literature (York/RA: stops on a trend "
            "book add no alpha; the venue stop is process-death insurance). "
            "TIGHTENING is the measured mistake: 5%/8% whipsaw -3.5/-4.5pp "
            "on ETH/SOL. Moving this in either direction needs a new "
            "measurement, not an intuition")

    def test_the_sweep_report_exists_with_the_verdict_inputs(self):
        rep = (REPO / "training" / "reports" / "stop_sweep_p419.json")
        if not rep.exists():
            import pytest
            pytest.skip("operator-local report (P213)")
        d = json.loads(rep.read_text(encoding="utf-8"))
        assert "0.1" in d["summary_net_delta_3asset"]
