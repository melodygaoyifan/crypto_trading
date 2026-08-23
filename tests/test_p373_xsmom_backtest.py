"""[P373] Pin the xsmom verdict so nobody relaxes the bar into a pass — and
[P382] re-point the pins to the COST-CORRECTED result without touching the bar.

xsmom was the ONE offline-backtestable September candidate never run. P373 ran
it (training/xsmom_backtest_lab.py) and recorded every 8-asset variant as
FAILING against equal-weight buy-and-hold. P382 then found the lab charged the
ROUND-TRIP cost on every LEG (a 2x overcharge, the class P281/P287 had fixed
elsewhere) and re-ran it at the repo's RT/2-per-leg convention. The bar is
UNCHANGED; what the bar says about the corrected numbers is different:

    8-asset, net of cost (pre-correction -> corrected), eqw B&H +453.5:
      top2_long_only   +385.7 -> +514.4   EARNS (beats, 2/3 eras positive)
      top3_long_only   +396.9 -> +502.6   EARNS (beats, 2/3 eras positive)
      top2_long_short   -96.0 -> +170.8   fails (positive, still loses to B&H)
      top3_long_short   -38.4 -> +177.8   fails (positive, still loses to B&H)

These tests pin (a) the pre-committed bar exactly as written in the lab's
docstring and (b) the DECIDED corrected values (P237: a silent revert to the
2x overcharge — or a silent loosening — both go red). The P373 executability
caveats are untouched and re-pinned: only BTC/ETH/SOL are routable (P292), so
an 8-asset EARNS is a SIGNAL finding, not a tradeable strategy.
"""
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / "training" / "reports" / "xsmom_backtest_p373.json"
PRECORRECTION = REPO / "training" / "reports" / "xsmom_backtest_p373_p382_precorrection.json"


def _verdict(variant: dict, bh_total: float) -> str:
    """The pre-committed rule, in one place: beat B&H on TOTAL RETURN AND be
    positive in >= 2 of 3 eras. Anything else is 'fails'."""
    beats = variant["total_pct"] > bh_total
    eras_pos = sum(1 for v in variant["eras"].values() if v > 0)
    return "EARNS" if (beats and eras_pos >= 2) else "fails"


def test_the_bar_requires_both_beat_bnh_and_two_of_three_eras():
    # beats but only 1/3 eras -> fails
    assert _verdict({"total_pct": 500.0, "eras": {"a": 10, "b": -5, "c": -5}}, 400.0) == "fails"
    # 3/3 eras but loses to B&H -> fails
    assert _verdict({"total_pct": 300.0, "eras": {"a": 100, "b": 100, "c": 100}}, 400.0) == "fails"
    # both -> earns (the only pass)
    assert _verdict({"total_pct": 500.0, "eras": {"a": 100, "b": 100, "c": -5}}, 400.0) == "EARNS"


@pytest.mark.skipif(not REPORT.exists(), reason="report is operator-local (built by the lab)")
def test_the_corrected_8_asset_read_is_the_decided_one():
    """[P382] The report must carry the corrected cost convention and the
    verdicts that fall from it — long-only EARNS, long-short fails. A revert
    to full-RT-per-leg drops top2_long_only back to ~+385.7 (below B&H) and
    long-short back below zero; both would fail here."""
    d = json.loads(REPORT.read_text(encoding="utf-8"))
    assert "COST_RT/2" in d.get("cost_convention", ""), (
        "report was not written by the P382 per-leg lab")
    u = d["universe8"]
    bh = u["buy_and_hold_eqw"]["total_pct"]
    verdicts = {tag: _verdict(v, bh) for tag, v in u["variants"].items()}
    assert verdicts["top2_long_only"] == "EARNS", verdicts
    assert verdicts["top3_long_only"] == "EARNS", verdicts
    assert verdicts["top2_long_short"] == "fails", verdicts
    assert verdicts["top3_long_short"] == "fails", verdicts
    # decided corrected values (P237); tolerance is the report's 0.1 rounding
    assert abs(bh - 453.5) < 0.2
    assert abs(u["variants"]["top2_long_only"]["total_pct"] - 514.4) < 0.2
    assert abs(u["variants"]["top3_long_only"]["total_pct"] - 502.6) < 0.2
    # long-short is POSITIVE now but still below B&H — and funding carry is
    # not modelled, which would only make the short legs worse.
    assert 0 < u["variants"]["top2_long_short"]["total_pct"] < bh
    assert 0 < u["variants"]["top3_long_short"]["total_pct"] < bh
    # every variant is still negative in the most recent era (P243/P244
    # era-fragility signature) — an EARNS on the bar is not era-robust
    for tag, v in u["variants"].items():
        assert v["eras"]["2025-26"] < 0, (tag, v["eras"])


@pytest.mark.skipif(not PRECORRECTION.exists(), reason="pre-correction copy is operator-local")
def test_the_precorrection_copy_is_kept_and_reads_the_old_numbers():
    """History is preserved, not overwritten: the P373-era numbers live on
    under the _p382_precorrection name so the before/after is auditable."""
    d = json.loads(PRECORRECTION.read_text(encoding="utf-8"))
    u = d["universe8"]
    assert abs(u["variants"]["top2_long_only"]["total_pct"] - 385.7) < 0.2
    assert u["variants"]["top2_long_short"]["total_pct"] < 0
    assert "cost_convention" not in d   # it predates the stamp


@pytest.mark.skipif(not REPORT.exists(), reason="report is operator-local (built by the lab)")
def test_the_3_asset_beater_is_the_p277_degenerate_case_not_a_cross_section():
    # The 3-asset top1 variant beats B&H — 'not really a cross-section'
    # (P277): k=1-of-3 correlated majors is a single-asset momentum TIMER and
    # rides SOL (P325). Pin that it is the 3-asset universe and pin the
    # corrected edge (P382: +151.7 pts over 6y, was +57.6 under the 2x
    # overcharge) so it is never read as an 8-asset xsmom win.
    d = json.loads(REPORT.read_text(encoding="utf-8"))
    e = d["universe3_executable"]
    edge = e["top1_long_only"]["total_pct"] - e["buy_and_hold_eqw"]["total_pct"]
    assert abs(edge - 151.7) < 0.3
