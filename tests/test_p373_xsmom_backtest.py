"""[P373] Pin the xsmom verdict so nobody relaxes the bar into a pass.

xsmom was the ONE offline-backtestable September candidate never run. It has now
been run (training/xsmom_backtest_lab.py) and the genuine 8-asset cross-section
FAILS every variant against equal-weight buy-and-hold on total return. These
tests pin (a) the pre-committed bar exactly as written in the docstring, and
(b) that the recorded 8-asset result does not clear it — so a future edit that
loosens the bar or re-reads the window until something passes goes red.
"""
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / "training" / "reports" / "xsmom_backtest_p373.json"


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
def test_the_genuine_8_asset_cross_section_does_not_earn():
    d = json.loads(REPORT.read_text(encoding="utf-8"))
    u = d["universe8"]
    bh = u["buy_and_hold_eqw"]["total_pct"]
    verdicts = {tag: _verdict(v, bh) for tag, v in u["variants"].items()}
    # Every 8-asset variant fails. If one ever EARNS, that is a real finding and
    # this test should be updated deliberately (P141), not to make CI green.
    assert all(v == "fails" for v in verdicts.values()), verdicts
    # And the long-short variants lose money outright before funding carry
    # (which is not modelled and would only make the shorts worse).
    assert u["variants"]["top2_long_short"]["total_pct"] < 0
    assert u["variants"]["top3_long_short"]["total_pct"] < 0


@pytest.mark.skipif(not REPORT.exists(), reason="report is operator-local (built by the lab)")
def test_the_3_asset_beater_is_the_p277_degenerate_case_not_a_cross_section():
    # The one variant that beats B&H is top1-of-3 majors — 'not really a
    # cross-section' (P277). Pin that it is the 3-asset universe and that its
    # edge over B&H is thin (rides SOL, per P325), so it is not read as an
    # 8-asset xsmom win.
    d = json.loads(REPORT.read_text(encoding="utf-8"))
    e = d["universe3_executable"]
    edge = e["top1_long_only"]["total_pct"] - e["buy_and_hold_eqw"]["total_pct"]
    assert 0 < edge < 100  # beats, but by a thin margin over 6 years
