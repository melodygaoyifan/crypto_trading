"""[P374] Pin the cross-sectional funding-carry verdict.

This was the LAST un-run offline candidate the P373-era research audit found:
6y of daily funding for all 8 assets, never ranked cross-sectionally. It has now
been run (training/funding_carry_xs_lab.py) net of cost AND carry and FAILS every
variant against equal-weight buy-and-hold. These tests pin the pre-committed bar
and that the recorded result does not clear it, so a future edit that relaxes the
bar or re-reads the window until something passes goes red.
"""
import json
from pathlib import Path
import pytest

REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / "training" / "reports" / "funding_carry_xs_p374.json"


def _verdict(variant: dict, bh_total: float) -> str:
    beats = variant["total_pct"] > bh_total
    eras_pos = sum(1 for v in variant["eras"].values() if v > 0)
    return "EARNS" if (beats and eras_pos >= 2) else "fails"


def test_the_bar_requires_both_beat_bnh_and_two_of_three_eras():
    assert _verdict({"total_pct": 500.0, "eras": {"a": 10, "b": -5, "c": -5}}, 400.0) == "fails"
    assert _verdict({"total_pct": 300.0, "eras": {"a": 100, "b": 100, "c": 100}}, 400.0) == "fails"
    assert _verdict({"total_pct": 500.0, "eras": {"a": 100, "b": 100, "c": -5}}, 400.0) == "EARNS"


@pytest.mark.skipif(not REPORT.exists(), reason="report is operator-local (built by the lab)")
def test_no_funding_carry_variant_earns():
    d = json.loads(REPORT.read_text(encoding="utf-8"))
    bh = d["buy_and_hold_eqw"]["total_pct"]
    verdicts = {t: _verdict(v, bh) for t, v in d["variants"].items()}
    # Every variant fails. An EARNS here is a real finding and must be updated
    # deliberately (P141), not to make CI green.
    assert all(v == "fails" for v in verdicts.values()), verdicts
    # The dollar-neutral variants lose money outright — shorting the highest
    # funder shorts the trending winner.
    assert d["variants"]["bottom2_dollar_neutral"]["total_pct"] < 0
    assert d["variants"]["bottom3_dollar_neutral"]["total_pct"] < 0
