"""[P374] Pin the cross-sectional funding-carry verdict — [P382] re-pointed to
the COST-CORRECTED result without touching the bar.

This was the LAST un-run offline candidate the P373-era research audit found:
6y of daily funding for all 8 assets, never ranked cross-sectionally. P374 ran
it (training/funding_carry_xs_lab.py) net of cost AND carry and recorded every
variant as FAILING against equal-weight buy-and-hold. P382 found the lab
charged the ROUND-TRIP cost on every LEG (2x overcharge) and re-ran it at the
repo's RT/2-per-leg convention. THE VERDICT STANDS — every variant still fails
— but the magnitudes moved a great deal, which is the honest before/after:

    8-asset, net of cost+carry (pre-correction -> corrected), eqw B&H +466.7:
      bottom2_long_only        -329.2 ->  +102.0   fails (below B&H)
      bottom3_long_only        -176.1 ->  +179.7   fails (below B&H)
      bottom2_dollar_neutral  -1433.0 ->  -515.4   fails (loses outright)
      bottom3_dollar_neutral  -1207.3 ->  -467.2   fails (loses outright)

These tests pin the pre-committed bar and the DECIDED corrected values (P237:
a silent revert to the 2x overcharge goes red too).
"""
import json
from pathlib import Path
import pytest

REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / "training" / "reports" / "funding_carry_xs_p374.json"
PRECORRECTION = REPO / "training" / "reports" / "funding_carry_xs_p374_p382_precorrection.json"


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
    assert "COST_RT/2" in d.get("cost_convention", ""), (
        "report was not written by the P382 per-leg lab")
    bh = d["buy_and_hold_eqw"]["total_pct"]
    verdicts = {t: _verdict(v, bh) for t, v in d["variants"].items()}
    # Every variant fails. An EARNS here is a real finding and must be updated
    # deliberately (P141), not to make CI green.
    assert all(v == "fails" for v in verdicts.values()), verdicts
    # The dollar-neutral variants lose money outright — shorting the highest
    # funder shorts the trending winner.
    assert d["variants"]["bottom2_dollar_neutral"]["total_pct"] < 0
    assert d["variants"]["bottom3_dollar_neutral"]["total_pct"] < 0


@pytest.mark.skipif(not REPORT.exists(), reason="report is operator-local (built by the lab)")
def test_the_corrected_magnitudes_are_the_decided_ones():
    """[P382] The long-only variants are POSITIVE at the honest per-leg cost
    (they were deeply negative under the 2x overcharge) and still below B&H.
    Pinned so a revert to full-RT-per-leg (-329.2 / -176.1) goes red."""
    d = json.loads(REPORT.read_text(encoding="utf-8"))
    bh = d["buy_and_hold_eqw"]["total_pct"]
    v = d["variants"]
    assert abs(bh - 466.7) < 0.2
    assert abs(v["bottom2_long_only"]["total_pct"] - 102.0) < 0.2
    assert abs(v["bottom3_long_only"]["total_pct"] - 179.7) < 0.2
    assert 0 < v["bottom2_long_only"]["total_pct"] < bh
    assert 0 < v["bottom3_long_only"]["total_pct"] < bh
    assert abs(v["bottom2_dollar_neutral"]["total_pct"] - (-515.4)) < 0.2
    assert abs(v["bottom3_dollar_neutral"]["total_pct"] - (-467.2)) < 0.2
    # every variant is still negative in 2025-26 — the P243/P244 carry
    # inversion the lab was built to price
    for tag, s in v.items():
        assert s["eras"]["2025-26"] < 0, (tag, s["eras"])


@pytest.mark.skipif(not PRECORRECTION.exists(), reason="pre-correction copy is operator-local")
def test_the_precorrection_copy_is_kept_and_reads_the_old_numbers():
    d = json.loads(PRECORRECTION.read_text(encoding="utf-8"))
    assert abs(d["variants"]["bottom2_long_only"]["total_pct"] - (-329.2)) < 0.2
    assert "cost_convention" not in d   # it predates the stamp
