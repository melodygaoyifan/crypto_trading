"""[P377] Pin the drawdown-reduction-overlay verdict so the product bar can't be
quietly relaxed — [P382] re-pointed to the COST-CORRECTED result.

The reframe (run the certified trend signal as a long-or-flat overlay on a
hold) was recorded by P377 as a MODEST product that FAILS the strict bar. P382
found the lab charged the ROUND-TRIP cost on every LEG (a 2x overcharge that
fell almost entirely on the overlay — hold pays only its two end legs) and
re-ran it at the repo's RT/2-per-leg convention. The bar is UNCHANGED; the
corrected numbers (pre-correction -> corrected):

    return kept   BTC 66% -> 85%   ETH 76% -> 98%   SOL 68% -> 80%
    DD cut        BTC 37% -> 40%   ETH 28% -> 42%   SOL  5% ->  8%
    per-asset verdict (full + >=2/3 eras): BTC fails (1/3 eras),
        ETH VIABLE PRODUCT (2/3 eras), SOL fails (DD barely cut)
    portfolio: DD cut 25% -> 37%, ret kept 70% -> 85%, Sharpe 1.05 -> 1.30
        -> VIABLE PRODUCT on the full-window bar AS CODED by P377 (the
           portfolio verdict never carried the era clause); the era
           breakdown is now REPORTED beside it (informational, P382) and
           reads 1/3 — so the portfolio pass is a magnitude pass, NOT an
           era-robust one. Deciding whether the portfolio verdict should
           carry the era clause is its own recorded decision.

These tests pin the DECIDED corrected values (P237): a silent revert to the
2x overcharge and a silent loosening of the bar both go red.
"""
import json
from pathlib import Path
import pytest

REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / "training" / "reports" / "overlay_backtest_p377.json"
PRECORRECTION = REPO / "training" / "reports" / "overlay_backtest_p377_p382_precorrection.json"


def _verdict(dd_cut, ret_keep, era_pass, dd_min=0.25, keep_min=0.50):
    full_ok = dd_cut >= dd_min and ret_keep >= keep_min
    return "VIABLE PRODUCT" if (full_ok and era_pass >= 2) else "fails"


def test_bar_requires_both_magnitude_and_era_robustness():
    # magnitude met but not era-robust -> fails (the BTC case)
    assert _verdict(0.25, 0.70, era_pass=1) == "fails"
    # magnitude not met -> fails even if era-robust
    assert _verdict(0.10, 0.70, era_pass=3) == "fails"
    # both -> viable (the ETH case)
    assert _verdict(0.30, 0.60, era_pass=2) == "VIABLE PRODUCT"


@pytest.mark.skipif(not REPORT.exists(), reason="report is operator-local (built by the lab)")
def test_overlay_cuts_drawdown_and_raises_sharpe_at_portfolio_level():
    # the genuine product value: less drawdown, higher Sharpe, most of the return
    d = json.loads(REPORT.read_text(encoding="utf-8"))
    assert "COST_RT/2" in d.get("cost_convention", ""), (
        "report was not written by the P382 per-leg lab")
    p = d["portfolio"]
    assert p["overlay"]["maxdd_pct"] < p["hold"]["maxdd_pct"]      # less drawdown
    assert p["overlay"]["sharpe"] >= p["hold"]["sharpe"]           # >= risk-adjusted
    assert p["ret_keep"] >= 0.50                                   # keeps most return
    assert p["overlay"]["total_pct"] < p["hold"]["total_pct"]      # but gives up some


@pytest.mark.skipif(not REPORT.exists(), reason="report is operator-local")
def test_recorded_verdicts_are_the_corrected_decided_ones():
    """[P382] Per-asset verdicts under the unchanged bar (full + >=2/3 eras):
    BTC fails on era robustness, ETH is VIABLE, SOL fails on drawdown. The
    portfolio reads VIABLE on the full-window rule as coded, with the era
    clause reported beside it at 1/3 — pinned both ways so nobody reads the
    portfolio pass as era-robust, and nobody reverts it to 'fails' by
    re-introducing the 2x overcharge (which took its DD cut to 0.249)."""
    d = json.loads(REPORT.read_text(encoding="utf-8"))
    a = d["assets"]
    assert a["BTC"]["verdict"] == "fails"
    assert a["ETH"]["verdict"] == "VIABLE PRODUCT"
    assert a["SOL"]["verdict"] == "fails"
    assert a["SOL"]["dd_cut"] < 0.25            # SOL's DD is barely cut
    assert sum(e["pass"] for e in a["BTC"]["eras"].values()) == 1
    assert sum(e["pass"] for e in a["ETH"]["eras"].values()) == 2
    assert sum(e["pass"] for e in a["SOL"]["eras"].values()) == 0
    # decided corrected magnitudes (tolerance = the report's rounding)
    assert abs(a["BTC"]["ret_keep"] - 0.848) < 0.005
    assert abs(a["ETH"]["ret_keep"] - 0.979) < 0.005
    assert abs(a["SOL"]["ret_keep"] - 0.800) < 0.005
    p = d["portfolio"]
    assert p["verdict"] == "VIABLE PRODUCT"
    assert abs(p["dd_cut"] - 0.37) < 0.005
    assert abs(p["ret_keep"] - 0.854) < 0.005
    assert "full-window" in p["verdict_rule"]
    assert p["era_pass_informational"] == 1, p["eras_informational"]
    # the era clause, applied to the portfolio the way it is applied per asset,
    # would NOT pass — keep that visible
    assert _verdict(p["dd_cut"], p["ret_keep"], p["era_pass_informational"]) == "fails"


@pytest.mark.skipif(not PRECORRECTION.exists(), reason="pre-correction copy is operator-local")
def test_the_precorrection_copy_is_kept_and_reads_the_old_numbers():
    d = json.loads(PRECORRECTION.read_text(encoding="utf-8"))
    assert abs(d["portfolio"]["ret_keep"] - 0.695) < 0.005
    assert d["portfolio"]["verdict"] == "fails"
    assert "cost_convention" not in d   # it predates the stamp
