"""[P377] Pin the drawdown-reduction-overlay verdict so the product bar can't be
quietly relaxed. The reframe (run the certified trend signal as a long-or-flat
overlay on a hold) is a MODEST risk-adjusted product: at the portfolio level it
cuts maxDD ~25% and keeps ~70% of return with a higher Sharpe, but it FAILS the
strict era-robustness bar (choppy-bull whipsaw; SOL barely helped).
"""
import json
from pathlib import Path
import pytest

REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / "training" / "reports" / "overlay_backtest_p377.json"


def _verdict(dd_cut, ret_keep, era_pass, dd_min=0.25, keep_min=0.50):
    full_ok = dd_cut >= dd_min and ret_keep >= keep_min
    return "VIABLE PRODUCT" if (full_ok and era_pass >= 2) else "fails"


def test_bar_requires_both_magnitude_and_era_robustness():
    # magnitude met but not era-robust -> fails (the real overlay case)
    assert _verdict(0.25, 0.70, era_pass=1) == "fails"
    # magnitude not met -> fails even if era-robust
    assert _verdict(0.10, 0.70, era_pass=3) == "fails"
    # both -> viable
    assert _verdict(0.30, 0.60, era_pass=2) == "VIABLE PRODUCT"


@pytest.mark.skipif(not REPORT.exists(), reason="report is operator-local (built by the lab)")
def test_overlay_cuts_drawdown_and_raises_sharpe_at_portfolio_level():
    # the genuine product value: less drawdown, higher Sharpe, most of the return
    d = json.loads(REPORT.read_text(encoding="utf-8"))
    p = d["portfolio"]
    assert p["overlay"]["maxdd_pct"] < p["hold"]["maxdd_pct"]      # less drawdown
    assert p["overlay"]["sharpe"] >= p["hold"]["sharpe"]           # >= risk-adjusted
    assert p["ret_keep"] >= 0.50                                   # keeps most return
    assert p["overlay"]["total_pct"] < p["hold"]["total_pct"]      # but gives up some


@pytest.mark.skipif(not REPORT.exists(), reason="report is operator-local")
def test_recorded_verdict_is_fails_not_relaxed():
    # pre-committed bar: the portfolio & per-asset verdicts are 'fails' (era
    # robustness). A future PASS is a real finding to act on, not a relaxed bar.
    d = json.loads(REPORT.read_text(encoding="utf-8"))
    assert d["portfolio"]["verdict"] == "fails"
    # SOL is the weakest — its DD is barely cut
    assert d["assets"]["SOL"]["dd_cut"] < 0.25
