"""[P248-GP2] Regime-book shadow harness: labels causal, funding causal,
book targets match the p247_leakfix winners verbatim, SOL degradation is
declared, absence never reads as signal, ledger and funding history persist.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from defense.regime_book_shadow import (
    RegimeBookShadow, regime_label, causal_funding_z, book_target,
    BOOKS_VERSION, MIN_BARS, FUND_Z_WINDOW,
)


# ---------------------------------------------------------------- labels
def _mk_closes(trend="bull", n=600):
    x = np.linspace(0, 1, n)
    if trend == "bull":
        return list(100 * (1 + 0.5 * x))          # rising: above SMA, mom>0
    if trend == "bear":
        return list(100 * (1 - 0.5 * x))
    # peace = the two indicators DISAGREE: long decline (540-bar momentum
    # negative) with a late spike that puts the close above SMA200
    closes = list(200 - 100 * x)
    closes[-1] = closes[-200] + 10                # above the 200-bar average
    return closes


def test_labels_truth_table():
    assert regime_label(_mk_closes("bull")) == "bull"
    assert regime_label(_mk_closes("bear")) == "bear"
    assert regime_label(_mk_closes("peace")) == "peace"


def test_warmup_refuses():
    assert regime_label(_mk_closes("bull", n=MIN_BARS - 1)) == "warmup"
    assert book_target("BTC", "warmup", 2.0) == (0.0, "warmup")


# ---------------------------------------------------------------- funding
def test_funding_z_needs_full_window():
    assert causal_funding_z(None) is None
    assert causal_funding_z([0.0001] * (FUND_Z_WINDOW - 1)) is None


def test_funding_z_of_a_spike_is_positive():
    series = [0.0001] * (FUND_Z_WINDOW - 1) + [0.001]
    assert causal_funding_z(series) > 2.0


def test_missing_funding_makes_cells_flat_with_named_reason():
    """Absence must never read as a neutral zero signal (P2/P199): the BTC
    funding cells go FLAT with a reason, not to a fabricated z=0 trade."""
    t, leg = book_target("BTC", "peace", None)
    assert t == 0.0 and leg == "flat_no_funding_history"
    t, leg = book_target("BTC", "bear", None)
    assert t == 0.0


# ---------------------------------------------------------------- books
def test_btc_book_matches_p247_winners():
    assert book_target("BTC", "bull", 0.0) == (1.0, "hold")
    assert book_target("BTC", "bear", 1.5)[0] == -1.0     # funding_short thr 1.0
    assert book_target("BTC", "bear", 0.5)[0] == 0.0
    assert book_target("BTC", "peace", 0.9)[0] == -1.0    # contrarian thr 0.5
    assert book_target("BTC", "peace", -0.9)[0] == 1.0
    assert book_target("BTC", "peace", 0.2)[0] == 0.0


def test_eth_book_is_trend_only():
    assert book_target("ETH", "bull", None)[0] == 1.0
    assert book_target("ETH", "bear", 3.0)[0] == 0.0, (
        "ETH must never short on funding — its measured book is trend-only"
    )
    assert book_target("ETH", "peace", -3.0)[0] == 0.0


def test_sol_book_is_declared_degraded():
    """SOL's bear ridge leg ships only with full feature parity; until then
    the book is hold-bull/flat-elsewhere AND says so in its version tag —
    its forward IC must never be mistaken for the full book's."""
    assert book_target("SOL", "bull", None)[0] == 1.0
    assert book_target("SOL", "bear", 2.0)[0] == 0.0
    assert BOOKS_VERSION["SOL"] == "v1_degraded_no_bear_leg"


# ---------------------------------------------------------------- harness
@pytest.fixture()
def harness(tmp_path):
    return RegimeBookShadow(data_dir=str(tmp_path))


def test_ledger_row_shape_and_confidence_rule(harness):
    rec = harness.record_tick("ETH", _mk_closes("bull"), price=1900.0)
    assert rec is not None
    assert rec["strategy"] == "regimebook" and rec["direction"] == 1.0
    # scorer multiplies direction x confidence (P236): |target|
    assert rec["confidence"] == 1.0
    rec2 = harness.record_tick("ETH", _mk_closes("peace"), price=1900.0)
    assert rec2["direction"] == 0.0 and rec2["confidence"] == 0.0, (
        "a flat row must contribute zero to the IC, never a saturated claim"
    )


def test_ledger_appends_jsonl(harness, tmp_path):
    harness.record_tick("BTC", _mk_closes("bull"), price=60000.0)
    harness.record_tick("BTC", _mk_closes("bull"), price=60100.0)
    path = tmp_path / "strategy_shadow" / "regimebook_BTC.jsonl"
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2 and rows[1]["price"] == 60100.0


def test_funding_history_persists_and_dedups(tmp_path):
    from datetime import date, timedelta
    h1 = RegimeBookShadow(data_dir=str(tmp_path))
    d0 = date(2026, 6, 1)
    for i in range(FUND_Z_WINDOW + 5):
        h1.record_daily_funding("BTC", (d0 + timedelta(days=i)).isoformat(), 0.0001)
    h1.record_daily_funding("BTC", "2026-08-01", 0.001)   # spike, completed day
    # a fresh instance (restart) restores the history — P154: RAM-only
    # baselines are not controls
    h2 = RegimeBookShadow(data_dir=str(tmp_path))
    z = causal_funding_z(h2._funding_series("BTC"))
    assert z is not None and z > 2.0
    # re-recording the same day is a no-op
    h2.record_daily_funding("BTC", "2026-08-01", 0.5)
    assert h2._fund_hist["BTC"]["2026-08-01"] == pytest.approx(0.001)


def test_broken_write_is_failsoft(harness, monkeypatch):
    monkeypatch.setattr("builtins.open",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("disk")))
    rec = harness.record_tick("BTC", _mk_closes("bull"), price=60000.0)
    assert rec is None   # logged, swallowed — the tick must survive


# ---------------------------------------------------------- SOL parity leg
def _fake_sol_model(features):
    n = len(features)
    return {"asset": "SOL", "cell": "bear", "family": "ridge_defensive",
            "features": features, "mean": [0.0] * n, "scale": [1.0] * n,
            "coef": [-1.0] + [0.0] * (n - 1),   # pred = -x0
            "intercept": 0.0, "train_sigma": 1.0}


def _sol_harness(tmp_path, features=("f_a", "f_b")):
    mdir = tmp_path / "configs" / "regimebook"
    mdir.mkdir(parents=True)
    (mdir / "SOL_bear_ridge.json").write_text(
        json.dumps(_fake_sol_model(list(features))), encoding="utf-8")
    return RegimeBookShadow(data_dir=str(tmp_path / "data"),
                            repo_root=str(tmp_path))


def test_sol_bear_stays_degraded_without_features(tmp_path):
    h = _sol_harness(tmp_path)
    rec = h.record_tick("SOL", _mk_closes("bear"), price=80.0)
    assert rec["direction"] == 0.0
    assert rec["book_version"] == "v1_degraded_no_bear_leg"
    assert rec["leg"] == "flat_degraded_stale_features"


def test_sol_bear_refuses_on_partial_coverage(tmp_path):
    """Coverage is COUNTED — a missing feature must never become a silent
    zero in the dot product (P2 class)."""
    h = _sol_harness(tmp_path)
    h.observe_features("SOL", {"f_a": 2.0}, {})     # f_b missing
    rec = h.record_tick("SOL", _mk_closes("bear"), price=80.0)
    assert rec["direction"] == 0.0
    assert rec["leg"] == "flat_degraded_parity_gap"
    assert "f_b" in str(rec["coverage_note"])


def test_sol_bear_activates_at_full_coverage(tmp_path):
    h = _sol_harness(tmp_path)
    h.observe_features("SOL", {"f_a": 2.0, "f_b": 0.1}, {})
    rec = h.record_tick("SOL", _mk_closes("bear"), price=80.0)
    # pred = -2.0, z = -2.0 -> short-only clip -> -1.0
    assert rec["direction"] == -1.0 and rec["leg"] == "ridge_defensive"
    assert rec["book_version"] == "v2_full_bear"


def test_sol_bear_deadband_and_short_only_clip(tmp_path):
    h = _sol_harness(tmp_path)
    h.observe_features("SOL", {"f_a": 0.1, "f_b": 0.0}, {})   # z = -0.1 < deadband
    rec = h.record_tick("SOL", _mk_closes("bear"), price=80.0)
    assert rec["direction"] == 0.0 and rec["leg"] == "ridge_defensive_flat"
    h.observe_features("SOL", {"f_a": -3.0, "f_b": 0.0}, {})  # z = +3 -> long forecast
    rec = h.record_tick("SOL", _mk_closes("bear"), price=80.0)
    assert rec["direction"] == 0.0, "defensive leg must never go LONG in a bear"


def test_sol_bull_ignores_the_bear_model(tmp_path):
    h = _sol_harness(tmp_path)
    h.observe_features("SOL", {"f_a": 5.0, "f_b": 5.0}, {})
    rec = h.record_tick("SOL", _mk_closes("bull"), price=80.0)
    assert rec["direction"] == 1.0 and rec["leg"] == "hold"


def test_stale_stash_degrades(tmp_path, monkeypatch):
    from defense import regime_book_shadow as mod
    h = _sol_harness(tmp_path)
    h.observe_features("SOL", {"f_a": 2.0, "f_b": 0.1}, {})
    ts, vals = h._feature_stash["SOL"]
    h._feature_stash["SOL"] = (ts - mod.FEATURE_STASH_MAX_AGE_S - 10, vals)
    rec = h.record_tick("SOL", _mk_closes("bear"), price=80.0)
    assert rec["leg"] == "flat_degraded_stale_features", (
        "a stale feature snapshot must degrade, not trade on old data"
    )


# ---------------------------------------------------------- orchestrator
def test_tick_orchestrator_is_per_asset_failsoft(harness, monkeypatch):
    calls = {}

    def fake_fetch(asset):
        calls[asset] = True
        if asset == "ETH":
            return None                       # feed outage for one asset
        return _mk_closes("bull")

    monkeypatch.setattr(harness, "fetch_closes_4h", fake_fetch)
    monkeypatch.setattr(harness, "refresh_funding_daily", lambda a: None)
    summary = harness.tick(("BTC", "ETH", "SOL"))
    assert calls == {"BTC": True, "ETH": True, "SOL": True}
    assert any(s.startswith("ETH=SKIP") for s in summary), (
        "an outage must be a visible SKIP, not silence (P155)"
    )
    assert any(s.startswith("BTC=bull") for s in summary)


# ---------------------------------------------------------- wiring pins
def test_mainpy_wiring_exists():
    src = (REPO / "main.py").read_text(encoding="utf-8", errors="ignore")
    assert "RegimeBookShadow(data_dir=" in src, "init missing (P152 shape)"
    assert "._regime_book_shadow.tick((" in src, "loop-level tick call missing"
    assert "observe_features(" in src, "SOL parity stash missing"


def test_scorer_registers_regimebook_at_both_sites():
    src = (REPO / "analytics" / "shadow_ic" / "compute_shadow_ic.py"
           ).read_text(encoding="utf-8", errors="ignore")
    assert src.count('"regimebook"') + src.count("regimebook") >= 2, (
        "prefix must be registered at BOTH default sites (P192/P236) or the "
        "ledger accumulates forever and is never scored"
    )
