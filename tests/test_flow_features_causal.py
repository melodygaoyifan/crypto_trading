"""[P200-FEATURES] Flow-v2 features must be causal BY CONSTRUCTION.

P164's rule: a feature transform is verified causal by perturbing the future
and asserting nothing earlier moves — never by inspection. That is the test
that would have caught the wavelet leak on day one; every new feature family
gets it from day one.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def bf():
    spec = importlib.util.spec_from_file_location(
        "build_flow_features_under_test",
        REPO / "training" / "scripts" / "build_flow_features.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _raw(n_hours=4000, seed=5):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.005, n_hours)))
    vol = rng.lognormal(10, 1, n_hours)
    taker = vol * rng.uniform(0.3, 0.7, n_hours)
    qv = vol * close
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n_hours, freq="1h"),
        "open": close, "high": close * 1.001, "low": close * 0.999,
        "close": close, "volume": vol, "quote_volume": qv,
        "count": rng.integers(1000, 50000, n_hours).astype(float),
        "taker_buy_base": taker, "taker_buy_quote": taker * close,
    })


def test_perturbing_the_future_does_not_move_the_past(bf):
    """The P164 construction test. Mutate the last 25% of the raw series
    violently; every feature value in the first 60% must be bit-identical."""
    raw = _raw()
    base = bf.flow_features_4h(raw)
    mutated = raw.copy()
    cut = int(len(mutated) * 0.75)
    mutated.loc[mutated.index[cut:], ["close", "volume", "quote_volume",
                                      "count", "taker_buy_base",
                                      "taker_buy_quote"]] *= 7.3
    after = bf.flow_features_4h(mutated)
    check_until = int(len(base) * 0.60)
    fcols = [c for c in base.columns if c.startswith("fv2_")]
    a = base[fcols].iloc[:check_until].to_numpy()
    b = after[fcols].iloc[:check_until].to_numpy()
    diff = np.nanmax(np.abs(a - b)) if a.size else 0.0
    assert diff == 0.0, (
        f"future perturbation moved past feature values (max delta {diff}) — "
        f"a non-causal statistic has entered flow_features_4h (P164 class)"
    )


def test_warmup_bars_are_nan_not_fabricated(bf):
    raw = _raw(n_hours=800)
    f = bf.flow_features_4h(raw)
    # first bars cannot have a 42-bar-minimum z-score
    assert f["fv2_taker_ratio_z"].iloc[:10].isna().all(), (
        "z-scores exist before the minimum window — early values are "
        "fabricated, not measured"
    )
    assert f["fv2_taker_ratio_z"].iloc[-100:].notna().any()


def test_seasonality_is_deterministic_and_bounded(bf):
    raw = _raw(n_hours=1200)
    f = bf.flow_features_4h(raw)
    for c in ("fv2_hour_sin", "fv2_hour_cos", "fv2_dow_sin", "fv2_dow_cos"):
        assert f[c].abs().max() <= 1.0 + 1e-9
        assert f[c].notna().all()


def test_cross_asset_uses_lagged_reference_return(bf):
    """The reference asset's return must be its PREVIOUS bar — the
    contemporaneous bar is not information, it is simultaneity."""
    ts = pd.date_range("2024-01-01", periods=100, freq="4h")
    a = pd.DataFrame({"timestamp": ts, "close": np.linspace(100, 110, 100)})
    r_close = np.full(100, 50.0)
    r_close[60] = 60.0  # reference spikes at bar 60
    r = pd.DataFrame({"timestamp": ts, "close": r_close})
    out = bf.cross_asset_features("SOL", {"SOL": a, "BTC": r})
    lag = out["fv2_ref_lag_ret_4h"].to_numpy()
    assert abs(lag[61] - 0.2) < 1e-9, "spike must appear at bar 61 (lagged)"
    assert abs(lag[60]) < 1e-9, (
        "reference spike leaked into its own bar — contemporaneous read"
    )


def test_z_window_matches_documented_30d():
    src = (REPO / "training" / "scripts" / "build_flow_features.py").read_text(
        encoding="utf-8")
    assert "Z_WINDOW = 180" in src and "Z_MIN = 42" in src
