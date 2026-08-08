"""[P216] The 12 GMM inputs must mean the same thing in training and at serve.

Audit prompted by the operator's "are we sure the GMM features are correctly
built?" — the answer was: causally yes (no lookahead anywhere), but the
train/serve contract held by accident in two places and was broken in one:

  * cross_asset_correlation: training uses the constant 0.87; runtime reads
    raw.get("cross_asset_correlation", 0.87) — which matches ONLY because the
    live value is written into raw AFTER _predict_gmm_regime runs. Fragile
    ordering, pinned here (P173 is_4h_bar_close shape: load-bearing accident).
  * fear_index: BOTH sides are the 100-RSI(14) proxy, not the real F&G index
    the name suggests. Consistent, so safe; pinned so one side can't drift.
  * vol_percentile: WAS a real skew — training ranked against the expanding
    6-year history, runtime ranks within its fetched ~1024-bar frame. With
    volume's secular drift those distributions differ materially (one of the
    GMM's 10 effective inputs; same family as the P214 wavelet skew, and a
    contributor to the old GMM's shift-driven saturation, P215). Fixed:
    training now uses a trailing GMM_VOL_PCT_WINDOW=1024 window.
"""

import importlib.util
import io
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent


def _load_rebuild():
    saved = {k: sys.modules.pop(k) for k in list(sys.modules)
             if k == "scripts" or k.startswith("scripts.")}
    tdir = str(REPO / "training")
    sys.path.insert(0, tdir)
    try:
        spec = importlib.util.spec_from_file_location(
            "rebuild_for_parity_test",
            REPO / "training" / "scripts" / "rebuild_pipeline.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        for k in [k for k in list(sys.modules)
                  if k == "scripts" or k.startswith("scripts.")]:
            sys.modules.pop(k, None)
        sys.modules.update(saved)
        if tdir in sys.path:
            sys.path.remove(tdir)


@pytest.fixture(scope="module")
def rp():
    return _load_rebuild()


def _runtime_src():
    return io.open(REPO / "data_mgmt" / "market_data_pipeline.py",
                   encoding="utf-8").read()


# ---------------------------------------------------------------------------
# vol_percentile: trailing window, mirroring runtime
# ---------------------------------------------------------------------------

def test_vol_percentile_ignores_history_beyond_the_runtime_window(rp):
    """Volumes older than GMM_VOL_PCT_WINDOW must not influence the rank —
    that is what the runtime cannot see."""
    n = 3000
    rng = np.random.default_rng(2)
    closes = 100 + np.cumsum(rng.normal(0, 0.1, n))
    rets = np.zeros(n)
    vols_a = rng.lognormal(10, 0.5, n)
    vols_b = vols_a.copy()
    vols_b[:n - rp.GMM_VOL_PCT_WINDOW - 100] *= 1000.0  # ancient history explodes
    i = n - 1
    fa = rp.compute_gmm_features_for_bar(closes, vols_a, rets, i)
    fb = rp.compute_gmm_features_for_bar(closes, vols_b, rets, i)
    vp_idx = rp.GMM_FEATURE_COLS.index("vol_percentile")
    assert fa[vp_idx] == fb[vp_idx], (
        "vol_percentile still ranks against history older than the runtime "
        "window — the train/serve skew P216 fixed has returned"
    )


def test_vol_percentile_window_constant_matches_runtime_frame():
    """Runtime frames are 721 fetched bars bootstrapped toward 1024
    (target_feature_bars). The training window must stay in that range —
    if the runtime frame size ever changes, change BOTH."""
    rp = _load_rebuild()
    assert rp.GMM_VOL_PCT_WINDOW == 1024
    rt = _runtime_src()
    assert "target_feature_bars" in rt


# ---------------------------------------------------------------------------
# cross_asset_correlation: constant-by-ordering, pinned from both sides
# ---------------------------------------------------------------------------

def test_cross_corr_constants_agree():
    rp = _load_rebuild()
    src = io.open(REPO / "training" / "scripts" / "rebuild_pipeline.py",
                  encoding="utf-8").read()
    assert "cross_corr = 0.87" in src
    assert 'raw.get("cross_asset_correlation", 0.87)' in _runtime_src(), (
        "runtime default for cross_asset_correlation is no longer 0.87 — "
        "training's constant and the runtime default have drifted"
    )


def test_cross_corr_is_written_after_the_gmm_predicts():
    """The load-bearing accident: the live correlation lands in `raw` only
    AFTER _predict_gmm_regime has read its default. If the write ever moves
    before the predict, the GMM's serve-time input starts varying while its
    training input was constant — a silent distribution change on one of the
    12 inputs. This pin makes that move loud."""
    rt = _runtime_src()
    predict_pos = rt.find("gmm_result = self._predict_gmm_regime(")
    write_pos = rt.find('raw["cross_asset_correlation"] = ')
    assert predict_pos != -1 and write_pos != -1
    assert predict_pos < write_pos, (
        "cross_asset_correlation is now written BEFORE the GMM predicts — "
        "the GMM's serve-time input is no longer the 0.87 constant it was "
        "trained on. Either feed the live value in training too (and refit) "
        "or restore the ordering."
    )


# ---------------------------------------------------------------------------
# fear_index and spread: consistent proxies, pinned
# ---------------------------------------------------------------------------

def test_fear_index_is_the_same_rsi_proxy_on_both_sides():
    src = io.open(REPO / "training" / "scripts" / "rebuild_pipeline.py",
                  encoding="utf-8").read()
    assert "fear_idx = 100.0 - rsi" in src
    assert 'fear_idx = 100.0 - raw.get("rsi", 50.0)' in _runtime_src(), (
        "runtime fear_index no longer mirrors training's 100-RSI proxy"
    )


def test_spread_defaults_agree():
    rp = _load_rebuild()
    assert rp.SPREAD_DEFAULTS == {"BTC": 5.0, "ETH": 8.0, "SOL": 12.0}
    rt = _runtime_src()
    assert '{"BTC": 5.0, "ETH": 8.0, "SOL": 12.0}' in rt, (
        "runtime spread defaults have drifted from training's SPREAD_DEFAULTS"
    )


# ---------------------------------------------------------------------------
# No lookahead anywhere in the 12 (the P164 construction test)
# ---------------------------------------------------------------------------

def test_perturbing_the_future_does_not_move_any_gmm_feature(rp):
    n = 2500
    rng = np.random.default_rng(9)
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    vols = rng.lognormal(10, 0.5, n)
    rets = np.zeros(n)
    rets[1:] = np.diff(closes) / closes[:-1]
    i = 1500
    base = rp.compute_gmm_features_for_bar(closes, vols, rets, i)
    closes2, vols2, rets2 = closes.copy(), vols.copy(), rets.copy()
    closes2[i + 1:] *= 5.0
    vols2[i + 1:] *= 5.0
    rets2[i + 1:] = 0.5
    after = rp.compute_gmm_features_for_bar(closes2, vols2, rets2, i)
    assert np.array_equal(base, after, equal_nan=True), (
        "a GMM feature at bar i moved when only bars > i changed — lookahead"
    )
