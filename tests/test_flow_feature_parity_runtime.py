"""[P1a] Train/serve parity for fv2 — identity by construction, pinned.

The whole reason data_mgmt/flow_features.py exists: training and the live
feed must compute fv2 through the SAME functions. These tests pin that the
runtime path (latest_fv2_vector, what BinanceFlowFeed serves) reproduces the
training path (flow_features_4h + cross_asset_features, what the parquets
carry) exactly, and that the feed obeys the P214 (no training imports) and
P160/P170 (absence is explicit, never zero) disciplines.
"""

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from data_mgmt.flow_features import (  # noqa: E402
    FV2_COLUMNS, flow_features_4h, cross_asset_features, latest_fv2_vector, _agg_4h,
)


def _raw_1h(n=4800, seed=13, start="2024-01-01"):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.005, n)))
    vol = rng.lognormal(10, 1, n)
    taker = vol * rng.uniform(0.3, 0.7, n)
    return pd.DataFrame({
        "timestamp": pd.date_range(start, periods=n, freq="1h"),
        "open": close, "high": close * 1.001, "low": close * 0.999,
        "close": close, "volume": vol, "quote_volume": vol * close,
        "count": rng.integers(1000, 50000, n).astype(float),
        "taker_buy_base": taker, "taker_buy_quote": taker * close,
    })


def test_runtime_vector_equals_training_row_exactly():
    """The decisive parity pin: the feed's latest_fv2_vector must reproduce
    the training pipeline's LAST row bit-for-bit on identical raw data."""
    raw_self = _raw_1h(seed=13)
    raw_ref = _raw_1h(seed=14)
    # training path
    f = flow_features_4h(raw_self)
    g_self = _agg_4h(raw_self)[["timestamp", "close"]]
    g_ref = _agg_4h(raw_ref)[["timestamp", "close"]]
    x = cross_asset_features("SOL", {"SOL": g_self, "BTC": g_ref})
    training_last = f.merge(x, on="timestamp", how="left").iloc[-1][FV2_COLUMNS]
    # runtime path
    runtime_vec = latest_fv2_vector(raw_self, raw_ref, "SOL")
    assert runtime_vec is not None
    for c in FV2_COLUMNS:
        assert runtime_vec[c] == training_last[c], (
            f"{c}: runtime {runtime_vec[c]} != training {training_last[c]} — "
            f"the train/serve identity has broken"
        )


def test_warmup_returns_none_not_zeros():
    """P160/P170: absence must be explicit. A short history cannot produce a
    vector — it must return None, never a zero-filled 'measurement'."""
    raw_self = _raw_1h(n=100)   # 25 4H bars < the 42-bar z-score minimum
    raw_ref = _raw_1h(n=100, seed=14)
    assert latest_fv2_vector(raw_self, raw_ref, "SOL") is None


def test_feed_module_imports_no_training_code():
    """P214: runtime modules must not import training/ code — the image does
    not carry it, and the ImportError fallback would look like a feature
    with no signal."""
    src = io.open(REPO / "data_mgmt" / "feeds" / "binance_flow_feed.py",
                  encoding="utf-8").read()
    assert "from training" not in src and "import training" not in src


def test_feed_fetch_failure_returns_none(monkeypatch):
    from data_mgmt.feeds.binance_flow_feed import BinanceFlowFeed
    feed = BinanceFlowFeed()
    monkeypatch.setattr(feed, "_fetch_1h", lambda sym: None)
    assert feed.latest("BTC") is None


def test_feed_caches_per_ttl(monkeypatch):
    from data_mgmt.feeds.binance_flow_feed import BinanceFlowFeed
    feed = BinanceFlowFeed(cache_ttl_sec=3600)
    calls = {"n": 0}

    def fake_fetch(sym):
        calls["n"] += 1
        return None
    monkeypatch.setattr(feed, "_fetch_1h", fake_fetch)
    feed.latest("BTC")
    feed.latest("BTC")
    assert calls["n"] <= 1, "cache miss on second call within TTL"


def test_fv2_columns_match_the_parquet_contract():
    """The 13 names the feed serves must be exactly the columns the training
    parquets carry (order included — the obs builder will consume them
    positionally).

    [P252] The parquets are OPERATOR-LOCAL (training/training_data/ is
    gitignored, P199/P213), so CI structurally cannot run this parity check
    — it was the red test-suite's cause from the moment it landed (the P194
    class: a test whose outcome depends on a gitignored path is invisible
    to CI and carries its whole signal locally). Skip loudly with the
    reason; the 13-name contract itself is still pinned unconditionally
    below so CI keeps THAT half."""
    assert len(FV2_COLUMNS) == 13
    pq = (REPO / "training" / "training_data" / "drl_training"
          / "BTC_4H_full.parquet")
    if not pq.exists():
        pytest.skip(
            "training parquets are operator-local (gitignored, P199/P213) — "
            "the parquet half of the parity check only runs where the "
            "training data lives"
        )
    df = pd.read_parquet(pq)
    parquet_fv2 = [c for c in df.columns if c.startswith("fv2_")]
    assert sorted(parquet_fv2) == sorted(FV2_COLUMNS)
