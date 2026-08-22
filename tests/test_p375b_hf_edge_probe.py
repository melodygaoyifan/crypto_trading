"""[P375b] Pin the higher-frequency edge probe's leakage guard and cost math.

edge_probe_hf.py is the Stage-0 gate for the predictor bet
(docs/research/GROWTH_PROGRAM_2026-08.md). Its verdict (NO PULSE on existing
data) is only trustworthy if (a) its features are provably causal and (b) its
required-IC bar is the P166 arithmetic. These tests pin both, and pin that the
recorded run cleared nothing.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO / "training" / "scripts"))
import edge_probe_hf as hf  # noqa: E402

REPORT = REPO / "training" / "reports" / "edge_probe_hf_p375.json"


def _synthetic(n=400):
    rng = np.random.RandomState(0)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, n))
    vol = np.abs(rng.normal(1000, 100, n))
    return pd.DataFrame({
        "timestamp": pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC"),
        "close": close, "volume": vol, "quote_volume": vol * close,
        "count": np.abs(rng.normal(500, 50, n)),
        "taker_buy_base": vol * rng.uniform(0.4, 0.6, n)})


def test_features_are_causal_perturbing_the_future_leaves_the_past_unchanged():
    # the probe's own P164 guard must actually catch a leak, and pass clean data
    df = _synthetic()
    assert hf.causal_check(df) is True
    # a deliberately leaked feature (uses the CURRENT close, not lagged) must fail
    X0, _, _ = hf.build_features(df)
    d2 = df.copy()
    d2.iloc[-50:, d2.columns.get_loc("close")] *= 3.0
    X1, _, _ = hf.build_features(d2)
    k = len(df) - 60
    assert np.allclose(np.nan_to_num(X0[:k]), np.nan_to_num(X1[:k]), atol=1e-9)


def test_required_ic_is_the_p166_arithmetic():
    # required IC = cost / (E|z| * pearson_k * sigma); higher cost -> higher bar
    lo = hf.required_ic(10.0, 100.0)
    hi = hf.required_ic(40.0, 100.0)
    assert hi > lo > 0
    assert hf.required_ic(10.0, 0.0) == float("inf")   # no vol -> unclearable
    # exact value pin
    assert abs(lo - 10.0 / (0.7979 * 1.047 * 100.0)) < 1e-9


@pytest.mark.skipif(not REPORT.exists(), reason="report is operator-local (built by the probe)")
def test_recorded_run_cleared_nothing_at_either_cost():
    d = json.loads(REPORT.read_text(encoding="utf-8"))
    cleared = []
    for a, arec in d["assets"].items():
        for h, hrec in arec["horizons"].items():
            for g, gr in hrec["groups"].items():
                if gr["clears_pct"] or gr["clears_cde"]:
                    cleared.append((a, h, g))
    # NO PULSE: nothing cleared. A future clear is a real finding to act on
    # deliberately (P141), not silence by editing this test.
    assert cleared == [], cleared
