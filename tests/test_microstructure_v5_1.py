"""Phase 4 microstructure strategies + shadow harness — unit tests.

Verifies Iron Laws 4 and 7:
  - Fail-closed on missing fields, NaN, partial data.
  - Shadow harness NEVER mutates input market_data dict (no fusion side-effect).
"""

import json
import math
import os
from pathlib import Path

import pytest

from defense.strategy_shadow_v5_1 import (
    MicrostructureShadowHarness,
    build_microstructure_shadow_harness,
)
from strategies.microstructure_v5_1 import (
    KyleLambdaStrategy,
    OrderFlowImbalanceStrategy,
    VPINSpikeStrategy,
    build_phase4_microstructure_strategies,
)


# ---------- Iron Law 4: fail-closed on bad inputs ----------------------------

def test_ofi_neutral_when_field_missing():
    s = OrderFlowImbalanceStrategy()
    out = s.evaluate("BTC", {})
    assert out.direction == 0.0 and out.confidence == 0.0
    assert "missing" in out.reason


def test_ofi_neutral_below_threshold():
    s = OrderFlowImbalanceStrategy()
    out = s.evaluate("BTC", {"ofi_zscore": 0.5, "order_book_imbalance": 0.2})
    assert out.direction == 0.0
    assert "below_threshold" in out.reason


def test_ofi_neutral_streak_too_short():
    s = OrderFlowImbalanceStrategy(persistence=2)
    md = {"ofi_zscore": 2.0, "order_book_imbalance": 0.3}
    out = s.evaluate("BTC", md)
    assert out.direction == 0.0
    assert "streak_too_short" in out.reason


def test_ofi_fires_after_persistence():
    s = OrderFlowImbalanceStrategy(persistence=2, entry_threshold=1.5, min_imbalance=0.1)
    md = {"ofi_zscore": 2.0, "order_book_imbalance": 0.3}
    s.evaluate("BTC", md)  # streak=1, NEUTRAL
    out = s.evaluate("BTC", md)  # streak=2, FIRES
    assert out.direction == 1.0
    assert 0.0 < out.confidence <= s.confidence_cap
    assert "ofi_sustained" in out.reason


def test_ofi_streak_resets_on_sign_flip():
    s = OrderFlowImbalanceStrategy(persistence=2)
    s.evaluate("BTC", {"ofi_zscore": 2.0, "order_book_imbalance": 0.3})
    s.evaluate("BTC", {"ofi_zscore": 2.0, "order_book_imbalance": 0.3})
    out = s.evaluate("BTC", {"ofi_zscore": -2.0, "order_book_imbalance": -0.3})
    # First negative tick after positive streak — streak now -1, below persistence
    assert out.direction == 0.0


def test_ofi_per_asset_streaks_independent():
    s = OrderFlowImbalanceStrategy(persistence=2)
    s.evaluate("BTC", {"ofi_zscore": 2.0, "order_book_imbalance": 0.3})
    s.evaluate("BTC", {"ofi_zscore": 2.0, "order_book_imbalance": 0.3})
    # ETH starts fresh — first tick should be NEUTRAL, not FIRE
    out = s.evaluate("ETH", {"ofi_zscore": 2.0, "order_book_imbalance": 0.3})
    assert out.direction == 0.0


def test_ofi_neutral_on_nan():
    s = OrderFlowImbalanceStrategy()
    out = s.evaluate("BTC", {"ofi_zscore": float("nan"), "order_book_imbalance": 0.3})
    assert out.direction == 0.0


# ---------- VPIN spike ------------------------------------------------------

def test_vpin_neutral_when_synthetic():
    s = VPINSpikeStrategy()
    md = {"vpin": 0.85, "vpin_source": "synthetic", "vpin_anomaly": "HIGH",
          "ofi_zscore": 2.0}
    out = s.evaluate("BTC", md)
    assert out.direction == 0.0
    assert "synthetic" in out.reason or "unreliable" in out.reason


def test_vpin_neutral_below_spike():
    s = VPINSpikeStrategy(spike_threshold=0.7)
    md = {"vpin": 0.5, "vpin_source": "computed", "vpin_anomaly": "NORMAL",
          "ofi_zscore": 2.0}
    out = s.evaluate("BTC", md)
    assert out.direction == 0.0


def test_vpin_contrarian_fades_ofi():
    """OFI positive (net buy) + VPIN spike → fade SHORT."""
    s = VPINSpikeStrategy(spike_threshold=0.70, min_ofi_z=1.0)
    md = {"vpin": 0.85, "vpin_source": "computed", "vpin_anomaly": "HIGH",
          "ofi_zscore": 2.5}
    out = s.evaluate("BTC", md)
    assert out.direction == -1.0  # contrarian to positive OFI
    assert 0.0 < out.confidence <= s.confidence_cap


def test_vpin_anomaly_high_alone_triggers():
    """Even if vpin number is just at threshold, anomaly=HIGH should fire."""
    s = VPINSpikeStrategy(spike_threshold=0.99, min_ofi_z=1.0)
    md = {"vpin": 0.50, "vpin_source": "computed", "vpin_anomaly": "HIGH",
          "ofi_zscore": -2.0}
    out = s.evaluate("BTC", md)
    assert out.direction == 1.0  # fade negative OFI


# ---------- Kyle's lambda ---------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_warmup_state(monkeypatch, tmp_path):
    """[P303] KyleLambdaStrategy now PERSISTS `_last_price` and its lambda
    history (they were RAM-only, which is why the strategy recorded 1
    directional row in 6,168 - every restart wiped them). These tests assume
    a cold start, and without isolation they would inherit whatever a
    previous run left in the repo's data/ dir - the P186 class, where the
    suite writes into the checkout and later runs read it back.
    """
    monkeypatch.setenv("HMATS_DATA_DIR", str(tmp_path))


def test_kyle_neutral_warmup():
    s = KyleLambdaStrategy(lookback=30)
    md = {"current_price": 60000, "order_book_imbalance": 0.2, "spread_bps": 10}
    out = s.evaluate("BTC", md)
    assert out.direction == 0.0
    assert "no_prev_price" in out.reason


def test_kyle_neutral_history_warmup():
    s = KyleLambdaStrategy(lookback=30)
    base = 60000
    for i in range(5):
        md = {"current_price": base * (1 + i*0.001),
              "order_book_imbalance": 0.2, "spread_bps": 10}
        out = s.evaluate("BTC", md)
    assert out.direction == 0.0
    assert "history_warmup" in out.reason or "no_prev_price" in out.reason


def test_kyle_fires_on_spike_and_fades():
    s = KyleLambdaStrategy(lookback=15, pct_threshold=0.85,
                           min_spread_bps=5.0, min_abs_return_4h=0.005)
    base = 60000.0
    # Build benign history: small moves, low lambda_proxy
    p = base
    for i in range(15):
        p = p * (1 + 0.0005)  # 0.05% per tick → tiny lambda_proxy
        md = {"current_price": p, "order_book_imbalance": 0.3, "spread_bps": 10}
        s.evaluate("BTC", md)

    # Now inject a spike: large positive return + tiny imbalance => big lambda_proxy
    p_spike = p * (1 + 0.02)  # +2%
    md = {"current_price": p_spike, "order_book_imbalance": 0.05, "spread_bps": 12}
    out = s.evaluate("BTC", md)
    assert out.direction == -1.0  # fade the up move
    assert out.confidence > 0.0


def test_kyle_neutral_when_spread_tight():
    s = KyleLambdaStrategy(lookback=15, min_spread_bps=8.0)
    base = 60000.0
    p = base
    for _ in range(15):
        p = p * 1.0005
        s.evaluate("BTC", {"current_price": p, "order_book_imbalance": 0.3, "spread_bps": 10})
    out = s.evaluate("BTC", {"current_price": p * 1.02, "order_book_imbalance": 0.05,
                              "spread_bps": 3.0})
    assert out.direction == 0.0


# ---------- Iron Law 7: shadow harness does NOT mutate market_data ----------

def test_shadow_harness_does_not_mutate_market_data(tmp_path: Path):
    h = MicrostructureShadowHarness(
        strategies=build_phase4_microstructure_strategies(),
        log_dir=tmp_path,
    )
    md = {
        "current_price": 60000.0,
        "ofi_zscore": 2.5,
        "order_book_imbalance": 0.4,
        "spread_bps": 10.0,
        "vpin": 0.85,
        "vpin_source": "computed",
        "vpin_anomaly": "HIGH",
    }
    md_snapshot_keys = set(md.keys())
    md_snapshot_vals = dict(md)

    h.observe("BTC", md)

    # Iron Law 7: no fusion side-effect — market_data unchanged
    assert set(md.keys()) == md_snapshot_keys
    for k, v in md_snapshot_vals.items():
        assert md[k] == v, f"market_data[{k}] mutated by shadow harness"


def test_shadow_harness_writes_jsonl(tmp_path: Path):
    h = MicrostructureShadowHarness(
        strategies=build_phase4_microstructure_strategies(),
        log_dir=tmp_path,
    )
    md = {
        "current_price": 60000.0, "ofi_zscore": 0.1, "order_book_imbalance": 0.05,
        "spread_bps": 5.0, "vpin": 0.30, "vpin_source": "computed",
        "vpin_anomaly": "NORMAL",
    }
    h.observe("BTC", md)
    h.observe("ETH", md)

    files = list(tmp_path.glob("microstructure_*.jsonl"))
    assert len(files) == 1
    lines = files[0].read_text(encoding="utf-8").strip().split("\n")
    # 3 strategies × 2 assets = 6 records
    assert len(lines) == 6
    rec = json.loads(lines[0])
    assert "ts" in rec and "asset" in rec and "strategy" in rec
    assert "direction" in rec and "confidence" in rec
    assert rec["asset"] in ("BTC", "ETH")
    assert rec["strategy"] in ("ofi", "vpin_spike", "kyle_lambda")


def test_shadow_harness_isolates_strategy_failure(tmp_path: Path):
    """Iron Law 4: a bad strategy must not break the others."""
    class _BadStrategy:
        def evaluate(self, asset, md):
            raise RuntimeError("boom")
    good = OrderFlowImbalanceStrategy()
    h = MicrostructureShadowHarness(strategies=[_BadStrategy(), good], log_dir=tmp_path)
    md = {"ofi_zscore": 0.1, "order_book_imbalance": 0.05}

    h.observe("BTC", md)  # should not raise

    # Good strategy still wrote — bad one was bypassed
    files = list(tmp_path.glob("microstructure_*.jsonl"))
    assert len(files) == 1
    lines = files[0].read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1  # only the good one
    rec = json.loads(lines[0])
    assert rec["strategy"] == "ofi"

    obs, errs = h.stats()
    assert obs == 1 and errs == 1


def test_shadow_harness_handles_empty_strategy_list(tmp_path: Path):
    h = MicrostructureShadowHarness(strategies=[], log_dir=tmp_path)
    h.observe("BTC", {"ofi_zscore": 1.0})  # no-op, no exception
    assert not list(tmp_path.glob("microstructure_*.jsonl"))


def test_builder_returns_three_strategies():
    strats = build_phase4_microstructure_strategies()
    assert len(strats) == 3
    names = {type(s).__name__ for s in strats}
    assert names == {"OrderFlowImbalanceStrategy", "VPINSpikeStrategy", "KyleLambdaStrategy"}
