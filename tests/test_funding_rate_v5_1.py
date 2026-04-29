"""Phase 3 funding-rate strategies + funding shadow harness — unit tests.

Verifies Iron Laws 4 and 7:
  - Fail-closed on missing fields.
  - Funding harness does NOT mutate market_data.
  - Funding sleeve registered into Phase 7 allocator.
"""

import json
from pathlib import Path

import pytest

from defense.strategy_shadow_v5_1 import (
    MicrostructureShadowHarness,
    build_funding_shadow_harness,
)
from strategies.funding_rate_v5_1 import (
    FundingRateExtremeStrategy,
    FundingRateMeanReversionStrategy,
    FundingRatePostETFRegimeStrategy,
    build_phase3_funding_strategies,
)
from risk.sleeve_allocator_v5_1 import build_phase7_sleeve_allocator


# ---------- FundingRateExtremeStrategy ------------------------------------

def test_funding_extreme_neutral_when_field_missing():
    s = FundingRateExtremeStrategy()
    out = s.evaluate("BTC", {})
    assert out.direction == 0.0 and "missing" in out.reason


def test_funding_extreme_falls_back_to_hourly():
    """If funding_rate_8h is missing, use hourly × 8 as proxy."""
    s = FundingRateExtremeStrategy(min_sustained_hours=24.0)
    out = s.evaluate("BTC", {
        "funding_rate_hourly": 0.00005,  # 0.005%/h × 8 = 0.04%/8h (above 0.03% threshold)
        "funding_sustained_hours": 30.0,
    })
    assert out.direction == -1.0  # long-crowded → SHORT


def test_funding_extreme_below_threshold_neutral():
    s = FundingRateExtremeStrategy(entry_threshold=0.0003)
    out = s.evaluate("BTC", {
        "funding_rate_8h": 0.0001, "funding_sustained_hours": 30.0,
    })
    assert out.direction == 0.0 and "below_threshold" in out.reason


def test_funding_extreme_not_sustained_neutral():
    s = FundingRateExtremeStrategy(min_sustained_hours=24.0)
    out = s.evaluate("BTC", {
        "funding_rate_8h": 0.0005, "funding_sustained_hours": 8.0,
    })
    assert out.direction == 0.0 and "not_sustained" in out.reason


def test_funding_extreme_long_crowded_short_signal():
    s = FundingRateExtremeStrategy(entry_threshold=0.0003, extreme_threshold=0.0005,
                                    min_sustained_hours=24.0)
    out = s.evaluate("BTC", {
        "funding_rate_8h": 0.0006, "funding_sustained_hours": 30.0,
    })
    assert out.direction == -1.0
    assert out.confidence > 0.0
    assert out.diagnostics["side"] == "long_crowded"


def test_funding_extreme_short_crowded_long_signal():
    s = FundingRateExtremeStrategy(entry_threshold=0.0003, min_sustained_hours=24.0)
    out = s.evaluate("ETH", {
        "funding_rate_8h": -0.0006, "funding_sustained_hours": 30.0,
    })
    assert out.direction == 1.0
    assert out.diagnostics["side"] == "short_crowded"


def test_funding_extreme_handles_nan():
    s = FundingRateExtremeStrategy()
    out = s.evaluate("BTC", {
        "funding_rate_8h": float("nan"), "funding_sustained_hours": 30.0,
    })
    # Falls back to hourly which is also missing → NEUTRAL
    assert out.direction == 0.0


# ---------- FundingRateMeanReversionStrategy -------------------------------

def test_funding_mean_reversion_warmup():
    s = FundingRateMeanReversionStrategy(min_history=12)
    for _ in range(5):
        out = s.evaluate("BTC", {"funding_rate_8h": 0.0001})
    assert out.direction == 0.0 and "warmup" in out.reason


def test_funding_mean_reversion_extreme_z_fires():
    s = FundingRateMeanReversionStrategy(min_history=12, zentry=2.0)
    # Build benign history with low variance
    for _ in range(15):
        s.evaluate("BTC", {"funding_rate_8h": 0.0001})
    # Spike: 10× the typical value
    out = s.evaluate("BTC", {"funding_rate_8h": 0.001})
    assert out.direction == -1.0  # high z → fade long crowding
    assert "z=" in out.reason


def test_funding_mean_reversion_negative_extreme_long():
    s = FundingRateMeanReversionStrategy(min_history=12, zentry=2.0)
    for _ in range(15):
        s.evaluate("BTC", {"funding_rate_8h": 0.0001})
    out = s.evaluate("BTC", {"funding_rate_8h": -0.001})
    assert out.direction == 1.0  # negative z → fade short crowding


def test_funding_mean_reversion_zero_stdev_neutral():
    """All identical history → stdev=0 → NEUTRAL."""
    s = FundingRateMeanReversionStrategy(min_history=12)
    for _ in range(20):
        s.evaluate("BTC", {"funding_rate_8h": 0.0001})
    # Identical input again → z would divide by zero
    out = s.evaluate("BTC", {"funding_rate_8h": 0.0001})
    # zero_stdev OR within-band → both are NEUTRAL
    assert out.direction == 0.0


def test_funding_mean_reversion_per_asset_independent():
    s = FundingRateMeanReversionStrategy(min_history=5)
    for _ in range(10):
        s.evaluate("BTC", {"funding_rate_8h": 0.0001})
    # ETH starts fresh — must be in warmup
    out = s.evaluate("ETH", {"funding_rate_8h": 0.0005})
    assert out.direction == 0.0
    assert "warmup" in out.reason


# ---------- FundingRatePostETFRegimeStrategy ------------------------------

def test_funding_post_etf_warmup():
    s = FundingRatePostETFRegimeStrategy(min_regime_obs=30)
    for _ in range(10):
        out = s.evaluate("BTC", {"funding_rate_8h": 0.0001})
    assert out.direction == 0.0 and "warmup" in out.reason


def test_funding_post_etf_normalized_extreme_fires():
    s = FundingRatePostETFRegimeStrategy(min_regime_obs=30, fire_threshold=0.85)
    # Build regime with ceiling ~0.0004
    import random
    random.seed(42)
    for _ in range(40):
        s.evaluate("BTC", {"funding_rate_8h": random.uniform(-0.0004, 0.0004)})
    # Normalized 0.95 above ceiling → fires
    out = s.evaluate("BTC", {"funding_rate_8h": 0.0008})
    assert out.direction == -1.0
    assert out.confidence > 0.0
    assert "post_etf" in out.reason


def test_funding_post_etf_within_regime_neutral():
    s = FundingRatePostETFRegimeStrategy(min_regime_obs=30, fire_threshold=0.85)
    import random
    random.seed(42)
    for _ in range(40):
        s.evaluate("BTC", {"funding_rate_8h": random.uniform(-0.0004, 0.0004)})
    out = s.evaluate("BTC", {"funding_rate_8h": 0.0001})  # well within ceiling
    assert out.direction == 0.0


def test_funding_post_etf_zero_ceiling_neutral():
    s = FundingRatePostETFRegimeStrategy(min_regime_obs=5)
    for _ in range(10):
        s.evaluate("BTC", {"funding_rate_8h": 0.0})
    out = s.evaluate("BTC", {"funding_rate_8h": 0.0001})
    assert out.direction == 0.0


# ---------- Funding harness — Iron Law 7 + ledger separation --------------

def test_funding_harness_writes_segregated_ledger(tmp_path: Path):
    h = build_funding_shadow_harness(log_dir=tmp_path)
    md = {
        "funding_rate_8h": 0.0006, "funding_sustained_hours": 30.0,
        "funding_rate_hourly": 0.000075,
    }
    h.observe("BTC", md)

    funding_files = list(tmp_path.glob("funding_*.jsonl"))
    micro_files = list(tmp_path.glob("microstructure_*.jsonl"))
    cascade_files = list(tmp_path.glob("cascade_*.jsonl"))
    assert len(funding_files) == 1
    assert len(micro_files) == 0
    assert len(cascade_files) == 0

    lines = funding_files[0].read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 3   # 3 funding strategies
    names = {json.loads(l)["strategy"] for l in lines}
    assert "funding_extreme" in names


def test_funding_harness_does_not_mutate_market_data(tmp_path: Path):
    h = build_funding_shadow_harness(log_dir=tmp_path)
    md = {"funding_rate_8h": 0.0005, "funding_sustained_hours": 30.0}
    snapshot = dict(md)
    h.observe("BTC", md)
    assert dict(md) == snapshot


def test_phase3_builder_returns_three():
    strats = build_phase3_funding_strategies()
    assert len(strats) == 3
    names = {type(s).__name__ for s in strats}
    assert names == {
        "FundingRateExtremeStrategy",
        "FundingRateMeanReversionStrategy",
        "FundingRatePostETFRegimeStrategy",
    }


# ---------- Sleeve allocator — funding registered ------------------------

def test_phase7_allocator_includes_funding_sleeve():
    a = build_phase7_sleeve_allocator()
    assert "funding" in a._sleeves
    funding = a._sleeves["funding"]
    assert funding.estimated_vol_pct == 0.15
    assert funding.sharpe_estimate == 1.5
    assert funding.is_live is False


def test_phase7_allocator_total_sleeve_count_after_funding():
    a = build_phase7_sleeve_allocator()
    # 4 sleeves: directional_short + microstructure + cascade + funding
    assert len(a._sleeves) == 4


def test_phase7_allocator_all_weights_below_cap():
    """With funding added (low vol 0.15), Iron Law 6 cap test."""
    a = build_phase7_sleeve_allocator()
    weights = a.compute_allocations()
    for name, w in weights.items():
        assert w <= a.max_sleeve_weight + 1e-9, \
            f"sleeve {name} weight {w} exceeds cap {a.max_sleeve_weight}"
