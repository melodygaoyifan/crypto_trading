"""Phase 8 cascade strategies + cascade-prefix harness — unit tests.

Verifies Iron Laws 4 and 7:
  - Fail-closed on missing fields, sub-threshold inputs, capitulation cap.
  - Cascade harness uses 'cascade' ledger prefix (segregated from Phase 4).
  - Output schema stable: ShadowSignal NamedTuple matches harness expectations.
"""

import json
from pathlib import Path

import pytest

from defense.strategy_shadow_v5_1 import (
    MicrostructureShadowHarness,
    build_cascade_shadow_harness,
)
from strategies.liquidation_cascade_v5_1 import (
    CascadeAnticipationStrategy,
    StopHuntDefenseStrategy,
    build_phase8_cascade_strategies,
)


# ---------- CascadeAnticipationStrategy ------------------------------------

def test_cascade_neutral_when_fields_missing():
    s = CascadeAnticipationStrategy()
    out = s.evaluate("BTC", {})
    assert out.direction == 0.0
    assert "missing" in out.reason


def test_cascade_neutral_when_total_liq_too_small():
    s = CascadeAnticipationStrategy(min_total_usd=1.0e8)
    out = s.evaluate("BTC", {
        "liquidation_imbalance": 0.8, "total_liquidations_24h": 1.0e6,
        "price_change_4h_pct": -0.03,
    })
    assert out.direction == 0.0
    assert "total_liq_too_small" in out.reason


def test_cascade_neutral_long_liq_no_drop_yet():
    """Long-side cascade implied but price hasn't dropped → wait."""
    s = CascadeAnticipationStrategy()
    out = s.evaluate("BTC", {
        "liquidation_imbalance": 0.75, "total_liquidations_24h": 5.0e8,
        "price_change_4h_pct": +0.001,  # tiny up move
    })
    assert out.direction == 0.0
    assert "no_drop_yet" in out.reason


def test_cascade_neutral_capitulation():
    """Price already crashed -12% → cascade exhausted, NEUTRAL."""
    s = CascadeAnticipationStrategy(cap_floor_pct=0.10)
    out = s.evaluate("BTC", {
        "liquidation_imbalance": 0.75, "total_liquidations_24h": 5.0e8,
        "price_change_4h_pct": -0.12,
    })
    assert out.direction == 0.0
    assert "capitulation" in out.reason


def test_cascade_short_signal_in_long_liq_window():
    """Long-side cascade in progress, price -3%, not capitulated → SHORT."""
    s = CascadeAnticipationStrategy()
    out = s.evaluate("BTC", {
        "liquidation_imbalance": 0.75, "total_liquidations_24h": 5.0e8,
        "price_change_4h_pct": -0.03,
    })
    assert out.direction == -1.0
    assert 0.0 < out.confidence <= s.confidence_cap
    assert "long_liq" in out.reason


def test_cascade_long_signal_in_short_squeeze_window():
    """Short squeeze: imbalance -0.7, price +3%, not capitulated → LONG."""
    s = CascadeAnticipationStrategy()
    out = s.evaluate("ETH", {
        "liquidation_imbalance": -0.70, "total_liquidations_24h": 4.0e8,
        "price_change_4h_pct": +0.03,
    })
    assert out.direction == 1.0
    assert "short_liq" in out.reason


def test_cascade_imbalance_below_threshold():
    s = CascadeAnticipationStrategy(imbalance_threshold=0.6)
    out = s.evaluate("BTC", {
        "liquidation_imbalance": 0.4, "total_liquidations_24h": 5.0e8,
        "price_change_4h_pct": -0.03,
    })
    assert out.direction == 0.0
    assert "imbalance_below_threshold" in out.reason


def test_cascade_neutral_on_nan():
    s = CascadeAnticipationStrategy()
    out = s.evaluate("BTC", {
        "liquidation_imbalance": float("nan"), "total_liquidations_24h": 5.0e8,
        "price_change_4h_pct": -0.03,
    })
    assert out.direction == 0.0


def test_cascade_vpin_confirm_gate():
    """When require_vpin_confirm=True, NEUTRAL until vpin_anomaly==HIGH."""
    s = CascadeAnticipationStrategy(require_vpin_confirm=True)
    md_no_vpin = {
        "liquidation_imbalance": 0.75, "total_liquidations_24h": 5.0e8,
        "price_change_4h_pct": -0.03, "vpin_anomaly": "NORMAL",
    }
    out_no = s.evaluate("BTC", md_no_vpin)
    assert out_no.direction == 0.0
    assert "awaiting_vpin_confirm" in out_no.reason

    md_no_vpin["vpin_anomaly"] = "HIGH"
    out_yes = s.evaluate("BTC", md_no_vpin)
    assert out_yes.direction == -1.0


# ---------- StopHuntDefenseStrategy ----------------------------------------

def test_stophunt_neutral_when_spread_missing():
    s = StopHuntDefenseStrategy()
    out = s.evaluate("BTC", {})
    assert out.direction == 0.0
    assert "missing_spread" in out.reason


def test_stophunt_quiet_regime_baseline_advisory():
    """All risk components zero → quiet, baseline offset, low confidence."""
    s = StopHuntDefenseStrategy(base_offset_bps=25.0, scale_offset_bps=75.0)
    out = s.evaluate("BTC", {
        "liquidation_imbalance": 0.0, "spread_bps": 3.0,
        "price_change_1h_pct": 0.001, "vpin_anomaly": "NORMAL",
        "vpin_source": "computed",
    })
    assert out.direction == 0.0
    assert out.confidence == 0.10  # quiet baseline
    assert out.diagnostics["recommended_stop_offset_bps"] == 25.0  # base only
    assert out.diagnostics["composite_risk"] < 0.10


def test_stophunt_elevated_risk_recommends_wider_offset():
    """High imbalance + HIGH vpin + wide spread + extreme move → high composite."""
    s = StopHuntDefenseStrategy(
        base_offset_bps=25.0, scale_offset_bps=75.0,
        imbalance_threshold=0.50, min_spread_for_alarm=8.0,
        extreme_move_1h_pct=0.010,
    )
    out = s.evaluate("BTC", {
        "liquidation_imbalance": 0.85,
        "spread_bps": 20.0,
        "price_change_1h_pct": -0.025,
        "vpin_anomaly": "HIGH",
        "vpin_source": "computed",
    })
    assert out.direction == 0.0  # advisory, non-directional
    assert out.confidence > 0.5
    assert out.diagnostics["recommended_stop_offset_bps"] > 50.0
    assert out.diagnostics["advisory_only"] is True
    assert "elevated_stop_hunt_risk" in out.reason


def test_stophunt_synthetic_vpin_does_not_trigger_toxic():
    """vpin_anomaly=HIGH + vpin_source=synthetic → toxic_score must be 0."""
    s = StopHuntDefenseStrategy()
    out = s.evaluate("BTC", {
        "liquidation_imbalance": 0.0, "spread_bps": 3.0,
        "price_change_1h_pct": 0.0,
        "vpin_anomaly": "HIGH", "vpin_source": "synthetic",
    })
    assert out.diagnostics["toxic_score"] == 0.0


def test_stophunt_partial_risk_emits_advisory():
    """Wide spread alone → moderate composite, advisory emitted."""
    s = StopHuntDefenseStrategy(min_spread_for_alarm=8.0)
    out = s.evaluate("ETH", {
        "liquidation_imbalance": 0.0, "spread_bps": 16.0,
        "price_change_1h_pct": 0.001,
        "vpin_anomaly": "NORMAL", "vpin_source": "computed",
    })
    assert out.diagnostics["spread_score"] > 0
    # composite is spread/4 ≈ 0.25, above quiet threshold 0.10
    assert out.confidence > 0.10
    assert out.diagnostics["recommended_stop_offset_bps"] > 25.0


# ---------- Cascade harness — Iron Law 7 + ledger separation ---------------

def test_cascade_harness_writes_segregated_ledger(tmp_path: Path):
    h = build_cascade_shadow_harness(log_dir=tmp_path)
    md = {
        "liquidation_imbalance": 0.75, "total_liquidations_24h": 5.0e8,
        "price_change_4h_pct": -0.03, "price_change_1h_pct": -0.01,
        "spread_bps": 5.0, "vpin_anomaly": "NORMAL", "vpin_source": "computed",
        "current_price": 60000.0,
    }
    h.observe("BTC", md)

    cascade_files = list(tmp_path.glob("cascade_*.jsonl"))
    micro_files = list(tmp_path.glob("microstructure_*.jsonl"))
    assert len(cascade_files) == 1
    assert len(micro_files) == 0  # cascade harness MUST NOT write to micro ledger

    lines = cascade_files[0].read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2  # cascade_anticipation + stop_hunt_defense
    names = {json.loads(l)["strategy"] for l in lines}
    assert names == {"cascade_anticipation", "stop_hunt_defense"}


def test_cascade_harness_does_not_mutate_market_data(tmp_path: Path):
    h = build_cascade_shadow_harness(log_dir=tmp_path)
    md = {
        "liquidation_imbalance": 0.85, "total_liquidations_24h": 6.0e8,
        "price_change_4h_pct": -0.04, "price_change_1h_pct": -0.02,
        "spread_bps": 12.0, "vpin_anomaly": "HIGH", "vpin_source": "computed",
        "current_price": 60000.0,
    }
    snapshot = dict(md)

    h.observe("BTC", md)

    assert dict(md) == snapshot, "cascade harness mutated market_data"


def test_cascade_harness_handles_strategy_exception(tmp_path: Path):
    """Iron Law 4: a bad strategy must not break siblings or write."""
    class _BadStrategy:
        def evaluate(self, asset, md):
            raise RuntimeError("boom")
    good = StopHuntDefenseStrategy()
    h = MicrostructureShadowHarness(
        strategies=[_BadStrategy(), good],
        log_dir=tmp_path,
        log_prefix="cascade",
    )
    h.observe("BTC", {"spread_bps": 5.0})
    obs, errs = h.stats()
    assert obs == 1 and errs == 1


def test_phase8_builder_returns_two_strategies():
    strats = build_phase8_cascade_strategies()
    assert len(strats) == 2
    names = {type(s).__name__ for s in strats}
    assert names == {"CascadeAnticipationStrategy", "StopHuntDefenseStrategy"}
