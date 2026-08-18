"""Tests for the Exit DRL promotion gate + outcome ledger."""
import json
import time
from pathlib import Path

import pytest

from agents.exit_drl_outcome_ledger import (
    ExitDRLOutcomeLedger, get_outcome_ledger, reset_singleton,
)
from risk.exit_drl_promotion_gate import (
    EXIT_ALL_RATIO_MAX, HOLD_RATIO_MAX, HOLD_RATIO_MIN,
    ExitDRLPromotionGate,
)


# ────────────────────────────────────────────────────────────────
# OutcomeLedger
# ────────────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def _reset():
    reset_singleton()
    yield
    reset_singleton()


def test_ledger_open_predict_close_roundtrip(tmp_path):
    ledger = ExitDRLOutcomeLedger(ledger_path=str(tmp_path / "ledger.jsonl"))
    ledger.record_open("BTC", entry_price=50000.0, direction=+1)
    ledger.record_prediction("BTC", "HOLD", 0.7, unrealized_pnl=0.005, bars_held=1)
    ledger.record_prediction("BTC", "PARTIAL_EXIT", 0.6, unrealized_pnl=0.012, bars_held=4)
    ledger.record_prediction("BTC", "HOLD", 0.55, unrealized_pnl=0.008, bars_held=6)
    record = ledger.record_close(
        "BTC", exit_price=50400.0, exit_reason="phase_transition",
        realized_pnl_bps=80.0,
    )
    assert record is not None
    assert record["closed"] is True
    assert record["asset"] == "BTC"
    assert record["predicted_action_at_close"] == "HOLD"
    assert record["predicted_action_at_peak"] == "PARTIAL_EXIT"  # peak was at PnL=0.012
    assert record["n_predictions"] == 3
    assert record["realized_pnl_bps"] == 80.0
    # Ledger file written
    line = (tmp_path / "ledger.jsonl").read_text(encoding="utf-8").strip()
    assert json.loads(line)["asset"] == "BTC"


def test_ledger_close_without_open_returns_none(tmp_path):
    ledger = ExitDRLOutcomeLedger(ledger_path=str(tmp_path / "ledger.jsonl"))
    out = ledger.record_close("BTC", 100.0, "stop_loss", -50.0)
    assert out is None
    assert not (tmp_path / "ledger.jsonl").exists() or \
        (tmp_path / "ledger.jsonl").read_text(encoding="utf-8") == ""


def test_ledger_predict_without_open_is_noop(tmp_path):
    ledger = ExitDRLOutcomeLedger(ledger_path=str(tmp_path / "ledger.jsonl"))
    ledger.record_prediction("BTC", "HOLD", 0.5, 0.0, 1)
    assert ledger.open_assets() == []


def test_ledger_handles_double_open_gracefully(tmp_path):
    ledger = ExitDRLOutcomeLedger(ledger_path=str(tmp_path / "ledger.jsonl"))
    ledger.record_open("BTC", 100.0, +1)
    ledger.record_prediction("BTC", "HOLD", 0.5, 0.0, 1)
    # Second open without close → first is abandoned
    ledger.record_open("BTC", 110.0, -1)
    # Close should reflect the second trade (direction=-1, entry=110)
    record = ledger.record_close("BTC", 105.0, "stop", -50.0)
    assert record["entry_price"] == 110.0
    assert record["direction"] == -1


def test_ledger_singleton_returns_same_instance(tmp_path):
    a = get_outcome_ledger(str(tmp_path / "l.jsonl"))
    b = get_outcome_ledger()  # default path arg ignored after first call
    assert a is b


# ────────────────────────────────────────────────────────────────
# PromotionGate
# ────────────────────────────────────────────────────────────────
def _write_ledger(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _write_validator(path: Path, asset: str, sharpe_lift: float, mean_pnl_lift: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "asset": asset,
        "drl": {"sharpe": 0.5, "mean_pnl_bps": 100.0},
        "baseline": {"sharpe": 0.4, "mean_pnl_bps": 80.0},
        "lift_pct": {"sharpe": sharpe_lift, "mean_pnl_bps": mean_pnl_lift},
    }), encoding="utf-8")


def _write_shadow_log(path: Path, days_ago: float):
    """Write a shadow log line whose first ts is `days_ago` days back."""
    path.parent.mkdir(parents=True, exist_ok=True)
    first_ts = time.time() - days_ago * 86400.0
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"ts": first_ts, "asset": "BTC", "action": "HOLD"}) + "\n")


def test_gate_blocks_when_no_evidence(tmp_path):
    gate = ExitDRLPromotionGate(
        outcome_ledger_path=str(tmp_path / "ledger.jsonl"),
        validator_dir=str(tmp_path / "val"),
        shadow_log_path=str(tmp_path / "shadow.jsonl"),
    )
    result = gate.evaluate("BTC")
    assert result["would_promote"] is False
    blockers = result["blockers"]
    assert any("min_shadow_days" in b for b in blockers)
    assert any("min_exit_events" in b for b in blockers)
    assert any("validator_not_run" in b for b in blockers)


def test_gate_blocks_on_short_shadow(tmp_path):
    _write_shadow_log(tmp_path / "shadow.jsonl", days_ago=5.0)
    _write_validator(tmp_path / "val" / "BTC_validation.json", "BTC",
                     sharpe_lift=15.0, mean_pnl_lift=20.0)
    # 30 closed exit events
    _write_ledger(tmp_path / "ledger.jsonl", [
        {"asset": "BTC", "closed": True, "predicted_action_at_close": "HOLD"}
    ] * 30)

    gate = ExitDRLPromotionGate(
        outcome_ledger_path=str(tmp_path / "ledger.jsonl"),
        validator_dir=str(tmp_path / "val"),
        shadow_log_path=str(tmp_path / "shadow.jsonl"),
    )
    result = gate.evaluate("BTC")
    assert result["would_promote"] is False
    assert any("min_shadow_days" in b for b in result["blockers"])


def test_gate_blocks_on_low_lift(tmp_path):
    _write_shadow_log(tmp_path / "shadow.jsonl", days_ago=45.0)
    _write_ledger(tmp_path / "ledger.jsonl", [
        {"asset": "BTC", "closed": True, "predicted_action_at_close": "HOLD"}
    ] * 30)
    _write_validator(tmp_path / "val" / "BTC_validation.json", "BTC",
                     sharpe_lift=5.0, mean_pnl_lift=2.0)  # below 10%

    gate = ExitDRLPromotionGate(
        outcome_ledger_path=str(tmp_path / "ledger.jsonl"),
        validator_dir=str(tmp_path / "val"),
        shadow_log_path=str(tmp_path / "shadow.jsonl"),
    )
    result = gate.evaluate("BTC")
    assert result["would_promote"] is False
    assert any("sharpe_lift_unmet" in b for b in result["blockers"])


def test_gate_blocks_on_action_distribution_skew(tmp_path):
    _write_shadow_log(tmp_path / "shadow.jsonl", days_ago=45.0)
    _write_validator(tmp_path / "val" / "BTC_validation.json", "BTC",
                     sharpe_lift=20.0, mean_pnl_lift=15.0)
    # Pathological: 50% EXIT_ALL > 30% threshold
    records = (
        [{"asset": "BTC", "closed": True, "predicted_action_at_close": "HOLD"}] * 15
        + [{"asset": "BTC", "closed": True, "predicted_action_at_close": "EXIT_ALL"}] * 15
    )
    _write_ledger(tmp_path / "ledger.jsonl", records)

    gate = ExitDRLPromotionGate(
        outcome_ledger_path=str(tmp_path / "ledger.jsonl"),
        validator_dir=str(tmp_path / "val"),
        shadow_log_path=str(tmp_path / "shadow.jsonl"),
    )
    result = gate.evaluate("BTC")
    assert result["would_promote"] is False
    assert any("exit_all_ratio_high" in b for b in result["blockers"])


def test_gate_promotes_when_all_thresholds_met(tmp_path):
    _write_shadow_log(tmp_path / "shadow.jsonl", days_ago=45.0)
    _write_validator(tmp_path / "val" / "BTC_validation.json", "BTC",
                     sharpe_lift=25.0, mean_pnl_lift=15.0)
    # 30 events, healthy distribution: 70% HOLD, 5% EXIT_ALL
    records = (
        [{"asset": "BTC", "closed": True, "predicted_action_at_close": "HOLD"}] * 21
        + [{"asset": "BTC", "closed": True, "predicted_action_at_close": "PARTIAL_EXIT"}] * 6
        + [{"asset": "BTC", "closed": True, "predicted_action_at_close": "RELEASE_RUNNER"}] * 1
        + [{"asset": "BTC", "closed": True, "predicted_action_at_close": "EXIT_ALL"}] * 2
    )
    _write_ledger(tmp_path / "ledger.jsonl", records)

    gate = ExitDRLPromotionGate(
        outcome_ledger_path=str(tmp_path / "ledger.jsonl"),
        validator_dir=str(tmp_path / "val"),
        shadow_log_path=str(tmp_path / "shadow.jsonl"),
    )
    result = gate.evaluate("BTC")
    assert result["would_promote"] is True, f"Expected promotion. Blockers: {result['blockers']}"
    assert result["evidence"]["sharpe_lift_pct"] == 25.0
    assert result["evidence"]["n_exit_events"] == 30
    assert result["evidence"]["shadow_days"] >= 30.0


def test_gate_evaluate_all_returns_per_asset(tmp_path):
    gate = ExitDRLPromotionGate(
        outcome_ledger_path=str(tmp_path / "ledger.jsonl"),
        validator_dir=str(tmp_path / "val"),
        shadow_log_path=str(tmp_path / "shadow.jsonl"),
    )
    out = gate.evaluate_all(["BTC", "ETH"])
    assert set(out.keys()) == {"BTC", "ETH"}
    assert all(v["would_promote"] is False for v in out.values())
