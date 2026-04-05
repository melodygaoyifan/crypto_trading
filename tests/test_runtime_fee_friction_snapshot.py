from types import SimpleNamespace

from main import HMATSProductionRunner


def test_runtime_fee_friction_snapshot_prefers_fee_context_over_stale_alpha_fee():
    runner = HMATSProductionRunner.__new__(HMATSProductionRunner)
    intent = SimpleNamespace(
        prefer_limit_order=False,
        execution_mode=SimpleNamespace(value="AGGRESSIVE"),
    )
    alpha_result = SimpleNamespace(
        friction_fee_bps=0.0,
        friction_slippage_bps=5.0,
        friction_latency_bps=2.0,
        friction_margin_bps=7.0,
    )
    market_data = {
        "fee_context": {
            "maker_fee_bps": 8.0,
            "taker_fee_bps": 12.0,
            "fee_source": "unit_test_fee_model",
            "fee_weight": 0.4,
            "monthly_volume_usd": 12_345.0,
            "in_free_tier": False,
        }
    }

    snapshot = runner._get_runtime_fee_friction_snapshot(
        intent=intent,
        market_data=market_data,
        alpha_result=alpha_result,
    )

    assert snapshot["friction_fee_bps"] == 12.0
    assert snapshot["friction_fee_bps_alpha_gate"] == 12.0
    assert snapshot["friction_margin_bps"] == 7.0
    assert snapshot["friction_total_bps"] == 26.0
    assert snapshot["friction_fee_source"] == "unit_test_fee_model"
