from defense.constitution import (
    get_constitution_guarantees,
    reset_constitution_guarantees,
)
from defense.p0_safety_integrator import get_p0_integrator, reset_p0_integrator
from defense.trade_gate import TradeGateConfig, get_trade_gate, reset_trade_gate


def test_trade_gate_refreshes_when_config_changes():
    reset_trade_gate()
    first = get_trade_gate(TradeGateConfig(max_data_age_seconds=60.0))
    second = get_trade_gate(TradeGateConfig(max_data_age_seconds=15.0))

    assert first is not second
    assert second.config.max_data_age_seconds == 15.0

    reset_trade_gate()


def test_constitution_refreshes_when_tranche_config_changes():
    reset_constitution_guarantees()
    first = get_constitution_guarantees({"t4_min_profit_bps": 50})
    second = get_constitution_guarantees({"t4_min_profit_bps": 80})

    assert first is not second
    assert second.tranche_scheduler.TRANCHE_4_MIN_PROFIT_BPS == 80

    reset_constitution_guarantees()


def test_p0_integrator_refreshes_when_config_changes():
    reset_p0_integrator()
    first = get_p0_integrator({"enabled": True, "risk": {"max_drawdown_pct": 0.10}})
    second = get_p0_integrator({"enabled": True, "risk": {"max_drawdown_pct": 0.25}})

    assert first is not second
    assert second.config["risk"]["max_drawdown_pct"] == 0.25

    reset_p0_integrator()
