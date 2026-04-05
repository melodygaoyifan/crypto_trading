from agents.model_alpha_agent import (
    ModelType,
    get_model_alpha_agent,
    reset_model_alpha_agent,
)
from core.authority_chain import get_authority_chain, reset_authority_chain
from core.runtime_spine import SpineConfig, get_runtime_spine, reset_runtime_spine
from defense.production_reliability import get_risk_veto_classifier


class _RiskManagerAllow:
    def check_trade_allowed(self, **kwargs):
        return {"approved": True}


class _LeverageGuardAllow:
    def validate_order(self, **kwargs):
        return True, "ok"


class _DataGuardAllow:
    def can_execute(self, **kwargs):
        return True, "ok", {}


def test_authority_chain_refreshes_when_constructor_inputs_change():
    reset_authority_chain()
    first = get_authority_chain(
        strict_mode=False,
        auto_load_modules=False,
        trade_gate=object(),
        risk_manager=_RiskManagerAllow(),
        leverage_guard=_LeverageGuardAllow(),
    )
    second = get_authority_chain(
        strict_mode=True,
        auto_load_modules=False,
        data_guard=_DataGuardAllow(),
        trade_gate=object(),
        risk_manager=_RiskManagerAllow(),
        leverage_guard=_LeverageGuardAllow(),
    )

    assert first is not second
    assert second.strict_mode is True

    reset_authority_chain()


def test_runtime_spine_refreshes_when_config_changes():
    reset_runtime_spine()
    first = get_runtime_spine(SpineConfig(master_tick_seconds=14400))
    second = get_runtime_spine(SpineConfig(master_tick_seconds=7200))

    assert first is not second
    assert second.config.master_tick_seconds == 7200

    reset_runtime_spine()


def test_risk_veto_classifier_refreshes_when_weekend_config_changes():
    import defense.production_reliability as mod

    mod._risk_veto_classifier = None
    first = get_risk_veto_classifier({"weekend_soft_veto_cap": 1.00})
    second = get_risk_veto_classifier({"weekend_soft_veto_cap": 0.55})

    assert first is not second
    assert second._weekend_config["weekend_soft_veto_cap"] == 0.55

    mod._risk_veto_classifier = None


def test_model_alpha_agent_refreshes_when_constructor_inputs_change():
    reset_model_alpha_agent()
    first = get_model_alpha_agent(
        model_type=ModelType.AUTO,
        assets=["BTC", "ETH"],
        allow_mock_model=False,
    )
    second = get_model_alpha_agent(
        model_type=ModelType.SEQUENCE_ALPHA,
        assets=["BTC", "ETH", "SOL"],
        allow_mock_model=True,
    )

    assert first is not second
    assert second.model_type == ModelType.SEQUENCE_ALPHA
    assert second.assets == ["BTC", "ETH", "SOL"]
    assert second.allow_mock_model is True

    reset_model_alpha_agent()
