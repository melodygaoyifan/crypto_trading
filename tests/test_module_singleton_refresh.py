from agents.microstructure_agent import (
    MicrostructureConfig,
    get_microstructure_agent,
    reset_microstructure_agent,
)
from orchestration.sota_integration import (
    SOTAIntegrationConfig,
    get_sota_integration,
    reset_sota_integration,
)
from signals.stablecoin_monitor import (
    get_stablecoin_monitor,
    reset_stablecoin_monitor,
)


def test_sota_integration_refreshes_when_config_changes():
    reset_sota_integration()
    first = get_sota_integration(SOTAIntegrationConfig(enable_alerts=False))
    second = get_sota_integration(SOTAIntegrationConfig(enable_alerts=True))

    assert first is not second
    assert second.config.enable_alerts is True

    reset_sota_integration()


def test_microstructure_agent_refreshes_when_config_changes():
    reset_microstructure_agent()
    first = get_microstructure_agent(MicrostructureConfig(min_confidence=0.5))
    second = get_microstructure_agent(MicrostructureConfig(min_confidence=0.8))

    assert first is not second
    assert second.config.min_confidence == 0.8

    reset_microstructure_agent()


def test_stablecoin_monitor_refreshes_when_api_key_changes():
    reset_stablecoin_monitor()
    first = get_stablecoin_monitor(api_key="")
    second = get_stablecoin_monitor(api_key="fresh-key")

    assert first is not second
    assert second._api_key == "fresh-key"
    assert second._mock_mode is False

    reset_stablecoin_monitor()
