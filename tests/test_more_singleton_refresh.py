from agents.options_sentiment_agent import (
    get_options_sentiment_agent,
    reset_options_sentiment_agent,
)
from agents.volatility_alpha_agent import (
    VolAlphaConfig,
    get_volatility_alpha_agent,
    reset_volatility_alpha_agent,
)
from execution.execution_quality_logger import (
    ExecutionQualityLoggerConfig,
    get_execution_quality_logger,
    reset_execution_quality_logger,
)
from execution.impact_calibration import (
    ImpactCalibrationConfig,
    get_impact_calibration_table,
    reset_impact_calibration_table,
)
from risk.leverage_guard import (
    LeverageConfig,
    get_leverage_guard,
    reset_leverage_guard,
)
from signals.alpha_signal_integrator import (
    AlphaIntegratorConfig,
    get_alpha_integrator,
    reset_alpha_integrator,
)
from signals.flow_signal_integrator import (
    FlowIntegratorConfig,
    get_flow_integrator,
    reset_flow_integrator,
)
from signals.macro_crowd_adapter import (
    get_macro_crowd_adapter,
    reset_macro_crowd_adapter,
)


def test_leverage_guard_refreshes_when_config_changes():
    reset_leverage_guard()
    first = get_leverage_guard(LeverageConfig(max_leverage=3.0))
    second = get_leverage_guard(LeverageConfig(max_leverage=2.0))

    assert first is not second
    assert second.config.max_leverage == 2.0

    reset_leverage_guard()


def test_macro_crowd_adapter_refreshes_when_inputs_change():
    reset_macro_crowd_adapter()
    first_feed = object()
    second_feed = object()

    first = get_macro_crowd_adapter(fred_feed=first_feed)
    second = get_macro_crowd_adapter(fred_feed=second_feed)

    assert first is not second
    assert getattr(second, "_fred_feed", None) is second_feed

    reset_macro_crowd_adapter()


def test_flow_integrator_refreshes_when_config_changes():
    reset_flow_integrator()
    first = get_flow_integrator(FlowIntegratorConfig(mock_mode=False))
    second = get_flow_integrator(FlowIntegratorConfig(mock_mode=True))

    assert first is not second
    assert second.config.mock_mode is True

    reset_flow_integrator()


def test_alpha_integrator_refreshes_when_config_changes():
    reset_alpha_integrator()
    first = get_alpha_integrator(AlphaIntegratorConfig(onchain_sentiment_weight=0.20))
    second = get_alpha_integrator(AlphaIntegratorConfig(onchain_sentiment_weight=0.40))

    assert first is not second
    assert second.config.onchain_sentiment_weight == 0.40

    reset_alpha_integrator()


def test_execution_quality_logger_refreshes_when_config_changes(tmp_path):
    reset_execution_quality_logger()
    first = get_execution_quality_logger(
        ExecutionQualityLoggerConfig(log_dir=str(tmp_path / "logs_a"))
    )
    second = get_execution_quality_logger(
        ExecutionQualityLoggerConfig(log_dir=str(tmp_path / "logs_b"))
    )

    assert first is not second
    assert second.config.log_dir.endswith("logs_b")

    reset_execution_quality_logger()


def test_impact_calibration_table_refreshes_when_config_changes(tmp_path):
    reset_impact_calibration_table()
    first = get_impact_calibration_table(
        ImpactCalibrationConfig(calibration_file=str(tmp_path / "impact_a.json"))
    )
    second = get_impact_calibration_table(
        ImpactCalibrationConfig(calibration_file=str(tmp_path / "impact_b.json"))
    )

    assert first is not second
    assert second.config.calibration_file.endswith("impact_b.json")

    reset_impact_calibration_table()


def test_options_sentiment_agent_refreshes_when_api_key_changes():
    reset_options_sentiment_agent()
    first = get_options_sentiment_agent(api_key="key_a")
    second = get_options_sentiment_agent(api_key="key_b")

    assert first is not second
    assert getattr(second, "_api_key", "") == "key_b"

    reset_options_sentiment_agent()


def test_volatility_alpha_agent_refreshes_when_config_changes():
    reset_volatility_alpha_agent()
    first = get_volatility_alpha_agent(VolAlphaConfig(min_intensity_for_output=0.3))
    second = get_volatility_alpha_agent(VolAlphaConfig(min_intensity_for_output=0.5))

    assert first is not second
    assert second.config.min_intensity_for_output == 0.5

    reset_volatility_alpha_agent()
