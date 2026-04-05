from core.risk_governor import (
    GovernorConfig,
    get_risk_governor,
    reset_risk_governor,
)
from defense.governor_integration import (
    GovernorIntegrationConfig,
    get_governor_integration,
    reset_governor_integration,
)
from defense.strategy_existence_fuse import (
    ExistenceFuseConfig,
    get_existence_fuse,
    reset_existence_fuse,
)
from execution.loop_controller import (
    ExecutionLoopConfig,
    get_execution_loop,
    reset_execution_loop,
)
from execution.orderbook_analyzer import (
    get_orderbook_analyzer,
    reset_orderbook_analyzer,
)
from execution.rl_fallback import (
    FallbackConfig,
    get_fallback_manager,
    reset_fallback_manager,
)
from execution.rl_drift_fallback import (
    DriftDetectorConfig,
    get_drift_detector,
    get_fallback_executor,
    reset_drift_detection,
)
from execution.sota_scheduler import (
    SchedulerConfig,
    get_execution_scheduler,
    reset_execution_scheduler,
)
from risk.global_exposure_cap import (
    ExposureCapConfig,
    get_exposure_cap_manager,
    reset_exposure_cap_manager,
)
from risk.regime_transition_buffer import (
    RegimeTransitionConfig,
    get_regime_transition_buffer,
    reset_regime_transition_buffer,
)
from risk.short_position_controller import (
    ShortRiskConfig,
    get_short_controller,
    reset_short_controller,
)
from risk.strategy_allocator import (
    get_strategy_allocator,
    reset_strategy_allocator,
)
from risk.tranche_manager import get_tranche_manager, reset_tranche_manager
from risk.unified_position_sizer import (
    PositionSizingConfig,
    get_unified_position_sizer,
    reset_unified_position_sizer,
)
from risk.volatility_targeting import (
    AssetVolConfig,
    get_volatility_sizer,
    reset_volatility_sizer,
)
from agents.short_bias_agent import (
    ShortBiasConfig,
    get_short_bias_agent,
    reset_short_bias_agent,
)
from risk.correlation_position_sizer import (
    CorrelationConfig,
    get_correlation_sizer,
    reset_correlation_sizer,
)
from risk.opportunity_budget_governor import (
    OpportunityBudgetConfig,
    get_opportunity_budget_governor,
    reset_opportunity_budget_governor,
)
from risk.cascade_exhaustion_governor import (
    CascadeExhaustionConfig,
    get_cascade_exhaustion_governor,
    reset_cascade_exhaustion_governor,
)
from risk.volatility_alpha_risk import (
    VolAlphaRiskConfig,
    get_vol_alpha_risk_integrator,
    reset_vol_alpha_risk_integrator,
)
from agents.onchain_graph_alpha import (
    OnChainGraphConfig,
    get_onchain_alpha_engine,
    get_onchain_graph_agent,
    reset_onchain_alpha_engine,
)
from agents.onchain_solana_agent import (
    OnChainConfig,
    get_onchain_agent,
    reset_onchain_agent,
)
from agents.sol_exposure_manager import (
    SOLExposureConfig,
    get_sol_exposure_manager,
    reset_sol_exposure_manager,
)
from agents.drl_agent import get_drl_agent, reset_drl_agent
from risk.stop_loss_authority import (
    StopConfig,
    StopOrder,
    StopType,
    get_stop_authority_manager,
    reset_stop_authority_manager,
)
from risk.sota_risk_controller import (
    SOTARiskConfig,
    get_sota_risk_controller,
    reset_sota_risk_controller,
)


def test_risk_governor_refreshes_when_config_changes():
    reset_risk_governor()
    first = get_risk_governor(GovernorConfig(hard_drawdown_halt=0.20))
    second = get_risk_governor(GovernorConfig(hard_drawdown_halt=0.15))

    assert first is not second
    assert second.config.hard_drawdown_halt == 0.15

    reset_risk_governor()


def test_governor_integration_refreshes_when_config_changes():
    reset_governor_integration()
    first = get_governor_integration(
        GovernorIntegrationConfig(enable_runtime_state=False)
    )
    second = get_governor_integration(
        GovernorIntegrationConfig(enable_runtime_state=True)
    )

    assert first is not second
    assert second.config.enable_runtime_state is True

    reset_governor_integration()


def test_existence_fuse_refreshes_when_config_changes():
    reset_existence_fuse()
    first = get_existence_fuse(ExistenceFuseConfig(rolling_window_days=28))
    second = get_existence_fuse(ExistenceFuseConfig(rolling_window_days=14))

    assert first is not second
    assert second.config.rolling_window_days == 14

    reset_existence_fuse()


def test_exposure_cap_manager_refreshes_when_config_changes():
    reset_exposure_cap_manager()
    first = get_exposure_cap_manager(ExposureCapConfig(max_gross_normal=1.50))
    second = get_exposure_cap_manager(ExposureCapConfig(max_gross_normal=1.20))

    assert first is not second
    assert second.config.max_gross_normal == 1.20

    reset_exposure_cap_manager()


def test_regime_transition_buffer_refreshes_when_config_changes():
    reset_regime_transition_buffer()
    first = get_regime_transition_buffer(RegimeTransitionConfig(min_conditions_for_transition=2))
    second = get_regime_transition_buffer(RegimeTransitionConfig(min_conditions_for_transition=3))

    assert first is not second
    assert second.config.min_conditions_for_transition == 3

    reset_regime_transition_buffer()


def test_tranche_manager_refreshes_when_config_changes():
    reset_tranche_manager()
    first = get_tranche_manager({"t4_min_profit_bps": 50, "sizes_cumulative": {"TRANCHE_1": 0.35}})
    second = get_tranche_manager({"t4_min_profit_bps": 80, "sizes_cumulative": {"TRANCHE_1": 0.40}})

    assert first is not second
    assert second.TRANCHE_4_MIN_PROFIT_BPS == 80
    assert second.TRANCHE_1_SIZE == 0.40

    reset_tranche_manager()


def test_fallback_manager_refreshes_when_config_changes():
    reset_fallback_manager()
    first = get_fallback_manager(FallbackConfig(min_rl_confidence=0.60))
    second = get_fallback_manager(FallbackConfig(min_rl_confidence=0.75))

    assert first is not second
    assert second.config.min_rl_confidence == 0.75

    reset_fallback_manager()


def test_execution_scheduler_refreshes_when_config_changes():
    reset_execution_scheduler()
    first = get_execution_scheduler(SchedulerConfig(twap_default_duration_min=15))
    second = get_execution_scheduler(SchedulerConfig(twap_default_duration_min=8))

    assert first is not second
    assert second.config.twap_default_duration_min == 8

    reset_execution_scheduler()


def test_orderbook_analyzer_refreshes_when_config_changes():
    reset_orderbook_analyzer()
    first = get_orderbook_analyzer({"depth_levels": 5})
    second = get_orderbook_analyzer({"depth_levels": 12})

    assert first is not second
    assert second.config["depth_levels"] == 12

    reset_orderbook_analyzer()


def test_execution_loop_refreshes_when_config_changes():
    reset_execution_loop()
    first = get_execution_loop(ExecutionLoopConfig(tick_interval_ms=200))
    second = get_execution_loop(ExecutionLoopConfig(tick_interval_ms=500))

    assert first is not second
    assert second.config.tick_interval_ms == 500

    reset_execution_loop()


def test_short_controller_refreshes_when_config_changes():
    reset_short_controller()
    first = get_short_controller(ShortRiskConfig(max_single_position_pct=0.30))
    second = get_short_controller(ShortRiskConfig(max_single_position_pct=0.20))

    assert first is not second
    assert second.config.max_single_position_pct == 0.20

    reset_short_controller()


def test_strategy_allocator_refreshes_when_inputs_change():
    reset_strategy_allocator()
    first = get_strategy_allocator(temperature=0.5, max_drawdown=0.15)
    second = get_strategy_allocator(temperature=0.2, max_drawdown=0.10)

    assert first is not second
    assert second.temperature == 0.2
    assert second.max_drawdown == 0.10

    reset_strategy_allocator()


def test_unified_position_sizer_refreshes_when_inputs_change():
    reset_unified_position_sizer()
    first = get_unified_position_sizer(
        PositionSizingConfig(risk_per_trade=0.01),
        initial_capital=10_000,
    )
    second = get_unified_position_sizer(
        PositionSizingConfig(risk_per_trade=0.02),
        initial_capital=20_000,
    )

    assert first is not second
    assert second.config.risk_per_trade == 0.02
    assert second._initial_capital == 20_000

    reset_unified_position_sizer()


def test_drift_detector_refreshes_when_config_changes():
    reset_drift_detection()
    first = get_drift_detector(DriftDetectorConfig(psi_warning_threshold=0.10))
    second = get_drift_detector(DriftDetectorConfig(psi_warning_threshold=0.20))

    assert first is not second
    assert second.config.psi_warning_threshold == 0.20

    reset_drift_detection()


def test_fallback_executor_refreshes_with_new_drift_detector():
    reset_drift_detection()
    first_detector = get_drift_detector(DriftDetectorConfig(psi_warning_threshold=0.10))
    first_executor = get_fallback_executor()
    second_detector = get_drift_detector(DriftDetectorConfig(psi_warning_threshold=0.40))
    second_executor = get_fallback_executor()

    assert first_detector is not second_detector
    assert first_executor is not second_executor
    assert second_executor.drift_detector is second_detector

    reset_drift_detection()


def test_l2_analyzer_refreshes_when_symbols_change():
    from execution.level2_analyzer import get_l2_analyzer, reset_l2_analyzer

    reset_l2_analyzer()
    first = get_l2_analyzer(["BTC", "ETH"])
    second = get_l2_analyzer(["BTC", "ETH", "SOL"])

    assert first is not second
    assert second.symbols == ["BTC", "ETH", "SOL"]

    reset_l2_analyzer()


def test_volatility_sizer_refreshes_when_inputs_change():
    reset_volatility_sizer()
    first = get_volatility_sizer(
        target_daily_vol=0.02,
        asset_configs={"BTC/USD": AssetVolConfig(target_daily_vol=0.015)},
    )
    second = get_volatility_sizer(
        target_daily_vol=0.03,
        asset_configs={"BTC/USD": AssetVolConfig(target_daily_vol=0.020)},
    )

    assert first is not second
    assert second.portfolio_target_daily_vol == 0.03
    assert second._asset_configs["BTC/USD"].target_daily_vol == 0.020

    reset_volatility_sizer()


def test_onchain_agent_refreshes_when_config_changes():
    reset_onchain_agent()
    first = get_onchain_agent(OnChainConfig(update_interval_seconds=120))
    second = get_onchain_agent(OnChainConfig(update_interval_seconds=30))

    assert first is not second
    assert second.config.update_interval_seconds == 30

    reset_onchain_agent()


def test_onchain_graph_engine_and_agent_refresh_with_new_engine():
    reset_onchain_alpha_engine()
    first_engine = get_onchain_alpha_engine(OnChainGraphConfig(whale_threshold_usd=100_000))
    first_agent = get_onchain_graph_agent(first_engine)
    second_engine = get_onchain_alpha_engine(OnChainGraphConfig(whale_threshold_usd=500_000))
    second_agent = get_onchain_graph_agent(second_engine)

    assert first_engine is not second_engine
    assert first_agent is not second_agent
    assert second_agent.engine is second_engine

    reset_onchain_alpha_engine()


def test_sol_exposure_manager_refreshes_when_config_changes():
    reset_sol_exposure_manager()
    first = get_sol_exposure_manager(SOLExposureConfig(min_mode_hold_seconds=300.0))
    second = get_sol_exposure_manager(SOLExposureConfig(min_mode_hold_seconds=60.0))

    assert first is not second
    assert second.config.min_mode_hold_seconds == 60.0

    reset_sol_exposure_manager()


def test_drl_agent_refreshes_when_constructor_inputs_change():
    reset_drl_agent()
    first = get_drl_agent(mode="SHADOW", obs_dim=126, n_stack=1)
    second = get_drl_agent(mode="EXIT_ONLY", obs_dim=126, n_stack=4)

    assert first is not second
    assert second.mode.value == "EXIT_ONLY"
    assert second.n_stack == 4

    reset_drl_agent()


def test_stop_authority_refreshes_when_config_changes():
    reset_stop_authority_manager()
    first = get_stop_authority_manager(StopConfig(soft_stop_pct=0.02))
    second = get_stop_authority_manager(StopConfig(soft_stop_pct=0.03))

    assert first is not second
    assert second.config.soft_stop_pct == 0.03

    reset_stop_authority_manager()


def test_stop_order_timestamps_are_timezone_aware():
    order = StopOrder(
        asset="SOL",
        stop_type=StopType.SOFT,
        trigger_price=100.0,
        direction="long",
        size=1.0,
    )
    assert order.created_at.tzinfo is not None
    assert order.updated_at.tzinfo is not None


def test_sota_risk_controller_refreshes_when_config_changes():
    reset_sota_risk_controller()
    first = get_sota_risk_controller(SOTARiskConfig(max_daily_loss=0.08))
    second = get_sota_risk_controller(SOTARiskConfig(max_daily_loss=0.05))

    assert first is not second
    assert second.config.max_daily_loss == 0.05

    reset_sota_risk_controller()


def test_short_bias_agent_refreshes_when_config_changes():
    reset_short_bias_agent()
    first = get_short_bias_agent(ShortBiasConfig(min_confidence_threshold=0.40))
    second = get_short_bias_agent(ShortBiasConfig(min_confidence_threshold=0.55))

    assert first is not second
    assert second.config.min_confidence_threshold == 0.55

    reset_short_bias_agent()


def test_correlation_sizer_refreshes_when_config_changes():
    reset_correlation_sizer()
    first = get_correlation_sizer(CorrelationConfig(high_corr_threshold=0.75))
    second = get_correlation_sizer(CorrelationConfig(high_corr_threshold=0.90))

    assert first is not second
    assert second.config.high_corr_threshold == 0.90

    reset_correlation_sizer()


def test_opportunity_budget_governor_refreshes_when_config_changes():
    reset_opportunity_budget_governor()
    first = get_opportunity_budget_governor(OpportunityBudgetConfig(global_budget=1.0))
    second = get_opportunity_budget_governor(OpportunityBudgetConfig(global_budget=0.6))

    assert first is not second
    assert second.config.global_budget == 0.6

    reset_opportunity_budget_governor()


def test_cascade_exhaustion_governor_refreshes_when_inputs_change():
    reset_cascade_exhaustion_governor()
    first = get_cascade_exhaustion_governor(
        CascadeExhaustionConfig(cascade_detect_liq_threshold=10_000_000),
        permissions_override={"DETECTED": {"T1": True, "T2": True, "T3": False, "T4": False}},
    )
    second = get_cascade_exhaustion_governor(
        CascadeExhaustionConfig(cascade_detect_liq_threshold=25_000_000),
        permissions_override={"DETECTED": {"T1": True, "T2": True, "T3": True, "T4": False}},
    )

    assert first is not second
    assert second.config.cascade_detect_liq_threshold == 25_000_000
    assert second.permissions_override["DETECTED"]["T3"] is True

    reset_cascade_exhaustion_governor()


def test_vol_alpha_risk_integrator_refreshes_when_config_changes():
    reset_vol_alpha_risk_integrator()
    first = get_vol_alpha_risk_integrator(VolAlphaRiskConfig(vol_alpha_base_weight=1.0))
    second = get_vol_alpha_risk_integrator(VolAlphaRiskConfig(vol_alpha_base_weight=0.6))

    assert first is not second
    assert second.config.vol_alpha_base_weight == 0.6

    reset_vol_alpha_risk_integrator()
