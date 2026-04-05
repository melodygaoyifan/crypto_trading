from signals.false_breakout_detector import (
    FalseBreakoutConfig,
    get_false_breakout_detector,
    reset_false_breakout_detector,
)
from signals.profit_max_adapter import (
    ProfitMaxConfig,
    get_profit_max_adapter,
    reset_profit_max_adapter,
)
from signals.regime_transition_risk import (
    RegimeTransitionConfig,
    get_regime_transition_risk,
    reset_regime_transition_risk,
)
from signals.signal_quality_scorer import (
    SignalQualityScorerConfig,
    get_signal_quality_scorer,
    reset_signal_quality_scorer,
)


def test_profit_max_from_dict_loads_extended_fields():
    cfg = ProfitMaxConfig.from_dict(
        {
            "profit_max": {
                "macro_shock_threshold": 0.55,
                "crowd_panic_urgency_mult": 0.72,
                "signal_quality_low_conviction_mult": 0.81,
            }
        }
    )

    assert cfg.macro_shock_threshold == 0.55
    assert cfg.crowd_panic_urgency_mult == 0.72
    assert cfg.signal_quality_low_conviction_mult == 0.81


def test_signal_quality_from_dict_loads_threshold_fields():
    cfg = SignalQualityScorerConfig.from_dict(
        {
            "signal_quality": {
                "conviction_score_for_min": 22.0,
                "urgency_score_threshold_high": 77.0,
                "default_score_on_error": 44.0,
            }
        }
    )

    assert cfg.conviction_score_for_min == 22.0
    assert cfg.urgency_score_threshold_high == 77.0
    assert cfg.default_score_on_error == 44.0


def test_false_breakout_from_dict_loads_safe_defaults():
    cfg = FalseBreakoutConfig.from_dict(
        {
            "false_breakout": {
                "default_atr": 0.02,
                "default_volume_ratio": 0.75,
            }
        }
    )

    assert cfg.default_atr == 0.02
    assert cfg.default_volume_ratio == 0.75


def test_regime_transition_from_dict_loads_known_regimes():
    cfg = RegimeTransitionConfig.from_dict(
        {
            "regime_transition": {
                "known_regimes": ["A", "B", "C"],
            }
        }
    )

    assert cfg.known_regimes == ["A", "B", "C"]


def test_profit_max_singleton_refreshes_when_config_changes():
    reset_profit_max_adapter()
    first = get_profit_max_adapter(ProfitMaxConfig(false_breakout_action="VETO"))
    second = get_profit_max_adapter(ProfitMaxConfig(false_breakout_action="SCALE"))

    assert first is not second
    assert second.config.false_breakout_action == "SCALE"

    reset_profit_max_adapter()


def test_signal_quality_singleton_refreshes_when_config_changes():
    reset_signal_quality_scorer()
    first = get_signal_quality_scorer(SignalQualityScorerConfig(min_score_for_add=60.0))
    second = get_signal_quality_scorer(SignalQualityScorerConfig(min_score_for_add=75.0))

    assert first is not second
    assert second.config.min_score_for_add == 75.0

    reset_signal_quality_scorer()


def test_false_breakout_singleton_refreshes_when_config_changes():
    reset_false_breakout_detector()
    first = get_false_breakout_detector(FalseBreakoutConfig(min_criteria_for_veto=2))
    second = get_false_breakout_detector(FalseBreakoutConfig(min_criteria_for_veto=3))

    assert first is not second
    assert second.config.min_criteria_for_veto == 3

    reset_false_breakout_detector()


def test_regime_transition_singleton_refreshes_when_config_changes():
    reset_regime_transition_risk()
    first = get_regime_transition_risk(RegimeTransitionConfig(decay_factor=0.95))
    second = get_regime_transition_risk(RegimeTransitionConfig(decay_factor=0.80))

    assert first is not second
    assert second.config.decay_factor == 0.80

    reset_regime_transition_risk()
