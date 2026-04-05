from integration.integration_v36 import HMATSv36Engine, TradeIntentV36


def _make_engine():
    engine = HMATSv36Engine.__new__(HMATSv36Engine)
    engine._last_alpha_result = None
    return engine


def test_pre_alpha_hold_applies_to_quiet_weak_long_volume_breakout_without_position():
    engine = _make_engine()
    intent = TradeIntentV36(asset="BTC", system_mode="OPPORTUNITY")

    applied = engine._maybe_apply_pre_alpha_hold(
        asset="BTC",
        intent=intent,
        agent_signals={
            "primary_strategy": "volume_breakout",
            "quant_direction": 0.041,
        },
        market_data={
            "regime_state": "QUIET_ACCUMULATION",
            "current_exposure": 0.0,
            "pre_alpha_hold_volume_breakout_long_in_quiet": True,
            "pre_alpha_hold_volume_breakout_long_max_direction": 0.05,
        },
        position_state={"current_exposure": 0.0},
    )

    assert applied is True
    assert intent.direction == 0.0
    assert intent.target_exposure == 0.0
    assert intent.tranche_action == "HOLD"
    assert intent.alpha_gate_passed is True
    assert intent.veto_active is False
    assert intent.veto_reason.startswith("[PRE_ALPHA_HOLD]")
    assert engine._last_alpha_result.gate_decision == "PRE_ALPHA_HOLD"


def test_pre_alpha_hold_does_not_apply_to_short_signal():
    engine = _make_engine()
    intent = TradeIntentV36(asset="BTC", system_mode="OPPORTUNITY")

    applied = engine._maybe_apply_pre_alpha_hold(
        asset="BTC",
        intent=intent,
        agent_signals={
            "primary_strategy": "volume_breakout",
            "quant_direction": -0.041,
        },
        market_data={
            "regime_state": "QUIET_ACCUMULATION",
            "current_exposure": 0.0,
            "pre_alpha_hold_volume_breakout_long_in_quiet": True,
            "pre_alpha_hold_volume_breakout_long_max_direction": 0.05,
        },
        position_state={"current_exposure": 0.0},
    )

    assert applied is False
    assert intent.direction == 0.0
    assert intent.target_exposure == 0.0
    assert intent.tranche_action == ""


def test_pre_alpha_hold_does_not_apply_when_position_exists():
    engine = _make_engine()
    intent = TradeIntentV36(asset="BTC", system_mode="OPPORTUNITY")

    applied = engine._maybe_apply_pre_alpha_hold(
        asset="BTC",
        intent=intent,
        agent_signals={
            "primary_strategy": "volume_breakout",
            "quant_direction": 0.041,
        },
        market_data={
            "regime_state": "QUIET_ACCUMULATION",
            "current_exposure": 0.02,
            "pre_alpha_hold_volume_breakout_long_in_quiet": True,
            "pre_alpha_hold_volume_breakout_long_max_direction": 0.05,
        },
        position_state={"current_exposure": 0.02},
    )

    assert applied is False


def test_pre_alpha_hold_does_not_apply_to_non_volume_breakout():
    engine = _make_engine()
    intent = TradeIntentV36(asset="BTC", system_mode="OPPORTUNITY")

    applied = engine._maybe_apply_pre_alpha_hold(
        asset="BTC",
        intent=intent,
        agent_signals={
            "primary_strategy": "momentum",
            "quant_direction": 0.041,
        },
        market_data={
            "regime_state": "QUIET_ACCUMULATION",
            "current_exposure": 0.0,
            "pre_alpha_hold_volume_breakout_long_in_quiet": True,
            "pre_alpha_hold_volume_breakout_long_max_direction": 0.05,
        },
        position_state={"current_exposure": 0.0},
    )

    assert applied is False
