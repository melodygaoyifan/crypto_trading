from signals.authority_fusion import (
    AUTHORITY_MATRIX_OPPORTUNITY,
    AgentSignal,
    Authority,
    AuthorityFusionEngine,
    FusionContext,
    RegimePhase,
    SystemMode,
)


def test_lead_lag_is_not_direction_decider_in_opportunity():
    assert AUTHORITY_MATRIX_OPPORTUNITY["lead_lag"] == Authority.EXECUTE


def test_opportunity_fusion_keeps_regime_direction_when_lead_lag_opposes():
    engine = AuthorityFusionEngine()
    signals = {
        "regime": AgentSignal(direction=-0.161, confidence=0.99),
        "lead_lag": AgentSignal(direction=0.204, confidence=0.47),
        "risk": AgentSignal(veto_active=False),
    }
    context = FusionContext(
        mode=SystemMode.OPPORTUNITY,
        regime_phase=RegimePhase.IGNITION,
        lead_lag_confident=True,
        lead_lag_edge=20.4,
        regime="QUIET_ACCUMULATION",
    )

    result = engine.fuse(signals, context)

    assert result.direction == -0.161
    assert result.execution_mode == "AGGRESSIVE_TAKER"
    assert result.decider_agent == "regime"
