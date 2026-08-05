import pytest
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

    # [P165] The point of this test is that lead_lag (EXECUTE) cannot flip the
    # direction regime chose — it sets execution style only. It is NOT that
    # fusion returns regime's float bit-for-bit.
    #
    # Two deliberate changes since it was written make exact equality wrong:
    # kraken_quant was promoted to DECIDE (2026-04-22), and P30 stopped
    # treating an abstaining DECIDE agent as disagreement. kraken_quant is
    # absent from `signals` here, so it abstains and pulls the result 0.0102%
    # toward zero — hence -0.160983... and decider "consensus(regime,
    # kraken_quant)". Direction and magnitude are otherwise regime's.
    assert result.direction < 0, (
        f"lead_lag (+0.204) must not flip the sign regime chose; "
        f"got {result.direction}"
    )
    assert result.direction == pytest.approx(-0.161, rel=1e-3)
    assert result.execution_mode == "AGGRESSIVE_TAKER"
    assert "regime" in result.decider_agent
