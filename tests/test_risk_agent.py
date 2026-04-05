"""
Tests for agents/risk_agent.py v6.4 alignment.

Groups:
  1. RiskAssessment.to_v6_dict (4 tests) - veto, leverage_cap, budget, metadata
  2. RiskAssessment persistence (3 tests) - to_dict, from_dict, round-trip
  3. RiskAgentV34 generate_signal (3 tests) - wrapper, drawdown veto, nominal
  4. RiskAgentV34 persistence (2 tests) - to_dict, from_dict
"""
import pytest
from datetime import datetime

from agents.risk_agent import (
    RiskAssessment,
    RiskAgentV34,
    RiskConfig,
    get_risk_agent,
    reset_risk_agent,
)
from market.phase_detector import RegimePhaseResult


# ============================================================================
# HELPERS
# ============================================================================

def _nominal_assessment(**kw) -> RiskAssessment:
    """Assessment with safe defaults - no veto, no exit acceleration."""
    defaults = dict(
        risk_multiplier=1.0,
        max_gross_cap=1.50,
        drawdown=0.02,
    )
    defaults.update(kw)
    return RiskAssessment(**defaults)


# ============================================================================
# GROUP 1: RiskAssessment.to_v6_dict (4 tests)
# ============================================================================

class TestToV6Dict:
    def test_nominal_no_veto(self):
        a = _nominal_assessment()
        d = a.to_v6_dict()
        assert d["risk_veto"] is False
        assert d["leverage_cap"] <= 3.0
        assert d["risk_budget_remaining"] > 0.0
        assert d["data_quality"] == "ok"
        assert d["asof_ts"].endswith("Z")
        assert "risk_agent_v34" in d["provenance"]

    def test_veto_active(self):
        a = _nominal_assessment(veto_active=True, veto_reason="HALT drawdown")
        d = a.to_v6_dict()
        assert d["risk_veto"] is True
        assert "HALT drawdown" in d["reasons"]

    def test_leverage_cap_capped_at_3(self):
        a = _nominal_assessment(max_gross_cap=5.0)
        d = a.to_v6_dict()
        assert d["leverage_cap"] == 3.0

    def test_budget_zero_at_full_drawdown(self):
        a = _nominal_assessment(drawdown=1.0)
        d = a.to_v6_dict()
        assert d["risk_budget_remaining"] == 0.0


# ============================================================================
# GROUP 2: RiskAssessment persistence (3 tests)
# ============================================================================

class TestRiskAssessmentPersistence:
    def test_to_dict_keys(self):
        a = _nominal_assessment()
        d = a.to_dict()
        assert "risk_multiplier" in d
        assert "veto_active" in d

    def test_from_dict_restores(self):
        d = {
            "risk_multiplier": 0.5,
            "max_gross_cap": 1.2,
            "veto_active": True,
            "veto_reason": "test",
            "drawdown": 0.08,
            "correlation": 0.7,
        }
        a = RiskAssessment.from_dict(d)
        assert a.risk_multiplier == 0.5
        assert a.veto_active is True
        assert a.drawdown == 0.08

    def test_round_trip(self):
        orig = _nominal_assessment(
            risk_multiplier=0.7,
            max_gross_cap=1.3,
            veto_active=False,
            drawdown=0.04,
        )
        d = orig.to_dict()
        # from_dict reconstructs from the dict - not all fields round-trip
        # via to_dict (which is minimal), but key fields should match
        restored = RiskAssessment.from_dict(d)
        assert restored.risk_multiplier == orig.risk_multiplier
        assert restored.veto_active == orig.veto_active


# ============================================================================
# GROUP 3: RiskAgentV34 generate_signal (3 tests)
# ============================================================================

class TestGenerateSignal:
    def _make_agent(self) -> RiskAgentV34:
        return RiskAgentV34(config=RiskConfig())

    def _phase_result(self) -> RegimePhaseResult:
        return RegimePhaseResult()

    def test_nominal_returns_v6_dict(self):
        agent = self._make_agent()
        sig = agent.generate_signal(
            system_mode="NORMAL",
            current_drawdown=0.01,
            current_correlation=0.3,
            current_exposures={"BTC": 0.1, "ETH": 0.05, "SOL": 0.05},
            phase_result=self._phase_result(),
        )
        assert "risk_veto" in sig
        assert "leverage_cap" in sig
        assert sig["risk_veto"] is False

    def test_critical_drawdown_vetos(self):
        agent = self._make_agent()
        sig = agent.generate_signal(
            system_mode="NORMAL",
            current_drawdown=0.20,  # > critical_drawdown (0.15)
            current_correlation=0.3,
            current_exposures={"BTC": 0.1},
            phase_result=self._phase_result(),
        )
        assert sig["risk_veto"] is True
        assert len(sig["reasons"]) > 0

    def test_leverage_cap_never_exceeds_3(self):
        agent = self._make_agent()
        sig = agent.generate_signal(
            system_mode="OPPORTUNITY",
            current_drawdown=0.0,
            current_correlation=0.1,
            current_exposures={},
            phase_result=self._phase_result(),
        )
        assert sig["leverage_cap"] <= 3.0


# ============================================================================
# GROUP 4: RiskAgentV34 persistence (2 tests)
# ============================================================================

class TestAgentPersistence:
    def test_to_dict_has_keys(self):
        agent = RiskAgentV34()
        d = agent.to_dict()
        assert "sol_dominance_bar" in d
        assert "fast_corr_buffer" in d

    def test_from_dict_restores_state(self):
        agent = RiskAgentV34()
        state = {
            "sol_dominance_bar": 5,
            "sol_dominance_active": True,
            "fast_corr_buffer": [0.3, 0.4, 0.5],
        }
        agent.from_dict(state)
        assert agent.sol_dominance._current_bar == 5
        assert agent.sol_dominance._override.active is True
        assert list(agent.fast_correlation._correlation_buffer) == [0.3, 0.4, 0.5]


# ============================================================================
# GROUP 5: Singleton lifecycle (2 tests)
# ============================================================================

class TestSingleton:
    def test_singleton_same(self):
        reset_risk_agent()
        a1 = get_risk_agent()
        a2 = get_risk_agent()
        assert a1 is a2
        reset_risk_agent()

    def test_reset_creates_fresh(self):
        reset_risk_agent()
        a1 = get_risk_agent()
        reset_risk_agent()
        a2 = get_risk_agent()
        assert a1 is not a2
        reset_risk_agent()
