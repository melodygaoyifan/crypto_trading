"""
tests/test_confidence_gate_staging.py - P1-2C Confidence Gate Staging tests

Verifies:
1. HIGH_RISK defaults: min_confidence=0.65, min_aligned=5
2. ULTRA staged schedule: WEEK1=0.50, WEEK2_3=0.45, aligned=3
3. Gate behaviour: confidence + alignment interaction
4. Proof fields populated in ConfidenceGateResult
5. Config round-trip from JSON
6. Edge cases: no direction, no agents, all neutral
"""

import json
import pytest
from pathlib import Path

from defense.confidence_gate import (
    ConfidenceGateConfig,
    ConfidenceGate,
    ConfidenceGateResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gate(
    min_confidence_default: float = 0.65,
    min_aligned_agents: int = 5,
    staged_schedule: list = None,
    stage_selector: str = "",
) -> ConfidenceGate:
    """Create a ConfidenceGate with specified config."""
    config = ConfidenceGateConfig(
        min_confidence_default=min_confidence_default,
        min_aligned_agents=min_aligned_agents,
        staged_schedule=staged_schedule or [],
        stage_selector=stage_selector,
    )
    return ConfidenceGate(config=config)


def _make_signals(
    quant_dir: float = 0.0,
    drl_dir: float = 0.0,
    sentiment_dir: float = 0.0,
    short_bias_dir: float = 0.0,
    two_stage_dir: float = 0.0,
) -> dict:
    """Create agent_signals dict with specified directions."""
    return {
        "quant_direction": quant_dir,
        "quant_confidence": 0.7,
        "drl_direction": drl_dir,
        "sentiment_direction": sentiment_dir,
        "short_bias_direction": short_bias_dir,
        "short_bias_confidence": 0.6,
        "two_stage_direction": two_stage_dir,
        "two_stage_confidence": 0.7,
    }


ULTRA_SCHEDULE = [
    {"stage": "WEEK1",   "min_confidence": 0.50},
    {"stage": "WEEK2_3", "min_confidence": 0.45},
]


# ===========================================================================
# Test Group 1: HIGH_RISK defaults
# ===========================================================================

class TestHighRiskDefaults:
    """HIGH_RISK: no staged schedule -> min_confidence=0.65, min_aligned=5."""

    def test_default_min_confidence(self):
        gate = _make_gate()
        assert gate._min_confidence == 0.65

    def test_default_min_aligned(self):
        gate = _make_gate()
        assert gate.config.min_aligned_agents == 5

    def test_high_risk_blocks_low_confidence(self):
        """conf=0.50 < 0.65 default -> blocked."""
        gate = _make_gate()
        signals = _make_signals(quant_dir=1.0, drl_dir=1.0, sentiment_dir=1.0,
                                short_bias_dir=1.0, two_stage_dir=1.0)
        result = gate.check(intent_direction=1, overall_confidence=0.50,
                            agent_signals=signals)
        assert result.passed is False
        assert result.reason == "LOW_CONF"

    def test_high_risk_blocks_low_alignment(self):
        """conf=0.70 OK, but only 3 aligned < 5 -> blocked."""
        gate = _make_gate()
        signals = _make_signals(quant_dir=1.0, drl_dir=1.0, sentiment_dir=1.0)
        result = gate.check(intent_direction=1, overall_confidence=0.70,
                            agent_signals=signals)
        assert result.passed is False
        assert result.reason == "LOW_ALIGNMENT"

    def test_high_risk_passes_when_met(self):
        """conf=0.70, all 5 aligned -> passes."""
        gate = _make_gate()
        signals = _make_signals(quant_dir=1.0, drl_dir=1.0, sentiment_dir=1.0,
                                short_bias_dir=1.0, two_stage_dir=1.0)
        result = gate.check(intent_direction=1, overall_confidence=0.70,
                            agent_signals=signals)
        assert result.passed is True
        assert result.reason == ""


# ===========================================================================
# Test Group 2: ULTRA staged schedule resolution
# ===========================================================================

class TestUltraStagedSchedule:
    """Staged schedule resolves correct min_confidence per stage."""

    def test_week1_resolves_to_050(self):
        gate = _make_gate(staged_schedule=ULTRA_SCHEDULE, stage_selector="WEEK1")
        assert gate._min_confidence == 0.50

    def test_week2_3_resolves_to_045(self):
        gate = _make_gate(staged_schedule=ULTRA_SCHEDULE, stage_selector="WEEK2_3")
        assert gate._min_confidence == 0.45

    def test_unknown_stage_falls_back_to_default(self):
        gate = _make_gate(
            min_confidence_default=0.55,
            staged_schedule=ULTRA_SCHEDULE,
            stage_selector="UNKNOWN_STAGE",
        )
        assert gate._min_confidence == 0.55

    def test_empty_stage_selector_falls_back(self):
        gate = _make_gate(
            min_confidence_default=0.60,
            staged_schedule=ULTRA_SCHEDULE,
            stage_selector="",
        )
        assert gate._min_confidence == 0.60

    def test_ultra_min_aligned_is_3(self):
        gate = _make_gate(min_aligned_agents=3)
        assert gate.config.min_aligned_agents == 3


# ===========================================================================
# Test Group 3: Gate behaviour - confidence + alignment interaction
# ===========================================================================

class TestGateBehaviour:
    """Cross-checks for confidence and alignment thresholds."""

    def test_conf_048_aligned_3_week1_fail(self):
        """conf=0.48 < WEEK1(0.50) -> fail LOW_CONF."""
        gate = _make_gate(
            staged_schedule=ULTRA_SCHEDULE,
            stage_selector="WEEK1",
            min_aligned_agents=3,
        )
        signals = _make_signals(quant_dir=1.0, drl_dir=1.0, sentiment_dir=1.0)
        result = gate.check(intent_direction=1, overall_confidence=0.48,
                            agent_signals=signals)
        assert result.passed is False
        assert result.reason == "LOW_CONF"

    def test_conf_048_aligned_3_week2_3_pass(self):
        """conf=0.48 > WEEK2_3(0.45), aligned=3 ≥ 3 -> pass."""
        gate = _make_gate(
            staged_schedule=ULTRA_SCHEDULE,
            stage_selector="WEEK2_3",
            min_aligned_agents=3,
        )
        signals = _make_signals(quant_dir=1.0, drl_dir=1.0, sentiment_dir=1.0)
        result = gate.check(intent_direction=1, overall_confidence=0.48,
                            agent_signals=signals)
        assert result.passed is True

    def test_conf_060_aligned_2_fail_alignment(self):
        """conf=0.60 OK, but aligned=2 < 3 -> fail LOW_ALIGNMENT."""
        gate = _make_gate(
            staged_schedule=ULTRA_SCHEDULE,
            stage_selector="WEEK1",
            min_aligned_agents=3,
        )
        signals = _make_signals(quant_dir=1.0, drl_dir=1.0)
        result = gate.check(intent_direction=1, overall_confidence=0.60,
                            agent_signals=signals)
        assert result.passed is False
        assert result.reason == "LOW_ALIGNMENT"
        assert result.aligned_count == 2

    def test_all_opposing_zero_aligned(self):
        """All agents oppose -> aligned=0 -> fail LOW_ALIGNMENT."""
        gate = _make_gate(
            staged_schedule=ULTRA_SCHEDULE,
            stage_selector="WEEK1",
            min_aligned_agents=3,
        )
        signals = _make_signals(quant_dir=-1.0, drl_dir=-1.0, sentiment_dir=-1.0)
        result = gate.check(intent_direction=1, overall_confidence=0.60,
                            agent_signals=signals)
        assert result.passed is False
        assert result.aligned_count == 0

    def test_short_direction_alignment(self):
        """Short direction: agents with negative direction are aligned."""
        gate = _make_gate(min_aligned_agents=3)
        signals = _make_signals(quant_dir=-1.0, drl_dir=-1.0, sentiment_dir=-1.0,
                                short_bias_dir=-1.0, two_stage_dir=-1.0)
        result = gate.check(intent_direction=-1, overall_confidence=0.70,
                            agent_signals=signals)
        assert result.passed is True
        assert result.aligned_count == 5

    def test_confidence_checked_before_alignment(self):
        """LOW_CONF takes precedence over LOW_ALIGNMENT."""
        gate = _make_gate(min_aligned_agents=5)
        signals = _make_signals(quant_dir=1.0)  # only 1 aligned
        result = gate.check(intent_direction=1, overall_confidence=0.30,
                            agent_signals=signals)
        assert result.passed is False
        assert result.reason == "LOW_CONF"  # checked first


# ===========================================================================
# Test Group 4: Result fields populated
# ===========================================================================

class TestResultFields:
    """ConfidenceGateResult has all required proof fields."""

    def test_result_fields_on_pass(self):
        gate = _make_gate(
            staged_schedule=ULTRA_SCHEDULE,
            stage_selector="WEEK1",
            min_aligned_agents=3,
        )
        signals = _make_signals(quant_dir=1.0, drl_dir=1.0, sentiment_dir=1.0)
        result = gate.check(intent_direction=1, overall_confidence=0.60,
                            agent_signals=signals)
        assert result.passed is True
        assert result.min_confidence_used == 0.50
        assert result.min_aligned_used == 3
        assert result.overall_confidence == 0.60
        assert result.aligned_count == 3
        assert result.stage_selector == "WEEK1"
        assert "quant" in result.aligned_agents
        assert "drl" in result.aligned_agents
        assert "sentiment" in result.aligned_agents

    def test_result_fields_on_fail(self):
        gate = _make_gate(min_aligned_agents=5)
        signals = _make_signals(quant_dir=1.0, drl_dir=1.0)
        result = gate.check(intent_direction=1, overall_confidence=0.40,
                            agent_signals=signals)
        assert result.passed is False
        assert result.reason == "LOW_CONF"
        assert result.overall_confidence == 0.40
        assert result.min_confidence_used == 0.65
        assert result.aligned_count == 2


# ===========================================================================
# Test Group 5: Config round-trip from JSON
# ===========================================================================

class TestConfigRoundTrip:
    """Config loads correctly from dict and ultra JSON."""

    def test_from_dict(self):
        d = {
            "min_confidence_default": 0.50,
            "staged_schedule": [
                {"stage": "WEEK1", "min_confidence": 0.50},
                {"stage": "WEEK2_3", "min_confidence": 0.45},
            ],
            "stage_selector": "WEEK1",
            "min_aligned_agents": 3,
        }
        config = ConfidenceGateConfig.from_dict(d)
        assert config.min_confidence_default == 0.50
        assert config.min_aligned_agents == 3
        assert len(config.staged_schedule) == 2
        assert config.stage_selector == "WEEK1"

    def test_from_dict_defaults(self):
        config = ConfidenceGateConfig.from_dict({})
        assert config.min_confidence_default == 0.65
        assert config.min_aligned_agents == 5
        assert config.stage_selector == ""

    def test_ultra_json_has_confidence_gate(self):
        json_path = Path("configs/ultra_aggressive_5y.json")
        if not json_path.exists():
            pytest.skip("ultra_aggressive_5y.json not found")
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        cg = data.get("confidence_gate")
        assert cg is not None
        assert cg["min_confidence_default"] == 0.50
        assert cg["min_aligned_agents"] == 3
        assert cg["stage_selector"] == "WEEK1"
        schedule = {e["stage"]: e["min_confidence"]
                    for e in cg["staged_schedule"]}
        assert schedule["WEEK1"] == 0.50
        assert schedule["WEEK2_3"] == 0.45

    def test_cloud_production_has_no_confidence_gate(self):
        """cloud_production.json should NOT have confidence_gate."""
        json_path = Path("configs/cloud_production.json")
        if not json_path.exists():
            pytest.skip("cloud_production.json not found")
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        assert data.get("confidence_gate") is None


# ===========================================================================
# Test Group 6: Edge cases
# ===========================================================================

class TestEdgeCases:
    """Edge cases: no direction, all neutral, empty signals."""

    def test_zero_direction_passes(self):
        """No direction (0) -> gate passes (nothing to align on)."""
        gate = _make_gate(min_aligned_agents=3)
        signals = _make_signals()
        result = gate.check(intent_direction=0, overall_confidence=0.70,
                            agent_signals=signals)
        assert result.passed is True
        assert result.aligned_count == 0

    def test_empty_signals_fail_alignment(self):
        """No agent signals -> 0 aligned -> fail if min > 0."""
        gate = _make_gate(min_aligned_agents=3)
        result = gate.check(intent_direction=1, overall_confidence=0.70,
                            agent_signals={})
        assert result.passed is False
        assert result.reason == "LOW_ALIGNMENT"

    def test_all_neutral_fail_alignment(self):
        """All agents at 0 direction -> 0 aligned -> fail."""
        gate = _make_gate(min_aligned_agents=3)
        signals = _make_signals()  # all 0
        result = gate.check(intent_direction=1, overall_confidence=0.70,
                            agent_signals=signals)
        assert result.passed is False
        assert result.reason == "LOW_ALIGNMENT"

    def test_min_aligned_zero_always_passes_alignment(self):
        """min_aligned=0 -> alignment check always passes."""
        gate = _make_gate(min_aligned_agents=0)
        signals = _make_signals()  # all 0
        result = gate.check(intent_direction=1, overall_confidence=0.70,
                            agent_signals=signals)
        assert result.passed is True

    def test_mixed_directions_only_counts_aligned(self):
        """Some agree, some oppose -> only counts agreeing agents."""
        gate = _make_gate(min_aligned_agents=2)
        signals = _make_signals(quant_dir=1.0, drl_dir=-1.0, sentiment_dir=1.0)
        result = gate.check(intent_direction=1, overall_confidence=0.70,
                            agent_signals=signals)
        assert result.passed is True
        assert result.aligned_count == 2
        assert "quant" in result.aligned_agents
        assert "sentiment" in result.aligned_agents
        assert "drl" not in result.aligned_agents
