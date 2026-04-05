"""
tests/test_false_breakout_scale.py - P1-2B FalseBreakout SCALE tests

Verifies:
1. HIGH_RISK (default) behaviour: false breakout -> HARD VETO (unchanged)
2. ULTRA_AGGRESSIVE_5Y behaviour: false breakout -> SCALE(0.5)
3. Non-false-breakout signals pass through unchanged in both modes
4. Config round-trip from JSON
5. Proof log / breakdown populated correctly for SCALE mode
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from signals.profit_max_adapter import (
    ProfitMaxConfig,
    ProfitMaxResult,
    ProfitMaxAdapter,
    VetoSource,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides) -> ProfitMaxConfig:
    """Create a ProfitMaxConfig with reasonable defaults for testing."""
    defaults = dict(
        enabled=True,
        false_breakout_enabled=True,
        false_breakout_action="VETO",
        false_breakout_scale=0.5,
        false_breakout_urgency_penalty=0.15,
        # Disable everything else so tests are isolated
        regime_quality_enabled=False,
        phase_filter_enabled=False,
        loss_streak_enabled=False,
        funding_bias_enabled=False,
        signal_quality_enabled=False,
        transition_risk_enabled=False,
        anti_floor_drift_enabled=False,
        entry_type_inference_enabled=False,
        sentiment_enabled=False,
        macro_crowd_enabled=False,
    )
    defaults.update(overrides)
    return ProfitMaxConfig(**defaults)


def _make_fb_result(is_false_breakout: bool, confidence: float = 0.8,
                    reasons: list = None):
    """Create a mock FalseBreakoutResult."""
    mock = MagicMock()
    mock.is_false_breakout = is_false_breakout
    mock.confidence = confidence
    mock.reasons = reasons or ["VOLUME_WEAK", "PRICE_REENTRY"]
    mock.to_dict.return_value = {
        "is_false_breakout": is_false_breakout,
        "confidence": confidence,
        "reasons": mock.reasons,
    }
    return mock


# ===========================================================================
# Test Group 1: HIGH_RISK (default) - VETO behaviour preserved
# ===========================================================================

class TestHighRiskVetoBehaviour:
    """Default VETO mode must be unchanged from pre-P1-2B."""

    def test_false_breakout_produces_veto(self):
        """When action=VETO and false breakout detected -> result.veto=True."""
        config = _make_config(false_breakout_action="VETO")
        adapter = ProfitMaxAdapter(config=config)

        # Mock the false breakout detector
        fb_mock = _make_fb_result(is_false_breakout=True, confidence=0.85)
        adapter._false_breakout_detector = MagicMock()
        adapter._false_breakout_detector.detect.return_value = fb_mock

        result = adapter.evaluate(
            regime="TREND",
            direction=1,
            entry_type="breakout",
        )

        assert result.veto is True
        assert result.veto_source == VetoSource.FALSE_BREAKOUT
        assert "FALSE_BREAKOUT_VETO" in result.veto_reason
        assert result.size_multiplier == 1.0  # Not scaled - vetoed instead

    def test_veto_includes_confidence_and_reasons(self):
        """VETO reason string must include confidence and detector reasons."""
        config = _make_config(false_breakout_action="VETO")
        adapter = ProfitMaxAdapter(config=config)

        fb_mock = _make_fb_result(
            is_false_breakout=True, confidence=0.92,
            reasons=["ATR_REJECTION", "WICK_RATIO_HIGH"]
        )
        adapter._false_breakout_detector = MagicMock()
        adapter._false_breakout_detector.detect.return_value = fb_mock

        result = adapter.evaluate(
            regime="TREND",
            direction=1,
            entry_type="breakout",
        )

        assert "0.92" in result.veto_reason
        assert "ATR_REJECTION" in result.veto_reason

    def test_no_false_breakout_no_veto(self):
        """When detector says NOT false breakout -> no veto, size_mult=1.0."""
        config = _make_config(false_breakout_action="VETO")
        adapter = ProfitMaxAdapter(config=config)

        fb_mock = _make_fb_result(is_false_breakout=False, confidence=0.3)
        adapter._false_breakout_detector = MagicMock()
        adapter._false_breakout_detector.detect.return_value = fb_mock

        result = adapter.evaluate(
            regime="TREND",
            direction=1,
            entry_type="breakout",
        )

        assert result.veto is False
        assert result.size_multiplier == 1.0


# ===========================================================================
# Test Group 2: ULTRA_AGGRESSIVE_5Y - SCALE behaviour
# ===========================================================================

class TestUltraAggressiveScaleBehaviour:
    """SCALE mode: false breakout -> half size, reduced urgency, NO veto."""

    def test_false_breakout_scales_not_vetoes(self):
        """When action=SCALE and false breakout detected -> size*=0.5, no veto."""
        config = _make_config(
            false_breakout_action="SCALE",
            false_breakout_scale=0.5,
        )
        adapter = ProfitMaxAdapter(config=config)

        fb_mock = _make_fb_result(is_false_breakout=True, confidence=0.85)
        adapter._false_breakout_detector = MagicMock()
        adapter._false_breakout_detector.detect.return_value = fb_mock

        result = adapter.evaluate(
            regime="TREND",
            direction=1,
            entry_type="breakout",
        )

        assert result.veto is False
        assert result.size_multiplier == 0.5
        assert result.veto_source == VetoSource.NONE

    def test_scale_reduces_urgency(self):
        """SCALE mode reduces execution_urgency by urgency_penalty."""
        config = _make_config(
            false_breakout_action="SCALE",
            false_breakout_scale=0.5,
            false_breakout_urgency_penalty=0.15,
        )
        adapter = ProfitMaxAdapter(config=config)

        fb_mock = _make_fb_result(is_false_breakout=True)
        adapter._false_breakout_detector = MagicMock()
        adapter._false_breakout_detector.detect.return_value = fb_mock

        result = adapter.evaluate(
            regime="TREND",
            direction=1,
            entry_type="breakout",
        )

        assert result.execution_urgency == pytest.approx(1.0 - 0.15, abs=1e-6)

    def test_scale_urgency_floor_at_0_1(self):
        """Urgency never drops below 0.1 even with large penalty."""
        config = _make_config(
            false_breakout_action="SCALE",
            false_breakout_scale=0.5,
            false_breakout_urgency_penalty=0.99,
        )
        adapter = ProfitMaxAdapter(config=config)

        fb_mock = _make_fb_result(is_false_breakout=True)
        adapter._false_breakout_detector = MagicMock()
        adapter._false_breakout_detector.detect.return_value = fb_mock

        result = adapter.evaluate(
            regime="TREND",
            direction=1,
            entry_type="breakout",
        )

        assert result.execution_urgency >= 0.1

    def test_custom_scale_factor(self):
        """Custom scale factor (0.3) is respected."""
        config = _make_config(
            false_breakout_action="SCALE",
            false_breakout_scale=0.3,
        )
        adapter = ProfitMaxAdapter(config=config)

        fb_mock = _make_fb_result(is_false_breakout=True)
        adapter._false_breakout_detector = MagicMock()
        adapter._false_breakout_detector.detect.return_value = fb_mock

        result = adapter.evaluate(
            regime="TREND",
            direction=1,
            entry_type="breakout",
        )

        assert result.size_multiplier == pytest.approx(0.3, abs=1e-6)

    def test_no_false_breakout_no_scale(self):
        """SCALE mode: when NOT false breakout -> size_multiplier=1.0."""
        config = _make_config(
            false_breakout_action="SCALE",
            false_breakout_scale=0.5,
        )
        adapter = ProfitMaxAdapter(config=config)

        fb_mock = _make_fb_result(is_false_breakout=False, confidence=0.2)
        adapter._false_breakout_detector = MagicMock()
        adapter._false_breakout_detector.detect.return_value = fb_mock

        result = adapter.evaluate(
            regime="TREND",
            direction=1,
            entry_type="breakout",
        )

        assert result.veto is False
        assert result.size_multiplier == 1.0

    def test_scale_continues_through_soft_modulations(self):
        """SCALE does NOT early-return - subsequent steps still apply."""
        config = _make_config(
            false_breakout_action="SCALE",
            false_breakout_scale=0.5,
            # Enable regime quality to verify it runs after false breakout
            regime_quality_enabled=True,
        )
        adapter = ProfitMaxAdapter(config=config)

        fb_mock = _make_fb_result(is_false_breakout=True)
        adapter._false_breakout_detector = MagicMock()
        adapter._false_breakout_detector.detect.return_value = fb_mock

        result = adapter.evaluate(
            regime="VOLATILE_CHOP",
            direction=1,
            entry_type="breakout",
        )

        # size_multiplier should be 0.5 from false breakout
        assert result.size_multiplier == pytest.approx(0.5, abs=1e-6)
        # regime_quality should also have run - check breakdown
        assert "regime_quality" in result.breakdown


# ===========================================================================
# Test Group 3: Breakdown / proof log for SCALE mode
# ===========================================================================

class TestScaleBreakdown:
    """SCALE mode populates breakdown["false_breakout_action"] for proof log."""

    def test_breakdown_populated_on_scale(self):
        """Breakdown includes action=SCALE, scale factor, urgency penalty."""
        config = _make_config(
            false_breakout_action="SCALE",
            false_breakout_scale=0.5,
            false_breakout_urgency_penalty=0.15,
        )
        adapter = ProfitMaxAdapter(config=config)

        fb_mock = _make_fb_result(
            is_false_breakout=True, confidence=0.88,
            reasons=["VOLUME_WEAK"]
        )
        adapter._false_breakout_detector = MagicMock()
        adapter._false_breakout_detector.detect.return_value = fb_mock

        result = adapter.evaluate(
            regime="TREND",
            direction=1,
            entry_type="breakout",
        )

        fb_action = result.breakdown.get("false_breakout_action")
        assert fb_action is not None
        assert fb_action["action"] == "SCALE"
        assert fb_action["scale"] == 0.5
        assert fb_action["urgency_penalty"] == 0.15
        assert fb_action["confidence"] == 0.88
        assert "VOLUME_WEAK" in fb_action["reasons"]

    def test_no_breakdown_on_veto(self):
        """VETO mode does NOT populate false_breakout_action breakdown."""
        config = _make_config(false_breakout_action="VETO")
        adapter = ProfitMaxAdapter(config=config)

        fb_mock = _make_fb_result(is_false_breakout=True)
        adapter._false_breakout_detector = MagicMock()
        adapter._false_breakout_detector.detect.return_value = fb_mock

        result = adapter.evaluate(
            regime="TREND",
            direction=1,
            entry_type="breakout",
        )

        assert "false_breakout_action" not in result.breakdown


# ===========================================================================
# Test Group 4: Config round-trip
# ===========================================================================

class TestConfigRoundTrip:
    """Config loads correctly from dict (as ultra_aggressive_5y.json would)."""

    def test_from_dict_with_scale_config(self):
        """from_dict loads false_breakout_action/scale/urgency_penalty."""
        d = {
            "profit_max": {
                "false_breakout_action": "SCALE",
                "false_breakout_scale": 0.5,
                "false_breakout_urgency_penalty": 0.15,
            }
        }
        config = ProfitMaxConfig.from_dict(d)
        assert config.false_breakout_action == "SCALE"
        assert config.false_breakout_scale == 0.5
        assert config.false_breakout_urgency_penalty == 0.15

    def test_from_dict_defaults_to_veto(self):
        """from_dict without false_breakout_action defaults to VETO."""
        d = {"profit_max": {}}
        config = ProfitMaxConfig.from_dict(d)
        assert config.false_breakout_action == "VETO"
        assert config.false_breakout_scale == 0.5

    def test_ultra_aggressive_json_has_scale(self):
        """ultra_aggressive_5y.json contains profit_max.false_breakout_action=SCALE."""
        json_path = Path("configs/ultra_aggressive_5y.json")
        if not json_path.exists():
            pytest.skip("ultra_aggressive_5y.json not found")
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        pm = data.get("profit_max", {})
        assert pm.get("false_breakout_action") == "SCALE"
        assert pm.get("false_breakout_scale") == 0.5
        assert pm.get("false_breakout_urgency_penalty") == 0.15

    def test_default_config_is_veto(self):
        """Default ProfitMaxConfig (no overrides) uses VETO."""
        config = ProfitMaxConfig()
        assert config.false_breakout_action == "VETO"
        assert config.false_breakout_enabled is True


# ===========================================================================
# Test Group 5: Edge cases
# ===========================================================================

class TestEdgeCases:
    """Edge cases for false breakout scale behaviour."""

    def test_detector_disabled_no_scale_no_veto(self):
        """When false_breakout_enabled=False, no veto or scale regardless of mode."""
        config = _make_config(
            false_breakout_enabled=False,
            false_breakout_action="SCALE",
        )
        adapter = ProfitMaxAdapter(config=config)
        # Detector won't even be created if disabled at ProfitMaxAdapter level
        # but let's be safe and also not mock it
        adapter._false_breakout_detector = None

        result = adapter.evaluate(
            regime="TREND",
            direction=1,
            entry_type="breakout",
        )

        assert result.veto is False
        assert result.size_multiplier == 1.0

    def test_detector_exception_fails_safe(self):
        """If detector raises, fail-safe: no veto, no scale."""
        config = _make_config(false_breakout_action="SCALE")
        adapter = ProfitMaxAdapter(config=config)

        adapter._false_breakout_detector = MagicMock()
        adapter._false_breakout_detector.detect.side_effect = RuntimeError("boom")

        result = adapter.evaluate(
            regime="TREND",
            direction=1,
            entry_type="breakout",
        )

        assert result.veto is False
        assert result.size_multiplier == 1.0

    def test_momentum_entry_type_bypasses_detector(self):
        """Momentum entry type is not checked by false breakout detector."""
        config = _make_config(false_breakout_action="SCALE")
        adapter = ProfitMaxAdapter(config=config)

        # The detector should return non-false-breakout for momentum entries
        fb_mock = _make_fb_result(is_false_breakout=False)
        adapter._false_breakout_detector = MagicMock()
        adapter._false_breakout_detector.detect.return_value = fb_mock

        result = adapter.evaluate(
            regime="TREND",
            direction=1,
            entry_type="momentum",
        )

        assert result.veto is False
        assert result.size_multiplier == 1.0
