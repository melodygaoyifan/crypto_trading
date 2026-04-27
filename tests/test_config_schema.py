"""
test_config_schema.py — verify the schema catches the bug families it claims
==============================================================================

[P113 2026-04-27] Property-style tests for configs/config_schema.py.

If this file's tests pass, the schema:
  - Accepts the current production config without false positives
  - Rejects the canonical-vs-JSON drift family (P112 fix)
  - Rejects drawdown-ordering inversions
  - Rejects correlation-threshold-ordering inversions
  - Rejects tier caps that exceed global cap (P112 9c safety)
  - Rejects tranche_percentages that don't sum to 1.0
"""
from __future__ import annotations

import pytest


class TestSchemaAcceptsValidConfig:
    """Current production config should pass with zero ERRORs."""

    def test_canonical_alone_passes(self):
        from configs.config_schema import validate_config_consistency
        issues = validate_config_consistency()
        errors = [m for s, m in issues if s == "ERROR"]
        assert not errors, f"Canonical-alone should validate clean: {errors}"

    def test_canonical_plus_matching_json_passes(self):
        """If JSON override AGREES with canonical, no warnings."""
        from configs.config_schema import validate_config_consistency
        # Match current canonical (post-P112)
        json_match = {"risk": {"hard_drawdown_halt": 0.25}}
        issues = validate_config_consistency(json_overrides=json_match)
        drift_warnings = [m for s, m in issues if "CONFIG-DRIFT" in m]
        assert not drift_warnings, (
            f"Matching JSON should not produce drift warnings: {drift_warnings}"
        )


class TestSchemaCatchesDrift:
    """The bug family P112 fixed should be detected automatically."""

    def test_p112_fix_drift_detection(self):
        """Pre-P112 state: canonical=0.20, JSON=0.25. Schema must warn."""
        from configs.config_schema import validate_config_consistency
        issues = validate_config_consistency(
            canonical={"hard_drawdown_halt": 0.20},  # pre-P112 value
            json_overrides={"risk": {"hard_drawdown_halt": 0.25}},
        )
        drift = [m for s, m in issues if "CONFIG-DRIFT" in m and "hard_drawdown_halt" in m]
        assert drift, (
            "Schema should detect canonical 0.20 vs JSON 0.25 drift "
            "(the P112 bug). If this test fails, the schema regressed."
        )

    def test_sota_flags_drift(self):
        """sota_flags MAX_LEVERAGE != canonical → warning."""
        from configs.config_schema import validate_config_consistency
        issues = validate_config_consistency(
            canonical={"max_leverage": 3.0},
            sota_flags={"MAX_LEVERAGE": 5.0},  # drift
        )
        drift = [m for s, m in issues if "MAX_LEVERAGE" in m]
        assert drift, "Schema should detect sota_flags MAX_LEVERAGE drift"


class TestSchemaRejectsInversions:
    """Ordering invariants must hold or schema raises ValueError."""

    def test_drawdown_inversion_rejected(self):
        from configs.config_schema import DrawdownGradient
        with pytest.raises(ValueError, match="critical_drawdown"):
            DrawdownGradient(
                critical_drawdown=0.30,  # higher than halt — inverted
                hard_drawdown_halt=0.20,
            )

    def test_kill_below_halt_rejected(self):
        from configs.config_schema import DrawdownGradient
        with pytest.raises(ValueError, match="kill_switch"):
            DrawdownGradient(
                hard_drawdown_halt=0.40,
                kill_switch_drawdown=0.35,  # less than halt — inverted
            )

    def test_correlation_inversion_rejected(self):
        from configs.config_schema import CorrelationThresholds
        with pytest.raises(ValueError, match="strictly increasing"):
            CorrelationThresholds(
                correlation_warning=0.95,
                correlation_danger=0.92,  # less than warning — inverted
                correlation_crisis=0.98,
            )

    def test_daily_loss_inversion_rejected(self):
        from configs.config_schema import DrawdownGradient
        with pytest.raises(ValueError, match="daily_loss"):
            DrawdownGradient(
                daily_loss_halt=0.15,  # higher than kill — inverted
                daily_loss_kill=0.10,
            )


class TestLeverageHierarchy:
    """P112 9c: tier/regime caps must NEVER loosen global cap."""

    def test_tier_cap_above_global_rejected(self):
        from configs.config_schema import LeverageHierarchy
        with pytest.raises(ValueError, match="exceeds global"):
            LeverageHierarchy(
                global_max_leverage=3.0,
                tier_caps={"VOL": 5.0},  # above global — would loosen
            )

    def test_regime_cap_above_global_rejected(self):
        from configs.config_schema import LeverageHierarchy
        with pytest.raises(ValueError, match="exceeds global"):
            LeverageHierarchy(
                global_max_leverage=3.0,
                regime_caps={"STRONG_TREND": 4.0},
            )

    def test_tier_cap_below_global_accepted(self):
        from configs.config_schema import LeverageHierarchy
        # Tightening is fine — that's the design intent (min wins)
        h = LeverageHierarchy(
            global_max_leverage=3.0,
            tier_caps={"CORE": 2.0, "EVENT": 2.0},
        )
        assert h.tier_caps == {"CORE": 2.0, "EVENT": 2.0}


class TestTrancheUnity:
    """tranche_percentages must sum to 1.0 (operator drift guard)."""

    def test_sum_unity_accepted(self):
        from configs.config_schema import TrancheConfig
        c = TrancheConfig(tranche_percentages={
            "TRANCHE_1": 0.35, "TRANCHE_2": 0.30,
            "TRANCHE_3": 0.20, "TRANCHE_4": 0.15,
        })
        assert sum(c.tranche_percentages.values()) == pytest.approx(1.0)

    def test_sum_below_unity_rejected(self):
        from configs.config_schema import TrancheConfig
        with pytest.raises(ValueError, match="must be 1.0"):
            TrancheConfig(tranche_percentages={
                "TRANCHE_1": 0.35, "TRANCHE_2": 0.30,
            })  # sums to 0.65

    def test_sum_above_unity_rejected(self):
        from configs.config_schema import TrancheConfig
        with pytest.raises(ValueError, match="must be 1.0"):
            TrancheConfig(tranche_percentages={
                "TRANCHE_1": 0.50, "TRANCHE_2": 0.50, "TRANCHE_3": 0.20,
            })  # sums to 1.20

    def test_empty_tranche_accepted(self):
        """Empty config → use defaults elsewhere (don't validate)."""
        from configs.config_schema import TrancheConfig
        c = TrancheConfig()
        assert c.tranche_percentages == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
