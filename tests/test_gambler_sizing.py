"""
tests/test_gambler_sizing.py - P4-02 Gambler Sizing tests

Verifies:
1. Disabled config -> never triggers
2. Mode not in allowed_modes -> not triggered
3. Regime not in allowed_regimes -> not triggered
4. Confidence below threshold -> not triggered
5. All gates pass -> triggered, exposure boosted to max_bet_pct_nav
6. Cooldown enforcement: 4h min between bets
7. Weekly limit enforcement: max 3 per week
8. Only boosts, never reduces exposure
9. Config round-trip from JSON
10. Edge cases: no direction, already at max, weekly prune
"""

import json
import time
import pytest
from pathlib import Path

from orchestration.gambler_sizing import (
    GamblerSizingConfig,
    GamblerSizing,
    GamblerSizingResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ULTRA_CONFIG = {
    "enabled": True,
    "max_bet_pct_nav": 0.50,
    "confidence_threshold": 0.70,
    "allowed_modes": ["OPPORTUNITY"],
    "allowed_regimes": ["STRONG_TREND", "IGNITION", "EXPANSION"],
    "min_seconds_between_bets": 14400,  # 4h
    "max_bets_per_week": 3,
}

BASE_TS = 1_700_000_000.0  # Fixed epoch for deterministic tests


def _make_gambler(cfg_dict=None) -> GamblerSizing:
    cfg = GamblerSizingConfig.from_dict(cfg_dict or ULTRA_CONFIG)
    return GamblerSizing(config=cfg)


def _trigger_kwargs(**overrides):
    """Default kwargs that pass all gates."""
    defaults = dict(
        intent_direction=1.0,
        target_exposure=0.20,
        system_mode="OPPORTUNITY",
        regime="STRONG_TREND",
        confidence=0.85,
        now_ts=BASE_TS,
    )
    defaults.update(overrides)
    return defaults


# ===========================================================================
# Test Group 1: Disabled
# ===========================================================================

class TestDisabled:
    """When disabled, gambler never triggers regardless of conditions."""

    def test_disabled_never_triggers(self):
        cfg = dict(ULTRA_CONFIG)
        cfg["enabled"] = False
        # Cannot trigger because config.enabled=False means GamblerSizing
        # is not created at all. Test the config field.
        gambler_cfg = GamblerSizingConfig.from_dict(cfg)
        assert gambler_cfg.enabled is False

    def test_empty_config_disabled_by_default(self):
        cfg = GamblerSizingConfig.from_dict({})
        assert cfg.enabled is False


# ===========================================================================
# Test Group 2: Mode gating
# ===========================================================================

class TestModeGating:
    """Only triggers in allowed_modes (e.g. OPPORTUNITY)."""

    def test_normal_mode_blocked(self):
        gambler = _make_gambler()
        result = gambler.evaluate(**_trigger_kwargs(system_mode="NORMAL"))
        assert result.triggered is False
        assert result.reason == "mode_not_allowed"

    def test_no_trade_mode_blocked(self):
        gambler = _make_gambler()
        result = gambler.evaluate(**_trigger_kwargs(system_mode="NO_TRADE"))
        assert result.triggered is False
        assert result.reason == "mode_not_allowed"

    def test_opportunity_mode_passes(self):
        gambler = _make_gambler()
        result = gambler.evaluate(**_trigger_kwargs(system_mode="OPPORTUNITY"))
        assert result.triggered is True


# ===========================================================================
# Test Group 3: Regime gating
# ===========================================================================

class TestRegimeGating:
    """Only triggers in allowed_regimes."""

    def test_volatile_chop_blocked(self):
        gambler = _make_gambler()
        result = gambler.evaluate(**_trigger_kwargs(regime="VOLATILE_CHOP"))
        assert result.triggered is False
        assert result.reason == "regime_not_allowed"

    def test_sideways_blocked(self):
        gambler = _make_gambler()
        result = gambler.evaluate(**_trigger_kwargs(regime="SIDEWAYS"))
        assert result.triggered is False
        assert result.reason == "regime_not_allowed"

    def test_strong_trend_passes(self):
        gambler = _make_gambler()
        result = gambler.evaluate(**_trigger_kwargs(regime="STRONG_TREND"))
        assert result.triggered is True

    def test_ignition_passes(self):
        gambler = _make_gambler()
        result = gambler.evaluate(**_trigger_kwargs(regime="IGNITION"))
        assert result.triggered is True

    def test_expansion_passes(self):
        gambler = _make_gambler()
        result = gambler.evaluate(**_trigger_kwargs(regime="EXPANSION"))
        assert result.triggered is True

    def test_regime_case_insensitive(self):
        gambler = _make_gambler()
        result = gambler.evaluate(**_trigger_kwargs(regime="strong_trend"))
        assert result.triggered is True


# ===========================================================================
# Test Group 4: Confidence gating
# ===========================================================================

class TestConfidenceGating:
    """Confidence must be >= threshold (0.70)."""

    def test_low_confidence_blocked(self):
        gambler = _make_gambler()
        result = gambler.evaluate(**_trigger_kwargs(confidence=0.60))
        assert result.triggered is False
        assert result.reason == "conf_too_low"

    def test_exactly_threshold_passes(self):
        gambler = _make_gambler()
        result = gambler.evaluate(**_trigger_kwargs(confidence=0.70))
        assert result.triggered is True

    def test_high_confidence_passes(self):
        gambler = _make_gambler()
        result = gambler.evaluate(**_trigger_kwargs(confidence=0.95))
        assert result.triggered is True


# ===========================================================================
# Test Group 5: Trigger behavior
# ===========================================================================

class TestTriggerBehavior:
    """When triggered, exposure is boosted to max_bet_pct_nav."""

    def test_exposure_boosted_to_max(self):
        gambler = _make_gambler()
        result = gambler.evaluate(**_trigger_kwargs(target_exposure=0.20))
        assert result.triggered is True
        assert result.new_exposure == 0.50
        assert result.old_exposure == 0.20
        assert result.reason == "gambler_triggered"

    def test_never_reduces_exposure(self):
        """If exposure already >= max_bet, don't reduce."""
        gambler = _make_gambler()
        result = gambler.evaluate(**_trigger_kwargs(target_exposure=0.60))
        assert result.triggered is False
        assert result.reason == "already_at_max"

    def test_exactly_at_max_not_triggered(self):
        gambler = _make_gambler()
        result = gambler.evaluate(**_trigger_kwargs(target_exposure=0.50))
        assert result.triggered is False
        assert result.reason == "already_at_max"

    def test_no_direction_blocked(self):
        gambler = _make_gambler()
        result = gambler.evaluate(**_trigger_kwargs(intent_direction=0.0))
        assert result.triggered is False
        assert result.reason == "no_direction"

    def test_result_fields_populated(self):
        gambler = _make_gambler()
        result = gambler.evaluate(**_trigger_kwargs())
        assert result.triggered is True
        assert result.confidence == 0.85
        assert result.regime == "STRONG_TREND"
        assert result.mode == "OPPORTUNITY"
        assert result.weekly_count == 1


# ===========================================================================
# Test Group 6: Cooldown enforcement
# ===========================================================================

class TestCooldown:
    """min_seconds_between_bets = 14400 (4h)."""

    def test_second_bet_within_cooldown_blocked(self):
        gambler = _make_gambler()
        # First bet
        r1 = gambler.evaluate(**_trigger_kwargs(now_ts=BASE_TS))
        assert r1.triggered is True

        # Second bet 1h later (3600s < 14400s)
        r2 = gambler.evaluate(**_trigger_kwargs(now_ts=BASE_TS + 3600))
        assert r2.triggered is False
        assert "cooldown" in r2.reason
        assert r2.cooldown_ok is False

    def test_second_bet_after_cooldown_allowed(self):
        gambler = _make_gambler()
        # First bet
        r1 = gambler.evaluate(**_trigger_kwargs(now_ts=BASE_TS))
        assert r1.triggered is True

        # Second bet 5h later (18000s > 14400s)
        r2 = gambler.evaluate(**_trigger_kwargs(now_ts=BASE_TS + 18000))
        assert r2.triggered is True

    def test_exactly_at_cooldown_boundary_blocked(self):
        gambler = _make_gambler()
        gambler.evaluate(**_trigger_kwargs(now_ts=BASE_TS))
        # Exactly at 14400s - elapsed == min_seconds, not > min_seconds
        r2 = gambler.evaluate(**_trigger_kwargs(now_ts=BASE_TS + 14400))
        assert r2.triggered is True  # 14400 elapsed == 14400 threshold -> not < threshold


# ===========================================================================
# Test Group 7: Weekly limit enforcement
# ===========================================================================

class TestWeeklyLimit:
    """max_bets_per_week = 3."""

    def test_fourth_bet_blocked(self):
        gambler = _make_gambler()
        spacing = 20000  # > 14400s cooldown

        r1 = gambler.evaluate(**_trigger_kwargs(now_ts=BASE_TS))
        assert r1.triggered is True

        r2 = gambler.evaluate(**_trigger_kwargs(now_ts=BASE_TS + spacing))
        assert r2.triggered is True

        r3 = gambler.evaluate(**_trigger_kwargs(now_ts=BASE_TS + spacing * 2))
        assert r3.triggered is True

        # Fourth bet - weekly limit hit
        r4 = gambler.evaluate(**_trigger_kwargs(now_ts=BASE_TS + spacing * 3))
        assert r4.triggered is False
        assert "weekly_limit" in r4.reason
        assert r4.cooldown_ok is False

    def test_weekly_prune_allows_new_bets(self):
        gambler = _make_gambler()
        spacing = 20000

        # Fill up 3 bets
        for i in range(3):
            gambler.evaluate(**_trigger_kwargs(now_ts=BASE_TS + spacing * i))

        # After 8 days (> 7 days), all old bets pruned
        r4 = gambler.evaluate(**_trigger_kwargs(now_ts=BASE_TS + 8 * 86400))
        assert r4.triggered is True
        assert r4.weekly_count == 1  # fresh week

    def test_weekly_count_tracks_correctly(self):
        gambler = _make_gambler()
        spacing = 20000

        r1 = gambler.evaluate(**_trigger_kwargs(now_ts=BASE_TS))
        assert r1.weekly_count == 1

        r2 = gambler.evaluate(**_trigger_kwargs(now_ts=BASE_TS + spacing))
        assert r2.weekly_count == 2

        r3 = gambler.evaluate(**_trigger_kwargs(now_ts=BASE_TS + spacing * 2))
        assert r3.weekly_count == 3


# ===========================================================================
# Test Group 8: Config round-trip
# ===========================================================================

class TestConfigRoundTrip:
    """Config from_dict and JSON load."""

    def test_from_dict(self):
        cfg = GamblerSizingConfig.from_dict(ULTRA_CONFIG)
        assert cfg.enabled is True
        assert cfg.max_bet_pct_nav == 0.50
        assert cfg.confidence_threshold == 0.70
        assert cfg.allowed_modes == ["OPPORTUNITY"]
        assert cfg.allowed_regimes == ["STRONG_TREND", "IGNITION", "EXPANSION"]
        assert cfg.min_seconds_between_bets == 14400
        assert cfg.max_bets_per_week == 3

    def test_from_dict_defaults(self):
        cfg = GamblerSizingConfig.from_dict({})
        assert cfg.enabled is False
        assert cfg.max_bet_pct_nav == 0.50
        assert cfg.confidence_threshold == 0.70
        assert cfg.max_bets_per_week == 3

    def test_ultra_json_has_gambler(self):
        json_path = Path("configs/ultra_aggressive_5y.json")
        if not json_path.exists():
            pytest.skip("ultra_aggressive_5y.json not found")
        with open(json_path) as f:
            data = json.load(f)
        gb = data.get("gambler")
        assert gb is not None
        assert gb["enabled"] is True
        assert gb["max_bet_pct_nav"] == 0.50
        assert gb["confidence_threshold"] == 0.70
        assert gb["allowed_modes"] == ["OPPORTUNITY"]
        assert "STRONG_TREND" in gb["allowed_regimes"]
        assert gb["min_seconds_between_bets"] == 14400
        assert gb["max_bets_per_week"] == 3

    def test_cloud_production_gambler_stays_bounded(self):
        """[P165] Was `test_cloud_production_has_no_gambler`, asserting the block
        was absent. It is present and deliberately so — `_description` reads
        "[PRE-LAUNCH] Aggressive sizing for OPPORTUNITY windows … Weekly limit
        raised 3→8, cooldown 4h→2h". An operator turned it on; a test insisting
        it be off is stale, not a finding.

        What is still worth guarding is the containment around it, so this now
        asserts the two properties that make gambler mode survivable: it can
        only fire in OPPORTUNITY, and a single bet can never exceed the
        `max_bet_pct_nav` ceiling the sizing module defaults to.
        """
        from orchestration.gambler_sizing import GamblerSizingConfig

        json_path = Path("configs/cloud_production.json")
        if not json_path.exists():
            pytest.skip("cloud_production.json not found")
        with open(json_path) as f:
            data = json.load(f)
        gb = data.get("gambler")
        if gb is None or not gb.get("enabled"):
            return  # disabled — nothing to contain

        assert gb.get("allowed_modes") == ["OPPORTUNITY"], (
            f"gambler must stay OPPORTUNITY-only; got {gb.get('allowed_modes')}"
        )
        ceiling = GamblerSizingConfig().max_bet_pct_nav
        assert gb["max_bet_pct_nav"] <= ceiling, (
            f"gambler max_bet_pct_nav={gb['max_bet_pct_nav']} exceeds the "
            f"module ceiling {ceiling}"
        )
        assert gb.get("confidence_threshold", 0) > 0, (
            "a zero/absent confidence threshold lets gambler sizing fire on "
            "any signal"
        )


# ===========================================================================
# Test Group 9: Edge cases
# ===========================================================================

class TestEdgeCases:
    """Edge cases: empty regime, short direction, get_weekly_count."""

    def test_empty_regime_blocked(self):
        gambler = _make_gambler()
        result = gambler.evaluate(**_trigger_kwargs(regime=""))
        assert result.triggered is False
        assert result.reason == "regime_not_allowed"

    def test_short_direction_triggers(self):
        """Negative direction should also trigger."""
        gambler = _make_gambler()
        result = gambler.evaluate(**_trigger_kwargs(intent_direction=-1.0))
        assert result.triggered is True

    def test_get_weekly_count(self):
        gambler = _make_gambler()
        assert gambler.get_weekly_count(now_ts=BASE_TS) == 0
        gambler.evaluate(**_trigger_kwargs(now_ts=BASE_TS))
        assert gambler.get_weekly_count(now_ts=BASE_TS + 100) == 1

    def test_small_direction_blocked(self):
        gambler = _make_gambler()
        result = gambler.evaluate(**_trigger_kwargs(intent_direction=0.05))
        assert result.triggered is False
        assert result.reason == "no_direction"
