"""
Tests for P1-02: ThesisBudgetGovernor under ULTRA_AGGRESSIVE_5Y profile.

Covers:
  1) HIGH_RISK vs ULTRA parameter differences
  2) Budget hit + cooldown behavior
  3) Re-entry allowance and exhaustion
  4) Re-entry size multiplier output
  5) Structured veto reasons
  6) Config overlay wiring
"""

import json
import time
from pathlib import Path
from dataclasses import dataclass

import pytest

from risk.thesis_budget_governor import (
    ThesisBudgetGovernor,
    ThesisBudgetConfig,
    ThesisBudgetResult,
    ThesisBudgetVetoReason,
    create_thesis_budget_governor,
)


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class MockIntent:
    """Minimal intent for testing."""
    asset: str = "SOL"
    direction: float = 1.0
    quant_strategy_id: str = "momentum"
    source: str = "test"


def _high_risk_config() -> ThesisBudgetConfig:
    """HIGH_RISK defaults."""
    return ThesisBudgetConfig(
        loss_budget_pct_nav=0.008,
        loss_streak_limit=3,
        cooldown_bars=6,
        max_reentry_after_budget_hit=0,
    )


def _ultra_config() -> ThesisBudgetConfig:
    """ULTRA_AGGRESSIVE_5Y config."""
    return ThesisBudgetConfig(
        loss_budget_pct_nav=0.025,
        loss_streak_limit=6,
        cooldown_bars=2,
        max_reentry_after_budget_hit=2,
    )


# ─────────────────────────────────────────────────────────────────────────
# 1) HIGH_RISK vs ULTRA parameter differences
# ─────────────────────────────────────────────────────────────────────────

class TestParameterDifferences:

    def test_high_risk_defaults(self):
        cfg = _high_risk_config()
        assert cfg.loss_budget_pct_nav == 0.008
        assert cfg.loss_streak_limit == 3
        assert cfg.cooldown_bars == 6
        assert cfg.max_reentry_after_budget_hit == 0

    def test_ultra_aggressive_values(self):
        cfg = _ultra_config()
        assert cfg.loss_budget_pct_nav == 0.025
        assert cfg.loss_streak_limit == 6
        assert cfg.cooldown_bars == 2
        assert cfg.max_reentry_after_budget_hit == 2

    def test_factory_with_ultra_dict(self):
        """create_thesis_budget_governor with ULTRA dict should produce ULTRA config."""
        config_dict = {
            "thesis_budget": {
                "loss_budget_pct_nav": 0.025,
                "loss_streak_limit": 6,
                "cooldown_bars": 2,
                "max_reentry_after_budget_hit": 2,
            }
        }
        gov = create_thesis_budget_governor(config_dict=config_dict, nav=100_000)
        assert gov.config.loss_budget_pct_nav == 0.025
        assert gov.config.loss_streak_limit == 6
        assert gov.config.cooldown_bars == 2
        assert gov.config.max_reentry_after_budget_hit == 2

    def test_factory_default_is_high_risk(self):
        """create_thesis_budget_governor with empty dict should produce HIGH_RISK defaults."""
        gov = create_thesis_budget_governor(config_dict={}, nav=100_000)
        assert gov.config.loss_budget_pct_nav == 0.008
        assert gov.config.loss_streak_limit == 3
        assert gov.config.cooldown_bars == 6
        assert gov.config.max_reentry_after_budget_hit == 0


# ─────────────────────────────────────────────────────────────────────────
# 2) Budget hit + cooldown behavior
# ─────────────────────────────────────────────────────────────────────────

class TestBudgetHitAndCooldown:

    def test_loss_streak_triggers_cooldown(self):
        """Reaching loss_streak_limit must trigger veto + cooldown."""
        gov = ThesisBudgetGovernor(config=_high_risk_config(), nav=100_000)
        intent = MockIntent()
        key = "SOL:long:momentum"

        # Record 3 losses (= loss_streak_limit for HIGH_RISK)
        for _ in range(3):
            gov.record_fill(key, -100.0)

        result = gov.evaluate(intent)
        assert result.allow is False
        assert result.veto_reason == ThesisBudgetVetoReason.LOSS_STREAK_LIMIT
        assert "LOSS_STREAK_LIMIT" in result.veto_reason.value

    def test_ultra_loss_streak_needs_6(self):
        """ULTRA needs 6 losses to trigger streak limit, not 3."""
        gov = ThesisBudgetGovernor(config=_ultra_config(), nav=100_000)
        intent = MockIntent()
        key = "SOL:long:momentum"

        # 3 losses - should still ALLOW under ULTRA
        for _ in range(3):
            gov.record_fill(key, -100.0)

        result = gov.evaluate(intent)
        assert result.allow is True

        # 3 more losses (total 6) - NOW should veto
        for _ in range(3):
            gov.record_fill(key, -100.0)

        result = gov.evaluate(intent)
        assert result.allow is False
        assert result.veto_reason == ThesisBudgetVetoReason.LOSS_STREAK_LIMIT

    def test_cooldown_blocks_during_period(self):
        """During cooldown, evaluate must veto."""
        gov = ThesisBudgetGovernor(config=_high_risk_config(), nav=100_000)
        intent = MockIntent()
        key = "SOL:long:momentum"

        # Trigger cooldown via loss streak
        for _ in range(3):
            gov.record_fill(key, -100.0)

        # First eval triggers cooldown
        result = gov.evaluate(intent)
        assert result.allow is False

        # Second eval should still be in cooldown
        result = gov.evaluate(intent)
        assert result.allow is False
        assert result.veto_reason == ThesisBudgetVetoReason.COOLDOWN_ACTIVE
        assert result.cooldown_remaining_seconds > 0

    def test_cooldown_expires(self):
        """After cooldown expires, evaluate should allow."""
        cfg = _high_risk_config()
        cfg.bar_duration_seconds = 1  # 1 second bars for fast testing
        cfg.cooldown_bars = 1         # 1 bar cooldown = 1 second
        gov = ThesisBudgetGovernor(config=cfg, nav=100_000)
        intent = MockIntent()
        key = "SOL:long:momentum"

        # Trigger cooldown
        for _ in range(3):
            gov.record_fill(key, -100.0)
        result = gov.evaluate(intent)
        assert result.allow is False

        # Manually expire cooldown
        state = gov._thesis_states[key]
        state.cooldown_until_ts = time.time() - 1  # expired
        state.loss_streak = 0  # reset streak (as if time passed)

        result = gov.evaluate(intent)
        assert result.allow is True


# ─────────────────────────────────────────────────────────────────────────
# 3) Re-entry allowance and exhaustion
# ─────────────────────────────────────────────────────────────────────────

class TestReentryLogic:

    def test_high_risk_no_reentry(self):
        """HIGH_RISK (max_reentry=0) must veto immediately on budget exhaustion."""
        gov = ThesisBudgetGovernor(config=_high_risk_config(), nav=100_000)
        intent = MockIntent()
        key = "SOL:long:momentum"

        # Exhaust budget: 0.8% of 100K = $800
        gov.record_fill(key, -900.0)

        result = gov.evaluate(intent)
        assert result.allow is False
        assert result.veto_reason == ThesisBudgetVetoReason.BUDGET_EXHAUSTED

    def test_ultra_allows_reentry_twice(self):
        """ULTRA (max_reentry=2) must allow 2 re-entries after budget exhaustion."""
        gov = ThesisBudgetGovernor(config=_ultra_config(), nav=100_000)
        intent = MockIntent()
        key = "SOL:long:momentum"

        # Exhaust budget: 2.5% of 100K = $2500
        gov.record_fill(key, -3000.0)

        # First re-entry - ALLOW
        result = gov.evaluate(intent)
        assert result.allow is True
        assert result.reentry_used == 1
        assert result.reentry_max == 2
        assert result.reentry_size_multiplier < 1.0

        # Second re-entry - ALLOW
        result = gov.evaluate(intent)
        assert result.allow is True
        assert result.reentry_used == 2
        assert result.reentry_size_multiplier < 1.0

        # Third attempt - VETO (used up)
        result = gov.evaluate(intent)
        assert result.allow is False
        assert result.veto_reason == ThesisBudgetVetoReason.REENTRY_USED_UP

    def test_reentry_used_up_reason(self):
        """After all re-entries used, reason must be REENTRY_USED_UP."""
        gov = ThesisBudgetGovernor(config=_ultra_config(), nav=100_000)
        key = "SOL:long:momentum"
        intent = MockIntent()

        # Exhaust budget
        gov.record_fill(key, -3000.0)

        # Use up both re-entries
        gov.evaluate(intent)  # reentry 1
        gov.evaluate(intent)  # reentry 2

        # Third - must be REENTRY_USED_UP
        result = gov.evaluate(intent)
        assert result.allow is False
        assert result.veto_reason == ThesisBudgetVetoReason.REENTRY_USED_UP
        assert "REENTRY_USED_UP" in result.veto_reason.value
        assert result.reentry_used == 2
        assert result.reentry_max == 2


# ─────────────────────────────────────────────────────────────────────────
# 4) Re-entry size multiplier output
# ─────────────────────────────────────────────────────────────────────────

class TestReentrySizeMultiplier:

    def test_normal_allow_has_multiplier_1(self):
        """Normal allow (no reentry) should have size_multiplier=1.0."""
        gov = ThesisBudgetGovernor(config=_ultra_config(), nav=100_000)
        intent = MockIntent()

        result = gov.evaluate(intent)
        assert result.allow is True
        assert result.reentry_size_multiplier == 1.0

    def test_reentry_has_reduced_multiplier(self):
        """Re-entry must output size_multiplier < 1.0."""
        gov = ThesisBudgetGovernor(config=_ultra_config(), nav=100_000)
        intent = MockIntent()
        key = "SOL:long:momentum"

        # Exhaust budget
        gov.record_fill(key, -3000.0)

        result = gov.evaluate(intent)
        assert result.allow is True
        assert result.reentry_size_multiplier == 0.5

    def test_veto_has_default_multiplier(self):
        """Veto results should have default size_multiplier=1.0."""
        gov = ThesisBudgetGovernor(config=_high_risk_config(), nav=100_000)
        intent = MockIntent()
        key = "SOL:long:momentum"

        # Exhaust budget
        gov.record_fill(key, -900.0)

        result = gov.evaluate(intent)
        assert result.allow is False
        assert result.reentry_size_multiplier == 1.0


# ─────────────────────────────────────────────────────────────────────────
# 5) Structured veto reasons + config snapshot
# ─────────────────────────────────────────────────────────────────────────

class TestStructuredVetoReasons:

    def test_cooldown_result_has_config_snapshot(self):
        """Cooldown veto result must carry config snapshot."""
        gov = ThesisBudgetGovernor(config=_ultra_config(), nav=100_000)
        intent = MockIntent()
        key = "SOL:long:momentum"

        # Trigger cooldown
        for _ in range(6):
            gov.record_fill(key, -100.0)

        result = gov.evaluate(intent)
        assert result.allow is False
        assert result.config_loss_budget_pct_nav == 0.025
        assert result.config_loss_streak_limit == 6
        assert result.config_cooldown_bars == 2
        assert result.config_max_reentry == 2

    def test_allow_result_has_config_snapshot(self):
        """Normal allow result must also carry config snapshot."""
        gov = ThesisBudgetGovernor(config=_ultra_config(), nav=100_000)
        intent = MockIntent()

        result = gov.evaluate(intent)
        assert result.allow is True
        assert result.config_loss_budget_pct_nav == 0.025
        assert result.config_loss_streak_limit == 6

    def test_budget_exhausted_result_has_reentry_info(self):
        """Budget exhausted result must show reentry_used and reentry_max."""
        gov = ThesisBudgetGovernor(config=_high_risk_config(), nav=100_000)
        intent = MockIntent()
        key = "SOL:long:momentum"

        gov.record_fill(key, -900.0)

        result = gov.evaluate(intent)
        assert result.allow is False
        assert result.reentry_used == 0
        assert result.reentry_max == 0


# ─────────────────────────────────────────────────────────────────────────
# 6) Config overlay wiring (ProductionConfig -> ThesisBudgetGovernor)
# ─────────────────────────────────────────────────────────────────────────

class TestConfigOverlayWiring:

    def _write_json(self, directory: Path, name: str, data: dict) -> Path:
        path = directory / name
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_production_config_ultra_thesis_budget(self, tmp_path):
        """ProductionConfig with ULTRA overlay must carry thesis budget params."""
        from main import ProductionConfig

        base = {
            "assets": ["BTC", "ETH", "SOL"],
            "risk": {"hard_drawdown_halt": 0.10},
            "dms_timeout_seconds": 60,
        }
        profile = {
            "risk_profile": "ULTRA_AGGRESSIVE_5Y",
            "thesis_budget": {
                "loss_budget_pct_nav": 0.025,
                "loss_streak_limit": 6,
                "cooldown_bars": 2,
                "max_reentry_after_budget_hit": 2,
            },
        }
        self._write_json(tmp_path, "base.json", base)
        self._write_json(tmp_path, "ultra_aggressive_5y.json", profile)

        cfg = ProductionConfig.from_file(
            tmp_path / "base.json",
            risk_profile="ULTRA_AGGRESSIVE_5Y",
        )
        assert cfg.thesis_budget_loss_pct_nav == 0.025
        assert cfg.thesis_budget_loss_streak_limit == 6
        assert cfg.thesis_budget_cooldown_bars == 2
        assert cfg.thesis_budget_max_reentry == 2

    def test_production_config_default_thesis_budget(self, tmp_path):
        """ProductionConfig without overlay must use HIGH_RISK defaults."""
        from main import ProductionConfig

        base = {
            "assets": ["BTC"],
            "risk": {"hard_drawdown_halt": 0.10},
            "dms_timeout_seconds": 60,
        }
        self._write_json(tmp_path, "base.json", base)

        cfg = ProductionConfig.from_file(tmp_path / "base.json")
        assert cfg.thesis_budget_loss_pct_nav == 0.008
        assert cfg.thesis_budget_loss_streak_limit == 3
        assert cfg.thesis_budget_cooldown_bars == 6
        assert cfg.thesis_budget_max_reentry == 0


# ─────────────────────────────────────────────────────────────────────────
# 7) Real config file validation
# ─────────────────────────────────────────────────────────────────────────

class TestRealUltraConfig:

    def test_ultra_json_has_thesis_budget(self):
        """configs/ultra_aggressive_5y.json must contain thesis_budget section."""
        path = Path("configs/ultra_aggressive_5y.json")
        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        tb = data["thesis_budget"]
        assert tb["loss_budget_pct_nav"] == 0.025
        assert tb["loss_streak_limit"] == 6
        assert tb["cooldown_bars"] == 2
        assert tb["max_reentry_after_budget_hit"] == 2
