"""
Tests for P1-01: ULTRA_AGGRESSIVE_5Y risk profile.

Covers:
  1. Config file loads correctly with all expected fields
  2. Deep merge overlay replaces base values
  3. Deep merge preserves base keys not in overlay
  4. ProductionConfig.from_file with risk_profile overlay
  5. LeverageConfig accepts max_leverage=4.0 (within 10.0 limit)
  6. GovernorConfig respects overridden thresholds
  7. HIGH_RISK defaults remain unchanged without overlay
  8. Missing profile file raises FileNotFoundError
  9. Exposure caps and post-leverage caps from config
  10. Regime leverage map merged from overlay
"""

import json
import logging
import tempfile
from pathlib import Path

import pytest


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────

def _write_json(directory: Path, name: str, data: dict) -> Path:
    """Write a JSON file to directory and return its path."""
    path = directory / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# ─────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def config_dir(tmp_path):
    """Create a temp config directory with base + profile configs."""
    base = {
        "assets": ["BTC", "ETH", "SOL"],
        "risk": {
            "hard_drawdown_halt": 0.10,
            "correlation_crisis": 0.95,
            "max_gross_exposure": 1.50,
            "max_single_asset_pct": 0.80,
        },
        "leverage": {
            "max_leverage": 3.0,
            "max_position_notional_usd": 50000.0,
            "max_total_notional_usd": 100000.0,
        },
        "dms_timeout_seconds": 60,
    }
    profile = {
        "_description": "Test ultra aggressive profile",
        "risk_profile": "ULTRA_AGGRESSIVE_5Y",
        "risk": {
            "hard_drawdown_halt": 0.25,
            "correlation_crisis": 0.98,
            "max_gross_exposure": 2.50,
            "max_single_asset_pct": 1.00,
        },
        "leverage": {
            "max_leverage": 4.0,
            "max_position_notional_usd": 200000.0,
            "max_total_notional_usd": 400000.0,
        },
        "regime_leverage": {
            "VOLATILE_CHOP": 4.0,
            "MOMENTUM_RALLY": 3.0,
        },
        "exposure_caps": {
            "BTC": 0.60,
            "ETH": 0.60,
            "SOL": 0.80,
        },
        "post_leverage_caps": {
            "BTC": 0.50,
            "ETH": 0.50,
            "SOL": 0.60,
        },
    }
    _write_json(tmp_path, "base.json", base)
    _write_json(tmp_path, "ultra_aggressive_5y.json", profile)
    return tmp_path


# ─────────────────────────────────────────────────────────────────────────
# Tests: deep_merge
# ─────────────────────────────────────────────────────────────────────────

class TestDeepMerge:

    def test_overlay_replaces_flat_keys(self):
        from main import ProductionConfig
        base = {"a": 1, "b": 2}
        overlay = {"b": 99}
        result = ProductionConfig._deep_merge(base, overlay)
        assert result == {"a": 1, "b": 99}

    def test_overlay_deep_merges_nested_dicts(self):
        from main import ProductionConfig
        base = {"risk": {"dd": 0.10, "corr": 0.95}}
        overlay = {"risk": {"dd": 0.25}}
        result = ProductionConfig._deep_merge(base, overlay)
        assert result["risk"]["dd"] == 0.25
        assert result["risk"]["corr"] == 0.95  # preserved

    def test_meta_fields_skipped(self):
        from main import ProductionConfig
        base = {"a": 1}
        overlay = {"_description": "skip me", "a": 2}
        result = ProductionConfig._deep_merge(base, overlay)
        assert result == {"a": 2}
        assert "_description" not in result

    def test_overlay_adds_new_keys(self):
        from main import ProductionConfig
        base = {"a": 1}
        overlay = {"b": 2}
        result = ProductionConfig._deep_merge(base, overlay)
        assert result == {"a": 1, "b": 2}


# ─────────────────────────────────────────────────────────────────────────
# Tests: ProductionConfig loading
# ─────────────────────────────────────────────────────────────────────────

class TestProductionConfigProfile:

    def test_default_profile_is_high_risk(self, config_dir):
        """Without --risk-profile, profile should be HIGH_RISK."""
        from main import ProductionConfig
        cfg = ProductionConfig.from_file(config_dir / "base.json")
        assert cfg.risk_profile == "HIGH_RISK"
        assert cfg.hard_drawdown_halt == 0.10
        assert cfg.max_leverage == 3.0

    def test_profile_overlay_applies(self, config_dir):
        """With --risk-profile, overlay values should replace base."""
        from main import ProductionConfig
        cfg = ProductionConfig.from_file(
            config_dir / "base.json",
            risk_profile="ULTRA_AGGRESSIVE_5Y",
        )
        assert cfg.risk_profile == "ULTRA_AGGRESSIVE_5Y"
        assert cfg.hard_drawdown_halt == 0.25
        assert cfg.correlation_crisis == 0.98
        assert cfg.max_gross_exposure == 2.50
        assert cfg.max_single_asset_pct == 1.00
        assert cfg.max_leverage == 4.0
        assert cfg.max_position_notional_usd == 200000.0
        assert cfg.max_total_notional_usd == 400000.0

    def test_profile_exposure_caps(self, config_dir):
        """Exposure caps should come from profile overlay."""
        from main import ProductionConfig
        cfg = ProductionConfig.from_file(
            config_dir / "base.json",
            risk_profile="ULTRA_AGGRESSIVE_5Y",
        )
        assert cfg.exposure_caps["BTC"] == 0.60
        assert cfg.exposure_caps["SOL"] == 0.80
        assert cfg.post_leverage_caps["BTC"] == 0.50
        assert cfg.post_leverage_caps["SOL"] == 0.60

    def test_profile_regime_leverage_merged(self, config_dir):
        """Regime leverage map should merge overlay onto defaults."""
        from main import ProductionConfig
        cfg = ProductionConfig.from_file(
            config_dir / "base.json",
            risk_profile="ULTRA_AGGRESSIVE_5Y",
        )
        # Overridden by profile
        assert cfg.regime_leverage_map["VOLATILE_CHOP"] == 4.0
        assert cfg.regime_leverage_map["MOMENTUM_RALLY"] == 3.0
        # Default preserved (not in overlay)
        assert cfg.regime_leverage_map["UNKNOWN"] == 1.0

    def test_missing_profile_raises(self, config_dir):
        """Missing profile config file should raise FileNotFoundError."""
        from main import ProductionConfig
        with pytest.raises(FileNotFoundError, match="nonexistent"):
            ProductionConfig.from_file(
                config_dir / "base.json",
                risk_profile="NONEXISTENT",
            )

    def test_base_preserved_when_no_overlay(self, config_dir):
        """Base config values should be untouched without overlay."""
        from main import ProductionConfig
        cfg = ProductionConfig.from_file(config_dir / "base.json")
        assert cfg.hard_drawdown_halt == 0.10
        assert cfg.correlation_crisis == 0.95
        assert cfg.max_gross_exposure == 1.50
        assert cfg.max_single_asset_pct == 0.80
        assert cfg.max_leverage == 3.0


# ─────────────────────────────────────────────────────────────────────────
# Tests: LeverageConfig with higher max_leverage
# ─────────────────────────────────────────────────────────────────────────

class TestLeverageConfigBounds:

    def test_leverage_4x_accepted(self):
        """max_leverage=4.0 must be accepted (within [1.0, 10.0])."""
        from risk.leverage_guard import LeverageConfig
        cfg = LeverageConfig(max_leverage=4.0)
        assert cfg.max_leverage == 4.0

    def test_leverage_above_10_rejected(self):
        """max_leverage > 10.0 must raise ValueError."""
        from risk.leverage_guard import LeverageConfig
        with pytest.raises(ValueError, match="max_leverage"):
            LeverageConfig(max_leverage=11.0)

    def test_leverage_guard_uses_config(self):
        """LeverageGuard must use the passed config's max_leverage."""
        from risk.leverage_guard import LeverageGuard, LeverageConfig
        cfg = LeverageConfig(max_leverage=4.0)
        guard = LeverageGuard(config=cfg)
        assert guard.config.max_leverage == 4.0

    def test_leverage_guard_rejects_over_limit(self):
        """Order exceeding configured max_leverage must be rejected."""
        from risk.leverage_guard import LeverageGuard, LeverageConfig
        cfg = LeverageConfig(max_leverage=4.0)
        guard = LeverageGuard(config=cfg)
        # 5x leverage: $500K position on $100K equity
        ok, reason = guard.validate_order(
            position_notional_usd=500_000,
            account_equity=100_000,
        )
        assert ok is False
        assert "exceeds maximum" in reason

    def test_leverage_guard_allows_within_limit(self):
        """Order within configured max_leverage must be allowed."""
        from risk.leverage_guard import LeverageGuard, LeverageConfig
        cfg = LeverageConfig(
            max_leverage=4.0,
            max_position_notional_usd=400_000,
            max_total_notional_usd=500_000,
        )
        guard = LeverageGuard(config=cfg)
        # 3.5x leverage: $350K position on $100K equity
        ok, reason = guard.validate_order(
            position_notional_usd=350_000,
            account_equity=100_000,
        )
        assert ok is True


# ─────────────────────────────────────────────────────────────────────────
# Tests: GovernorConfig with overridden thresholds
# ─────────────────────────────────────────────────────────────────────────

class TestGovernorConfigProfile:

    def test_governor_accepts_ultra_aggressive_thresholds(self):
        """GovernorConfig should accept ultra-aggressive values."""
        from core.risk_governor import GovernorConfig
        cfg = GovernorConfig(
            hard_drawdown_halt=0.25,
            correlation_crisis=0.98,
            max_gross_exposure=2.50,
            max_asset_exposure=1.00,
        )
        assert cfg.hard_drawdown_halt == 0.25
        assert cfg.correlation_crisis == 0.98
        assert cfg.max_gross_exposure == 2.50
        assert cfg.max_asset_exposure == 1.00

    def test_governor_default_is_high_risk(self):
        """Default GovernorConfig must agree with the canonical rule table.

        [P165] Was pinned at hard_drawdown_halt=0.10 / correlation_crisis=0.95,
        pre-UNLEASH values (`core/constants.py` now carries 0.20 [UL-5] and 0.98
        [UL-4]) that this repo has never held — red since the initial commit.

        The literals are gone on purpose. `core/constants.OPPORTUNITY_RULE_TABLE`
        is the declared single source of truth for these thresholds and
        `core/risk_governor.py` is one of several consumers; the failure worth
        catching is the two disagreeing, not either one having a specific value.
        """
        from core.constants import get_rule
        from core.risk_governor import GovernorConfig

        cfg = GovernorConfig()
        for field in ("hard_drawdown_halt", "correlation_crisis",
                      "max_gross_exposure", "max_asset_exposure"):
            expected = get_rule(field, is_opportunity=False)
            if expected is None:
                continue  # not tracked in the rule table
            assert getattr(cfg, field) == expected, (
                f"GovernorConfig.{field}={getattr(cfg, field)} disagrees with "
                f"OPPORTUNITY_RULE_TABLE['{field}']['normal']={expected}. "
                f"Update BOTH or neither."
            )


# ─────────────────────────────────────────────────────────────────────────
# Tests: Actual config file validation
# ─────────────────────────────────────────────────────────────────────────

class TestRealConfigFile:

    def test_ultra_aggressive_5y_json_exists(self):
        """configs/ultra_aggressive_5y.json must exist."""
        path = Path("configs/ultra_aggressive_5y.json")
        assert path.exists(), f"Config file not found: {path}"

    def test_ultra_aggressive_5y_json_valid(self):
        """configs/ultra_aggressive_5y.json must be valid JSON with required fields."""
        path = Path("configs/ultra_aggressive_5y.json")
        with open(path) as f:
            data = json.load(f)
        assert data["risk_profile"] == "ULTRA_AGGRESSIVE_5Y"
        assert data["risk"]["hard_drawdown_halt"] == 0.25
        assert data["risk"]["correlation_crisis"] == 0.98
        assert data["risk"]["max_gross_exposure"] == 2.50
        # [P165] Was 4.0. `configs/canonical_config.MAX_LEVERAGE` is 3.0 and the
        # profile tracks it; assert the parity, not the literal.
        from configs.canonical_config import MAX_LEVERAGE
        assert data["leverage"]["max_leverage"] == MAX_LEVERAGE
        assert data["exposure_caps"]["BTC"] == 0.60
        assert data["post_leverage_caps"]["SOL"] == 0.60
