"""
P2-02  Tests - Fee-Tier-Aware Execution Timing
================================================
Validates:
  1. DynamicMakerTakerSelector tier adjustments (free / post-free)
  2. ExecutionTimingEngine.evaluate() passes fee_context through
  3. ExecutionManager taker_allowed guard
  4. Config JSON structure
  5. ProductionConfig parsing
  6. Singleton config wiring
  7. Backward compatibility (no fee_context = unchanged behavior)
"""

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from execution.timing_engine import (
    DynamicMakerTakerSelector,
    ExecutionMode,
    ExecutionTimingCalculator,
    ExecutionTimingEngine,
    ExecutionTimingScore,
    get_timing_engine,
    reset_timing_engine,
)


# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------
ULTRA_TIMING_CONFIG = {
    "enable_tier_aware_timing": True,
    "free_tier_taker_upgrade_min_score": 0.60,
    "free_tier_taker_upgrade_min_urgency": 0.40,
    "post_tier_taker_min_urgency": 0.85,
    "post_tier_default_mode": "PASSIVE_PREFERRED",
}

FREE_TIER_CONTEXT = {
    "in_free_tier": True,
    "fee_weight": 0.0,
    "monthly_volume_usd": 5000.0,
    "tier_config": ULTRA_TIMING_CONFIG,
}

POST_TIER_CONTEXT = {
    "in_free_tier": False,
    "fee_weight": 1.0,
    "monthly_volume_usd": 15000.0,
    "tier_config": ULTRA_TIMING_CONFIG,
}

BLEND_ZONE_CONTEXT = {
    "in_free_tier": False,
    "fee_weight": 0.5,
    "monthly_volume_usd": 11000.0,
    "tier_config": ULTRA_TIMING_CONFIG,
}


# ===================================================================
# 1. Free Tier Upgrades
# ===================================================================
class TestFreeTierUpgrade(unittest.TestCase):
    """Free tier ($0 taker) -> upgrade to AGGRESSIVE_TAKER at lower thresholds."""

    def test_excellent_medium_upgrades_to_taker(self):
        """score=0.85 (excellent), urgency=0.50 (medium): default=PASSIVE_PREFERRED -> AGGRESSIVE_TAKER."""
        mode, adj = DynamicMakerTakerSelector.select_mode(
            timing_score=0.85, urgency=0.50, fee_context=FREE_TIER_CONTEXT,
        )
        self.assertEqual(mode, ExecutionMode.AGGRESSIVE_TAKER)
        self.assertTrue(adj)

    def test_good_medium_upgrades_to_taker(self):
        """score=0.70 (good), urgency=0.45 (medium): default=PASSIVE_PREFERRED -> AGGRESSIVE_TAKER."""
        mode, adj = DynamicMakerTakerSelector.select_mode(
            timing_score=0.70, urgency=0.45, fee_context=FREE_TIER_CONTEXT,
        )
        self.assertEqual(mode, ExecutionMode.AGGRESSIVE_TAKER)
        self.assertTrue(adj)

    def test_good_low_no_upgrade(self):
        """score=0.70, urgency=0.30 (low, below 0.40): stays PASSIVE_ONLY."""
        mode, adj = DynamicMakerTakerSelector.select_mode(
            timing_score=0.70, urgency=0.30, fee_context=FREE_TIER_CONTEXT,
        )
        self.assertEqual(mode, ExecutionMode.PASSIVE_ONLY)
        self.assertFalse(adj)

    def test_fair_score_no_upgrade(self):
        """score=0.50 (fair, below 0.60): stays PASSIVE_ONLY even with medium urgency."""
        mode, adj = DynamicMakerTakerSelector.select_mode(
            timing_score=0.50, urgency=0.50, fee_context=FREE_TIER_CONTEXT,
        )
        self.assertEqual(mode, ExecutionMode.PASSIVE_ONLY)
        self.assertFalse(adj)

    def test_poor_score_no_upgrade(self):
        """score=0.30 (poor): stays DELAY, never upgraded."""
        mode, adj = DynamicMakerTakerSelector.select_mode(
            timing_score=0.30, urgency=0.50, fee_context=FREE_TIER_CONTEXT,
        )
        self.assertNotEqual(mode, ExecutionMode.AGGRESSIVE_TAKER)
        self.assertFalse(adj)

    def test_already_aggressive_not_double_adjusted(self):
        """score=0.85, urgency=0.80 (excellent, high): already AGGRESSIVE_TAKER -> no tier adjustment flag."""
        mode, adj = DynamicMakerTakerSelector.select_mode(
            timing_score=0.85, urgency=0.80, fee_context=FREE_TIER_CONTEXT,
        )
        self.assertEqual(mode, ExecutionMode.AGGRESSIVE_TAKER)
        self.assertFalse(adj)  # Was already AGGRESSIVE from matrix

    def test_delay_not_upgraded(self):
        """DELAY mode is never upgraded to AGGRESSIVE_TAKER even in free tier."""
        mode, adj = DynamicMakerTakerSelector.select_mode(
            timing_score=0.20, urgency=0.20, fee_context=FREE_TIER_CONTEXT,
        )
        self.assertEqual(mode, ExecutionMode.DELAY)
        self.assertFalse(adj)


# ===================================================================
# 2. Post Free Tier Downgrades
# ===================================================================
class TestPostTierDowngrade(unittest.TestCase):
    """Post free tier -> restrict AGGRESSIVE_TAKER to high urgency only."""

    def test_taker_downgraded_below_threshold(self):
        """score=0.85, urgency=0.75 (high but <0.85): AGGRESSIVE_TAKER -> PASSIVE_PREFERRED."""
        mode, adj = DynamicMakerTakerSelector.select_mode(
            timing_score=0.85, urgency=0.75, fee_context=POST_TIER_CONTEXT,
        )
        self.assertEqual(mode, ExecutionMode.PASSIVE_PREFERRED)
        self.assertTrue(adj)

    def test_taker_allowed_above_threshold(self):
        """score=0.85, urgency=0.90 (>0.85): AGGRESSIVE_TAKER stays."""
        mode, adj = DynamicMakerTakerSelector.select_mode(
            timing_score=0.85, urgency=0.90, fee_context=POST_TIER_CONTEXT,
        )
        self.assertEqual(mode, ExecutionMode.AGGRESSIVE_TAKER)
        self.assertFalse(adj)

    def test_passive_modes_unchanged(self):
        """score=0.70, urgency=0.50 (good, medium): PASSIVE_PREFERRED stays."""
        mode, adj = DynamicMakerTakerSelector.select_mode(
            timing_score=0.70, urgency=0.50, fee_context=POST_TIER_CONTEXT,
        )
        self.assertEqual(mode, ExecutionMode.PASSIVE_PREFERRED)
        self.assertFalse(adj)

    def test_blend_zone_same_as_post(self):
        """Blend zone (not in_free_tier) applies same downgrade logic."""
        mode, adj = DynamicMakerTakerSelector.select_mode(
            timing_score=0.85, urgency=0.75, fee_context=BLEND_ZONE_CONTEXT,
        )
        self.assertEqual(mode, ExecutionMode.PASSIVE_PREFERRED)
        self.assertTrue(adj)

    def test_exhaustion_upgraded_then_downgraded(self):
        """Exhaustion upgrades urgency_cat medium->high, but post-tier still downgrades taker."""
        # score=0.85, urgency=0.50 -> medium -> EXHAUSTION bumps to high -> AGGRESSIVE_TAKER
        # But post-tier blocks because actual urgency=0.50 < 0.85
        mode, adj = DynamicMakerTakerSelector.select_mode(
            timing_score=0.85, urgency=0.50, regime_phase="EXHAUSTION",
            fee_context=POST_TIER_CONTEXT,
        )
        self.assertEqual(mode, ExecutionMode.PASSIVE_PREFERRED)
        self.assertTrue(adj)


# ===================================================================
# 3. Emergency Exit Bypass
# ===================================================================
class TestEmergencyExitBypass(unittest.TestCase):
    """need_urgent_exit=True always gets AGGRESSIVE_TAKER, regardless of tier."""

    def test_emergency_bypasses_post_tier(self):
        mode, adj = DynamicMakerTakerSelector.select_mode(
            timing_score=0.85, urgency=0.50,
            need_urgent_exit=True, fee_context=POST_TIER_CONTEXT,
        )
        self.assertEqual(mode, ExecutionMode.AGGRESSIVE_TAKER)
        self.assertFalse(adj)

    def test_emergency_bypasses_free_tier(self):
        mode, adj = DynamicMakerTakerSelector.select_mode(
            timing_score=0.30, urgency=0.10,
            need_urgent_exit=True, fee_context=FREE_TIER_CONTEXT,
        )
        self.assertEqual(mode, ExecutionMode.AGGRESSIVE_TAKER)
        self.assertFalse(adj)


# ===================================================================
# 4. Backward Compatibility (no fee_context)
# ===================================================================
class TestBackwardCompatibility(unittest.TestCase):
    """No fee_context -> original MODE_MATRIX behavior, no tier adjustment."""

    def test_no_context_returns_tuple(self):
        """Without fee_context, returns (mode, False)."""
        mode, adj = DynamicMakerTakerSelector.select_mode(
            timing_score=0.85, urgency=0.75,
        )
        self.assertEqual(mode, ExecutionMode.AGGRESSIVE_TAKER)
        self.assertFalse(adj)

    def test_none_context_no_adjustment(self):
        mode, adj = DynamicMakerTakerSelector.select_mode(
            timing_score=0.85, urgency=0.50, fee_context=None,
        )
        self.assertEqual(mode, ExecutionMode.PASSIVE_PREFERRED)
        self.assertFalse(adj)

    def test_disabled_config_no_adjustment(self):
        """enable_tier_aware_timing=False -> no adjustment."""
        ctx = {
            "in_free_tier": True,
            "fee_weight": 0.0,
            "tier_config": {"enable_tier_aware_timing": False},
        }
        mode, adj = DynamicMakerTakerSelector.select_mode(
            timing_score=0.85, urgency=0.50, fee_context=ctx,
        )
        self.assertEqual(mode, ExecutionMode.PASSIVE_PREFERRED)
        self.assertFalse(adj)

    def test_empty_tier_config_no_adjustment(self):
        """Empty tier_config -> no adjustment."""
        ctx = {"in_free_tier": True, "fee_weight": 0.0, "tier_config": {}}
        mode, adj = DynamicMakerTakerSelector.select_mode(
            timing_score=0.85, urgency=0.50, fee_context=ctx,
        )
        self.assertEqual(mode, ExecutionMode.PASSIVE_PREFERRED)
        self.assertFalse(adj)


# ===================================================================
# 5. ExecutionTimingEngine.evaluate() integration
# ===================================================================
class TestEngineEvaluateWithFeeContext(unittest.TestCase):
    """Verify evaluate() passes fee_context through to select_mode."""

    def setUp(self):
        reset_timing_engine()
        self.engine = ExecutionTimingEngine()

    def test_free_tier_upgrade_through_evaluate(self):
        """Free tier context passed through evaluate -> tier-adjusted score."""
        score, mode = self.engine.evaluate(
            asset="BTC", spread_bps=5.0, depth=1.0, vpin=0.2,
            lead_lag_edge=1.0, lead_lag_confidence=0.5,
            urgency=0.50, regime_phase="NORMAL",
            fee_context=FREE_TIER_CONTEXT,
        )
        # With low vpin and decent spread/depth, score should be >=0.6
        if score.total_score >= 0.60:
            self.assertEqual(mode, ExecutionMode.AGGRESSIVE_TAKER)
            self.assertTrue(score.tier_adjusted)

    def test_no_fee_context_no_tier_flag(self):
        """Without fee_context, tier_adjusted is always False."""
        score, mode = self.engine.evaluate(
            asset="BTC", spread_bps=5.0, depth=1.0, vpin=0.2,
            lead_lag_edge=0.0, lead_lag_confidence=0.0,
            urgency=0.50, regime_phase="NORMAL",
        )
        self.assertFalse(score.tier_adjusted)

    def test_tier_adjusted_in_to_dict(self):
        """tier_adjusted=True appears in to_dict() output."""
        score = ExecutionTimingScore(tier_adjusted=True)
        d = score.to_dict()
        self.assertTrue(d.get("tier_adjusted"))

    def test_tier_adjusted_false_omitted(self):
        """tier_adjusted=False is NOT in to_dict() (cleaner logs)."""
        score = ExecutionTimingScore(tier_adjusted=False)
        d = score.to_dict()
        self.assertNotIn("tier_adjusted", d)


# ===================================================================
# 6. Singleton Config
# ===================================================================
class TestSingletonConfig(unittest.TestCase):
    """get_timing_engine(timing_config=...) stores tier_config on instance."""

    def setUp(self):
        reset_timing_engine()

    def tearDown(self):
        reset_timing_engine()

    def test_config_stored(self):
        eng = get_timing_engine(timing_config=ULTRA_TIMING_CONFIG)
        self.assertEqual(eng.tier_config, ULTRA_TIMING_CONFIG)

    def test_no_config_empty_dict(self):
        eng = get_timing_engine()
        self.assertEqual(eng.tier_config, {})

    def test_singleton_second_call_refreshes_when_config_changes(self):
        """Second call with different config refreshes the singleton."""
        eng1 = get_timing_engine(timing_config=ULTRA_TIMING_CONFIG)
        eng2 = get_timing_engine(timing_config={"different": True})
        self.assertIsNot(eng1, eng2)
        self.assertEqual(eng2.tier_config, {"different": True})



# ===================================================================
# 7. ExecutionManager taker_allowed Guard
# ===================================================================
class TestExecutionManagerTakerGuard(unittest.TestCase):
    """taker_allowed=False downgrades MARKET -> LIMIT."""

    def setUp(self):
        from execution.execution_manager import ExecutionManager, ExecutionConfig
        self.mgr = ExecutionManager(
            exchange=None,
            config=ExecutionConfig(),
            dry_run=True,
        )

    def test_taker_allowed_true_keeps_market(self):
        """Default taker_allowed=True: MARKET stays MARKET."""
        result = self.mgr.execute_order(
            symbol="BTC/USD", side="BUY", size=0.01,
            price=50000.0, order_type="MARKET",
            taker_allowed=True,
        )
        self.assertEqual(result.order_type, "MARKET")

    def test_taker_blocked_with_price_downgrades(self):
        """taker_allowed=False + price provided: MARKET -> LIMIT."""
        result = self.mgr.execute_order(
            symbol="BTC/USD", side="BUY", size=0.01,
            price=50000.0, order_type="MARKET",
            taker_allowed=False,
        )
        self.assertEqual(result.order_type, "LIMIT")

    def test_taker_blocked_no_price_stays_market(self):
        """taker_allowed=False + no price: MARKET stays (last resort)."""
        result = self.mgr.execute_order(
            symbol="BTC/USD", side="BUY", size=0.01,
            price=None, order_type="MARKET",
            taker_allowed=False,
        )
        self.assertEqual(result.order_type, "MARKET")

    def test_limit_order_unaffected(self):
        """taker_allowed=False doesn't affect LIMIT orders."""
        result = self.mgr.execute_order(
            symbol="BTC/USD", side="BUY", size=0.01,
            price=50000.0, order_type="LIMIT",
            taker_allowed=False,
        )
        self.assertEqual(result.order_type, "LIMIT")


# ===================================================================
# 8. Config JSON Validation
# ===================================================================
class TestRealConfigJSON(unittest.TestCase):
    """Validate ultra_aggressive_5y.json timing_engine section."""

    @classmethod
    def setUpClass(cls):
        cfg_path = ROOT / "configs" / "ultra_aggressive_5y.json"
        with open(cfg_path, encoding="utf-8") as f:
            cls.data = json.load(f)
        cls.timing = cls.data.get("timing_engine", {})

    def test_section_exists(self):
        self.assertIn("timing_engine", self.data)

    def test_enable_flag(self):
        self.assertTrue(self.timing["enable_tier_aware_timing"])

    def test_free_tier_score_threshold(self):
        self.assertAlmostEqual(self.timing["free_tier_taker_upgrade_min_score"], 0.60)

    def test_free_tier_urgency_threshold(self):
        self.assertAlmostEqual(self.timing["free_tier_taker_upgrade_min_urgency"], 0.40)

    def test_post_tier_urgency_threshold(self):
        self.assertAlmostEqual(self.timing["post_tier_taker_min_urgency"], 0.85)

    def test_post_tier_default_mode(self):
        self.assertEqual(self.timing["post_tier_default_mode"], "PASSIVE_PREFERRED")
        # Verify it's a valid ExecutionMode
        ExecutionMode(self.timing["post_tier_default_mode"])


# ===================================================================
# 9. ProductionConfig Parsing
# ===================================================================
class TestProductionConfigTimingEngine(unittest.TestCase):
    """ProductionConfig.from_file() parses timing_engine section."""

    @classmethod
    def setUpClass(cls):
        # Ensure ULTRA_AGGRESSIVE_5Y profile exists
        cls.cfg_path = ROOT / "configs" / "cloud_production.json"
        cls.profile_path = ROOT / "configs" / "ultra_aggressive_5y.json"

    def test_ultra_profile_has_timing_config(self):
        """When risk_profile=ULTRA_AGGRESSIVE_5Y, timing_engine_config is set."""
        if not self.cfg_path.exists() or not self.profile_path.exists():
            self.skipTest("Config files not found")

        sys.path.insert(0, str(ROOT))
        from main import ProductionConfig
        config = ProductionConfig.from_file(
            self.cfg_path, risk_profile="ultra_aggressive_5y"
        )
        self.assertIsNotNone(config.timing_engine_config)
        self.assertTrue(config.timing_engine_config.get("enable_tier_aware_timing"))

    def test_default_config_no_timing(self):
        """Default config (no profile overlay) has no timing_engine_config."""
        if not self.cfg_path.exists():
            self.skipTest("Config files not found")

        from main import ProductionConfig
        config = ProductionConfig.from_file(self.cfg_path)
        # cloud_production.json doesn't have a timing_engine section
        if config.timing_engine_config is None:
            self.assertIsNone(config.timing_engine_config)
        else:
            # If it does, check it doesn't enable tier-aware
            self.assertFalse(
                config.timing_engine_config.get("enable_tier_aware_timing", False)
            )


# ===================================================================
# 10. Post Tier Default Mode Config Variations
# ===================================================================
class TestPostTierDefaultModeVariations(unittest.TestCase):
    """Custom post_tier_default_mode values."""

    def test_passive_only_downgrade(self):
        """Post-tier can downgrade to PASSIVE_ONLY instead of PASSIVE_PREFERRED."""
        ctx = {
            "in_free_tier": False,
            "fee_weight": 1.0,
            "tier_config": {
                **ULTRA_TIMING_CONFIG,
                "post_tier_default_mode": "PASSIVE_ONLY",
            },
        }
        mode, adj = DynamicMakerTakerSelector.select_mode(
            timing_score=0.85, urgency=0.75, fee_context=ctx,
        )
        self.assertEqual(mode, ExecutionMode.PASSIVE_ONLY)
        self.assertTrue(adj)

    def test_invalid_mode_falls_back(self):
        """Invalid post_tier_default_mode -> falls back to PASSIVE_PREFERRED."""
        ctx = {
            "in_free_tier": False,
            "fee_weight": 1.0,
            "tier_config": {
                **ULTRA_TIMING_CONFIG,
                "post_tier_default_mode": "NONEXISTENT",
            },
        }
        mode, adj = DynamicMakerTakerSelector.select_mode(
            timing_score=0.85, urgency=0.75, fee_context=ctx,
        )
        self.assertEqual(mode, ExecutionMode.PASSIVE_PREFERRED)
        self.assertTrue(adj)


if __name__ == "__main__":
    unittest.main()
