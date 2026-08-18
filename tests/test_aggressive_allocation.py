"""
tests/test_aggressive_allocation.py - P4-01 AggressiveAllocation tests

Coverage:
- Score computation (basic, zero, negative, max)
- Tier selection (STRONG, MID, BASE)
- Cap-aware normalization (never exceeds cap, overflow redistribution)
- Anti-churn guard (preserves existing positions)
- Edge cases (first tick, stale scores, disabled)
- Config loading from ultra_aggressive_5y.json
- Proof snapshot format
"""

import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestration.aggressive_allocator import (
    AggressiveAllocationConfig,
    AggressiveAllocator,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ULTRA_CAPS = {"SOL": 0.80, "BTC": 0.60, "ETH": 0.60}


def _ultra_config() -> AggressiveAllocationConfig:
    return AggressiveAllocationConfig(
        enabled=True,
        strong_threshold=0.80,
        mid_threshold=0.65,
        strong_weights=[1.0, 0.0, 0.0],
        mid_weights=[0.7, 0.3, 0.0],
        base_weights=[0.5, 0.3, 0.2],
    )


def _allocator(config=None) -> AggressiveAllocator:
    return AggressiveAllocator(config=config or _ultra_config())


def _signals(qd=0.0, qc=0.5, rc=0.5):
    return {
        "quant_direction": qd,
        "quant_confidence": qc,
        "regime_confidence": rc,
    }


# ---------------------------------------------------------------------------
# Test: Score computation
# ---------------------------------------------------------------------------

class TestScoreComputation(unittest.TestCase):
    """Score = abs(quant_direction) * quant_confidence * regime_confidence."""

    def test_score_basic(self):
        s = AggressiveAllocator.compute_score(0.5, 0.8, 0.7)
        self.assertAlmostEqual(s, 0.5 * 0.8 * 0.7)

    def test_score_zero_direction(self):
        s = AggressiveAllocator.compute_score(0.0, 0.8, 0.7)
        self.assertAlmostEqual(s, 0.0)

    def test_score_negative_direction_uses_abs(self):
        s = AggressiveAllocator.compute_score(-0.6, 0.8, 0.7)
        self.assertAlmostEqual(s, 0.6 * 0.8 * 0.7)

    def test_score_all_max(self):
        s = AggressiveAllocator.compute_score(1.0, 1.0, 1.0)
        self.assertAlmostEqual(s, 1.0)

    def test_score_rewards_net_alpha_and_realized_pnl_and_penalizes_toxicity(self):
        base = AggressiveAllocator.compute_score(0.6, 0.8, 0.8)
        improved = AggressiveAllocator.compute_score(
            0.6,
            0.8,
            0.8,
            expected_net_alpha_bps=16.0,
            toxicity_score=0.10,
            side_avg_realized_pnl_bps=18.0,
            side_total_realized_pnl_usd=40.0,
        )
        toxic = AggressiveAllocator.compute_score(
            0.6,
            0.8,
            0.8,
            expected_net_alpha_bps=16.0,
            toxicity_score=0.90,
            side_avg_realized_pnl_bps=18.0,
            side_total_realized_pnl_usd=40.0,
        )
        self.assertGreater(improved, base)
        self.assertLess(toxic, improved)


# ---------------------------------------------------------------------------
# Test: Tier selection and allocation
# ---------------------------------------------------------------------------

class TestTierSelection(unittest.TestCase):
    """Tier selection based on top score."""

    def test_strong_tier_concentration(self):
        """top_score > 0.80 -> top asset gets maximum cap, overflow redistributed."""
        alloc = _allocator()
        alloc.update_score("SOL", _signals(qd=0.95, qc=0.95, rc=0.95))
        alloc.update_score("BTC", _signals(qd=0.3, qc=0.5, rc=0.5))
        alloc.update_score("ETH", _signals(qd=0.2, qc=0.5, rc=0.5))

        sol_cap = alloc.get_allocation("SOL", ULTRA_CAPS)
        btc_cap = alloc.get_allocation("BTC", ULTRA_CAPS)
        eth_cap = alloc.get_allocation("ETH", ULTRA_CAPS)

        # SOL should get maximum (capped at 0.80)
        self.assertAlmostEqual(sol_cap, 0.80)
        # SOL weight=1.0 -> capped at 0.80 -> overflow 0.20 redistributed
        # BTC/ETH get overflow (total allocation preserved)
        total = sol_cap + btc_cap + eth_cap
        self.assertAlmostEqual(total, 1.0, places=5)
        # SOL dominates
        self.assertGreater(sol_cap, btc_cap)
        self.assertGreater(sol_cap, eth_cap)

    def test_mid_tier_concentration(self):
        """top_score between 0.65 and 0.80 -> [0.7, 0.3, 0.0]."""
        alloc = _allocator()
        alloc.update_score("SOL", _signals(qd=0.85, qc=0.85, rc=0.95))  # ~0.686
        alloc.update_score("BTC", _signals(qd=0.3, qc=0.5, rc=0.5))
        alloc.update_score("ETH", _signals(qd=0.2, qc=0.5, rc=0.5))

        # Verify top score is in MID range
        top = alloc._scores["SOL"]
        self.assertGreater(top, 0.65)
        self.assertLessEqual(top, 0.80)

        sol_cap = alloc.get_allocation("SOL", ULTRA_CAPS)
        btc_cap = alloc.get_allocation("BTC", ULTRA_CAPS)
        eth_cap = alloc.get_allocation("ETH", ULTRA_CAPS)

        # SOL: 0.7 * 0.80 = 0.56 (weight * cap)
        # But cap-aware normalize may redistribute
        self.assertGreater(sol_cap, btc_cap)
        self.assertAlmostEqual(eth_cap, 0.0)

    def test_base_tier_equal(self):
        """top_score < 0.65 -> [0.5, 0.3, 0.2]."""
        alloc = _allocator()
        alloc.update_score("SOL", _signals(qd=0.5, qc=0.5, rc=0.5))  # 0.125
        alloc.update_score("BTC", _signals(qd=0.4, qc=0.5, rc=0.5))  # 0.100
        alloc.update_score("ETH", _signals(qd=0.3, qc=0.5, rc=0.5))  # 0.075

        sol_cap = alloc.get_allocation("SOL", ULTRA_CAPS)
        btc_cap = alloc.get_allocation("BTC", ULTRA_CAPS)
        eth_cap = alloc.get_allocation("ETH", ULTRA_CAPS)

        # All three should get positive allocation
        self.assertGreater(sol_cap, 0)
        self.assertGreater(btc_cap, 0)
        self.assertGreater(eth_cap, 0)

    def test_ranking_order_descending(self):
        """Assets are ranked by score descending."""
        alloc = _allocator()
        alloc.update_score("ETH", _signals(qd=0.9, qc=0.9, rc=0.9))  # highest
        alloc.update_score("BTC", _signals(qd=0.5, qc=0.5, rc=0.5))  # mid
        alloc.update_score("SOL", _signals(qd=0.2, qc=0.5, rc=0.5))  # lowest

        eth_cap = alloc.get_allocation("ETH", ULTRA_CAPS)
        btc_cap = alloc.get_allocation("BTC", ULTRA_CAPS)
        sol_cap = alloc.get_allocation("SOL", ULTRA_CAPS)

        self.assertGreater(eth_cap, btc_cap)
        self.assertGreaterEqual(btc_cap, sol_cap)


# ---------------------------------------------------------------------------
# Test: Cap-aware normalization
# ---------------------------------------------------------------------------

class TestCapAwareNormalization(unittest.TestCase):
    """Cap-aware normalization prevents downstream clamp churn."""

    def test_normalization_never_exceeds_cap(self):
        """Output for each asset must be <= exposure_caps[asset]."""
        alloc = _allocator()
        alloc.update_score("SOL", _signals(qd=1.0, qc=1.0, rc=1.0))
        alloc.update_score("BTC", _signals(qd=0.5, qc=0.5, rc=0.5))
        alloc.update_score("ETH", _signals(qd=0.3, qc=0.5, rc=0.5))

        for asset in ["SOL", "BTC", "ETH"]:
            cap = alloc.get_allocation(asset, ULTRA_CAPS)
            self.assertLessEqual(
                cap, ULTRA_CAPS[asset] + 1e-9,
                f"{asset} allocation {cap} exceeds cap {ULTRA_CAPS[asset]}"
            )

    def test_normalization_overflow_redistribution(self):
        """When top asset is capped, overflow goes to lower-ranked assets."""
        # Use tight caps to force overflow
        tight_caps = {"SOL": 0.30, "BTC": 0.60, "ETH": 0.60}
        alloc = _allocator()
        alloc.update_score("SOL", _signals(qd=1.0, qc=1.0, rc=1.0))  # top
        alloc.update_score("BTC", _signals(qd=0.5, qc=0.5, rc=0.5))
        alloc.update_score("ETH", _signals(qd=0.3, qc=0.5, rc=0.5))

        sol_cap = alloc.get_allocation("SOL", tight_caps)
        btc_cap = alloc.get_allocation("BTC", tight_caps)

        # SOL capped at 0.30, overflow should boost BTC
        self.assertAlmostEqual(sol_cap, 0.30)
        self.assertGreater(btc_cap, 0.0)

    def test_cap_aware_with_max_single_asset_60(self):
        """User-specified test: max_single_asset=0.60, strong, sum==1.0, top1==0.60."""
        caps = {"SOL": 0.60, "BTC": 0.60, "ETH": 0.60}
        alloc = _allocator()
        alloc.update_score("SOL", _signals(qd=1.0, qc=1.0, rc=1.0))
        alloc.update_score("BTC", _signals(qd=0.3, qc=0.5, rc=0.5))
        alloc.update_score("ETH", _signals(qd=0.2, qc=0.5, rc=0.5))

        sol_cap = alloc.get_allocation("SOL", caps)
        btc_cap = alloc.get_allocation("BTC", caps)
        eth_cap = alloc.get_allocation("ETH", caps)

        # SOL capped at 0.60 (not 1.0)
        self.assertAlmostEqual(sol_cap, 0.60)
        # Overflow 0.40 redistributed to BTC/ETH
        total = sol_cap + btc_cap + eth_cap
        self.assertAlmostEqual(total, 1.0, places=5)
        # top1 == 0.60
        self.assertAlmostEqual(sol_cap, 0.60)


# ---------------------------------------------------------------------------
# Test: Anti-churn guard
# ---------------------------------------------------------------------------

class TestAntiChurnGuard(unittest.TestCase):
    """Anti-churn: allocation doesn't force-close existing positions."""

    def test_anti_churn_preserves_existing_position(self):
        """Allocation=0 but current exposure=0.50 -> cap >= 0.50."""
        alloc = _allocator()
        alloc.update_score("SOL", _signals(qd=1.0, qc=1.0, rc=1.0))
        alloc.update_score("BTC", _signals(qd=0.1, qc=0.3, rc=0.3))
        alloc.update_score("ETH", _signals(qd=0.1, qc=0.3, rc=0.3))

        current_exp = {"SOL": 0.0, "BTC": 0.50, "ETH": 0.0}

        # BTC has weight=0 in STRONG tier, but current_exposure=0.50
        btc_cap = alloc.get_allocation("BTC", ULTRA_CAPS, current_exp)
        self.assertGreaterEqual(btc_cap, 0.50)

    def test_anti_churn_allows_growth(self):
        """Anti-churn doesn't prevent growing if allocation allows."""
        alloc = _allocator()
        alloc.update_score("SOL", _signals(qd=1.0, qc=1.0, rc=1.0))
        alloc.update_score("BTC", _signals(qd=0.1, qc=0.3, rc=0.3))
        alloc.update_score("ETH", _signals(qd=0.1, qc=0.3, rc=0.3))

        current_exp = {"SOL": 0.30, "BTC": 0.0, "ETH": 0.0}

        # SOL: allocation should allow up to cap (0.80)
        sol_cap = alloc.get_allocation("SOL", ULTRA_CAPS, current_exp)
        self.assertGreaterEqual(sol_cap, 0.30)
        self.assertLessEqual(sol_cap, 0.80)


# ---------------------------------------------------------------------------
# Test: Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases(unittest.TestCase):
    """Edge cases and fail-open behavior."""

    def test_first_tick_no_scores_failopen(self):
        """No scores at all -> returns exposure_caps (fail-open)."""
        alloc = _allocator()
        cap = alloc.get_allocation("SOL", ULTRA_CAPS)
        self.assertAlmostEqual(cap, 0.80)  # SOL cap

    def test_stale_scores_used(self):
        """Only SOL has a fresh score - BTC/ETH use previous tick."""
        alloc = _allocator()
        # Simulate previous tick
        alloc.update_score("BTC", _signals(qd=0.8, qc=0.8, rc=0.8))  # 0.512
        alloc.update_score("ETH", _signals(qd=0.3, qc=0.5, rc=0.5))  # 0.075

        # New tick - only SOL updates
        alloc.update_score("SOL", _signals(qd=0.9, qc=0.9, rc=0.9))  # 0.729

        # Should use SOL(fresh)=0.729, BTC(stale)=0.512, ETH(stale)=0.075
        sol_cap = alloc.get_allocation("SOL", ULTRA_CAPS)
        btc_cap = alloc.get_allocation("BTC", ULTRA_CAPS)

        # SOL should be top (MID tier since 0.729 > 0.65)
        self.assertGreater(sol_cap, btc_cap)

    def test_disabled_config_passthrough(self):
        """enabled=False -> allocator not created (tested via init pattern)."""
        cfg = AggressiveAllocationConfig(enabled=False)
        self.assertFalse(cfg.enabled)

    def test_single_asset_scored(self):
        """Only one asset has a score."""
        alloc = _allocator()
        alloc.update_score("BTC", _signals(qd=0.9, qc=0.9, rc=0.9))

        # Should still return a valid cap
        cap = alloc.get_allocation("BTC", ULTRA_CAPS)
        self.assertGreater(cap, 0)
        self.assertLessEqual(cap, ULTRA_CAPS["BTC"])


# ---------------------------------------------------------------------------
# Test: Config from JSON
# ---------------------------------------------------------------------------

class TestConfigFromJSON(unittest.TestCase):
    """Config loading from ultra_aggressive_5y.json."""

    def test_config_from_dict(self):
        d = {
            "enabled": True,
            "strong_threshold": 0.85,
            "mid_threshold": 0.70,
            "strong_weights": [1.0, 0.0, 0.0],
            "mid_weights": [0.7, 0.3, 0.0],
            "base_weights": [0.4, 0.35, 0.25],
            "correlation_merge_threshold": 0.91,
            "correlation_reduce_threshold": 0.77,
            "correlation_reduce_factor": 0.25,
        }
        cfg = AggressiveAllocationConfig.from_dict(d)
        self.assertTrue(cfg.enabled)
        self.assertAlmostEqual(cfg.strong_threshold, 0.85)
        self.assertAlmostEqual(cfg.mid_threshold, 0.70)
        self.assertEqual(cfg.base_weights, [0.4, 0.35, 0.25])
        self.assertAlmostEqual(cfg.correlation_merge_threshold, 0.91)
        self.assertAlmostEqual(cfg.correlation_reduce_threshold, 0.77)
        self.assertAlmostEqual(cfg.correlation_reduce_factor, 0.25)

    def test_config_from_ultra_aggressive_json(self):
        config_path = Path(__file__).parent.parent / "configs" / "ultra_aggressive_5y.json"
        if not config_path.exists():
            self.skipTest("ultra_aggressive_5y.json not found")

        with open(config_path, encoding="utf-8") as f:
            data = json.load(f)

        alloc = data.get("allocation")
        self.assertIsNotNone(alloc)

        cfg = AggressiveAllocationConfig.from_dict(alloc)
        self.assertTrue(cfg.enabled)
        self.assertAlmostEqual(cfg.strong_threshold, 0.80)
        self.assertAlmostEqual(cfg.mid_threshold, 0.65)
        self.assertEqual(cfg.strong_weights, [1.0, 0.0, 0.0])
        self.assertEqual(cfg.mid_weights, [0.7, 0.3, 0.0])
        self.assertEqual(cfg.base_weights, [0.5, 0.3, 0.2])

    def test_config_from_none(self):
        cfg = AggressiveAllocationConfig.from_dict(None)
        self.assertFalse(cfg.enabled)

    def test_cloud_production_allocation_block_is_loadable(self):
        """[P165] Was `test_cloud_production_has_no_allocation`.

        The block is now present and enabled in that profile (deliberately —
        `live_high_risk.json` carries a fuller version with the same flags), so
        "must be absent" is stale. The property still worth holding is that
        whatever is in the file round-trips through `from_dict` without silently
        falling back to defaults, since a typo'd key there would disable
        allocation while the JSON still reads as enabled.
        """
        config_path = Path(__file__).parent.parent / "configs" / "cloud_production.json"
        if not config_path.exists():
            self.skipTest("cloud_production.json not found")

        with open(config_path, encoding="utf-8") as f:
            data = json.load(f)

        block = data.get("allocation")
        if block is None:
            return  # absent is still a valid state

        cfg = AggressiveAllocationConfig.from_dict(block)
        self.assertEqual(cfg.enabled, bool(block.get("enabled", False)))
        # `live_enabled` is NOT a field on this dataclass — main.py:2303 reads it
        # straight off the raw dict. Assert the shape so a rename on either side
        # surfaces here rather than silently defaulting the live gate to False.
        self.assertIn("live_enabled", block)
        self.assertFalse(hasattr(cfg, "live_enabled"))


# ---------------------------------------------------------------------------
# Test: Proof snapshot
# ---------------------------------------------------------------------------

class TestProofSnapshot(unittest.TestCase):
    """Proof log snapshot format."""

    def test_proof_snapshot_format(self):
        alloc = _allocator()
        alloc.update_score("SOL", _signals(qd=0.9, qc=0.9, rc=0.95))
        alloc.update_score("BTC", _signals(qd=0.4, qc=0.5, rc=0.5))
        alloc.update_score("ETH", _signals(qd=0.3, qc=0.5, rc=0.5))

        # Must call get_allocation to populate snapshot
        alloc.get_allocation("SOL", ULTRA_CAPS)

        snap = alloc.get_proof_snapshot()
        self.assertIsNotNone(snap)
        self.assertIn("tier", snap)
        self.assertIn("top_score", snap)
        self.assertIn("scores", snap)
        self.assertIn("final_alloc", snap)
        self.assertIn(snap["tier"], ["STRONG", "MID", "BASE"])

    def test_proof_snapshot_none_before_allocation(self):
        alloc = _allocator()
        self.assertIsNone(alloc.get_proof_snapshot())


# ---------------------------------------------------------------------------
# Test: update_score from agent_signals
# ---------------------------------------------------------------------------

class TestUpdateScore(unittest.TestCase):
    """update_score extracts correct keys from agent_signals."""

    def test_update_score_extracts_keys(self):
        alloc = _allocator()
        signals = {
            "quant_direction": -0.7,
            "quant_confidence": 0.8,
            "regime_confidence": 0.9,
            "other_key": "ignored",
        }
        score = alloc.update_score("SOL", signals)
        expected = 0.7 * 0.8 * 0.9
        self.assertAlmostEqual(score, expected)
        self.assertAlmostEqual(alloc._scores["SOL"], expected)

    def test_update_score_defaults(self):
        """Missing keys default to safe values."""
        alloc = _allocator()
        score = alloc.update_score("BTC", {})
        expected = 0.0 * 0.5 * 0.5  # qd=0, qc=0.5, rc=0.5
        self.assertAlmostEqual(score, expected)

    def test_update_score_tracks_profitability_components_in_snapshot(self):
        alloc = _allocator()
        signals = {
            "quant_direction": -0.7,
            "quant_confidence": 0.8,
            "regime_confidence": 0.9,
            "aggressive_allocator_expected_net_alpha_bps": 14.0,
            "aggressive_allocator_toxicity_score": 0.15,
            "aggressive_allocator_side_avg_realized_pnl_bps": 12.0,
            "aggressive_allocator_side_total_realized_pnl_usd": 25.0,
        }
        alloc.update_score("SOL", signals)
        alloc.update_score("BTC", _signals(qd=0.2, qc=0.5, rc=0.5))
        alloc.update_score("ETH", _signals(qd=0.1, qc=0.5, rc=0.5))
        alloc.get_allocation("SOL", ULTRA_CAPS)
        snap = alloc.get_proof_snapshot()

        self.assertIn("score_components", snap)
        self.assertEqual(snap["score_components"]["SOL"]["expected_net_alpha_bps"], 14.0)
        self.assertEqual(snap["score_components"]["SOL"]["toxicity_score"], 0.15)


if __name__ == "__main__":
    unittest.main()
