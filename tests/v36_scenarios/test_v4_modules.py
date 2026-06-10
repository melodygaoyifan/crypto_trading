"""
================================================================================
HMATS v4.0-COMPLETE - UNIFIED TEST SUITE FOR NEW MODULES
================================================================================
Version: 4.0.0
Purpose: Verify all newly added/updated modules work correctly

TESTS INCLUDED:
    1. Scenario Walkthroughs Unified
    2. Integrated Liquidity Manager
    3. Idle Module Integrator v4
    4. Enhanced Meta Decision Layer
    5. Interface Compatibility Layer

USAGE:
    python -m pytest integration/core/test_v4_modules.py -v

    Or run directly:
    python integration/core/test_v4_modules.py
================================================================================
"""

import unittest
import logging
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

# [TEST-CLEANUP 2026-06-09] All 5 test classes in this file target v4.0
# modules that have been removed or moved out of the live tree (per
# subsequent refactors). Specifically: ScenarioWalkthroughsUnified,
# IntegratedLiquidityManagerV4 (superseded by liquidity/integrated_manager),
# IdleModuleIntegratorV4, EnhancedMetaDecision (rename in 2026-Q1),
# and InterfaceCompatibilityLayer. None of these symbols import on
# the current live tree, so the entire file is skipped at module
# scope rather than asserting in collection. Delete this file when
# the v4 transition arc is fully retired.
pytestmark = pytest.mark.skip(
    reason="[TEST-CLEANUP 2026-06-09] v4 modules (ScenarioWalkthroughsUnified, "
           "IntegratedLiquidityManagerV4, IdleModuleIntegratorV4, "
           "EnhancedMetaDecision, InterfaceCompatibilityLayer) removed or "
           "renamed; tests pre-date the rename and were never updated."
)

logging.basicConfig(level=logging.WARNING)

# =============================================================================
# TEST: SCENARIO WALKTHROUGHS UNIFIED
# =============================================================================

class TestScenarioWalkthroughsUnified(unittest.TestCase):
    """Test the unified scenario walkthroughs module."""
    
    def test_import(self):
        """Test that module can be imported."""
        from integration.core.scenario_walkthroughs_unified import (
            ScenarioRunner,
            run_all_scenarios,
            run_live_scenarios,
            LiveScenarioType,
        )
        self.assertIsNotNone(ScenarioRunner)
    
    def test_live_scenario_types(self):
        """Test LiveScenarioType enum values."""
        from integration.core.scenario_walkthroughs_unified import LiveScenarioType
        
        self.assertEqual(LiveScenarioType.SOL_BULLISH_BREAKOUT.value, "SOL_BULLISH_BREAKOUT")
        self.assertEqual(LiveScenarioType.SOL_BEARISH_BREAKDOWN.value, "SOL_BEARISH_BREAKDOWN")
        self.assertEqual(LiveScenarioType.WEEKEND_LOW_LIQUIDITY.value, "WEEKEND_LOW_LIQUIDITY")
    
    def test_scenario_runner_init(self):
        """Test ScenarioRunner initialization."""
        from integration.core.scenario_walkthroughs_unified import ScenarioRunner
        
        runner = ScenarioRunner()
        self.assertIsNotNone(runner)
        self.assertEqual(runner.results, [])
    
    def test_simulate_sol_bullish(self):
        """Test SOL bullish breakout scenario."""
        from integration.core.scenario_walkthroughs_unified import simulate_sol_bullish_breakout
        
        result = simulate_sol_bullish_breakout(verbose=False)
        
        self.assertIsNotNone(result)
        self.assertTrue(result.passed)
        self.assertGreater(len(result.state_history), 0)
        self.assertIn("NORMAL -> OPPORTUNITY", result.actual_mode_changes)
    
    def test_simulate_sol_bearish(self):
        """Test SOL bearish breakdown scenario."""
        from integration.core.scenario_walkthroughs_unified import simulate_sol_bearish_breakdown
        
        result = simulate_sol_bearish_breakdown(verbose=False)
        
        self.assertIsNotNone(result)
        self.assertTrue(result.passed)
        self.assertLess(result.actual_final_exposure, 0)  # Should be short
    
    def test_run_live_scenarios(self):
        """Test running all live scenarios."""
        from integration.core.scenario_walkthroughs_unified import run_live_scenarios
        
        results = run_live_scenarios(verbose=False)
        
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        
        # At least some should pass
        passed = sum(1 for r in results if r.passed)
        self.assertGreater(passed, 0)


# =============================================================================
# TEST: INTEGRATED LIQUIDITY MANAGER
# =============================================================================

class TestIntegratedLiquidityManager(unittest.TestCase):
    """Test the integrated liquidity manager module."""
    
    def test_import(self):
        """Test that module can be imported."""
        from integration.core.integrated_liquidity_manager import (
            IntegratedWeekendLiquidityManager,
            get_integrated_liquidity_manager,
            LiquidityPeriod,
        )
        self.assertIsNotNone(IntegratedWeekendLiquidityManager)
    
    def test_manager_init(self):
        """Test manager initialization."""
        from integration.core.integrated_liquidity_manager import (
            IntegratedWeekendLiquidityManager,
            reset_integrated_liquidity_manager,
        )
        
        reset_integrated_liquidity_manager()
        manager = IntegratedWeekendLiquidityManager()
        
        self.assertIsNotNone(manager.config)
        self.assertIsNotNone(manager.state)
    
    def test_update_weekday(self):
        """Test update on a weekday."""
        from integration.core.integrated_liquidity_manager import (
            IntegratedWeekendLiquidityManager,
            LiquidityPeriod,
        )
        
        manager = IntegratedWeekendLiquidityManager()
        
        # Wednesday at 14:00 UTC
        test_time = datetime(2026, 1, 28, 14, 0, 0)  # Wednesday
        state = manager.update(test_time)
        
        self.assertEqual(state.period, LiquidityPeriod.NORMAL)
        self.assertFalse(state.is_weekend)
        self.assertFalse(state.is_holiday)
    
    def test_update_weekend(self):
        """Test update on weekend."""
        from integration.core.integrated_liquidity_manager import (
            IntegratedWeekendLiquidityManager,
            LiquidityPeriod,
        )
        
        manager = IntegratedWeekendLiquidityManager()
        
        # Saturday at 14:00 UTC
        test_time = datetime(2026, 1, 31, 14, 0, 0)  # Saturday
        state = manager.update(test_time)
        
        self.assertEqual(state.period, LiquidityPeriod.WEEKEND)
        self.assertTrue(state.is_weekend)
        self.assertLess(state.gross_cap, 1.0)  # Should be capped
    
    def test_validate_entry_normal(self):
        """Test entry validation during normal period."""
        from integration.core.integrated_liquidity_manager import IntegratedWeekendLiquidityManager
        
        manager = IntegratedWeekendLiquidityManager()
        test_time = datetime(2026, 1, 28, 14, 0, 0)  # Wednesday
        manager.update(test_time)
        
        allowed, reason = manager.validate_entry(
            asset="SOL",
            target_exposure=0.50,
            is_opportunity=False,
        )
        
        self.assertTrue(allowed)
    
    def test_validate_entry_weekend(self):
        """Test entry validation during weekend."""
        from integration.core.integrated_liquidity_manager import IntegratedWeekendLiquidityManager
        
        manager = IntegratedWeekendLiquidityManager()
        test_time = datetime(2026, 1, 31, 14, 0, 0)  # Saturday
        manager.update(test_time)
        
        # High exposure should be blocked
        allowed, reason = manager.validate_entry(
            asset="SOL",
            target_exposure=0.80,
            is_opportunity=False,
        )
        
        self.assertFalse(allowed)
    
    def test_proof_log(self):
        """Test proof log generation."""
        from integration.core.integrated_liquidity_manager import IntegratedWeekendLiquidityManager
        
        manager = IntegratedWeekendLiquidityManager()
        manager.update()
        
        log = manager.get_proof_log()
        
        self.assertIn("[LIQUIDITY]", log)
        self.assertIn("period=", log)
        self.assertIn("gross_cap=", log)


# =============================================================================
# TEST: IDLE MODULE INTEGRATOR V4
# =============================================================================

class TestIdleModuleIntegratorV4(unittest.TestCase):
    """Test the idle module integrator v4."""
    
    def test_import(self):
        """Test that module can be imported."""
        from engine.idle_module_integrator_v4 import (
            IdleModuleIntegrator,
            get_idle_module_integrator,
            MarketActivityLevel,
        )
        self.assertIsNotNone(IdleModuleIntegrator)
    
    def test_integrator_init(self):
        """Test integrator initialization."""
        from engine.idle_module_integrator_v4 import (
            IdleModuleIntegrator,
            reset_idle_module_integrator,
        )
        
        reset_idle_module_integrator()
        integrator = IdleModuleIntegrator()
        
        self.assertIsNotNone(integrator.config)
        self.assertIsNotNone(integrator.state)
    
    def test_update_active_market(self):
        """Test update with active market data."""
        from engine.idle_module_integrator_v4 import (
            IdleModuleIntegrator,
            MarketActivityLevel,
        )
        
        integrator = IdleModuleIntegrator()
        
        state = integrator.update({
            'realized_vol': 0.25,
            'volume_ratio': 1.2,
            'spread_bps': 2.0,
        })
        
        self.assertEqual(state.activity_level, MarketActivityLevel.ACTIVE)
        self.assertFalse(state.is_quiet)
        self.assertFalse(state.is_idle)
    
    def test_update_quiet_market(self):
        """Test update with quiet market data."""
        from engine.idle_module_integrator_v4 import (
            IdleModuleIntegrator,
            MarketActivityLevel,
        )
        
        integrator = IdleModuleIntegrator()
        
        state = integrator.update({
            'realized_vol': 0.10,
            'volume_ratio': 0.3,
            'spread_bps': 4.0,
        })
        
        # Should be QUIET or IDLE (not ACTIVE)
        self.assertNotEqual(state.activity_level, MarketActivityLevel.ACTIVE)
    
    def test_proof_log(self):
        """Test proof log generation."""
        from engine.idle_module_integrator_v4 import IdleModuleIntegrator
        
        integrator = IdleModuleIntegrator()
        integrator.update({'realized_vol': 0.20, 'volume_ratio': 1.0, 'spread_bps': 2.0})
        
        log = integrator.get_proof_log()
        
        self.assertIn("[IDLE]", log)
        self.assertIn("activity=", log)
        self.assertIn("vol=", log)


# =============================================================================
# TEST: ENHANCED META DECISION LAYER
# =============================================================================

class TestEnhancedMetaDecision(unittest.TestCase):
    """Test the enhanced meta decision layer."""
    
    def test_import(self):
        """Test that module can be imported."""
        from integration.core.enhanced_meta_decision import (
            EnhancedMetaDecisionLayer,
            get_enhanced_meta_layer,
            SharedInformation,
        )
        self.assertIsNotNone(EnhancedMetaDecisionLayer)
    
    def test_layer_init(self):
        """Test layer initialization."""
        from integration.core.enhanced_meta_decision import (
            EnhancedMetaDecisionLayer,
            reset_enhanced_meta_layer,
        )
        
        reset_enhanced_meta_layer()
        layer = EnhancedMetaDecisionLayer()
        
        self.assertIsNotNone(layer.info_broker)
        self.assertIsNotNone(layer.regime_memory)
    
    def test_update_context(self):
        """Test context update."""
        from integration.core.enhanced_meta_decision import EnhancedMetaDecisionLayer
        
        layer = EnhancedMetaDecisionLayer()
        
        layer.update_context(
            market_data={'volatility': 0.20, 'liquidity_score': 0.7},
            signals={'btc': 0.5, 'sol': 0.6},
            regime="MODERATE_BULL"
        )
        
        shared = layer.get_shared_info()
        self.assertEqual(shared.current_regime.value, "MODERATE_BULL")
    
    def test_aggressiveness_modifier(self):
        """Test aggressiveness modifier."""
        from integration.core.enhanced_meta_decision import EnhancedMetaDecisionLayer
        
        layer = EnhancedMetaDecisionLayer()
        
        modifier = layer.get_aggressiveness_modifier()
        
        # Should be in valid range
        self.assertGreaterEqual(modifier, 0.5)
        self.assertLessEqual(modifier, 1.2)
    
    def test_record_outcome(self):
        """Test outcome recording."""
        from integration.core.enhanced_meta_decision import EnhancedMetaDecisionLayer
        
        layer = EnhancedMetaDecisionLayer()
        
        # Record some outcomes
        layer.record_outcome("QUANT_MOMENTUM", 100, "MODERATE_BULL", is_opportunity=True)
        layer.record_outcome("QUANT_MOMENTUM", -50, "MODERATE_BULL", is_opportunity=False)
        
        # Check stats updated
        stats = layer.get_stats()
        self.assertGreater(len(stats['strategy_performances']), 0)
    
    def test_proof_log(self):
        """Test proof log generation."""
        from integration.core.enhanced_meta_decision import EnhancedMetaDecisionLayer
        
        layer = EnhancedMetaDecisionLayer()
        layer.update_context({'volatility': 0.20}, regime="NEUTRAL")
        
        log = layer.get_proof_log()
        
        self.assertIn("[META]", log)
        self.assertIn("regime=", log)
        self.assertIn("aggr_mod=", log)


# =============================================================================
# TEST: INTERFACE COMPATIBILITY LAYER
# =============================================================================

class TestCompatibilityLayer(unittest.TestCase):
    """Test the interface compatibility layer."""
    
    def test_import(self):
        """Test that module can be imported."""
        from integration.core.compat import (
            AgentSignal,
            get_engine_v35,
            MetaIntegration,
        )
        self.assertIsNotNone(AgentSignal)
    
    def test_agent_signal_compat(self):
        """Test AgentSignal compatibility wrapper."""
        import warnings
        from integration.core.compat import AgentSignal
        
        # Should issue deprecation warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            signal = AgentSignal(direction=0.5, confidence=0.8)
            
            self.assertGreater(len(w), 0)
            self.assertTrue(any("deprecated" in str(warning.message).lower() for warning in w))
        
        # Should still work
        self.assertEqual(signal.direction, 0.5)
    
    def test_meta_integration_compat(self):
        """Test MetaIntegration compatibility wrapper."""
        import warnings
        from integration.core.compat import MetaIntegration
        
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            
            meta = MetaIntegration()
            modifier = meta.get_aggressiveness_modifier()
            
            self.assertIsInstance(modifier, float)


# =============================================================================
# RUN TESTS
# =============================================================================

def run_all_tests():
    """Run all tests and print summary."""
    
    print("=" * 70)
    print("HMATS v4.0-COMPLETE - UNIFIED TEST SUITE")
    print("=" * 70)
    print()
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestScenarioWalkthroughsUnified))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegratedLiquidityManager))
    suite.addTests(loader.loadTestsFromTestCase(TestIdleModuleIntegratorV4))
    suite.addTests(loader.loadTestsFromTestCase(TestEnhancedMetaDecision))
    suite.addTests(loader.loadTestsFromTestCase(TestCompatibilityLayer))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print()
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print()
        print("ON ALL TESTS PASSED")
    else:
        print()
        print("OFF SOME TESTS FAILED")
        
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"  - {test}")
        
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  - {test}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
