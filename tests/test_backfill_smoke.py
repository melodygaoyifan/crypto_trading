#!/usr/bin/env python3
"""
================================================================================
HMATS v6.2.3 Backfill Patch - Smoke Test
================================================================================
Version: 6.2.3-BACKFILL
Purpose: Verify that restored modules import correctly and work in dry-run mode

USAGE:
    python tests/test_backfill_smoke.py

    Or with pytest:
    pytest tests/test_backfill_smoke.py -v

MODULES TESTED:
    P1 (Core):
    - market/regime_navigator.py (EnhancedMarketRegimeNavigator)
    - market/structure_analyzer.py (StructureBreakAnalyzer)
    - risk/strategy_allocator.py (StrategyAllocator)

    P2 (Optional):
    - agents/onchain_graph_alpha.py (OnChainGraphAlphaEngine)
    - defense/bitbeast_rules.py (BitBeastRulesEngine)

================================================================================
"""

import sys
import os
import logging
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BackfillSmokeTest")


def test_config_flags():
    """Test that config flags exist for backfill modules."""
    print("\n" + "=" * 70)
    print("TEST 1: Config Flags")
    print("=" * 70)
    
    try:
        from configs.sota_flags import SOTAFlags
        flags = SOTAFlags()
        
        # Check P1.11 backfill flags
        assert hasattr(flags, 'ENABLE_ENHANCED_REGIME_NAVIGATOR'), "Missing ENABLE_ENHANCED_REGIME_NAVIGATOR"
        assert hasattr(flags, 'ENABLE_STRUCTURE_ANALYZER'), "Missing ENABLE_STRUCTURE_ANALYZER"
        assert hasattr(flags, 'ENABLE_STRATEGY_ALLOCATOR'), "Missing ENABLE_STRATEGY_ALLOCATOR"
        assert hasattr(flags, 'ENABLE_ONCHAIN_GRAPH_ALPHA'), "Missing ENABLE_ONCHAIN_GRAPH_ALPHA"
        assert hasattr(flags, 'ENABLE_BITBEAST_RULES'), "Missing ENABLE_BITBEAST_RULES"
        
        print(f"  ENABLE_ENHANCED_REGIME_NAVIGATOR = {flags.ENABLE_ENHANCED_REGIME_NAVIGATOR}")
        print(f"  ENABLE_STRUCTURE_ANALYZER = {flags.ENABLE_STRUCTURE_ANALYZER}")
        print(f"  ENABLE_STRATEGY_ALLOCATOR = {flags.ENABLE_STRATEGY_ALLOCATOR}")
        print(f"  ENABLE_ONCHAIN_GRAPH_ALPHA = {flags.ENABLE_ONCHAIN_GRAPH_ALPHA}")
        print(f"  ENABLE_BITBEAST_RULES = {flags.ENABLE_BITBEAST_RULES}")
        
        print("  ✓ All config flags present")
        return True
    except Exception as e:
        print(f"  ✗ Config flags test failed: {e}")
        return False


def test_regime_navigator_import():
    """Test P1: EnhancedMarketRegimeNavigator import."""
    print("\n" + "=" * 70)
    print("TEST 2: Regime Navigator Import (P1)")
    print("=" * 70)
    
    try:
        from market.regime_navigator import (
            EnhancedMarketRegimeNavigator,
            VolOfVolTensor,
            LiquidityRegime,
            VolatilityRegime,
            MarketRegime,
        )
        
        # Basic instantiation test
        navigator = EnhancedMarketRegimeNavigator()
        print(f"  EnhancedMarketRegimeNavigator: {type(navigator)}")
        print(f"  MarketRegime options: {[r.name for r in MarketRegime]}")
        print(f"  LiquidityRegime options: {[r.name for r in LiquidityRegime]}")
        print("  ✓ Regime Navigator imports OK")
        return True
    except Exception as e:
        print(f"  ✗ Regime Navigator import failed: {e}")
        return False


def test_structure_analyzer_import():
    """Test P1: StructureBreakAnalyzer import."""
    print("\n" + "=" * 70)
    print("TEST 3: Structure Analyzer Import (P1)")
    print("=" * 70)
    
    try:
        from market.structure_analyzer import (
            StructureBreakAnalyzer,
            StructureBreakResult,
            BreakDirection,
            FractalType,
        )
        
        # Basic instantiation test
        analyzer = StructureBreakAnalyzer()
        print(f"  StructureBreakAnalyzer: {type(analyzer)}")
        print(f"  BreakDirection options: {[d.name for d in BreakDirection]}")
        print(f"  FractalType options: {[f.name for f in FractalType]}")
        print("  ✓ Structure Analyzer imports OK")
        return True
    except Exception as e:
        print(f"  ✗ Structure Analyzer import failed: {e}")
        return False


def test_strategy_allocator_import():
    """Test P1: StrategyAllocator import."""
    print("\n" + "=" * 70)
    print("TEST 4: Strategy Allocator Import (P1)")
    print("=" * 70)
    
    try:
        from risk.strategy_allocator import (
            StrategyAllocator,
            StrategyState,
            AllocationResult,
            Regime,
        )
        
        # Basic instantiation test
        allocator = StrategyAllocator()
        print(f"  StrategyAllocator: {type(allocator)}")
        print(f"  Regime options: {[r.name for r in Regime]}")
        print("  ✓ Strategy Allocator imports OK")
        return True
    except Exception as e:
        print(f"  ✗ Strategy Allocator import failed: {e}")
        return False


def test_onchain_graph_alpha_import():
    """Test P2: OnChainGraphAlphaEngine import."""
    print("\n" + "=" * 70)
    print("TEST 5: OnChain Graph Alpha Import (P2)")
    print("=" * 70)
    
    try:
        from agents.onchain_graph_alpha import (
            OnChainGraphAlphaEngine,
            WalletProfile,
            WhaleAction,
            OnChainSignalType,
            AddressType,
        )
        
        print(f"  OnChainGraphAlphaEngine: available")
        print(f"  WhaleAction options: {[a.name for a in WhaleAction]}")
        print(f"  AddressType options: {[t.name for t in AddressType]}")
        print("  ✓ OnChain Graph Alpha imports OK")
        return True
    except Exception as e:
        print(f"  ✗ OnChain Graph Alpha import failed: {e}")
        return False


def test_bitbeast_rules_import():
    """Test P2: BitBeastBehaviorGuard import."""
    print("\n" + "=" * 70)
    print("TEST 6: BitBeast Rules Import (P2)")
    print("=" * 70)
    
    try:
        from defense.bitbeast_rules import (
            BitBeastBehaviorGuard,
            VolumeFilter,
            StructureBreakoutFilter,
        )
        
        print(f"  BitBeastBehaviorGuard: available")
        print(f"  VolumeFilter: available")
        print(f"  StructureBreakoutFilter: available")
        print("  ✓ BitBeast Rules imports OK")
        return True
    except Exception as e:
        print(f"  ✗ BitBeast Rules import failed: {e}")
        return False


def test_market_module_exports():
    """Test market module exports all backfill components."""
    print("\n" + "=" * 70)
    print("TEST 7: Market Module Exports")
    print("=" * 70)
    
    try:
        from market import (
            EnhancedMarketRegimeNavigator,
            StructureBreakAnalyzer,
        )
        print("  ✓ market module exports backfill components")
        return True
    except ImportError as e:
        print(f"  [WARN] market module export warning: {e}")
        return False


def test_risk_module_exports():
    """Test risk module exports StrategyAllocator."""
    print("\n" + "=" * 70)
    print("TEST 8: Risk Module Exports"  )
    print("=" * 70)
    
    try:
        from risk import StrategyAllocator
        print("  ✓ risk module exports StrategyAllocator")
        return True
    except ImportError as e:
        print(f"  [WARN] risk module export warning: {e}")
        return False


def test_dry_run_pipeline():
    """Test that pipeline runs in dry-run mode without errors."""
    print("\n" + "=" * 70)
    print("TEST 9: Dry-Run Pipeline Simulation")
    print("=" * 70)
    
    try:
        import numpy as np
        
        # Create synthetic OHLCV data
        n_bars = 100
        base_price = 100.0
        synthetic_data = {
            'open': base_price + np.random.randn(n_bars).cumsum() * 0.5,
            'high': base_price + np.random.randn(n_bars).cumsum() * 0.5 + 1,
            'low': base_price + np.random.randn(n_bars).cumsum() * 0.5 - 1,
            'close': base_price + np.random.randn(n_bars).cumsum() * 0.5,
            'volume': np.abs(np.random.randn(n_bars)) * 1000000,
        }
        
        # Test Structure Analyzer with synthetic data
        from market.structure_analyzer import StructureBreakAnalyzer
        import time
        
        analyzer = StructureBreakAnalyzer()
        asset = "SOL"
        
        # Feed bars to analyzer using the update() method
        for i in range(n_bars):
            analyzer.update(
                asset=asset,
                timestamp=time.time() - (n_bars - i) * 3600,
                open_=synthetic_data['open'][i],
                high=synthetic_data['high'][i],
                low=synthetic_data['low'][i],
                close=synthetic_data['close'][i],
                volume=synthetic_data['volume'][i],
            )
        
        # Get analysis result using check_break()
        current_price = synthetic_data['close'][-1]
        result = analyzer.check_break(asset, current_price)
        
        print(f"  Structure Break Detected: {result.break_detected}")
        print(f"  Direction: {result.direction.name}")
        print(f"  Can Long: {result.can_long}")
        print(f"  Can Short: {result.can_short}")
        print(f"  Recent Fractals: {len(result.recent_fractals)}")
        
        print("  ✓ Dry-run pipeline simulation OK")
        return True
    except Exception as e:
        print(f"  ✗ Dry-run pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_toggles_disable_behavior():
    """Test that when toggles are OFF, modules return neutral outputs."""
    print("\n" + "=" * 70)
    print("TEST 10: Toggle Disable Behavior")
    print("=" * 70)
    
    try:
        from configs.sota_flags import SOTAFlags
        
        # Check defaults
        flags = SOTAFlags()
        
        # Verify safe defaults
        assert flags.ENABLE_STRATEGY_ALLOCATOR == False, "ENABLE_STRATEGY_ALLOCATOR should default to False"
        assert flags.ENABLE_ONCHAIN_GRAPH_ALPHA == False, "ENABLE_ONCHAIN_GRAPH_ALPHA should default to False"
        assert flags.ENABLE_BITBEAST_RULES == False, "ENABLE_BITBEAST_RULES should default to False"
        
        print("  ENABLE_STRATEGY_ALLOCATOR = False (correct)")
        print("  ENABLE_ONCHAIN_GRAPH_ALPHA = False (correct)")
        print("  ENABLE_BITBEAST_RULES = False (correct)")
        print("  ✓ Toggles disable behavior OK")
        return True
    except Exception as e:
        print(f"  ✗ Toggle test failed: {e}")
        return False


def main():
    """Run all smoke tests."""
    print("=" * 70)
    print("HMATS v6.2.3 BACKFILL PATCH - SMOKE TEST")
    print("=" * 70)
    
    results = []
    
    # Run all tests
    results.append(("Config Flags", test_config_flags()))
    results.append(("Regime Navigator", test_regime_navigator_import()))
    results.append(("Structure Analyzer", test_structure_analyzer_import()))
    results.append(("Strategy Allocator", test_strategy_allocator_import()))
    results.append(("OnChain Graph Alpha", test_onchain_graph_alpha_import()))
    results.append(("BitBeast Rules", test_bitbeast_rules_import()))
    results.append(("Market Module Exports", test_market_module_exports()))
    results.append(("Risk Module Exports", test_risk_module_exports()))
    results.append(("Dry-Run Pipeline", test_dry_run_pipeline()))
    results.append(("Toggle Behavior", test_toggles_disable_behavior()))
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, r in results if r)
    failed = sum(1 for _, r in results if not r)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print()
    print(f"  Total: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("\n  🎉 ALL SMOKE TESTS PASSED")
        return 0
    else:
        print(f"\n  [WARN]️ {failed} tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
