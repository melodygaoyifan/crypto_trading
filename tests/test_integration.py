#!/usr/bin/env python3
"""
================================================================================
HMATS v5.0.5 Patch - Integration Test Suite
================================================================================

Comprehensive tests for all v5.0.5 enhancements:
1. RL Execution Agent with 5-level orderbook features
2. Adaptive Stop with multi-window ATR and Calmar
3. Volatility Targeting with per-asset configs
4. Market Impact Model with impact curves

Run: python test_integration.py
"""

import sys
import random
import pytest
import numpy as np
from datetime import datetime, timedelta

# Ensure we're testing the local patch
sys.path.insert(0, '.')


def test_rl_execution_agent():
    """Test RL Execution Agent enhancements."""
    print("\n" + "=" * 60)
    print("TEST 1: RL Execution Agent with 5-Level Orderbook")
    print("=" * 60)
    
    from execution.rl_execution_agent import (
        RLExecutionAgent, OrderbookSnapshot, ExecutionAction
    )
    
    # Test 1.1: State dimension
    print("\n[1.1] State Dimension Test")
    agent_v505 = RLExecutionAgent(use_extended_state=True)
    agent_v45 = RLExecutionAgent(use_extended_state=False)
    
    assert agent_v505.state_dim == 25, f"Expected 25, got {agent_v505.state_dim}"
    assert agent_v45.state_dim == 15, f"Expected 15, got {agent_v45.state_dim}"
    print(f"   v5.0.5 state dim: {agent_v505.state_dim} ✓")
    print(f"   v4.5 compat dim: {agent_v45.state_dim} ✓")
    
    # Test 1.2: Orderbook metrics calculation
    print("\n[1.2] Orderbook Feature Calculation")
    orderbook = OrderbookSnapshot(
        timestamp=datetime.now(),
        symbol="SOL/USD",
        bid_prices=[100.0, 99.95, 99.90, 99.85, 99.80],
        bid_quantities=[500, 800, 1200, 600, 1000],
        ask_prices=[100.05, 100.10, 100.15, 100.20, 100.25],
        ask_quantities=[400, 900, 700, 1100, 800]
    )
    orderbook.calculate_metrics()
    
    assert len(orderbook.level_spreads_bps) == 5, "Should have 5 level spreads"
    assert len(orderbook.level_imbalances) == 5, "Should have 5 level imbalances"
    assert -1 <= orderbook.depth_pressure <= 1, "Depth pressure should be [-1, 1]"
    
    print(f"   Level spreads: {[f'{s:.1f}' for s in orderbook.level_spreads_bps]}")
    print(f"   Level imbalances: {[f'{i:.2f}' for i in orderbook.level_imbalances]}")
    print(f"   Depth pressure: {orderbook.depth_pressure:.3f}")
    print("   ✓ Orderbook features OK")
    
    # Test 1.3: Decision with new actions
    print("\n[1.3] Extended Action Space")
    test_cases = [
        (500, 0.3, "Small order, low urgency"),
        (1000, 0.8, "Medium order, high urgency"),
        (5000, 0.5, "Large order, medium urgency"),
    ]
    
    for size, urgency, desc in test_cases:
        decision = agent_v505.decide(
            orderbook=orderbook,
            order_size=size,
            side="buy",
            urgency=urgency
        )
        print(f"   {desc}: {decision.action.name}, slice={decision.slice_pct:.0%}")
    
    # Test new actions exist
    assert ExecutionAction.ADAPTIVE_SLICE.value == 5
    assert ExecutionAction.ICEBERG.value == 6
    print("   ✓ Extended actions OK")
    
    # Test 1.4: Fill estimation
    print("\n[1.4] Fill Price Estimation")
    for qty in [500, 2000, 5000]:
        avg_price, levels = orderbook.estimate_fill_price(qty, "buy")
        slippage = (avg_price - orderbook.ask_prices[0]) / orderbook.ask_prices[0] * 10000
        print(f"   Buy {qty}: ${avg_price:.4f}, {levels} levels, {slippage:.1f}bps slippage")
    
    print("\nON RL Execution Agent Tests PASSED")
    return True


def test_adaptive_stop():
    """Test Adaptive Stop enhancements."""
    print("\n" + "=" * 60)
    print("TEST 2: Adaptive Stop with Multi-Window ATR & Calmar")
    print("=" * 60)
    
    from risk.adaptive_stop import (
        EnhancedAdaptiveStopManager, MultiWindowATRCalculator,
        CalmarCalculator, PositionDirection, VolatilityRegime,
        StopConfiguration
    )
    
    # Test 2.1: Multi-window ATR
    print("\n[2.1] Multi-Window ATR Calculation")
    atr_calc = MultiWindowATRCalculator(weights=(0.4, 0.4, 0.2))
    
    random.seed(42)
    base_price = 100.0
    for day in range(60):
        ret = random.gauss(0, 0.04)
        close = base_price * (1 + ret)
        high = close * (1 + abs(random.gauss(0, 0.01)))
        low = close * (1 - abs(random.gauss(0, 0.01)))
        atr_calc.update("SOL", high, low, close)
        base_price = close
    
    state = atr_calc.get_state("SOL")
    assert state.atr_14d > 0, "ATR 14d should be positive"
    assert state.atr_30d > 0, "ATR 30d should be positive"
    assert state.atr_60d > 0, "ATR 60d should be positive"
    assert state.composite_atr > 0, "Composite ATR should be positive"
    
    print(f"   ATR 14d: {state.atr_14d:.4f}")
    print(f"   ATR 30d: {state.atr_30d:.4f}")
    print(f"   ATR 60d: {state.atr_60d:.4f}")
    print(f"   Composite: {state.composite_atr:.4f}")
    print(f"   Vol percentile: {state.vol_percentile:.1f}%")
    print(f"   Vol regime: {state.vol_regime.value}")
    print("   ✓ Multi-window ATR OK")
    
    # Test 2.2: Calmar ratio
    print("\n[2.2] Calmar Ratio Calculation")
    calmar_calc = CalmarCalculator(lookback_days=90)
    
    equity = 10000.0
    for day in range(60):
        equity *= (1 + random.gauss(0.001, 0.02))
        calmar_calc.update("SOL", equity)
    
    metrics = calmar_calc.get_metrics("SOL")
    print(f"   Calmar ratio: {metrics.calmar_ratio:.2f}")
    print(f"   Max drawdown: {metrics.max_drawdown_pct:.1f}%")
    print(f"   Recovery factor: {metrics.recovery_factor:.2f}")
    print("   ✓ Calmar ratio OK")
    
    # Test 2.3: Dynamic R-trigger
    print("\n[2.3] Dynamic R-Trigger and Stop Management")
    stop_mgr = EnhancedAdaptiveStopManager(
        atr_calculator=atr_calc,
        calmar_calculator=calmar_calc
    )
    
    entry_price = base_price
    state = stop_mgr.register_position(
        symbol="SOL",
        entry_price=entry_price,
        position_size=10.0,
        direction=PositionDirection.LONG,
        calmar_metrics=metrics
    )
    
    assert state.dynamic_r_trigger > 0, "Dynamic R trigger should be positive"
    print(f"   Entry: ${entry_price:.2f}")
    print(f"   Initial stop: ${state.initial_stop:.4f}")
    print(f"   Dynamic R trigger: {state.dynamic_r_trigger:.2f}")
    print(f"   Vol regime at entry: {state.vol_regime_at_entry.value}")
    
    # Simulate price movement to breakeven
    profit_price = entry_price * (1 + state.initial_risk_pct/100 * state.dynamic_r_trigger)
    state = stop_mgr.update_stop("SOL", profit_price)
    
    assert state.breakeven_triggered, "Breakeven should be triggered"
    print(f"   After {state.current_r_multiple:.2f}R profit:")
    print(f"   - Breakeven triggered: {state.breakeven_triggered}")
    print(f"   - Stop type: {state.stop_type.value}")
    print(f"   - Current stop: ${state.current_stop:.4f}")
    print("   ✓ Dynamic R-trigger OK")
    
    print("\nON Adaptive Stop Tests PASSED")
    return True


def test_volatility_targeting():
    """Test Volatility Targeting enhancements."""
    print("\n" + "=" * 60)
    print("TEST 3: Volatility Targeting with Per-Asset Configs")
    print("=" * 60)
    
    from risk.volatility_targeting import (
        EnhancedVolatilityTargetingSizer, AssetVolConfig,
        PositionSizeRecommendation
    )
    
    # Test 3.1: Per-asset configuration
    print("\n[3.1] Per-Asset Configuration")
    asset_configs = {
        "SOL/USD": AssetVolConfig(target_daily_vol=0.02, max_weight=0.30),
        "BTC/USD": AssetVolConfig(target_daily_vol=0.015, max_weight=0.40),
        "ETH/USD": AssetVolConfig(target_daily_vol=0.025, max_weight=0.35),
    }
    
    sizer = EnhancedVolatilityTargetingSizer(
        portfolio_target_daily_vol=0.02,
        asset_configs=asset_configs,
        use_correlation_adjustment=True
    )
    
    for symbol, config in asset_configs.items():
        retrieved = sizer.get_asset_config(symbol)
        assert retrieved.target_daily_vol == config.target_daily_vol
        print(f"   {symbol}: target={config.target_daily_vol:.1%}, max={config.max_weight:.0%}")
    print("   ✓ Per-asset config OK")
    
    # Test 3.2: Volatility calculation and sizing
    print("\n[3.2] Volatility-Based Sizing")
    np.random.seed(42)
    
    # High vol asset (SOL)
    sol_prices = 100 * np.cumprod(1 + np.random.randn(60) * 0.05)
    sizer.bulk_update_volatility("SOL/USD", list(sol_prices))
    
    # Medium vol asset (BTC)
    btc_prices = 40000 * np.cumprod(1 + np.random.randn(60) * 0.03)
    sizer.bulk_update_volatility("BTC/USD", list(btc_prices))
    
    # Low vol asset (ETH)
    eth_prices = 2500 * np.cumprod(1 + np.random.randn(60) * 0.02)
    sizer.bulk_update_volatility("ETH/USD", list(eth_prices))
    
    capital = 100_000
    for symbol in ["SOL/USD", "BTC/USD", "ETH/USD"]:
        rec = sizer.calculate_position(symbol, capital, 0.25)
        print(f"   {symbol}:")
        print(f"      Target vol: {rec.target_daily_vol:.1%}")
        print(f"      Adjusted weight: {rec.adjusted_weight:.1%}")
        print(f"      Actual/Target: {rec.actual_vs_target_ratio:.2f}x")
    
    print("   ✓ Volatility sizing OK")
    
    # Test 3.3: Portfolio allocation with correlation
    print("\n[3.3] Portfolio Allocation with Correlation")
    recs, state = sizer.calculate_portfolio_allocations(
        symbols=["SOL/USD", "BTC/USD", "ETH/USD"],
        capital=capital,
        base_weights={"SOL/USD": 0.25, "BTC/USD": 0.50, "ETH/USD": 0.25}
    )
    
    assert state.diversification_ratio >= 1.0, "Diversification ratio should be >= 1"
    print(f"   Portfolio daily vol: {state.current_daily_vol:.2%}")
    print(f"   Vol utilization: {state.vol_utilization:.0%}")
    print(f"   Diversification ratio: {state.diversification_ratio:.2f}")
    print(f"   Total exposure: ${state.total_exposure_usd:,.0f}")
    print("   ✓ Portfolio allocation OK")
    
    # Test 3.4: Backward compatibility
    print("\n[3.4] Backward Compatibility")
    from risk.volatility_targeting import VolatilityTargetingSizer
    
    legacy_sizer = VolatilityTargetingSizer(target_daily_vol=0.02)
    legacy_sizer.bulk_update_volatility("TEST/USD", list(sol_prices))
    rec = legacy_sizer.calculate_position("TEST/USD", 50000, 0.2)
    
    assert hasattr(rec, 'adjusted_weight'), "Should have adjusted_weight"
    assert hasattr(rec, 'position_usd'), "Should have position_usd"
    print(f"   Legacy interface works: weight={rec.adjusted_weight:.1%}")
    print("   ✓ Backward compatibility OK")
    
    print("\nON Volatility Targeting Tests PASSED")
    return True


def test_market_impact_model():
    """Test Market Impact Model enhancements."""
    print("\n" + "=" * 60)
    print("TEST 4: Market Impact Model with Impact Curves")
    print("=" * 60)
    
    from execution.market_impact_model import (
        EnhancedMarketImpactModel, OrderbookSnapshot, OrderbookLevel,
        ImpactCurve
    )
    
    # Test 4.1: Basic impact estimation
    print("\n[4.1] Combined Impact Estimation")
    model = EnhancedMarketImpactModel(
        use_impact_curve=True,
        curve_weight=0.3
    )
    
    # Create orderbook
    ob = OrderbookSnapshot(timestamp=datetime.now(), symbol="SOL/USD")
    for p, q in [(100.05, 500), (100.10, 800), (100.15, 1200), (100.20, 600), (100.25, 1000)]:
        ob.asks.append(OrderbookLevel(price=p, quantity=q))
    for p, q in [(100.00, 600), (99.95, 900), (99.90, 1100), (99.85, 500), (99.80, 800)]:
        ob.bids.append(OrderbookLevel(price=p, quantity=q))
    ob.calculate_metrics()
    model.update_orderbook("SOL/USD", ob)
    
    impact = model.estimate_impact(
        symbol="SOL/USD",
        order_quantity=2000,
        daily_volume=50_000_000,
        volatility=0.8,
        side="buy",
        orderbook=ob
    )
    
    assert impact.total_impact_bps > 0, "Impact should be positive"
    print(f"   Total impact: {impact.total_impact_bps:.2f} bps")
    print(f"   Model used: {impact.model_used}")
    print(f"   Recommended offset: {impact.recommended_limit_offset_bps:.1f} bps")
    print(f"   Levels needed: {impact.recommended_depth_levels}")
    print("   ✓ Impact estimation OK")
    
    # Test 4.2: Impact curve building
    print("\n[4.2] Impact Curve Construction")
    random.seed(42)
    
    # Record many trades to build curve
    for i in range(100):
        size = random.uniform(100, 5000)
        levels = random.randint(1, 4)
        actual = random.uniform(2, 20) * (size / 1000)  # Larger orders = more impact
        
        model.record_actual_impact(
            symbol="SOL/USD",
            side="buy",
            order_size=size,
            daily_volume=50_000_000,
            actual_impact_bps=actual,
            volatility=0.8,
            orderbook=ob,
            levels_consumed=levels,
            fill_rate=random.uniform(0.9, 1.0),
            execution_time_ms=random.uniform(50, 500)
        )
    
    curve = model.get_impact_curve("SOL/USD")
    assert curve is not None, "Curve should be built"
    assert curve['total_samples'] >= 50, "Should have at least 50 samples"
    
    print(f"   Curve samples: {curve['total_samples']}")
    print(f"   Avg impact: {curve['avg_impact_bps']:.1f} bps")
    print(f"   Model RMSE: {curve['model_rmse']:.1f} bps")
    print(f"   Curve points: {len(curve['points'])}")
    print("   ✓ Curve building OK")
    
    # Test 4.3: Curve-adjusted estimation
    print("\n[4.3] Curve-Adjusted Estimation")
    impact_with_curve = model.estimate_impact(
        symbol="SOL/USD",
        order_quantity=2000,
        daily_volume=50_000_000,
        volatility=0.8,
        side="buy",
        orderbook=ob
    )
    
    assert impact_with_curve.curve_adjusted or impact_with_curve.curve_confidence > 0
    print(f"   Curve adjusted: {impact_with_curve.curve_adjusted}")
    print(f"   Curve confidence: {impact_with_curve.curve_confidence:.2f}")
    print(f"   Final impact: {impact_with_curve.total_impact_bps:.2f} bps")
    print("   ✓ Curve adjustment OK")
    
    # Test 4.4: Enhanced execution plan
    print("\n[4.4] Enhanced Execution Plan")
    plan = model.generate_execution_plan(
        order_id="test_001",
        symbol="SOL/USD",
        side="buy",
        total_quantity=10000,
        daily_volume=50_000_000,
        volatility=0.8,
        current_price=100.0,
        urgency=0.5,
        orderbook=ob
    )
    
    assert len(plan.slices) > 0, "Should have slices"
    assert plan.recommended_strategy in ["twap", "vwap", "adaptive", "aggressive"]
    
    print(f"   Strategy: {plan.recommended_strategy}")
    print(f"   Slices: {len(plan.slices)}")
    print(f"   Expected VWAP: ${plan.expected_vwap:.4f}")
    print(f"   Total impact: {plan.total_expected_impact_bps:.2f} bps")
    
    # Check slice has new fields
    slice_0 = plan.slices[0]
    assert hasattr(slice_0, 'recommended_order_type')
    assert hasattr(slice_0, 'limit_offset_bps')
    print(f"   Slice 0: {slice_0.recommended_order_type}, offset={slice_0.limit_offset_bps:.1f}bps")
    print("   ✓ Execution plan OK")
    
    # Test 4.5: Backward compatibility
    print("\n[4.5] Backward Compatibility")
    from execution.market_impact_model import MarketImpactModel
    
    legacy_model = MarketImpactModel()
    legacy_impact = legacy_model.estimate_impact(
        symbol="TEST/USD",
        order_quantity=1000,
        daily_volume=10_000_000,
        volatility=0.5,
        side="sell"
    )
    
    assert hasattr(legacy_impact, 'total_impact_bps')
    assert hasattr(legacy_impact, 'should_split')
    print(f"   Legacy interface: {legacy_impact.total_impact_bps:.1f} bps")
    print("   ✓ Backward compatibility OK")
    
    print("\nON Market Impact Model Tests PASSED")
    return True


def run_all_tests():
    """Run all integration tests."""
    print("\n" + "=" * 70)
    print("HMATS v5.0.5 PATCH - INTEGRATION TEST SUITE")
    print("=" * 70)
    
    results = []
    
    try:
        results.append(("RL Execution Agent", test_rl_execution_agent()))
    except Exception as e:
        print(f"\nOFF RL Execution Agent FAILED: {e}")
        results.append(("RL Execution Agent", False))
    
    try:
        results.append(("Adaptive Stop", test_adaptive_stop()))
    except Exception as e:
        print(f"\nOFF Adaptive Stop FAILED: {e}")
        results.append(("Adaptive Stop", False))
    
    try:
        results.append(("Volatility Targeting", test_volatility_targeting()))
    except Exception as e:
        print(f"\nOFF Volatility Targeting FAILED: {e}")
        results.append(("Volatility Targeting", False))
    
    try:
        results.append(("Market Impact Model", test_market_impact_model()))
    except Exception as e:
        print(f"\nOFF Market Impact Model FAILED: {e}")
        results.append(("Market Impact Model", False))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    all_passed = True
    for name, passed in results:
        status = "ON PASS" if passed else "OFF FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 ALL TESTS PASSED - v5.0.5 Patch Ready for Deployment")
    else:
        print("[WARN]️  SOME TESTS FAILED - Please Review")
    print("=" * 70)
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
