#!/usr/bin/env python3
"""
HMATS v5 Multi-Asset - End-to-End System Diagnostic
=====================================================
Comprehensive diagnostic covering all components.

Tests:
1. Configuration validation
2. Module imports
3. Core components
4. Training pipeline
5. Data flow
6. Signal generation
7. Portfolio management
8. Integration test

Usage: python diagnostic.py
"""

import os
import sys
import json
import time
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

# Track results
RESULTS = {
    'passed': [],
    'failed': [],
    'warnings': [],
    'skipped': []
}


def log_pass(test_name: str, message: str = ""):
    """Log passed test."""
    RESULTS['passed'].append(test_name)
    print(f"  ON {test_name}" + (f": {message}" if message else ""))


def log_fail(test_name: str, error: str):
    """Log failed test."""
    RESULTS['failed'].append((test_name, error))
    print(f"  OFF {test_name}: {error}")


def log_warn(test_name: str, message: str):
    """Log warning."""
    RESULTS['warnings'].append((test_name, message))
    print(f"  [WARN]️  {test_name}: {message}")


def log_skip(test_name: str, reason: str):
    """Log skipped test."""
    RESULTS['skipped'].append((test_name, reason))
    print(f"  ⏭️  {test_name}: {reason}")


# ============================================================================
# TEST 1: CONFIGURATION VALIDATION
# ============================================================================

def test_configuration():
    """Test configuration files and values."""
    print("\n" + "=" * 60)
    print("1️⃣  CONFIGURATION VALIDATION")
    print("=" * 60)
    
    # Check default config exists
    config_path = Path("configs/default.json")
    if not config_path.exists():
        log_fail("config_file", "configs/default.json not found")
        return
    
    log_pass("config_file", "configs/default.json exists")
    
    # Load and validate config
    try:
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        log_pass("config_parse", "JSON valid")
    except json.JSONDecodeError as e:
        log_fail("config_parse", str(e))
        return
    
    # Check required fields
    required_fields = ['risk_profile', 'initial_capital', 'assets', 'portfolio']
    for field in required_fields:
        if field in config:
            log_pass(f"config_{field}", f"Present")
        else:
            log_fail(f"config_{field}", f"Missing required field: {field}")
    
    # Validate allocation
    assets = config.get('assets', [])
    total_target = sum(a.get('target_allocation_pct', 0) for a in assets)
    
    if 0.99 <= total_target <= 1.01:
        log_pass("allocation_sum", f"Targets sum to {total_target:.0%}")
    else:
        log_warn("allocation_sum", f"Targets sum to {total_target:.0%}, expected 100%")
    
    # Check SOL is primary
    sol_config = next((a for a in assets if a['base'] == 'SOL'), None)
    if sol_config:
        sol_pct = sol_config.get('target_allocation_pct', 0)
        if sol_pct >= 0.50:
            log_pass("sol_primary", f"SOL target: {sol_pct:.0%}")
        else:
            log_warn("sol_primary", f"SOL target only {sol_pct:.0%}, expected 50%+")
    else:
        log_fail("sol_primary", "SOL not in assets")
    
    # Check no cash buffer
    portfolio = config.get('portfolio', {})
    exposure = portfolio.get('max_total_exposure', 0)
    if exposure >= 1.0:
        log_pass("no_cash", f"Max exposure: {exposure:.0%} (no cash buffer)")
    else:
        log_warn("no_cash", f"Max exposure: {exposure:.0%}, some cash reserved")


# ============================================================================
# TEST 2: MODULE IMPORTS
# ============================================================================

def test_imports():
    """Test all module imports."""
    print("\n" + "=" * 60)
    print("2️⃣  MODULE IMPORTS")
    print("=" * 60)
    
    # Core modules
    core_modules = [
        ("core.multi_asset_intelligence", "MultiAssetConfig"),
        ("core.multi_asset_intelligence", "MultiAssetHeavyModel"),
        ("core.multi_asset_intelligence", "MultiAssetLightModel"),
        ("core.multi_asset_intelligence", "PortfolioManager"),
        ("core.gpu_ta", "GPUTechnicalAnalyzer"),
    ]
    
    for module_path, class_name in core_modules:
        try:
            module = __import__(module_path, fromlist=[class_name])
            cls = getattr(module, class_name)
            log_pass(f"import_{class_name}", f"from {module_path}")
        except ImportError as e:
            log_fail(f"import_{class_name}", str(e))
        except AttributeError as e:
            log_fail(f"import_{class_name}", f"Class not found: {e}")
    
    # Training modules
    training_modules = [
        ("training.envs.multi_asset_env", "MultiAssetTradingEnv"),
        ("training.envs.multi_asset_env", "SingleAssetEnv"),
        ("training.data.multi_asset_collector", "MultiAssetDataCollector"),
    ]
    
    for module_path, class_name in training_modules:
        try:
            module = __import__(module_path, fromlist=[class_name])
            cls = getattr(module, class_name)
            log_pass(f"import_{class_name}", f"from {module_path}")
        except ImportError as e:
            log_fail(f"import_{class_name}", str(e))
        except AttributeError as e:
            log_fail(f"import_{class_name}", f"Class not found: {e}")
    
    # External dependencies
    external = [
        ("numpy", "np"),
        ("pandas", "pd"),
    ]
    
    for module_name, alias in external:
        try:
            __import__(module_name)
            log_pass(f"import_{module_name}", "Available")
        except ImportError:
            log_fail(f"import_{module_name}", "Not installed")
    
    # Optional dependencies (training)
    training_deps = [
        ("gymnasium", "DRL environments"),
    ]
    
    for module_name, purpose in training_deps:
        try:
            __import__(module_name)
            log_pass(f"optional_{module_name}", f"{purpose}")
        except ImportError:
            log_warn(f"optional_{module_name}", f"Not installed ({purpose} - install for training)")
    
    # Optional dependencies (production)
    optional = [
        ("ccxt", "Exchange API"),
        ("stable_baselines3", "DRL training"),
        ("cupy", "GPU acceleration"),
        ("openai", "GPT-4 API"),
        ("anthropic", "Claude API"),
    ]
    
    for module_name, purpose in optional:
        try:
            __import__(module_name)
            log_pass(f"optional_{module_name}", f"{purpose}")
        except ImportError:
            log_warn(f"optional_{module_name}", f"Not installed ({purpose})")


# ============================================================================
# TEST 3: CORE COMPONENTS
# ============================================================================

def test_core_components():
    """Test core component initialization."""
    print("\n" + "=" * 60)
    print("3️⃣  CORE COMPONENTS")
    print("=" * 60)
    
    try:
        from core.multi_asset_intelligence import (
            MultiAssetConfig, MultiAssetHeavyModel, MultiAssetLightModel,
            PortfolioManager, RiskProfile, create_multi_asset_config
        )
        
        # Test config creation
        config = create_multi_asset_config("aggressive", 100_000)
        log_pass("config_create", f"Capital: ${config.initial_capital:,}")
        
        # Verify SOL-focused
        sol_asset = next((a for a in config.assets if a.base == "SOL"), None)
        if sol_asset and sol_asset.max_allocation_pct >= 0.50:
            log_pass("config_sol_focus", f"SOL max: {sol_asset.max_allocation_pct:.0%}")
        else:
            log_fail("config_sol_focus", "SOL not configured as primary")
        
        # Test risk profile
        if config.risk_profile == RiskProfile.AGGRESSIVE:
            log_pass("config_risk", f"Risk profile: {config.risk_profile.value}")
        else:
            log_warn("config_risk", f"Risk profile: {config.risk_profile.value}")
        
        # Test max exposure (should be 100%)
        if config.max_total_exposure >= 1.0:
            log_pass("config_exposure", f"Max exposure: {config.max_total_exposure:.0%}")
        else:
            log_warn("config_exposure", f"Max exposure: {config.max_total_exposure:.0%}")
        
        # Test Heavy Model init (without LLM)
        heavy = MultiAssetHeavyModel(config)
        log_pass("heavy_model_init", "Initialized")
        
        # Test Light Model init
        light = MultiAssetLightModel(config)
        log_pass("light_model_init", "Initialized")
        
        # Test Portfolio Manager init
        portfolio = PortfolioManager(config)
        log_pass("portfolio_manager_init", f"Capital: ${portfolio.capital:,}")
        
    except Exception as e:
        log_fail("core_components", f"{type(e).__name__}: {e}")
        traceback.print_exc()


# ============================================================================
# TEST 4: TRAINING PIPELINE
# ============================================================================

def test_training_pipeline():
    """Test training environment and components."""
    print("\n" + "=" * 60)
    print("4️⃣  TRAINING PIPELINE")
    print("=" * 60)
    
    try:
        from training.envs.multi_asset_env import (
            MultiAssetTradingEnv, SingleAssetEnv, Action
        )
        
        # Create synthetic data
        n_rows = 1000
        dates = pd.date_range(start='2024-01-01', periods=n_rows, freq='1h')
        
        def make_ohlcv(base_price: float) -> pd.DataFrame:
            """Create synthetic OHLCV data."""
            returns = np.random.randn(n_rows) * 0.02
            prices = base_price * np.exp(np.cumsum(returns))
            
            df = pd.DataFrame({
                'open': prices * (1 + np.random.randn(n_rows) * 0.001),
                'high': prices * (1 + abs(np.random.randn(n_rows) * 0.01)),
                'low': prices * (1 - abs(np.random.randn(n_rows) * 0.01)),
                'close': prices,
                'volume': np.random.uniform(1000, 10000, n_rows),
            }, index=dates)
            
            # Add features
            df['rsi'] = 50 + np.random.randn(n_rows) * 15
            df['macd_histogram'] = np.random.randn(n_rows) * 100
            df['ema_alignment'] = np.random.choice([-1, 0, 1], n_rows)
            df['volume_ratio'] = np.random.uniform(0.5, 2, n_rows)
            df['atr_pct'] = np.random.uniform(1, 5, n_rows)
            
            return df
        
        data = {
            'SOL': make_ohlcv(200),
            'BTC': make_ohlcv(95000),
            'ETH': make_ohlcv(3200),
        }
        
        log_pass("synthetic_data", f"Created {n_rows} rows for 3 assets")
        
        # Test MultiAssetTradingEnv
        env = MultiAssetTradingEnv(
            data=data,
            initial_capital=100_000,
            assets=["SOL", "BTC", "ETH"],
            max_position_pct=0.6,
            max_total_exposure=1.0,
            mode="individual"
        )
        
        log_pass("multi_env_create", "MultiAssetTradingEnv created")
        
        # Test reset
        obs, info = env.reset()
        log_pass("multi_env_reset", f"Obs shape: {obs.shape}")
        
        # Test step
        action = np.array([1, 0, 0])  # LONG SOL, HOLD BTC, HOLD ETH
        obs, reward, terminated, truncated, info = env.step(action)
        
        log_pass("multi_env_step", f"Reward: {reward:.4f}, Equity: ${info['equity']:,.2f}")
        
        # Run a few more steps
        for i in range(10):
            action = np.random.randint(0, 4, 3)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                break
        
        log_pass("multi_env_episode", f"Steps completed, Drawdown: {info['drawdown']:.2%}")
        
        # Test SingleAssetEnv
        single_env = SingleAssetEnv(
            data=data['SOL'],
            asset_name="SOL",
            initial_capital=100_000
        )
        
        obs, info = single_env.reset()
        log_pass("single_env_create", f"SingleAssetEnv (SOL), Obs shape: {obs.shape}")
        
        # Test action space
        log_pass("action_space", f"Actions: {[a.name for a in Action]}")
        
    except Exception as e:
        log_fail("training_pipeline", f"{type(e).__name__}: {e}")
        traceback.print_exc()


# ============================================================================
# TEST 5: DATA FLOW
# ============================================================================

def test_data_flow():
    """Test data collection and feature engineering."""
    print("\n" + "=" * 60)
    print("5️⃣  DATA FLOW")
    print("=" * 60)
    
    try:
        from training.data.multi_asset_collector import MultiAssetDataCollector
        
        # Create collector
        collector = MultiAssetDataCollector(
            data_dir="./test_data",
            exchange="kraken",
            timeframe="1h"
        )
        
        log_pass("collector_create", "MultiAssetDataCollector created")
        
        # Test feature engineering with synthetic data
        n_rows = 500
        dates = pd.date_range(start='2024-01-01', periods=n_rows, freq='1h')
        
        df = pd.DataFrame({
            'open': np.random.uniform(190, 210, n_rows),
            'high': np.random.uniform(200, 220, n_rows),
            'low': np.random.uniform(180, 200, n_rows),
            'close': np.random.uniform(190, 210, n_rows),
            'volume': np.random.uniform(1000, 10000, n_rows),
        }, index=dates)
        
        # Add features
        df_with_features = collector.add_features(df)
        
        # Check features were added
        expected_features = [
            'rsi', 'macd', 'macd_histogram', 
            'ema_21', 'ema_55', 'ema_alignment',
            'bb_upper', 'bb_lower', 'bb_position',
            'atr', 'atr_pct', 'volume_ratio',
            'roc_1', 'roc_24', 'adx'
        ]
        
        missing = [f for f in expected_features if f not in df_with_features.columns]
        
        if not missing:
            log_pass("feature_engineering", f"All {len(expected_features)} features added")
        else:
            log_fail("feature_engineering", f"Missing: {missing}")
        
        # Check for NaN
        nan_counts = df_with_features.isna().sum()
        nan_cols = nan_counts[nan_counts > 0]
        
        if len(nan_cols) == 0:
            log_pass("feature_nan_check", "No NaN values")
        else:
            log_warn("feature_nan_check", f"Some NaN values (expected at start due to lookback)")
        
        # Test feature values are reasonable
        rsi = df_with_features['rsi'].dropna()
        if rsi.min() >= 0 and rsi.max() <= 100:
            log_pass("rsi_range", f"RSI range: {rsi.min():.1f} - {rsi.max():.1f}")
        else:
            log_fail("rsi_range", f"RSI out of bounds: {rsi.min():.1f} - {rsi.max():.1f}")
        
    except Exception as e:
        log_fail("data_flow", f"{type(e).__name__}: {e}")
        traceback.print_exc()


# ============================================================================
# TEST 6: SIGNAL GENERATION
# ============================================================================

def test_signal_generation():
    """Test signal generation through both stages."""
    print("\n" + "=" * 60)
    print("6️⃣  SIGNAL GENERATION")
    print("=" * 60)
    
    try:
        from core.multi_asset_intelligence import (
            MultiAssetConfig, MultiAssetHeavyModel, MultiAssetLightModel,
            PortfolioStrategicState, AssetStrategicState, AssetScenario,
            create_multi_asset_config
        )
        from datetime import datetime, timedelta
        
        config = create_multi_asset_config("aggressive", 100_000)
        
        # Create synthetic data
        n_rows = 500
        dates = pd.date_range(start='2024-01-01', periods=n_rows, freq='1h')
        
        def make_data(base_price):
            return pd.DataFrame({
                'open': np.random.uniform(base_price*0.99, base_price*1.01, n_rows),
                'high': np.random.uniform(base_price*1.0, base_price*1.02, n_rows),
                'low': np.random.uniform(base_price*0.98, base_price*1.0, n_rows),
                'close': np.random.uniform(base_price*0.99, base_price*1.01, n_rows),
                'volume': np.random.uniform(1000, 10000, n_rows),
            }, index=dates)
        
        asset_data = {
            'SOL': make_data(200),
            'BTC': make_data(95000),
            'ETH': make_data(3200),
        }
        
        # Create features
        asset_features = {}
        for asset, df in asset_data.items():
            asset_features[asset] = {
                'close': float(df['close'].iloc[-1]),
                'rsi': 55.0,
                'macd_histogram': 10.0,
                'adx': 28.0,
                'atr_pct': 3.0,
                'volume_ratio': 1.2,
                'ema_alignment': 1,
                'roc_1': 0.5,
                'roc_4': 1.2,
                'roc_24': 3.5,
            }
        
        log_pass("signal_data_prepared", f"Data for {len(asset_data)} assets")
        
        # Test Heavy Model (fallback - no LLM)
        heavy = MultiAssetHeavyModel(config)
        
        # Run analysis (will use fallback since no LLM)
        state = heavy.analyze(asset_data, asset_features)
        
        log_pass("heavy_analysis", f"Market regime: {state.market_regime}")
        
        # Check state structure
        if len(state.asset_states) == 3:
            log_pass("state_assets", f"States for: {list(state.asset_states.keys())}")
        else:
            log_fail("state_assets", f"Expected 3, got {len(state.asset_states)}")
        
        # Check SOL allocation is highest
        sol_alloc = state.allocation_weights.get('SOL', 0)
        btc_alloc = state.allocation_weights.get('BTC', 0)
        eth_alloc = state.allocation_weights.get('ETH', 0)
        
        log_pass("allocations", f"SOL: {sol_alloc:.0%}, BTC: {btc_alloc:.0%}, ETH: {eth_alloc:.0%}")
        
        if sol_alloc >= btc_alloc and sol_alloc >= eth_alloc:
            log_pass("sol_primary_alloc", "SOL has highest allocation")
        else:
            log_warn("sol_primary_alloc", "SOL is not primary allocation")
        
        # Test Light Model
        light = MultiAssetLightModel(config)
        light.update_strategy(state)
        
        log_pass("light_updated", f"Loaded {sum(len(s.scenarios) for s in state.asset_states.values())} scenarios")
        
        # Test evaluation
        prices = {
            'SOL': 205.0,  # Price moved up
            'BTC': 96000.0,
            'ETH': 3300.0,
        }
        
        # Modify features to trigger signal
        trigger_features = dict(asset_features)
        trigger_features['SOL'] = {
            **asset_features['SOL'],
            'roc_1': 2.5,  # Strong move
            'volume_ratio': 2.5,  # Volume spike
        }
        
        signals = light.evaluate(prices, trigger_features)
        
        if signals:
            log_pass("signal_generated", f"Generated {len(signals)} signal(s)")
            for sig in signals:
                log_pass(f"signal_{sig.asset}", f"{sig.action} @ ${sig.entry_price:,.2f}")
        else:
            log_warn("signal_generated", "No signals triggered (may need specific conditions)")
        
    except Exception as e:
        log_fail("signal_generation", f"{type(e).__name__}: {e}")
        traceback.print_exc()


# ============================================================================
# TEST 7: PORTFOLIO MANAGEMENT
# ============================================================================

def test_portfolio_management():
    """Test portfolio constraints and position sizing."""
    print("\n" + "=" * 60)
    print("7️⃣  PORTFOLIO MANAGEMENT")
    print("=" * 60)
    
    try:
        from core.multi_asset_intelligence import (
            MultiAssetConfig, PortfolioManager, MultiAssetSignal,
            PortfolioStrategicState, AssetStrategicState, AssetScenario,
            create_multi_asset_config
        )
        from datetime import datetime, timedelta
        
        config = create_multi_asset_config("aggressive", 100_000)
        portfolio = PortfolioManager(config)
        
        log_pass("portfolio_init", f"Capital: ${portfolio.capital:,}")
        
        # Create mock strategic state
        state = PortfolioStrategicState(
            timestamp=datetime.now(),
            asset_states={},
            market_regime="risk_on",
            correlation_btc_eth=0.85,
            correlation_btc_sol=0.75,
            correlation_eth_sol=0.70,
            total_recommended_exposure=1.0,
            allocation_weights={'SOL': 0.50, 'BTC': 0.25, 'ETH': 0.25},
            portfolio_var_24h=0.05,
            max_position_value=60_000,
            valid_until=datetime.now() + timedelta(hours=1)
        )
        
        # Test SOL signal (50% allocation)
        sol_signal = MultiAssetSignal(
            timestamp=datetime.now(),
            asset="SOL",
            action="LONG",
            size_pct=0.50,
            entry_price=200.0,
            stop_loss=194.0,  # 3% stop
            take_profit=212.0,  # 6% TP
            triggered_scenario="sol_momentum",
            trigger_reason="Strong momentum",
            confidence=0.75,
            urgency="high"
        )
        
        valid, reason, sizing = portfolio.validate_signal(sol_signal, state)
        
        if valid:
            log_pass("sol_signal_valid", f"SOL 50% position approved")
            log_pass("sol_sizing", f"Value: ${sizing['position_value']:,.0f}, Risk: {sizing['risk_pct']:.2%}")
        else:
            log_fail("sol_signal_valid", f"Rejected: {reason}")
        
        # Simulate position opened
        portfolio.positions['SOL'] = {
            'side': 'long',
            'value': sizing['position_value'],
            'size': sizing['size_units']
        }
        
        # Test BTC signal (should work - 25%)
        btc_signal = MultiAssetSignal(
            timestamp=datetime.now(),
            asset="BTC",
            action="LONG",
            size_pct=0.25,
            entry_price=95000.0,
            stop_loss=92150.0,
            take_profit=100700.0,
            triggered_scenario="btc_breakout",
            trigger_reason="Resistance break",
            confidence=0.70,
            urgency="medium"
        )
        
        valid, reason, sizing = portfolio.validate_signal(btc_signal, state)
        
        if valid:
            log_pass("btc_signal_valid", f"BTC 25% position approved")
        else:
            log_fail("btc_signal_valid", f"Rejected: {reason}")
        
        # Simulate BTC position
        portfolio.positions['BTC'] = {
            'side': 'long',
            'value': sizing['position_value'],
            'size': sizing['size_units']
        }
        
        # Test oversized signal (should be reduced)
        big_signal = MultiAssetSignal(
            timestamp=datetime.now(),
            asset="ETH",
            action="LONG",
            size_pct=0.40,  # Too big
            entry_price=3200.0,
            stop_loss=3104.0,
            take_profit=3392.0,
            triggered_scenario="eth_test",
            trigger_reason="Test",
            confidence=0.60,
            urgency="low"
        )
        
        valid, reason, sizing = portfolio.validate_signal(big_signal, state)
        
        # Should be reduced due to exposure limits
        current_exposure = portfolio._get_current_exposure()
        log_pass("exposure_tracking", f"Current exposure: {current_exposure:.0%}")
        
        if valid and sizing.get('size_pct', 0) < 0.40:
            log_pass("exposure_limit", f"ETH reduced to {sizing.get('size_pct', 0):.0%} (limit enforced)")
        elif not valid:
            log_pass("exposure_limit", f"ETH blocked: {reason}")
        else:
            log_warn("exposure_limit", "Exposure limit may not be enforced")
        
        # Test correlated exposure limit
        corr_btc = portfolio._get_asset_exposure('BTC')
        corr_eth = portfolio._get_asset_exposure('ETH')
        total_corr = corr_btc + corr_eth
        
        log_pass("correlation_check", f"BTC+ETH combined: {total_corr:.0%} (limit: {config.max_correlated_exposure:.0%})")
        
    except Exception as e:
        log_fail("portfolio_management", f"{type(e).__name__}: {e}")
        traceback.print_exc()


# ============================================================================
# TEST 8: INTEGRATION TEST
# ============================================================================

def test_integration():
    """Full integration test simulating real operation."""
    print("\n" + "=" * 60)
    print("8️⃣  INTEGRATION TEST")
    print("=" * 60)
    
    try:
        from core.multi_asset_intelligence import (
            MultiAssetConfig, MultiAssetHeavyModel, MultiAssetLightModel,
            PortfolioManager, create_multi_asset_config
        )
        from core.gpu_ta import GPUTechnicalAnalyzer
        
        # Initialize components
        config = create_multi_asset_config("aggressive", 100_000)
        heavy = MultiAssetHeavyModel(config)
        light = MultiAssetLightModel(config)
        portfolio = PortfolioManager(config)
        ta = GPUTechnicalAnalyzer()
        
        log_pass("components_init", "All components initialized")
        
        # Create realistic synthetic data
        n_rows = 500
        dates = pd.date_range(end=datetime.now(), periods=n_rows, freq='1h')
        
        np.random.seed(42)
        
        def make_realistic_data(base_price, vol=0.02):
            returns = np.random.randn(n_rows) * vol
            prices = base_price * np.exp(np.cumsum(returns))
            
            return pd.DataFrame({
                'open': prices * (1 + np.random.randn(n_rows) * 0.001),
                'high': prices * (1 + abs(np.random.randn(n_rows) * 0.005)),
                'low': prices * (1 - abs(np.random.randn(n_rows) * 0.005)),
                'close': prices,
                'volume': np.abs(np.random.randn(n_rows) * 1000 + 5000),
            }, index=dates)
        
        asset_data = {
            'SOL': make_realistic_data(200, vol=0.04),  # SOL more volatile
            'BTC': make_realistic_data(95000, vol=0.02),
            'ETH': make_realistic_data(3200, vol=0.025),
        }
        
        # Calculate technical indicators
        asset_data_with_ta = ta.calculate_multi_asset(asset_data)
        
        log_pass("ta_calculated", "Technical indicators computed")
        
        # Extract features
        asset_features = {}
        for asset, df in asset_data_with_ta.items():
            last = df.iloc[-1]
            asset_features[asset] = {
                'close': float(last['close']),
                'rsi': float(last.get('rsi', 50)),
                'macd_histogram': float(last.get('macd_histogram', 0)),
                'adx': float(last.get('adx', 20)),
                'atr_pct': float(last.get('atr_pct', 2)),
                'volume_ratio': float(last.get('volume_ratio', 1)),
                'ema_alignment': int(last.get('ema_alignment', 0)),
                'roc_1': float(last.get('roc_1', 0)),
                'roc_24': float(last.get('roc_24', 0)),
            }
        
        # Stage 1: Heavy analysis
        print("\n  Running Stage 1 (Heavy) analysis...")
        state = heavy.analyze(asset_data_with_ta, asset_features)
        
        log_pass("stage1_complete", f"Regime: {state.market_regime}")
        
        for asset, asset_state in state.asset_states.items():
            log_pass(f"  {asset}_state", f"Regime: {asset_state.regime}, Bias: {asset_state.directional_bias:+.2f}")
        
        # Stage 2: Light model monitoring
        light.update_strategy(state)
        
        # Simulate price ticks
        print("\n  Simulating 10 price ticks...")
        signals_count = 0
        
        for i in range(10):
            # Simulate price movement
            prices = {}
            tick_features = {}
            
            for asset in ['SOL', 'BTC', 'ETH']:
                base_price = asset_features[asset]['close']
                move = np.random.randn() * 0.005
                prices[asset] = base_price * (1 + move)
                
                tick_features[asset] = dict(asset_features[asset])
                tick_features[asset]['close'] = prices[asset]
                tick_features[asset]['roc_1'] = move * 100
            
            # Check for signals
            signals = light.evaluate(prices, tick_features)
            
            if signals:
                signals_count += len(signals)
                for sig in signals:
                    # Validate through portfolio
                    valid, reason, sizing = portfolio.validate_signal(sig, state)
                    
                    if valid:
                        log_pass(f"tick_{i+1}_signal", f"{sig.asset} {sig.action} validated")
        
        log_pass("stage2_complete", f"10 ticks processed, {signals_count} signals generated")
        
        # Final portfolio state
        summary = portfolio.get_portfolio_summary()
        log_pass("portfolio_final", f"Capital: ${summary['capital']:,.0f}, Exposure: {summary['total_exposure']:.0%}")
        
        log_pass("integration_complete", "Full pipeline executed successfully")
        
    except Exception as e:
        log_fail("integration", f"{type(e).__name__}: {e}")
        traceback.print_exc()


# ============================================================================
# TEST 9: P0 MULTI-MODAL DATA & MODEL ALPHA VERIFICATION
# ============================================================================

def test_p0_components():
    """
    P0 验证: 多模态数据管道 + 模型 Alpha
    
    验收标准:
    1. 每个 feed 的 last_timestamp / staleness / health
    2. 是否触发降级
    3. --mode verify 下不允许任何 feed 静默失败
    4. ModelIntent 是否被 authority_fusion 接纳
    5. 是否被 veto（以及 veto 原因）
    """
    print("\n" + "=" * 60)
    print("9️⃣  P0 MULTI-MODAL DATA & MODEL ALPHA VERIFICATION")
    print("=" * 60)
    
    # =========================================================================
    # P0-1: 数据管道验证
    # =========================================================================
    print("\n  [P0-1] Data Pipeline Verification...")
    
    try:
        from data_mgmt.feeds import (
            get_onchain_feed, get_sentiment_feed, 
            get_macro_feed, get_lob_feed,
            OnChainFeed, SentimentFeed, MacroFeed, LOBFeed
        )
        log_pass("p0_feeds_import", "All feeds imported successfully")
    except ImportError as e:
        log_fail("p0_feeds_import", f"Import error: {e}")
        return
    
    # 测试各个 Feed
    feeds_status = {}
    
    # OnChain Feed
    try:
        onchain_feed = get_onchain_feed()
        health = onchain_feed.get_health()
        feeds_status['onchain'] = health
        
        if health.get('last_fetch_time') is None:
            # 首次获取需要 fetch
            import asyncio
            loop = asyncio.new_event_loop()
            tick = loop.run_until_complete(onchain_feed.fetch())
            loop.close()
            health = onchain_feed.get_health()
        
        log_pass("p0_onchain_feed", f"Source: {health.get('source')}, "
                f"Staleness: {health.get('staleness_sec', 'N/A'):.1f}s")
    except Exception as e:
        log_fail("p0_onchain_feed", str(e))
    
    # Sentiment Feed  
    try:
        from data_mgmt.feeds.sentiment_feed import reset_sentiment_feed
        reset_sentiment_feed()  # 重置以确保干净测试
        sentiment_feed = get_sentiment_feed()
        
        import asyncio
        loop = asyncio.new_event_loop()
        tick = loop.run_until_complete(sentiment_feed.fetch())
        loop.close()
        
        health = sentiment_feed.get_health()
        feeds_status['sentiment'] = health
        
        log_pass("p0_sentiment_feed", f"Source: {health.get('source')}, "
                f"Staleness: {health.get('staleness_sec', 'N/A'):.1f}s")
    except Exception as e:
        log_fail("p0_sentiment_feed", str(e))
    
    # Macro Feed
    try:
        from data_mgmt.feeds.macro_feed import reset_macro_feed
        reset_macro_feed()
        macro_feed = get_macro_feed()
        
        import asyncio
        loop = asyncio.new_event_loop()
        tick = loop.run_until_complete(macro_feed.fetch())
        loop.close()
        
        health = macro_feed.get_health()
        feeds_status['macro'] = health
        
        log_pass("p0_macro_feed", f"Source: {health.get('source')}, "
                f"Staleness: {health.get('staleness_sec', 'N/A'):.1f}s")
    except Exception as e:
        log_fail("p0_macro_feed", str(e))
    
    # LOB Feed
    try:
        from data_mgmt.feeds.lob_feed import reset_lob_feed
        reset_lob_feed()
        lob_feed = get_lob_feed()
        
        import asyncio
        loop = asyncio.new_event_loop()
        tick = loop.run_until_complete(lob_feed.fetch())
        loop.close()
        
        health = lob_feed.get_health()
        feeds_status['lob'] = health
        
        log_pass("p0_lob_feed", f"Source: {health.get('source')}, "
                f"Symbols: {health.get('symbols_covered')}")
    except Exception as e:
        log_fail("p0_lob_feed", str(e))
    
    # 数据健康检查
    try:
        from data_mgmt.data_health import (
            get_data_health, DataSourceType, DegradationLevel, HealthStatus
        )
        
        data_health = get_data_health()
        
        # 模拟更新各数据源状态
        for source_name, feed_health in feeds_status.items():
            try:
                source_type = DataSourceType(source_name.upper()) if hasattr(DataSourceType, source_name.upper()) else None
                if source_type:
                    data_health._monitor.update_source(
                        source_type=source_type,
                        available=not feed_health.get('is_stale', True),
                        staleness_sec=feed_health.get('staleness_sec', 0),
                        confidence=feed_health.get('confidence', 0.5),
                    )
            except Exception:
                pass
        
        snapshot = data_health.get_snapshot()
        
        log_pass("p0_data_health", f"Status: {snapshot.overall_status.value}, "
                f"Degradation: {snapshot.degradation_level.value}")
        
        # 检查降级是否正常工作
        if snapshot.degradation_level == DegradationLevel.NO_TRADE:
            log_warn("p0_degradation", "System in NO_TRADE mode due to data issues")
        elif snapshot.degradation_level == DegradationLevel.EXIT_ONLY:
            log_warn("p0_degradation", "System in EXIT_ONLY mode")
        else:
            log_pass("p0_degradation", "No critical degradation")
            
    except Exception as e:
        log_fail("p0_data_health", str(e))
    
    # =========================================================================
    # P0-2: 模型 Alpha 验证
    # =========================================================================
    print("\n  [P0-2] Model Alpha Verification...")
    
    try:
        from agents.model_alpha_agent import (
            get_model_alpha_agent, ModelIntent, UnifiedState, IntentDirection
        )
        
        agent = get_model_alpha_agent()
        
        # 创建测试状态
        state = UnifiedState(
            timestamp=datetime.now(),
            prices={"BTC": 100000, "ETH": 3500, "SOL": 200},
            returns_1h={"BTC": 0.01, "ETH": 0.02, "SOL": -0.01},
            returns_24h={"BTC": 0.05, "ETH": 0.08, "SOL": -0.03},
            volatility={"BTC": 0.5, "ETH": 0.6, "SOL": 0.8},
            lob_imbalance={"BTC": 0.1, "ETH": -0.2, "SOL": 0.3},
            fear_greed=30,
            sentiment_score=-0.2,
        )
        
        # 生成意图
        intent = agent.generate_intent(state, "SOL")
        
        if intent:
            log_pass("p0_model_intent", f"Asset: {intent.asset}, Dir: {intent.direction.value}, "
                    f"Conf: {intent.confidence:.2f}")
            
            # 记录意图
            log_pass("p0_intent_explanation", f"Tokens: {intent.explanation_tokens[:3]}")
        else:
            log_warn("p0_model_intent", "No intent generated (confidence too low)")
        
    except Exception as e:
        log_fail("p0_model_alpha", str(e))
    
    # 验证 authority_fusion 集成
    try:
        from signals.authority_fusion import (
            AUTHORITY_MATRIX_NORMAL, AUTHORITY_MATRIX_OPPORTUNITY
        )
        
        # 检查 model_alpha 是否在矩阵中
        if "model_alpha" in AUTHORITY_MATRIX_NORMAL:
            log_pass("p0_authority_integration", 
                    f"model_alpha in NORMAL: {AUTHORITY_MATRIX_NORMAL['model_alpha'].value}")
        else:
            log_fail("p0_authority_integration", "model_alpha not in AUTHORITY_MATRIX_NORMAL")
        
        if "model_alpha" in AUTHORITY_MATRIX_OPPORTUNITY:
            log_pass("p0_authority_opportunity", 
                    f"model_alpha in OPPORTUNITY: {AUTHORITY_MATRIX_OPPORTUNITY['model_alpha'].value}")
        else:
            log_fail("p0_authority_opportunity", "model_alpha not in AUTHORITY_MATRIX_OPPORTUNITY")
            
    except Exception as e:
        log_fail("p0_authority_fusion", str(e))
    
    # 验证权重控制器
    try:
        from risk.correlation_realtime_controller import get_model_alpha_weight_controller
        
        weight_ctrl = get_model_alpha_weight_controller()
        status = weight_ctrl.get_status()
        
        log_pass("p0_weight_controller", f"Max weight: {status['current_max_weight']:.2f}, "
                f"Model allowed: {status['model_allowed']}")
        
    except Exception as e:
        log_fail("p0_weight_controller", str(e))
    
    # 验证权重自适应
    try:
        from signals.adaptive_strategy_weight_manager import get_model_alpha_weight_adapter
        
        adapter = get_model_alpha_weight_adapter()
        metrics = adapter.get_metrics()
        
        log_pass("p0_weight_adapter", f"Weight: {metrics['current_weight']:.2f}, "
                f"Consecutive failures: {metrics['consecutive_failures']}")
        
    except Exception as e:
        log_fail("p0_weight_adapter", str(e))
    
    # 验证 DRL 仍然禁止 entry authority
    try:
        from signals.authority_fusion import AUTHORITY_MATRIX_NORMAL, Authority
        
        drl_authority = AUTHORITY_MATRIX_NORMAL.get("drl")
        if drl_authority == Authority.ADVISE:
            log_pass("p0_drl_constraint", "DRL is ADVISE only (correct)")
        elif drl_authority in [Authority.DECIDE, Authority.TRIGGER]:
            log_fail("p0_drl_constraint", f"DRL has {drl_authority.value} - should be ADVISE only!")
        else:
            log_pass("p0_drl_constraint", f"DRL authority: {drl_authority.value}")
            
    except Exception as e:
        log_fail("p0_drl_constraint", str(e))
    
    print("\n  [P0] Verification Complete")


# ============================================================================
# TEST 10: GPU ACCELERATION
# ============================================================================

def test_gpu():
    """Test GPU availability and acceleration."""
    print("\n" + "=" * 60)
    print("9️⃣  GPU ACCELERATION")
    print("=" * 60)
    
    # Check CuPy
    try:
        import cupy as cp
        
        # Get device info
        device = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(device.id)
        name = props['name'].decode() if isinstance(props['name'], bytes) else props['name']
        
        log_pass("cupy_available", f"GPU: {name}")
        
        # Test computation
        x = cp.random.randn(10000, 10000)
        y = cp.dot(x, x.T)
        cp.cuda.Stream.null.synchronize()
        
        log_pass("gpu_compute", "Matrix multiplication successful")
        
        # Get memory info
        mem_info = cp.cuda.runtime.memGetInfo()
        free_gb = mem_info[0] / 1e9
        total_gb = mem_info[1] / 1e9
        
        log_pass("gpu_memory", f"Free: {free_gb:.1f}GB / {total_gb:.1f}GB")
        
    except ImportError:
        log_warn("cupy_available", "CuPy not installed (GPU acceleration disabled)")
    except Exception as e:
        log_fail("gpu_test", str(e))
    
    # Check PyTorch CUDA
    try:
        import torch
        
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            log_pass("pytorch_cuda", f"Available: {device_name}")
        else:
            log_warn("pytorch_cuda", "CUDA not available for PyTorch")
    except ImportError:
        log_skip("pytorch_cuda", "PyTorch not installed")


# ============================================================================
# MAIN
# ============================================================================

def print_summary():
    """Print diagnostic summary."""
    print("\n" + "=" * 60)
    print("📊 DIAGNOSTIC SUMMARY")
    print("=" * 60)
    
    total = len(RESULTS['passed']) + len(RESULTS['failed'])
    
    print(f"\nON Passed: {len(RESULTS['passed'])}")
    print(f"OFF Failed: {len(RESULTS['failed'])}")
    print(f"[WARN]️  Warnings: {len(RESULTS['warnings'])}")
    print(f"⏭️  Skipped: {len(RESULTS['skipped'])}")
    
    if RESULTS['failed']:
        print("\nOFF FAILURES:")
        for name, error in RESULTS['failed']:
            print(f"   • {name}: {error}")
    
    if RESULTS['warnings']:
        print("\n[WARN]️  WARNINGS:")
        for name, msg in RESULTS['warnings']:
            print(f"   • {name}: {msg}")
    
    # Overall status
    print("\n" + "=" * 60)
    if not RESULTS['failed']:
        print("ON ALL TESTS PASSED - System ready for deployment")
    else:
        print(f"OFF {len(RESULTS['failed'])} TESTS FAILED - Review issues above")
    print("=" * 60)
    
    return len(RESULTS['failed']) == 0


def main():
    """Run all diagnostics."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║      HMATS v5 MULTI-ASSET END-TO-END DIAGNOSTIC                 ║
║                                                                  ║
║      SOL: 50% | BTC: 25% | ETH: 25% | Cash: 0%                  ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")
    
    start_time = time.time()
    
    # Change to project directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Add to path
    sys.path.insert(0, str(script_dir))
    
    # Run all tests
    test_configuration()
    test_imports()
    test_core_components()
    test_training_pipeline()
    test_data_flow()
    test_signal_generation()
    test_portfolio_management()
    test_integration()
    test_p0_components()  # P0 验证
    test_gpu()
    
    elapsed = time.time() - start_time
    
    print(f"\n⏱️  Total time: {elapsed:.1f}s")
    
    success = print_summary()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
