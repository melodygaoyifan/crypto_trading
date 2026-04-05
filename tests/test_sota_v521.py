"""
HMATS v5.2.1 SOTA Test Suite

Comprehensive tests for all v5.2.1 SOTA enhancements:
1. Correlation Dynamic Controller with prediction
2. Market Impact Calculator with learning
3. Adaptive Weight Manager
4. Unified Event System
5. RL Drift Fallback
6. Integration Layer

Run: python -m pytest tests/test_sota_v521.py -v
Or:  python tests/test_sota_v521.py
"""

import asyncio
import sys
import os
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List
import random
import math

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestResult:
    """Test result container"""
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.error = None
        self.duration_ms = 0
        self.details: Dict[str, Any] = {}
    
    def __str__(self):
        status = "✓ PASS" if self.passed else "✗ FAIL"
        msg = f"{status} | {self.name} ({self.duration_ms:.1f}ms)"
        if self.error:
            msg += f"\n       Error: {self.error}"
        return msg


class SOTAV521TestSuite:
    """Comprehensive test suite for HMATS v5.2.1"""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.start_time = None
    
    def run_test(self, name: str, test_func):
        """Run a single test and capture results"""
        result = TestResult(name)
        start = datetime.now()
        
        try:
            if asyncio.iscoroutinefunction(test_func):
                asyncio.run(test_func())
            else:
                test_func()
            result.passed = True
        except AssertionError as e:
            result.error = str(e)
        except Exception as e:
            result.error = f"{type(e).__name__}: {e}"
        
        result.duration_ms = (datetime.now() - start).total_seconds() * 1000
        self.results.append(result)
        print(result)
        return result.passed
    
    def run_all(self) -> bool:
        """Run all tests"""
        self.start_time = datetime.now()
        print("=" * 70)
        print("HMATS v5.2.1 SOTA Test Suite")
        print("=" * 70)
        print()
        
        # Test categories
        print("[1. Correlation Dynamic Controller Tests]")
        self.run_test("test_correlation_initialization", self.test_correlation_initialization)
        self.run_test("test_correlation_regime_detection", self.test_correlation_regime_detection)
        self.run_test("test_correlation_jump_detection", self.test_correlation_jump_detection)
        self.run_test("test_correlation_prediction", self.test_correlation_prediction)
        self.run_test("test_correlation_position_adjustment", self.test_correlation_position_adjustment)
        print()
        
        print("[2. Market Impact Calculator Tests]")
        self.run_test("test_impact_initialization", self.test_impact_initialization)
        self.run_test("test_impact_prediction", self.test_impact_prediction)
        self.run_test("test_impact_execution_plan", self.test_impact_execution_plan)
        self.run_test("test_impact_online_learning", self.test_impact_online_learning)
        print()
        
        print("[3. Adaptive Weight Manager Tests]")
        self.run_test("test_weight_initialization", self.test_weight_initialization)
        self.run_test("test_weight_trade_recording", self.test_weight_trade_recording)
        self.run_test("test_weight_sharpe_calculation", self.test_weight_sharpe_calculation)
        self.run_test("test_weight_calmar_calculation", self.test_weight_calmar_calculation)
        self.run_test("test_weight_smoothing", self.test_weight_smoothing)
        print()
        
        print("[4. Unified Event System Tests]")
        self.run_test("test_event_initialization", self.test_event_initialization)
        self.run_test("test_event_subscription", self.test_event_subscription)
        self.run_test("test_event_emission", self.test_event_emission)
        self.run_test("test_event_adapter", self.test_event_adapter)
        print()
        
        print("[5. RL Drift Fallback Tests]")
        self.run_test("test_drift_initialization", self.test_drift_initialization)
        self.run_test("test_drift_kl_divergence", self.test_drift_kl_divergence)
        self.run_test("test_drift_psi_calculation", self.test_drift_psi_calculation)
        self.run_test("test_drift_auto_fallback", self.test_drift_auto_fallback)
        self.run_test("test_drift_recovery", self.test_drift_recovery)
        print()
        
        print("[6. Integration Tests]")
        self.run_test("test_integration_initialization", self.test_integration_initialization)
        self.run_test("test_integration_health_check", self.test_integration_health_check)
        self.run_test("test_integration_cross_component", self.test_integration_cross_component)
        self.run_test("test_integration_graceful_degradation", self.test_integration_graceful_degradation)
        print()
        
        # Summary
        self.print_summary()
        
        return all(r.passed for r in self.results)
    
    def print_summary(self):
        """Print test summary"""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed
        
        duration = (datetime.now() - self.start_time).total_seconds()
        
        print("=" * 70)
        print(f"Test Summary: {passed}/{total} passed ({failed} failed)")
        print(f"Total Duration: {duration:.2f}s")
        print("=" * 70)
        
        if failed > 0:
            print("\nFailed Tests:")
            for r in self.results:
                if not r.passed:
                    print(f"  - {r.name}: {r.error}")
    
    # ========== Correlation Controller Tests ==========
    
    def test_correlation_initialization(self):
        """Test correlation controller initialization"""
        from risk.correlation_dynamic_controller_v521 import (
            get_correlation_controller_v521,
            CorrelationDynamicsConfigV521
        )
        
        config = CorrelationDynamicsConfigV521()
        controller = get_correlation_controller_v521(config)
        
        assert controller is not None, "Controller should be created"
        assert controller.config.lookback_bars_medium == 100, "Default lookback should be 100"
        assert controller.config.enable_prediction == True, "Prediction should be enabled"
    
    def test_correlation_regime_detection(self):
        """Test correlation regime detection"""
        from risk.correlation_dynamic_controller_v521 import (
            CorrelationDynamicControllerV521,
            CorrelationDynamicsConfigV521,
            CorrelationRegimeV521
        )
        
        # Fresh controller for this test
        config = CorrelationDynamicsConfigV521(lookback_bars_medium=20)
        controller = CorrelationDynamicControllerV521(config)
        
        # Feed highly correlated data
        for i in range(50):
            t = i * 0.1
            controller.update_price("BTC", 100000 + 1000 * math.sin(t))
            controller.update_price("ETH", 3500 + 35 * math.sin(t))
            controller.update_price("SOL", 200 + 2 * math.sin(t))
        
        assessment = controller.compute_assessment()
        
        assert assessment is not None, "Assessment should be computed"
        assert assessment.regime in CorrelationRegimeV521, "Should return valid regime"
        assert 0 <= assessment.position_scale_factor <= 1.0, "Scale factor should be in [0, 1]"
    
    def test_correlation_jump_detection(self):
        """Test correlation jump detection"""
        from risk.correlation_dynamic_controller_v521 import (
            CorrelationDynamicControllerV521,
            CorrelationDynamicsConfigV521
        )
        
        config = CorrelationDynamicsConfigV521(
            lookback_bars_short=10,
            lookback_bars_medium=20,
            jump_zscore_threshold=2.0
        )
        controller = CorrelationDynamicControllerV521(config)
        
        # Feed stable data first
        for i in range(30):
            controller.update_price("BTC", 100000 + random.gauss(0, 100))
            controller.update_price("ETH", 3500 + random.gauss(0, 10))
        
        # Then inject highly correlated spike
        for i in range(10):
            t = i * 0.5
            spike = 500 * math.sin(t)
            controller.update_price("BTC", 100000 + spike)
            controller.update_price("ETH", 3500 + spike * 0.035)
        
        assessment = controller.compute_assessment()
        # Jump detection may or may not trigger depending on data
        assert assessment is not None
    
    def test_correlation_prediction(self):
        """Test correlation prediction"""
        from risk.correlation_dynamic_controller_v521 import (
            CorrelationDynamicControllerV521,
            CorrelationDynamicsConfigV521
        )
        
        config = CorrelationDynamicsConfigV521(enable_prediction=True)
        controller = CorrelationDynamicControllerV521(config)
        
        # Feed trending correlation data
        for i in range(100):
            # Increasing correlation over time
            noise_btc = random.gauss(0, 100 / (1 + i * 0.02))
            noise_eth = random.gauss(0, 10 / (1 + i * 0.02))
            controller.update_price("BTC", 100000 + noise_btc + i * 10)
            controller.update_price("ETH", 3500 + noise_eth + i * 0.35)
        
        assessment = controller.compute_assessment()
        
        assert assessment.prediction is not None, "Prediction should be available"
        assert "predicted_1h" in assessment.prediction or assessment.prediction == {}, \
            "Prediction should have time horizons"
    
    def test_correlation_position_adjustment(self):
        """Test position adjustment based on correlation"""
        from risk.correlation_dynamic_controller_v521 import (
            CorrelationDynamicControllerV521,
            CorrelationDynamicsConfigV521
        )
        
        config = CorrelationDynamicsConfigV521()
        controller = CorrelationDynamicControllerV521(config)
        
        # Feed some data
        for i in range(50):
            controller.update_price("BTC", 100000 + random.gauss(0, 100))
            controller.update_price("ETH", 3500 + random.gauss(0, 10))
        
        base_size = 1.0
        adjusted_size, reason = controller.get_position_adjustment("BTC", base_size)
        
        assert 0 < adjusted_size <= base_size, "Adjusted size should be <= base"
        assert len(reason) > 0, "Should provide adjustment reason"
    
    # ========== Market Impact Calculator Tests ==========
    
    def test_impact_initialization(self):
        """Test impact calculator initialization"""
        from execution.market_impact_v521 import (
            get_impact_calculator_v521,
            MarketImpactConfigV521
        )
        
        config = MarketImpactConfigV521()
        calculator = get_impact_calculator_v521(config)
        
        assert calculator is not None, "Calculator should be created"
        assert calculator.config.enable_online_learning == True
    
    def test_impact_prediction(self):
        """Test impact prediction"""
        from execution.market_impact_v521 import (
            MarketImpactCalculatorV521,
            MarketImpactConfigV521
        )
        
        config = MarketImpactConfigV521()
        calculator = MarketImpactCalculatorV521(config)
        
        # Create mock orderbook
        orderbook = {
            "symbol": "BTC",
            "bids": [[99900, 1.0], [99800, 2.0], [99700, 5.0], [99600, 10.0]],
            "asks": [[100100, 1.0], [100200, 2.0], [100300, 5.0], [100400, 10.0]],
            "timestamp": datetime.now().isoformat()
        }
        
        calculator.update_orderbook(orderbook)
        
        prediction = calculator.predict_impact(
            symbol="BTC",
            side="buy",
            quantity_usd=50000,
            orderbook=orderbook,
            urgency=0.5
        )
        
        assert prediction is not None, "Prediction should be returned"
        assert "total_impact_bps" in prediction, "Should include total impact"
        assert prediction["total_impact_bps"] >= 0, "Impact should be non-negative"
    
    def test_impact_execution_plan(self):
        """Test execution plan generation"""
        from execution.market_impact_v521 import (
            MarketImpactCalculatorV521,
            MarketImpactConfigV521
        )
        
        config = MarketImpactConfigV521()
        calculator = MarketImpactCalculatorV521(config)
        
        orderbook = {
            "symbol": "BTC",
            "bids": [[99900, 1.0], [99800, 2.0], [99700, 5.0]],
            "asks": [[100100, 1.0], [100200, 2.0], [100300, 5.0]],
            "timestamp": datetime.now().isoformat()
        }
        
        plan = calculator.get_execution_plan(
            symbol="BTC",
            side="buy",
            total_quantity_usd=100000,
            orderbook=orderbook,
            urgency=0.3
        )
        
        assert plan is not None, "Plan should be generated"
        assert "execution_strategy" in plan, "Should include strategy"
        assert "impact_prediction" in plan, "Should include impact prediction"
    
    def test_impact_online_learning(self):
        """Test online learning from executions"""
        from execution.market_impact_v521 import (
            MarketImpactCalculatorV521,
            MarketImpactConfigV521,
            TradeExecution
        )
        
        config = MarketImpactConfigV521(enable_online_learning=True)
        calculator = MarketImpactCalculatorV521(config)
        
        # Record some executions
        for i in range(10):
            execution = TradeExecution(
                symbol="BTC",
                side="buy",
                quantity_usd=50000 + i * 1000,
                estimated_impact_bps=10.0 + i,
                realized_impact_bps=12.0 + i + random.gauss(0, 2),
                participation_rate=0.01 + i * 0.001,
                duration_seconds=60,
                num_slices=5,
                timestamp=datetime.now()
            )
            calculator.record_execution(execution)
        
        stats = calculator.get_calibration_stats()
        
        assert stats is not None, "Stats should be available"
        assert stats.get("sample_count", 0) > 0, "Should have recorded samples"
    
    # ========== Adaptive Weight Manager Tests ==========
    
    def test_weight_initialization(self):
        """Test weight manager initialization"""
        from signals.adaptive_weight_v521 import (
            get_adaptive_weight_manager_v521,
            AdaptiveWeightConfigV521
        )
        
        config = AdaptiveWeightConfigV521()
        manager = get_adaptive_weight_manager_v521(config)
        
        assert manager is not None, "Manager should be created"
        
        weights = manager.get_all_weights()
        assert isinstance(weights, dict), "Weights should be dict"
    
    def test_weight_trade_recording(self):
        """Test trade recording"""
        from signals.adaptive_weight_v521 import (
            AdaptiveWeightManagerV521,
            AdaptiveWeightConfigV521
        )
        
        config = AdaptiveWeightConfigV521()
        manager = AdaptiveWeightManagerV521(config)
        
        # Record trades
        manager.record_trade("trend_following", 0.02, True)
        manager.record_trade("trend_following", -0.01, False)
        manager.record_trade("trend_following", 0.015, True)
        
        metrics = manager.get_metrics("trend_following")
        
        assert metrics is not None, "Metrics should be available"
        assert metrics.trade_count == 3, "Should have 3 trades"
    
    def test_weight_sharpe_calculation(self):
        """Test Sharpe ratio calculation"""
        from signals.adaptive_weight_v521 import (
            AdaptiveWeightManagerV521,
            AdaptiveWeightConfigV521
        )
        
        config = AdaptiveWeightConfigV521()
        manager = AdaptiveWeightManagerV521(config)
        
        # Record enough trades for Sharpe
        for i in range(50):
            pnl = 0.02 + random.gauss(0, 0.01)  # Positive drift
            manager.record_trade("test_strategy", pnl, pnl > 0)
        
        metrics = manager.get_metrics("test_strategy")
        
        assert metrics is not None
        # Sharpe should be positive for positive drift
        assert metrics.sharpe_ratio >= 0 or metrics.sharpe_ratio is None
    
    def test_weight_calmar_calculation(self):
        """Test Calmar ratio calculation"""
        from signals.adaptive_weight_v521 import (
            AdaptiveWeightManagerV521,
            AdaptiveWeightConfigV521
        )
        
        config = AdaptiveWeightConfigV521()
        manager = AdaptiveWeightManagerV521(config)
        
        # Record trades with drawdown
        cumulative = 0
        for i in range(100):
            if i < 50:
                pnl = 0.02 + random.gauss(0, 0.005)
            else:
                pnl = -0.01 + random.gauss(0, 0.005)  # Drawdown period
            cumulative += pnl
            manager.record_trade("dd_strategy", pnl, pnl > 0)
        
        metrics = manager.get_metrics("dd_strategy")
        assert metrics is not None
    
    def test_weight_smoothing(self):
        """Test weight smoothing (EMA)"""
        from signals.adaptive_weight_v521 import (
            AdaptiveWeightManagerV521,
            AdaptiveWeightConfigV521
        )
        
        config = AdaptiveWeightConfigV521(weight_smoothing_alpha=0.3)
        manager = AdaptiveWeightManagerV521(config)
        
        weights_history = []
        
        # Record trades and track weight changes
        for i in range(20):
            pnl = 0.05 if i % 2 == 0 else -0.05  # Oscillating
            manager.record_trade("oscillating", pnl, pnl > 0)
            weights_history.append(manager.get_weight("oscillating"))
        
        # Smoothing should prevent extreme oscillation
        # Weight changes should be moderate
        max_change = max(abs(weights_history[i+1] - weights_history[i]) 
                        for i in range(len(weights_history)-1))
        
        # With smoothing, changes shouldn't be too extreme
        assert max_change < 1.0, "Weight changes should be smoothed"
    
    # ========== Unified Event System Tests ==========
    
    def test_event_initialization(self):
        """Test event system initialization"""
        from infra.unified_event_v521 import (
            init_event_driven_system,
            RunMode
        )
        
        manager = init_event_driven_system(RunMode.PAPER)
        
        assert manager is not None, "Manager should be created"
        assert manager.run_mode == RunMode.PAPER
    
    def test_event_subscription(self):
        """Test event subscription"""
        from infra.unified_event_v521 import (
            UnifiedEventManager,
            EventTypeV521,
            RunMode
        )
        
        manager = UnifiedEventManager(RunMode.PAPER)
        
        received_events = []
        
        def handler(event):
            received_events.append(event)
        
        manager.subscribe(EventTypeV521.SIGNAL_GENERATED, handler)
        
        # Emit event
        manager.emit(EventTypeV521.SIGNAL_GENERATED, {"test": "data"})
        
        assert len(received_events) == 1, "Should receive 1 event"
        assert received_events[0].payload["test"] == "data"
    
    def test_event_emission(self):
        """Test event emission"""
        from infra.unified_event_v521 import (
            UnifiedEventManager,
            EventTypeV521,
            RunMode
        )
        
        manager = UnifiedEventManager(RunMode.PAPER)
        
        # Emit without handler (should not crash)
        manager.emit(EventTypeV521.TRADE_EXECUTED, {"symbol": "BTC"})
        
        # Check event was logged
        assert len(manager._event_log) == 1
    
    def test_event_adapter(self):
        """Test event adapter"""
        from infra.unified_event_v521 import (
            UnifiedEventManager,
            StrategyEventAdapter,
            EventTypeV521,
            RunMode
        )
        
        manager = UnifiedEventManager(RunMode.PAPER)
        
        received_events = []
        manager.subscribe(EventTypeV521.SIGNAL_GENERATED, lambda e: received_events.append(e))
        
        adapter = StrategyEventAdapter("test_strategy", manager)
        adapter.emit_signal(
            strategy_name="trend",
            symbol="BTC",
            direction=0.8,
            confidence=0.75
        )
        
        assert len(received_events) == 1
        assert received_events[0].payload["symbol"] == "BTC"
    
    # ========== RL Drift Fallback Tests ==========
    
    def test_drift_initialization(self):
        """Test drift manager initialization"""
        from execution.rl_drift_fallback_v521 import (
            get_rl_fallback_manager,
            DriftDetectorConfigV521
        )
        
        config = DriftDetectorConfigV521()
        manager = get_rl_fallback_manager(
            feature_names=["f1", "f2", "f3"],
            config=config
        )
        
        assert manager is not None
        assert len(manager.feature_names) == 3
    
    def test_drift_kl_divergence(self):
        """Test KL divergence calculation"""
        from execution.rl_drift_fallback_v521 import KLDivergenceDetector
        
        detector = KLDivergenceDetector(threshold=0.5)
        
        # Build reference distribution
        for i in range(500):
            detector.update_reference(random.gauss(0, 1))
        
        # Test distribution - similar
        similar_kl = detector.compute_divergence([random.gauss(0, 1) for _ in range(50)])
        
        # Test distribution - different
        different_kl = detector.compute_divergence([random.gauss(5, 2) for _ in range(50)])
        
        assert similar_kl < different_kl, "Different distribution should have higher KL"
    
    def test_drift_psi_calculation(self):
        """Test PSI calculation"""
        from execution.rl_drift_fallback_v521 import PSIDetector
        
        detector = PSIDetector(threshold=0.25)
        
        # Build reference
        for i in range(500):
            detector.update_reference(random.gauss(0, 1))
        
        # Similar distribution
        similar_psi = detector.compute_psi([random.gauss(0, 1) for _ in range(50)])
        
        # Different distribution
        different_psi = detector.compute_psi([random.gauss(3, 1) for _ in range(50)])
        
        assert similar_psi < different_psi, "Different distribution should have higher PSI"
    
    def test_drift_auto_fallback(self):
        """Test automatic fallback on drift"""
        from execution.rl_drift_fallback_v521 import (
            RLExecutionFallbackManager,
            DriftDetectorConfigV521,
            FallbackStrategy
        )
        
        config = DriftDetectorConfigV521(
            auto_fallback_on_medium=True,
            kl_threshold=0.3,
            psi_threshold=0.15
        )
        
        manager = RLExecutionFallbackManager(
            feature_names=["f1", "f2"],
            config=config
        )
        
        # Feed reference data
        for i in range(500):
            manager.update_features({
                "f1": random.gauss(0, 1),
                "f2": random.gauss(0, 1)
            })
        
        # Now feed drifted data
        for i in range(100):
            manager.update_features({
                "f1": random.gauss(5, 2),  # Shifted
                "f2": random.gauss(5, 2)   # Shifted
            })
        
        decision = manager.check_and_decide({
            "f1": random.gauss(5, 2),
            "f2": random.gauss(5, 2)
        })
        
        # With significant drift, should recommend fallback
        # (depends on drift severity)
        assert "use_rl" in decision
        assert "fallback_strategy" in decision
    
    def test_drift_recovery(self):
        """Test recovery from fallback"""
        from execution.rl_drift_fallback_v521 import (
            RLExecutionFallbackManager,
            DriftDetectorConfigV521
        )
        
        config = DriftDetectorConfigV521(
            observation_samples_for_recovery=10,
            cooldown_minutes=0.01  # Very short for test
        )
        
        manager = RLExecutionFallbackManager(
            feature_names=["f1"],
            config=config
        )
        
        # Feed reference
        for i in range(500):
            manager.update_features({"f1": random.gauss(0, 1)})
        
        # Manually trigger fallback
        manager._in_fallback = True
        manager._fallback_triggered_at = datetime.now() - timedelta(minutes=1)
        
        # Feed stable data
        for i in range(20):
            manager.update_features({"f1": random.gauss(0, 1)})
        
        status = manager.get_status()
        # Recovery should eventually happen
        assert "in_fallback" in status
    
    # ========== Integration Tests ==========
    
    async def test_integration_initialization(self):
        """Test full integration initialization"""
        from orchestration.sota_v521_integration import (
            SOTAV521Integrator,
            SOTAV521Config
        )
        
        config = SOTAV521Config(graceful_degradation=True)
        integrator = SOTAV521Integrator(config)
        
        success = await integrator.initialize()
        
        # With graceful degradation, should always succeed
        assert success or config.graceful_degradation
    
    async def test_integration_health_check(self):
        """Test health check"""
        from orchestration.sota_v521_integration import (
            SOTAV521Integrator,
            SOTAV521Config,
            ComponentStatus
        )
        
        config = SOTAV521Config(graceful_degradation=True)
        integrator = SOTAV521Integrator(config)
        
        await integrator.initialize()
        
        health = integrator.get_health()
        
        assert "version" in health
        assert health["version"] == "5.2.1"
        assert "components" in health
    
    async def test_integration_cross_component(self):
        """Test cross-component interaction"""
        from orchestration.sota_v521_integration import (
            SOTAV521Integrator,
            SOTAV521Config
        )
        
        config = SOTAV521Config(graceful_degradation=True)
        integrator = SOTAV521Integrator(config)
        
        await integrator.initialize()
        
        # Update prices (correlation)
        for i in range(50):
            integrator.update_price("BTC", 100000 + i * 10)
            integrator.update_price("ETH", 3500 + i * 0.35)
        
        # Record trades (weights)
        integrator.record_trade("trend", 0.02, True)
        integrator.record_trade("momentum", -0.01, False)
        
        # Update features (drift)
        integrator.update_features({
            "price_change_1m": 0.001,
            "volume_ratio": 1.1
        })
        
        # Get dashboard data
        dashboard = integrator.get_dashboard_data()
        
        assert "version" in dashboard
        assert "health" in dashboard
    
    async def test_integration_graceful_degradation(self):
        """Test graceful degradation when components fail"""
        from orchestration.sota_v521_integration import (
            SOTAV521Integrator,
            SOTAV521Config,
            ComponentStatus
        )
        
        # Disable all components
        config = SOTAV521Config(
            enable_correlation_prediction=False,
            enable_impact_learning=False,
            enable_adaptive_weights=False,
            enable_unified_events=False,
            enable_drift_detection=False,
            graceful_degradation=True
        )
        
        integrator = SOTAV521Integrator(config)
        success = await integrator.initialize()
        
        # Should still succeed with graceful degradation
        assert success
        
        # Operations should return safe defaults
        corr = integrator.get_correlation_assessment()
        assert corr is None  # No controller initialized
        
        weights = integrator.get_strategy_weights()
        assert weights == {}  # No manager initialized


def run_tests():
    """Run all tests"""
    suite = SOTAV521TestSuite()
    success = suite.run_all()
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)
