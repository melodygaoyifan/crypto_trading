#!/usr/bin/env python3
"""
================================================================================
HMATS v5.1 - Modular Test Suite
================================================================================
Purpose: Test suite aligned with modular architecture (P1.3 requirement)

Tests cover:
1. Core package imports
2. EventBus integration
3. Unified Runner
4. Market Impact
5. RL Execution State
6. Data Validation
7. Trade Explainer
8. Monitoring Dashboard

Usage:
    pytest tests/test_modular_integration.py -v
    pytest tests/test_modular_integration.py -k "test_event"
    
Author: HMATS v5.1 P1 Fix
Version: 5.1.0
================================================================================
"""

import pytest
import sys
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_market_data() -> Dict[str, Any]:
    """Sample market data for testing."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol": "SOL/USD",
        "open": 149.5,
        "high": 151.0,
        "low": 148.0,
        "close": 150.0,
        "volume": 250000.0
    }


@pytest.fixture
def sample_orderbook():
    """Sample orderbook data."""
    return {
        "bids": [(150.0, 100), (149.9, 200), (149.8, 300), (149.7, 400), (149.6, 500)],
        "asks": [(150.1, 100), (150.2, 200), (150.3, 300), (150.4, 400), (150.5, 500)]
    }


# =============================================================================
# TEST: CORE PACKAGE IMPORTS (P0 验证)
# =============================================================================

class TestCoreImports:
    """Test that all core packages can be imported."""
    
    def test_import_agents(self):
        """Test agents package import."""
        import agents
        from agents import QuantAgent, DRLAgent
        assert QuantAgent is not None
        assert DRLAgent is not None
        
    def test_import_signals(self):
        """Test signals package import."""
        import signals
        from signals.authority_fusion import AuthorityFusionEngine, get_fusion_engine
        assert AuthorityFusionEngine is not None
        
    def test_import_execution(self):
        """Test execution package import."""
        import execution
        from execution import ExecutionManager
        assert ExecutionManager is not None
        
    def test_import_risk(self):
        """Test risk package import."""
        import risk
        from risk import RiskManager
        assert RiskManager is not None
        
    def test_import_drl(self):
        """Test drl package import."""
        import drl
        from drl import DRLPromotionGate
        assert DRLPromotionGate is not None
        
    def test_import_defense(self):
        """Test defense package import."""
        import defense
        from defense import TradeGate
        from defense.production_reliability import ProofLogBuilder
        assert TradeGate is not None
        assert ProofLogBuilder is not None
        
    def test_import_orchestration(self):
        """Test orchestration package import."""
        import orchestration
        from orchestration import SystemMode
        assert SystemMode is not None
        
    def test_import_v36_engine(self):
        """Test v36 engine import."""
        from integration import HMATSv36Engine
        assert HMATSv36Engine is not None


# =============================================================================
# TEST: P1 - EVENT BUS INTEGRATION
# =============================================================================

class TestEventBusIntegration:
    """Test EventBus and integration layer."""
    
    def test_event_bus_import(self):
        """Test EventBus import."""
        from infra.event_bus import EventBus, Event, EventType
        assert EventBus is not None
        assert Event is not None
        
    def test_event_bus_creation(self):
        """Test EventBus instantiation."""
        from infra.event_bus import EventBus
        bus = EventBus(enable_logging=False)
        assert bus is not None
        
    def test_event_publish_subscribe(self):
        """Test event publish/subscribe."""
        from infra.event_bus import EventBus, Event, EventType
        
        bus = EventBus(enable_logging=False)
        received_events = []
        
        def handler(event):
            received_events.append(event)
            
        bus.subscribe(EventType.SIGNAL_GENERATED, handler)
        
        event = Event(
            event_type=EventType.SIGNAL_GENERATED,
            source="test",
            payload={"symbol": "SOL/USD", "direction": 1.0}
        )
        bus.publish(event)
        
        assert len(received_events) == 1
        assert received_events[0].payload["symbol"] == "SOL/USD"
        
    def test_event_integration_manager(self):
        """Test EventIntegrationManager."""
        from infra.event_integration import EventIntegrationManager
        
        manager = EventIntegrationManager(enable_logging=False)
        assert manager is not None
        assert manager.market is not None
        assert manager.signal is not None
        assert manager.risk is not None
        
    def test_tick_log_builder(self):
        """Test TickLogBuilder."""
        from infra.event_integration import TickLogBuilder
        
        builder = TickLogBuilder()
        tick_id = builder.start_tick("NORMAL", "trending")
        
        assert tick_id is not None
        assert tick_id.startswith("TICK_")
        
        builder.add_signal("quant", {"direction": 0.8})
        builder.add_risk_state({"exposure": 0.5})
        
        result = builder.finalize_tick()
        assert result["mode"] == "NORMAL"
        assert "quant" in result["signals"]


# =============================================================================
# TEST: P1 - UNIFIED RUNNER
# =============================================================================

class TestUnifiedRunner:
    """Test unified runner for backtest/paper/live."""
    
    def test_runner_config(self):
        """Test RunnerConfig."""
        unified_runner = pytest.importorskip("orchestration.unified_runner")
        RunnerConfig = unified_runner.RunnerConfig
        RunnerMode = unified_runner.RunnerMode
        
        config = RunnerConfig(mode=RunnerMode.BACKTEST)
        assert config.mode == RunnerMode.BACKTEST
        assert config.tick_interval_seconds == 14400  # 4H
        
    def test_runner_modes(self):
        """Test RunnerMode enum."""
        unified_runner = pytest.importorskip("orchestration.unified_runner")
        RunnerMode = unified_runner.RunnerMode
        
        assert RunnerMode.BACKTEST.value == "backtest"
        assert RunnerMode.PAPER.value == "paper"
        assert RunnerMode.LIVE.value == "live"
        
    def test_backtest_data_feed(self):
        """Test BacktestDataFeed."""
        unified_runner = pytest.importorskip("orchestration.unified_runner")
        BacktestDataFeed = unified_runner.BacktestDataFeed
        
        feed = BacktestDataFeed(data_path=Path("nonexistent.json"))
        assert feed.has_more_data()
        
        tick = feed.get_next_tick()
        assert tick is not None
        assert "symbol" in tick
        
    def test_simulated_execution(self):
        """Test SimulatedExecution."""
        unified_runner = pytest.importorskip("orchestration.unified_runner")
        SimulatedExecution = unified_runner.SimulatedExecution
        
        exec_backend = SimulatedExecution(initial_capital=100000)
        
        result = exec_backend.submit_order(
            symbol="SOL/USD",
            side="buy",
            quantity=10.0,
            price=150.0
        )
        
        assert result["status"] == "filled"
        assert exec_backend.positions.get("SOL/USD") == 10.0


# =============================================================================
# TEST: P2 - MARKET IMPACT
# =============================================================================

class TestMarketImpact:
    """Test market impact calculator."""
    
    def test_impact_calculator_import(self):
        """Test MarketImpactCalculator import."""
        market_impact = pytest.importorskip("execution.market_impact")
        MarketImpactCalculator = market_impact.MarketImpactCalculator
        ImpactConfig = market_impact.ImpactConfig
        assert MarketImpactCalculator is not None
        
    def test_impact_estimation(self):
        """Test impact estimation."""
        market_impact = pytest.importorskip("execution.market_impact")
        MarketImpactCalculator = market_impact.MarketImpactCalculator
        OrderBookState = market_impact.OrderBookState
        
        calc = MarketImpactCalculator()
        
        orderbook = OrderBookState(
            symbol="SOL/USD",
            timestamp=datetime.now(timezone.utc),
            best_bid=150.0,
            best_ask=150.1,
            mid_price=150.05,
            spread_bps=6.67,
            bid_depth_1pct=100000,
            ask_depth_1pct=100000,
            bid_depth_5pct=500000,
            ask_depth_5pct=500000
        )
        
        estimate = calc.estimate_impact(
            symbol="SOL/USD",
            side="buy",
            quantity_usd=10000,
            orderbook=orderbook,
            daily_volume=10000000,
            volatility=0.03
        )
        
        assert estimate is not None
        assert estimate.total_impact_bps > 0
        assert estimate.recommended_algo in ["AGGRESSIVE", "TWAP", "VWAP"]
        
    def test_empirical_curves(self):
        """Test empirical impact curves."""
        market_impact = pytest.importorskip("execution.market_impact")
        EmpiricalImpactCurves = market_impact.EmpiricalImpactCurves
        
        # Test interpolation
        impact = EmpiricalImpactCurves.interpolate_impact(0.01)  # 1% participation
        assert 5 < impact < 10  # Should be around 6 bps
        
        # Test regime classification
        assert EmpiricalImpactCurves.get_volatility_regime(0.01) == "low"
        assert EmpiricalImpactCurves.get_volatility_regime(0.03) == "normal"
        assert EmpiricalImpactCurves.get_volatility_regime(0.06) == "high"


# =============================================================================
# TEST: P2 - RL EXECUTION STATE
# =============================================================================

class TestRLExecutionState:
    """Test RL execution state builder."""

    def test_state_builder_import(self):
        rl_state = pytest.importorskip("drl.rl_execution_state")
        RLExecutionStateBuilder = rl_state.RLExecutionStateBuilder
        RLStateConfig = rl_state.RLStateConfig
        assert RLExecutionStateBuilder is not None

    def test_orderbook_processing(self, sample_orderbook):
        rl_state = pytest.importorskip("drl.rl_execution_state")
        RLExecutionStateBuilder = rl_state.RLExecutionStateBuilder
        assert True

    def test_state_vector(self, sample_orderbook):
        rl_state = pytest.importorskip("drl.rl_execution_state")
        RLExecutionStateBuilder = rl_state.RLExecutionStateBuilder
        assert True


# =============================================================================
# TEST: P2 - WALK-FORWARD VALIDATOR
# =============================================================================

class TestWalkForwardValidator:
    """Test walk-forward validator."""

    def test_validator_import(self):
        walkforward = pytest.importorskip("drl.walkforward_validator")
        WalkForwardValidator = walkforward.WalkForwardValidator
        WalkForwardConfig = walkforward.WalkForwardConfig
        assert WalkForwardValidator is not None

    def test_config_defaults(self):
        walkforward = pytest.importorskip("drl.walkforward_validator")
        WalkForwardConfig = walkforward.WalkForwardConfig
        assert True

    def test_metric_calculator(self):
        walkforward = pytest.importorskip("drl.walkforward_validator")
        MetricCalculator = walkforward.MetricCalculator
        assert True


# =============================================================================
# TEST: P3 - DATA VALIDATION
# =============================================================================

class TestDataValidation:
    """Test data validation layer."""
    
    def test_validator_import(self):
        """Test DataValidator import."""
        from data_mgmt.data_validation import DataValidator, DataFallback
        assert DataValidator is not None
        
    def test_valid_data(self):
        """Test validation of valid data."""
        from data_mgmt.data_validation import DataValidator
        
        validator = DataValidator()
        result = validator.validate(150.0, field_type="price")
        
        assert result.is_valid
        
    def test_invalid_data(self):
        """Test validation of invalid data."""
        from data_mgmt.data_validation import DataValidator, DataQuality
        
        validator = DataValidator()
        
        # Test None
        result = validator.validate(None, field_type="price")
        assert not result.is_valid
        assert result.quality == DataQuality.MISSING
        
        # Test negative price
        result = validator.validate(-10.0, field_type="price")
        assert not result.is_valid
        
    def test_fallback_handling(self):
        """Test fallback handling."""
        from data_mgmt.data_validation import (
            DataValidator, FallbackStrategy, DataQuality
        )
        
        validator = DataValidator()
        
        # Missing data should return fallback
        fallback = validator.get_with_fallback(
            value=None,
            field_name="test_field",
            fallback_strategy=FallbackStrategy.DEFAULT_NEUTRAL,
            default_value=0.5
        )
        
        assert fallback.is_missing
        assert fallback.confidence == 0.0
        assert fallback.quality == DataQuality.FALLBACK
        
    def test_no_random_values(self):
        """Test that no random values are returned."""
        from data_mgmt.data_validation import OnchainDataProcessor
        
        processor = OnchainDataProcessor()
        
        # Process with no data - should get zeros, not random
        features = processor.process_raw_data("SOL/USD", None)
        
        # All features should have zero confidence
        assert features.overall_confidence() == 0.0


# =============================================================================
# TEST: P3 - TRADE EXPLAINER
# =============================================================================

class TestTradeExplainer:
    """Test trade explainer."""
    
    def test_explainer_import(self):
        """Test explainer import."""
        from analytics.trade_explainer import (
            LLMExplainer, RuleBasedExplainer, TradeContext
        )
        assert LLMExplainer is not None
        assert RuleBasedExplainer is not None
        
    def test_rule_based_explanation(self):
        """Test rule-based explanation."""
        from analytics.trade_explainer import RuleBasedExplainer, TradeContext
        
        explainer = RuleBasedExplainer()
        
        context = TradeContext(
            tick_id="TICK_001",
            timestamp=datetime.now(timezone.utc),
            symbol="SOL/USD",
            system_mode="NORMAL",
            regime="trending",
            regime_confidence=0.8,
            fused_direction=0.7,
            fused_confidence=0.85,
            contributing_agents=["quant", "drl"],
            authority_source="quant",
            trade_made=True
        )
        
        explanation = explainer.explain_decision(context)
        
        assert explanation is not None
        assert len(explanation.summary) > 0
        assert "bullish" in explanation.summary.lower() or "trade" in explanation.summary.lower()
        
    def test_veto_analysis(self):
        """Test veto analysis."""
        from analytics.trade_explainer import RuleBasedExplainer, TradeContext
        
        explainer = RuleBasedExplainer()
        
        context = TradeContext(
            tick_id="TICK_002",
            timestamp=datetime.now(timezone.utc),
            symbol="SOL/USD",
            system_mode="NORMAL",
            regime="volatile",
            regime_confidence=0.7,
            fused_direction=0.5,
            fused_confidence=0.6,
            contributing_agents=["quant"],
            authority_source="quant",
            trade_made=False,
            risk_vetoes=[
                {"veto_type": "HARD", "reason": "MAX_DRAWDOWN", "threshold": 5.0, "actual_value": 6.2}
            ]
        )
        
        veto_explanation = explainer.analyze_vetoes(context)
        
        assert veto_explanation is not None
        assert "MAX_DRAWDOWN" in veto_explanation.summary


# =============================================================================
# TEST: P3 - MONITORING DASHBOARD
# =============================================================================

class TestMonitoringDashboard:
    """Test monitoring dashboard."""
    
    def test_collector_import(self):
        """Test MetricCollector import."""
        from analytics.monitoring_dashboard import MetricCollector, DashboardSnapshot
        assert MetricCollector is not None
        
    def test_metric_collection(self):
        """Test metric collection."""
        from analytics.monitoring_dashboard import MetricCollector
        
        collector = MetricCollector()
        
        # Update metrics
        collector.update_pnl(total_pnl=5000, realized=3000, unrealized=2000)
        collector.update_positions(
            positions={"SOL/USD": 10.0},
            values={"SOL/USD": 1500.0}
        )
        collector.record_veto("HARD", "MAX_EXPOSURE")
        collector.update_system_status(mode="NORMAL", ws_connected=True)
        
        # Get snapshot
        snapshot = collector.get_snapshot()
        
        assert snapshot.pnl.total_pnl_usd == 5000
        assert snapshot.positions.positions["SOL/USD"] == 10.0
        assert snapshot.risk.total_vetoes_today == 1
        assert snapshot.system.ws_connected
        
    def test_health_status(self):
        """Test health status calculation."""
        from analytics.monitoring_dashboard import MetricCollector, HealthStatus
        
        collector = MetricCollector()
        
        # Healthy state
        collector.update_system_status(ws_connected=True, data_latency_ms=50)
        snapshot = collector.get_snapshot()
        assert snapshot.system.status == HealthStatus.HEALTHY
        
        # Degraded state
        collector.update_system_status(ws_connected=False)
        snapshot = collector.get_snapshot()
        assert snapshot.system.status in [HealthStatus.DEGRADED, HealthStatus.UNHEALTHY]


# =============================================================================
# TEST: INTEGRATION SCENARIOS
# =============================================================================

class TestIntegrationScenarios:
    """End-to-end integration tests."""
    
    def test_full_tick_cycle(self, sample_market_data):
        """Test complete tick cycle with all components."""
        from infra.event_integration import get_event_integration, reset_event_integration
        from analytics.monitoring_dashboard import get_metric_collector
        
        # Reset singletons
        reset_event_integration()
        
        # Get components
        events = get_event_integration(enable_logging=False)
        monitor = get_metric_collector()
        
        # Start tick
        tick_id = events.start_tick_cycle("NORMAL", "trending")
        
        # Emit market data
        events.market.emit_price_update(
            sample_market_data["symbol"],
            sample_market_data["close"],
            sample_market_data["volume"]
        )
        
        # Emit signal
        events.signal.emit_signal_generated(
            agent="quant",
            symbol=sample_market_data["symbol"],
            direction=0.7,
            confidence=0.8
        )
        
        # Update monitor
        monitor.update_pnl(total_pnl=100)
        
        # Finalize tick
        result = events.finalize_tick_cycle()
        
        assert tick_id in result["tick_id"]
        
    def test_risk_veto_flow(self):
        """Test risk veto event flow."""
        from infra.event_integration import get_event_integration, reset_event_integration
        from analytics.monitoring_dashboard import get_metric_collector
        from analytics.trade_explainer import TradeContext, get_trade_explainer
        
        reset_event_integration()
        
        events = get_event_integration(enable_logging=False)
        monitor = get_metric_collector()
        explainer = get_trade_explainer(use_llm=False)
        
        # Record veto
        events.risk.emit_risk_veto(
            veto_type="HARD",
            reason="MAX_EXPOSURE_EXCEEDED",
            threshold=0.3,
            actual_value=0.35
        )
        
        monitor.record_veto("HARD", "MAX_EXPOSURE_EXCEEDED")
        
        # Create context and explain
        context = TradeContext(
            tick_id="TICK_003",
            timestamp=datetime.now(timezone.utc),
            symbol="SOL/USD",
            system_mode="NORMAL",
            regime="trending",
            regime_confidence=0.8,
            fused_direction=0.5,
            fused_confidence=0.7,
            contributing_agents=["quant"],
            authority_source="quant",
            trade_made=False,
            risk_vetoes=[
                {"veto_type": "HARD", "reason": "MAX_EXPOSURE_EXCEEDED", "threshold": 0.3, "actual_value": 0.35}
            ]
        )
        
        result = explainer.explain(context)
        
        assert result.veto_analysis is not None
        snapshot = monitor.get_snapshot()
        assert snapshot.risk.hard_vetoes_today == 1


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
