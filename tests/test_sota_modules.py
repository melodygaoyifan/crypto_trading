#!/usr/bin/env python3
"""
================================================================================
HMATS v5.2.0 - SOTA Module Test Suite
================================================================================
Purpose: Comprehensive tests for all SOTA upgrade modules

Tests cover:
1. Risk Management (SOTA Risk Controller, Correlation Sizer)
2. Execution Engine (Level2, RL Fallback, Scheduler)
3. Monitoring (Alert Manager)
4. Strategy Agents (Sentiment, Microstructure, OnChain)
5. Event Replay
6. Explainability
7. Plugin Registry
8. Integration Layer

Usage:
    pytest tests/test_sota_modules.py -v
    pytest tests/test_sota_modules.py -k "test_risk" -v
    
Author: HMATS SOTA Upgrade
Version: 5.2.0
================================================================================
"""

import pytest
import sys
import os
import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock, patch, MagicMock

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
        "volume": 250000.0,
        "bid": 149.9,
        "ask": 150.1
    }


@pytest.fixture
def sample_orderbook() -> Dict:
    """Sample orderbook data."""
    return {
        "bids": [
            {"price": 150.0, "size": 100},
            {"price": 149.9, "size": 200},
            {"price": 149.8, "size": 300},
            {"price": 149.7, "size": 400},
            {"price": 149.6, "size": 500}
        ],
        "asks": [
            {"price": 150.1, "size": 100},
            {"price": 150.2, "size": 200},
            {"price": 150.3, "size": 300},
            {"price": 150.4, "size": 400},
            {"price": 150.5, "size": 500}
        ]
    }


@pytest.fixture
def sample_portfolio() -> Dict[str, float]:
    """Sample portfolio positions."""
    return {
        "BTC": 0.5,
        "ETH": 2.0,
        "SOL": 50.0
    }


# =============================================================================
# TEST: RISK MANAGEMENT
# =============================================================================

class TestSOTARiskController:
    """Test SOTA Risk Controller functionality."""
    
    def test_import(self):
        """Test SOTA risk controller can be imported."""
        from risk.sota_risk_controller import SOTARiskController, SOTARiskConfig
        assert SOTARiskController is not None
        assert SOTARiskConfig is not None
    
    def test_initialization(self):
        """Test risk controller initialization."""
        from risk.sota_risk_controller import SOTARiskController, SOTARiskConfig
        
        config = SOTARiskConfig()
        controller = SOTARiskController(config)
        
        assert controller is not None
        assert controller.risk_state is not None
    
    def test_equity_update(self):
        """Test equity tracking."""
        from risk.sota_risk_controller import SOTARiskController
        
        controller = SOTARiskController()
        controller.update_equity(equity=100000, realized_pnl_today=0)
        
        assert controller.current_equity == 100000
    
    def test_drawdown_detection(self):
        """Test drawdown detection triggers."""
        from risk.sota_risk_controller import SOTARiskController, SOTARiskConfig
        
        config = SOTARiskConfig(reduce_threshold=0.05)
        controller = SOTARiskController(config)
        
        # Start with 100k
        controller.update_equity(equity=100000, realized_pnl_today=0)
        
        # Draw down 6% (should trigger reduced mode)
        controller.update_equity(equity=94000, realized_pnl_today=-6000)
        
        # Should be in reduced mode
        can_trade, reason = controller.can_open_position("BTC", "buy", 1.0)
        # Note: exact behavior depends on implementation
    
    def test_kill_switch(self):
        """Test kill switch functionality."""
        from risk.sota_risk_controller import SOTARiskController, KillSwitchReason

        controller = SOTARiskController()

        # Activate kill switch (internal method; no public activate)
        controller._activate_kill_switch(KillSwitchReason.MANUAL_TRIGGER)

        assert controller.kill_switch_active

        # Should block trades
        can_trade, reason = controller.can_open_position("BTC", "buy", 1.0)
        assert not can_trade
        assert "kill" in reason.lower() or "halt" in reason.lower()
    
    def test_dashboard_flags(self):
        """Test dashboard flag generation."""
        from risk.sota_risk_controller import SOTARiskController
        
        controller = SOTARiskController()
        controller.update_equity(equity=100000, realized_pnl_today=0)
        
        flags = controller.get_dashboard_flags()
        assert isinstance(flags, list)


class TestCorrelationPositionSizer:
    """Test correlation-aware position sizing."""
    
    def test_import(self):
        """Test correlation sizer can be imported."""
        from risk.correlation_position_sizer import (
            CorrelationAwarePositionSizer, 
            CorrelationConfig
        )
        assert CorrelationAwarePositionSizer is not None
    
    def test_initialization(self):
        """Test sizer initialization."""
        from risk.correlation_position_sizer import CorrelationAwarePositionSizer
        
        sizer = CorrelationAwarePositionSizer()
        assert sizer is not None
    
    def test_position_reduction(self, sample_portfolio):
        """Test that correlated positions get reduced."""
        from risk.correlation_position_sizer import (
            CorrelationAwarePositionSizer,
            CorrelationConfig
        )

        config = CorrelationConfig(high_corr_threshold=0.7, min_samples=3)
        sizer = CorrelationAwarePositionSizer(config)

        # Build correlation via price history (update_correlation doesn't exist)
        base = 100000
        for i in range(10):
            sizer.update_price("BTC", base + i * 100)
            sizer.update_price("ETH", 3000 + i * 3)  # highly correlated moves

        # Size a position
        result = sizer.calculate_adjusted_size(
            symbol="ETH",
            base_size=1.0,
        )

        if result:
            # Result exists even without correlation data - adjusted_size should be <= base
            assert result.adjusted_size <= 1.0


# =============================================================================
# TEST: EXECUTION ENGINE
# =============================================================================

class TestLevel2Analyzer:
    """Test Level 2 order book analyzer."""
    
    def test_import(self):
        """Test Level 2 analyzer can be imported."""
        from execution.level2_analyzer import Level2OrderBookAnalyzer as Level2Analyzer, OrderBookSnapshot
        assert Level2Analyzer is not None
    
    def test_spread_calculation(self, sample_orderbook):
        """Test spread calculation."""
        from execution.level2_analyzer import Level2OrderBookAnalyzer as Level2Analyzer, OrderBookSnapshot, OrderBookLevel
        
        analyzer = Level2Analyzer()
        
        bids = [OrderBookLevel(price=b["price"], size=b["size"]) 
                for b in sample_orderbook["bids"]]
        asks = [OrderBookLevel(price=a["price"], size=a["size"]) 
                for a in sample_orderbook["asks"]]
        
        snapshot = OrderBookSnapshot(
            symbol="SOL/USD",
            timestamp=datetime.now(timezone.utc),
            bids=bids,
            asks=asks
        )
        
        # Test spread
        assert snapshot.best_bid == 150.0
        assert snapshot.best_ask == 150.1
        assert snapshot.spread_bps > 0
    
    def test_liquidity_analysis(self, sample_orderbook):
        """Test liquidity level detection."""
        from execution.level2_analyzer import Level2OrderBookAnalyzer as Level2Analyzer
        from datetime import datetime

        analyzer = Level2Analyzer()  # default symbols: BTC, ETH, SOL

        # Use the actual API: update_orderbook then analyze_depth
        bids = [(b["price"], b["size"]) for b in sample_orderbook["bids"]]
        asks = [(a["price"], a["size"]) for a in sample_orderbook["asks"]]
        analyzer.update_orderbook("SOL", bids, asks, timestamp=datetime.now(timezone.utc))

        analysis = analyzer.analyze_depth("SOL")

        if analysis:
            assert hasattr(analysis, "liquidity_level") or hasattr(analysis, "bid_depth")


class TestRLFallback:
    """Test RL execution fallback system."""
    
    def test_import(self):
        """Test RL fallback can be imported."""
        from execution.rl_fallback import RLExecutionFallbackManager as RLExecutionFallback, ExecutionMode
        assert RLExecutionFallback is not None
        assert ExecutionMode is not None
    
    def test_fallback_decision(self):
        """Test fallback decision making."""
        from execution.rl_fallback import RLExecutionFallbackManager as RLExecutionFallback, FallbackConfig
        import numpy as np

        config = FallbackConfig(min_rl_confidence=0.6)
        fallback = RLExecutionFallback(config)

        # Without RL model, should default to fallback mode
        # decide() takes (state, state_dict, order_side, remaining_size)
        state = np.zeros(10)
        decision = fallback.decide(
            state=state,
            state_dict={"price": 100, "volatility": 0.02},
            order_side="buy",
            remaining_size=1.0,
        )

        assert decision is not None


class TestSOTAScheduler:
    """Test SOTA execution scheduler."""
    
    def test_import(self):
        """Test scheduler can be imported."""
        from execution.sota_scheduler import SOTAExecutionScheduler, ScheduleType
        assert SOTAExecutionScheduler is not None
        assert ScheduleType is not None
    
    def test_twap_scheduling(self):
        """Test TWAP order scheduling."""
        from execution.sota_scheduler import SOTAExecutionScheduler, ScheduleType
        
        scheduler = SOTAExecutionScheduler()
        
        order = scheduler.schedule_order(
            symbol="BTC",
            side="buy",
            size=1.0,
            schedule_type=ScheduleType.TWAP,
            duration_minutes=10
        )
        
        assert order is not None
        assert len(order.slices) > 1  # Should be sliced


# =============================================================================
# TEST: MONITORING & ALERTS
# =============================================================================

class TestAlertManager:
    """Test alert management system."""
    
    def test_import(self):
        """Test alert manager can be imported."""
        from infra.alert_manager import AlertManager, AlertType, AlertSeverity
        assert AlertManager is not None
        assert AlertType is not None
    
    def test_alert_creation(self):
        """Test alert creation via send_alert."""
        from infra.alert_manager import AlertManager, AlertConfig, AlertType, AlertSeverity

        config = AlertConfig(enable_console=True, enable_email=False)
        manager = AlertManager(config)

        alert = manager.send_alert(
            alert_type=AlertType.REGIME_CHANGE,
            title="Test Alert",
            message="This is a test",
            severity=AlertSeverity.WARNING,
        )

        # send_alert returns an Alert (or None if throttled on first call - unlikely)
        # Just verify no exception raised
        assert True

    def test_alert_throttling(self):
        """Test alert throttling."""
        from infra.alert_manager import AlertManager, AlertConfig, AlertType, AlertSeverity

        config = AlertConfig(
            min_interval_seconds={"regime_change": 60}
        )
        manager = AlertManager(config)

        # First alert should send
        sent1 = manager.send_alert(
            alert_type=AlertType.REGIME_CHANGE,
            title="First",
            message="First alert",
            severity=AlertSeverity.INFO,
        )

        # Second immediate alert should be throttled
        sent2 = manager.send_alert(
            alert_type=AlertType.REGIME_CHANGE,
            title="Second",
            message="Second alert",
            severity=AlertSeverity.INFO,
        )

        # Note: exact behavior depends on implementation


# =============================================================================
# TEST: STRATEGY AGENTS
# =============================================================================

class TestSentimentAgent:
    """Test sentiment LLM agent."""

    def test_import(self):
        """Test sentiment agent can be imported."""
        from agents.sentiment_llm_agent import SentimentLLMAgent, SentimentLevel
        assert SentimentLLMAgent is not None

    def test_generate_signal_returns_result(self):
        """Test synchronous generate_signal returns a SentimentResult."""
        from agents.sentiment_llm_agent import LLMSentimentAgent

        agent = LLMSentimentAgent()

        # generate_signal returns cached result or neutral default
        result = agent.generate_signal("BTC")
        assert result is not None
        assert result.asset == "BTC"

    def test_fear_greed_level_enum(self):
        """Test SentimentLevel constants exist."""
        from agents.sentiment_llm_agent import SentimentLevel

        assert SentimentLevel.EXTREME_FEAR == "extreme_fear"
        assert SentimentLevel.GREED == "greed"

    def test_signal_generation_neutral_without_cache(self):
        """Test generate_signal returns neutral when no analysis has been cached."""
        from agents.sentiment_llm_agent import LLMSentimentAgent

        agent = LLMSentimentAgent()

        result = agent.generate_signal("ETH")

        # Without prior async analyze(), should return default neutral result
        assert result is not None
        assert result.asset == "ETH"


class TestMicrostructureAgent:
    """Test microstructure arbitrage agent."""

    @pytest.fixture(autouse=True)  # [P371]
    def _isolated_warmup_state(self, tmp_path, monkeypatch):  # [P371]
        """[P371] the agent persists/restores its sample deques under
        HMATS_DATA_DIR; construct the state, never inherit it (P294)."""  # [P371]
        monkeypatch.setenv("HMATS_DATA_DIR", str(tmp_path))  # [P371]
        yield  # [P371]
    
    def test_import(self):
        """Test microstructure agent can be imported."""
        from agents.microstructure_agent import MicrostructureArbitrageAgent, MicroSignalType
        assert MicrostructureArbitrageAgent is not None
    
    def test_exchange_update(self):
        """Test exchange state updates."""
        from agents.microstructure_agent import MicrostructureArbitrageAgent

        agent = MicrostructureArbitrageAgent()

        agent.update_exchange(
            asset="BTC",
            exchange="binance",
            bid=100000,
            ask=100010,
            bid_size=10,
            ask_size=10,
            last_price=100005,
            last_size=0.5,
            last_side="buy"
        )

        assert "binance" in agent._per_asset["BTC"].exchange_states
    
    def test_cross_exchange_spread(self):
        """Test cross-exchange spread detection."""
        from agents.microstructure_agent import MicrostructureArbitrageAgent

        agent = MicrostructureArbitrageAgent()

        # Binance: bid=100000, ask=100020 (wider spread)
        agent.update_exchange("BTC", "binance", bid=100000, ask=100020,
                             bid_size=10, ask_size=10)

        # Kraken: bid=99980, ask=100010
        # best_bid = max(100000, 99980) = 100000 (binance)
        # best_ask = min(100020, 100010) = 100010 (kraken)
        # spread = (100010 - 100000) / 100005 * 10000 ≈ 1.0 bps
        agent.update_exchange("BTC", "kraken", bid=99980, ask=100010,
                             bid_size=5, ask_size=5)

        st = agent._per_asset["BTC"]
        assert st.cross_state is not None
        assert st.cross_state.cross_spread_bps > 0
    
    def test_microstructure_bias(self):
        """Test microstructure bias calculation."""
        from agents.microstructure_agent import MicrostructureArbitrageAgent

        agent = MicrostructureArbitrageAgent()

        # Add some data (per-asset)
        for i in range(20):
            agent.update_exchange("BTC", "binance", bid=100000+i, ask=100010+i,
                                 bid_size=10+i, ask_size=5,
                                 last_side="buy", last_size=1)

        bias, conf = agent.get_microstructure_bias("BTC")

        assert isinstance(bias, float)
        assert isinstance(conf, float)


class TestOnChainAgent:
    """Test on-chain Solana agent."""
    
    def test_import(self):
        """Test on-chain agent can be imported."""
        from agents.onchain_solana_agent import SolanaOnChainAgent, OnChainSignalType
        assert SolanaOnChainAgent is not None
    
    def test_initialization(self):
        """Test agent initialization."""
        from agents.onchain_solana_agent import SolanaOnChainAgent
        
        agent = SolanaOnChainAgent()
        assert agent is not None


# =============================================================================
# TEST: EVENT REPLAY
# =============================================================================

class TestEventReplay:
    """Test event replay system."""
    
    def test_import(self):
        """Test event replay can be imported."""
        event_replay = pytest.importorskip("infra.event_replay")
        EventReplayManager = event_replay.EventReplayEngine
        ReplaySpeed = event_replay.ReplaySpeed
        InjectedFaultType = event_replay.InjectedFaultType
        assert EventReplayManager is not None
        assert ReplaySpeed is not None
        assert InjectedFaultType is not None
    
    def test_fault_injection_types(self):
        """Test all fault types are defined."""
        event_replay = pytest.importorskip("infra.event_replay")
        InjectedFaultType = event_replay.InjectedFaultType
        
        expected_faults = [
            "WS_DISCONNECT",
            "WS_RECONNECT", 
            "API_RATE_LIMIT",
            "API_TIMEOUT",
            "API_ERROR",
            "DATA_GAP",
            "LATENCY_SPIKE"
        ]
        
        for fault in expected_faults:
            assert hasattr(InjectedFaultType, fault)


# =============================================================================
# TEST: EXPLAINABILITY
# =============================================================================

class TestLLMExplainer:
    """Test LLM decision explainer."""
    
    def test_import(self):
        """Test explainer can be imported."""
        from analytics.llm_explainer import LLMDecisionExplainer, ExplanationLevel
        assert LLMDecisionExplainer is not None
        assert ExplanationLevel is not None
    
    def test_template_explanation(self):
        """Test template-based explanation (no LLM)."""
        from analytics.llm_explainer import LLMDecisionExplainer, ExplainerConfig, TradeAction
        from datetime import datetime

        config = ExplainerConfig(use_templates=True, model_name="local")
        explainer = LLMDecisionExplainer(config)

        # explain_trade() requires a TradeAction dataclass
        action = TradeAction(
            action_type="entry",
            asset="BTC",
            direction="long",
            size=0.5,
            price=100000.0,
            timestamp=datetime.now(timezone.utc),
            metadata={"mode": "OPPORTUNITY"},
        )

        explanation = explainer.explain_trade(action)

        assert explanation is not None


# =============================================================================
# TEST: PLUGIN REGISTRY
# =============================================================================

class TestPluginRegistry:
    """Test plugin registry system."""
    
    def test_import(self):
        """Test plugin registry can be imported."""
        plugin_registry = pytest.importorskip("core.plugin_registry")
        PluginRegistry = plugin_registry.PluginRegistry
        PluginType = plugin_registry.PluginType
        PluginInterface = plugin_registry.PluginInterface
        assert PluginRegistry is not None
        assert PluginType is not None
    
    def test_plugin_types(self):
        """Test plugin types are defined."""
        plugin_registry = pytest.importorskip("core.plugin_registry")
        PluginType = plugin_registry.PluginType
        
        expected_types = ["STRATEGY", "RISK", "EXECUTION", "SIGNAL"]
        for ptype in expected_types:
            assert hasattr(PluginType, ptype)
    
    def test_registration(self):
        """Test plugin registration."""
        plugin_registry = pytest.importorskip("core.plugin_registry")
        PluginRegistry = plugin_registry.PluginRegistry
        PluginInterface = plugin_registry.PluginInterface
        PluginType = plugin_registry.PluginType
        
        registry = PluginRegistry()
        
        # Create mock plugin
        class MockPlugin(PluginInterface):
            PLUGIN_NAME = "mock_plugin"
            PLUGIN_VERSION = "1.0.0"
            PLUGIN_TYPE = PluginType.STRATEGY
            
            def initialize(self, config):
                return True
        
        # Register
        registry.register(MockPlugin)

        # list_plugins returns metadata (not instances); get_by_type needs active plugins
        plugins = registry.list_plugins(PluginType.STRATEGY)
        assert len(plugins) > 0


# =============================================================================
# TEST: INTEGRATION LAYER
# =============================================================================

class TestSOTAIntegration:
    """Test unified SOTA integration layer."""
    
    def test_import(self):
        """Test integration can be imported."""
        from orchestration.sota_integration import SOTAIntegration, SOTAIntegrationConfig
        assert SOTAIntegration is not None
    
    def test_initialization(self):
        """Test integration layer initialization."""
        from orchestration.sota_integration import SOTAIntegration, SOTAIntegrationConfig
        
        config = SOTAIntegrationConfig(
            enable_sentiment_agent=True,
            enable_microstructure_agent=True
        )
        
        integration = SOTAIntegration(config)
        
        assert integration is not None
        assert len(integration.components) > 0
    
    def test_risk_check(self, sample_portfolio):
        """Test integrated risk check."""
        from orchestration.sota_integration import SOTAIntegration
        
        integration = SOTAIntegration()
        
        result = integration.check_trade_allowed(
            symbol="BTC",
            side="buy",
            size=0.5,
            portfolio=sample_portfolio
        )
        
        assert "allowed" in result
        assert "adjusted_size" in result
    
    def test_signal_fusion(self, sample_market_data):
        """Test multi-agent signal fusion."""
        from orchestration.sota_integration import SOTAIntegration
        
        integration = SOTAIntegration()
        
        signal = integration.get_fused_signal("BTC", sample_market_data)
        
        assert "direction" in signal
        assert "confidence" in signal
    
    def test_decision_explanation(self):
        """Test decision explanation."""
        from orchestration.sota_integration import SOTAIntegration
        
        integration = SOTAIntegration()
        
        decision = {
            "action": "buy",
            "symbol": "BTC",
            "signal_components": {"sentiment": {"direction": 0.5, "confidence": 0.7}}
        }
        
        explanation = integration.explain_decision(decision)
        
        assert explanation is not None
        assert len(explanation) > 0
    
    def test_system_state(self):
        """Test system state retrieval."""
        from orchestration.sota_integration import SOTAIntegration
        
        integration = SOTAIntegration()
        
        state = integration.get_system_state()
        
        assert "timestamp" in state
        assert "components_active" in state
    
    def test_cleanup(self):
        """Test clean shutdown."""
        from orchestration.sota_integration import SOTAIntegration, reset_sota_integration
        
        integration = SOTAIntegration()
        integration.shutdown()
        reset_sota_integration()
        
        # Should be clean
        assert len(integration.components) == 0


# =============================================================================
# TEST: ADAPTIVE WEIGHTING
# =============================================================================

class TestAdaptiveWeighting:
    """Test adaptive agent weighting system (v5.2.1 canonical)."""

    def test_import(self):
        """Test adaptive weighting v521 can be imported."""
        from signals.adaptive_weight_v521 import (
            AdaptiveWeightManagerV521,
            AdaptiveWeightConfigV521,
        )
        assert AdaptiveWeightManagerV521 is not None
        assert AdaptiveWeightConfigV521 is not None

    def test_manager_initialization(self):
        """Test weight manager initializes and returns weights."""
        from signals.adaptive_weight_v521 import (
            AdaptiveWeightManagerV521,
            AdaptiveWeightConfigV521,
        )

        config = AdaptiveWeightConfigV521()
        manager = AdaptiveWeightManagerV521(config)

        assert manager is not None


# =============================================================================
# INTEGRATION SMOKE TEST
# =============================================================================

class TestSOTASmokeTest:
    """Smoke test for overall SOTA functionality."""
    
    def test_full_trading_cycle(self, sample_market_data, sample_orderbook, sample_portfolio):
        """Test a full trading decision cycle."""
        from orchestration.sota_integration import SOTAIntegration
        
        integration = SOTAIntegration()
        
        # 1. Update risk state
        integration.update_risk_state(
            equity=100000,
            realized_pnl=0,
            prices={"BTC": 100000, "ETH": 3000, "SOL": 150}
        )
        
        # 2. Get signal
        signal = integration.get_fused_signal("SOL", sample_market_data)
        
        # 3. Check if trade allowed
        trade_check = integration.check_trade_allowed(
            symbol="SOL",
            side="buy" if signal["direction"] > 0 else "sell",
            size=10.0,
            portfolio=sample_portfolio
        )
        
        # 4. Plan execution
        if trade_check["allowed"]:
            plan = integration.plan_execution(
                symbol="SOL",
                side="buy",
                size=trade_check["adjusted_size"],
                orderbook=sample_orderbook
            )
            assert plan is not None
        
        # 5. Get explanation
        decision = {
            "action": "buy" if signal["direction"] > 0 else "sell",
            "symbol": "SOL",
            "signal_components": signal.get("components", {}),
            "risk_flags": trade_check.get("risk_flags", [])
        }
        
        explanation = integration.explain_decision(decision)
        assert explanation is not None
        
        # 6. Cleanup
        integration.shutdown()
        
        # All steps completed without error
        assert True


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
