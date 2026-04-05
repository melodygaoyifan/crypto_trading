"""
HMATS v5.2.1 SOTA Integration Module

Unified integration layer connecting all v5.2.1 SOTA enhancements:
1. Correlation Dynamic Risk Control with prediction
2. Market Impact Model with real orderbook training
3. Adaptive Strategy Weights (Sharpe/Calmar/Winrate)
4. Unified Event-Driven Architecture
5. RL Drift Detection with Safe Fallback

This module provides:
- Single initialization point for all v5.2.1 components
- Cross-component event wiring
- Unified status monitoring
- Graceful degradation when components unavailable
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
import traceback

logger = logging.getLogger(__name__)


class ComponentStatus(Enum):
    """Status of individual SOTA components"""
    NOT_INITIALIZED = "not_initialized"
    INITIALIZING = "initializing"
    RUNNING = "running"
    DEGRADED = "degraded"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class ComponentHealth:
    """Health status for a component"""
    name: str
    status: ComponentStatus
    last_update: datetime
    error_count: int = 0
    last_error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SOTAV521Config:
    """Configuration for v5.2.1 SOTA integration"""
    # Correlation Controller
    enable_correlation_prediction: bool = True
    correlation_jump_threshold: float = 3.0
    preemptive_reduction_enabled: bool = True
    
    # Market Impact
    enable_impact_learning: bool = True
    impact_urgency_default: float = 0.5
    
    # Adaptive Weights
    enable_adaptive_weights: bool = True
    weight_update_interval_hours: float = 4.0
    
    # Event System
    enable_unified_events: bool = True
    event_log_path: str = "logs/events"
    
    # RL Fallback
    enable_drift_detection: bool = True
    auto_fallback_enabled: bool = True
    
    # Integration
    health_check_interval_seconds: float = 60.0
    graceful_degradation: bool = True


class SOTAV521Integrator:
    """
    Main integration class for HMATS v5.2.1 SOTA features.
    
    Coordinates all v5.2.1 components and provides:
    - Unified initialization
    - Cross-component event wiring
    - Health monitoring
    - Graceful degradation
    """
    
    VERSION = "5.2.1"
    
    def __init__(self, config: Optional[SOTAV521Config] = None):
        self.config = config or SOTAV521Config()
        self.components: Dict[str, ComponentHealth] = {}
        self._initialized = False
        self._running = False
        
        # Component instances
        self._correlation_controller = None
        self._impact_calculator = None
        self._weight_manager = None
        self._event_manager = None
        self._fallback_manager = None
        
        # Callbacks
        self._trade_callbacks: List[Callable] = []
        self._alert_callbacks: List[Callable] = []
        
        logger.info(f"SOTA v{self.VERSION} Integrator created")
    
    async def initialize(self) -> bool:
        """Initialize all v5.2.1 components"""
        logger.info(f"Initializing HMATS v{self.VERSION} SOTA components...")
        
        success = True
        
        # 1. Initialize Unified Event System (first, others depend on it)
        if self.config.enable_unified_events:
            success &= await self._init_event_system()
        
        # 2. Initialize Correlation Controller
        if self.config.enable_correlation_prediction:
            success &= await self._init_correlation_controller()
        
        # 3. Initialize Market Impact Calculator
        if self.config.enable_impact_learning:
            success &= await self._init_impact_calculator()
        
        # 4. Initialize Adaptive Weights
        if self.config.enable_adaptive_weights:
            success &= await self._init_weight_manager()
        
        # 5. Initialize RL Fallback Manager
        if self.config.enable_drift_detection:
            success &= await self._init_fallback_manager()
        
        # Wire up cross-component events
        if success:
            await self._wire_events()
        
        self._initialized = success
        
        if success:
            logger.info(f"SOTA v{self.VERSION} initialization complete")
        else:
            logger.warning(f"SOTA v{self.VERSION} initialization partial - some components failed")
        
        return success
    
    async def _init_event_system(self) -> bool:
        """Initialize unified event system"""
        try:
            self._update_component_status("event_system", ComponentStatus.INITIALIZING)
            
            from infra.unified_event_v521 import init_event_driven_system, RunMode
            
            self._event_manager = init_event_driven_system(RunMode.PAPER)
            
            self._update_component_status(
                "event_system", 
                ComponentStatus.RUNNING,
                metrics={"adapters_registered": len(self._event_manager._adapters)}
            )
            
            logger.info("Unified Event System v5.2.1 initialized")
            return True
            
        except ImportError as e:
            logger.warning(f"Event system not available: {e}")
            self._update_component_status(
                "event_system", 
                ComponentStatus.FAILED,
                error=str(e)
            )
            return self.config.graceful_degradation
            
        except Exception as e:
            logger.error(f"Event system initialization failed: {e}")
            self._update_component_status(
                "event_system",
                ComponentStatus.FAILED,
                error=str(e)
            )
            return self.config.graceful_degradation
    
    async def _init_correlation_controller(self) -> bool:
        """Initialize correlation dynamic controller"""
        try:
            self._update_component_status("correlation_controller", ComponentStatus.INITIALIZING)
            
            from risk.correlation_dynamic_controller_v521 import (
                get_correlation_controller_v521,
                CorrelationDynamicsConfigV521
            )
            
            config = CorrelationDynamicsConfigV521(
                jump_zscore_threshold=self.config.correlation_jump_threshold,
                enable_prediction=self.config.enable_correlation_prediction
            )
            
            self._correlation_controller = get_correlation_controller_v521(config)
            
            self._update_component_status(
                "correlation_controller",
                ComponentStatus.RUNNING,
                metrics={"prediction_enabled": config.enable_prediction}
            )
            
            logger.info("Correlation Controller v5.2.1 initialized")
            return True
            
        except ImportError as e:
            logger.warning(f"Correlation controller not available: {e}")
            self._update_component_status(
                "correlation_controller",
                ComponentStatus.FAILED,
                error=str(e)
            )
            return self.config.graceful_degradation
            
        except Exception as e:
            logger.error(f"Correlation controller initialization failed: {e}")
            self._update_component_status(
                "correlation_controller",
                ComponentStatus.FAILED,
                error=str(e)
            )
            return self.config.graceful_degradation
    
    async def _init_impact_calculator(self) -> bool:
        """Initialize market impact calculator"""
        try:
            self._update_component_status("impact_calculator", ComponentStatus.INITIALIZING)
            
            from execution.market_impact_v521 import (
                get_impact_calculator_v521,
                MarketImpactConfigV521
            )
            
            config = MarketImpactConfigV521(
                enable_online_learning=self.config.enable_impact_learning
            )
            
            self._impact_calculator = get_impact_calculator_v521(config)
            
            self._update_component_status(
                "impact_calculator",
                ComponentStatus.RUNNING,
                metrics={"learning_enabled": config.enable_online_learning}
            )
            
            logger.info("Market Impact Calculator v5.2.1 initialized")
            return True
            
        except ImportError as e:
            logger.warning(f"Impact calculator not available: {e}")
            self._update_component_status(
                "impact_calculator",
                ComponentStatus.FAILED,
                error=str(e)
            )
            return self.config.graceful_degradation
            
        except Exception as e:
            logger.error(f"Impact calculator initialization failed: {e}")
            self._update_component_status(
                "impact_calculator",
                ComponentStatus.FAILED,
                error=str(e)
            )
            return self.config.graceful_degradation
    
    async def _init_weight_manager(self) -> bool:
        """Initialize adaptive weight manager"""
        try:
            self._update_component_status("weight_manager", ComponentStatus.INITIALIZING)
            
            from signals.adaptive_weight_v521 import (
                get_adaptive_weight_manager_v521,
                AdaptiveWeightConfigV521
            )
            
            config = AdaptiveWeightConfigV521(
                evaluation_interval_hours=self.config.weight_update_interval_hours
            )
            
            self._weight_manager = get_adaptive_weight_manager_v521(config)
            
            self._update_component_status(
                "weight_manager",
                ComponentStatus.RUNNING,
                metrics={"update_interval_hours": config.evaluation_interval_hours}
            )
            
            logger.info("Adaptive Weight Manager v5.2.1 initialized")
            return True
            
        except ImportError as e:
            logger.warning(f"Weight manager not available: {e}")
            self._update_component_status(
                "weight_manager",
                ComponentStatus.FAILED,
                error=str(e)
            )
            return self.config.graceful_degradation
            
        except Exception as e:
            logger.error(f"Weight manager initialization failed: {e}")
            self._update_component_status(
                "weight_manager",
                ComponentStatus.FAILED,
                error=str(e)
            )
            return self.config.graceful_degradation
    
    async def _init_fallback_manager(self) -> bool:
        """Initialize RL drift fallback manager"""
        try:
            self._update_component_status("fallback_manager", ComponentStatus.INITIALIZING)
            
            from execution.rl_drift_fallback_v521 import (
                get_rl_fallback_manager,
                DriftDetectorConfigV521
            )
            
            # Standard feature set for drift detection
            feature_names = [
                "price_change_1m", "price_change_5m", "price_change_15m",
                "volume_ratio", "spread_bps", "volatility_1h",
                "orderbook_imbalance", "trade_intensity"
            ]
            
            config = DriftDetectorConfigV521(
                auto_fallback_on_medium=self.config.auto_fallback_enabled
            )
            
            self._fallback_manager = get_rl_fallback_manager(
                feature_names=feature_names,
                config=config
            )
            
            self._update_component_status(
                "fallback_manager",
                ComponentStatus.RUNNING,
                metrics={
                    "features_tracked": len(feature_names),
                    "auto_fallback": config.auto_fallback_on_medium
                }
            )
            
            logger.info("RL Fallback Manager v5.2.1 initialized")
            return True
            
        except ImportError as e:
            logger.warning(f"Fallback manager not available: {e}")
            self._update_component_status(
                "fallback_manager",
                ComponentStatus.FAILED,
                error=str(e)
            )
            return self.config.graceful_degradation
            
        except Exception as e:
            logger.error(f"Fallback manager initialization failed: {e}")
            self._update_component_status(
                "fallback_manager",
                ComponentStatus.FAILED,
                error=str(e)
            )
            return self.config.graceful_degradation
    
    async def _wire_events(self) -> None:
        """Wire up cross-component events"""
        if not self._event_manager:
            logger.warning("Event manager not available, skipping event wiring")
            return
        
        try:
            from infra.unified_event_v521 import EventTypeV521
            
            # Correlation events -> Risk actions
            if self._correlation_controller:
                self._event_manager.subscribe(
                    EventTypeV521.CORRELATION_REGIME_CHANGE,
                    self._on_correlation_regime_change
                )
                self._event_manager.subscribe(
                    EventTypeV521.CORRELATION_JUMP,
                    self._on_correlation_jump
                )
            
            # Drift events -> Execution fallback
            if self._fallback_manager:
                self._event_manager.subscribe(
                    EventTypeV521.DRIFT_DETECTED,
                    self._on_drift_detected
                )
            
            # Weight adjustment events
            if self._weight_manager:
                self._event_manager.subscribe(
                    EventTypeV521.WEIGHT_ADJUSTED,
                    self._on_weight_adjusted
                )
            
            logger.info("Cross-component events wired successfully")
            
        except Exception as e:
            logger.warning(f"Event wiring failed: {e}")
    
    def _on_correlation_regime_change(self, event) -> None:
        """Handle correlation regime change events"""
        logger.info(f"Correlation regime changed: {event.payload}")
        for callback in self._alert_callbacks:
            try:
                callback("correlation_regime", event.payload)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def _on_correlation_jump(self, event) -> None:
        """Handle correlation jump events"""
        logger.warning(f"Correlation jump detected: {event.payload}")
        for callback in self._alert_callbacks:
            try:
                callback("correlation_jump", event.payload)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def _on_drift_detected(self, event) -> None:
        """Handle drift detection events"""
        logger.warning(f"RL drift detected: {event.payload}")
        for callback in self._alert_callbacks:
            try:
                callback("drift_detected", event.payload)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def _on_weight_adjusted(self, event) -> None:
        """Handle weight adjustment events"""
        logger.info(f"Strategy weights adjusted: {event.payload}")
    
    def _update_component_status(
        self,
        name: str,
        status: ComponentStatus,
        error: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update component health status"""
        if name not in self.components:
            self.components[name] = ComponentHealth(
                name=name,
                status=status,
                last_update=datetime.now()
            )
        
        health = self.components[name]
        health.status = status
        health.last_update = datetime.now()
        
        if error:
            health.error_count += 1
            health.last_error = error
        
        if metrics:
            health.metrics.update(metrics)
    
    # ========== Public API ==========
    
    def update_price(self, symbol: str, price: float) -> None:
        """Update price for correlation tracking"""
        if self._correlation_controller:
            try:
                self._correlation_controller.update_price(symbol, price)
            except Exception as e:
                logger.error(f"Price update failed: {e}")
    
    def update_orderbook(self, symbol: str, orderbook: Dict[str, Any]) -> None:
        """Update orderbook for impact calculation"""
        if self._impact_calculator:
            try:
                self._impact_calculator.update_orderbook(orderbook)
            except Exception as e:
                logger.error(f"Orderbook update failed: {e}")
    
    def record_trade(
        self,
        strategy: str,
        pnl_pct: float,
        is_winner: bool
    ) -> None:
        """Record trade for weight adjustment"""
        if self._weight_manager:
            try:
                self._weight_manager.record_trade(strategy, pnl_pct, is_winner)
            except Exception as e:
                logger.error(f"Trade record failed: {e}")
        
        # Notify callbacks
        for callback in self._trade_callbacks:
            try:
                callback(strategy, pnl_pct, is_winner)
            except Exception as e:
                logger.error(f"Trade callback failed: {e}")
    
    def update_features(self, features: Dict[str, float]) -> None:
        """Update features for drift detection"""
        if self._fallback_manager:
            try:
                self._fallback_manager.update_features(features)
            except Exception as e:
                logger.error(f"Feature update failed: {e}")
    
    def get_correlation_assessment(self) -> Optional[Dict[str, Any]]:
        """Get current correlation assessment"""
        if not self._correlation_controller:
            return None
        
        try:
            assessment = self._correlation_controller.compute_assessment()
            return {
                "regime": assessment.regime.value,
                "action": assessment.action.value,
                "position_scale": assessment.position_scale_factor,
                "avg_correlation": assessment.avg_correlation,
                "prediction": assessment.prediction,
                "jump_detected": assessment.jump_detected
            }
        except Exception as e:
            logger.error(f"Correlation assessment failed: {e}")
            return None
    
    def get_execution_plan(
        self,
        symbol: str,
        side: str,
        quantity_usd: float,
        orderbook: Dict[str, Any],
        urgency: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """Get execution plan with impact prediction"""
        if not self._impact_calculator:
            return None
        
        try:
            urgency = urgency or self.config.impact_urgency_default
            return self._impact_calculator.get_execution_plan(
                symbol=symbol,
                side=side,
                total_quantity_usd=quantity_usd,
                orderbook=orderbook,
                urgency=urgency
            )
        except Exception as e:
            logger.error(f"Execution plan failed: {e}")
            return None
    
    def get_strategy_weights(self) -> Dict[str, float]:
        """Get current strategy weights"""
        if not self._weight_manager:
            return {}
        
        try:
            return self._weight_manager.get_all_weights()
        except Exception as e:
            logger.error(f"Weight retrieval failed: {e}")
            return {}
    
    def check_rl_drift(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Check for RL drift and get execution decision"""
        if not self._fallback_manager:
            return {"use_rl": True, "fallback_strategy": None, "reason": "Drift detection unavailable"}
        
        try:
            return self._fallback_manager.check_and_decide(features)
        except Exception as e:
            logger.error(f"Drift check failed: {e}")
            return {"use_rl": False, "fallback_strategy": "TWAP", "reason": f"Error: {e}"}
    
    def can_add_position(
        self,
        symbol: str,
        existing_positions: List[str]
    ) -> tuple[bool, str]:
        """Check if new position can be added based on correlation"""
        if not self._correlation_controller:
            return True, "Correlation controller unavailable"
        
        try:
            return self._correlation_controller.can_add_position(symbol, existing_positions)
        except Exception as e:
            logger.error(f"Position check failed: {e}")
            return True, f"Error: {e}"
    
    def get_position_adjustment(
        self,
        symbol: str,
        base_size: float
    ) -> tuple[float, str]:
        """Get position size adjustment based on correlation regime"""
        if not self._correlation_controller:
            return base_size, "Correlation controller unavailable"
        
        try:
            return self._correlation_controller.get_position_adjustment(symbol, base_size)
        except Exception as e:
            logger.error(f"Position adjustment failed: {e}")
            return base_size, f"Error: {e}"
    
    # ========== Callbacks ==========
    
    def register_trade_callback(self, callback: Callable) -> None:
        """Register callback for trade events"""
        self._trade_callbacks.append(callback)
    
    def register_alert_callback(self, callback: Callable) -> None:
        """Register callback for alerts"""
        self._alert_callbacks.append(callback)
    
    # ========== Health & Status ==========
    
    def get_health(self) -> Dict[str, Any]:
        """Get health status of all components"""
        return {
            "version": self.VERSION,
            "initialized": self._initialized,
            "running": self._running,
            "timestamp": datetime.now().isoformat(),
            "components": {
                name: {
                    "status": health.status.value,
                    "last_update": health.last_update.isoformat(),
                    "error_count": health.error_count,
                    "last_error": health.last_error,
                    "metrics": health.metrics
                }
                for name, health in self.components.items()
            }
        }
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive data for dashboard display"""
        data = {
            "version": self.VERSION,
            "health": self.get_health(),
            "timestamp": datetime.now().isoformat()
        }
        
        # Correlation data
        if self._correlation_controller:
            try:
                data["correlation"] = self._correlation_controller.get_dashboard_data()
            except Exception as e:
                data["correlation"] = {"error": str(e)}
        
        # Impact data
        if self._impact_calculator:
            try:
                data["impact"] = self._impact_calculator.get_calibration_stats()
            except Exception as e:
                data["impact"] = {"error": str(e)}
        
        # Weight data
        if self._weight_manager:
            try:
                data["weights"] = self._weight_manager.get_dashboard_data()
            except Exception as e:
                data["weights"] = {"error": str(e)}
        
        # Fallback data
        if self._fallback_manager:
            try:
                data["fallback"] = self._fallback_manager.get_status()
            except Exception as e:
                data["fallback"] = {"error": str(e)}
        
        return data
    
    async def start(self) -> bool:
        """Start the integration layer"""
        if not self._initialized:
            success = await self.initialize()
            if not success and not self.config.graceful_degradation:
                return False
        
        self._running = True
        logger.info(f"SOTA v{self.VERSION} Integration started")
        return True
    
    async def stop(self) -> None:
        """Stop the integration layer"""
        self._running = False
        
        # Update all component statuses
        for name in self.components:
            self._update_component_status(name, ComponentStatus.STOPPED)
        
        logger.info(f"SOTA v{self.VERSION} Integration stopped")


# Singleton instance
_integrator_instance: Optional[SOTAV521Integrator] = None


def get_sota_v521_integrator(config: Optional[SOTAV521Config] = None) -> SOTAV521Integrator:
    """Get singleton instance of SOTA v5.2.1 integrator"""
    global _integrator_instance
    if _integrator_instance is None:
        _integrator_instance = SOTAV521Integrator(config)
    return _integrator_instance


async def init_sota_v521(config: Optional[SOTAV521Config] = None) -> SOTAV521Integrator:
    """Initialize and start SOTA v5.2.1 integration"""
    integrator = get_sota_v521_integrator(config)
    await integrator.start()
    return integrator


# ========== Quick Start Functions ==========

def quick_correlation_check() -> Dict[str, Any]:
    """Quick correlation assessment without async"""
    integrator = get_sota_v521_integrator()
    return integrator.get_correlation_assessment() or {"error": "Not initialized"}


def quick_weight_check() -> Dict[str, float]:
    """Quick strategy weight check without async"""
    integrator = get_sota_v521_integrator()
    return integrator.get_strategy_weights()


def quick_health_check() -> Dict[str, Any]:
    """Quick health check without async"""
    integrator = get_sota_v521_integrator()
    return integrator.get_health()


# ========== Example Usage ==========

if __name__ == "__main__":
    import json
    
    async def main():
        print("=" * 60)
        print("HMATS v5.2.1 SOTA Integration Example")
        print("=" * 60)
        
        # Create config
        config = SOTAV521Config(
            enable_correlation_prediction=True,
            enable_impact_learning=True,
            enable_adaptive_weights=True,
            enable_unified_events=True,
            enable_drift_detection=True,
            graceful_degradation=True
        )
        
        # Initialize
        integrator = await init_sota_v521(config)
        
        # Check health
        print("\n[Health Status]")
        health = integrator.get_health()
        print(json.dumps(health, indent=2, default=str))
        
        # Simulate some updates
        print("\n[Simulating Updates]")
        
        # Update prices
        for i in range(50):
            integrator.update_price("BTC", 100000 + i * 10)
            integrator.update_price("ETH", 3500 + i * 0.5)
            integrator.update_price("SOL", 200 + i * 0.1)
        print("Updated prices for 50 ticks")
        
        # Record trades
        integrator.record_trade("trend_following", 0.02, True)
        integrator.record_trade("momentum", -0.01, False)
        integrator.record_trade("mean_reversion", 0.015, True)
        print("Recorded 3 trades")
        
        # Update features
        integrator.update_features({
            "price_change_1m": 0.001,
            "price_change_5m": 0.003,
            "price_change_15m": 0.008,
            "volume_ratio": 1.2,
            "spread_bps": 5.0,
            "volatility_1h": 0.02,
            "orderbook_imbalance": 0.15,
            "trade_intensity": 1.5
        })
        print("Updated drift features")
        
        # Get assessments
        print("\n[Correlation Assessment]")
        corr = integrator.get_correlation_assessment()
        if corr:
            print(json.dumps(corr, indent=2, default=str))
        
        print("\n[Strategy Weights]")
        weights = integrator.get_strategy_weights()
        print(json.dumps(weights, indent=2, default=str))
        
        print("\n[RL Drift Check]")
        drift = integrator.check_rl_drift({
            "price_change_1m": 0.005,
            "price_change_5m": 0.01,
            "price_change_15m": 0.02,
            "volume_ratio": 1.5,
            "spread_bps": 8.0,
            "volatility_1h": 0.03,
            "orderbook_imbalance": 0.25,
            "trade_intensity": 2.0
        })
        print(json.dumps(drift, indent=2, default=str))
        
        # Position check
        print("\n[Position Check]")
        can_add, reason = integrator.can_add_position("SOL", ["BTC", "ETH"])
        print(f"Can add SOL: {can_add} - {reason}")
        
        # Position adjustment
        print("\n[Position Adjustment]")
        adjusted_size, adj_reason = integrator.get_position_adjustment("BTC", 1.0)
        print(f"Adjusted size: {adjusted_size:.3f} - {adj_reason}")
        
        # Dashboard data
        print("\n[Dashboard Data Summary]")
        dashboard = integrator.get_dashboard_data()
        print(f"Components: {list(dashboard.get('health', {}).get('components', {}).keys())}")
        
        # Stop
        await integrator.stop()
        print("\n[Integration Stopped]")
        print("=" * 60)
    
    asyncio.run(main())
