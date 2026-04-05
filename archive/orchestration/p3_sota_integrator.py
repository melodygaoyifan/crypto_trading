"""
================================================================================
HMATS v6.1.0 - P3 SOTA Integrator
================================================================================
Purpose: Simplified wrapper for SOTA v5.2.1 integration layer

This provides a clean interface to access SOTA features:
- Correlation control (real-time position adjustment)
- Market impact model (slippage prediction)
- Alpha fusion (onchain + sentiment)
- Adaptive weights (strategy weighting)

All modules are optional and fail gracefully.

Usage:
    integrator = get_p3_sota_integrator()
    
    # In tick processing:
    integrator.update_prices(prices_dict)
    
    # Before execution:
    adjustment = integrator.get_correlation_adjustment(positions)
    impact = integrator.predict_impact(asset, side, size, orderbook)
    
    # For alpha signal:
    signal = integrator.get_alpha_signal(asset)

================================================================================
"""

import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


# =============================================================================
# IMPORTS (with fallbacks)
# =============================================================================

CORRELATION_CONTROLLER_AVAILABLE = False
try:
    from risk.correlation_realtime_controller import (
        CorrelationRealtimeController,
        CorrelationConfig,
    )
    CORRELATION_CONTROLLER_AVAILABLE = True
except ImportError:
    pass

IMPACT_MODEL_AVAILABLE = False
try:
    from execution.market_impact_model import (
        ProductionMarketImpactModel,
        ImpactConfig,
    )
    IMPACT_MODEL_AVAILABLE = True
except ImportError:
    pass

ALPHA_FUSION_AVAILABLE = False
try:
    from agents.onchain_sentiment_fusion import (
        OnchainSentimentFusion,
        FusionConfig,
    )
    ALPHA_FUSION_AVAILABLE = True
except ImportError:
    pass

WEIGHT_MANAGER_AVAILABLE = False
try:
    from signals.adaptive_strategy_weight_manager import (
        AdaptiveStrategyWeightManager,
        WeightManagerConfig,
    )
    WEIGHT_MANAGER_AVAILABLE = True
except ImportError:
    pass

# Full SOTA integration layer
FULL_SOTA_AVAILABLE = False
try:
    from orchestration.sota_v521_complete_integration import (
        SOTAV521CompleteIntegrator,
        SOTAV521CompleteConfig,
    )
    FULL_SOTA_AVAILABLE = True
except ImportError:
    pass


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class P3SOTAConfig:
    """P3 SOTA integrator configuration."""
    
    # Enable flags
    enable_correlation: bool = True
    enable_impact_model: bool = True
    enable_alpha_fusion: bool = True
    enable_adaptive_weights: bool = True
    
    # Correlation settings
    correlation_warning_threshold: float = 0.90
    correlation_critical_threshold: float = 0.95
    correlation_cooldown_minutes: int = 15
    
    # Impact model settings
    max_participation_rate: float = 0.10
    impact_learning_rate: float = 0.01
    
    # Alpha fusion settings
    onchain_weight: float = 0.4
    sentiment_weight: float = 0.3
    funding_weight: float = 0.3
    
    # Weight manager settings
    min_weight: float = 0.1
    max_weight: float = 3.0


# =============================================================================
# RESULT TYPES
# =============================================================================

@dataclass
class CorrelationAdjustment:
    """Result from correlation controller."""
    should_reduce: bool = False
    scale_factor: float = 1.0
    reason: str = ""
    current_correlation: float = 0.0
    threshold_breached: bool = False


@dataclass
class ImpactPrediction:
    """Result from impact model."""
    expected_slippage_bps: float = 0.0
    recommended_size: float = 0.0
    participation_rate: float = 0.0
    urgency_score: float = 0.5
    vwap_recommendation: bool = False


@dataclass
class AlphaSignal:
    """Result from alpha fusion."""
    direction: float = 0.0
    confidence: float = 0.0
    onchain_contribution: float = 0.0
    sentiment_contribution: float = 0.0
    funding_contribution: float = 0.0


# =============================================================================
# P3 SOTA INTEGRATOR
# =============================================================================

class P3SOTAIntegrator:
    """
    Simplified SOTA integration layer.
    
    Provides clean access to advanced SOTA features
    with graceful degradation when modules unavailable.
    """
    
    def __init__(self, config: P3SOTAConfig = None):
        self.config = config or P3SOTAConfig()
        
        # Full integrator (if available)
        self._full_integrator: Optional[SOTAV521CompleteIntegrator] = None
        
        # Individual components (fallback)
        self._correlation_controller = None
        self._impact_model = None
        self._alpha_fusion = None
        self._weight_manager = None
        
        # State
        self._initialized = False
        self._lock = threading.Lock()
        self._last_prices: Dict[str, float] = {}
        
        self._init_components()
    
    def _init_components(self):
        """Initialize available components."""
        
        # Try full integrator first
        if FULL_SOTA_AVAILABLE:
            try:
                full_config = SOTAV521CompleteConfig(
                    enable_realtime_correlation=self.config.enable_correlation,
                    enable_production_impact=self.config.enable_impact_model,
                    enable_alpha_fusion=self.config.enable_alpha_fusion,
                    enable_adaptive_weights=self.config.enable_adaptive_weights,
                )
                self._full_integrator = SOTAV521CompleteIntegrator(full_config)
                self._initialized = True
                logger.info("[P3 SOTA] Full integrator loaded")
                return
            except Exception as e:
                logger.warning(f"[P3 SOTA] Full integrator failed: {e}")
        
        # Fall back to individual components
        component_count = 0
        
        if CORRELATION_CONTROLLER_AVAILABLE and self.config.enable_correlation:
            try:
                self._correlation_controller = CorrelationRealtimeController()
                component_count += 1
                logger.info("  [P3] CorrelationController: ACTIVE")
            except Exception as e:
                logger.warning(f"  [P3] CorrelationController: FAILED ({e})")
        
        if IMPACT_MODEL_AVAILABLE and self.config.enable_impact_model:
            try:
                self._impact_model = ProductionMarketImpactModel()
                component_count += 1
                logger.info("  [P3] ImpactModel: ACTIVE")
            except Exception as e:
                logger.warning(f"  [P3] ImpactModel: FAILED ({e})")
        
        if ALPHA_FUSION_AVAILABLE and self.config.enable_alpha_fusion:
            try:
                self._alpha_fusion = OnchainSentimentFusion()
                component_count += 1
                logger.info("  [P3] AlphaFusion: ACTIVE")
            except Exception as e:
                logger.warning(f"  [P3] AlphaFusion: FAILED ({e})")
        
        if WEIGHT_MANAGER_AVAILABLE and self.config.enable_adaptive_weights:
            try:
                self._weight_manager = AdaptiveStrategyWeightManager()
                component_count += 1
                logger.info("  [P3] WeightManager: ACTIVE")
            except Exception as e:
                logger.warning(f"  [P3] WeightManager: FAILED ({e})")
        
        self._initialized = component_count > 0
        logger.info(f"[P3 SOTA] Initialized with {component_count} components")
    
    # =========================================================================
    # UPDATE METHODS
    # =========================================================================
    
    def update_prices(self, prices: Dict[str, float]):
        """Update price data for correlation tracking."""
        with self._lock:
            self._last_prices = prices.copy()
            
            if self._full_integrator:
                try:
                    self._full_integrator.update_prices(prices)
                except Exception as e:
                    logger.warning(f"[P3] Price update failed: {e}")
            elif self._correlation_controller:
                try:
                    for asset, price in prices.items():
                        self._correlation_controller.update_price(asset, price)
                except Exception as e:
                    logger.warning(f"[P3] Correlation update failed: {e}")
    
    def update_orderbook(self, asset: str, orderbook: Dict):
        """Update orderbook for impact model."""
        if self._full_integrator:
            # Full integrator uses orderbook in predict_market_impact calls
            # Store for later use
            with self._lock:
                if not hasattr(self, '_orderbooks'):
                    self._orderbooks = {}
                self._orderbooks[asset] = orderbook
        elif self._impact_model:
            try:
                self._impact_model.update_orderbook(asset, orderbook)
            except Exception as e:
                logger.debug(f"[P3] Orderbook update failed: {e}")
    
    # =========================================================================
    # QUERY METHODS
    # =========================================================================
    
    def get_correlation_adjustment(
        self,
        positions: Dict[str, float],
    ) -> CorrelationAdjustment:
        """
        Get correlation-based position adjustment.
        
        Returns recommendation to reduce position if
        cross-asset correlation is dangerously high.
        """
        result = CorrelationAdjustment()
        
        if not self._initialized:
            return result
        
        try:
            if self._full_integrator:
                adjustment = self._full_integrator.get_correlation_position_adjustment(
                    positions
                )
                if adjustment:
                    result.should_reduce = adjustment.get("should_reduce", False)
                    result.scale_factor = adjustment.get("scale_factor", 1.0)
                    result.reason = adjustment.get("reason", "")
                    result.current_correlation = adjustment.get("correlation", 0.0)
                    
            elif self._correlation_controller:
                state = self._correlation_controller.get_state()
                corr = state.get("btc_eth_correlation", 0.0)
                
                result.current_correlation = corr
                
                if corr > self.config.correlation_critical_threshold:
                    result.should_reduce = True
                    result.scale_factor = 0.5
                    result.reason = f"CRITICAL: correlation {corr:.2f}"
                    result.threshold_breached = True
                    
                elif corr > self.config.correlation_warning_threshold:
                    result.should_reduce = True
                    result.scale_factor = 0.75
                    result.reason = f"WARNING: correlation {corr:.2f}"
                    
        except Exception as e:
            logger.warning(f"[P3] Correlation adjustment error: {e}")
        
        return result
    
    def predict_impact(
        self,
        asset: str,
        side: str,
        size: float,
        orderbook: Dict = None,
    ) -> ImpactPrediction:
        """
        Predict market impact for an order.
        
        Returns expected slippage and recommendations.
        """
        result = ImpactPrediction()
        
        if not self._initialized:
            return result
        
        try:
            if self._full_integrator:
                impact = self._full_integrator.predict_market_impact(
                    asset, side, size, orderbook
                )
                if impact:
                    result.expected_slippage_bps = impact.get("slippage_bps", 0.0)
                    result.recommended_size = impact.get("recommended_size", size)
                    result.participation_rate = impact.get("participation", 0.0)
                    result.urgency_score = impact.get("urgency", 0.5)
                    result.vwap_recommendation = impact.get("use_vwap", False)
                    
            elif self._impact_model:
                impact = self._impact_model.predict(
                    asset=asset,
                    side=side,
                    size=size,
                    orderbook=orderbook,
                )
                if impact:
                    result.expected_slippage_bps = impact.slippage_bps
                    result.recommended_size = min(size, impact.max_size)
                    
        except Exception as e:
            logger.warning(f"[P3] Impact prediction error: {e}")
        
        return result
    
    def get_alpha_signal(self, asset: str) -> AlphaSignal:
        """
        Get combined onchain + sentiment alpha signal.
        
        Returns fused signal with component breakdown.
        """
        result = AlphaSignal()
        
        if not self._initialized:
            return result
        
        try:
            if self._full_integrator:
                signal = self._full_integrator.generate_alpha_signal(asset)
                if signal:
                    result.direction = signal.get("direction", 0.0)
                    result.confidence = signal.get("confidence", 0.0)
                    result.onchain_contribution = signal.get("onchain", 0.0)
                    result.sentiment_contribution = signal.get("sentiment", 0.0)
                    result.funding_contribution = signal.get("funding", 0.0)
                    
            elif self._alpha_fusion:
                signal = self._alpha_fusion.generate_signal(asset)
                if signal:
                    result.direction = signal.direction
                    result.confidence = signal.confidence
                    
        except Exception as e:
            logger.warning(f"[P3] Alpha signal error: {e}")
        
        return result
    
    def get_strategy_weights(self) -> Dict[str, float]:
        """Get adaptive strategy weights."""
        if not self._initialized:
            return {}
        
        try:
            if self._full_integrator:
                return self._full_integrator.get_strategy_weights() or {}
            elif self._weight_manager:
                return self._weight_manager.get_weights() or {}
        except Exception as e:
            logger.warning(f"[P3] Weights error: {e}")
        
        return {}
    
    # =========================================================================
    # STATUS
    # =========================================================================
    
    def get_status(self) -> Dict[str, Any]:
        """Get integrator status."""
        return {
            "initialized": self._initialized,
            "full_integrator": self._full_integrator is not None,
            "correlation_controller": self._correlation_controller is not None,
            "impact_model": self._impact_model is not None,
            "alpha_fusion": self._alpha_fusion is not None,
            "weight_manager": self._weight_manager is not None,
            "last_prices": self._last_prices,
        }
    
    def is_available(self) -> bool:
        """Check if any P3 functionality is available."""
        return self._initialized


# =============================================================================
# SINGLETON
# =============================================================================

_p3_integrator: Optional[P3SOTAIntegrator] = None
_p3_lock = threading.Lock()


def get_p3_sota_integrator(config: P3SOTAConfig = None) -> P3SOTAIntegrator:
    """Get singleton P3 SOTA integrator."""
    global _p3_integrator
    if _p3_integrator is None:
        with _p3_lock:
            if _p3_integrator is None:
                _p3_integrator = P3SOTAIntegrator(config)
    return _p3_integrator


def reset_p3_sota_integrator():
    """Reset singleton (for testing)."""
    global _p3_integrator
    with _p3_lock:
        _p3_integrator = None


if __name__ == "__main__":
    # Test
    integrator = get_p3_sota_integrator()
    print("Status:", integrator.get_status())
    
    # Simulate updates
    integrator.update_prices({"BTC": 50000, "ETH": 3000, "SOL": 100})
    
    # Get adjustments
    corr_adj = integrator.get_correlation_adjustment({"BTC": 0.5, "ETH": 0.3})
    print("Correlation adjustment:", corr_adj)
    
    impact = integrator.predict_impact("SOL", "buy", 100000)
    print("Impact prediction:", impact)
    
    alpha = integrator.get_alpha_signal("SOL")
    print("Alpha signal:", alpha)
    
    weights = integrator.get_strategy_weights()
    print("Strategy weights:", weights)
