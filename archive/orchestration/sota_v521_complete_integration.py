"""
HMATS v5.2.1-SOTA-COMPLETE Integration Layer

Complete integration of all production-grade v5.2.1 modules:
- Real-time correlation controller with prediction
- Production market impact model with online learning
- On-chain + sentiment alpha fusion engine
- Adaptive strategy weight manager

This layer provides unified access to all SOTA features with:
- Single initialization point
- Cross-component event wiring
- Health monitoring
- Graceful degradation
- Dashboard data aggregation

Usage:
    config = SOTAV521CompleteConfig(enable_all=True)
    integrator = SOTAV521CompleteIntegrator(config)
    await integrator.initialize()
    
    # Use individual components
    adjustment = integrator.get_correlation_position_adjustment(positions)
    impact = integrator.predict_market_impact("BTC", "buy", 100000, orderbook)
    signal = integrator.generate_alpha_signal("SOL")
    weights = integrator.get_strategy_weights()
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from datetime import datetime
import logging
import asyncio

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SOTAV521CompleteConfig:
    """Configuration for complete v5.2.1 integration"""
    
    # Module enables
    enable_realtime_correlation: bool = True
    enable_production_impact: bool = True
    enable_alpha_fusion: bool = True
    enable_adaptive_weights: bool = True
    enable_unified_events: bool = True
    enable_drift_detection: bool = True
    enable_explainer_dashboard: bool = True
    
    # Correlation settings
    correlation_short_window: int = 20
    correlation_medium_window: int = 60
    correlation_long_window: int = 200
    correlation_jump_threshold: float = 2.5
    correlation_preemptive_scale: float = 0.85
    correlation_cooldown_minutes: int = 15
    
    # Market impact settings
    impact_learning_rate: float = 0.01
    impact_max_participation_critical: float = 0.30
    impact_max_participation_high: float = 0.15
    impact_max_participation_medium: float = 0.08
    impact_max_participation_low: float = 0.05
    
    # Alpha fusion settings
    alpha_onchain_weight: float = 0.4
    alpha_sentiment_weight: float = 0.3
    alpha_funding_weight: float = 0.3
    alpha_extreme_fear_threshold: int = 20
    alpha_extreme_greed_threshold: int = 80
    alpha_cooldown_minutes: int = 30
    
    # Adaptive weights settings
    weights_sharpe_component: float = 0.4
    weights_calmar_component: float = 0.25
    weights_short_winrate_component: float = 0.2
    weights_medium_winrate_component: float = 0.15
    weights_ema_alpha: float = 0.3
    weights_min: float = 0.1
    weights_max: float = 3.0
    
    # Drift detection settings
    drift_kl_threshold: float = 0.5
    drift_psi_threshold: float = 0.25
    drift_ks_threshold: float = 0.1
    drift_recovery_cooldown_minutes: int = 30
    drift_recovery_samples: int = 100
    
    # Convenience
    @classmethod
    def enable_all(cls) -> 'SOTAV521CompleteConfig':
        """Create config with all features enabled"""
        return cls(
            enable_realtime_correlation=True,
            enable_production_impact=True,
            enable_alpha_fusion=True,
            enable_adaptive_weights=True,
            enable_unified_events=True,
            enable_drift_detection=True,
            enable_explainer_dashboard=True
        )


class IntegrationState(Enum):
    """Integration health states"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    PARTIAL = "partial"
    FAILED = "failed"


@dataclass
class ComponentHealth:
    """Health status of individual component"""
    name: str
    enabled: bool
    initialized: bool
    healthy: bool
    last_error: Optional[str] = None
    last_update: Optional[datetime] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Complete Integration Layer
# =============================================================================

class SOTAV521CompleteIntegrator:
    """
    Complete integration layer for all v5.2.1 SOTA modules.
    
    Provides unified access to:
    - Real-time correlation control with prediction
    - Production market impact model
    - On-chain + sentiment alpha fusion
    - Adaptive strategy weight management
    - RL drift detection and fallback
    - Unified event architecture
    """
    
    def __init__(self, config: Optional[SOTAV521CompleteConfig] = None):
        self.config = config or SOTAV521CompleteConfig()
        
        # Component instances (lazy loaded)
        self._correlation_controller = None
        self._market_impact_model = None
        self._alpha_engine = None
        self._weight_manager = None
        self._event_bus = None
        self._drift_detector = None
        self._dashboard = None
        
        # Health tracking
        self._component_health: Dict[str, ComponentHealth] = {}
        self._initialized = False
        self._state = IntegrationState.PARTIAL
        
        # EventBus subscriptions
        self._subscriptions = []
        
    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------
    
    async def initialize(self) -> bool:
        """Initialize all enabled components"""
        logger.info("Initializing SOTA v5.2.1 Complete Integration")
        
        success_count = 0
        total_count = 0
        
        # Initialize unified events first (other components depend on it)
        if self.config.enable_unified_events:
            total_count += 1
            if await self._init_event_bus():
                success_count += 1
        
        # Initialize correlation controller
        if self.config.enable_realtime_correlation:
            total_count += 1
            if await self._init_correlation_controller():
                success_count += 1
        
        # Initialize market impact model
        if self.config.enable_production_impact:
            total_count += 1
            if await self._init_market_impact():
                success_count += 1
        
        # Initialize alpha fusion engine
        if self.config.enable_alpha_fusion:
            total_count += 1
            if await self._init_alpha_engine():
                success_count += 1
        
        # Initialize adaptive weight manager
        if self.config.enable_adaptive_weights:
            total_count += 1
            if await self._init_weight_manager():
                success_count += 1
        
        # Initialize drift detector
        if self.config.enable_drift_detection:
            total_count += 1
            if await self._init_drift_detector():
                success_count += 1
        
        # Initialize dashboard
        if self.config.enable_explainer_dashboard:
            total_count += 1
            if await self._init_dashboard():
                success_count += 1
        
        # Wire cross-component events
        await self._wire_events()
        
        # Determine state
        if success_count == total_count:
            self._state = IntegrationState.HEALTHY
        elif success_count > 0:
            self._state = IntegrationState.DEGRADED
        else:
            self._state = IntegrationState.FAILED
        
        self._initialized = True
        logger.info(f"Integration initialized: {success_count}/{total_count} components, state={self._state.value}")
        
        return self._state in (IntegrationState.HEALTHY, IntegrationState.DEGRADED)
    
    def initialize_sync(self) -> bool:
        """Synchronous initialization wrapper for testing"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If already in async context, use nest_asyncio pattern
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self.initialize())
                    return future.result(timeout=30)
            else:
                return loop.run_until_complete(self.initialize())
        except RuntimeError:
            # No event loop exists
            return asyncio.run(self.initialize())
    
    def is_initialized(self) -> bool:
        """Check if integrator is initialized"""
        return self._initialized
    
    def get_health_status(self) -> Dict:
        """Get health status of all components (alias for get_health)"""
        return self.get_health()
    
    async def _init_event_bus(self) -> bool:
        """Initialize unified event bus"""
        try:
            from infra.unified_event_v521 import UnifiedEventBus, RunMode
            self._event_bus = UnifiedEventBus(run_mode=RunMode.LIVE)
            self._component_health['event_bus'] = ComponentHealth(
                name='event_bus',
                enabled=True,
                initialized=True,
                healthy=True,
                last_update=datetime.now()
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to init event bus: {e}")
            self._component_health['event_bus'] = ComponentHealth(
                name='event_bus',
                enabled=True,
                initialized=False,
                healthy=False,
                last_error=str(e)
            )
            # Create fallback
            self._event_bus = _FallbackEventBus()
            return False
    
    async def _init_correlation_controller(self) -> bool:
        """Initialize real-time correlation controller"""
        try:
            from risk.correlation_realtime_controller import CorrelationRealtimeController
            
            self._correlation_controller = CorrelationRealtimeController(
                short_window=self.config.correlation_short_window,
                medium_window=self.config.correlation_medium_window,
                long_window=self.config.correlation_long_window,
                jump_threshold=self.config.correlation_jump_threshold,
                cooldown_minutes=self.config.correlation_cooldown_minutes,
                event_bus=self._event_bus
            )
            
            self._component_health['correlation'] = ComponentHealth(
                name='correlation',
                enabled=True,
                initialized=True,
                healthy=True,
                last_update=datetime.now()
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to init correlation controller: {e}")
            self._component_health['correlation'] = ComponentHealth(
                name='correlation',
                enabled=True,
                initialized=False,
                healthy=False,
                last_error=str(e)
            )
            return False
    
    async def _init_market_impact(self) -> bool:
        """Initialize production market impact model"""
        try:
            from execution.production_market_impact import AdaptiveMarketImpactModel
            
            self._market_impact_model = AdaptiveMarketImpactModel(
                learning_rate=self.config.impact_learning_rate
            )
            
            self._component_health['market_impact'] = ComponentHealth(
                name='market_impact',
                enabled=True,
                initialized=True,
                healthy=True,
                last_update=datetime.now()
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to init market impact model: {e}")
            self._component_health['market_impact'] = ComponentHealth(
                name='market_impact',
                enabled=True,
                initialized=False,
                healthy=False,
                last_error=str(e)
            )
            return False
    
    async def _init_alpha_engine(self) -> bool:
        """Initialize on-chain + sentiment alpha fusion engine"""
        try:
            from agents.onchain_sentiment_fusion import OnChainSentimentAlphaEngine
            
            self._alpha_engine = OnChainSentimentAlphaEngine(
                onchain_weight=self.config.alpha_onchain_weight,
                sentiment_weight=self.config.alpha_sentiment_weight,
                funding_weight=self.config.alpha_funding_weight,
                extreme_fear_threshold=self.config.alpha_extreme_fear_threshold,
                extreme_greed_threshold=self.config.alpha_extreme_greed_threshold,
                cooldown_minutes=self.config.alpha_cooldown_minutes
            )
            
            self._component_health['alpha_fusion'] = ComponentHealth(
                name='alpha_fusion',
                enabled=True,
                initialized=True,
                healthy=True,
                last_update=datetime.now()
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to init alpha engine: {e}")
            self._component_health['alpha_fusion'] = ComponentHealth(
                name='alpha_fusion',
                enabled=True,
                initialized=False,
                healthy=False,
                last_error=str(e)
            )
            return False
    
    async def _init_weight_manager(self) -> bool:
        """Initialize adaptive strategy weight manager"""
        try:
            from signals.adaptive_strategy_weight_manager import AdaptiveStrategyWeightManager
            
            self._weight_manager = AdaptiveStrategyWeightManager(
                sharpe_weight=self.config.weights_sharpe_component,
                calmar_weight=self.config.weights_calmar_component,
                short_winrate_weight=self.config.weights_short_winrate_component,
                medium_winrate_weight=self.config.weights_medium_winrate_component,
                ema_alpha=self.config.weights_ema_alpha,
                min_weight=self.config.weights_min,
                max_weight=self.config.weights_max
            )
            
            self._component_health['weight_manager'] = ComponentHealth(
                name='weight_manager',
                enabled=True,
                initialized=True,
                healthy=True,
                last_update=datetime.now()
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to init weight manager: {e}")
            self._component_health['weight_manager'] = ComponentHealth(
                name='weight_manager',
                enabled=True,
                initialized=False,
                healthy=False,
                last_error=str(e)
            )
            return False
    
    async def _init_drift_detector(self) -> bool:
        """Initialize RL drift detector"""
        try:
            from execution.rl_drift_fallback_v521 import EnhancedDriftDetector
            
            self._drift_detector = EnhancedDriftDetector(
                kl_threshold=self.config.drift_kl_threshold,
                psi_threshold=self.config.drift_psi_threshold,
                ks_threshold=self.config.drift_ks_threshold,
                recovery_cooldown_minutes=self.config.drift_recovery_cooldown_minutes,
                recovery_samples=self.config.drift_recovery_samples
            )
            
            self._component_health['drift_detector'] = ComponentHealth(
                name='drift_detector',
                enabled=True,
                initialized=True,
                healthy=True,
                last_update=datetime.now()
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to init drift detector: {e}")
            self._component_health['drift_detector'] = ComponentHealth(
                name='drift_detector',
                enabled=True,
                initialized=False,
                healthy=False,
                last_error=str(e)
            )
            return False
    
    async def _init_dashboard(self) -> bool:
        """Initialize explainer dashboard"""
        try:
            from dashboard.integrated_explainer_dashboard import IntegratedExplainerDashboard
            
            self._dashboard = IntegratedExplainerDashboard()
            
            self._component_health['dashboard'] = ComponentHealth(
                name='dashboard',
                enabled=True,
                initialized=True,
                healthy=True,
                last_update=datetime.now()
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to init dashboard: {e}")
            self._component_health['dashboard'] = ComponentHealth(
                name='dashboard',
                enabled=True,
                initialized=False,
                healthy=False,
                last_error=str(e)
            )
            return False
    
    async def _wire_events(self):
        """Wire cross-component event handlers"""
        if not self._event_bus:
            return
        
        try:
            # Correlation events -> Weight adjustment
            if self._correlation_controller and self._weight_manager:
                self._event_bus.subscribe(
                    'correlation.correlation_jump',
                    self._on_correlation_jump
                )
            
            # Alpha signals -> Dashboard
            if self._alpha_engine and self._dashboard:
                self._event_bus.subscribe(
                    'alpha.signal_generated',
                    self._on_alpha_signal
                )
            
            # Drift events -> Dashboard alerts
            if self._drift_detector and self._dashboard:
                self._event_bus.subscribe(
                    'drift.detected',
                    self._on_drift_detected
                )
            
            logger.info("Cross-component events wired")
        except Exception as e:
            logger.warning(f"Failed to wire some events: {e}")
    
    def _on_correlation_jump(self, event: Dict):
        """Handle correlation jump events"""
        try:
            if self._dashboard:
                self._dashboard.add_alert(
                    "correlation",
                    f"Correlation jump detected: {event.get('severity', 'unknown')}",
                    event
                )
        except Exception as e:
            logger.error(f"Error handling correlation jump: {e}")
    
    def _on_alpha_signal(self, event: Dict):
        """Handle alpha signal events"""
        try:
            if self._dashboard:
                self._dashboard.add_signal(
                    event.get('token', 'unknown'),
                    event.get('direction', 0),
                    event.get('confidence', 0),
                    event.get('components', {})
                )
        except Exception as e:
            logger.error(f"Error handling alpha signal: {e}")
    
    def _on_drift_detected(self, event: Dict):
        """Handle drift detection events"""
        try:
            if self._dashboard:
                self._dashboard.add_alert(
                    "drift",
                    f"RL drift detected: {event.get('severity', 'unknown')}",
                    event
                )
        except Exception as e:
            logger.error(f"Error handling drift event: {e}")
    
    # -------------------------------------------------------------------------
    # Correlation Control API
    # -------------------------------------------------------------------------
    
    def update_prices(self, prices: Dict[str, float]):
        """
        Update price data for correlation analysis.
        
        Args:
            prices: Dict of asset -> price (e.g., {"BTC": 100000, "ETH": 3500})
        """
        if self._correlation_controller:
            self._correlation_controller.update_prices_batch(prices)
            self._component_health['correlation'].last_update = datetime.now()
    
    def get_correlation_position_adjustment(
        self,
        current_positions: Dict[str, float]
    ) -> Optional[Dict]:
        """
        Get position adjustment recommendation based on correlation analysis.
        
        Args:
            current_positions: Dict of asset -> position size
            
        Returns:
            Dict with: state, action, scale_factor, asset_adjustments, is_preemptive
            None if correlation controller not available
        """
        if not self._correlation_controller:
            return None
        
        try:
            adjustment = self._correlation_controller.get_position_adjustment(
                current_positions
            )
            
            return {
                'state': adjustment.state.value if hasattr(adjustment, 'state') else 'unknown',
                'action': adjustment.action.value if hasattr(adjustment, 'action') else 'hold',
                'scale_factor': adjustment.scale_factor if hasattr(adjustment, 'scale_factor') else 1.0,
                'asset_adjustments': adjustment.asset_adjustments if hasattr(adjustment, 'asset_adjustments') else {},
                'is_preemptive': adjustment.is_preemptive if hasattr(adjustment, 'is_preemptive') else False,
                'reason': adjustment.reason if hasattr(adjustment, 'reason') else ''
            }
        except Exception as e:
            logger.error(f"Error getting correlation adjustment: {e}")
            self._component_health['correlation'].healthy = False
            self._component_health['correlation'].last_error = str(e)
            return None
    
    def get_correlation_state(self) -> Optional[str]:
        """Get current correlation state"""
        if not self._correlation_controller:
            return None
        
        try:
            return self._correlation_controller.get_current_state().value
        except Exception as e:
            logger.error(f"Error getting correlation state: {e}")
            return None
    
    def get_correlation_prediction(self, horizon_hours: int = 1) -> Optional[Dict]:
        """Get correlation prediction for given horizon"""
        if not self._correlation_controller:
            return None
        
        try:
            pred = self._correlation_controller.get_prediction(horizon_hours)
            return {
                'predicted_correlation': pred.predicted_value if pred else None,
                'spike_warning': pred.spike_warning if pred else False,
                'collapse_warning': pred.collapse_warning if pred else False,
                'confidence': pred.confidence if pred else 0.0
            }
        except Exception as e:
            logger.error(f"Error getting correlation prediction: {e}")
            return None
    
    # -------------------------------------------------------------------------
    # Market Impact API
    # -------------------------------------------------------------------------
    
    def update_market_data(
        self,
        symbol: str,
        daily_volume: float,
        volatility: float,
        bid_ask_spread: float = 0.001
    ):
        """
        Update market data for impact model.
        
        Args:
            symbol: Trading pair (e.g., "BTC-USD")
            daily_volume: 24h trading volume in USD
            volatility: Annualized volatility
            bid_ask_spread: Current bid-ask spread as fraction
        """
        if self._market_impact_model:
            self._market_impact_model.update_market_data(
                symbol, daily_volume, volatility, bid_ask_spread
            )
            self._component_health['market_impact'].last_update = datetime.now()
    
    def predict_market_impact(
        self,
        symbol: str,
        side: str,
        size_usd: float,
        orderbook: Optional[Dict] = None
    ) -> Optional[Dict]:
        """
        Predict market impact for a trade.
        
        Args:
            symbol: Trading pair
            side: "buy" or "sell"
            size_usd: Order size in USD
            orderbook: Optional L2 orderbook data
            
        Returns:
            Dict with: temp_impact_bps, perm_impact_bps, total_impact_bps, cost_usd
        """
        if not self._market_impact_model:
            return None
        
        try:
            impact = self._market_impact_model.predict_impact(
                symbol, side, size_usd, orderbook
            )
            
            return {
                'temp_impact_bps': impact.temp_impact_bps if hasattr(impact, 'temp_impact_bps') else 0,
                'perm_impact_bps': impact.perm_impact_bps if hasattr(impact, 'perm_impact_bps') else 0,
                'total_impact_bps': impact.total_impact_bps if hasattr(impact, 'total_impact_bps') else 0,
                'cost_usd': impact.cost_usd if hasattr(impact, 'cost_usd') else 0,
                'confidence_low': impact.confidence_low if hasattr(impact, 'confidence_low') else 0,
                'confidence_high': impact.confidence_high if hasattr(impact, 'confidence_high') else 0
            }
        except Exception as e:
            logger.error(f"Error predicting impact: {e}")
            self._component_health['market_impact'].healthy = False
            return None
    
    def get_optimal_slicing(
        self,
        symbol: str,
        side: str,
        total_size_usd: float,
        urgency: str = "medium",
        orderbook: Optional[Dict] = None
    ) -> Optional[Dict]:
        """
        Compute optimal order slicing plan.
        
        Args:
            symbol: Trading pair
            side: "buy" or "sell"
            total_size_usd: Total order size in USD
            urgency: "critical", "high", "medium", "low"
            orderbook: Optional L2 orderbook data
            
        Returns:
            Dict with: num_slices, slice_sizes, order_type, limit_price, duration
        """
        if not self._market_impact_model:
            return None
        
        try:
            # Map urgency string to enum
            from execution.production_market_impact import UrgencyLevel
            urgency_map = {
                'critical': UrgencyLevel.CRITICAL,
                'high': UrgencyLevel.HIGH,
                'medium': UrgencyLevel.MEDIUM,
                'low': UrgencyLevel.LOW
            }
            urgency_level = urgency_map.get(urgency.lower(), UrgencyLevel.MEDIUM)
            
            plan = self._market_impact_model.compute_optimal_slicing(
                symbol, side, total_size_usd, urgency_level, orderbook
            )
            
            return {
                'num_slices': plan.num_slices if hasattr(plan, 'num_slices') else 1,
                'slice_sizes': plan.slice_sizes if hasattr(plan, 'slice_sizes') else [total_size_usd],
                'order_type': plan.order_type.value if hasattr(plan, 'order_type') else 'market',
                'suggested_limit_price': plan.suggested_limit_price if hasattr(plan, 'suggested_limit_price') else None,
                'estimated_duration_seconds': plan.estimated_duration_seconds if hasattr(plan, 'estimated_duration_seconds') else 0,
                'predicted_total_impact_bps': plan.predicted_total_impact_bps if hasattr(plan, 'predicted_total_impact_bps') else 0
            }
        except Exception as e:
            logger.error(f"Error computing slicing: {e}")
            self._component_health['market_impact'].healthy = False
            return None
    
    def record_execution(
        self,
        symbol: str,
        side: str,
        size_btc: float,
        notional_usd: float,
        execution_price: float,
        expected_price: Optional[float] = None
    ):
        """
        Record execution for model calibration.
        
        Args:
            symbol: Trading pair
            side: "buy" or "sell"
            size_btc: Size in base currency
            notional_usd: Size in USD
            execution_price: Actual execution price
            expected_price: Expected price (mid or limit)
        """
        if self._market_impact_model:
            self._market_impact_model.record_execution_simple(
                symbol, side, size_btc, notional_usd,
                execution_price, expected_price
            )
    
    # -------------------------------------------------------------------------
    # Alpha Fusion API
    # -------------------------------------------------------------------------
    
    def record_onchain_activity(
        self,
        address: str,
        activity_type: str,
        amount: float,
        token: str,
        value_usd: float,
        is_whale: bool = False,
        is_smart_money: bool = False
    ):
        """
        Record on-chain wallet activity.
        
        Args:
            address: Wallet address
            activity_type: "buy", "sell", "transfer", "stake", etc.
            amount: Token amount
            token: Token symbol
            value_usd: USD value
            is_whale: Whether address is known whale
            is_smart_money: Whether address is tracked smart money
        """
        if self._alpha_engine:
            from agents.onchain_sentiment_fusion import WalletActivity
            
            activity = WalletActivity(
                address=address,
                activity_type=activity_type,
                amount=amount,
                token=token,
                value_usd=value_usd,
                is_whale=is_whale,
                is_smart_money=is_smart_money,
                timestamp=datetime.now()
            )
            
            self._alpha_engine.record_onchain_activity(activity)
            self._component_health['alpha_fusion'].last_update = datetime.now()
    
    def update_sentiment(
        self,
        fear_greed: Optional[int] = None,
        twitter: Optional[float] = None,
        reddit: Optional[float] = None,
        news_sentiment: Optional[float] = None,
        btc_funding: Optional[float] = None,
        eth_funding: Optional[float] = None,
        sol_funding: Optional[float] = None
    ):
        """
        Update sentiment data.
        
        Args:
            fear_greed: Fear & Greed index (0-100)
            twitter: Twitter sentiment (-1 to 1)
            reddit: Reddit sentiment (-1 to 1)
            news_sentiment: News sentiment (-1 to 1)
            btc_funding: BTC perpetual funding rate
            eth_funding: ETH perpetual funding rate
            sol_funding: SOL perpetual funding rate
        """
        if self._alpha_engine:
            from agents.onchain_sentiment_fusion import SentimentData
            
            sentiment = SentimentData(
                fear_greed_index=fear_greed,
                twitter_sentiment=twitter,
                reddit_sentiment=reddit,
                news_sentiment=news_sentiment,
                btc_funding_rate=btc_funding,
                eth_funding_rate=eth_funding,
                sol_funding_rate=sol_funding,
                timestamp=datetime.now()
            )
            
            self._alpha_engine.update_sentiment(sentiment)
            self._component_health['alpha_fusion'].last_update = datetime.now()
    
    def ingest_news(self, headline: str, source: str = "unknown"):
        """Ingest news headline for sentiment analysis"""
        if self._alpha_engine:
            self._alpha_engine.ingest_news_text(headline, source)
    
    def generate_alpha_signal(self, token: str) -> Optional[Dict]:
        """
        Generate fused alpha signal for token.
        
        Args:
            token: Token symbol (e.g., "SOL")
            
        Returns:
            Dict with: direction, confidence, trade_bias, urgency, components
        """
        if not self._alpha_engine:
            return None
        
        try:
            signal = self._alpha_engine.generate_signals(token)
            
            if signal:
                return {
                    'direction': signal.direction if hasattr(signal, 'direction') else 0,
                    'confidence': signal.confidence if hasattr(signal, 'confidence') else 0,
                    'trade_bias': signal.trade_bias.value if hasattr(signal, 'trade_bias') else 'neutral',
                    'urgency': signal.urgency.value if hasattr(signal, 'urgency') else 'low',
                    'components': {
                        'onchain': signal.onchain_component if hasattr(signal, 'onchain_component') else 0,
                        'sentiment': signal.sentiment_component if hasattr(signal, 'sentiment_component') else 0,
                        'funding': signal.funding_component if hasattr(signal, 'funding_component') else 0
                    },
                    'signals': [
                        {'type': s.signal_type.value, 'direction': s.direction, 'confidence': s.confidence}
                        for s in (signal.component_signals if hasattr(signal, 'component_signals') else [])
                    ]
                }
            return None
        except Exception as e:
            logger.error(f"Error generating alpha signal: {e}")
            self._component_health['alpha_fusion'].healthy = False
            return None
    
    def get_alpha_signal_for_fusion(self, token: str) -> Optional[Dict]:
        """Get alpha signal formatted for signal fusion"""
        if not self._alpha_engine:
            return None
        
        try:
            return self._alpha_engine.get_signal_for_fusion(token)
        except Exception as e:
            logger.error(f"Error getting fusion signal: {e}")
            return None
    
    # -------------------------------------------------------------------------
    # Adaptive Weights API
    # -------------------------------------------------------------------------
    
    def record_strategy_trade(
        self,
        strategy_name: str,
        asset: str,
        pnl_pct: float,
        holding_period_hours: float = 4.0
    ):
        """
        Record strategy trade for weight tracking.
        
        Args:
            strategy_name: Name of strategy
            asset: Asset traded
            pnl_pct: P&L percentage (e.g., 0.03 for 3%)
            holding_period_hours: Trade holding period
        """
        if self._weight_manager:
            self._weight_manager.record_trade_simple(
                strategy_name, asset, pnl_pct, holding_period_hours
            )
            self._component_health['weight_manager'].last_update = datetime.now()
    
    def get_strategy_weight(self, strategy_name: str) -> float:
        """Get weight for single strategy"""
        if self._weight_manager:
            return self._weight_manager.get_weight(strategy_name)
        return 1.0
    
    def get_strategy_weights(self) -> Dict[str, float]:
        """Get all strategy weights"""
        if self._weight_manager:
            return self._weight_manager.get_all_weights()
        return {}
    
    def get_normalized_strategy_weights(self) -> Dict[str, float]:
        """Get normalized strategy weights (sum to 1)"""
        if self._weight_manager:
            return self._weight_manager.get_normalized_weights()
        return {}
    
    def fuse_strategy_signals(self, strategy_signals: Dict[str, float]) -> Tuple[float, float]:
        """
        Fuse strategy signals using adaptive weights.
        
        Args:
            strategy_signals: Dict of strategy_name -> signal (-1 to 1)
            
        Returns:
            Tuple of (fused_direction, confidence)
        """
        if self._weight_manager:
            return self._weight_manager.fuse_signals(strategy_signals)
        
        # Fallback: equal weighting
        if not strategy_signals:
            return (0.0, 0.0)
        avg = sum(strategy_signals.values()) / len(strategy_signals)
        return (avg, 0.5)
    
    def get_strategy_metrics(self, strategy_name: str) -> Optional[Dict]:
        """Get detailed metrics for strategy"""
        if not self._weight_manager:
            return None
        
        try:
            metrics = self._weight_manager.get_metrics(strategy_name)
            if metrics:
                return {
                    'sharpe_ratio': metrics.sharpe_ratio if hasattr(metrics, 'sharpe_ratio') else None,
                    'calmar_ratio': metrics.calmar_ratio if hasattr(metrics, 'calmar_ratio') else None,
                    'short_winrate': metrics.short_winrate if hasattr(metrics, 'short_winrate') else None,
                    'medium_winrate': metrics.medium_winrate if hasattr(metrics, 'medium_winrate') else None,
                    'total_trades': metrics.total_trades if hasattr(metrics, 'total_trades') else 0,
                    'max_drawdown': metrics.max_drawdown if hasattr(metrics, 'max_drawdown') else 0
                }
            return None
        except Exception as e:
            logger.error(f"Error getting strategy metrics: {e}")
            return None
    
    # -------------------------------------------------------------------------
    # Drift Detection API
    # -------------------------------------------------------------------------
    
    def check_rl_drift(self, features: Dict[str, float]) -> Optional[Dict]:
        """
        Check for RL model drift.
        
        Args:
            features: Current feature values
            
        Returns:
            Dict with: use_rl, fallback_strategy, severity, reason
        """
        if not self._drift_detector:
            return {'use_rl': True, 'fallback_strategy': None, 'severity': 'none', 'reason': 'detector not available'}
        
        try:
            result = self._drift_detector.check_drift(features)
            
            return {
                'use_rl': not result.drift_detected if hasattr(result, 'drift_detected') else True,
                'fallback_strategy': result.fallback_strategy if hasattr(result, 'fallback_strategy') else None,
                'severity': result.severity.value if hasattr(result, 'severity') else 'none',
                'reason': result.reason if hasattr(result, 'reason') else '',
                'metrics': {
                    'kl_divergence': result.kl_divergence if hasattr(result, 'kl_divergence') else 0,
                    'psi': result.psi if hasattr(result, 'psi') else 0,
                    'ks_statistic': result.ks_statistic if hasattr(result, 'ks_statistic') else 0
                }
            }
        except Exception as e:
            logger.error(f"Error checking drift: {e}")
            return {'use_rl': True, 'fallback_strategy': None, 'severity': 'error', 'reason': str(e)}
    
    def update_rl_baseline(self, features: Dict[str, float]):
        """Update RL baseline distribution"""
        if self._drift_detector:
            self._drift_detector.update_baseline(features)
    
    # -------------------------------------------------------------------------
    # Health & Status API
    # -------------------------------------------------------------------------
    
    def get_health(self) -> Dict:
        """Get integration health status"""
        return {
            'state': self._state.value,
            'initialized': self._initialized,
            'components': {
                name: {
                    'enabled': health.enabled,
                    'initialized': health.initialized,
                    'healthy': health.healthy,
                    'last_error': health.last_error,
                    'last_update': health.last_update.isoformat() if health.last_update else None
                }
                for name, health in self._component_health.items()
            }
        }
    
    def get_dashboard_data(self) -> Dict:
        """Get aggregated data for dashboard"""
        data = {
            'health': self.get_health(),
            'correlation': {},
            'market_impact': {},
            'alpha': {},
            'weights': {},
            'drift': {}
        }
        
        # Correlation data
        if self._correlation_controller:
            try:
                state = self._correlation_controller.get_current_state()
                data['correlation'] = {
                    'state': state.value if state else 'unknown',
                    'prediction': self.get_correlation_prediction(1)
                }
            except Exception:
                pass
        
        # Weights data
        if self._weight_manager:
            try:
                data['weights'] = self.get_strategy_weights()
            except Exception:
                pass
        
        return data
    
    async def shutdown(self):
        """Gracefully shutdown all components"""
        logger.info("Shutting down SOTA v5.2.1 integration")
        
        # Unsubscribe from events
        if self._event_bus:
            for sub in self._subscriptions:
                try:
                    self._event_bus.unsubscribe(sub)
                except Exception:
                    pass
        
        self._initialized = False
        self._state = IntegrationState.PARTIAL
        
        logger.info("Shutdown complete")


# =============================================================================
# Fallback EventBus
# =============================================================================

class _FallbackEventBus:
    """Fallback event bus when unified events unavailable"""
    
    def __init__(self):
        self._handlers = {}
    
    def subscribe(self, event_type: str, handler):
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
    
    def unsubscribe(self, event_type: str):
        self._handlers.pop(event_type, None)
    
    def emit(self, event_type: str, data: Dict):
        for handler in self._handlers.get(event_type, []):
            try:
                handler(data)
            except Exception as e:
                logger.warning(f"Handler error for {event_type}: {e}")


# =============================================================================
# Convenience Functions
# =============================================================================

_integrator_instance: Optional[SOTAV521CompleteIntegrator] = None


async def init_sota_v521_complete(
    config: Optional[SOTAV521CompleteConfig] = None
) -> SOTAV521CompleteIntegrator:
    """
    Initialize and return global SOTA v5.2.1 integrator.
    
    Usage:
        integrator = await init_sota_v521_complete()
    """
    global _integrator_instance
    
    if _integrator_instance is None:
        _integrator_instance = SOTAV521CompleteIntegrator(config)
        await _integrator_instance.initialize()
    
    return _integrator_instance


def get_sota_v521_integrator() -> Optional[SOTAV521CompleteIntegrator]:
    """Get global integrator instance"""
    return _integrator_instance


# =============================================================================
# Test / Demo
# =============================================================================

async def _demo():
    """Demo usage of complete integration"""
    print("=" * 60)
    print("SOTA v5.2.1 COMPLETE INTEGRATION DEMO")
    print("=" * 60)
    
    # Initialize with all features
    config = SOTAV521CompleteConfig.enable_all()
    integrator = SOTAV521CompleteIntegrator(config)
    
    print("\n[1] Initializing...")
    success = await integrator.initialize()
    print(f"    Init success: {success}")
    
    health = integrator.get_health()
    print(f"    State: {health['state']}")
    for comp, status in health['components'].items():
        init_status = "✓" if status['initialized'] else "✗"
        print(f"    {comp}: {init_status}")
    
    # Simulate price updates
    print("\n[2] Updating prices for correlation analysis...")
    for i in range(30):
        prices = {
            "BTC": 100000 + i * 100,
            "ETH": 3500 + i * 10,
            "SOL": 200 + i * 1
        }
        integrator.update_prices(prices)
    
    # Get correlation adjustment
    positions = {"BTC": 1.0, "ETH": 2.0, "SOL": 10.0}
    adjustment = integrator.get_correlation_position_adjustment(positions)
    if adjustment:
        print(f"    Correlation state: {adjustment['state']}")
        print(f"    Action: {adjustment['action']}")
        print(f"    Scale factor: {adjustment['scale_factor']}")
    
    # Simulate market impact
    print("\n[3] Testing market impact prediction...")
    integrator.update_market_data("BTC-USD", daily_volume=5e9, volatility=0.6)
    impact = integrator.predict_market_impact("BTC-USD", "buy", 100000)
    if impact:
        print(f"    Temp impact: {impact['temp_impact_bps']:.2f} bps")
        print(f"    Total impact: {impact['total_impact_bps']:.2f} bps")
    
    # Get slicing plan
    plan = integrator.get_optimal_slicing("BTC-USD", "buy", 500000, "medium")
    if plan:
        print(f"    Slices: {plan['num_slices']}")
        print(f"    Order type: {plan['order_type']}")
    
    # Simulate alpha fusion
    print("\n[4] Testing alpha fusion...")
    integrator.record_onchain_activity(
        "whale_001", "buy", 50000, "SOL", 2_000_000, is_whale=True
    )
    integrator.update_sentiment(fear_greed=18, twitter=0.3, sol_funding=0.015)
    integrator.ingest_news("Solana ecosystem sees major breakout")
    
    signal = integrator.generate_alpha_signal("SOL")
    if signal:
        print(f"    Direction: {signal['direction']:.2f}")
        print(f"    Confidence: {signal['confidence']:.2f}")
        print(f"    Trade bias: {signal['trade_bias']}")
    
    # Simulate adaptive weights
    print("\n[5] Testing adaptive weights...")
    strategies = ["trend_following", "momentum", "mean_reversion"]
    for _ in range(20):
        for strat in strategies:
            pnl = 0.02 if strat == "trend_following" else -0.01
            integrator.record_strategy_trade(strat, "BTC", pnl)
    
    weights = integrator.get_strategy_weights()
    print(f"    Weights: {weights}")
    
    # Fuse signals
    signals = {"trend_following": 0.6, "momentum": 0.3, "mean_reversion": -0.2}
    fused, conf = integrator.fuse_strategy_signals(signals)
    print(f"    Fused signal: {fused:.3f} (conf: {conf:.3f})")
    
    # Test drift detection
    print("\n[6] Testing drift detection...")
    drift = integrator.check_rl_drift({"volatility": 0.8, "momentum": 0.3})
    if drift:
        print(f"    Use RL: {drift['use_rl']}")
        print(f"    Severity: {drift['severity']}")
    
    # Get dashboard data
    print("\n[7] Dashboard data...")
    dashboard = integrator.get_dashboard_data()
    print(f"    Keys: {list(dashboard.keys())}")
    
    # Shutdown
    print("\n[8] Shutting down...")
    await integrator.shutdown()
    print("    Done!")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(_demo())
