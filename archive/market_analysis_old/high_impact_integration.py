"""
================================================================================
HIGH IMPACT INTEGRATION - Unified Access to All v3.3 Safety Components
================================================================================
Version: 3.3.0
Purpose: Provides single integration point for all HIGH impact audit fixes

This module connects:
1. UnifiedPositionSizer - Position sizing with tranche integration
2. GlobalExposureCapManager - Hard gross exposure limits
3. IntradayCorrelationMonitor - Correlation collapse detection
4. ExecutionLoopController - 200ms timing control
5. StopLossAuthorityManager - Stops that respect authority
6. HotRestartPersistence - State persistence for restarts
7. WeekendLiquidityManager - Time-based exposure reduction

USAGE:
    from core.high_impact_integration import HighImpactSafetyLayer
    
    safety = HighImpactSafetyLayer()
    safety.initialize()
    
    # Before any trade
    validation = safety.validate_trade_request(
        asset="SOL",
        requested_exposure=0.30,
        direction="long",
        ...
    )
    
    if validation.is_allowed:
        # Proceed with trade at validation.adjusted_size
        pass
================================================================================
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Any
from datetime import datetime

# Import all HIGH impact modules
from risk.unified_position_sizer import (
    UnifiedPositionSizer,
    PositionSizingConfig,
    PositionSizingResult,
    TrancheLevel,
)
from risk.global_exposure_cap import (
    GlobalExposureCapManager,
    ExposureCapConfig,
    ExposureValidationResult,
)
from market.intraday_correlation_monitor import (
    IntradayCorrelationMonitor,
    CorrelationConfig,
    CorrelationState,
    CorrelationState,
)
from execution.execution_loop_controller import (
    ExecutionLoopController,
    ExecutionLoopConfig,
    ExecutionModifier,
    ExecutionMode,
)
from risk.stop_loss_authority import (
    StopLossAuthorityManager,
    StopConfig,
    StopDecision,
    AuthorityState,
    SystemMode,
)
from infra.hot_restart_persistence import (
    HotRestartPersistence,
    SystemState,
    PersistenceEvent,
)
from liquidity.weekend_manager import (
    WeekendLiquidityManager,
    WeekendConfig,
    WeekendState,
    TradingSession,
)

logger = logging.getLogger(__name__)


@dataclass
class TradeValidationRequest:
    """Request for trade validation."""
    asset: str
    direction: str  # "long" or "short"
    requested_exposure_delta: float
    tranche_level: TrancheLevel
    account_balance: float
    stop_loss_pct: float
    current_btc_exposure: float = 0.0
    current_eth_exposure: float = 0.0
    current_sol_exposure: float = 0.0
    is_sol_dominant: bool = False


@dataclass
class TradeValidationResult:
    """Result of trade validation through safety layer."""
    
    # Core decision
    is_allowed: bool
    adjusted_size: float
    adjusted_exposure: float
    
    # Blocking reasons (if any)
    blocked_by: str = ""  # Which safety check blocked
    block_reason: str = ""
    
    # Component results
    position_sizing: Optional[PositionSizingResult] = None
    exposure_cap: Optional[ExposureValidationResult] = None
    correlation: Optional[CorrelationState] = None
    liquidity: Optional[WeekendState] = None
    
    # Execution guidance
    execution_mode: ExecutionMode = ExecutionMode.PASSIVE_PREFERRED
    stop_decision: Optional[StopDecision] = None
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict:
        return {
            "is_allowed": self.is_allowed,
            "adjusted_size": self.adjusted_size,
            "adjusted_exposure": self.adjusted_exposure,
            "blocked_by": self.blocked_by,
            "block_reason": self.block_reason,
            "execution_mode": self.execution_mode.value,
            "timestamp": self.timestamp.isoformat(),
        }


class HighImpactSafetyLayer:
    """
    Unified safety layer integrating all HIGH impact fixes.
    
    This is the GATEKEEPER for all trading decisions.
    Every trade request MUST pass through this layer.
    
    VALIDATION ORDER:
    1. Weekend/Liquidity check (is now a reduced period?)
    2. Correlation check (is correlation dangerously high?)
    3. Global exposure cap (will this exceed gross limits?)
    4. Position sizing (what size is allowed by risk rules?)
    5. Stop calculation (what stops are required?)
    
    If any check fails, the trade is blocked or adjusted.
    """
    
    def __init__(
        self,
        position_config: Optional[PositionSizingConfig] = None,
        exposure_config: Optional[ExposureCapConfig] = None,
        correlation_config: Optional[CorrelationConfig] = None,
        execution_config: Optional[ExecutionLoopConfig] = None,
        stop_config: Optional[StopConfig] = None,
        persistence_db_path: str = "hmats_state.db",
        liquidity_config: Optional[WeekendConfig] = None,
    ):
        # Initialize all components
        self.position_sizer = UnifiedPositionSizer(position_config)
        self.exposure_cap = GlobalExposureCapManager(exposure_config)
        self.correlation_monitor = IntradayCorrelationMonitor(correlation_config)
        self.execution_loop = ExecutionLoopController(execution_config)
        self.stop_manager = StopLossAuthorityManager(stop_config)
        self.persistence = HotRestartPersistence(persistence_db_path)
        self.liquidity_manager = WeekendLiquidityManager(liquidity_config)
        
        self._initialized = False
        
        logger.info("HighImpactSafetyLayer created")
    
    def initialize(self):
        """Initialize all components and load persisted state."""
        
        # Load persisted state
        saved_state = self.persistence.load()
        if saved_state:
            logger.info(f"Loaded persisted state: mode={saved_state.system_mode}")
            # Restore component states from persisted state
            try:
                if hasattr(saved_state, 'system_mode') and saved_state.system_mode:
                    self.mode_manager.set_mode(saved_state.system_mode)
                if hasattr(saved_state, 'positions') and saved_state.positions:
                    for asset, pos in saved_state.positions.items():
                        self.position_sizer.restore_position(asset, pos)
                if hasattr(saved_state, 'risk_state') and saved_state.risk_state:
                    self.risk_manager.restore_state(saved_state.risk_state)
                logger.info("Successfully restored component states from persistence")
            except Exception as e:
                logger.warning(f"Failed to restore some component states: {e}")
        
        # Start execution loop
        self.execution_loop.start()
        
        self._initialized = True
        logger.info("HighImpactSafetyLayer initialized")
    
    def shutdown(self):
        """Shutdown components and persist state."""
        
        # Stop execution loop
        self.execution_loop.stop()
        
        # Persist current state
        self._persist_current_state(PersistenceEvent.SHUTDOWN)
        
        self._initialized = False
        logger.info("HighImpactSafetyLayer shutdown")
    
    def validate_trade_request(
        self,
        request: TradeValidationRequest,
        authority_state: Optional[AuthorityState] = None,
    ) -> TradeValidationResult:
        """
        Validate a trade request through all safety checks.
        
        Returns adjusted size and whether trade is allowed.
        """
        
        if not self._initialized:
            return TradeValidationResult(
                is_allowed=False,
                adjusted_size=0.0,
                adjusted_exposure=0.0,
                blocked_by="system",
                block_reason="Safety layer not initialized",
            )
        
        # Step 1: Weekend/Liquidity check
        liquidity_state = self.liquidity_manager.get_current_state()
        
        if liquidity_state.period in [TradingSession.HOLIDAY]:
            return TradeValidationResult(
                is_allowed=False,
                adjusted_size=0.0,
                adjusted_exposure=0.0,
                blocked_by="liquidity",
                block_reason=f"Trading blocked during {liquidity_state.period.value}",
                liquidity=liquidity_state,
            )
        
        # Step 2: Correlation check
        correlation = self.correlation_monitor.get_current_state()
        
        if correlation.state == CorrelationState.COLLAPSE:
            return TradeValidationResult(
                is_allowed=False,
                adjusted_size=0.0,
                adjusted_exposure=0.0,
                blocked_by="correlation",
                block_reason=f"Correlation collapse ({correlation.average_correlation:.3f})",
                correlation=correlation,
                liquidity=liquidity_state,
            )
        
        # Step 3: Update exposure state and validate
        self.exposure_cap.update_positions(
            btc_exposure=request.current_btc_exposure,
            eth_exposure=request.current_eth_exposure,
            sol_exposure=request.current_sol_exposure,
        )
        
        exposure_result = self.exposure_cap.validate_new_exposure(
            asset=request.asset,
            requested_exposure_delta=request.requested_exposure_delta,
            correlation_score=correlation.average_correlation,
            is_sol_dominant=request.is_sol_dominant,
            is_crisis=correlation.state == CorrelationState.DANGER,
        )
        
        if not exposure_result.is_allowed:
            return TradeValidationResult(
                is_allowed=False,
                adjusted_size=0.0,
                adjusted_exposure=0.0,
                blocked_by="exposure_cap",
                block_reason=exposure_result.adjustment_reason,
                exposure_cap=exposure_result,
                correlation=correlation,
                liquidity=liquidity_state,
            )
        
        # Step 4: Calculate position size
        # Get drawdown status from RiskManager
        drawdown_status = "NORMAL"
        if hasattr(self, 'risk_manager') and self.risk_manager:
            try:
                risk_state = self.risk_manager.get_state()
                if hasattr(risk_state, 'drawdown_status'):
                    drawdown_status = risk_state.drawdown_status
                elif hasattr(risk_state, 'current_drawdown'):
                    # Infer status from drawdown percentage
                    dd = risk_state.current_drawdown
                    if dd > 0.15:
                        drawdown_status = "CRITICAL"
                    elif dd > 0.10:
                        drawdown_status = "WARNING"
                    elif dd > 0.05:
                        drawdown_status = "CAUTION"
            except Exception:
                pass
        
        # Calculate current gross for position sizer
        current_gross = (
            abs(request.current_btc_exposure) +
            abs(request.current_eth_exposure) +
            abs(request.current_sol_exposure)
        )
        
        current_asset_exposure = getattr(
            request,
            f"current_{request.asset.lower()}_exposure",
            0.0
        )
        
        # Apply liquidity period caps
        liquidity_caps = self.liquidity_manager.get_adjusted_caps(request.is_sol_dominant)
        asset_cap_override = liquidity_caps.get(f"{request.asset.lower()}_cap")
        
        position_result = self.position_sizer.calculate_position_size(
            asset=request.asset,
            account_balance=request.account_balance,
            stop_loss_pct=request.stop_loss_pct,
            tranche_level=request.tranche_level,
            current_gross_exposure=current_gross,
            current_asset_exposure=current_asset_exposure,
            drawdown_status=drawdown_status,
            asset_cap_override=asset_cap_override,
        )
        
        if not position_result.is_valid:
            return TradeValidationResult(
                is_allowed=False,
                adjusted_size=0.0,
                adjusted_exposure=0.0,
                blocked_by="position_sizing",
                block_reason=position_result.rejection_reason,
                position_sizing=position_result,
                exposure_cap=exposure_result,
                correlation=correlation,
                liquidity=liquidity_state,
            )
        
        # Step 5: Calculate stops
        if authority_state is None:
            authority_state = AuthorityState()
        
        # Get entry price from request or market data
        entry_price = getattr(request, 'entry_price', None) or getattr(request, 'current_price', None)
        if entry_price is None:
            # Try to get from market data
            entry_price = getattr(request, 'market_price', 100.0)
        
        stop_decision = self.stop_manager.calculate_stops(
            asset=request.asset,
            entry_price=entry_price,
            direction=request.direction,
            authority_state=authority_state,
            size=position_result.final_size_usd,
        )
        
        # Get execution mode
        exec_modifier = self.execution_loop.get_current_modifier()
        
        # Success - return validated trade
        return TradeValidationResult(
            is_allowed=True,
            adjusted_size=min(
                position_result.final_size_usd,
                exposure_result.adjusted_size * request.account_balance
            ),
            adjusted_exposure=min(
                position_result.final_exposure_pct,
                exposure_result.adjusted_size
            ),
            position_sizing=position_result,
            exposure_cap=exposure_result,
            correlation=correlation,
            liquidity=liquidity_state,
            execution_mode=exec_modifier.mode,
            stop_decision=stop_decision,
        )
    
    def update_correlation_data(
        self,
        btc_price: float,
        eth_price: float,
        sol_price: float,
    ):
        """Update correlation monitor with new prices."""
        self.correlation_monitor.add_prices(btc_price, eth_price, sol_price)
    
    def set_execution_modifier(self, modifier: ExecutionModifier):
        """Set execution modifier (from Lead-Lag, VPIN, etc.)."""
        self.execution_loop.set_modifier(modifier)
    
    def update_authority_state(self, state: AuthorityState):
        """Update authority state (affects stops, etc.)."""
        self.stop_manager.update_authority(state)
    
    def should_force_derisk(self) -> Tuple[bool, float, str]:
        """
        Check if forced de-risking is required.
        
        Returns:
            (should_derisk, target_scale, reason)
        """
        # Check correlation
        should_derisk, scale = self.correlation_monitor.should_force_derisk()
        if should_derisk:
            return True, scale, "Correlation collapse"
        
        # Check weekend/liquidity
        should_derisk, scale = self.liquidity_manager.should_reduce_exposure()
        if should_derisk:
            return True, scale, f"Liquidity period: {self.liquidity_manager.get_current_state().period.value}"
        
        return False, 1.0, ""
    
    def get_current_caps(self) -> Dict[str, float]:
        """Get all current exposure caps."""
        liquidity = self.liquidity_manager.get_current_state()
        correlation = self.correlation_monitor.get_current_state()
        
        # Get caps from exposure manager considering correlation
        gross_cap, _ = self.exposure_cap.get_current_gross_cap(
            correlation_score=correlation.average_correlation,
            is_crisis=correlation.state == CorrelationState.DANGER,
        )
        
        # Apply liquidity caps
        liquidity_caps = self.liquidity_manager.get_adjusted_caps()
        
        return {
            "gross_cap": min(gross_cap, liquidity.current_cap),
            "btc_cap": liquidity_caps["btc_cap"],
            "eth_cap": liquidity_caps["eth_cap"],
            "sol_cap": liquidity_caps["sol_cap"],
            "correlation_scale": correlation.recommended_scale,
            "liquidity_period": liquidity.period.value,
            "correlation_state": correlation.state.value,
        }
    
    def _persist_current_state(self, event: PersistenceEvent):
        """Persist current state."""
        # Get system mode from ModeManager
        current_mode = "NORMAL"
        if hasattr(self, 'mode_manager') and self.mode_manager:
            try:
                mode_state = self.mode_manager.get_mode()
                current_mode = mode_state.name if hasattr(mode_state, 'name') else str(mode_state)
            except Exception:
                pass
        
        # Build state from components
        state = SystemState(
            system_mode=current_mode,
            opportunity_trigger=None,
        )
        
        # Add positions from position manager
        if hasattr(self, 'position_sizer') and self.position_sizer:
            try:
                state.positions = self.position_sizer.get_all_positions()
            except Exception:
                pass
        
        self.persistence.persist(state, event)
    
    def get_status(self) -> Dict[str, Any]:
        """Get complete safety layer status."""
        correlation = self.correlation_monitor.get_current_state()
        liquidity = self.liquidity_manager.get_current_state()
        
        return {
            "initialized": self._initialized,
            "correlation": {
                "state": correlation.state.value,
                "average": correlation.average_correlation,
                "scale": correlation.recommended_scale,
            },
            "liquidity": {
                "period": liquidity.period.value,
                "cap": liquidity.current_cap,
                "next_transition": str(liquidity.time_until_next_transition),
            },
            "execution": {
                "mode": self.execution_loop.get_current_modifier().mode.value,
                "tick_stats": self.execution_loop.get_tick_stats(),
            },
            "caps": self.get_current_caps(),
        }


# Singleton instance
_safety_layer: Optional[HighImpactSafetyLayer] = None


def get_safety_layer() -> HighImpactSafetyLayer:
    """Get or create singleton safety layer."""
    global _safety_layer
    if _safety_layer is None:
        _safety_layer = HighImpactSafetyLayer()
    return _safety_layer


def reset_safety_layer():
    """Reset singleton (for testing)."""
    global _safety_layer
    if _safety_layer:
        _safety_layer.shutdown()
    _safety_layer = None
