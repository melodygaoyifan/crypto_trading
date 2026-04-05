"""
================================================================================
HMATS v4.0-COMPLETE - INTERFACE COMPATIBILITY LAYER
================================================================================
Version: 4.0.0-COMPAT
Purpose: Backward-compatible wrappers for deprecated interfaces

MIGRATION NOTES:
    This module provides compatibility wrappers for interfaces that were
    renamed, moved, or restructured during the v3.3 -> v3.4 -> v3.5 -> v3.6 -> v4
    upgrade path.

DEPRECATED INTERFACES SUPPORTED:
    1. AgentSignal -> authority_fusion.AgentSignal (v3.4)
    2. AuthorityFusionEngine -> authority_fusion.AuthorityFusionEngine (v3.4)
    3. get_engine_v35() -> get_fusion_engine() (v3.5)
    4. RiskAgent -> risk_agent.RiskAgent (v3.4)
    5. MetaIntegration -> enhanced_meta_decision (v4)
    6. WeekendManager -> integrated_liquidity_manager (v4)
    7. ScenarioWalkthrough -> scenario_walkthroughs_unified (v4)

USAGE:
    # Old code (still works):
    from hmats.compat import AgentSignal, get_engine_v35
    
    # New recommended code:
    from integration.core.authority_fusion import AgentSignal, get_fusion_engine

WARNING:
    These wrappers are provided for backward compatibility only.
    New code should use the canonical imports from integration/core/.
================================================================================
"""

import logging
import warnings
from typing import Dict, Optional, Any, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# DEPRECATION WARNING HELPER
# =============================================================================

def _deprecated(old_name: str, new_name: str, version: str = "4.0"):
    """Issue deprecation warning for old interface usage."""
    warnings.warn(
        f"{old_name} is deprecated since v{version}. Use {new_name} instead.",
        DeprecationWarning,
        stacklevel=3
    )


# =============================================================================
# AUTHORITY FUSION COMPATIBILITY (v3.2 -> v3.4)
# =============================================================================

# Import canonical versions
try:
    from ..integration.core.authority_fusion import (
        AgentSignal as _AgentSignal,
        AuthorityFusionEngine as _AuthorityFusionEngine,
        FusionContext as _FusionContext,
        FusionResult as _FusionResult,
        Authority,
        SystemMode,
        get_fusion_engine as _get_fusion_engine,
        reset_fusion_engine as _reset_fusion_engine,
    )
    V36_AUTHORITY_AVAILABLE = True
except ImportError:
    V36_AUTHORITY_AVAILABLE = False


# Compatibility class for AgentSignal
class AgentSignal:
    """
    Backward-compatible AgentSignal wrapper.
    
    DEPRECATED: Use integration.core.authority_fusion.AgentSignal instead.
    """
    
    def __init__(
        self,
        direction: float = 0.0,
        confidence: float = 0.0,
        veto_active: bool = False,
        veto_direction: Optional[str] = None,
        leverage_cap: float = 1.0,
        exit_pressure: float = 0.0,
        tranche_advice: Optional[str] = None,
    ):
        _deprecated("AgentSignal", "integration.core.authority_fusion.AgentSignal")
        
        if V36_AUTHORITY_AVAILABLE:
            self._inner = _AgentSignal(
                direction=direction,
                confidence=confidence,
                veto_active=veto_active,
                veto_direction=veto_direction,
                leverage_cap=leverage_cap,
                exit_pressure=exit_pressure,
                tranche_advice=tranche_advice,
            )
        else:
            self.direction = direction
            self.confidence = confidence
            self.veto_active = veto_active
            self.veto_direction = veto_direction
            self.leverage_cap = leverage_cap
            self.exit_pressure = exit_pressure
            self.tranche_advice = tranche_advice
    
    def to_dict(self) -> Dict:
        if V36_AUTHORITY_AVAILABLE and hasattr(self, '_inner'):
            return self._inner.to_dict()
        return {
            "direction": self.direction,
            "confidence": self.confidence,
            "veto_active": self.veto_active,
            "leverage_cap": self.leverage_cap,
        }


def get_engine_v35() -> Any:
    """
    Backward-compatible get_engine_v35() function.
    
    DEPRECATED: Use integration.core.authority_fusion.get_fusion_engine() instead.
    """
    _deprecated("get_engine_v35()", "get_fusion_engine()")
    
    if V36_AUTHORITY_AVAILABLE:
        return _get_fusion_engine()
    else:
        raise ImportError("authority_fusion module not available")


def get_fusion_engine_v34() -> Any:
    """
    Backward-compatible get_fusion_engine_v34() function.
    
    DEPRECATED: Use integration.core.authority_fusion.get_fusion_engine() instead.
    """
    _deprecated("get_fusion_engine_v34()", "get_fusion_engine()")
    
    if V36_AUTHORITY_AVAILABLE:
        return _get_fusion_engine()
    else:
        raise ImportError("authority_fusion module not available")


# =============================================================================
# RISK AGENT COMPATIBILITY (v3.3 -> v3.4)
# =============================================================================

try:
    from ..integration.core.risk_agent import (
        RiskAgent as _RiskAgent,
        RiskConfig as _RiskConfig,
        RiskState as _RiskState,
        VetoReason,
        get_risk_agent as _get_risk_agent,
    )
    V36_RISK_AVAILABLE = True
except ImportError:
    V36_RISK_AVAILABLE = False


class RiskAgentV33:
    """
    Backward-compatible RiskAgent wrapper for v3.3 interface.
    
    DEPRECATED: Use integration.core.risk_agent.RiskAgent instead.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        _deprecated("RiskAgentV33", "integration.core.risk_agent.RiskAgent")
        
        if V36_RISK_AVAILABLE:
            self._inner = _get_risk_agent()
        else:
            self._inner = None
    
    def check_veto(
        self,
        direction: float,
        target_exposure: float,
        market_data: Dict,
        is_opportunity: bool = False,
    ) -> Tuple[bool, str]:
        """
        Check if risk should veto the trade.
        
        DEPRECATED interface - new interface uses check_entry_allowed().
        """
        _deprecated("check_veto()", "check_entry_allowed()")
        
        if self._inner:
            result = self._inner.check_entry_allowed(
                direction=direction,
                target_exposure=target_exposure,
                market_data=market_data,
                mode="OPPORTUNITY" if is_opportunity else "NORMAL",
            )
            return (not result.allowed, result.reason if not result.allowed else "")
        
        return (False, "")
    
    def get_exposure_cap(self, asset: str = "SOL") -> float:
        """Get current exposure cap."""
        if self._inner:
            return self._inner.get_exposure_cap(asset)
        return 1.0


def get_risk_agent_v33() -> RiskAgentV33:
    """
    Backward-compatible get_risk_agent_v33() function.
    
    DEPRECATED: Use integration.core.risk_agent.get_risk_agent() instead.
    """
    _deprecated("get_risk_agent_v33()", "get_risk_agent()")
    return RiskAgentV33()


# =============================================================================
# META INTEGRATION COMPATIBILITY (v3.3 -> v4)
# =============================================================================

try:
    from .enhanced_meta_decision import (
        EnhancedMetaDecisionLayer,
        SharedInformation,
        get_enhanced_meta_layer,
    )
    META_LAYER_AVAILABLE = True
except ImportError:
    META_LAYER_AVAILABLE = False


class MetaIntegration:
    """
    Backward-compatible MetaIntegration wrapper.
    
    DEPRECATED: Use integration.core.enhanced_meta_decision.EnhancedMetaDecisionLayer instead.
    """
    
    def __init__(self):
        _deprecated("MetaIntegration", "EnhancedMetaDecisionLayer")
        
        if META_LAYER_AVAILABLE:
            self._inner = get_enhanced_meta_layer()
        else:
            self._inner = None
    
    def update(self, market_data: Dict, signals: Dict = None):
        """Update meta integration with market data."""
        if self._inner:
            self._inner.update_context(market_data, signals)
    
    def get_aggressiveness_modifier(self) -> float:
        """Get aggressiveness modifier."""
        if self._inner:
            return self._inner.get_aggressiveness_modifier()
        return 1.0
    
    def get_adjusted_confidence(self, strategy: str, base: float) -> float:
        """Get adjusted confidence for strategy."""
        if self._inner:
            return self._inner.get_adjusted_confidence(strategy, base)
        return base


def get_meta_integration() -> MetaIntegration:
    """
    Backward-compatible get_meta_integration() function.
    
    DEPRECATED: Use integration.core.enhanced_meta_decision.get_enhanced_meta_layer() instead.
    """
    _deprecated("get_meta_integration()", "get_enhanced_meta_layer()")
    return MetaIntegration()


# =============================================================================
# WEEKEND MANAGER COMPATIBILITY (v3.3 -> v4)
# =============================================================================

try:
    from .integrated_liquidity_manager import (
        IntegratedWeekendLiquidityManager,
        IntegratedLiquidityState,
        LiquidityPeriod,
        get_integrated_liquidity_manager,
    )
    LIQUIDITY_MANAGER_AVAILABLE = True
except ImportError:
    LIQUIDITY_MANAGER_AVAILABLE = False


class WeekendManager:
    """
    Backward-compatible WeekendManager wrapper.
    
    DEPRECATED: Use integration.core.integrated_liquidity_manager instead.
    """
    
    def __init__(self):
        _deprecated("WeekendManager", "IntegratedWeekendLiquidityManager")
        
        if LIQUIDITY_MANAGER_AVAILABLE:
            self._inner = get_integrated_liquidity_manager()
        else:
            self._inner = None
    
    def is_weekend(self) -> bool:
        """Check if currently weekend."""
        if self._inner:
            state = self._inner.update()
            return state.is_weekend
        return False
    
    def get_position_cap(self, asset: str = "SOL") -> float:
        """Get position cap for weekend."""
        if self._inner:
            return self._inner.get_exposure_cap(asset, is_opportunity=False)
        return 1.0
    
    def validate_entry(
        self,
        asset: str,
        target_exposure: float,
        is_opportunity: bool = False,
    ) -> Tuple[bool, str]:
        """Validate weekend entry."""
        if self._inner:
            return self._inner.validate_entry(
                asset=asset,
                target_exposure=target_exposure,
                is_opportunity=is_opportunity,
            )
        return (True, "")


def get_weekend_manager() -> WeekendManager:
    """
    Backward-compatible get_weekend_manager() function.
    
    DEPRECATED: Use get_integrated_liquidity_manager() instead.
    """
    _deprecated("get_weekend_manager()", "get_integrated_liquidity_manager()")
    return WeekendManager()


# =============================================================================
# SCENARIO WALKTHROUGH COMPATIBILITY (v3.2 -> v4)
# =============================================================================

try:
    from .scenario_walkthroughs_unified import (
        ScenarioRunner,
        run_all_scenarios,
        run_live_scenarios,
        LiveScenarioType,
        LiveScenarioResult,
    )
    SCENARIO_AVAILABLE = True
except ImportError:
    SCENARIO_AVAILABLE = False


def scenario_1_sol_bullish():
    """
    Backward-compatible scenario_1_sol_bullish() function.
    
    DEPRECATED: Use scenario_walkthroughs_unified.simulate_sol_bullish_breakout() instead.
    """
    _deprecated("scenario_1_sol_bullish()", "simulate_sol_bullish_breakout()")
    
    if SCENARIO_AVAILABLE:
        from .scenario_walkthroughs_unified import simulate_sol_bullish_breakout
        result = simulate_sol_bullish_breakout(verbose=True)
        return result.passed
    return False


def scenario_2_sol_bearish():
    """
    Backward-compatible scenario_2_sol_bearish() function.
    
    DEPRECATED: Use scenario_walkthroughs_unified.simulate_sol_bearish_breakdown() instead.
    """
    _deprecated("scenario_2_sol_bearish()", "simulate_sol_bearish_breakdown()")
    
    if SCENARIO_AVAILABLE:
        from .scenario_walkthroughs_unified import simulate_sol_bearish_breakdown
        result = simulate_sol_bearish_breakdown(verbose=True)
        return result.passed
    return False


def run_v32_scenarios():
    """
    Backward-compatible run_v32_scenarios() function.
    
    DEPRECATED: Use run_live_scenarios() instead.
    """
    _deprecated("run_v32_scenarios()", "run_live_scenarios()")
    
    if SCENARIO_AVAILABLE:
        return run_live_scenarios(verbose=True)
    return []


# =============================================================================
# IDLE MODULE COMPATIBILITY (v3.2 -> v4)
# =============================================================================

try:
    from .idle_module_integrator_v4 import (
        IdleModuleIntegrator,
        IdleState,
        MarketActivityLevel,
        get_idle_module_integrator,
    )
    IDLE_AVAILABLE = True
except ImportError:
    IDLE_AVAILABLE = False


class IdleIntegrator:
    """
    Backward-compatible IdleIntegrator wrapper.
    
    DEPRECATED: Use integration.core.idle_module_integrator_v4.IdleModuleIntegrator instead.
    """
    
    def __init__(self):
        _deprecated("IdleIntegrator", "IdleModuleIntegrator")
        
        if IDLE_AVAILABLE:
            self._inner = get_idle_module_integrator()
        else:
            self._inner = None
    
    def update(self, market_data: Dict) -> Dict:
        """Update idle state."""
        if self._inner:
            state = self._inner.update(market_data)
            return state.to_dict()
        return {"activity_level": "ACTIVE"}
    
    def is_idle(self) -> bool:
        """Check if market is idle."""
        if self._inner:
            return self._inner.state.is_idle
        return False


def get_idle_integrator() -> IdleIntegrator:
    """
    Backward-compatible get_idle_integrator() function.
    
    DEPRECATED: Use get_idle_module_integrator() instead.
    """
    _deprecated("get_idle_integrator()", "get_idle_module_integrator()")
    return IdleIntegrator()


# =============================================================================
# CONSTITUTION GUARANTEES COMPATIBILITY (v3.3 -> v3.6)
# =============================================================================

try:
    from ..integration.core.constitution_guarantees import (
        ConstitutionGuaranteesModule,
        NoTradeState,
        OpportunityState,
        AlphaGatingResult,
        TrancheDecision,
        get_constitution_guarantees,
    )
    CONSTITUTION_AVAILABLE = True
except ImportError:
    CONSTITUTION_AVAILABLE = False


def get_constitution_v33():
    """
    Backward-compatible get_constitution_v33() function.
    
    DEPRECATED: Use get_constitution_guarantees() instead.
    """
    _deprecated("get_constitution_v33()", "get_constitution_guarantees()")
    
    if CONSTITUTION_AVAILABLE:
        return get_constitution_guarantees()
    raise ImportError("constitution_guarantees module not available")


# =============================================================================
# SIGNAL GENERATORS COMPATIBILITY (v3.2 -> v4)
# =============================================================================

try:
    from ..core.signal_generators_real import (
        QuantSignalGenerator,
        get_quant_generator,
    )
    SIGNAL_GEN_AVAILABLE = True
except ImportError:
    SIGNAL_GEN_AVAILABLE = False


def get_quant_generator_v32():
    """
    Backward-compatible get_quant_generator_v32() function.
    
    DEPRECATED: Use core.signal_generators_real.get_quant_generator() instead.
    """
    _deprecated("get_quant_generator_v32()", "get_quant_generator()")
    
    if SIGNAL_GEN_AVAILABLE:
        return get_quant_generator()
    raise ImportError("signal_generators_real module not available")


# =============================================================================
# TRANCHE MANAGER COMPATIBILITY (v3.2 -> v4)
# =============================================================================

try:
    from ..core.tranche_manager import (
        TrancheManager,
        TrancheLevel,
        get_tranche_manager,
    )
    TRANCHE_AVAILABLE = True
except ImportError:
    TRANCHE_AVAILABLE = False


def get_tranche_manager_v32():
    """
    Backward-compatible get_tranche_manager_v32() function.
    
    DEPRECATED: Use core.tranche_manager.get_tranche_manager() instead.
    """
    _deprecated("get_tranche_manager_v32()", "get_tranche_manager()")
    
    if TRANCHE_AVAILABLE:
        return get_tranche_manager()
    raise ImportError("tranche_manager module not available")


# =============================================================================
# ALL COMPATIBILITY EXPORTS
# =============================================================================

__all__ = [
    # Authority Fusion (v3.2 -> v3.4)
    "AgentSignal",
    "get_engine_v35",
    "get_fusion_engine_v34",
    
    # Risk Agent (v3.3 -> v3.4)
    "RiskAgentV33",
    "get_risk_agent_v33",
    
    # Meta Integration (v3.3 -> v4)
    "MetaIntegration",
    "get_meta_integration",
    
    # Weekend Manager (v3.3 -> v4)
    "WeekendManager",
    "get_weekend_manager",
    
    # Scenario Walkthroughs (v3.2 -> v4)
    "scenario_1_sol_bullish",
    "scenario_2_sol_bearish",
    "run_v32_scenarios",
    
    # Idle Module (v3.2 -> v4)
    "IdleIntegrator",
    "get_idle_integrator",
    
    # Constitution (v3.3 -> v3.6)
    "get_constitution_v33",
    
    # Signal Generators (v3.2 -> v4)
    "get_quant_generator_v32",
    
    # Tranche Manager (v3.2 -> v4)
    "get_tranche_manager_v32",
]


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("HMATS v4.0-COMPLETE - INTERFACE COMPATIBILITY LAYER")
    print("=" * 70)
    print()
    print("This module provides backward-compatible wrappers for deprecated interfaces.")
    print()
    print("Supported deprecated interfaces:")
    for name in __all__:
        print(f"  - {name}")
    print()
    print("WARNING: These wrappers will issue deprecation warnings when used.")
    print("New code should use the canonical imports from integration/core/.")
