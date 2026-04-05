"""
================================================================================
HMATS Risk Module - 风险管理系统
================================================================================
Version: 5.4.1-P0FIX

SOTA对照: "波动率驱动的杠杆调节（Volatility Targeting）"

核心组件:
- VolatilityTargeting: 波动率目标仓位调整
- PositionSizer: 统一仓位计算
- StopLoss: 自适应止损管理
- TrancheManager: 梯队仓位管理
- ExposureCap: 全局敞口上限
- LeverageGuard: P0 FIX - 杠杆/保证金硬限制 (v5.4.1)

IMPORT STRATEGY:
    P0 modules (LeverageGuard) are imported FIRST and UNCONDITIONALLY.
    Legacy modules use lazy import to prevent cascade failures.
================================================================================
"""

__version__ = "5.4.1-P0FIX"

import logging as _logging
_logger = _logging.getLogger(__name__)

# =============================================================================
# P0 FIX: Leverage Guard (v5.4.1) - UNCONDITIONAL IMPORT
# This module has NO external dependencies and MUST always be available
# =============================================================================

from .leverage_guard import (
    LeverageGuard,
    LeverageConfig,
    MarginMode,
    get_leverage_guard,
    reset_leverage_guard,
)

# =============================================================================
# LEGACY MODULES - LAZY IMPORT (may have complex dependencies)
# =============================================================================

# Availability flags
_VOLATILITY_TARGETING_AVAILABLE = False
_RISK_MANAGER_AVAILABLE = False
_UNIFIED_POSITION_SIZER_AVAILABLE = False
_GLOBAL_EXPOSURE_CAP_AVAILABLE = False
_STOP_LOSS_AUTHORITY_AVAILABLE = False
_ADAPTIVE_STOP_AVAILABLE = False
_SOTA_RISK_CONTROLLER_AVAILABLE = False

# Lazy import containers
VolatilityTargeting = None
RiskManager = None
UnifiedPositionSizer = None
PositionSizingResult = None
PositionSizingConfig = None
TrancheLevel = None
get_unified_position_sizer = None
reset_unified_position_sizer = None
GlobalExposureCapManager = None
ExposureValidationResult = None
ExposureState = None
ExposureCapConfig = None
ExposureCapLevel = None
get_exposure_cap_manager = None
reset_exposure_cap_manager = None
StopLossAuthorityManager = None
StopDecision = None
StopOrder = None
StopConfig = None
StopType = None
StopStatus = None
get_stop_authority_manager = None
reset_stop_authority_manager = None
MultiWindowATRState = None
AdaptiveStopType = None
StopTriggerReason = None
VolatilityRegime = None
SOTARiskController = None
SOTARiskConfig = None
RiskState = None
CorrelationState = None
KillSwitchReason = None
get_sota_risk_controller = None
reset_sota_risk_controller = None

# Try to import volatility_targeting
try:
    from .volatility_targeting import VolatilityTargeting
    _VOLATILITY_TARGETING_AVAILABLE = True
except ImportError as e:
    _logger.debug(f"volatility_targeting not available: {e}")

# Try to import risk_manager
try:
    from .risk_manager import RiskManager
    _RISK_MANAGER_AVAILABLE = True
except ImportError as e:
    _logger.debug(f"risk_manager not available: {e}")

# Try to import unified_position_sizer
try:
    from .unified_position_sizer import (
        UnifiedPositionSizer,
        PositionSizingResult,
        PositionSizingConfig,
        TrancheLevel,
        get_unified_position_sizer,
        reset_unified_position_sizer
    )
    _UNIFIED_POSITION_SIZER_AVAILABLE = True
except ImportError as e:
    _logger.debug(f"unified_position_sizer not available: {e}")

# Try to import global_exposure_cap
try:
    from .global_exposure_cap import (
        GlobalExposureCapManager,
        ExposureValidationResult,
        ExposureState,
        ExposureCapConfig,
        ExposureCapLevel,
        get_exposure_cap_manager,
        reset_exposure_cap_manager
    )
    _GLOBAL_EXPOSURE_CAP_AVAILABLE = True
except ImportError as e:
    _logger.debug(f"global_exposure_cap not available: {e}")

# Try to import stop_loss_authority
try:
    from .stop_loss_authority import (
        StopLossAuthorityManager,
        StopDecision,
        StopOrder,
        StopConfig,
        StopType,
        StopStatus,
        get_stop_authority_manager,
        reset_stop_authority_manager
    )
    _STOP_LOSS_AUTHORITY_AVAILABLE = True
except ImportError as e:
    _logger.debug(f"stop_loss_authority not available: {e}")

# Try to import adaptive_stop
try:
    from .adaptive_stop import (
        MultiWindowATRState,
        StopType as AdaptiveStopType,
        StopTriggerReason,
        VolatilityRegime
    )
    _ADAPTIVE_STOP_AVAILABLE = True
except ImportError as e:
    _logger.debug(f"adaptive_stop not available: {e}")

# Try to import sota_risk_controller
try:
    from .sota_risk_controller import (
        SOTARiskController,
        SOTARiskConfig,
        RiskState,
        CorrelationState,
        KillSwitchReason,
        get_sota_risk_controller,
        reset_sota_risk_controller
    )
    _SOTA_RISK_CONTROLLER_AVAILABLE = True
except ImportError as e:
    _logger.debug(f"sota_risk_controller not available: {e}")


# =============================================================================
# __all__ - Only export what's actually available
# =============================================================================

__all__ = [
    # P0 FIX: Leverage Guard (ALWAYS available)
    "LeverageGuard",
    "LeverageConfig",
    "MarginMode",
    "get_leverage_guard",
    "reset_leverage_guard",
]

# Add legacy exports if available
if _VOLATILITY_TARGETING_AVAILABLE:
    __all__.append("VolatilityTargeting")

if _RISK_MANAGER_AVAILABLE:
    __all__.append("RiskManager")

if _UNIFIED_POSITION_SIZER_AVAILABLE:
    __all__.extend([
        "UnifiedPositionSizer",
        "PositionSizingResult",
        "PositionSizingConfig",
        "TrancheLevel",
        "get_unified_position_sizer",
        "reset_unified_position_sizer",
    ])

if _GLOBAL_EXPOSURE_CAP_AVAILABLE:
    __all__.extend([
        "GlobalExposureCapManager",
        "ExposureValidationResult",
        "ExposureState",
        "ExposureCapConfig",
        "ExposureCapLevel",
        "get_exposure_cap_manager",
        "reset_exposure_cap_manager",
    ])

if _STOP_LOSS_AUTHORITY_AVAILABLE:
    __all__.extend([
        "StopLossAuthorityManager",
        "StopDecision",
        "StopOrder",
        "StopConfig",
        "StopType",
        "StopStatus",
        "get_stop_authority_manager",
        "reset_stop_authority_manager",
    ])

if _ADAPTIVE_STOP_AVAILABLE:
    __all__.extend([
        "MultiWindowATRState",
        "AdaptiveStopType",
        "StopTriggerReason",
        "VolatilityRegime",
    ])

if _SOTA_RISK_CONTROLLER_AVAILABLE:
    __all__.extend([
        "SOTARiskController",
        "SOTARiskConfig",
        "RiskState",
        "CorrelationState",
        "KillSwitchReason",
        "get_sota_risk_controller",
        "reset_sota_risk_controller",
    ])


# =============================================================================
# MODULE STATUS CHECK
# =============================================================================

def get_module_status():
    """Get availability status of all risk modules."""
    return {
        "version": __version__,
        # P0 modules (always True)
        "leverage_guard": True,
        # Legacy modules (may be False)
        "volatility_targeting": _VOLATILITY_TARGETING_AVAILABLE,
        "risk_manager": _RISK_MANAGER_AVAILABLE,
        "unified_position_sizer": _UNIFIED_POSITION_SIZER_AVAILABLE,
        "global_exposure_cap": _GLOBAL_EXPOSURE_CAP_AVAILABLE,
        "stop_loss_authority": _STOP_LOSS_AUTHORITY_AVAILABLE,
        "adaptive_stop": _ADAPTIVE_STOP_AVAILABLE,
        "sota_risk_controller": _SOTA_RISK_CONTROLLER_AVAILABLE,
    }


# =============================================================================
# v6.1.3 GOVERNORS (Long-Term Survival)
# =============================================================================

# Opportunity Budget Governor
try:
    from .opportunity_budget_governor import (
        OpportunityBudgetGovernor,
        OpportunityBudgetConfig,
        OpportunityType,
        BudgetCheckResult,
        get_opportunity_budget_governor,
        reset_opportunity_budget_governor,
        check_opportunity_budget,
    )
    _OPPORTUNITY_BUDGET_AVAILABLE = True
    __all__.extend([
        "OpportunityBudgetGovernor",
        "OpportunityBudgetConfig",
        "OpportunityType",
        "BudgetCheckResult",
        "get_opportunity_budget_governor",
        "reset_opportunity_budget_governor",
        "check_opportunity_budget",
    ])
except ImportError:
    _OPPORTUNITY_BUDGET_AVAILABLE = False

# Regime Transition Buffer
try:
    from .regime_transition_buffer import (
        RegimeTransitionBuffer,
        RegimeTransitionConfig,
        TransitionState,
        TransitionCondition,
        TransitionCheckResult,
        get_regime_transition_buffer,
        reset_regime_transition_buffer,
        check_regime_transition,
    )
    _REGIME_TRANSITION_AVAILABLE = True
    __all__.extend([
        "RegimeTransitionBuffer",
        "RegimeTransitionConfig",
        "TransitionState",
        "TransitionCondition",
        "TransitionCheckResult",
        "get_regime_transition_buffer",
        "reset_regime_transition_buffer",
        "check_regime_transition",
    ])
except ImportError:
    _REGIME_TRANSITION_AVAILABLE = False

# Cascade Exhaustion Governor
try:
    from .cascade_exhaustion_governor import (
        CascadeExhaustionGovernor,
        CascadeExhaustionConfig,
        CascadePhase,
        TrancheLevel,
        TrancheCheckResult,
        get_cascade_exhaustion_governor,
        reset_cascade_exhaustion_governor,
        check_cascade_tranche,
    )
    _CASCADE_EXHAUSTION_AVAILABLE = True
    __all__.extend([
        "CascadeExhaustionGovernor",
        "CascadeExhaustionConfig",
        "CascadePhase",
        "TrancheLevel",
        "TrancheCheckResult",
        "get_cascade_exhaustion_governor",
        "reset_cascade_exhaustion_governor",
        "check_cascade_tranche",
    ])
except ImportError:
    _CASCADE_EXHAUSTION_AVAILABLE = False


# =============================================================================
# v6.2.3 BACKFILL: Strategy Allocator
# =============================================================================

_STRATEGY_ALLOCATOR_AVAILABLE = False
StrategyAllocator = None
StrategyState = None
AllocationResult = None

try:
    from .strategy_allocator import (
        StrategyAllocator,
        StrategyState,
        AllocationResult,
        Regime as AllocatorRegime,
    )
    _STRATEGY_ALLOCATOR_AVAILABLE = True
    __all__.extend([
        "StrategyAllocator",
        "StrategyState",
        "AllocationResult",
        "AllocatorRegime",
    ])
except ImportError as e:
    _logger.debug(f"strategy_allocator not available: {e}")
