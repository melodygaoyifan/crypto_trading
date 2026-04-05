"""
================================================================================
HMATS v5.4.1 - CORE RUNTIME SPINE
================================================================================
Version: 5.4.1-P0FIX
Purpose: Canonical runtime spine that wires all components together

This module provides the single source of truth for system flow:
    data -> regime -> agents -> fusion -> execution -> risk -> feedback

P0 FIX ADDITIONS (v5.4.1):
    - unit_system: Canonical unit type enforcement
    - account_sync: Real-time account equity provider
    - authority_chain: Risk check pipeline with one-veto-kill
    - exchange_guard: Kraken-only runtime enforcement

IMPORT STRATEGY:
    P0 modules are imported FIRST and UNCONDITIONALLY.
    Legacy modules use lazy import to prevent cascade failures.

All other modules connect through this spine.
================================================================================
"""

__version__ = "5.4.1-P0FIX"

import logging as _logging
_logger = _logging.getLogger(__name__)

# =============================================================================
# P0 FIX MODULES - UNCONDITIONAL IMPORT (v5.4.1)
# These modules have NO external dependencies and MUST always be available
# =============================================================================

# P0 FIX: Unit System (v5.4.1) - CRITICAL
from .unit_system import (
    UnitType,
    TypedValue,
    exposure_to_quantity,
    quantity_to_exposure,
    validate_exposure_fraction,
    validate_base_quantity,
)

# P0 FIX: Account Sync (v5.4.1) - CRITICAL
from .account_sync import (
    AccountSyncManager,
    AccountState,
    EquityStatus,
    get_account_sync,
    reset_account_sync,
    validate_exchange_kraken_only,
)

# P0 FIX: Exchange Guard (v5.4.1) - CRITICAL
from .exchange_guard import (
    ExchangeGuard,
    validate_kraken_only,
    assert_not_dex_execution,
    detect_exchange_from_client,
    detect_exchange_from_url,
    get_exchange_guard,
    reset_exchange_guard,
    NonKrakenExchangeError,
    DEXExecutionBlockedError,
)

# P0 FIX: Authority Chain (v5.4.1) - CRITICAL
from .authority_chain import (
    AuthorityChain,
    AuthorityStage,
    ChainResult,
    StageResult,
    VetoSeverity,
    get_authority_chain,
    reset_authority_chain,
)
from .runtime_authority import (
    RuntimeAuthorityState,
    ALLOWED_RUNTIME_AUTHORITY_STATES,
    normalize_runtime_authority,
    build_runtime_authority_snapshot,
)
from .runtime_control_service import (
    build_g6_runtime_settings,
    build_aggressive_allocator_runtime_settings,
    build_fast_market_execution_decision,
    build_asset_runtime_authority_snapshot,
    build_runtime_authority_matrix,
)
from .execution_advisory_service import (
    build_execution_advisory_config_snapshot,
    evaluate_execution_advisory_guard,
)
from .profit_calibration_service import (
    build_profit_calibration_shadow,
    build_profit_calibration_live_adjustments,
)
from .runtime_export_service import (
    build_exit_alpha_review_summary,
    build_step15_asset_status,
    inject_asset_authority_states,
)

# =============================================================================
# LEGACY MODULES - LAZY IMPORT (may have complex dependencies)
# =============================================================================

# Lazy import flags
_RUNTIME_SPINE_AVAILABLE = False
_RISK_GOVERNOR_AVAILABLE = False
_FEEDBACK_LOOP_AVAILABLE = False

# Lazy import containers
RuntimeSpine = None
SpineConfig = None
TickResult = None
get_runtime_spine = None
reset_runtime_spine = None

RiskGovernor = None
GovernorDecision = None
VetoReason = None
GovernorConfig = None
get_risk_governor = None

FeedbackLoop = None
FeedbackEvent = None
FeedbackType = None
get_feedback_loop = None

# Try to import runtime_spine
try:
    from .runtime_spine import (
        RuntimeSpine,
        SpineConfig,
        TickResult,
        get_runtime_spine,
        reset_runtime_spine,
    )
    _RUNTIME_SPINE_AVAILABLE = True
except ImportError as e:
    _logger.debug(f"runtime_spine not available: {e}")

# Try to import risk_governor
try:
    from .risk_governor import (
        RiskGovernor,
        GovernorDecision,
        VetoReason,
        GovernorConfig,
        get_risk_governor,
    )
    _RISK_GOVERNOR_AVAILABLE = True
except ImportError as e:
    _logger.debug(f"risk_governor not available: {e}")

# Try to import feedback_loop
try:
    from .feedback_loop import (
        FeedbackLoop,
        FeedbackEvent,
        FeedbackType,
        get_feedback_loop,
    )
    _FEEDBACK_LOOP_AVAILABLE = True
except ImportError as e:
    _logger.debug(f"feedback_loop not available: {e}")


# =============================================================================
# __all__ - Only export what's actually available
# =============================================================================

__all__ = [
    # P0 FIX: Unit System (ALWAYS available)
    "UnitType",
    "TypedValue",
    "exposure_to_quantity",
    "quantity_to_exposure",
    "validate_exposure_fraction",
    "validate_base_quantity",
    # P0 FIX: Account Sync (ALWAYS available)
    "AccountSyncManager",
    "AccountState",
    "EquityStatus",
    "get_account_sync",
    "reset_account_sync",
    "validate_exchange_kraken_only",
    # P0 FIX: Exchange Guard (ALWAYS available)
    "ExchangeGuard",
    "validate_kraken_only",
    "assert_not_dex_execution",
    "detect_exchange_from_client",
    "detect_exchange_from_url",
    "get_exchange_guard",
    "reset_exchange_guard",
    "NonKrakenExchangeError",
    "DEXExecutionBlockedError",
    # P0 FIX: Authority Chain (ALWAYS available)
    "AuthorityChain",
    "AuthorityStage",
    "ChainResult",
    "StageResult",
    "VetoSeverity",
    "get_authority_chain",
    "reset_authority_chain",
    "RuntimeAuthorityState",
    "ALLOWED_RUNTIME_AUTHORITY_STATES",
    "normalize_runtime_authority",
    "build_runtime_authority_snapshot",
    "build_g6_runtime_settings",
    "build_aggressive_allocator_runtime_settings",
    "build_fast_market_execution_decision",
    "build_asset_runtime_authority_snapshot",
    "build_runtime_authority_matrix",
    "build_execution_advisory_config_snapshot",
    "evaluate_execution_advisory_guard",
    "build_profit_calibration_shadow",
    "build_profit_calibration_live_adjustments",
    "build_exit_alpha_review_summary",
    "build_step15_asset_status",
    "inject_asset_authority_states",
]

# Add legacy exports if available
if _RUNTIME_SPINE_AVAILABLE:
    __all__.extend([
        "RuntimeSpine",
        "SpineConfig",
        "TickResult",
        "get_runtime_spine",
        "reset_runtime_spine",
    ])

if _RISK_GOVERNOR_AVAILABLE:
    __all__.extend([
        "RiskGovernor",
        "GovernorDecision",
        "VetoReason",
        "GovernorConfig",
        "get_risk_governor",
    ])

if _FEEDBACK_LOOP_AVAILABLE:
    __all__.extend([
        "FeedbackLoop",
        "FeedbackEvent",
        "FeedbackType",
        "get_feedback_loop",
    ])


# =============================================================================
# MODULE STATUS CHECK
# =============================================================================

def get_module_status():
    """Get availability status of all core modules."""
    return {
        "version": __version__,
        # P0 modules (always True)
        "unit_system": True,
        "account_sync": True,
        "exchange_guard": True,
        "authority_chain": True,
        # Legacy modules (may be False)
        "runtime_spine": _RUNTIME_SPINE_AVAILABLE,
        "risk_governor": _RISK_GOVERNOR_AVAILABLE,
        "feedback_loop": _FEEDBACK_LOOP_AVAILABLE,
    }
