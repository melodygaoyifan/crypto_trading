"""
================================================================================
HMATS v5.4.0 - CANONICAL IMPORTS
================================================================================

This module provides THE SINGLE IMPORT POINT for all production runtime code.

IMPORT RULES (v5.4.0 Clarification):
-------------------------------------
OFF FORBIDDEN: Direct import from legacy/ in runtime paths
   - legacy/ contains deprecated/archived code
   - Will trigger NO_TRADE mode if imported during production runtime

ON ALLOWED: integration/ imports through canonical entrypoint (main.py)
   - integration/integration_v36.py is the CANONICAL decision engine
   - Contains irreplaceable: decide(), TradeIntentV36, SystemMode
   - Must only be imported via main.py or canonical mapping

ON ALLOWED: Import from this module (core/canonical_imports) for runtime components

Usage:
    from core.canonical_imports import (
        AuthorityFusionEngine,
        RiskGovernor,
        TradeGate,
        ExecutionManager,
        RuntimeSpine,
    )

Engineering Rules:
1. All production code should import from here OR through main.py canonical path
2. Tests can import directly from source modules
3. legacy/ imports are FORBIDDEN in runtime paths (triggers NO_TRADE)
4. integration/ imports are allowed ONLY through canonical entrypoint (main.py)

================================================================================
"""

import logging
import sys
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

# =============================================================================
# RUNTIME PROTECTION FLAG
# =============================================================================

import os

# =============================================================================
# RUNTIME MODE PROTECTION (v5.4.0 - Enhanced with HMATS_RUNTIME_MODE)
# =============================================================================

# Internal flag set by main.py during production startup
_RUNTIME_MODE_ACTIVE = False

# Environment variable for runtime mode (DEV/PROD)
# - DEV: Development mode, allows legacy imports (default)
# - PROD: Production mode, blocks legacy imports, enforces canonical weights
HMATS_RUNTIME_MODE = os.getenv("HMATS_RUNTIME_MODE", "DEV")


def activate_runtime_mode():
    """
    Activate production runtime mode protection.
    Called by main.py at startup.
    
    Effects:
    - Sets HMATS_RUNTIME_MODE=PROD
    - Blocks legacy/ imports
    - Enforces canonical weight system (signals.adaptive_weight_v521)
    """
    global _RUNTIME_MODE_ACTIVE, HMATS_RUNTIME_MODE
    _RUNTIME_MODE_ACTIVE = True
    HMATS_RUNTIME_MODE = "PROD"
    os.environ["HMATS_RUNTIME_MODE"] = "PROD"
    logger.info("[CANONICAL] Runtime mode ACTIVATED - HMATS_RUNTIME_MODE=PROD")
    logger.info("[CANONICAL] Legacy imports will trigger protection")
    logger.info("[CANONICAL] Canonical weight system: signals.adaptive_weight_v521")


def is_runtime_mode():
    """Check if runtime mode is active (PROD mode)."""
    return _RUNTIME_MODE_ACTIVE or os.getenv("HMATS_RUNTIME_MODE") == "PROD"


def get_runtime_mode() -> str:
    """Get current runtime mode (DEV or PROD)."""
    return os.getenv("HMATS_RUNTIME_MODE", "DEV")


def check_legacy_import_violation(module_name: str):
    """
    Check if a legacy import is being attempted in runtime mode.
    
    NOTE: This only applies to legacy/ imports, NOT integration/.
    integration/ is the canonical decision engine and is allowed through main.py.
    
    If runtime mode is active and legacy is imported:
    1. Log a clear error
    2. Set a global flag that triggers NO_TRADE
    
    This is called from legacy/__init__.py
    """
    global _LEGACY_IMPORT_VIOLATION
    
    if _RUNTIME_MODE_ACTIVE:
        logger.error(
            f"[CANONICAL] LEGACY IMPORT VIOLATION: {module_name} was imported in runtime mode! "
            f"This is forbidden. System will enter NO_TRADE mode for safety."
        )
        _LEGACY_IMPORT_VIOLATION = True
        return True
    return False


# Global flag for legacy import violation (checked by risk governor)
_LEGACY_IMPORT_VIOLATION = False


def has_legacy_import_violation() -> bool:
    """Check if a legacy import violation occurred."""
    return _LEGACY_IMPORT_VIOLATION


# =============================================================================
# CANONICAL EXPORTS - AUTHORITY & FUSION (includes SystemMode)
# =============================================================================

from signals.authority_fusion import (
    AuthorityFusionEngine,
    Authority,
    FusionResult,
    FusionContext,
    AgentSignal,
    SystemMode,  # System mode enum
    AUTHORITY_MATRIX_NORMAL,
    AUTHORITY_MATRIX_OPPORTUNITY,
    AUTHORITY_MATRIX_NO_TRADE,
)

# =============================================================================
# CANONICAL EXPORTS - RISK GOVERNANCE
# =============================================================================

from core.risk_governor import (
    RiskGovernor,
    GovernorConfig,
    GovernorDecision,
    VetoType,
    VetoReason,
)

# =============================================================================
# CANONICAL EXPORTS - TRADE GATE
# =============================================================================

from defense.trade_gate import (
    TradeGate,
    TradeGateConfig,
    GateDecision,
    RejectReason,  # Note: GateReason was renamed to RejectReason
)

# =============================================================================
# CANONICAL EXPORTS - EXECUTION
# =============================================================================

from execution.execution_manager import (
    ExecutionManager,
    ExecutionConfig,
    OrderResult,
    OrderType,
)

from execution.learned_execution_policy import (
    LearnedExecutionPolicy,
    LearnedExecutionConfig,
)

# =============================================================================
# CANONICAL EXPORTS - RUNTIME SPINE
# =============================================================================

from core.runtime_spine import (
    RuntimeSpine,
    SpineConfig,
    TickResult,
    TickStage,
    ExecutionAcceptance,
    FusedSignal,
    ExecutionIntent,
    ExecutionResult,
    MarketSnapshot,
)

# =============================================================================
# CANONICAL EXPORTS - REGIME
# =============================================================================

from market.regime_detector import (
    EnhancedRegimeNavigator as RegimePhaseDetector,  # Alias for canonical naming
    EnhancedRegimeState as RegimeState,
    MarketRegimeState as RegimePhase,
    get_regime_navigator,
)

# =============================================================================
# CANONICAL EXPORTS - SYSTEM MODE (from authority_fusion above)
# =============================================================================

# SystemMode is re-exported from authority_fusion above

# =============================================================================
# CANONICAL EXPORTS - DATA HEALTH
# =============================================================================

from data_mgmt.data_health import (
    DataHealthIntegration,
    HealthStatus,
    DegradationLevel,
)

# =============================================================================
# CANONICAL EXPORTS - SENTIMENT
# =============================================================================

from engine.compute.vllm_inference_wrapper import (
    get_sentiment_config,
    is_sentiment_mock,
    sentiment_can_trigger_opportunity,
    SentimentConfig,
)

# =============================================================================
# ALL EXPORTS
# =============================================================================

__all__ = [
    # Runtime protection (v5.4.0)
    'activate_runtime_mode',
    'is_runtime_mode',
    'get_runtime_mode',
    'HMATS_RUNTIME_MODE',
    'has_legacy_import_violation',
    
    # Authority & Fusion
    'AuthorityFusionEngine',
    'Authority',
    'FusionResult',
    'FusionContext',
    'AgentSignal',
    'AUTHORITY_MATRIX_NORMAL',
    'AUTHORITY_MATRIX_OPPORTUNITY',
    'AUTHORITY_MATRIX_NO_TRADE',
    
    # Risk Governance
    'RiskGovernor',
    'GovernorConfig',
    'GovernorDecision',
    'VetoType',
    'VetoReason',
    
    # Trade Gate
    'TradeGate',
    'TradeGateConfig',
    'GateDecision',
    'RejectReason',
    
    # Execution
    'ExecutionManager',
    'ExecutionConfig',
    'OrderResult',
    'OrderType',
    'LearnedExecutionPolicy',
    'LearnedExecutionConfig',
    
    # Runtime Spine
    'RuntimeSpine',
    'SpineConfig',
    'TickResult',
    'TickStage',
    'ExecutionAcceptance',
    'FusedSignal',
    'ExecutionIntent',
    'ExecutionResult',
    'MarketSnapshot',
    
    # Regime
    'RegimePhaseDetector',
    'RegimeState',
    'RegimePhase',
    'get_regime_navigator',
    
    # System Mode
    'SystemMode',
    
    # Data Health
    'DataHealthIntegration',
    'HealthStatus',
    'DegradationLevel',
    
    # Sentiment
    'get_sentiment_config',
    'is_sentiment_mock',
    'sentiment_can_trigger_opportunity',
    'SentimentConfig',
]


# =============================================================================
# STARTUP LOGGING
# =============================================================================

logger.info(
    "[CANONICAL] canonical_imports loaded - "
    f"exports: {len(__all__)} symbols"
)
