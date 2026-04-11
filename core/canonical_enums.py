"""
================================================================================
CANONICAL ENUMS - Single Source of Truth for Shared Enumerations
================================================================================
Version: 1.0.0
Purpose: Eliminate enum duplication across 450+ Python files.

MIGRATION GUIDE:
    Before:  from integration.integration_v36 import SystemMode, ExecutionMode
    After:   from core.canonical_enums import SystemMode, ExecutionMode

CRITICAL: All enum VALUES are preserved exactly as they were in integration_v36.py
(the canonical decision engine). Any module that defined different values for the
same enum name was a BUG, not a feature.

SPECIAL CASE - SystemMode in authority_fusion.py:
    authority_fusion.py historically used int values (0, 1, 2) instead of strings.
    The _FUSION_MODE_MAP in integration_v36.py bridged this.
    This module defines SystemMode with STRING values (the canonical form).
    authority_fusion.py is updated to use these canonical enums directly.

================================================================================
"""

from enum import Enum, auto


# [N5] str,Enum pattern: all string-valued enums inherit from (str, Enum)
# so that `SystemMode.NORMAL == "NORMAL"` is True without .value access.
# This eliminates the entire class of enum identity comparison bugs that
# HMATS has had across 8+ enum definitions (see MEMORY.md bug patterns).


# =============================================================================
# SYSTEM MODE (Mode Hierarchy: NO_TRADE > OPPORTUNITY > NORMAL)
# =============================================================================

class SystemMode(str, Enum):
    """System operating mode.

    Precedence: NO_TRADE > OPPORTUNITY > NORMAL (LOCKED).

    Canonical source: was integration/integration_v36.py.
    Duplicates existed in: authority_fusion.py (int values!),
    orchestration/system_mode.py, core/runtime_spine.py,
    risk/stop_loss_authority.py.
    """
    NO_TRADE = "NO_TRADE"
    OPPORTUNITY = "OPPORTUNITY"
    NORMAL = "NORMAL"


# =============================================================================
# EXECUTION MODE (Trade Intent level - how to execute)
# =============================================================================

class ExecutionMode(str, Enum):
    """Trade execution mode at the intent level.

    Canonical source: was integration/integration_v36.py.

    NOTE: Other modules define LOCAL execution mode enums for subsystem-specific
    purposes (e.g., DVOLExecutionMode in execution_guards.py, TimingExecutionMode
    in timing_engine.py). Those are DIFFERENT concepts and should be renamed,
    not merged here.
    """
    AGGRESSIVE = "AGGRESSIVE"
    AGGRESSIVE_MAKER = "AGGRESSIVE_MAKER"
    BALANCED = "BALANCED"
    PASSIVE_PREFERRED = "PASSIVE_PREFERRED"
    PASSIVE_ONLY = "PASSIVE_ONLY"


# =============================================================================
# DEADLOCK RESOLUTION
# =============================================================================

class DeadlockResolution(str, Enum):
    """Fusion deadlock resolution.

    Canonical source: was integration/integration_v36.py.
    Duplicates existed in: defense/production_reliability.py (had CONTINUE
    instead of NONE, and extra REDUCE_SIZE), signals/fusion_patience.py.
    """
    NONE = "NONE"
    WAIT_ONE_MORE = "WAIT_ONE_MORE"
    FORCE_AGGRESSIVE = "FORCE_AGGRESSIVE"
    ABORT_OPPORTUNITY = "ABORT_OPPORTUNITY"


# =============================================================================
# REGIME PHASE (Where within a move)
# =============================================================================

class RegimePhase(str, Enum):
    """Regime phase within a trend.

    NOT the same as regime_state (BULL/BEAR/SIDEWAYS).
    Describes WHERE we are within a move.

    Canonical source: was market/phase_detector.py.
    Duplicates existed in: core/runtime_spine.py, integration/integration_v36.py,
    execution/exit_alpha.py (with lowercase values!).
    """
    UNKNOWN = "UNKNOWN"
    IGNITION = "IGNITION"
    EXPANSION = "EXPANSION"
    SATURATION = "SATURATION"
    EXHAUSTION = "EXHAUSTION"


# =============================================================================
# DRL AUTHORITY LEVEL
# =============================================================================

class DRLAuthorityLevel(str, Enum):
    """DRL authority gating level.

    Promotion path: DISABLED -> SHADOW -> EXIT_ONLY -> ACTIVE
    Demotion: consecutive losses / drawdown -> EXIT_ONLY (auto-demote)

    Canonical source: was integration/integration_v36.py.
    Duplicate existed in: drl/authority_gate.py.
    """
    DISABLED = "DISABLED"
    SHADOW = "SHADOW"
    EXIT_ONLY = "EXIT_ONLY"
    ACTIVE = "ACTIVE"


# =============================================================================
# TRANCHE LEVEL
# =============================================================================

class TrancheLevel(Enum):
    """Position tranche levels (pyramiding).

    Canonical source: was defense/constitution.py.
    Duplicates existed in: risk/unified_position_sizer.py,
    risk/tranche_manager.py, risk/cascade_exhaustion_governor.py (with T1-T4
    names instead of TRANCHE_N).
    """
    NONE = 0
    TRANCHE_1 = 1
    TRANCHE_2 = 2
    TRANCHE_3 = 3
    TRANCHE_4 = 4


# =============================================================================
# RUN MODE (CLI mode selection)
# =============================================================================

class RunMode(str, Enum):
    """System run mode (CLI argument)."""
    VERIFY = "verify"
    PAPER = "paper"
    BACKTEST = "backtest"
    LIVE = "live"


# =============================================================================
# AUTHORITY TYPES (for agent roles in fusion)
# =============================================================================

class Authority(str, Enum):
    """Authority types for agent roles in the fusion engine.

    Canonical source: was signals/authority_fusion.py.
    """
    DECIDE = "DECIDE"
    CONFIRM = "CONFIRM"
    ADVISE = "ADVISE"
    VETO = "VETO"
    CAP = "CAP"
    TRIGGER = "TRIGGER"
    EXECUTE = "EXECUTE"
    ABSOLUTE = "ABSOLUTE"
    NONE = "NONE"
