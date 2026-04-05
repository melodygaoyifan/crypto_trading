"""
================================================================================
HMATS v3.6 - Runnable Canonical Package
================================================================================
Version: 3.6.0
Purpose: Fused v3.3-HR + v3.4 + v3.5 canonical integration

PHILOSOPHY (v3.3-HR):
    "Missing a high-conviction move is worse than taking a controlled loss"

ARCHITECTURE (LOCKED):
    Information -> Regime -> Quant -> DRL -> Fusion -> Execution -> Risk

MODE HIERARCHY (LOCKED):
    NO_TRADE > OPPORTUNITY > NORMAL

DRL LIFECYCLE (v6.7):
    DISABLED -> SHADOW -> EXIT_ONLY -> ACTIVE (with auto-demotion)
================================================================================
"""

VERSION = "3.6.0"
VERSION_NAME = "HMATS v3.6 (Runnable Canonical)"

from .integration_v36 import (
    HMATSv36Engine,
    TradeIntentV36,
    get_v36_engine,
    reset_v36_engine,
    SystemMode,
    RegimePhase,
    ExecutionMode,
    DRLAuthorityLevel,
    DeadlockResolution,
    V33HR_PHILOSOPHY,
    V33HR_SOL_DOMINANCE,
)

__version__ = VERSION
__all__ = [
    "VERSION",
    "VERSION_NAME",
    "HMATSv36Engine",
    "TradeIntentV36",
    "get_v36_engine",
    "reset_v36_engine",
    "SystemMode",
    "RegimePhase",
    "ExecutionMode",
    "DRLAuthorityLevel",
    "DeadlockResolution",
    "V33HR_PHILOSOPHY",
    "V33HR_SOL_DOMINANCE",
]

# P1: Event Integration
try:
    from .event_integration import (
        V36EventIntegration,
        get_v36_event_integration,
        MarketDataEvent,
        RegimeUpdateEvent,
        SignalGeneratedEvent,
        OrderSubmittedEvent,
        RiskVetoEvent,
        TickEventLog,
    )
except ImportError:
    pass
