"""
HMATS Signals Module - 信号融合系统
===================================

SOTA对照: "信号融合优化，多源信号整合决策"
"""

from .fusion_patience import (
    DeadlockResolution,
    DeadlockState,
    FusionPatienceManager,
    get_patience_manager,
)

__all__ = [
    "DeadlockResolution",
    "FusionPatienceManager",
    "get_patience_manager",
]

# Authority fusion exports
from signals.authority_fusion import (
    Authority,
    AgentSignal,
    FusionContext,
    FusionResult,
    AuthorityFusionEngine,
    get_fusion_engine,
)

# Flow signal integrator (大资金流向监控)
try:
    from signals.flow_signal_integrator import (
        FlowSignalIntegrator,
        FlowSignalType,
        FlowState,
        FlowIntegratorConfig,
        get_flow_integrator,
    )
    __all__.extend([
        "FlowSignalIntegrator",
        "FlowSignalType",
        "FlowState",
        "FlowIntegratorConfig",
        "get_flow_integrator",
    ])
except ImportError:
    pass

# Alpha signal integrator (高优先级 Alpha 信号)
try:
    from signals.alpha_signal_integrator import (
        AlphaSignalIntegrator,
        AlphaSignalSource,
        AlphaState,
        AlphaIntegratorConfig,
        get_alpha_integrator,
    )
    __all__.extend([
        "AlphaSignalIntegrator",
        "AlphaSignalSource",
        "AlphaState",
        "AlphaIntegratorConfig",
        "get_alpha_integrator",
    ])
except ImportError:
    pass
