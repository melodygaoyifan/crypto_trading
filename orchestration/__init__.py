"""
HMATS Orchestration Module - 系统编排层
========================================

核心组件:
- SystemMode: 系统模式管理
- SOTAIntegration: SOTA v5.2.0 集成
- StrategicCoordinator: 策略协调器
"""

from orchestration.system_mode import SystemMode, ModeState

__all__ = [
    "SystemMode",
    "ModeState",
]

# SOTA v5.2.0 Integration Layer
try:
    from .sota_integration import (
        SOTAIntegration,
        SOTAIntegrationConfig,
        get_sota_integration,
        reset_sota_integration,
    )
    __all__.extend([
        "SOTAIntegration",
        "SOTAIntegrationConfig",
        "get_sota_integration",
        "reset_sota_integration",
    ])
except ImportError:
    pass

# Strategic Coordinator
try:
    from .strategic_coordinator import StrategicCoordinator, get_strategic_coordinator
    __all__.extend(["StrategicCoordinator", "get_strategic_coordinator"])
except ImportError:
    pass