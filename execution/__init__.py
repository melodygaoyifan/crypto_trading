"""
HMATS Execution Module - 智能执行算法
=====================================

SOTA对照: "智能执行算法模块如TWAP、VWAP、Implementation Shortfall"

核心组件:
- TimingEngine: 执行时机引擎 (资产特定配置)
- PassiveAggressive: 被动/激进执行模式切换
- SOLExecution: SOL专用执行逻辑 (DEX特殊处理)
- ExecutionManager: 执行管理器
"""

import logging

logger = logging.getLogger(__name__)

from .timing_engine import (
    ExecutionMode,
    AssetExecutionProfile,
    ExecutionTimingScore,
    ExecutionTimingEngine,
    EXECUTION_PROFILES,
)

# Alias for backward compatibility
TimingEngine = ExecutionTimingEngine
ExecutionTimingCalculator = ExecutionTimingEngine

# Try to import ExecutionManager
try:
    from .execution_manager import ExecutionManager
except ImportError:
    ExecutionManager = None

__all__ = [
    "ExecutionMode",
    "ExecutionTimingEngine",
    "TimingEngine",
    "ExecutionTimingCalculator",
    "EXECUTION_PROFILES",
    "ExecutionManager",
    "AssetExecutionProfile",
    "ExecutionTimingScore",
]

# Execution Loop Controller
try:
    from .loop_controller import (
        ExecutionLoopController,
        ExecutionLoopConfig,
        ExecutionModifier,
        LoopTickResult,
        get_execution_loop,
        get_execution_loop_controller,
        reset_execution_loop,
    )
    __all__.extend([
        "ExecutionLoopController",
        "ExecutionLoopConfig",
        "ExecutionModifier",
        "LoopTickResult",
        "get_execution_loop",
        "get_execution_loop_controller",
        "reset_execution_loop",
    ])
except ImportError as e:
    logger.warning("Execution loop controller unavailable: %s", e)

# SOTA Scheduler
try:
    from .sota_scheduler import (
        SOTAExecutionScheduler,
        SchedulerConfig,
        ExecutionPlan,
    )
    __all__.extend(["SOTAExecutionScheduler", "SchedulerConfig", "ExecutionPlan"])
except ImportError:
    logger.warning("SOTA scheduler unavailable")

# Production Market Impact (actual implementation)
try:
    from .production_market_impact import (
        AdaptiveMarketImpactModel,
        UrgencyLevel,
        SlicingPlan,
    )
    __all__.extend(["AdaptiveMarketImpactModel", "UrgencyLevel", "SlicingPlan"])
except ImportError:
    logger.warning("Production market impact model unavailable")
