"""
================================================================================
HMATS v6.1.4 - Shadow Module Package
================================================================================
Purpose: Shadow Mode 观察模块，只允许计算/记录/展示，无决策权

Shadow Observers:
    1. LeadLagShadowObserver - L0 观察级
    2. PassiveAggressiveShadowObserver - L1 执行观察级
    3. SolDexShadowObserver - L1 特征源

核心约束:
    OFF 不能触发 entry
    OFF 不能 veto 任何交易
    OFF 不能改变仓位、方向、杠杆
    ON 只允许：计算 -> 记录 -> 展示

================================================================================
"""

from .shadow_observers import (
    # Event types
    ShadowEventType,
    ShadowObservation,
    
    # Individual observers
    LeadLagShadowObserver,
    PassiveAggressiveShadowObserver,
    SolDexShadowObserver,
    
    # Manager
    ShadowObserverManager,
    get_shadow_observer_manager,
    reset_shadow_observer_manager,
    
    # Convenience functions
    observe_shadows,
    analyze_execution_shadow,
)

__all__ = [
    "ShadowEventType",
    "ShadowObservation",
    "LeadLagShadowObserver",
    "PassiveAggressiveShadowObserver",
    "SolDexShadowObserver",
    "ShadowObserverManager",
    "get_shadow_observer_manager",
    "reset_shadow_observer_manager",
    "observe_shadows",
    "analyze_execution_shadow",
]

__version__ = "6.1.4"
