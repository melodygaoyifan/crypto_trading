"""
================================================================================
HMATS v5.2.1 - Volatility Alpha 风险集成
Volatility Alpha Risk Integration
================================================================================

Purpose: 将 VolatilityAlphaAgent 与风险管理系统集成

核心功能:
1. VolAlpha intensity 作为上层 constraint
2. 当组合波动率 > target 时 VolAlpha 自动降权
3. 与 volatility_targeting.py 的桥接

================================================================================
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class VolAlphaRiskConfig:
    """VolAlpha 风险配置"""
    
    # 组合波动率目标
    portfolio_vol_target_daily: float = 0.02       # 2% 日波动率目标
    portfolio_vol_target_annual: float = 0.32      # ~32% 年化
    
    # VolAlpha 权重控制
    vol_alpha_base_weight: float = 1.0             # 基础权重
    vol_alpha_min_weight: float = 0.0              # 最低权重
    
    # 降权阈值
    vol_utilization_warning: float = 0.8           # 80% 时开始警告
    vol_utilization_critical: float = 1.0          # 100% 时强制降权
    vol_utilization_disable: float = 1.3           # 130% 时禁用
    
    # 降权速率
    downweight_rate_per_pct: float = 0.1           # 每超过 1% 降权 10%
    
    # 熊市特殊处理
    bear_market_vol_boost: float = 1.2             # 熊市允许更高波动
    
    # EventBus
    publish_to_eventbus: bool = True


# =============================================================================
# VOLATILITY ALPHA RISK CONTROLLER
# =============================================================================

class VolAlphaRiskController:
    """
    VolAlpha 风险控制器
    
    职责:
    1. 监控组合波动率
    2. 根据波动率利用率调整 VolAlpha 权重
    3. 与 volatility_targeting.py 集成
    """
    
    def __init__(self, config: VolAlphaRiskConfig = None):
        self.config = config or VolAlphaRiskConfig()
        
        # 当前状态
        self._current_portfolio_vol: float = 0.0
        self._vol_utilization: float = 0.0
        self._vol_alpha_weight: float = self.config.vol_alpha_base_weight
        self._last_status: str = "INIT"
        
        # 市场状态
        self._is_bear_market: bool = False
        
        # 统计
        self._stats = {
            "weight_updates": 0,
            "downweight_events": 0,
            "disable_events": 0,
            "bear_boost_events": 0,
        }
        
        # 回调
        self._event_callback: Optional[Callable] = None
        
        logger.info("VolAlphaRiskController initialized")
    
    def set_event_callback(self, callback: Callable):
        """设置 EventBus 回调"""
        self._event_callback = callback
    
    # =========================================================================
    # MAIN INTERFACE
    # =========================================================================
    
    def update_portfolio_vol(
        self,
        current_portfolio_vol: float,
        is_bear_market: bool = False,
    ) -> float:
        """
        更新组合波动率并计算 VolAlpha 权重
        
        Args:
            current_portfolio_vol: 当前组合日波动率
            is_bear_market: 是否熊市
        
        Returns:
            VolAlpha 权重 (0-1)
        """
        self._current_portfolio_vol = current_portfolio_vol
        self._is_bear_market = is_bear_market
        
        # 计算目标 (熊市可以容忍更高波动)
        effective_target = self.config.portfolio_vol_target_daily
        if is_bear_market:
            effective_target *= self.config.bear_market_vol_boost
        
        # 计算利用率
        self._vol_utilization = current_portfolio_vol / effective_target if effective_target > 0 else 0
        
        # 计算权重
        self._vol_alpha_weight = self._compute_weight(self._vol_utilization)
        
        self._stats["weight_updates"] += 1
        
        # 日志
        self._log_update()
        
        # 发布事件
        self._publish_event()
        
        return self._vol_alpha_weight
    
    def _compute_weight(self, utilization: float) -> float:
        """计算 VolAlpha 权重"""
        
        # 低于警告阈值: 全权重
        if utilization < self.config.vol_utilization_warning:
            return self.config.vol_alpha_base_weight
        
        # 超过禁用阈值: 零权重
        if utilization >= self.config.vol_utilization_disable:
            self._stats["disable_events"] += 1
            return self.config.vol_alpha_min_weight
        
        # 在警告和临界之间: 线性降权
        if utilization < self.config.vol_utilization_critical:
            # 从 warning 到 critical，线性降到 0.5
            range_pct = (utilization - self.config.vol_utilization_warning) / (
                self.config.vol_utilization_critical - self.config.vol_utilization_warning
            )
            weight = self.config.vol_alpha_base_weight * (1 - range_pct * 0.5)
            
            self._stats["downweight_events"] += 1
            return max(weight, 0.5)
        
        # 在 critical 和 disable 之间: 快速降权
        range_pct = (utilization - self.config.vol_utilization_critical) / (
            self.config.vol_utilization_disable - self.config.vol_utilization_critical
        )
        weight = 0.5 * (1 - range_pct)
        
        self._stats["downweight_events"] += 1
        return max(weight, self.config.vol_alpha_min_weight)
    
    def _log_update(self):
        """记录更新日志"""
        status = "NORMAL"
        if self._vol_utilization >= self.config.vol_utilization_disable:
            status = "DISABLED"
        elif self._vol_utilization > self.config.vol_utilization_critical:
            status = "CRITICAL"
        elif self._vol_utilization >= self.config.vol_utilization_warning:
            status = "WARNING"

        _msg = (
            f"VolAlphaRisk: portfolio_vol={self._current_portfolio_vol:.2%} "
            f"utilization={self._vol_utilization:.1%} "
            f"weight={self._vol_alpha_weight:.2f} status={status} "
            f"bear_market={self._is_bear_market}"
        )
        if status != self._last_status:
            logger.info(_msg)
            self._last_status = status
        elif self._stats["weight_updates"] % 20 == 0:
            logger.info(_msg)
        else:
            logger.debug(_msg)
    
    def _publish_event(self):
        """发布事件到 EventBus"""
        if not self.config.publish_to_eventbus or not self._event_callback:
            return
        
        try:
            self._event_callback("VOL_ALPHA_RISK_UPDATE", {
                "portfolio_vol": self._current_portfolio_vol,
                "vol_utilization": self._vol_utilization,
                "vol_alpha_weight": self._vol_alpha_weight,
                "is_bear_market": self._is_bear_market,
                "timestamp": datetime.now().isoformat(),
            })
        except Exception as e:
            logger.warning(f"Failed to publish risk event: {e}")
    
    # =========================================================================
    # INTEGRATION WITH VOLATILITY TARGETING
    # =========================================================================
    
    def connect_to_volatility_targeting(self):
        """连接到 volatility_targeting.py"""
        try:
            from risk.volatility_targeting import (
                get_volatility_position_sizer,
                PortfolioRiskState,
            )
            
            sizer = get_volatility_position_sizer()
            
            # 注册回调
            logger.info("VolAlphaRisk connected to volatility_targeting")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to connect to volatility_targeting: {e}")
            return False
    
    def get_constraint_for_intent(self, vol_intent_intensity: float) -> float:
        """
        获取对 VolAlpha intent 的约束
        
        Args:
            vol_intent_intensity: VolAlpha 原始 intensity
        
        Returns:
            约束后的 intensity
        """
        return vol_intent_intensity * self._vol_alpha_weight
    
    # =========================================================================
    # STATUS
    # =========================================================================
    
    def get_status(self) -> Dict[str, Any]:
        """获取状态"""
        return {
            "portfolio_vol": self._current_portfolio_vol,
            "vol_utilization": self._vol_utilization,
            "vol_alpha_weight": self._vol_alpha_weight,
            "is_bear_market": self._is_bear_market,
            "config": {
                "target_daily": self.config.portfolio_vol_target_daily,
                "warning": self.config.vol_utilization_warning,
                "critical": self.config.vol_utilization_critical,
                "disable": self.config.vol_utilization_disable,
            },
            "stats": self._stats,
        }
    
    @property
    def is_disabled(self) -> bool:
        """VolAlpha 是否被禁用"""
        return self._vol_alpha_weight < 0.1
    
    @property
    def current_weight(self) -> float:
        """当前权重"""
        return self._vol_alpha_weight


# =============================================================================
# VOLATILITY TARGETING BRIDGE
# =============================================================================

class VolatilityTargetingBridge:
    """
    volatility_targeting.py 桥接
    
    提供与现有波动率目标系统的集成
    """
    
    def __init__(self):
        self._volatility_sizer = None
        self._last_portfolio_state = None
    
    def initialize(self) -> bool:
        """初始化桥接"""
        try:
            from risk.volatility_targeting import get_volatility_position_sizer
            
            self._volatility_sizer = get_volatility_position_sizer()
            logger.info("VolatilityTargetingBridge initialized")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to initialize VolatilityTargetingBridge: {e}")
            return False
    
    def get_portfolio_risk_state(self) -> Optional[Dict[str, Any]]:
        """获取组合风险状态"""
        if not self._volatility_sizer:
            return None
        
        try:
            state = self._volatility_sizer.get_portfolio_risk_state()
            self._last_portfolio_state = state
            
            return {
                "target_daily_vol": state.target_daily_vol,
                "current_daily_vol": state.current_daily_vol,
                "vol_utilization": state.vol_utilization,
                "total_exposure_usd": state.total_exposure_usd,
                "is_over_risk": state.is_over_risk(),
            }
            
        except Exception as e:
            logger.warning(f"Failed to get portfolio risk state: {e}")
            return None
    
    def get_current_portfolio_vol(self) -> float:
        """获取当前组合波动率"""
        state = self.get_portfolio_risk_state()
        if state:
            return state.get("current_daily_vol", 0.0)
        return 0.0


# =============================================================================
# COMPLETE RISK INTEGRATOR
# =============================================================================

class VolAlphaRiskIntegrator:
    """
    VolAlpha 风险完整集成器
    
    组合:
    - VolAlphaRiskController
    - VolatilityTargetingBridge
    - EventBus 集成
    """
    
    def __init__(self, config: VolAlphaRiskConfig = None):
        self.config = config or VolAlphaRiskConfig()
        
        self._risk_controller = VolAlphaRiskController(self.config)
        self._vol_bridge = VolatilityTargetingBridge()
        
        # VolAlpha agent 引用
        self._vol_alpha_agent = None
        
        # 初始化状态
        self._initialized = False
    
    def initialize(self) -> bool:
        """初始化"""
        # 连接 volatility_targeting
        bridge_ok = self._vol_bridge.initialize()
        
        # 连接 VolAlpha agent
        try:
            from agents.volatility_alpha_agent import get_volatility_alpha_agent
            self._vol_alpha_agent = get_volatility_alpha_agent()
        except Exception as e:
            logger.warning(f"Could not get VolAlpha agent: {e}")
        
        self._initialized = bridge_ok
        logger.info(f"VolAlphaRiskIntegrator initialized: {self._initialized}")
        return self._initialized
    
    def update(self, is_bear_market: bool = False) -> Dict[str, Any]:
        """
        执行更新周期
        
        1. 从 volatility_targeting 获取组合波动率
        2. 更新 VolAlpha 权重
        3. 应用约束到 VolAlpha agent
        """
        result = {
            "updated": False,
            "portfolio_vol": 0.0,
            "vol_alpha_weight": 1.0,
        }
        
        # 获取组合波动率
        portfolio_vol = self._vol_bridge.get_current_portfolio_vol()
        result["portfolio_vol"] = portfolio_vol
        
        # 更新风险控制器
        weight = self._risk_controller.update_portfolio_vol(
            portfolio_vol, is_bear_market
        )
        result["vol_alpha_weight"] = weight
        result["updated"] = True
        
        return result
    
    def apply_constraint_to_intent(self, vol_intent) -> Any:
        """
        应用风险约束到 VolAlpha intent
        
        Args:
            vol_intent: VolatilityIntent 对象
        
        Returns:
            约束后的 VolatilityIntent
        """
        from agents.volatility_alpha_agent import VolatilityIntent, VolBias
        
        if not isinstance(vol_intent, VolatilityIntent):
            return vol_intent
        
        constrained_intensity = self._risk_controller.get_constraint_for_intent(
            vol_intent.intensity
        )
        
        # 如果约束后 intensity 很低，改为 neutral
        if constrained_intensity < 0.15:
            return VolatilityIntent(
                asset=vol_intent.asset,
                vol_bias=VolBias.NEUTRAL,
                intensity=0.0,
                expected_horizon=vol_intent.expected_horizon,
                rationale_tags=vol_intent.rationale_tags + ["risk_constrained"],
                data_quality=vol_intent.data_quality,
                is_valid=True,
            )
        
        return VolatilityIntent(
            asset=vol_intent.asset,
            vol_bias=vol_intent.vol_bias,
            intensity=constrained_intensity,
            expected_horizon=vol_intent.expected_horizon,
            rationale_tags=vol_intent.rationale_tags,
            sub_signals=vol_intent.sub_signals,
            data_quality=vol_intent.data_quality,
            is_valid=vol_intent.is_valid,
        )
    
    def get_status(self) -> Dict[str, Any]:
        """获取状态"""
        return {
            "initialized": self._initialized,
            "risk_controller": self._risk_controller.get_status(),
            "bridge_connected": self._vol_bridge._volatility_sizer is not None,
        }


# =============================================================================
# SINGLETON
# =============================================================================

_vol_alpha_risk_integrator: Optional[VolAlphaRiskIntegrator] = None


def get_vol_alpha_risk_integrator(
    config: VolAlphaRiskConfig = None
) -> VolAlphaRiskIntegrator:
    """获取 VolAlpha 风险集成器单例"""
    global _vol_alpha_risk_integrator
    if _vol_alpha_risk_integrator is None:
        _vol_alpha_risk_integrator = VolAlphaRiskIntegrator(config)
    elif config is not None and _vol_alpha_risk_integrator.config != config:
        logger.info("[VOL_ALPHA_RISK] Config changed; refreshing singleton instance")
        _vol_alpha_risk_integrator = VolAlphaRiskIntegrator(config)
    return _vol_alpha_risk_integrator


def reset_vol_alpha_risk_integrator():
    """重置单例"""
    global _vol_alpha_risk_integrator
    _vol_alpha_risk_integrator = None


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'VolAlphaRiskConfig',
    'VolAlphaRiskController',
    'VolatilityTargetingBridge',
    'VolAlphaRiskIntegrator',
    'get_vol_alpha_risk_integrator',
    'reset_vol_alpha_risk_integrator',
]


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Testing VolAlpha Risk Integration")
    print("=" * 60)
    
    # 创建风险控制器
    config = VolAlphaRiskConfig()
    controller = VolAlphaRiskController(config)
    
    # 测试不同组合波动率水平
    test_cases = [
        (0.01, False, "Low vol"),
        (0.016, False, "Near warning"),
        (0.02, False, "At target"),
        (0.022, False, "Above target"),
        (0.026, False, "Critical"),
        (0.03, False, "Near disable"),
        (0.02, True, "Bear market (relaxed)"),
    ]
    
    print("\n1. Weight computation tests:")
    
    for vol, bear, desc in test_cases:
        weight = controller.update_portfolio_vol(vol, bear)
        print(f"   {desc}: vol={vol:.1%} bear={bear} -> weight={weight:.2f}")
    
    # 测试约束应用
    print("\n2. Constraint application:")
    
    from agents.volatility_alpha_agent import VolatilityIntent, VolBias, VolHorizon
    
    # 创建高 intensity intent
    intent = VolatilityIntent(
        asset="BTC",
        vol_bias=VolBias.INCREASE,
        intensity=0.8,
        expected_horizon=VolHorizon.SHORT,
        rationale_tags=["test"],
    )
    
    # 高波动环境
    controller.update_portfolio_vol(0.025, False)  # 超过 critical
    
    constrained = controller.get_constraint_for_intent(intent.intensity)
    print(f"   Original intensity: {intent.intensity:.2f}")
    print(f"   Controller weight: {controller.current_weight:.2f}")
    print(f"   Constrained intensity: {constrained:.2f}")
    
    # 测试状态
    print("\n3. Status:")
    status = controller.get_status()
    print(f"   portfolio_vol: {status['portfolio_vol']:.2%}")
    print(f"   vol_utilization: {status['vol_utilization']:.1%}")
    print(f"   vol_alpha_weight: {status['vol_alpha_weight']:.2f}")
    print(f"   is_disabled: {controller.is_disabled}")
    print(f"   stats: {status['stats']}")
    
    print("\n✓ VolAlpha Risk Integration test complete")
