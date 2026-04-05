"""
================================================================================
HMATS v5.2.1 - Volatility Alpha 与 Authority Fusion 的集成
Authority Fusion Volatility Alpha Integration
================================================================================

Purpose: 将 VolatilityAlphaAgent 集成到 Authority Fusion 系统

核心规则:
1. VolatilityAlphaAgent 是独立 authority source
2. 其输出不能单独触发 entry
3. 只能:
   - 放大/压制已有 direction intent
   - 影响 execution aggressiveness
4. 可被以下 veto:
   - RiskAgent
   - TradeGate
   - DataHealth

与 ModelAlpha 的关系:
- ModelAlpha = short + VolAlpha = increase -> 允许更激进执行
- ModelAlpha = long  + VolAlpha = increase -> 强制压制或拒绝 (熊市保护)

================================================================================
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple, Any
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# IMPORTS
# =============================================================================

from signals.authority_fusion import (
    Authority,
    SystemMode,
    AgentSignal,
    FusionContext,
    FusionResult,
    AuthorityFusionEngine,
    get_fusion_engine,
    AUTHORITY_MATRIX_NORMAL,
    AUTHORITY_MATRIX_OPPORTUNITY,
    AUTHORITY_MATRIX_NO_TRADE,
)

from agents.volatility_alpha_agent import (
    VolatilityAlphaAgent,
    VolatilityIntent,
    VolBias,
    get_volatility_alpha_agent,
)


# =============================================================================
# VOLATILITY ALPHA AUTHORITY DEFINITIONS
# =============================================================================

"""
VolatilityAlpha 权限定义:

NORMAL 模式:
    - ADVISE: 只能建议，不能决定
    - 与 ModelAlpha 类似的低权限

OPPORTUNITY 模式:
    - TRIGGER: 可以放大但不能减少 (与 Sentiment 类似)
    - 用于增强熊市中的激进执行

NO_TRADE 模式:
    - NONE: 完全禁用
"""

# 扩展的 Authority Matrix
VOLATILITY_ALPHA_AUTHORITY = {
    SystemMode.NORMAL: Authority.ADVISE,
    SystemMode.OPPORTUNITY: Authority.TRIGGER,
    SystemMode.NO_TRADE: Authority.NONE,
}


# =============================================================================
# VOLATILITY ALPHA SIGNAL ADAPTER
# =============================================================================

@dataclass
class VolAlphaSignal(AgentSignal):
    """
    Volatility Alpha 信号适配器
    
    将 VolatilityIntent 转换为 AgentSignal 格式
    """
    vol_bias: str = "neutral"           # increase/neutral/suppress
    vol_intensity: float = 0.0          # 0-1
    rationale_tags: List[str] = field(default_factory=list)
    
    @classmethod
    def from_intent(cls, intent: VolatilityIntent) -> 'VolAlphaSignal':
        """从 VolatilityIntent 创建 VolAlphaSignal"""
        
        # 不输出方向！这里的 direction 仅用于 TRIGGER 放大
        # VolAlpha 永远不产生实际的方向
        direction = 0.0  # 方向无关
        
        # 置信度基于 intensity
        confidence = intent.intensity
        
        # Veto 逻辑: suppress 时 veto long (熊市保护)
        veto_active = False
        veto_direction = None
        
        if intent.vol_bias == VolBias.SUPPRESS:
            # 低波动环境不应开新仓
            veto_active = True
            veto_direction = "both"  # 特殊：两个方向都谨慎
        
        return cls(
            direction=direction,
            confidence=confidence,
            veto_active=veto_active,
            veto_direction=veto_direction,
            leverage_cap=1.0,
            exit_pressure=0.0,
            tranche_advice=None,
            vol_bias=intent.vol_bias.value,
            vol_intensity=intent.intensity,
            rationale_tags=intent.rationale_tags,
        )


# =============================================================================
# VOLATILITY ALPHA FUSION RULES
# =============================================================================

class VolAlphaFusionRules:
    """
    Volatility Alpha 融合规则
    
    核心规则:
    1. VolAlpha 不能单独触发 entry
    2. 只能影响已有 direction intent 的执行强度
    3. 熊市保护: ModelAlpha=long + VolAlpha=increase -> 压制
    """
    
    @staticmethod
    def apply_vol_alpha(
        base_result: FusionResult,
        vol_signal: VolAlphaSignal,
        model_alpha_signal: Optional[AgentSignal],
        context: FusionContext,
    ) -> FusionResult:
        """
        应用 Volatility Alpha 到基础融合结果
        
        Args:
            base_result: 原始 Authority Fusion 输出
            vol_signal: VolatilityAlpha 信号
            model_alpha_signal: ModelAlpha 信号 (如果有)
            context: 融合上下文
        
        Returns:
            修改后的 FusionResult
        """
        # 获取 VolAlpha 权限
        vol_authority = VOLATILITY_ALPHA_AUTHORITY.get(context.mode, Authority.NONE)
        
        # NO_TRADE 模式：VolAlpha 禁用
        if vol_authority == Authority.NONE:
            logger.debug("VolAlpha: NONE authority in NO_TRADE mode")
            return base_result
        
        # 记录应用
        vol_applied_info = {
            "vol_bias": vol_signal.vol_bias,
            "vol_intensity": vol_signal.vol_intensity,
            "vol_authority": vol_authority.value,
        }
        
        # =====================================================================
        # RULE 1: 熊市保护 (ModelAlpha=long + VolAlpha=increase -> 压制)
        # =====================================================================
        
        if vol_signal.vol_bias == "increase" and vol_signal.vol_intensity > 0.5:
            if base_result.direction > 0:  # 想做多
                # 熊市高波动环境 + 想做多 = 危险！
                logger.warning(
                    "VolAlpha BEAR PROTECTION: suppressing LONG position "
                    f"(vol_intensity={vol_signal.vol_intensity:.2f})"
                )
                
                # 压制 exposure
                suppression_factor = 1 - (vol_signal.vol_intensity * 0.5)
                base_result.target_exposure *= suppression_factor
                
                # 增加保守性
                base_result.execution_mode = "PASSIVE_ONLY"
                base_result.delay_allowed = True
                base_result.urgency = max(base_result.urgency - 0.3, 0.1)
                
                vol_applied_info["action"] = "BEAR_PROTECTION_SUPPRESS"
                vol_applied_info["suppression_factor"] = suppression_factor
        
        # =====================================================================
        # RULE 2: 激进执行 (ModelAlpha=short + VolAlpha=increase -> 放大)
        # =====================================================================
        
        elif vol_signal.vol_bias == "increase" and vol_signal.vol_intensity > 0.5:
            if base_result.direction < 0:  # 想做空
                # 熊市高波动环境 + 想做空 = 正确方向！
                logger.info(
                    "VolAlpha AMPLIFY: supporting SHORT position "
                    f"(vol_intensity={vol_signal.vol_intensity:.2f})"
                )
                
                # 只在 OPPORTUNITY 模式放大
                if vol_authority == Authority.TRIGGER:
                    amplification_factor = 1 + (vol_signal.vol_intensity * 0.2)
                    base_result.target_exposure = min(
                        base_result.target_exposure * amplification_factor,
                        1.0
                    )
                    
                    # 更激进执行
                    base_result.execution_mode = "AGGRESSIVE_TAKER"
                    base_result.urgency = min(base_result.urgency + 0.2, 1.0)
                    base_result.allow_escalation = True
                    
                    vol_applied_info["action"] = "AMPLIFY_SHORT"
                    vol_applied_info["amplification_factor"] = amplification_factor
        
        # =====================================================================
        # RULE 3: 压制信号 (VolAlpha=suppress -> 减少执行激进性)
        # =====================================================================
        
        elif vol_signal.vol_bias == "suppress":
            logger.info(
                "VolAlpha SUPPRESS: reducing execution aggressiveness "
                f"(vol_intensity={vol_signal.vol_intensity:.2f})"
            )
            
            # 压制两个方向
            base_result.execution_mode = "PASSIVE_PREFERRED"
            base_result.urgency *= 0.7
            base_result.delay_allowed = True
            
            vol_applied_info["action"] = "SUPPRESS_EXECUTION"
        
        # =====================================================================
        # RULE 4: 中性信号 (VolAlpha=neutral -> 不干预)
        # =====================================================================
        
        else:
            vol_applied_info["action"] = "NO_ACTION"
        
        # 记录到 caps_applied
        base_result.caps_applied["vol_alpha"] = vol_applied_info
        
        return base_result
    
    @staticmethod
    def check_vol_alpha_veto(
        vol_signal: VolAlphaSignal,
        base_result: FusionResult,
    ) -> Tuple[bool, str]:
        """
        检查 VolAlpha 是否应该 veto
        
        Returns:
            (should_veto, reason)
        """
        # 只有 suppress 模式才有 veto 可能
        if vol_signal.vol_bias != "suppress":
            return False, ""
        
        # 如果 vol_intensity 很高且想做新仓
        if vol_signal.vol_intensity > 0.8 and abs(base_result.direction) > 0.5:
            if base_result.target_exposure > 0.3:
                return True, "VOL_ALPHA_SUPPRESS_NEW_POSITION"
        
        return False, ""


# =============================================================================
# EXTENDED AUTHORITY FUSION ENGINE
# =============================================================================

class VolAlphaAwareFusionEngine:
    """
    支持 VolatilityAlpha 的 Authority Fusion Engine
    
    包装原有的 AuthorityFusionEngine，添加 VolAlpha 处理
    """
    
    def __init__(self):
        self._base_engine = get_fusion_engine()
        self._vol_alpha_agent: Optional[VolatilityAlphaAgent] = None
        
        # 统计
        self._stats = {
            "fusions_with_vol_alpha": 0,
            "vol_alpha_amplify_count": 0,
            "vol_alpha_suppress_count": 0,
            "vol_alpha_bear_protect_count": 0,
            "vol_alpha_veto_count": 0,
        }
        
        logger.info("VolAlphaAwareFusionEngine initialized")
    
    def set_vol_alpha_agent(self, agent: VolatilityAlphaAgent):
        """设置 VolatilityAlpha Agent"""
        self._vol_alpha_agent = agent
    
    def fuse_with_vol_alpha(
        self,
        signals: Dict[str, AgentSignal],
        context: FusionContext,
        vol_intent: Optional[VolatilityIntent] = None,
    ) -> FusionResult:
        """
        带 VolatilityAlpha 的融合
        
        Args:
            signals: 原有 agent 信号
            context: 融合上下文
            vol_intent: VolatilityAlpha 意图 (可选，会自动评估)
        
        Returns:
            FusionResult
        """
        # 1. 执行基础融合
        base_result = self._base_engine.fuse(signals, context)
        
        # 2. 获取或评估 VolAlpha 意图
        if vol_intent is None and self._vol_alpha_agent:
            vol_intent = self._vol_alpha_agent.evaluate()
        
        if vol_intent is None or not vol_intent.is_valid:
            return base_result
        
        # 3. 转换为信号格式
        vol_signal = VolAlphaSignal.from_intent(vol_intent)
        
        # 4. 获取 ModelAlpha 信号
        model_alpha_signal = signals.get("model_alpha")
        
        # 5. 检查 veto
        should_veto, veto_reason = VolAlphaFusionRules.check_vol_alpha_veto(
            vol_signal, base_result
        )
        
        if should_veto:
            logger.warning(f"VolAlpha VETO: {veto_reason}")
            self._stats["vol_alpha_veto_count"] += 1
            
            # 记录 veto
            base_result.vetoes_active.append(f"vol_alpha:{veto_reason}")
            base_result.target_exposure = 0.0
            
            # 通知 VolAlpha agent
            if self._vol_alpha_agent:
                self._vol_alpha_agent.record_fusion_result(
                    accepted=False, veto_reason=veto_reason
                )
            
            return base_result
        
        # 6. 应用 VolAlpha 规则
        result = VolAlphaFusionRules.apply_vol_alpha(
            base_result, vol_signal, model_alpha_signal, context
        )
        
        # 7. 更新统计
        self._stats["fusions_with_vol_alpha"] += 1
        
        vol_action = result.caps_applied.get("vol_alpha", {}).get("action", "")
        if vol_action == "AMPLIFY_SHORT":
            self._stats["vol_alpha_amplify_count"] += 1
        elif vol_action == "SUPPRESS_EXECUTION":
            self._stats["vol_alpha_suppress_count"] += 1
        elif vol_action == "BEAR_PROTECTION_SUPPRESS":
            self._stats["vol_alpha_bear_protect_count"] += 1
        
        # 8. 通知 VolAlpha agent
        if self._vol_alpha_agent:
            self._vol_alpha_agent.record_fusion_result(accepted=True)
        
        # 9. 日志
        self._log_fusion(vol_signal, result, context)
        
        return result
    
    def _log_fusion(
        self,
        vol_signal: VolAlphaSignal,
        result: FusionResult,
        context: FusionContext,
    ):
        """记录融合日志 (必须)"""
        vol_info = result.caps_applied.get("vol_alpha", {})
        
        logger.info(
            f"VolAlpha Fusion: mode={context.mode.name} "
            f"vol_bias={vol_signal.vol_bias} vol_intensity={vol_signal.vol_intensity:.2f} "
            f"action={vol_info.get('action', 'NONE')} "
            f"direction={result.direction:.2f} exposure={result.target_exposure:.2f} "
            f"exec_mode={result.execution_mode} urgency={result.urgency:.2f}"
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        return self._stats.copy()


# =============================================================================
# RISK INTEGRATION
# =============================================================================

class VolAlphaRiskConstraint:
    """
    VolAlpha 风险约束
    
    当组合波动率 > target 时，VolAlpha 自动降权
    """
    
    def __init__(
        self,
        portfolio_vol_target: float = 0.3,  # 30% 年化目标
        max_vol_alpha_weight: float = 1.0,
    ):
        self.portfolio_vol_target = portfolio_vol_target
        self.max_vol_alpha_weight = max_vol_alpha_weight
        
        self._current_weight = 1.0
    
    def compute_weight(self, current_portfolio_vol: float) -> float:
        """
        计算 VolAlpha 权重
        
        当组合波动率超过目标时降权
        """
        if current_portfolio_vol <= self.portfolio_vol_target:
            self._current_weight = self.max_vol_alpha_weight
        else:
            # 超过目标波动率，按比例降权
            excess_ratio = (current_portfolio_vol - self.portfolio_vol_target) / self.portfolio_vol_target
            self._current_weight = max(
                self.max_vol_alpha_weight * (1 - excess_ratio),
                0.0
            )
        
        logger.debug(
            f"VolAlpha risk weight: portfolio_vol={current_portfolio_vol:.2%} "
            f"target={self.portfolio_vol_target:.2%} weight={self._current_weight:.2f}"
        )
        
        return self._current_weight
    
    def apply_weight_to_intent(self, intent: VolatilityIntent) -> VolatilityIntent:
        """
        将权重应用到 VolatilityIntent
        """
        weighted_intensity = intent.intensity * self._current_weight
        
        # 如果权重降到很低，改为 neutral
        if self._current_weight < 0.2:
            return VolatilityIntent(
                asset=intent.asset,
                vol_bias=VolBias.NEUTRAL,
                intensity=0.0,
                expected_horizon=intent.expected_horizon,
                rationale_tags=intent.rationale_tags + ["risk_downweighted"],
                data_quality=intent.data_quality,
                is_valid=True,
            )
        
        return VolatilityIntent(
            asset=intent.asset,
            vol_bias=intent.vol_bias,
            intensity=weighted_intensity,
            expected_horizon=intent.expected_horizon,
            rationale_tags=intent.rationale_tags,
            sub_signals=intent.sub_signals,
            data_quality=intent.data_quality,
            is_valid=intent.is_valid,
        )


# =============================================================================
# DATA HEALTH INTEGRATION
# =============================================================================

class VolAlphaDataHealthGate:
    """
    VolAlpha 数据健康门控
    
    若 LOB/链上/情绪中 ≥1 个关键源失效 -> 权重直接降为 0
    """
    
    @staticmethod
    def check_data_health(
        vol_agent: VolatilityAlphaAgent,
    ) -> Tuple[bool, List[str]]:
        """
        检查数据健康
        
        Returns:
            (is_healthy, missing_sources)
        """
        status = vol_agent.get_status()
        data_sources = status.get("data_sources", {})
        
        # 关键数据源
        critical_sources = ["lob", "volatility"]
        supplementary_sources = ["onchain", "sentiment", "correlation"]
        
        missing = []
        
        # 检查关键源
        for source in critical_sources:
            if not data_sources.get(source, False):
                missing.append(source)
        
        # 检查补充源 (至少需要 1 个)
        supplementary_available = sum(
            1 for s in supplementary_sources
            if data_sources.get(s, False)
        )
        
        if supplementary_available == 0:
            missing.append("all_supplementary")
        
        is_healthy = len(missing) == 0
        
        if not is_healthy:
            logger.warning(f"VolAlpha data health check FAILED: missing {missing}")
        
        return is_healthy, missing
    
    @staticmethod
    def gate_intent(
        intent: VolatilityIntent,
        is_healthy: bool,
        missing_sources: List[str],
    ) -> VolatilityIntent:
        """
        门控意图
        
        如果数据不健康，返回无效意图
        """
        if is_healthy:
            return intent
        
        return VolatilityIntent(
            asset=intent.asset,
            vol_bias=VolBias.NEUTRAL,
            intensity=0.0,
            expected_horizon=intent.expected_horizon,
            rationale_tags=[f"data_health_fail:{','.join(missing_sources)}"],
            data_quality=0.0,
            is_valid=False,
        )


# =============================================================================
# SINGLETON & FACTORY
# =============================================================================

_vol_alpha_fusion_engine: Optional[VolAlphaAwareFusionEngine] = None


def get_vol_alpha_fusion_engine() -> VolAlphaAwareFusionEngine:
    """获取带 VolAlpha 的 Fusion Engine 单例"""
    global _vol_alpha_fusion_engine
    if _vol_alpha_fusion_engine is None:
        _vol_alpha_fusion_engine = VolAlphaAwareFusionEngine()
        
        # 自动连接 VolAlpha agent
        vol_agent = get_volatility_alpha_agent()
        _vol_alpha_fusion_engine.set_vol_alpha_agent(vol_agent)
    
    return _vol_alpha_fusion_engine


def reset_vol_alpha_fusion_engine():
    """重置单例"""
    global _vol_alpha_fusion_engine
    _vol_alpha_fusion_engine = None


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'VOLATILITY_ALPHA_AUTHORITY',
    'VolAlphaSignal',
    'VolAlphaFusionRules',
    'VolAlphaAwareFusionEngine',
    'VolAlphaRiskConstraint',
    'VolAlphaDataHealthGate',
    'get_vol_alpha_fusion_engine',
    'reset_vol_alpha_fusion_engine',
]


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Testing VolAlpha Authority Fusion Integration")
    print("=" * 60)
    
    from market.phase_detector import RegimePhase
    
    # 创建 VolAlpha Agent
    vol_agent = get_volatility_alpha_agent()
    
    # 模拟熊市数据
    vol_agent.on_volatility_state({
        "asset": "BTC",
        "vol_percentile_30d": 85.0,
        "vol_zscore": 2.0,
    })
    vol_agent.on_sentiment_tick({
        "asset": "BTC",
        "fear_index": 18.0,
    })
    vol_agent.on_onchain_tick({
        "asset": "BTC",
        "exchange_inflow_spike": True,
    })
    vol_agent.on_lob_tick({
        "asset": "BTC",
        "spread_bps": 15.0,
    })
    vol_agent.on_correlation_state({
        "correlation_spike": True,
    })
    
    # 评估 VolAlpha
    vol_intent = vol_agent.evaluate("BTC")
    print(f"\n1. VolAlpha Intent:")
    print(f"   vol_bias: {vol_intent.vol_bias.value}")
    print(f"   intensity: {vol_intent.intensity:.2f}")
    print(f"   tags: {vol_intent.rationale_tags}")
    
    # 创建融合引擎
    fusion_engine = get_vol_alpha_fusion_engine()
    
    # 场景 1: ModelAlpha = short (正确方向)
    print("\n2. Scenario: ModelAlpha=SHORT + VolAlpha=INCREASE")
    
    signals = {
        "quant": AgentSignal(direction=-0.7, confidence=0.8),
        "model_alpha": AgentSignal(direction=-0.6, confidence=0.75),
        "regime": AgentSignal(direction=-0.5, confidence=0.7),
    }
    
    context = FusionContext(
        mode=SystemMode.OPPORTUNITY,
        regime_phase=RegimePhase.EXPANSION,
        data_valid=True,
    )
    
    result = fusion_engine.fuse_with_vol_alpha(signals, context, vol_intent)
    
    print(f"   direction: {result.direction:.2f}")
    print(f"   exposure: {result.target_exposure:.2f}")
    print(f"   exec_mode: {result.execution_mode}")
    print(f"   urgency: {result.urgency:.2f}")
    print(f"   vol_alpha_info: {result.caps_applied.get('vol_alpha', {})}")
    
    # 场景 2: ModelAlpha = long (危险方向)
    print("\n3. Scenario: ModelAlpha=LONG + VolAlpha=INCREASE (Bear Protection)")
    
    signals["quant"] = AgentSignal(direction=0.6, confidence=0.7)
    signals["model_alpha"] = AgentSignal(direction=0.5, confidence=0.6)
    
    result = fusion_engine.fuse_with_vol_alpha(signals, context, vol_intent)
    
    print(f"   direction: {result.direction:.2f}")
    print(f"   exposure: {result.target_exposure:.2f}")
    print(f"   exec_mode: {result.execution_mode}")
    print(f"   vol_alpha_info: {result.caps_applied.get('vol_alpha', {})}")
    
    # 场景 3: 风险约束
    print("\n4. Risk Constraint Test")
    
    risk_constraint = VolAlphaRiskConstraint(portfolio_vol_target=0.3)
    
    weight_low = risk_constraint.compute_weight(0.2)  # 低于目标
    print(f"   Portfolio vol 20%: weight = {weight_low:.2f}")
    
    weight_high = risk_constraint.compute_weight(0.5)  # 高于目标
    print(f"   Portfolio vol 50%: weight = {weight_high:.2f}")
    
    # 获取统计
    print("\n5. Stats:")
    stats = fusion_engine.get_stats()
    print(f"   {stats}")
    
    print("\n✓ VolAlpha Authority Fusion Integration test complete")
