"""
================================================================================
HMATS v5.2.1 P1 - 执行智能完整集成
P1 Complete Integration
================================================================================

Purpose: 统一管理 P1 所有组件的集成层

P1 组件:
- P1-A: DRL Execution Advisory (rl_execution_advisor.py)
- P1-B-1: Execution Quality Logger (execution_quality_logger.py)
- P1-B-2: Impact Calibration (impact_calibration.py)
- P1-B-3: Execution Quality Report (execution_quality_report.py)

关键约束 (P1-C):
- DRL 不允许 entry authority
- DRL 不允许绕过 TradeGate / Risk
- DRL advisory 可被 correlation / volatility / DataHealth 强制降权

================================================================================
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class P1IntegrationConfig:
    """P1 集成配置"""
    
    # 启用开关
    enable_rl_advisor: bool = True              # 启用 RL 执行顾问
    enable_quality_logging: bool = True         # 启用质量日志
    enable_impact_calibration: bool = True      # 启用冲击校准
    
    # RL Advisor 模式
    rl_advisor_mode: str = "shadow"             # "disabled", "shadow", "advisory", "active"
    
    # 自动降权条件
    correlation_spike_downweight: float = 0.5   # 相关性 spike 时降权
    volatility_spike_downweight: float = 0.3    # 波动率 spike 时降权
    data_health_critical_disable: bool = True   # 数据健康严重时禁用
    
    # 熔断条件
    max_consecutive_slippage_violations: int = 3
    slippage_violation_threshold_bps: float = 10.0
    
    # 报告
    auto_report_interval_hours: int = 24
    report_output_dir: str = "reports"


# =============================================================================
# P1 EXECUTION INTEGRATOR
# =============================================================================

class P1ExecutionIntegrator:
    """
    P1 执行智能集成器
    
    职责:
    1. 统一管理 RL Advisor, Quality Logger, Impact Calibration
    2. 提供执行主路径的集成 API
    3. 强制执行 P1-C 权力边界
    """
    
    def __init__(self, config: P1IntegrationConfig = None):
        self.config = config or P1IntegrationConfig()
        
        # 组件实例
        self._rl_advisor = None
        self._quality_logger = None
        self._calibration_bridge = None
        self._report_generator = None
        
        # 状态
        self._initialized = False
        self._rl_weight = 1.0                   # RL 当前权重
        self._rl_disabled_reason: str = ""
        
        # 统计
        self._stats = {
            "total_executions": 0,
            "rl_used_count": 0,
            "fallback_count": 0,
            "fuse_count": 0,
            "downweight_count": 0,
        }
        
        # EventBus 回调
        self._event_callback: Optional[Callable] = None
        
        logger.info("P1ExecutionIntegrator created")
    
    def initialize(self) -> bool:
        """初始化所有 P1 组件"""
        if self._initialized:
            return True
        
        try:
            # 1. 初始化 RL Advisor
            if self.config.enable_rl_advisor:
                from execution.rl_execution_advisor import (
                    RLExecutionAdvisor,
                    RLExecutionAdvisorConfig,
                    AdvisoryMode,
                )
                
                mode_map = {
                    "disabled": AdvisoryMode.DISABLED,
                    "shadow": AdvisoryMode.SHADOW,
                    "advisory": AdvisoryMode.ADVISORY,
                    "active": AdvisoryMode.ACTIVE,
                }
                
                advisor_config = RLExecutionAdvisorConfig(
                    mode=mode_map.get(self.config.rl_advisor_mode, AdvisoryMode.SHADOW),
                    max_consecutive_slippage_violations=self.config.max_consecutive_slippage_violations,
                    slippage_violation_threshold_bps=self.config.slippage_violation_threshold_bps,
                )
                
                self._rl_advisor = RLExecutionAdvisor(advisor_config)
                logger.info(f"RL Advisor initialized: mode={self.config.rl_advisor_mode}")
            
            # 2. 初始化 Quality Logger
            if self.config.enable_quality_logging:
                from execution.execution_quality_logger import (
                    ExecutionQualityLogger,
                    ExecutionQualityLoggerConfig,
                )
                
                logger_config = ExecutionQualityLoggerConfig()
                self._quality_logger = ExecutionQualityLogger(logger_config)
                logger.info("Execution Quality Logger initialized")
            
            # 3. 初始化 Calibration Bridge
            if self.config.enable_impact_calibration:
                from execution.impact_calibration import (
                    ProductionMarketImpactCalibrationBridge,
                )
                
                self._calibration_bridge = ProductionMarketImpactCalibrationBridge()
                logger.info("Impact Calibration Bridge initialized")
            
            # 4. 初始化 Report Generator
            from analytics.execution_quality_report import (
                ExecutionQualityReportGenerator,
            )
            
            self._report_generator = ExecutionQualityReportGenerator(
                self.config.report_output_dir
            )
            
            self._initialized = True
            logger.info("P1 Integration initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"P1 Integration initialization failed: {e}")
            return False
    
    def set_event_callback(self, callback: Callable):
        """设置 EventBus 回调"""
        self._event_callback = callback
        
        if self._rl_advisor:
            self._rl_advisor.set_event_callback(callback)
        if self._quality_logger:
            self._quality_logger.set_event_callback(callback)
    
    # =========================================================================
    # MAIN EXECUTION API
    # =========================================================================
    
    def get_execution_advice(
        self,
        asset: str,
        side: str,
        total_quantity: float,
        executed_quantity: float,
        mid_price: float,
        spread_bps: float,
        volatility: float,
        lob_imbalance: float,
        bid_depth_usd: float,
        ask_depth_usd: float,
        urgency: float = 0.5,
        recent_slippage: List[float] = None,
        recent_fill_ratio: List[float] = None,
    ) -> Dict[str, Any]:
        """
        获取执行建议
        
        这是主入口，production_market_impact 应调用此方法
        
        Returns:
            {
                "slice_size_pct": float,
                "wait_time_sec": float,
                "limit_offset_bps": float,
                "cancel_replace_policy": str,
                "source": "rl" | "baseline" | "fallback",
                "confidence": float,
                "reasoning": List[str],
            }
        """
        self._stats["total_executions"] += 1
        
        # 默认 baseline 建议
        baseline = self._compute_baseline_advice(
            asset=asset,
            side=side,
            total_quantity=total_quantity,
            executed_quantity=executed_quantity,
            spread_bps=spread_bps,
            volatility=volatility,
            lob_imbalance=lob_imbalance,
            urgency=urgency,
        )
        
        # 如果 RL 不可用，直接返回 baseline
        if not self._rl_advisor or not self.config.enable_rl_advisor:
            return {**baseline, "source": "baseline"}
        
        # 如果 RL 被禁用
        if self._rl_weight == 0:
            self._stats["fallback_count"] += 1
            return {
                **baseline,
                "source": "fallback",
                "reasoning": [f"RL disabled: {self._rl_disabled_reason}"],
            }
        
        try:
            from execution.rl_execution_advisor import ExecutionState
            
            # 构建状态
            state = ExecutionState(
                timestamp=datetime.now(),
                asset=asset,
                side=side,
                total_quantity=total_quantity,
                executed_quantity=executed_quantity,
                remaining_quantity=total_quantity - executed_quantity,
                mid_price=mid_price,
                spread_bps=spread_bps,
                volatility=volatility,
                lob_imbalance=lob_imbalance,
                bid_depth_usd=bid_depth_usd,
                ask_depth_usd=ask_depth_usd,
                recent_slippage=recent_slippage or [],
                recent_fill_ratio=recent_fill_ratio or [],
                recent_wait_time=[],
                urgency=urgency,
            )
            
            # 获取 RL 建议
            advice = self._rl_advisor.get_advice(state)
            
            # 如果是回退
            if advice.is_fallback:
                self._stats["fallback_count"] += 1
                return {
                    **baseline,
                    "source": "fallback",
                    "reasoning": advice.reasoning,
                }
            
            # 应用权重调整
            confidence = advice.confidence * self._rl_weight
            
            # 如果置信度太低，使用 baseline
            if confidence < 0.5:
                return {
                    **baseline,
                    "source": "baseline",
                    "reasoning": [f"Low RL confidence: {confidence:.2f}"],
                }
            
            self._stats["rl_used_count"] += 1
            
            return {
                "slice_size_pct": advice.slice_size_pct,
                "wait_time_sec": advice.wait_time_sec,
                "limit_offset_bps": advice.limit_offset_bps,
                "cancel_replace_policy": advice.cancel_replace_policy.value,
                "source": "rl",
                "confidence": confidence,
                "reasoning": advice.reasoning,
            }
            
        except Exception as e:
            logger.warning(f"RL advisor error: {e}")
            self._stats["fallback_count"] += 1
            return {
                **baseline,
                "source": "fallback",
                "reasoning": [f"Error: {str(e)}"],
            }
    
    def _compute_baseline_advice(
        self,
        asset: str,
        side: str,
        total_quantity: float,
        executed_quantity: float,
        spread_bps: float,
        volatility: float,
        lob_imbalance: float,
        urgency: float,
    ) -> Dict[str, Any]:
        """计算 baseline 执行建议"""
        progress = executed_quantity / max(total_quantity, 1e-8)
        remaining_pct = 1 - progress
        
        # 拆单比例
        if urgency > 0.8:
            slice_pct = min(0.4, remaining_pct)
        elif remaining_pct < 0.2:
            slice_pct = remaining_pct
        else:
            slice_pct = min(0.2, remaining_pct)
        
        # 等待时间
        if volatility > 0.8:
            wait_time = 8.0
        elif urgency > 0.7:
            wait_time = 2.0
        else:
            wait_time = 5.0
        
        # 限价偏移
        offset = spread_bps * 0.5
        if volatility > 0.6:
            offset *= 1.3
        
        return {
            "slice_size_pct": slice_pct,
            "wait_time_sec": wait_time,
            "limit_offset_bps": offset,
            "cancel_replace_policy": "on_timeout",
            "confidence": 0.8,
            "reasoning": ["baseline_heuristic"],
        }
    
    # =========================================================================
    # FEEDBACK API
    # =========================================================================
    
    def record_execution_result(
        self,
        asset: str,
        side: str,
        intended_qty: float,
        executed_qty: float,
        arrival_price: float,
        avg_exec_price: float,
        expected_slippage_bps: float,
        time_to_fill_sec: float,
        rl_advice_used: bool,
        num_slices: int = 1,
        cancel_count: int = 0,
        volatility: float = 0.0,
        spread_bps: float = 0.0,
        rl_confidence: float = 0.0,
    ):
        """
        记录执行结果
        
        用于:
        1. 质量日志
        2. RL 反馈学习
        3. 冲击模型校准
        """
        # 计算实际滑点
        if side == "buy":
            realized_slippage_bps = (avg_exec_price - arrival_price) / arrival_price * 10000
        else:
            realized_slippage_bps = (arrival_price - avg_exec_price) / arrival_price * 10000
        
        # 1. 记录到质量日志
        if self._quality_logger:
            self._quality_logger.log_simple(
                asset=asset,
                side=side,
                intended_qty=intended_qty,
                executed_qty=executed_qty,
                arrival_price=arrival_price,
                avg_exec_price=avg_exec_price,
                expected_slippage_bps=expected_slippage_bps,
                time_to_fill_sec=time_to_fill_sec,
                rl_advice_used=rl_advice_used,
                num_slices=num_slices,
                cancel_count=cancel_count,
                volatility=volatility,
                spread_bps=spread_bps,
                rl_confidence=rl_confidence,
            )
        
        # 2. RL 反馈
        if self._rl_advisor:
            fill_ratio = executed_qty / intended_qty if intended_qty > 0 else 0
            self._rl_advisor.record_execution_result(
                advice_used=rl_advice_used,
                realized_slippage_bps=realized_slippage_bps,
                fill_ratio=fill_ratio,
                time_to_fill_sec=time_to_fill_sec,
            )
        
        # 3. 冲击模型校准
        if self._calibration_bridge:
            size_usd = intended_qty * arrival_price
            self._calibration_bridge.record_execution(
                asset=asset,
                side=side,
                size_usd=size_usd,
                expected_impact_bps=expected_slippage_bps,
                realized_impact_bps=realized_slippage_bps,
                volatility=volatility,
                spread_bps=spread_bps,
            )
    
    # =========================================================================
    # CALIBRATION API
    # =========================================================================
    
    def get_calibrated_impact_params(
        self,
        asset: str,
        volatility: float,
        spread_bps: float,
    ) -> Tuple[float, float, float]:
        """
        获取校准的冲击模型参数
        
        供 production_market_impact 使用
        
        Returns:
            (eta, gamma, confidence)
        """
        if not self._calibration_bridge:
            return 0.1, 0.1, 0.0
        
        return self._calibration_bridge.get_calibrated_params(
            asset=asset,
            volatility=volatility,
            spread_bps=spread_bps,
        )
    
    # =========================================================================
    # P1-C: WEIGHT CONTROL
    # =========================================================================
    
    def on_correlation_spike(self, severity: str):
        """
        相关性 spike 回调
        
        P1-C 要求: 可被 correlation 强制降权
        """
        if severity in ["spiking", "crisis"]:
            self._rl_weight *= self.config.correlation_spike_downweight
            self._stats["downweight_count"] += 1
            logger.info(f"P1: RL weight reduced due to correlation spike: {self._rl_weight:.2f}")
            
            if self._rl_advisor:
                self._rl_advisor.on_correlation_spike(severity)
    
    def on_volatility_spike(self, severity: str):
        """
        波动率 spike 回调
        
        P1-C 要求: 可被 volatility 强制降权
        """
        if severity == "extreme":
            self._rl_weight *= self.config.volatility_spike_downweight
            self._stats["downweight_count"] += 1
            logger.info(f"P1: RL weight reduced due to volatility spike: {self._rl_weight:.2f}")
            
            if self._rl_advisor:
                self._rl_advisor.on_volatility_spike(severity)
    
    def on_data_health_change(self, degradation_level: str):
        """
        数据健康变化回调
        
        P1-C 要求: 可被 DataHealth 直接禁用
        """
        if self.config.data_health_critical_disable:
            if degradation_level in ["exit_only", "no_trade"]:
                self._rl_weight = 0.0
                self._rl_disabled_reason = f"DataHealth: {degradation_level}"
                logger.warning(f"P1: RL disabled due to data health: {degradation_level}")
                
                if self._rl_advisor:
                    self._rl_advisor.on_data_health_degraded(degradation_level)
    
    def reset_rl_weight(self):
        """重置 RL 权重"""
        self._rl_weight = 1.0
        self._rl_disabled_reason = ""
        logger.info("P1: RL weight reset to 1.0")
    
    # =========================================================================
    # REPORTING
    # =========================================================================
    
    def generate_report(self) -> Tuple[str, str]:
        """
        生成执行质量报告
        
        Returns:
            (json_path, html_path)
        """
        if not self._quality_logger or not self._report_generator:
            raise RuntimeError("Quality logger or report generator not initialized")
        
        records = list(self._quality_logger._records)
        report = self._report_generator.generate_report(records)
        
        json_path = self._report_generator.save_json(report)
        html_path = self._report_generator.save_html(report)
        
        return json_path, html_path
    
    # =========================================================================
    # STATUS
    # =========================================================================
    
    def get_status(self) -> Dict[str, Any]:
        """获取 P1 集成状态"""
        status = {
            "initialized": self._initialized,
            "rl_advisor_enabled": self._rl_advisor is not None,
            "rl_weight": self._rl_weight,
            "rl_disabled_reason": self._rl_disabled_reason,
            "quality_logger_enabled": self._quality_logger is not None,
            "calibration_enabled": self._calibration_bridge is not None,
            "stats": self._stats,
        }
        
        if self._rl_advisor:
            status["rl_advisor_status"] = self._rl_advisor.get_status()
        
        if self._quality_logger:
            status["quality_summary"] = self._quality_logger.get_summary()
        
        return status
    
    def get_diagnostic_output(self) -> str:
        """获取诊断输出"""
        lines = [
            "=" * 60,
            "P1 Execution Intelligence Status",
            "=" * 60,
            "",
            f"Initialized: {self._initialized}",
            f"RL Advisor: {'Enabled' if self._rl_advisor else 'Disabled'}",
            f"RL Weight: {self._rl_weight:.2f}",
            f"RL Mode: {self.config.rl_advisor_mode}",
            "",
            "Statistics:",
            f"  Total Executions: {self._stats['total_executions']}",
            f"  RL Used: {self._stats['rl_used_count']}",
            f"  Fallback: {self._stats['fallback_count']}",
            f"  Fuse Count: {self._stats['fuse_count']}",
            f"  Downweight Count: {self._stats['downweight_count']}",
        ]
        
        if self._rl_advisor:
            rl_stats = self._rl_advisor.get_stats()
            lines.extend([
                "",
                "RL Advisor Details:",
                f"  Mode: {rl_stats.get('mode', 'N/A')}",
                f"  Is Fused: {rl_stats.get('is_fused', False)}",
                f"  Usage Rate: {rl_stats.get('usage_rate', 0):.1%}",
            ])
        
        if self._quality_logger:
            comparison = self._quality_logger.get_rl_comparison()
            if comparison.get('rl_count'):
                lines.extend([
                    "",
                    "RL vs Baseline:",
                    f"  RL Slippage: {comparison.get('rl_slippage_mean', 0):.2f} bps",
                    f"  Baseline Slippage: {comparison.get('baseline_slippage_mean', 0):.2f} bps",
                ])
                if 'slippage_improvement_bps' in comparison:
                    lines.append(
                        f"  Improvement: {comparison['slippage_improvement_bps']:.2f} bps "
                        f"({comparison['slippage_improvement_pct']:.1f}%)"
                    )
        
        lines.append("=" * 60)
        return "\n".join(lines)


# =============================================================================
# SINGLETON
# =============================================================================

_p1_integrator_instance: Optional[P1ExecutionIntegrator] = None


def get_p1_execution_integrator(
    config: P1IntegrationConfig = None
) -> P1ExecutionIntegrator:
    """获取 P1 集成单例"""
    global _p1_integrator_instance
    if _p1_integrator_instance is None:
        _p1_integrator_instance = P1ExecutionIntegrator(config)
    return _p1_integrator_instance


def reset_p1_execution_integrator():
    """重置单例"""
    global _p1_integrator_instance
    _p1_integrator_instance = None


# =============================================================================
# CONVENIENCE API
# =============================================================================

def initialize_p1_execution() -> bool:
    """初始化 P1 执行智能"""
    integrator = get_p1_execution_integrator()
    return integrator.initialize()


def get_execution_advice_p1(
    asset: str,
    side: str,
    total_quantity: float,
    executed_quantity: float,
    mid_price: float,
    spread_bps: float,
    volatility: float,
    lob_imbalance: float = 0.0,
    urgency: float = 0.5,
) -> Dict[str, Any]:
    """便捷方法: 获取 P1 执行建议"""
    integrator = get_p1_execution_integrator()
    
    if not integrator._initialized:
        integrator.initialize()
    
    return integrator.get_execution_advice(
        asset=asset,
        side=side,
        total_quantity=total_quantity,
        executed_quantity=executed_quantity,
        mid_price=mid_price,
        spread_bps=spread_bps,
        volatility=volatility,
        lob_imbalance=lob_imbalance,
        bid_depth_usd=1000000,  # 默认值
        ask_depth_usd=1000000,
        urgency=urgency,
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'P1IntegrationConfig',
    'P1ExecutionIntegrator',
    'get_p1_execution_integrator',
    'reset_p1_execution_integrator',
    'initialize_p1_execution',
    'get_execution_advice_p1',
]


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Testing P1ExecutionIntegrator")
    print("=" * 60)
    
    # 创建集成器
    config = P1IntegrationConfig(
        rl_advisor_mode="advisory",
    )
    integrator = P1ExecutionIntegrator(config)
    
    # 初始化
    print("\n1. Initializing P1...")
    success = integrator.initialize()
    print(f"   Initialized: {success}")
    
    # 获取执行建议
    print("\n2. Getting execution advice...")
    advice = integrator.get_execution_advice(
        asset="BTC",
        side="buy",
        total_quantity=1.0,
        executed_quantity=0.3,
        mid_price=100000,
        spread_bps=5.0,
        volatility=0.5,
        lob_imbalance=0.2,
        bid_depth_usd=500000,
        ask_depth_usd=450000,
        urgency=0.5,
    )
    print(f"   Source: {advice['source']}")
    print(f"   Slice Size: {advice['slice_size_pct']:.1%}")
    print(f"   Wait Time: {advice['wait_time_sec']:.1f}s")
    print(f"   Limit Offset: {advice['limit_offset_bps']:.1f} bps")
    
    # 记录执行结果
    print("\n3. Recording execution result...")
    integrator.record_execution_result(
        asset="BTC",
        side="buy",
        intended_qty=1.0,
        executed_qty=0.95,
        arrival_price=100000,
        avg_exec_price=100050,
        expected_slippage_bps=5.0,
        time_to_fill_sec=4.2,
        rl_advice_used=advice['source'] == 'rl',
        volatility=0.5,
        spread_bps=5.0,
    )
    print("   Recorded")
    
    # 测试降权
    print("\n4. Testing weight control...")
    print(f"   Initial weight: {integrator._rl_weight:.2f}")
    integrator.on_correlation_spike("spiking")
    print(f"   After correlation spike: {integrator._rl_weight:.2f}")
    integrator.on_volatility_spike("extreme")
    print(f"   After volatility spike: {integrator._rl_weight:.2f}")
    
    # 重置
    integrator.reset_rl_weight()
    print(f"   After reset: {integrator._rl_weight:.2f}")
    
    # 获取状态
    print("\n5. Status:")
    status = integrator.get_status()
    print(f"   RL Enabled: {status['rl_advisor_enabled']}")
    print(f"   RL Weight: {status['rl_weight']:.2f}")
    print(f"   Total Executions: {status['stats']['total_executions']}")
    
    # 诊断输出
    print("\n6. Diagnostic Output:")
    print(integrator.get_diagnostic_output())
    
    print("\n✓ P1ExecutionIntegrator test complete")
