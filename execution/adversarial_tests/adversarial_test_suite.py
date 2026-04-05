"""
================================================================================
HMATS v5.2.1 SOTA - 执行对抗测试 (熊市专用)
Adversarial Execution Tests
================================================================================

Purpose: 验证在最差流动性 + 做空环境下系统是否会"不该下单却下单"

测试场景:
1. 流动性枯竭 (Liquidity Dryup)
2. Spread 爆炸 (Spread Blowout)
3. 恐慌资金流 (Panic Flow)

验收标准:
- 系统在这些条件下不会盲目下单
- 执行失败率控制在可接受范围
- 不会被过度止损

================================================================================
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


# =============================================================================
# TEST CONFIGURATION
# =============================================================================

@dataclass
class AdversarialTestConfig:
    """对抗测试配置"""
    
    # 流动性枯竭
    liquidity_dryup_depth_reduction: float = 0.9   # 深度减少 90%
    liquidity_dryup_spread_multiplier: float = 5.0 # spread 放大 5x
    
    # Spread 爆炸
    spread_blowout_bps_min: float = 50.0
    spread_blowout_bps_max: float = 200.0
    spread_blowout_duration_bars: int = 10
    
    # 恐慌资金流
    panic_flow_imbalance_threshold: float = 0.8    # 极端不平衡
    panic_flow_inflow_spike_multiplier: float = 10.0
    
    # 验收阈值
    max_acceptable_failure_rate: float = 0.3       # 最多 30% 失败率
    max_acceptable_slippage_bps: float = 100.0
    max_unintended_orders_rate: float = 0.05       # 最多 5% 不该下的单


# =============================================================================
# TEST SCENARIOS
# =============================================================================

@dataclass
class TestScenario:
    """测试场景"""
    name: str
    description: str
    
    # LOB 条件
    spread_bps: float = 0.0
    depth_reduction: float = 0.0     # 0-1, 减少比例
    imbalance: float = 0.0           # -1 to 1
    
    # 市场条件
    volatility: float = 0.0
    momentum: float = 0.0            # 负 = 下跌
    fear_index: float = 50.0
    
    # 链上条件
    exchange_inflow_multiplier: float = 1.0
    
    # 预期行为
    should_allow_order: bool = True
    expected_action: str = "limit_passive"  # wait, limit_passive, limit_aggressive
    
    def __repr__(self):
        return f"TestScenario(name={self.name}, spread={self.spread_bps}bps, depth_reduction={self.depth_reduction:.0%})"


# =============================================================================
# LIQUIDITY DRYUP TEST
# =============================================================================

class LiquidityDryupTest:
    """
    流动性枯竭测试
    
    模拟:
    - 深度骤降
    - Spread 扩大
    - Fill probability 下降
    
    验证:
    - 系统是否会在流动性不足时停止下单
    """
    
    def __init__(self, config: AdversarialTestConfig = None):
        self.config = config or AdversarialTestConfig()
        self._results = []
    
    def generate_scenarios(self) -> List[TestScenario]:
        """生成流动性枯竭场景"""
        scenarios = []
        
        # 渐进式枯竭
        for depth_reduction in [0.5, 0.7, 0.9, 0.95]:
            spread_mult = 1 + (self.config.liquidity_dryup_spread_multiplier - 1) * depth_reduction
            
            scenarios.append(TestScenario(
                name=f"liquidity_dryup_{int(depth_reduction*100)}pct",
                description=f"Depth reduced by {depth_reduction:.0%}, spread {spread_mult:.1f}x",
                spread_bps=10 * spread_mult,
                depth_reduction=depth_reduction,
                imbalance=-0.3,  # 轻微卖压
                volatility=0.5 + depth_reduction * 0.3,
                momentum=-0.02,
                fear_index=30 - depth_reduction * 20,
                should_allow_order=depth_reduction < 0.9,
                expected_action="wait" if depth_reduction >= 0.9 else "limit_passive",
            ))
        
        return scenarios
    
    def run_test(
        self,
        execution_policy,
        scenarios: List[TestScenario] = None,
    ) -> Dict[str, Any]:
        """
        运行测试
        
        Args:
            execution_policy: 执行策略实例
            scenarios: 测试场景列表
        
        Returns:
            测试结果
        """
        from execution.learned_execution_policy import (
            LOBFeatures, ExecutionState, ExecutionAction
        )
        
        scenarios = scenarios or self.generate_scenarios()
        results = []
        
        for scenario in scenarios:
            # 构建 LOB
            base_depth = 500000
            adjusted_depth = base_depth * (1 - scenario.depth_reduction)
            
            lob = LOBFeatures(
                spread_bps=scenario.spread_bps,
                imbalance=scenario.imbalance,
                bid_depth_usd=adjusted_depth * (1 - scenario.imbalance) / 2,
                ask_depth_usd=adjusted_depth * (1 + scenario.imbalance) / 2,
                vpin=0.5 + scenario.depth_reduction * 0.3,
            )
            
            # 构建执行状态
            exec_state = ExecutionState(
                target_quantity=1.0,
                executed_quantity=0.3,
                side="sell",  # 做空
                arrival_price=50000,
                current_mid_price=49000,
                deadline_sec=600,
            )
            
            # 获取建议
            market_embedding = np.random.randn(128) * 0.1
            advice = execution_policy.get_advice(lob, market_embedding, exec_state)
            
            # 评估
            actual_action = advice.recommended_action.value
            
            # 检查是否符合预期
            if scenario.should_allow_order:
                passed = actual_action != "wait" or scenario.expected_action == "wait"
            else:
                passed = actual_action == "wait" or advice.aggressiveness < 0.3
            
            result = {
                "scenario": scenario.name,
                "expected_action": scenario.expected_action,
                "actual_action": actual_action,
                "should_allow": scenario.should_allow_order,
                "passed": passed,
                "aggressiveness": advice.aggressiveness,
                "confidence": advice.confidence,
            }
            
            results.append(result)
            self._results.append(result)
        
        # 汇总
        passed_count = sum(1 for r in results if r["passed"])
        
        return {
            "test_name": "LiquidityDryupTest",
            "total_scenarios": len(results),
            "passed": passed_count,
            "failed": len(results) - passed_count,
            "pass_rate": passed_count / len(results) if results else 0,
            "details": results,
        }


# =============================================================================
# SPREAD BLOWOUT TEST
# =============================================================================

class SpreadBlowoutTest:
    """
    Spread 爆炸测试
    
    模拟:
    - Spread 突然放大到 50-200 bps
    - 持续多根 bar
    
    验证:
    - 系统是否会在 spread 爆炸时暂停或变被动
    """
    
    def __init__(self, config: AdversarialTestConfig = None):
        self.config = config or AdversarialTestConfig()
        self._results = []
    
    def generate_scenarios(self) -> List[TestScenario]:
        """生成 spread 爆炸场景"""
        scenarios = []
        
        for spread_bps in [30, 50, 80, 120, 200]:
            scenarios.append(TestScenario(
                name=f"spread_blowout_{int(spread_bps)}bps",
                description=f"Spread blowout to {spread_bps} bps",
                spread_bps=spread_bps,
                depth_reduction=0.3,  # 轻微深度减少
                imbalance=-0.2,
                volatility=0.7,
                momentum=-0.03,
                fear_index=20,
                should_allow_order=spread_bps < 80,
                expected_action="wait" if spread_bps >= 80 else "limit_passive",
            ))
        
        return scenarios
    
    def run_test(
        self,
        execution_policy,
        scenarios: List[TestScenario] = None,
    ) -> Dict[str, Any]:
        """运行测试"""
        from execution.learned_execution_policy import (
            LOBFeatures, ExecutionState, ExecutionAction
        )
        
        scenarios = scenarios or self.generate_scenarios()
        results = []
        
        for scenario in scenarios:
            lob = LOBFeatures(
                spread_bps=scenario.spread_bps,
                imbalance=scenario.imbalance,
                bid_depth_usd=400000,
                ask_depth_usd=500000,
                vpin=0.6,
            )
            
            exec_state = ExecutionState(
                target_quantity=1.0,
                executed_quantity=0.2,
                side="sell",
                arrival_price=50000,
                current_mid_price=48500,
                deadline_sec=900,
            )
            
            market_embedding = np.random.randn(128) * 0.1
            advice = execution_policy.get_advice(lob, market_embedding, exec_state)
            
            actual_action = advice.recommended_action.value
            
            # 高 spread 时应该被动或等待
            if scenario.spread_bps >= 80:
                passed = actual_action in ["wait", "limit_passive"] or advice.aggressiveness < 0.4
            else:
                passed = True  # 低 spread 允许任何动作
            
            result = {
                "scenario": scenario.name,
                "spread_bps": scenario.spread_bps,
                "actual_action": actual_action,
                "passed": passed,
                "aggressiveness": advice.aggressiveness,
            }
            
            results.append(result)
            self._results.append(result)
        
        passed_count = sum(1 for r in results if r["passed"])
        
        return {
            "test_name": "SpreadBlowoutTest",
            "total_scenarios": len(results),
            "passed": passed_count,
            "failed": len(results) - passed_count,
            "pass_rate": passed_count / len(results) if results else 0,
            "details": results,
        }


# =============================================================================
# PANIC FLOW TEST
# =============================================================================

class PanicFlowTest:
    """
    恐慌资金流测试
    
    模拟:
    - 交易所流入激增
    - 极端卖压 (imbalance)
    - 恐慌情绪
    
    验证:
    - 系统是否会在恐慌时保持理性
    - 做空方向是否能正确执行
    """
    
    def __init__(self, config: AdversarialTestConfig = None):
        self.config = config or AdversarialTestConfig()
        self._results = []
    
    def generate_scenarios(self) -> List[TestScenario]:
        """生成恐慌资金流场景"""
        scenarios = []
        
        # 不同程度的恐慌
        panic_levels = [
            ("mild_panic", -0.4, 25, 3.0),
            ("moderate_panic", -0.6, 15, 5.0),
            ("extreme_panic", -0.8, 8, 10.0),
            ("market_crash", -0.95, 3, 20.0),
        ]
        
        for name, imbalance, fear, inflow_mult in panic_levels:
            scenarios.append(TestScenario(
                name=name,
                description=f"Panic: imbalance={imbalance}, fear={fear}",
                spread_bps=20 + abs(imbalance) * 30,
                depth_reduction=0.3 + abs(imbalance) * 0.4,
                imbalance=imbalance,
                volatility=0.5 + abs(imbalance) * 0.4,
                momentum=-0.05 * abs(imbalance) / 0.4,
                fear_index=fear,
                exchange_inflow_multiplier=inflow_mult,
                should_allow_order=fear > 5,  # 极端崩溃时不下单
                expected_action="limit_passive" if fear > 10 else "wait",
            ))
        
        return scenarios
    
    def run_test(
        self,
        execution_policy,
        scenarios: List[TestScenario] = None,
    ) -> Dict[str, Any]:
        """运行测试"""
        from execution.learned_execution_policy import (
            LOBFeatures, ExecutionState, ExecutionAction
        )
        
        scenarios = scenarios or self.generate_scenarios()
        results = []
        
        # 设置熊市模式
        execution_policy.set_bear_market(True)
        
        for scenario in scenarios:
            base_depth = 500000
            adjusted_depth = base_depth * (1 - scenario.depth_reduction)
            
            lob = LOBFeatures(
                spread_bps=scenario.spread_bps,
                imbalance=scenario.imbalance,
                bid_depth_usd=adjusted_depth * 0.3,  # 极少买盘
                ask_depth_usd=adjusted_depth * 0.7,  # 大量卖盘
                vpin=0.8,  # 高 VPIN
            )
            
            exec_state = ExecutionState(
                target_quantity=1.0,
                executed_quantity=0.1,
                side="sell",  # 做空
                arrival_price=50000,
                current_mid_price=47000,  # 已下跌
                deadline_sec=1200,
                slippage_bps=-60,  # 负 = 有利 (卖在高点)
            )
            
            market_embedding = np.random.randn(128) * 0.1
            # 加入熊市信号
            market_embedding[0:10] = -0.5  # 模拟熊市 embedding
            
            advice = execution_policy.get_advice(lob, market_embedding, exec_state)
            
            actual_action = advice.recommended_action.value
            
            # 评估
            if scenario.fear_index <= 5:
                # 极端崩溃: 应该等待
                passed = actual_action == "wait" or advice.aggressiveness < 0.2
            else:
                # 可执行的恐慌: 应该被动但可以下单
                passed = advice.aggressiveness < 0.6  # 不应太激进
            
            result = {
                "scenario": scenario.name,
                "fear_index": scenario.fear_index,
                "imbalance": scenario.imbalance,
                "actual_action": actual_action,
                "passed": passed,
                "aggressiveness": advice.aggressiveness,
                "bear_adjusted": advice.bear_market_adjusted,
            }
            
            results.append(result)
            self._results.append(result)
        
        passed_count = sum(1 for r in results if r["passed"])
        
        return {
            "test_name": "PanicFlowTest",
            "total_scenarios": len(results),
            "passed": passed_count,
            "failed": len(results) - passed_count,
            "pass_rate": passed_count / len(results) if results else 0,
            "details": results,
        }


# =============================================================================
# COMPREHENSIVE ADVERSARIAL TEST SUITE
# =============================================================================

class AdversarialTestSuite:
    """
    完整对抗测试套件
    """
    
    def __init__(self, config: AdversarialTestConfig = None):
        self.config = config or AdversarialTestConfig()
        
        self._liquidity_test = LiquidityDryupTest(self.config)
        self._spread_test = SpreadBlowoutTest(self.config)
        self._panic_test = PanicFlowTest(self.config)
    
    def run_all_tests(self, execution_policy) -> Dict[str, Any]:
        """运行所有测试"""
        results = {}
        
        logger.info("Running Adversarial Test Suite...")
        
        # 1. Liquidity Dryup
        logger.info("Test 1: Liquidity Dryup")
        results["liquidity_dryup"] = self._liquidity_test.run_test(execution_policy)
        
        # 2. Spread Blowout
        logger.info("Test 2: Spread Blowout")
        results["spread_blowout"] = self._spread_test.run_test(execution_policy)
        
        # 3. Panic Flow
        logger.info("Test 3: Panic Flow")
        results["panic_flow"] = self._panic_test.run_test(execution_policy)
        
        # 汇总
        total_scenarios = sum(r["total_scenarios"] for r in results.values())
        total_passed = sum(r["passed"] for r in results.values())
        
        overall_pass_rate = total_passed / total_scenarios if total_scenarios > 0 else 0
        
        # 验收判断
        acceptance = overall_pass_rate >= (1 - self.config.max_acceptable_failure_rate)
        
        return {
            "suite_name": "AdversarialTestSuite",
            "timestamp": datetime.now().isoformat(),
            "total_scenarios": total_scenarios,
            "total_passed": total_passed,
            "total_failed": total_scenarios - total_passed,
            "overall_pass_rate": overall_pass_rate,
            "acceptance_threshold": 1 - self.config.max_acceptable_failure_rate,
            "accepted": acceptance,
            "test_results": results,
        }
    
    def print_report(self, results: Dict[str, Any]):
        """打印报告"""
        print("\n" + "=" * 70)
        print("ADVERSARIAL TEST SUITE REPORT")
        print("=" * 70)
        
        print(f"\nOverall: {results['total_passed']}/{results['total_scenarios']} passed")
        print(f"Pass Rate: {results['overall_pass_rate']:.1%}")
        print(f"Acceptance Threshold: {results['acceptance_threshold']:.1%}")
        print(f"Status: {'✓ ACCEPTED' if results['accepted'] else '✗ FAILED'}")
        
        print("\n" + "-" * 70)
        print("Test Breakdown:")
        print("-" * 70)
        
        for test_name, test_result in results["test_results"].items():
            status = "✓" if test_result["pass_rate"] >= 0.7 else "✗"
            print(f"  {status} {test_name}: {test_result['passed']}/{test_result['total_scenarios']} ({test_result['pass_rate']:.1%})")
        
        print("\n" + "=" * 70)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'AdversarialTestConfig',
    'TestScenario',
    'LiquidityDryupTest',
    'SpreadBlowoutTest',
    'PanicFlowTest',
    'AdversarialTestSuite',
]


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Running Adversarial Execution Tests")
    print("=" * 60)
    
    # 导入执行策略
    from execution.learned_execution_policy import (
        LearnedExecutionPolicy, LearnedExecutionConfig
    )
    
    # 创建策略 (advisory mode)
    config = LearnedExecutionConfig(mode="advisory")
    policy = LearnedExecutionPolicy(config)
    
    # 运行测试套件
    test_suite = AdversarialTestSuite()
    results = test_suite.run_all_tests(policy)
    
    # 打印报告
    test_suite.print_report(results)
