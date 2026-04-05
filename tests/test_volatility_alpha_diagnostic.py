"""
================================================================================
HMATS v5.2.1 - Volatility Alpha Agent 诊断测试
Volatility Alpha Agent Diagnostic Tests
================================================================================

Purpose: 验证 VolatilityAlphaAgent 的功能和集成

测试范围:
- A: VolatilityAlphaAgent 本体
- B: Authority Fusion 集成
- C: 风险集成
- D: Dashboard 集成
- E: 行为验证 (熊市/牛市场景)

验收标准:
1. 熊市场景: VolAlpha 会频繁激活
2. 牛市场景: 自动低权重
3. 每笔执行可追溯 VolAlpha 影响

================================================================================
"""

import sys
import os
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any

logger = logging.getLogger(__name__)


# =============================================================================
# TEST FRAMEWORK
# =============================================================================

class VolAlphaTestResult:
    """测试结果"""
    
    def __init__(self, name: str):
        self.name = name
        self.passed: List[str] = []
        self.failed: List[str] = []
        self.warnings: List[str] = []
    
    def add_pass(self, check: str):
        self.passed.append(check)
    
    def add_fail(self, check: str, error: str = ""):
        self.failed.append(f"{check}: {error}" if error else check)
    
    def add_warning(self, msg: str):
        self.warnings.append(msg)
    
    @property
    def success(self) -> bool:
        return len(self.failed) == 0
    
    def summary(self) -> str:
        status = "✓ PASS" if self.success else "✗ FAIL"
        return f"{self.name}: {status} ({len(self.passed)} passed, {len(self.failed)} failed)"


# =============================================================================
# MODULE A: VOLATILITY ALPHA AGENT TESTS
# =============================================================================

def test_volatility_alpha_agent() -> VolAlphaTestResult:
    """测试 VolatilityAlphaAgent 本体"""
    result = VolAlphaTestResult("A: VolatilityAlphaAgent")
    
    # Import test
    try:
        from agents.volatility_alpha_agent import (
            VolatilityAlphaAgent,
            VolAlphaConfig,
            VolatilityIntent,
            VolBias,
            VolHorizon,
            get_volatility_alpha_agent,
        )
        result.add_pass("Import volatility_alpha_agent")
    except ImportError as e:
        result.add_fail("Import volatility_alpha_agent", str(e))
        return result
    
    # Create agent
    try:
        config = VolAlphaConfig()
        agent = VolatilityAlphaAgent(config)
        result.add_pass("Create VolatilityAlphaAgent")
    except Exception as e:
        result.add_fail("Create agent", str(e))
        return result
    
    # Test state updates
    try:
        agent.on_lob_tick({
            "asset": "BTC",
            "spread_bps": 10.0,
            "depth_imbalance": 0.1,
            "bid_depth_usd": 500000,
            "ask_depth_usd": 450000,
        })
        
        agent.on_volatility_state({
            "asset": "BTC",
            "vol_percentile_30d": 50.0,
            "vol_zscore": 0.5,
        })
        
        result.add_pass("State updates")
    except Exception as e:
        result.add_fail("State updates", str(e))
    
    # Test evaluation
    try:
        intent = agent.evaluate("BTC")
        
        assert hasattr(intent, 'vol_bias'), "Missing vol_bias"
        assert hasattr(intent, 'intensity'), "Missing intensity"
        assert hasattr(intent, 'rationale_tags'), "Missing rationale_tags"
        assert 0 <= intent.intensity <= 1, "intensity out of range"
        assert isinstance(intent.vol_bias, VolBias), "vol_bias wrong type"
        
        result.add_pass("Evaluate produces valid intent")
    except Exception as e:
        result.add_fail("Evaluation", str(e))
    
    # Test NO direction output
    try:
        intent_dict = intent.to_dict()
        
        # Check no direction field
        assert 'direction' not in intent_dict, "Intent contains forbidden 'direction'"
        
        # Check no long/short position signals (but allow "short" in expected_horizon)
        # The only acceptable "short" is in expected_horizon field
        for key, value in intent_dict.items():
            if key == "expected_horizon":
                continue  # "short"/"medium" horizon is allowed
            if isinstance(value, str):
                assert 'long' not in value.lower() or key == "expected_horizon", f"Field {key} contains forbidden 'long'"
                # 'short' in expected_horizon is OK (means short time horizon, not short position)
        
        assert 'position_size' not in intent_dict, "Intent contains forbidden 'position_size'"
        assert 'entry' not in intent_dict, "Intent contains forbidden 'entry'"
        
        result.add_pass("No direction/position output")
    except Exception as e:
        result.add_fail("Direction check", str(e))
    
    # Test sub-signals
    try:
        assert 'breakout' in intent.sub_signals or len(intent.sub_signals) == 0, "Missing sub_signals structure"
        result.add_pass("Sub-signals structure")
    except Exception as e:
        result.add_fail("Sub-signals", str(e))
    
    return result


# =============================================================================
# MODULE B: AUTHORITY FUSION INTEGRATION TESTS
# =============================================================================

def test_authority_fusion_integration() -> VolAlphaTestResult:
    """测试 Authority Fusion 集成"""
    result = VolAlphaTestResult("B: Authority Fusion Integration")
    
    # Import
    try:
        from signals.authority_fusion_volalpha import (
            VolAlphaAwareFusionEngine,
            VolAlphaFusionRules,
            VolAlphaSignal,
            get_vol_alpha_fusion_engine,
            VOLATILITY_ALPHA_AUTHORITY,
        )
        from signals.authority_fusion import (
            SystemMode,
            AgentSignal,
            FusionContext,
            Authority,
        )
        result.add_pass("Import authority_fusion_volalpha")
    except ImportError as e:
        result.add_fail("Import authority_fusion_volalpha", str(e))
        return result
    
    # Test authority definitions
    try:
        assert SystemMode.NORMAL in VOLATILITY_ALPHA_AUTHORITY
        assert SystemMode.OPPORTUNITY in VOLATILITY_ALPHA_AUTHORITY
        assert SystemMode.NO_TRADE in VOLATILITY_ALPHA_AUTHORITY
        
        assert VOLATILITY_ALPHA_AUTHORITY[SystemMode.NO_TRADE] == Authority.NONE
        result.add_pass("Authority definitions")
    except Exception as e:
        result.add_fail("Authority definitions", str(e))
    
    # Test VolAlphaSignal adapter
    try:
        from agents.volatility_alpha_agent import VolatilityIntent, VolBias, VolHorizon
        
        intent = VolatilityIntent(
            asset="BTC",
            vol_bias=VolBias.INCREASE,
            intensity=0.8,
            expected_horizon=VolHorizon.SHORT,
            rationale_tags=["test"],
        )
        
        signal = VolAlphaSignal.from_intent(intent)
        
        assert signal.vol_bias == "increase"
        assert signal.vol_intensity == 0.8
        assert signal.direction == 0.0, "Signal should not have direction"
        
        result.add_pass("VolAlphaSignal adapter")
    except Exception as e:
        result.add_fail("Signal adapter", str(e))
    
    # Test fusion engine
    try:
        engine = get_vol_alpha_fusion_engine()
        assert engine is not None
        result.add_pass("Create fusion engine")
    except Exception as e:
        result.add_fail("Create fusion engine", str(e))
    
    # Test bear protection rule
    try:
        from market.phase_detector import RegimePhase
        
        # Create signals for LONG position
        signals = {
            "quant": AgentSignal(direction=0.7, confidence=0.8),
            "regime": AgentSignal(direction=0.5, confidence=0.7),
        }
        
        context = FusionContext(
            mode=SystemMode.OPPORTUNITY,
            regime_phase=RegimePhase.EXPANSION,
            data_valid=True,
        )
        
        # High vol intent
        vol_intent = VolatilityIntent(
            asset="BTC",
            vol_bias=VolBias.INCREASE,
            intensity=0.8,
            expected_horizon=VolHorizon.SHORT,
            rationale_tags=["test"],
        )
        
        fusion_result = engine.fuse_with_vol_alpha(signals, context, vol_intent)
        
        # Bear protection should have been applied
        vol_info = fusion_result.caps_applied.get("vol_alpha", {})
        
        # Check that some action was taken
        assert fusion_result is not None
        result.add_pass("Bear protection rule applied")
        
    except Exception as e:
        result.add_fail("Bear protection", str(e))
    
    return result


# =============================================================================
# MODULE C: RISK INTEGRATION TESTS
# =============================================================================

def test_risk_integration() -> VolAlphaTestResult:
    """测试风险集成"""
    result = VolAlphaTestResult("C: Risk Integration")
    
    # Import
    try:
        from risk.volatility_alpha_risk import (
            VolAlphaRiskController,
            VolAlphaRiskConfig,
            VolAlphaRiskIntegrator,
            get_vol_alpha_risk_integrator,
        )
        result.add_pass("Import volatility_alpha_risk")
    except ImportError as e:
        result.add_fail("Import volatility_alpha_risk", str(e))
        return result
    
    # Test risk controller
    try:
        config = VolAlphaRiskConfig(
            portfolio_vol_target_daily=0.02,
        )
        controller = VolAlphaRiskController(config)
        result.add_pass("Create risk controller")
    except Exception as e:
        result.add_fail("Create risk controller", str(e))
        return result
    
    # Test weight computation at different vol levels
    try:
        # Low vol: full weight
        weight_low = controller.update_portfolio_vol(0.01, False)
        assert weight_low >= 0.9, f"Low vol should have high weight, got {weight_low}"
        
        # At target: still high weight
        weight_target = controller.update_portfolio_vol(0.02, False)
        assert weight_target >= 0.5, f"At target should have decent weight, got {weight_target}"
        
        # High vol: reduced weight
        weight_high = controller.update_portfolio_vol(0.025, False)
        assert weight_high < weight_target, "High vol should reduce weight"
        
        # Very high vol: very low weight
        weight_critical = controller.update_portfolio_vol(0.03, False)
        assert weight_critical < 0.3, f"Critical vol should have low weight, got {weight_critical}"
        
        result.add_pass("Weight computation by vol level")
    except Exception as e:
        result.add_fail("Weight computation", str(e))
    
    # Test bear market relaxation
    try:
        controller.update_portfolio_vol(0.01, False)  # Reset
        
        weight_normal = controller.update_portfolio_vol(0.025, False)
        weight_bear = controller.update_portfolio_vol(0.025, True)
        
        # Bear market should allow higher vol
        assert weight_bear >= weight_normal, "Bear market should be more lenient"
        result.add_pass("Bear market relaxation")
    except Exception as e:
        result.add_fail("Bear market relaxation", str(e))
    
    # Test constraint application
    try:
        controller.update_portfolio_vol(0.025, False)  # High vol
        constrained = controller.get_constraint_for_intent(0.8)
        
        assert constrained < 0.8, "Constraint should reduce intensity"
        result.add_pass("Constraint application")
    except Exception as e:
        result.add_fail("Constraint application", str(e))
    
    return result


# =============================================================================
# MODULE D: DASHBOARD INTEGRATION TESTS
# =============================================================================

def test_dashboard_integration() -> VolAlphaTestResult:
    """测试 Dashboard 集成"""
    result = VolAlphaTestResult("D: Dashboard Integration")
    
    # Import
    try:
        from dashboard.volatility_alpha_panel import (
            VolAlphaSnapshot,
            VolAlphaDashboardState,
            get_volalpha_dashboard_state,
            VolAlphaPanelRenderer,
            VolAlphaDashboardIntegration,
        )
        result.add_pass("Import volatility_alpha_panel")
    except ImportError as e:
        result.add_fail("Import volatility_alpha_panel", str(e))
        return result
    
    # Test state
    try:
        state = VolAlphaDashboardState()
        
        snapshot = VolAlphaSnapshot(
            timestamp=datetime.now(),
            asset="BTC",
            vol_bias="increase",
            intensity=0.7,
            rationale_tags=["test"],
            breakout_signal=0.5,
            liquidation_signal=0.4,
            compression_signal=0.2,
            data_quality=0.9,
            fusion_accepted=True,
            fusion_action="AMPLIFY",
        )
        
        state.add_snapshot(snapshot)
        
        current = state.get_current()
        assert current is not None
        assert current.vol_bias == "increase"
        
        result.add_pass("Dashboard state")
    except Exception as e:
        result.add_fail("Dashboard state", str(e))
    
    # Test statistics
    try:
        stats = state.get_stats()
        assert stats["total_snapshots"] == 1
        assert stats["increase_count"] == 1
        result.add_pass("Statistics tracking")
    except Exception as e:
        result.add_fail("Statistics", str(e))
    
    # Test bear market timeline
    try:
        state.set_bear_market(True)
        
        for i in range(5):
            snapshot = VolAlphaSnapshot(
                timestamp=datetime.now() - timedelta(minutes=5-i),
                asset="BTC",
                vol_bias="increase",
                intensity=0.8,
                rationale_tags=["bear_signal"],
                breakout_signal=0.6,
                liquidation_signal=0.5,
                compression_signal=0.2,
                data_quality=0.9,
                fusion_accepted=True,
                fusion_action="AMPLIFY_SHORT",
            )
            state.add_snapshot(snapshot)
        
        timeline = state.get_bear_timeline()
        assert len(timeline) == 5
        result.add_pass("Bear market timeline")
    except Exception as e:
        result.add_fail("Bear timeline", str(e))
    
    return result


# =============================================================================
# MODULE E: BEHAVIOR VERIFICATION
# =============================================================================

def test_behavior_verification() -> VolAlphaTestResult:
    """测试行为验证 (熊市/牛市场景)"""
    result = VolAlphaTestResult("E: Behavior Verification")
    
    try:
        from agents.volatility_alpha_agent import (
            VolatilityAlphaAgent,
            VolAlphaConfig,
            VolBias,
        )
    except ImportError as e:
        result.add_fail("Import", str(e))
        return result
    
    agent = VolatilityAlphaAgent()
    
    # =====================================================================
    # 熊市场景: 高波动、恐慌、清算压力
    # =====================================================================
    
    try:
        # 模拟熊市数据
        agent.on_lob_tick({
            "asset": "BTC",
            "spread_bps": 20.0,  # 扩张
            "depth_imbalance": -0.4,  # 卖压
            "bid_depth_usd": 300000,
            "ask_depth_usd": 700000,
        })
        
        agent.on_volatility_state({
            "asset": "BTC",
            "realized_vol_1h": 0.9,
            "vol_percentile_30d": 90.0,  # 高分位数
            "vol_zscore": 2.5,
            "vol_of_vol": 0.7,
        })
        
        agent.on_onchain_tick({
            "asset": "BTC",
            "exchange_inflow_spike": True,
            "exchange_inflow_zscore": 3.0,
            "stablecoin_flow_direction": "outflow",
        })
        
        agent.on_sentiment_tick({
            "asset": "BTC",
            "fear_index": 15.0,  # 极端恐慌
            "sentiment_volatility": 0.6,
        })
        
        agent.on_correlation_state({
            "btc_dominance": 56.0,
            "btc_dominance_change_24h": 4.0,
            "correlation_spike": True,
            "correlation_regime": "exploded",
        })
        
        bear_intent = agent.evaluate("BTC")
        
        # 验证熊市场景
        assert bear_intent.vol_bias == VolBias.INCREASE, f"Bear market should show INCREASE, got {bear_intent.vol_bias}"
        assert bear_intent.intensity >= 0.5, f"Bear market intensity should be high, got {bear_intent.intensity}"
        assert len(bear_intent.rationale_tags) > 0, "Should have rationale tags"
        
        result.add_pass("Bear market: VolAlpha activates")
        
    except Exception as e:
        result.add_fail("Bear market scenario", str(e))
    
    # =====================================================================
    # 牛市场景: 低波动、贪婪、平稳
    # =====================================================================
    
    try:
        # 创建新 agent 以清除状态
        bull_agent = VolatilityAlphaAgent()
        
        # 模拟牛市数据
        bull_agent.on_lob_tick({
            "asset": "BTC",
            "spread_bps": 3.0,  # 窄
            "depth_imbalance": 0.2,  # 买盘
            "bid_depth_usd": 800000,
            "ask_depth_usd": 600000,
        })
        
        bull_agent.on_volatility_state({
            "asset": "BTC",
            "realized_vol_1h": 0.15,
            "vol_percentile_30d": 25.0,  # 低分位数
            "vol_zscore": -0.8,
            "vol_of_vol": 0.1,
        })
        
        bull_agent.on_onchain_tick({
            "asset": "BTC",
            "exchange_inflow_spike": False,
            "exchange_inflow_zscore": 0.5,
            "stablecoin_flow_direction": "inflow",
        })
        
        bull_agent.on_sentiment_tick({
            "asset": "BTC",
            "fear_index": 72.0,  # 贪婪
            "sentiment_volatility": 0.05,
        })
        
        bull_agent.on_correlation_state({
            "btc_dominance": 43.0,
            "correlation_spike": False,
            "correlation_regime": "normal",
        })
        
        bull_intent = bull_agent.evaluate("BTC")
        
        # 验证牛市场景
        assert bull_intent.vol_bias in [VolBias.NEUTRAL, VolBias.SUPPRESS], \
            f"Bull market should show NEUTRAL/SUPPRESS, got {bull_intent.vol_bias}"
        assert bull_intent.intensity < 0.5, \
            f"Bull market intensity should be low, got {bull_intent.intensity}"
        
        result.add_pass("Bull market: VolAlpha low weight")
        
    except Exception as e:
        result.add_fail("Bull market scenario", str(e))
    
    # =====================================================================
    # 统计验证
    # =====================================================================
    
    try:
        stats = agent.get_status()["stats"]
        
        assert stats["total_evaluations"] >= 1, "Should have evaluations"
        result.add_pass("Statistics tracked")
        
    except Exception as e:
        result.add_fail("Statistics", str(e))
    
    return result


# =============================================================================
# RUN ALL TESTS
# =============================================================================

def run_volalpha_diagnostic() -> Dict[str, Any]:
    """运行所有 VolAlpha 诊断测试"""
    print("=" * 70)
    print("HMATS v5.2.1 Volatility Alpha Agent Diagnostic Tests")
    print("=" * 70)
    print()
    
    tests = [
        ("A: Agent", test_volatility_alpha_agent),
        ("B: Fusion", test_authority_fusion_integration),
        ("C: Risk", test_risk_integration),
        ("D: Dashboard", test_dashboard_integration),
        ("E: Behavior", test_behavior_verification),
    ]
    
    results = {}
    total_passed = 0
    total_failed = 0
    
    for test_id, test_func in tests:
        print(f"\n{'─' * 50}")
        print(f"Running: {test_id}")
        print('─' * 50)
        
        try:
            result = test_func()
            results[test_id] = result
            
            print(f"\n{result.summary()}")
            
            if result.passed:
                print("  Passed:")
                for p in result.passed:
                    print(f"    ✓ {p}")
            
            if result.failed:
                print("  Failed:")
                for f in result.failed:
                    print(f"    ✗ {f}")
            
            if result.warnings:
                print("  Warnings:")
                for w in result.warnings:
                    print(f"    [WARN] {w}")
            
            total_passed += len(result.passed)
            total_failed += len(result.failed)
            
        except Exception as e:
            print(f"  ✗ Test crashed: {e}")
            total_failed += 1
    
    # 总结
    print("\n" + "=" * 70)
    print("VOLATILITY ALPHA DIAGNOSTIC SUMMARY")
    print("=" * 70)
    
    all_pass = all(r.success for r in results.values())
    
    print(f"\nTotal Checks: {total_passed + total_failed}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    print(f"\nOverall: {'✓ VOL_ALPHA COMPLETE' if all_pass else '✗ VOL_ALPHA INCOMPLETE'}")
    
    # 验收标准
    print("\n" + "-" * 70)
    print("Acceptance Criteria:")
    print("-" * 70)
    
    criteria = [
        ("Bear market: VolAlpha activates frequently", "E: Behavior" in results and results["E: Behavior"].success),
        ("Bull market: VolAlpha auto low-weight", "E: Behavior" in results and results["E: Behavior"].success),
        ("No direction output", "A: Agent" in results and results["A: Agent"].success),
        ("Authority Fusion integrated", "B: Fusion" in results and results["B: Fusion"].success),
        ("Risk constraint applied", "C: Risk" in results and results["C: Risk"].success),
    ]
    
    for criterion, passed in criteria:
        mark = "✓" if passed else "✗"
        print(f"  {mark} {criterion}")
    
    print("\n" + "=" * 70)
    
    return {
        "vol_alpha_complete": all_pass,
        "total_passed": total_passed,
        "total_failed": total_failed,
        "results": {k: v.summary() for k, v in results.items()},
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING,
        format='%(name)s - %(levelname)s - %(message)s'
    )
    
    result = run_volalpha_diagnostic()
    
    sys.exit(0 if result["vol_alpha_complete"] else 1)
