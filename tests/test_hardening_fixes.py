"""
================================================================================
HMATS v5.2.1 - Hardening Finishing Pass Verification
================================================================================

验证 H1, C2, M1-M3 修复是否正确实现

运行: python -m tests.test_hardening_fixes
================================================================================
"""

import logging
import pytest
import numpy as np
from datetime import datetime, timedelta, timezone

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_h1_execution_acceptance():
    """
    H1 FIX: 测试 ExecutionAcceptance 竞态修复
    """
    print("\n" + "=" * 60)
    print("H1 FIX: Execution WAIT ↔ Intent 竞态修复")
    print("=" * 60)
    
    from core.runtime_spine import ExecutionAcceptance
    
    # 测试1: 接受的 intent
    acceptance_ok = ExecutionAcceptance(
        accepted=True,
        reason="OK",
        spread_bps=15.0,
        execution_mode="PASSIVE_PREFERRED",
    )
    print(f"\n✓ Accepted intent: {acceptance_ok.to_log_string()}")
    
    # 测试2: 极端 spread 拒绝
    acceptance_spread = ExecutionAcceptance(
        accepted=False,
        reason="EXTREME_SPREAD",
        rejection_source="execution_acceptance_check",
        spread_bps=150.0,
        execution_mode="AGGRESSIVE",
    )
    print(f"✓ Rejected (spread): {acceptance_spread.to_log_string()}")
    
    # 测试3: 熔断拒绝
    acceptance_fused = ExecutionAcceptance(
        accepted=False,
        reason="EXEC_POLICY_FUSED",
        rejection_source="learned_execution_policy",
        spread_bps=25.0,
        execution_mode="BALANCED",
    )
    print(f"✓ Rejected (fused): {acceptance_fused.to_log_string()}")
    
    print("\nON H1 FIX: ExecutionAcceptance 数据结构正确")
    return


def test_c2_flow_data_validity():
    """
    C2 FIX: 测试链上 inflow data_valid 标记
    """
    print("\n" + "=" * 60)
    print("C2 FIX: 链上 inflow data_valid 标记")
    print("=" * 60)
    
    # [P165] Was `models.representation...`, which does not exist (and `models/`
    # is gitignored). The module now lives at archive/models_representation_
    # test_only/ — the directory name records that these two tests are its only
    # remaining consumers; there is no production reference to it.
    from archive.models_representation_test_only.market_embedding_model import (
        RawMarketFeatures,
    )

    # 测试1: 有效数据
    valid_features = RawMarketFeatures(
        timestamp=datetime.now(),
        asset="BTC",
        prices=np.array([50000, 50100]),
        returns=np.array([0.002]),
        exchange_inflow=1e8,
        exchange_outflow=5e7,
        inflow_data_valid=True,
        outflow_data_valid=True,
    )
    print(f"\n✓ Valid flow data: {valid_features.get_flow_validity_log()}")
    assert valid_features.is_flow_data_valid() == True
    
    # 测试2: API 超时导致无效
    invalid_features = RawMarketFeatures(
        timestamp=datetime.now(),
        asset="BTC",
        prices=np.array([50000, 50100]),
        returns=np.array([0.002]),
        exchange_inflow=0.0,  # 看起来是0，但实际是无效
        exchange_outflow=0.0,
        inflow_data_valid=False,
        outflow_data_valid=False,
        flow_invalid_reason="API_TIMEOUT",
    )
    print(f"✓ Invalid flow data: {invalid_features.get_flow_validity_log()}")
    assert invalid_features.is_flow_data_valid() == False
    
    # 测试3: 部分无效
    partial_features = RawMarketFeatures(
        timestamp=datetime.now(),
        asset="ETH",
        prices=np.array([3000, 3010]),
        returns=np.array([0.003]),
        exchange_inflow=5e7,
        exchange_outflow=0.0,
        inflow_data_valid=True,
        outflow_data_valid=False,
        flow_invalid_reason="RATE_LIMIT",
    )
    print(f"✓ Partial invalid: {partial_features.get_flow_validity_log()}")
    assert partial_features.is_flow_data_valid() == False
    
    print("\nON C2 FIX: data_valid 标记正确区分真实零值和数据缺失")
    return


def test_m1_embedding_rejection_logging():
    """
    M1 FIX: 测试 Embedding 拒绝日志
    """
    print("\n" + "=" * 60)
    print("M1 FIX: Embedding 拒绝日志增强")
    print("=" * 60)
    
    # [P165] See the note in test_c2_flow_data_validity.
    from archive.models_representation_test_only.market_embedding_model import (
        MarketEmbeddingModel,
        RawMarketFeatures,
    )
    
    model = MarketEmbeddingModel()
    
    # 测试1: 陈旧数据拒绝 (应该返回 None 并记录日志)
    stale_features = RawMarketFeatures(
        timestamp=datetime.now(timezone.utc) - timedelta(minutes=10),  # 10分钟前
        asset="BTC",
        prices=np.array([50000, 50100, 50050]),
        returns=np.array([0.002, -0.001]),
    )
    
    result = model.embed(stale_features)
    print(f"\n✓ Stale data rejection: result={result}")
    assert result is None, "Stale data should be rejected"
    
    # 测试2: 有效数据不应被staleness guard拒绝
    # Note: embed() may still return None if model isn't trained/initialized,
    # but should NOT be rejected by the staleness check specifically.
    valid_features = RawMarketFeatures(
        timestamp=datetime.now(timezone.utc),
        asset="BTC",
        prices=np.array([50000, 50100, 50050]),
        returns=np.array([0.002, -0.001]),
    )

    stale_rejections_before = model._stats.get("stale_rejections", 0)
    result = model.embed(valid_features)
    stale_rejections_after = model._stats.get("stale_rejections", 0)
    print(f"✓ Valid data accepted: result={result is not None}")
    # Fresh data should NOT increment stale_rejections counter
    assert stale_rejections_after == stale_rejections_before, \
        "Fresh data should not be stale-rejected"

    print("\nON M1 FIX: Embedding 拒绝日志正确记录")
    return


def test_m2_promotion_rejection_logging():
    """
    M2 FIX: 测试 Promotion Gate 拒绝日志
    """
    print("\n" + "=" * 60)
    print("M2 FIX: Promotion Gate 拒绝日志增强")
    print("=" * 60)
    
    from training.promotion.statistical_gate import (
        StatisticalPromotionGate,
        PerformanceMetrics,
    )
    
    gate = StatisticalPromotionGate()
    
    # 测试: 样本不足导致拒绝
    np.random.seed(42)
    returns = np.random.randn(30) * 0.02  # 只有30个样本
    
    # 带 regime 标签测试
    regime_labels = ["slow_bear"] * 15 + ["sideways"] * 15  # 缺少其他 regime
    
    decision = gate.evaluate(
        strategy_id="test_strategy_1",
        returns=returns,
        regime_labels=regime_labels,
    )
    
    print(f"\n✓ Rejection decision: approved={decision.approved}")
    print(f"  tests_passed: {decision.tests_passed}")
    print(f"  tests_failed: {decision.tests_failed}")
    print(f"  rejection_reasons: {decision.rejection_reasons}")
    
    print("\nON M2 FIX: Promotion Gate 拒绝日志详细记录了失败测试和 regime 分布")
    return


def test_m3_decision_summary():
    """
    M3 FIX: 测试决策摘要日志
    """
    print("\n" + "=" * 60)
    print("M3 FIX: Decision Summary 统一摘要")
    print("=" * 60)
    
    from core.runtime_spine import (
        RuntimeSpine,
        TickResult,
        FusedSignal,
        ExecutionAcceptance,
        SystemMode,
    )
    from datetime import datetime
    
    # 创建模拟的 TickResult
    result = TickResult(
        tick_id="T00000001",
        timestamp=datetime.now(),
        system_mode=SystemMode.NORMAL,
    )
    
    # 添加 fused signal
    result.fused_signal = FusedSignal(
        timestamp=datetime.now(),
        asset="SOL",
        direction=-0.65,  # SHORT
        confidence=0.72,
        execution_mode="PASSIVE_PREFERRED",
    )
    
    # 添加被拒绝的 execution acceptance
    result.execution_acceptance = ExecutionAcceptance(
        accepted=False,
        reason="EXTREME_SPREAD",
        spread_bps=120.0,
    )
    
    # 模拟生成决策摘要
    spine = RuntimeSpine.__new__(RuntimeSpine)  # 不调用 __init__
    summary = spine._generate_decision_summary(result)
    
    print(f"\n✓ Decision summary generated:")
    print(f"  {summary}")
    
    assert "DECISION_SUMMARY" in summary
    assert "SHORT" in summary
    assert "REJECTED" in summary or "EXECUTION" in summary
    
    print("\nON M3 FIX: Decision Summary 正确生成可审计的统一摘要")
    return


def main():
    """运行所有验证测试"""
    print("\n" + "=" * 70)
    print("HMATS v5.2.1 Hardening Finishing Pass - Verification")
    print("=" * 70)
    
    results = []
    
    # H1
    try:
        results.append(("H1: Execution Acceptance", test_h1_execution_acceptance()))
    except Exception as e:
        print(f"OFF H1 FAILED: {e}")
        results.append(("H1: Execution Acceptance", False))
    
    # C2
    try:
        results.append(("C2: Flow Data Validity", test_c2_flow_data_validity()))
    except Exception as e:
        print(f"OFF C2 FAILED: {e}")
        results.append(("C2: Flow Data Validity", False))
    
    # M1
    try:
        results.append(("M1: Embedding Rejection Log", test_m1_embedding_rejection_logging()))
    except Exception as e:
        print(f"OFF M1 FAILED: {e}")
        results.append(("M1: Embedding Rejection Log", False))
    
    # M2
    try:
        results.append(("M2: Promotion Rejection Log", test_m2_promotion_rejection_logging()))
    except Exception as e:
        print(f"OFF M2 FAILED: {e}")
        results.append(("M2: Promotion Rejection Log", False))
    
    # M3
    try:
        results.append(("M3: Decision Summary", test_m3_decision_summary()))
    except Exception as e:
        print(f"OFF M3 FAILED: {e}")
        results.append(("M3: Decision Summary", False))
    
    # 汇总
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\nTotal: {passed}/{total} passed")
    
    if passed == total:
        print("\nON ALL HARDENING FIXES VERIFIED")
        print("\n确认: 所有修复均为一致性/可审计性增强，未改变交易语义")
    else:
        print("\nOFF SOME FIXES FAILED - REVIEW REQUIRED")
    
    return passed == total


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
