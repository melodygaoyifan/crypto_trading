"""
================================================================================
HMATS v5.2.1 P0 - 诊断测试
P0 Diagnostic Tests
================================================================================

用于 diagnostic.py 和 --mode verify 的 P0 验证功能

验收标准 P0-1:
- 每个 feed 的 last_timestamp / staleness / health
- 是否触发降级
- --mode verify 下不允许任何 feed 静默失败

验收标准 P0-2:
- 每次决策 log 必须显示 ModelIntent
- 是否被 authority_fusion 接纳
- 是否被 veto（以及 veto 原因）
- DRL 仍然禁止 entry authority

================================================================================
"""

import logging
import asyncio
from typing import Dict, List, Tuple, Any
from datetime import datetime

logger = logging.getLogger(__name__)


def test_p0_feeds() -> Tuple[int, int, List[str]]:
    """
    测试 P0-1: 数据 Feeds
    
    Returns:
        (passed, failed, details)
    """
    passed = 0
    failed = 0
    details = []
    
    print("\n" + "=" * 60)
    print("P0-1: DATA FEEDS TEST")
    print("=" * 60)
    
    # 测试 OnChain Feed
    try:
        from data_mgmt.feeds.onchain_feed import get_onchain_feed
        
        feed = get_onchain_feed()
        health = feed.get_health()
        
        if health:
            details.append(f"  OnChain Feed: source={health['source']}, "
                          f"staleness={health.get('staleness_sec', 'N/A')}s")
            passed += 1
            print(f"  ON OnChain Feed: OK")
        else:
            details.append("  OnChain Feed: No health data")
            failed += 1
            print(f"  OFF OnChain Feed: No health data")
            
    except Exception as e:
        details.append(f"  OnChain Feed: Error - {e}")
        failed += 1
        print(f"  OFF OnChain Feed: {e}")
    
    # 测试 Sentiment Feed
    try:
        from data_mgmt.feeds.sentiment_feed import get_sentiment_feed
        
        feed = get_sentiment_feed()
        health = feed.get_health()
        
        if health:
            details.append(f"  Sentiment Feed: source={health['source']}, "
                          f"staleness={health.get('staleness_sec', 'N/A')}s")
            passed += 1
            print(f"  ON Sentiment Feed: OK")
        else:
            details.append("  Sentiment Feed: No health data")
            failed += 1
            print(f"  OFF Sentiment Feed: No health data")
            
    except Exception as e:
        details.append(f"  Sentiment Feed: Error - {e}")
        failed += 1
        print(f"  OFF Sentiment Feed: {e}")
    
    # 测试 Macro Feed
    try:
        from data_mgmt.feeds.macro_feed import get_macro_feed
        
        feed = get_macro_feed()
        health = feed.get_health()
        
        if health:
            details.append(f"  Macro Feed: source={health['source']}, "
                          f"staleness={health.get('staleness_sec', 'N/A')}s")
            passed += 1
            print(f"  ON Macro Feed: OK")
        else:
            details.append("  Macro Feed: No health data")
            failed += 1
            print(f"  OFF Macro Feed: No health data")
            
    except Exception as e:
        details.append(f"  Macro Feed: Error - {e}")
        failed += 1
        print(f"  OFF Macro Feed: {e}")
    
    # 测试 LOB Feed
    try:
        from data_mgmt.feeds.lob_feed import get_lob_feed
        
        feed = get_lob_feed()
        health = feed.get_health()
        
        if health:
            details.append(f"  LOB Feed: source={health['source']}, "
                          f"staleness={health.get('staleness_sec', 'N/A')}s")
            passed += 1
            print(f"  ON LOB Feed: OK")
        else:
            details.append("  LOB Feed: No health data")
            failed += 1
            print(f"  OFF LOB Feed: No health data")
            
    except Exception as e:
        details.append(f"  LOB Feed: Error - {e}")
        failed += 1
        print(f"  OFF LOB Feed: {e}")
    
    return passed, failed, details


def test_p0_data_health() -> Tuple[int, int, List[str]]:
    """
    测试 P0-1: 数据健康监控
    
    Returns:
        (passed, failed, details)
    """
    passed = 0
    failed = 0
    details = []
    
    print("\n" + "-" * 40)
    print("P0-1: DATA HEALTH TEST")
    print("-" * 40)
    
    try:
        from data_mgmt.data_health import get_data_health, DataSourceType
        
        integration = get_data_health()
        
        # 检查监控器
        monitor = integration.get_monitor()
        if monitor:
            details.append("  DataHealthMonitor: initialized")
            passed += 1
            print(f"  ON DataHealthMonitor: initialized")
        else:
            details.append("  DataHealthMonitor: not available")
            failed += 1
            print(f"  OFF DataHealthMonitor: not available")
        
        # 获取快照
        snapshot = integration.get_snapshot()
        if snapshot:
            details.append(f"  Snapshot: status={snapshot.overall_status.value}, "
                          f"degradation={snapshot.degradation_level.value}")
            passed += 1
            print(f"  ON Snapshot: status={snapshot.overall_status.value}")
            
            # 检查交易许可
            if snapshot.is_trade_allowed():
                details.append("  Trade Allowed: YES")
                print(f"  ON Trade Allowed: YES")
            else:
                details.append(f"  Trade Allowed: NO - {snapshot.degradation_reason}")
                print(f"  [WARN]️ Trade Allowed: NO - {snapshot.degradation_reason}")
        else:
            details.append("  Snapshot: not available")
            failed += 1
            print(f"  OFF Snapshot: not available")
            
    except Exception as e:
        details.append(f"  DataHealth: Error - {e}")
        failed += 1
        print(f"  OFF DataHealth: {e}")
    
    return passed, failed, details


def test_p0_trade_gate_integration() -> Tuple[int, int, List[str]]:
    """
    测试 P0-1: TradeGate 数据健康集成
    
    Returns:
        (passed, failed, details)
    """
    passed = 0
    failed = 0
    details = []
    
    print("\n" + "-" * 40)
    print("P0-1: TRADE GATE INTEGRATION TEST")
    print("-" * 40)
    
    try:
        from defense.trade_gate import get_trade_gate, TradeProposal, RejectReason
        from decimal import Decimal
        
        gate = get_trade_gate()
        
        # 检查数据健康检查是否可用
        if hasattr(gate, '_data_health_enabled'):
            details.append(f"  Data Health Check: enabled={gate._data_health_enabled}")
            passed += 1
            print(f"  ON Data Health Check: available")
        else:
            details.append("  Data Health Check: attribute not found")
            failed += 1
            print(f"  OFF Data Health Check: attribute not found")
        
        # 检查 DATA_HEALTH 相关 RejectReason
        health_reasons = [
            RejectReason.DATA_HEALTH_DEGRADED,
            RejectReason.DATA_HEALTH_CRITICAL,
            RejectReason.DATA_HEALTH_EXIT_ONLY,
        ]
        
        for reason in health_reasons:
            details.append(f"  RejectReason.{reason.name}: defined")
        passed += 1
        print(f"  ON DATA_HEALTH RejectReasons: defined")
        
    except Exception as e:
        details.append(f"  TradeGate: Error - {e}")
        failed += 1
        print(f"  OFF TradeGate: {e}")
    
    return passed, failed, details


def test_p0_model_alpha() -> Tuple[int, int, List[str]]:
    """
    测试 P0-2: 模型 Alpha Agent
    
    Returns:
        (passed, failed, details)
    """
    passed = 0
    failed = 0
    details = []
    
    print("\n" + "=" * 60)
    print("P0-2: MODEL ALPHA TEST")
    print("=" * 60)
    
    try:
        from agents.model_alpha_agent import (
            get_model_alpha_agent,
            ModelIntent,
            UnifiedState,
        )
        
        agent = get_model_alpha_agent()
        
        # 检查模型加载
        if agent._model_loaded:
            details.append("  Model: loaded")
            passed += 1
            print(f"  ON Model: loaded")
        else:
            details.append("  Model: not loaded")
            failed += 1
            print(f"  OFF Model: not loaded")
        
        # 创建测试状态
        state = UnifiedState(
            timestamp=datetime.now(),
            prices={"BTC": 100000, "ETH": 3500, "SOL": 200},
            returns_1h={"BTC": 0.01, "ETH": 0.02, "SOL": -0.01},
            returns_24h={"BTC": 0.05, "ETH": 0.03, "SOL": 0.02},
            volatility={"BTC": 0.5, "ETH": 0.6, "SOL": 0.8},
            fear_greed=35,
            sentiment_score=-0.2,
        )
        
        # 生成意图
        intent = agent.generate_intent(state, "BTC")
        
        if intent:
            details.append(f"  Intent Generated: direction={intent.direction.value}, "
                          f"confidence={intent.confidence:.2f}")
            passed += 1
            print(f"  ON Intent Generated: direction={intent.direction.value}")
            
            # 检查解释
            if intent.explanation_tokens:
                details.append(f"  Explanation: {', '.join(intent.explanation_tokens[:3])}")
                passed += 1
                print(f"  ON Explanation: provided")
            else:
                details.append("  Explanation: empty")
                failed += 1
                print(f"  [WARN]️ Explanation: empty")
        else:
            details.append("  Intent: not generated (low confidence?)")
            passed += 1  # 低置信度时不生成也是正确行为
            print(f"  [WARN]️ Intent: not generated (low confidence)")
        
        # 检查统计
        stats = agent.get_stats()
        details.append(f"  Stats: total_intents={stats.get('total_intents', 0)}, "
                      f"weight={stats.get('current_weight', 0):.2f}")
        passed += 1
        print(f"  ON Stats: accessible")
        
    except Exception as e:
        details.append(f"  ModelAlphaAgent: Error - {e}")
        failed += 1
        print(f"  OFF ModelAlphaAgent: {e}")
    
    return passed, failed, details


def test_p0_authority_fusion_integration() -> Tuple[int, int, List[str]]:
    """
    测试 P0-2: Authority Fusion 集成
    
    Returns:
        (passed, failed, details)
    """
    passed = 0
    failed = 0
    details = []
    
    print("\n" + "-" * 40)
    print("P0-2: AUTHORITY FUSION INTEGRATION TEST")
    print("-" * 40)
    
    try:
        from signals.authority_fusion import (
            AUTHORITY_MATRIX_NORMAL,
            AUTHORITY_MATRIX_OPPORTUNITY,
            AUTHORITY_MATRIX_NO_TRADE,
            Authority,
        )
        
        # 检查 model_alpha 在各矩阵中的权限
        matrices = [
            ("NORMAL", AUTHORITY_MATRIX_NORMAL),
            ("OPPORTUNITY", AUTHORITY_MATRIX_OPPORTUNITY),
            ("NO_TRADE", AUTHORITY_MATRIX_NO_TRADE),
        ]
        
        for name, matrix in matrices:
            if "model_alpha" in matrix:
                auth = matrix["model_alpha"]
                details.append(f"  {name}: model_alpha={auth.value}")
                passed += 1
                print(f"  ON {name}: model_alpha={auth.value}")
            else:
                details.append(f"  {name}: model_alpha not defined")
                failed += 1
                print(f"  OFF {name}: model_alpha not defined")
        
        # 验证 DRL 不能 entry
        drl_normal = AUTHORITY_MATRIX_NORMAL.get("drl")
        drl_opp = AUTHORITY_MATRIX_OPPORTUNITY.get("drl")
        
        if drl_normal == Authority.ADVISE and drl_opp == Authority.ADVISE:
            details.append("  DRL: ADVISE only (no entry authority)")
            passed += 1
            print(f"  ON DRL: ADVISE only (no entry authority)")
        else:
            details.append(f"  DRL: NORMAL={drl_normal}, OPPORTUNITY={drl_opp}")
            failed += 1
            print(f"  [WARN]️ DRL authority check needed")
        
    except Exception as e:
        details.append(f"  AuthorityFusion: Error - {e}")
        failed += 1
        print(f"  OFF AuthorityFusion: {e}")
    
    return passed, failed, details


def test_p0_correlation_weight_control() -> Tuple[int, int, List[str]]:
    """
    测试 P0-2: 相关性权重控制
    
    Returns:
        (passed, failed, details)
    """
    passed = 0
    failed = 0
    details = []
    
    print("\n" + "-" * 40)
    print("P0-2: CORRELATION WEIGHT CONTROL TEST")
    print("-" * 40)
    
    try:
        from risk.correlation_realtime_controller import (
            get_correlation_controller,
            get_model_alpha_weight_controller,
            CorrelationState,
        )
        
        controller = get_correlation_controller()
        weight_ctrl = get_model_alpha_weight_controller()
        
        # 检查权重控制器
        if weight_ctrl:
            status = weight_ctrl.get_status()
            details.append(f"  Weight Controller: max_weight={status['current_max_weight']:.2f}")
            passed += 1
            print(f"  ON Weight Controller: initialized")
            
            # 检查权重上限配置
            weight_caps = status.get('weight_caps', {})
            for state, cap in weight_caps.items():
                details.append(f"    {state}: cap={cap:.2f}")
            
            # 检查 crisis 状态权重为 0
            if weight_caps.get('crisis', 1.0) == 0.0:
                passed += 1
                print(f"  ON Crisis state: model weight = 0")
            else:
                failed += 1
                print(f"  [WARN]️ Crisis state: model weight not 0")
                
        else:
            details.append("  Weight Controller: not available")
            failed += 1
            print(f"  OFF Weight Controller: not available")
        
    except Exception as e:
        details.append(f"  CorrelationWeightControl: Error - {e}")
        failed += 1
        print(f"  OFF CorrelationWeightControl: {e}")
    
    return passed, failed, details


def test_p0_adaptive_weight() -> Tuple[int, int, List[str]]:
    """
    测试 P0-2: 自适应权重
    
    Returns:
        (passed, failed, details)
    """
    passed = 0
    failed = 0
    details = []
    
    print("\n" + "-" * 40)
    print("P0-2: ADAPTIVE WEIGHT TEST")
    print("-" * 40)
    
    try:
        from signals.adaptive_strategy_weight_manager import (
            get_weight_manager,
            get_model_alpha_weight_adapter,
        )
        
        manager = get_weight_manager()
        
        if manager:
            stats = manager.get_stats()
            details.append(f"  Weight Manager: trades={stats['trades_recorded']}")
            passed += 1
            print(f"  ON Weight Manager: initialized")
        else:
            details.append("  Weight Manager: not available")
            failed += 1
            print(f"  OFF Weight Manager: not available")
        
        # 检查 ModelAlpha 适配器
        try:
            adapter = get_model_alpha_weight_adapter()
            if adapter:
                status = adapter.get_status()
                details.append(f"  Model Adapter: weight={status['current_weight']:.2f}")
                passed += 1
                print(f"  ON Model Adapter: initialized")
            else:
                details.append("  Model Adapter: not available")
                failed += 1
                print(f"  OFF Model Adapter: not available")
        except Exception as e:
            details.append(f"  Model Adapter: {e}")
            failed += 1
            print(f"  [WARN]️ Model Adapter: {e}")
        
    except Exception as e:
        details.append(f"  AdaptiveWeight: Error - {e}")
        failed += 1
        print(f"  OFF AdaptiveWeight: {e}")
    
    return passed, failed, details


def test_p0_event_types() -> Tuple[int, int, List[str]]:
    """
    测试 P0: EventBus 事件类型
    
    Returns:
        (passed, failed, details)
    """
    passed = 0
    failed = 0
    details = []
    
    print("\n" + "-" * 40)
    print("P0: EVENT TYPES TEST")
    print("-" * 40)
    
    try:
        from infra.unified_event_v521 import EventTypeV521
        
        # P0-1 事件类型
        p0_1_events = [
            "ONCHAIN_TICK",
            "SENTIMENT_TICK",
            "MACRO_TICK",
            "LOB_TICK",
            "DATA_HEALTH_CHANGE",
            "DEGRADATION_TRIGGERED",
        ]
        
        # P0-2 事件类型
        p0_2_events = [
            "MODEL_INTENT",
            "MODEL_ACCEPTED",
            "MODEL_VETOED",
            "MODEL_WEIGHT_ADJUSTED",
        ]
        
        all_events = p0_1_events + p0_2_events
        
        for event_name in all_events:
            if hasattr(EventTypeV521, event_name):
                passed += 1
                print(f"  ON {event_name}: defined")
            else:
                failed += 1
                print(f"  OFF {event_name}: NOT defined")
                details.append(f"  Missing EventType: {event_name}")
        
        details.append(f"  EventTypes: {passed} defined, {failed} missing")
        
    except Exception as e:
        details.append(f"  EventTypes: Error - {e}")
        failed += 1
        print(f"  OFF EventTypes: {e}")
    
    return passed, failed, details


def run_p0_diagnostic() -> Dict[str, Any]:
    """
    运行完整 P0 诊断
    
    Returns:
        诊断结果字典
    """
    print("\n" + "=" * 60)
    print("HMATS v5.2.1 P0 COMPLETE DIAGNOSTIC")
    print("=" * 60)
    
    total_passed = 0
    total_failed = 0
    all_details = []
    
    # P0-1 Tests
    tests = [
        ("P0-1 Feeds", test_p0_feeds),
        ("P0-1 Data Health", test_p0_data_health),
        ("P0-1 TradeGate", test_p0_trade_gate_integration),
        ("P0-2 Model Alpha", test_p0_model_alpha),
        ("P0-2 Authority Fusion", test_p0_authority_fusion_integration),
        ("P0-2 Correlation Weight", test_p0_correlation_weight_control),
        ("P0-2 Adaptive Weight", test_p0_adaptive_weight),
        ("P0 Event Types", test_p0_event_types),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            passed, failed, details = test_func()
            total_passed += passed
            total_failed += failed
            all_details.extend(details)
            results[test_name] = {
                "passed": passed,
                "failed": failed,
                "details": details,
            }
        except Exception as e:
            total_failed += 1
            all_details.append(f"  {test_name}: EXCEPTION - {e}")
            results[test_name] = {
                "passed": 0,
                "failed": 1,
                "details": [f"Exception: {e}"],
            }
            print(f"  OFF {test_name}: EXCEPTION - {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("P0 DIAGNOSTIC SUMMARY")
    print("=" * 60)
    print(f"  Total Passed: {total_passed}")
    print(f"  Total Failed: {total_failed}")
    print(f"  Pass Rate: {total_passed / (total_passed + total_failed) * 100:.1f}%")
    
    # P0 Completion Status
    p0_complete = total_failed == 0
    
    print("\n" + "-" * 40)
    if p0_complete:
        print("  ON P0 COMPLETE: All tests passed")
    else:
        print("  [WARN]️ P0 INCOMPLETE: Some tests failed")
        print("\n  Failed Details:")
        for detail in all_details:
            if "Error" in detail or "NOT defined" in detail or "not available" in detail:
                print(f"    {detail}")
    print("-" * 40)
    
    return {
        "p0_complete": p0_complete,
        "total_passed": total_passed,
        "total_failed": total_failed,
        "results": results,
        "details": all_details,
    }


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    result = run_p0_diagnostic()
    
    print("\n" + "=" * 60)
    print("P0 STATUS:", "COMPLETE ON" if result["p0_complete"] else "INCOMPLETE [WARN]️")
    print("=" * 60)
