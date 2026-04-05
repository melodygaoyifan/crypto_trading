"""
================================================================================
HMATS v5.2.1 P1 - 诊断测试
P1 Diagnostic Tests
================================================================================

Purpose: 验证 P1 所有组件的功能和集成

测试范围:
- P1-A: DRL Execution Advisory
- P1-B-1: Execution Quality Logger
- P1-B-2: Impact Calibration
- P1-B-3: Execution Quality Report
- P1-C: DRL Authority Boundaries

验收标准:
1. 每笔执行 log 显示 RL 使用情况
2. 报告包含 RL vs baseline 对比
3. 明确区分 production-ready vs shadow/advisory

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

class P1TestResult:
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
# P1-A: DRL EXECUTION ADVISORY TESTS
# =============================================================================

def test_p1a_rl_execution_advisor() -> P1TestResult:
    """测试 RL 执行顾问"""
    result = P1TestResult("P1-A: RL Execution Advisor")
    
    try:
        from execution.rl_execution_advisor import (
            RLExecutionAdvisor,
            RLExecutionAdvisorConfig,
            AdvisoryMode,
            ExecutionState,
            RLExecutionAdvice,
            get_rl_execution_advisor,
        )
        result.add_pass("Import rl_execution_advisor")
    except ImportError as e:
        result.add_fail("Import rl_execution_advisor", str(e))
        return result
    
    # 测试创建
    try:
        config = RLExecutionAdvisorConfig(mode=AdvisoryMode.ADVISORY)
        advisor = RLExecutionAdvisor(config)
        result.add_pass("Create RLExecutionAdvisor")
    except Exception as e:
        result.add_fail("Create RLExecutionAdvisor", str(e))
        return result
    
    # 测试 ExecutionState
    try:
        state = ExecutionState(
            timestamp=datetime.now(),
            asset="BTC",
            side="buy",
            total_quantity=1.0,
            executed_quantity=0.3,
            remaining_quantity=0.7,
            mid_price=100000,
            spread_bps=5.0,
            volatility=0.5,
            lob_imbalance=0.2,
            bid_depth_usd=500000,
            ask_depth_usd=450000,
            urgency=0.5,
        )
        result.add_pass("Create ExecutionState")
    except Exception as e:
        result.add_fail("Create ExecutionState", str(e))
        return result
    
    # 测试获取建议
    try:
        advice = advisor.get_advice(state)
        
        # 验证建议结构
        assert hasattr(advice, 'slice_size_pct'), "Missing slice_size_pct"
        assert hasattr(advice, 'wait_time_sec'), "Missing wait_time_sec"
        assert hasattr(advice, 'limit_offset_bps'), "Missing limit_offset_bps"
        assert hasattr(advice, 'cancel_replace_policy'), "Missing cancel_replace_policy"
        assert hasattr(advice, 'confidence'), "Missing confidence"
        
        # 验证边界约束
        assert 0 <= advice.slice_size_pct <= 1, "slice_size_pct out of range"
        assert advice.wait_time_sec >= 0, "wait_time_sec negative"
        assert advice.limit_offset_bps >= 0, "limit_offset_bps negative"
        assert 0 <= advice.confidence <= 1, "confidence out of range"
        
        result.add_pass("Get advice with valid structure")
    except Exception as e:
        result.add_fail("Get advice", str(e))
    
    # 验证 P1-C: DRL 不输出方向
    try:
        # 检查建议中没有方向字段
        advice_dict = advice.to_dict()
        assert 'direction' not in advice_dict, "Advice contains forbidden 'direction'"
        assert 'long' not in str(advice_dict).lower(), "Advice contains 'long'"
        assert 'short' not in str(advice_dict).lower(), "Advice contains 'short'"
        result.add_pass("No direction in advice (P1-C)")
    except Exception as e:
        result.add_fail("P1-C direction check", str(e))
    
    # 测试降权
    try:
        initial_weight = advisor._current_weight
        advisor.on_correlation_spike("spiking")
        assert advisor._current_weight < initial_weight, "Weight not reduced"
        result.add_pass("Weight control on correlation spike")
    except Exception as e:
        result.add_fail("Weight control", str(e))
    
    # 测试熔断
    try:
        advisor._slippage_violations = 0
        advisor._is_fused = False
        
        for i in range(5):
            advisor.record_execution_result(
                advice_used=True,
                realized_slippage_bps=15.0,  # 超过阈值
                fill_ratio=0.8,
                time_to_fill_sec=10.0,
            )
        
        assert advisor._is_fused, "Should be fused after violations"
        result.add_pass("Fuse trigger on slippage violations")
    except Exception as e:
        result.add_fail("Fuse trigger", str(e))
    
    # 测试回退建议
    try:
        fallback = RLExecutionAdvice.create_fallback("test reason")
        assert fallback.is_fallback, "Should be marked as fallback"
        assert fallback.confidence == 1.0, "Fallback should have confidence=1"
        result.add_pass("Create fallback advice")
    except Exception as e:
        result.add_fail("Fallback advice", str(e))
    
    return result


# =============================================================================
# P1-B-1: EXECUTION QUALITY LOGGER TESTS
# =============================================================================

def test_p1b_execution_quality_logger() -> P1TestResult:
    """测试执行质量日志"""
    result = P1TestResult("P1-B-1: Execution Quality Logger")
    
    try:
        from execution.execution_quality_logger import (
            ExecutionQualityLogger,
            ExecutionQualityLoggerConfig,
            ExecutionQualityRecord,
            ExecutionSource,
            get_execution_quality_logger,
        )
        result.add_pass("Import execution_quality_logger")
    except ImportError as e:
        result.add_fail("Import execution_quality_logger", str(e))
        return result
    
    # 创建 logger
    try:
        config = ExecutionQualityLoggerConfig(
            log_dir="/tmp/p1_test_logs",
            flush_interval_records=1000,  # 避免测试时写入
        )
        quality_logger = ExecutionQualityLogger(config)
        result.add_pass("Create ExecutionQualityLogger")
    except Exception as e:
        result.add_fail("Create logger", str(e))
        return result
    
    # 记录执行
    try:
        for i in range(50):
            rl_used = i % 3 == 0
            
            quality_logger.log_simple(
                asset="BTC" if i % 2 == 0 else "ETH",
                side="buy" if i % 4 < 2 else "sell",
                intended_qty=1.0,
                executed_qty=0.95,
                arrival_price=100000,
                avg_exec_price=100050,
                expected_slippage_bps=5.0,
                time_to_fill_sec=3.0,
                rl_advice_used=rl_used,
                volatility=0.5,
                spread_bps=10.0,
            )
        
        assert quality_logger._total_records == 50
        result.add_pass("Log executions")
    except Exception as e:
        result.add_fail("Log executions", str(e))
    
    # 获取摘要
    try:
        summary = quality_logger.get_summary()
        assert summary['total_records'] == 50
        assert summary['unique_assets'] == 2
        result.add_pass("Get summary")
    except Exception as e:
        result.add_fail("Get summary", str(e))
    
    # RL 对比
    try:
        comparison = quality_logger.get_rl_comparison()
        assert comparison['rl_count'] > 0
        assert comparison['baseline_count'] > 0
        assert 'rl_slippage_mean' in comparison
        assert 'baseline_slippage_mean' in comparison
        result.add_pass("RL vs Baseline comparison")
    except Exception as e:
        result.add_fail("RL comparison", str(e))
    
    # 导出校准数据
    try:
        calibration_data = quality_logger.export_for_calibration()
        assert len(calibration_data) == 50
        assert 'asset' in calibration_data[0]
        assert 'realized_impact_bps' in calibration_data[0]
        result.add_pass("Export for calibration")
    except Exception as e:
        result.add_fail("Export calibration", str(e))
    
    return result


# =============================================================================
# P1-B-2: IMPACT CALIBRATION TESTS
# =============================================================================

def test_p1b_impact_calibration() -> P1TestResult:
    """测试冲击模型校准"""
    result = P1TestResult("P1-B-2: Impact Calibration")
    
    try:
        from execution.impact_calibration import (
            ImpactCalibrationTable,
            ImpactCalibrationConfig,
            CalibrationSample,
            ProductionMarketImpactCalibrationBridge,
            get_impact_calibration_table,
        )
        result.add_pass("Import impact_calibration")
    except ImportError as e:
        result.add_fail("Import impact_calibration", str(e))
        return result
    
    # 创建校准表
    try:
        config = ImpactCalibrationConfig(
            calibration_file="/tmp/p1_test_calibration.json",
            min_samples_for_calibration=10,
        )
        table = ImpactCalibrationTable(config)
        result.add_pass("Create ImpactCalibrationTable")
    except Exception as e:
        result.add_fail("Create table", str(e))
        return result
    
    # 生成样本
    try:
        samples = []
        for i in range(100):
            sample = CalibrationSample(
                timestamp=datetime.now(),
                asset="BTC" if i % 2 == 0 else "ETH",
                side="buy",
                size_usd=50000,
                expected_impact_bps=5.0,
                realized_impact_bps=5.0 + 2.0 * np.random.random(),
                volatility=0.3 + 0.4 * np.random.random(),
                spread_bps=5 + 15 * np.random.random(),
                session="us_morning" if i % 2 == 0 else "europe_morning",
            )
            samples.append(sample)
        result.add_pass("Generate calibration samples")
    except Exception as e:
        result.add_fail("Generate samples", str(e))
        return result
    
    # 离线校准
    try:
        table.calibrate_offline(samples)
        entries = table.get_all_entries()
        assert len(entries) > 0, "No calibration entries created"
        result.add_pass("Offline calibration")
    except Exception as e:
        result.add_fail("Offline calibration", str(e))
    
    # 查询参数
    try:
        eta, gamma, conf = table.get_params("BTC", "us_morning", "medium", "normal")
        assert eta > 0, "eta should be positive"
        assert gamma > 0, "gamma should be positive"
        result.add_pass("Query calibrated params")
    except Exception as e:
        result.add_fail("Query params", str(e))
    
    # 测试桥接
    try:
        bridge = ProductionMarketImpactCalibrationBridge(table)
        eta, gamma, conf = bridge.get_calibrated_params("BTC", volatility=0.5, spread_bps=10)
        result.add_pass("Calibration bridge")
    except Exception as e:
        result.add_fail("Calibration bridge", str(e))
    
    return result


# =============================================================================
# P1-B-3: EXECUTION QUALITY REPORT TESTS
# =============================================================================

def test_p1b_execution_quality_report() -> P1TestResult:
    """测试执行质量报告"""
    result = P1TestResult("P1-B-3: Execution Quality Report")
    
    try:
        from analytics.execution_quality_report import (
            ExecutionQualityReportGenerator,
            ExecutionQualityReportV521,
            get_execution_quality_report_generator,
        )
        result.add_pass("Import execution_quality_report")
    except ImportError as e:
        result.add_fail("Import execution_quality_report", str(e))
        return result
    
    # 创建生成器
    try:
        generator = ExecutionQualityReportGenerator("/tmp/p1_test_reports")
        result.add_pass("Create report generator")
    except Exception as e:
        result.add_fail("Create generator", str(e))
        return result
    
    # 生成模拟记录
    try:
        from dataclasses import dataclass
        
        @dataclass
        class MockRecord:
            timestamp: datetime
            asset: str
            side: str
            intended_quantity: float
            executed_quantity: float
            arrival_price: float
            avg_execution_price: float
            realized_slippage_bps: float
            fill_ratio: float
            time_to_fill_sec: float
            cancel_count: int
            rl_advice_used: bool
            utc_session: str
            volatility_bucket: str
        
        records = []
        for i in range(100):
            rl_used = i % 3 == 0
            slip = 4.0 if rl_used else 6.0
            slip += np.random.random() * 2
            
            records.append(MockRecord(
                timestamp=datetime.now() - timedelta(hours=i),
                asset=["BTC", "ETH", "SOL"][i % 3],
                side="buy" if i % 2 == 0 else "sell",
                intended_quantity=1.0,
                executed_quantity=0.95,
                arrival_price=100000,
                avg_execution_price=100000 * (1 + slip/10000),
                realized_slippage_bps=slip,
                fill_ratio=0.95,
                time_to_fill_sec=3.0,
                cancel_count=0,
                rl_advice_used=rl_used,
                utc_session=["us_morning", "europe_morning"][i % 2],
                volatility_bucket=["low", "medium", "high"][i % 3],
            ))
        
        result.add_pass("Generate mock records")
    except Exception as e:
        result.add_fail("Generate records", str(e))
        return result
    
    # 生成报告
    try:
        report = generator.generate_report(records)
        
        # 验证报告内容
        assert report.total_executions == 100
        assert report.rl_executions > 0
        assert report.baseline_executions > 0
        assert report.rl_slippage_mean_bps > 0
        assert report.baseline_slippage_mean_bps > 0
        
        result.add_pass("Generate report")
    except Exception as e:
        result.add_fail("Generate report", str(e))
        return result
    
    # 保存 JSON
    try:
        json_path = generator.save_json(report)
        assert os.path.exists(json_path)
        result.add_pass("Save JSON report")
    except Exception as e:
        result.add_fail("Save JSON", str(e))
    
    # 保存 HTML
    try:
        html_path = generator.save_html(report)
        assert os.path.exists(html_path)
        result.add_pass("Save HTML report")
    except Exception as e:
        result.add_fail("Save HTML", str(e))
    
    # 验证 RL vs Baseline 对比
    try:
        assert 'slippage_improvement_bps' in report.__dict__
        # RL 应该更好 (因为模拟数据这样设置的)
        assert report.slippage_improvement_bps > 0, "RL should show improvement"
        result.add_pass("RL vs Baseline improvement")
    except Exception as e:
        result.add_fail("RL improvement check", str(e))
    
    return result


# =============================================================================
# P1-C: DRL AUTHORITY BOUNDARY TESTS
# =============================================================================

def test_p1c_drl_authority_boundaries() -> P1TestResult:
    """测试 DRL 权力边界"""
    result = P1TestResult("P1-C: DRL Authority Boundaries")
    
    # 测试 1: DRL 不能产生 entry
    try:
        from execution.rl_execution_advisor import RLExecutionAdvisor, RLExecutionAdvisorConfig
        
        advisor = RLExecutionAdvisor()
        
        # 检查没有 entry 相关方法
        assert not hasattr(advisor, 'generate_entry'), "Should not have generate_entry"
        assert not hasattr(advisor, 'entry_signal'), "Should not have entry_signal"
        
        result.add_pass("No entry authority in RLExecutionAdvisor")
    except Exception as e:
        result.add_fail("Entry authority check", str(e))
    
    # 测试 2: DRL 可被 correlation 降权
    try:
        from orchestration.p1_complete_integration import P1ExecutionIntegrator
        
        integrator = P1ExecutionIntegrator()
        integrator.initialize()
        
        initial_weight = integrator._rl_weight
        integrator.on_correlation_spike("crisis")
        
        assert integrator._rl_weight < initial_weight, "Weight should decrease"
        result.add_pass("Correlation spike downweight")
    except Exception as e:
        result.add_fail("Correlation downweight", str(e))
    
    # 测试 3: DRL 可被 volatility 降权
    try:
        integrator.reset_rl_weight()
        initial_weight = integrator._rl_weight
        integrator.on_volatility_spike("extreme")
        
        assert integrator._rl_weight < initial_weight, "Weight should decrease"
        result.add_pass("Volatility spike downweight")
    except Exception as e:
        result.add_fail("Volatility downweight", str(e))
    
    # 测试 4: DRL 可被 DataHealth 禁用
    try:
        integrator.reset_rl_weight()
        integrator.on_data_health_change("no_trade")
        
        assert integrator._rl_weight == 0, "Weight should be 0"
        result.add_pass("DataHealth disable")
    except Exception as e:
        result.add_fail("DataHealth disable", str(e))
    
    return result


# =============================================================================
# P1 COMPLETE INTEGRATION TESTS
# =============================================================================

def test_p1_complete_integration() -> P1TestResult:
    """测试 P1 完整集成"""
    result = P1TestResult("P1 Complete Integration")
    
    try:
        from orchestration.p1_complete_integration import (
            P1ExecutionIntegrator,
            P1IntegrationConfig,
            get_p1_execution_integrator,
        )
        result.add_pass("Import p1_complete_integration")
    except ImportError as e:
        result.add_fail("Import p1_complete_integration", str(e))
        return result
    
    # 创建集成器
    try:
        config = P1IntegrationConfig(
            rl_advisor_mode="advisory",
        )
        integrator = P1ExecutionIntegrator(config)
        result.add_pass("Create P1ExecutionIntegrator")
    except Exception as e:
        result.add_fail("Create integrator", str(e))
        return result
    
    # 初始化
    try:
        success = integrator.initialize()
        assert success, "Initialization failed"
        result.add_pass("Initialize P1")
    except Exception as e:
        result.add_fail("Initialize", str(e))
        return result
    
    # 获取执行建议
    try:
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
        )
        
        assert 'source' in advice
        assert 'slice_size_pct' in advice
        assert advice['source'] in ['rl', 'baseline', 'fallback']
        result.add_pass("Get execution advice")
    except Exception as e:
        result.add_fail("Get advice", str(e))
    
    # 记录执行结果
    try:
        integrator.record_execution_result(
            asset="BTC",
            side="buy",
            intended_qty=1.0,
            executed_qty=0.95,
            arrival_price=100000,
            avg_exec_price=100050,
            expected_slippage_bps=5.0,
            time_to_fill_sec=4.0,
            rl_advice_used=advice['source'] == 'rl',
            volatility=0.5,
            spread_bps=5.0,
        )
        result.add_pass("Record execution result")
    except Exception as e:
        result.add_fail("Record result", str(e))
    
    # 获取状态
    try:
        status = integrator.get_status()
        assert 'initialized' in status
        assert 'rl_weight' in status
        assert 'stats' in status
        result.add_pass("Get status")
    except Exception as e:
        result.add_fail("Get status", str(e))
    
    # 诊断输出
    try:
        diagnostic = integrator.get_diagnostic_output()
        assert "P1 Execution Intelligence" in diagnostic
        result.add_pass("Diagnostic output")
    except Exception as e:
        result.add_fail("Diagnostic", str(e))
    
    return result


# =============================================================================
# RUN ALL TESTS
# =============================================================================

def run_p1_diagnostic() -> Dict[str, Any]:
    """运行所有 P1 诊断测试"""
    print("=" * 70)
    print("HMATS v5.2.1 P1 Diagnostic Tests")
    print("=" * 70)
    print()
    
    tests = [
        ("P1-A", test_p1a_rl_execution_advisor),
        ("P1-B-1", test_p1b_execution_quality_logger),
        ("P1-B-2", test_p1b_impact_calibration),
        ("P1-B-3", test_p1b_execution_quality_report),
        ("P1-C", test_p1c_drl_authority_boundaries),
        ("P1-Integration", test_p1_complete_integration),
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
            
            # 打印结果
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
    print("P1 DIAGNOSTIC SUMMARY")
    print("=" * 70)
    
    all_pass = all(r.success for r in results.values())
    
    print(f"\nTotal Checks: {total_passed + total_failed}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    print(f"\nOverall: {'✓ P1 COMPLETE' if all_pass else '✗ P1 INCOMPLETE'}")
    
    # 状态说明
    print("\n" + "-" * 70)
    print("Component Status:")
    print("-" * 70)
    
    status_info = [
        ("P1-A: RL Execution Advisor", "advisory", "DRL 仅输出执行参数，不允许 entry"),
        ("P1-B-1: Execution Quality Logger", "production-ready", "记录每笔执行质量"),
        ("P1-B-2: Impact Calibration", "production-ready", "按 bucket 校准冲击模型"),
        ("P1-B-3: Execution Quality Report", "production-ready", "RL vs Baseline 报告"),
        ("P1-C: Authority Boundaries", "enforced", "DRL 可降权/禁用"),
    ]
    
    for name, status, desc in status_info:
        result_mark = "✓" if results.get(name.split(":")[0], P1TestResult("")).success else "[WARN]"
        print(f"  {result_mark} {name}")
        print(f"      Status: {status}")
        print(f"      {desc}")
    
    print("\n" + "=" * 70)
    
    return {
        "p1_complete": all_pass,
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
    
    result = run_p1_diagnostic()
    
    # 退出码
    sys.exit(0 if result["p1_complete"] else 1)
