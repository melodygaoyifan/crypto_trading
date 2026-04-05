"""
================================================================================
HMATS v5.2.1 P1 - 执行质量报告
Execution Quality Report
================================================================================

Purpose: 生成执行质量分析报告，重点对比 RL vs Baseline

P1-B 核心输出:
1. RL vs 非 RL 执行对比
2. 冲击成本占比
3. 滑点分位数
4. 执行失败率
5. 熔断触发统计

输出格式:
- JSON (程序读取)
- HTML (人工查看)

================================================================================
"""

import logging
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# REPORT DATA STRUCTURES
# =============================================================================

@dataclass
class ExecutionQualityReportV521:
    """
    执行质量报告
    
    P1 验收标准要求的完整报告结构
    """
    # 元信息
    report_id: str
    generated_at: datetime
    period_start: datetime
    period_end: datetime
    version: str = "5.2.1-P1"
    
    # 总体统计
    total_executions: int = 0
    total_volume_usd: float = 0.0
    
    # RL vs Baseline 对比 (核心)
    rl_executions: int = 0
    baseline_executions: int = 0
    rl_usage_rate: float = 0.0
    
    rl_slippage_mean_bps: float = 0.0
    rl_slippage_p50_bps: float = 0.0
    rl_slippage_p90_bps: float = 0.0
    rl_slippage_p99_bps: float = 0.0
    
    baseline_slippage_mean_bps: float = 0.0
    baseline_slippage_p50_bps: float = 0.0
    baseline_slippage_p90_bps: float = 0.0
    baseline_slippage_p99_bps: float = 0.0
    
    # 改进分析
    slippage_improvement_bps: float = 0.0
    slippage_improvement_pct: float = 0.0
    
    # 冲击成本
    total_impact_cost_usd: float = 0.0
    impact_cost_ratio_pct: float = 0.0  # impact_cost / total_volume
    
    # 执行质量
    fill_ratio_mean: float = 0.0
    time_to_fill_mean_sec: float = 0.0
    cancel_rate: float = 0.0
    
    # 熔断统计
    fuse_trigger_count: int = 0
    fuse_reasons: Dict[str, int] = field(default_factory=dict)
    
    # 降权统计
    downweight_count: int = 0
    downweight_reasons: Dict[str, int] = field(default_factory=dict)
    
    # 分桶分析
    by_asset: Dict[str, Dict] = field(default_factory=dict)
    by_session: Dict[str, Dict] = field(default_factory=dict)
    by_volatility: Dict[str, Dict] = field(default_factory=dict)
    
    # 状态
    status: str = "complete"
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        d = asdict(self)
        d["generated_at"] = self.generated_at.isoformat()
        d["period_start"] = self.period_start.isoformat()
        d["period_end"] = self.period_end.isoformat()
        return d


# =============================================================================
# REPORT GENERATOR
# =============================================================================

class ExecutionQualityReportGenerator:
    """
    执行质量报告生成器
    
    从 ExecutionQualityLogger 读取数据并生成报告
    """
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 缓存
        self._last_report: Optional[ExecutionQualityReportV521] = None
    
    def generate_report(
        self,
        records: List[Any],  # ExecutionQualityRecord
        period_start: datetime = None,
        period_end: datetime = None,
        include_fuse_stats: bool = True,
    ) -> ExecutionQualityReportV521:
        """
        生成执行质量报告
        
        Args:
            records: 执行质量记录列表
            period_start: 报告开始时间
            period_end: 报告结束时间
            include_fuse_stats: 是否包含熔断统计
        
        Returns:
            ExecutionQualityReportV521
        """
        if not records:
            return ExecutionQualityReportV521(
                report_id=f"exec_quality_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                generated_at=datetime.now(),
                period_start=period_start or datetime.now(),
                period_end=period_end or datetime.now(),
                status="no_data",
            )
        
        # 确定时间范围
        timestamps = [r.timestamp for r in records]
        period_start = period_start or min(timestamps)
        period_end = period_end or max(timestamps)
        
        # 过滤记录
        filtered = [
            r for r in records
            if period_start <= r.timestamp <= period_end
        ]
        
        if not filtered:
            return ExecutionQualityReportV521(
                report_id=f"exec_quality_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                generated_at=datetime.now(),
                period_start=period_start,
                period_end=period_end,
                status="no_data_in_period",
            )
        
        # 分离 RL 和 Baseline
        rl_records = [r for r in filtered if r.rl_advice_used]
        baseline_records = [r for r in filtered if not r.rl_advice_used]
        
        # 创建报告
        report = ExecutionQualityReportV521(
            report_id=f"exec_quality_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            generated_at=datetime.now(),
            period_start=period_start,
            period_end=period_end,
        )
        
        # 总体统计
        report.total_executions = len(filtered)
        report.total_volume_usd = sum(r.intended_quantity * r.arrival_price for r in filtered)
        
        # RL vs Baseline
        report.rl_executions = len(rl_records)
        report.baseline_executions = len(baseline_records)
        report.rl_usage_rate = len(rl_records) / len(filtered) if filtered else 0
        
        # RL 滑点统计
        if rl_records:
            rl_slippages = [r.realized_slippage_bps for r in rl_records]
            report.rl_slippage_mean_bps = np.mean(rl_slippages)
            report.rl_slippage_p50_bps = np.percentile(rl_slippages, 50)
            report.rl_slippage_p90_bps = np.percentile(rl_slippages, 90)
            report.rl_slippage_p99_bps = np.percentile(rl_slippages, 99)
        
        # Baseline 滑点统计
        if baseline_records:
            baseline_slippages = [r.realized_slippage_bps for r in baseline_records]
            report.baseline_slippage_mean_bps = np.mean(baseline_slippages)
            report.baseline_slippage_p50_bps = np.percentile(baseline_slippages, 50)
            report.baseline_slippage_p90_bps = np.percentile(baseline_slippages, 90)
            report.baseline_slippage_p99_bps = np.percentile(baseline_slippages, 99)
        
        # 改进分析
        if rl_records and baseline_records:
            report.slippage_improvement_bps = (
                report.baseline_slippage_mean_bps - report.rl_slippage_mean_bps
            )
            report.slippage_improvement_pct = (
                report.slippage_improvement_bps / report.baseline_slippage_mean_bps * 100
                if report.baseline_slippage_mean_bps != 0 else 0
            )
        
        # 冲击成本
        total_slippage_cost = sum(
            r.realized_slippage_bps / 10000 * r.intended_quantity * r.arrival_price
            for r in filtered
        )
        report.total_impact_cost_usd = total_slippage_cost
        report.impact_cost_ratio_pct = (
            total_slippage_cost / report.total_volume_usd * 100
            if report.total_volume_usd > 0 else 0
        )
        
        # 执行质量
        report.fill_ratio_mean = np.mean([r.fill_ratio for r in filtered])
        report.time_to_fill_mean_sec = np.mean([r.time_to_fill_sec for r in filtered])
        report.cancel_rate = np.mean([1 if r.cancel_count > 0 else 0 for r in filtered])
        
        # 分桶分析
        report.by_asset = self._analyze_by_dimension(filtered, lambda r: r.asset)
        report.by_session = self._analyze_by_dimension(filtered, lambda r: r.utc_session)
        report.by_volatility = self._analyze_by_dimension(filtered, lambda r: r.volatility_bucket)
        
        # 警告
        if report.rl_usage_rate < 0.1:
            report.warnings.append("RL usage rate is low (<10%)")
        if report.slippage_improvement_pct < 0:
            report.warnings.append("RL shows worse slippage than baseline")
        if report.fill_ratio_mean < 0.9:
            report.warnings.append("Low overall fill ratio (<90%)")
        
        self._last_report = report
        return report
    
    def _analyze_by_dimension(
        self,
        records: List[Any],
        key_func: Callable,
    ) -> Dict[str, Dict]:
        """按维度分析"""
        groups: Dict[str, List] = defaultdict(list)
        
        for r in records:
            key = key_func(r)
            groups[key].append(r)
        
        result = {}
        for key, group in groups.items():
            rl_count = len([r for r in group if r.rl_advice_used])
            result[key] = {
                "count": len(group),
                "rl_count": rl_count,
                "rl_ratio": rl_count / len(group) if group else 0,
                "slippage_mean": np.mean([r.realized_slippage_bps for r in group]),
                "fill_ratio": np.mean([r.fill_ratio for r in group]),
            }
        
        return result
    
    # =========================================================================
    # OUTPUT
    # =========================================================================
    
    def save_json(self, report: ExecutionQualityReportV521) -> str:
        """保存为 JSON"""
        filepath = os.path.join(self.output_dir, f"{report.report_id}.json")
        
        with open(filepath, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        
        logger.info(f"Saved JSON report: {filepath}")
        return filepath
    
    def save_html(self, report: ExecutionQualityReportV521) -> str:
        """保存为 HTML"""
        filepath = os.path.join(self.output_dir, f"{report.report_id}.html")
        
        html = self._generate_html(report)
        
        with open(filepath, "w") as f:
            f.write(html)
        
        logger.info(f"Saved HTML report: {filepath}")
        return filepath
    
    def _generate_html(self, report: ExecutionQualityReportV521) -> str:
        """生成 HTML 报告"""
        
        # 确定改进颜色
        improvement_color = "#28a745" if report.slippage_improvement_bps > 0 else "#dc3545"
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Execution Quality Report - {report.report_id}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .header h1 {{
            margin: 0 0 10px 0;
        }}
        .header .meta {{
            opacity: 0.9;
            font-size: 14px;
        }}
        .card {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .card h2 {{
            margin-top: 0;
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        .metric {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .metric .value {{
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }}
        .metric .label {{
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }}
        .comparison {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        .comparison-item {{
            padding: 20px;
            border-radius: 8px;
        }}
        .rl-side {{
            background: #e8f4fd;
            border-left: 4px solid #0d6efd;
        }}
        .baseline-side {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
        }}
        .improvement {{
            text-align: center;
            padding: 20px;
            background: {improvement_color}20;
            border-radius: 8px;
            margin-top: 20px;
        }}
        .improvement .value {{
            font-size: 32px;
            font-weight: bold;
            color: {improvement_color};
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #f8f9fa;
            font-weight: 600;
        }}
        .warning {{
            background: #fff3cd;
            border: 1px solid #ffc107;
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
        }}
        .status-badge {{
            display: inline-block;
            padding: 3px 10px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
        }}
        .status-complete {{
            background: #d4edda;
            color: #155724;
        }}
        .status-warning {{
            background: #fff3cd;
            color: #856404;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Execution Quality Report</h1>
        <div class="meta">
            <strong>Report ID:</strong> {report.report_id}<br>
            <strong>Period:</strong> {report.period_start.strftime('%Y-%m-%d %H:%M')} - {report.period_end.strftime('%Y-%m-%d %H:%M')}<br>
            <strong>Generated:</strong> {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}<br>
            <strong>Version:</strong> {report.version}
            <span class="status-badge status-{'complete' if report.status == 'complete' else 'warning'}">{report.status}</span>
        </div>
    </div>
    
    <div class="card">
        <h2>📈 Overall Statistics</h2>
        <div class="metrics">
            <div class="metric">
                <div class="value">{report.total_executions:,}</div>
                <div class="label">Total Executions</div>
            </div>
            <div class="metric">
                <div class="value">${report.total_volume_usd:,.0f}</div>
                <div class="label">Total Volume</div>
            </div>
            <div class="metric">
                <div class="value">{report.rl_usage_rate:.1%}</div>
                <div class="label">RL Usage Rate</div>
            </div>
            <div class="metric">
                <div class="value">{report.fill_ratio_mean:.1%}</div>
                <div class="label">Avg Fill Ratio</div>
            </div>
        </div>
    </div>
    
    <div class="card">
        <h2>🤖 RL vs Baseline Comparison</h2>
        <div class="comparison">
            <div class="comparison-item rl-side">
                <h3>🔵 RL Advisory ({report.rl_executions} executions)</h3>
                <table>
                    <tr><td>Mean Slippage</td><td><strong>{report.rl_slippage_mean_bps:.2f} bps</strong></td></tr>
                    <tr><td>P50 Slippage</td><td>{report.rl_slippage_p50_bps:.2f} bps</td></tr>
                    <tr><td>P90 Slippage</td><td>{report.rl_slippage_p90_bps:.2f} bps</td></tr>
                    <tr><td>P99 Slippage</td><td>{report.rl_slippage_p99_bps:.2f} bps</td></tr>
                </table>
            </div>
            <div class="comparison-item baseline-side">
                <h3>🟡 Baseline ({report.baseline_executions} executions)</h3>
                <table>
                    <tr><td>Mean Slippage</td><td><strong>{report.baseline_slippage_mean_bps:.2f} bps</strong></td></tr>
                    <tr><td>P50 Slippage</td><td>{report.baseline_slippage_p50_bps:.2f} bps</td></tr>
                    <tr><td>P90 Slippage</td><td>{report.baseline_slippage_p90_bps:.2f} bps</td></tr>
                    <tr><td>P99 Slippage</td><td>{report.baseline_slippage_p99_bps:.2f} bps</td></tr>
                </table>
            </div>
        </div>
        <div class="improvement">
            <div class="value">{report.slippage_improvement_bps:+.2f} bps ({report.slippage_improvement_pct:+.1f}%)</div>
            <div class="label">Slippage Improvement (RL vs Baseline)</div>
        </div>
    </div>
    
    <div class="card">
        <h2>💰 Impact Cost Analysis</h2>
        <div class="metrics">
            <div class="metric">
                <div class="value">${report.total_impact_cost_usd:,.2f}</div>
                <div class="label">Total Impact Cost</div>
            </div>
            <div class="metric">
                <div class="value">{report.impact_cost_ratio_pct:.3f}%</div>
                <div class="label">Cost / Volume Ratio</div>
            </div>
            <div class="metric">
                <div class="value">{report.time_to_fill_mean_sec:.1f}s</div>
                <div class="label">Avg Time to Fill</div>
            </div>
            <div class="metric">
                <div class="value">{report.cancel_rate:.1%}</div>
                <div class="label">Cancel Rate</div>
            </div>
        </div>
    </div>
    
    <div class="card">
        <h2>📊 By Asset</h2>
        <table>
            <thead>
                <tr>
                    <th>Asset</th>
                    <th>Count</th>
                    <th>RL Ratio</th>
                    <th>Slippage (bps)</th>
                    <th>Fill Ratio</th>
                </tr>
            </thead>
            <tbody>
                {"".join(f'''
                <tr>
                    <td><strong>{asset}</strong></td>
                    <td>{data['count']}</td>
                    <td>{data['rl_ratio']:.1%}</td>
                    <td>{data['slippage_mean']:.2f}</td>
                    <td>{data['fill_ratio']:.1%}</td>
                </tr>
                ''' for asset, data in report.by_asset.items())}
            </tbody>
        </table>
    </div>
    
    <div class="card">
        <h2>🕐 By Session</h2>
        <table>
            <thead>
                <tr>
                    <th>Session</th>
                    <th>Count</th>
                    <th>RL Ratio</th>
                    <th>Slippage (bps)</th>
                    <th>Fill Ratio</th>
                </tr>
            </thead>
            <tbody>
                {"".join(f'''
                <tr>
                    <td><strong>{session}</strong></td>
                    <td>{data['count']}</td>
                    <td>{data['rl_ratio']:.1%}</td>
                    <td>{data['slippage_mean']:.2f}</td>
                    <td>{data['fill_ratio']:.1%}</td>
                </tr>
                ''' for session, data in report.by_session.items())}
            </tbody>
        </table>
    </div>
    
    {"".join(f'<div class="warning">[WARN]️ {w}</div>' for w in report.warnings)}
    
    <div style="text-align: center; margin-top: 30px; color: #666; font-size: 12px;">
        HMATS v5.2.1-P1 | Execution Quality Report | Generated by ExecutionQualityReportGenerator
    </div>
</body>
</html>
"""
        return html


# =============================================================================
# SINGLETON
# =============================================================================

_report_generator_instance: Optional[ExecutionQualityReportGenerator] = None


def get_execution_quality_report_generator(
    output_dir: str = "reports"
) -> ExecutionQualityReportGenerator:
    """获取报告生成器单例"""
    global _report_generator_instance
    if _report_generator_instance is None:
        _report_generator_instance = ExecutionQualityReportGenerator(output_dir)
    return _report_generator_instance


def reset_execution_quality_report_generator():
    """重置单例"""
    global _report_generator_instance
    _report_generator_instance = None


# =============================================================================
# INTEGRATION WITH QUALITY LOGGER
# =============================================================================

def generate_execution_quality_report_from_logger(
    output_dir: str = "reports",
    days: int = 7,
) -> Tuple[str, str]:
    """
    从质量日志生成报告
    
    便捷方法，自动从 ExecutionQualityLogger 获取数据
    
    Returns:
        (json_path, html_path)
    """
    try:
        from execution.execution_quality_logger import get_execution_quality_logger
        
        quality_logger = get_execution_quality_logger()
        records = list(quality_logger._records)
        
        if not records:
            # 尝试加载历史
            records = quality_logger.load_historical(days)
        
        generator = ExecutionQualityReportGenerator(output_dir)
        report = generator.generate_report(records)
        
        json_path = generator.save_json(report)
        html_path = generator.save_html(report)
        
        return json_path, html_path
        
    except Exception as e:
        logger.error(f"Failed to generate report: {e}")
        raise


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'ExecutionQualityReportV521',
    'ExecutionQualityReportGenerator',
    'get_execution_quality_report_generator',
    'reset_execution_quality_report_generator',
    'generate_execution_quality_report_from_logger',
]


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Testing ExecutionQualityReportGenerator")
    print("=" * 60)
    
    # 模拟 ExecutionQualityRecord
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
    
    # 生成模拟数据
    print("\n1. Generating mock records...")
    
    records = []
    for i in range(200):
        rl_used = i % 3 == 0
        
        # RL 通常更好
        base_slip = 5.0
        if rl_used:
            slip = base_slip * (0.85 + 0.2 * np.random.random())
        else:
            slip = base_slip * (1.0 + 0.3 * np.random.random())
        
        records.append(MockRecord(
            timestamp=datetime.now() - timedelta(hours=np.random.randint(0, 168)),
            asset=["BTC", "ETH", "SOL"][i % 3],
            side="buy" if i % 2 == 0 else "sell",
            intended_quantity=1.0,
            executed_quantity=0.95 + 0.05 * np.random.random(),
            arrival_price=100000,
            avg_execution_price=100000 * (1 + slip / 10000),
            realized_slippage_bps=slip,
            fill_ratio=0.9 + 0.1 * np.random.random(),
            time_to_fill_sec=2 + 5 * np.random.random(),
            cancel_count=1 if np.random.random() > 0.8 else 0,
            rl_advice_used=rl_used,
            utc_session=["us_morning", "europe_morning", "asia_morning"][i % 3],
            volatility_bucket=["low", "medium", "high"][i % 3],
        ))
    
    print(f"   Generated {len(records)} records")
    
    # 生成报告
    print("\n2. Generating report...")
    generator = ExecutionQualityReportGenerator("/tmp/execution_reports")
    report = generator.generate_report(records)
    
    print(f"   Report ID: {report.report_id}")
    print(f"   Total Executions: {report.total_executions}")
    print(f"   RL Usage: {report.rl_usage_rate:.1%}")
    print(f"   RL Slippage: {report.rl_slippage_mean_bps:.2f} bps")
    print(f"   Baseline Slippage: {report.baseline_slippage_mean_bps:.2f} bps")
    print(f"   Improvement: {report.slippage_improvement_bps:.2f} bps ({report.slippage_improvement_pct:.1f}%)")
    
    # 保存
    print("\n3. Saving reports...")
    json_path = generator.save_json(report)
    print(f"   JSON: {json_path}")
    
    html_path = generator.save_html(report)
    print(f"   HTML: {html_path}")
    
    # 显示警告
    if report.warnings:
        print("\n4. Warnings:")
        for w in report.warnings:
            print(f"   [WARN]️ {w}")
    
    print("\n✓ ExecutionQualityReportGenerator test complete")
