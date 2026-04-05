#!/usr/bin/env python3
"""
================================================================================
HMATS Training Pipeline - P0/P1 改进工具
================================================================================

P0 改进 (必须):
  ON torch.compile 支持 (已集成到训练脚本)
  ON BF16 精度支持 (已集成到训练脚本)
  ON Regime 名称映射 (已集成到 DT v3.2)
  ON 数据泄露检测

P1 改进 (推荐):
  ON 消融实验框架
  ON Walk-Forward 验证
  ON TTA 推理控制 (已集成到 Sentiment v2.2)

用法:
    python utils/improvements.py --check-leakage data.parquet
    python utils/improvements.py --ablation drl
    python utils/improvements.py --walkforward data.parquet

================================================================================
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import time

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger("Improvements")


# =============================================================================
# P0: 数据泄露检测
# =============================================================================

class DataLeakageDetector:
    """
    数据泄露检测器
    
    检测特征是否包含未来信息，避免虚假的高准确率
    """
    
    def __init__(self, lookahead: int = 24, threshold: float = 1.5):
        """
        Args:
            lookahead: 前瞻时间 (小时)
            threshold: 泄露阈值 (future_corr / past_corr > threshold 视为泄露)
        """
        self.lookahead = lookahead
        self.threshold = threshold
    
    def check_features(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = 'future_return'
    ) -> Dict[str, Dict]:
        """
        检测特征是否存在数据泄露
        
        Args:
            df: 数据框
            feature_cols: 特征列名列表
            target_col: 目标列名
        
        Returns:
            泄露报告 {feature: {score, corr_future, corr_past, suspicious}}
        """
        # 确保目标列存在
        if target_col not in df.columns:
            # 尝试计算 future_return
            if 'close' in df.columns:
                df = df.copy()
                df[target_col] = df['close'].shift(-self.lookahead) / df['close'] - 1
            else:
                raise ValueError(f"目标列 '{target_col}' 不存在且无法计算")
        
        leakage_report = {}
        suspicious_count = 0
        
        logger.info(f"检测 {len(feature_cols)} 个特征的数据泄露...")
        logger.info(f"前瞻时间: {self.lookahead}h, 阈值: {self.threshold}")
        
        for col in feature_cols:
            if col not in df.columns or col == target_col:
                continue
            
            try:
                # 与未来目标的相关性
                corr_future = df[col].corr(df[target_col].shift(-self.lookahead))
                
                # 与过去目标的相关性
                corr_past = df[col].corr(df[target_col].shift(self.lookahead))
                
                # 计算泄露分数
                if abs(corr_past) > 0.01:
                    leakage_score = abs(corr_future) / abs(corr_past)
                else:
                    leakage_score = abs(corr_future) * 10 if abs(corr_future) > 0.1 else 0
                
                is_suspicious = leakage_score > self.threshold
                
                if is_suspicious:
                    suspicious_count += 1
                    logger.warning(f"[WARN]️ 潜在泄露: {col}")
                    logger.warning(f"   future_corr={corr_future:.4f}, past_corr={corr_past:.4f}, score={leakage_score:.2f}")
                
                leakage_report[col] = {
                    'leakage_score': leakage_score,
                    'corr_future': corr_future,
                    'corr_past': corr_past,
                    'suspicious': is_suspicious
                }
                
            except Exception as e:
                logger.debug(f"跳过 {col}: {e}")
        
        # 汇总
        logger.info(f"\n检测完成: {suspicious_count}/{len(leakage_report)} 个特征可疑")
        
        if suspicious_count == 0:
            logger.info("ON 未检测到明显的数据泄露")
        else:
            logger.warning(f"OFF 发现 {suspicious_count} 个可疑特征，建议检查")
        
        return leakage_report
    
    def generate_report(self, leakage_report: Dict, output_path: str = None) -> str:
        """生成泄露报告"""
        lines = ["# 数据泄露检测报告\n"]
        
        # 可疑特征
        suspicious = {k: v for k, v in leakage_report.items() if v['suspicious']}
        
        lines.append(f"## 总结\n")
        lines.append(f"- 检测特征数: {len(leakage_report)}")
        lines.append(f"- 可疑特征数: {len(suspicious)}")
        lines.append(f"- 泄露率: {len(suspicious)/max(len(leakage_report),1)*100:.1f}%\n")
        
        if suspicious:
            lines.append("## 可疑特征\n")
            lines.append("| 特征 | 泄露分数 | 未来相关 | 过去相关 |")
            lines.append("|------|----------|----------|----------|")
            
            for feat, info in sorted(suspicious.items(), key=lambda x: -x[1]['leakage_score']):
                lines.append(f"| {feat} | {info['leakage_score']:.2f} | {info['corr_future']:.4f} | {info['corr_past']:.4f} |")
        
        report = "\n".join(lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"报告已保存: {output_path}")
        
        return report


# =============================================================================
# P0: 严格时间分割
# =============================================================================

def strict_temporal_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    gap_hours: int = 168  # 1周间隔
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    严格按时间分割数据集，避免泄露
    
    Args:
        df: 数据框 (必须按时间排序)
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        gap_hours: 分割间隔 (小时)
    
    Returns:
        (train, val, test)
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end + gap_hours:val_end].copy()
    test = df.iloc[val_end + gap_hours:].copy()
    
    logger.info("数据分割 (严格时间):")
    logger.info(f"  训练集: {len(train)} ({len(train)/n*100:.1f}%)")
    logger.info(f"  验证集: {len(val)} ({len(val)/n*100:.1f}%)")
    logger.info(f"  测试集: {len(test)} ({len(test)/n*100:.1f}%)")
    logger.info(f"  Gap: {gap_hours} 小时")
    
    return train, val, test


# =============================================================================
# P1: 消融实验框架
# =============================================================================

@dataclass
class AblationResult:
    """消融实验结果"""
    technique: str
    enabled: bool
    accuracy: float
    contribution: float
    training_time: float


class AblationStudy:
    """
    消融实验框架
    
    用于验证每个技术的实际贡献
    """
    
    # DT v3.2 可消融技术
    DT_TECHNIQUES = [
        'use_regime_aware_expert',
        'use_smart_oracle',
        'use_ohem',
        'use_rdrop',
        'use_fgm',
        'use_dynamic_loss_weights',
        'use_label_smoothing',
        'use_curriculum',
        'use_ema',
    ]
    
    # Sentiment v2.2 可消融技术
    SENTIMENT_TECHNIQUES = [
        'use_contrastive',
        'use_ohem',
        'use_rdrop',
        'use_fgm',
        'use_dynamic_loss_weights',
        'use_focal_loss',
        'use_label_smoothing',
        'use_mixup',
        'use_ema',
        'use_tta',
    ]
    
    def __init__(self, model_type: str = 'drl'):
        """
        Args:
            model_type: 'drl' 或 'sentiment'
        """
        self.model_type = model_type
        self.techniques = self.DT_TECHNIQUES if model_type == 'drl' else self.SENTIMENT_TECHNIQUES
        self.results: List[AblationResult] = []
    
    def run(
        self,
        base_config: dict,
        train_fn: Callable,
        eval_fn: Callable,
        epochs: int = 30
    ) -> Dict:
        """
        运行消融实验
        
        Args:
            base_config: 基础配置字典
            train_fn: 训练函数 (config) -> model
            eval_fn: 评估函数 (model) -> accuracy
            epochs: 每次实验的轮数
        
        Returns:
            消融结果
        """
        self.results = []
        
        # 1. 完整模型基线
        logger.info("="*60)
        logger.info("测试完整模型 (所有技术启用)...")
        logger.info("="*60)
        
        full_config = base_config.copy()
        full_config['epochs'] = epochs
        
        start = time.time()
        full_model = train_fn(full_config)
        full_time = time.time() - start
        full_acc = eval_fn(full_model)
        
        logger.info(f"完整模型准确率: {full_acc:.4f} ({full_time/60:.1f} min)")
        
        self.results.append(AblationResult(
            technique='FULL',
            enabled=True,
            accuracy=full_acc,
            contribution=0.0,
            training_time=full_time
        ))
        
        # 2. 逐一禁用技术
        for tech in self.techniques:
            if tech not in base_config:
                continue
                
            logger.info("="*60)
            logger.info(f"禁用 {tech}...")
            logger.info("="*60)
            
            config = base_config.copy()
            config['epochs'] = epochs
            config[tech] = False
            
            start = time.time()
            model = train_fn(config)
            train_time = time.time() - start
            acc = eval_fn(model)
            
            contribution = full_acc - acc
            
            self.results.append(AblationResult(
                technique=tech,
                enabled=False,
                accuracy=acc,
                contribution=contribution,
                training_time=train_time
            ))
            
            status = "ON" if contribution > 0.001 else "[WARN]️" if contribution > 0 else "OFF"
            logger.info(f"{status} {tech}: acc={acc:.4f}, 贡献={contribution:+.4f}")
        
        return self._generate_report()
    
    def _generate_report(self) -> Dict:
        """生成消融报告"""
        full_result = self.results[0]
        technique_results = self.results[1:]
        
        # 按贡献排序
        sorted_results = sorted(technique_results, key=lambda x: -x.contribution)
        
        report = {
            'full_accuracy': full_result.accuracy,
            'contributions': {},
            'sorted_techniques': []
        }
        
        logger.info("\n" + "="*60)
        logger.info("消融实验结果")
        logger.info("="*60)
        logger.info(f"完整模型准确率: {full_result.accuracy:.4f}")
        logger.info("-"*60)
        
        for r in sorted_results:
            status = "ON" if r.contribution > 0.005 else "[WARN]️" if r.contribution > 0 else "OFF"
            logger.info(f"{status} {r.technique}: {r.contribution:+.4f} ({r.accuracy:.4f})")
            
            report['contributions'][r.technique] = r.contribution
            report['sorted_techniques'].append({
                'technique': r.technique,
                'contribution': r.contribution,
                'accuracy': r.accuracy,
                'significant': r.contribution > 0.005
            })
        
        # 总结
        significant = [t for t in report['sorted_techniques'] if t['significant']]
        insignificant = [t for t in report['sorted_techniques'] if not t['significant']]
        
        logger.info("-"*60)
        logger.info(f"有效技术: {len(significant)}/{len(technique_results)}")
        
        if insignificant:
            logger.warning(f"[WARN]️ 以下技术贡献不明显，可考虑禁用:")
            for t in insignificant:
                logger.warning(f"   - {t['technique']} ({t['contribution']:+.4f})")
        
        return report


# =============================================================================
# P1: Walk-Forward 验证
# =============================================================================

class WalkForwardValidator:
    """
    Walk-Forward 验证器
    
    滚动训练-测试，更真实地评估模型性能
    """
    
    def __init__(
        self,
        train_window: int = 180 * 24,  # 180 天
        test_window: int = 30 * 24,    # 30 天
        step: int = 7 * 24,            # 7 天步进
        gap: int = 24                  # 1 天间隔
    ):
        self.train_window = train_window
        self.test_window = test_window
        self.step = step
        self.gap = gap
        self.fold_results = []
    
    def run(
        self,
        df: pd.DataFrame,
        train_fn: Callable,
        eval_fn: Callable
    ) -> Dict:
        """
        运行 Walk-Forward 验证
        
        Args:
            df: 数据框
            train_fn: 训练函数 (train_data) -> model
            eval_fn: 评估函数 (model, test_data) -> metrics_dict
        
        Returns:
            聚合结果
        """
        n = len(df)
        self.fold_results = []
        
        total_folds = (n - self.train_window - self.test_window - self.gap) // self.step
        logger.info(f"Walk-Forward 验证: {total_folds} folds")
        logger.info(f"  训练窗口: {self.train_window} ({self.train_window/24:.0f} 天)")
        logger.info(f"  测试窗口: {self.test_window} ({self.test_window/24:.0f} 天)")
        logger.info(f"  步进: {self.step} ({self.step/24:.0f} 天)")
        
        for start in range(0, n - self.train_window - self.test_window - self.gap, self.step):
            train_end = start + self.train_window
            test_start = train_end + self.gap
            test_end = test_start + self.test_window
            
            if test_end > n:
                break
            
            fold_num = len(self.fold_results) + 1
            
            train_data = df.iloc[start:train_end]
            test_data = df.iloc[test_start:test_end]
            
            logger.info(f"\nFold {fold_num}/{total_folds}:")
            logger.info(f"  训练: [{start}:{train_end}]")
            logger.info(f"  测试: [{test_start}:{test_end}]")
            
            # 训练
            start_time = time.time()
            model = train_fn(train_data)
            train_time = time.time() - start_time
            
            # 评估
            metrics = eval_fn(model, test_data)
            metrics['fold'] = fold_num
            metrics['train_start'] = start
            metrics['test_start'] = test_start
            metrics['train_time'] = train_time
            
            self.fold_results.append(metrics)
            
            acc = metrics.get('accuracy', metrics.get('sharpe', 0))
            logger.info(f"  结果: {acc:.4f} ({train_time/60:.1f} min)")
        
        return self._aggregate_results()
    
    def _aggregate_results(self) -> Dict:
        """聚合所有 fold 的结果"""
        if not self.fold_results:
            return {}
        
        aggregated = {}
        
        # 数值型指标聚合
        numeric_keys = [k for k in self.fold_results[0].keys() 
                       if isinstance(self.fold_results[0][k], (int, float))]
        
        for key in numeric_keys:
            values = [r[key] for r in self.fold_results if key in r]
            if values:
                aggregated[f'{key}_mean'] = np.mean(values)
                aggregated[f'{key}_std'] = np.std(values)
                aggregated[f'{key}_min'] = np.min(values)
                aggregated[f'{key}_max'] = np.max(values)
        
        aggregated['num_folds'] = len(self.fold_results)
        aggregated['fold_results'] = self.fold_results
        
        # 打印汇总
        logger.info("\n" + "="*60)
        logger.info("Walk-Forward 验证结果")
        logger.info("="*60)
        
        for key in ['accuracy', 'sharpe', 'return']:
            mean_key = f'{key}_mean'
            std_key = f'{key}_std'
            if mean_key in aggregated:
                logger.info(f"{key}: {aggregated[mean_key]:.4f} ± {aggregated[std_key]:.4f}")
        
        return aggregated


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='HMATS Training 改进工具')
    
    parser.add_argument('--check-leakage', type=str, metavar='FILE',
                       help='检查数据泄露 (parquet 文件)')
    parser.add_argument('--ablation', type=str, choices=['drl', 'sentiment'],
                       help='运行消融实验框架示例')
    parser.add_argument('--walkforward', type=str, metavar='FILE',
                       help='运行 Walk-Forward 验证示例')
    parser.add_argument('--lookahead', type=int, default=24,
                       help='数据泄露检测的前瞻时间 (默认: 24)')
    parser.add_argument('--output', type=str,
                       help='输出报告路径')
    
    args = parser.parse_args()
    
    if args.check_leakage:
        # 数据泄露检测
        logger.info(f"📊 检查数据泄露: {args.check_leakage}")
        
        df = pd.read_parquet(args.check_leakage)
        
        # 获取所有数值列作为特征
        feature_cols = [col for col in df.columns 
                       if df[col].dtype in ['float64', 'float32', 'int64', 'int32']
                       and col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        detector = DataLeakageDetector(lookahead=args.lookahead)
        report = detector.check_features(df, feature_cols)
        
        if args.output:
            detector.generate_report(report, args.output)
    
    elif args.ablation:
        # 消融实验示例
        logger.info(f"🔬 消融实验框架 ({args.ablation})")
        
        ablation = AblationStudy(args.ablation)
        logger.info(f"可消融技术: {ablation.techniques}")
        logger.info("\n使用方法:")
        logger.info("  ablation = AblationStudy('drl')")
        logger.info("  results = ablation.run(config, train_fn, eval_fn)")
    
    elif args.walkforward:
        # Walk-Forward 验证示例
        logger.info(f"📈 Walk-Forward 验证: {args.walkforward}")
        
        validator = WalkForwardValidator()
        logger.info(f"配置: train={validator.train_window}h, test={validator.test_window}h")
        logger.info("\n使用方法:")
        logger.info("  validator = WalkForwardValidator()")
        logger.info("  results = validator.run(df, train_fn, eval_fn)")
    
    else:
        parser.print_help()
        print("\n" + "="*60)
        print("改进清单:")
        print("="*60)
        print("\nP0 (已集成):")
        print("  ON torch.compile - 已添加到 DT v3.2 和 Sentiment v2.2")
        print("  ON BF16 精度 - 已添加到 DT v3.2 和 Sentiment v2.2")
        print("  ON Regime 名称映射 - 已添加到 DT v3.2")
        print("  ON 数据泄露检测 - 本工具提供")
        print("\nP1 (已集成/提供):")
        print("  ON 消融实验框架 - 本工具提供")
        print("  ON Walk-Forward 验证 - 本工具提供")
        print("  ON TTA 推理控制 - 已添加到 Sentiment v2.2")


if __name__ == "__main__":
    main()
