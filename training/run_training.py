#!/usr/bin/env python3
"""
================================================================================
HMATS 完整训练流程编排
================================================================================

训练组件:
  [GMM]       6-Regime GMM 预训练 -> Regime 检测器 (scripts/retrain_gmm.py)
  [DT v3.2]   Decision Transformer -> 方向预测 (MoE, EMA, FGM, OHEM, GradClip)
  [DRL v7]    TQC RL -> 交易决策 (train_drl_full.py, ULTIMATE preset, GradClip)
  [Sentiment] Agent v2.2 -> 情感分析 (EMA, FGM, OHEM)

用法:
    python run_training.py --all              # 完整流程 (BTC + ETH + SOL)
    python run_training.py --quick            # 快速测试
    python run_training.py --gmm              # 只训练 GMM
    python run_training.py --dt               # 只训练 DT
    python run_training.py --drl              # 只训练 DRL v7 (all assets)
    python run_training.py --drl --asset BTC  # 只训练 DRL v7 (单个 asset)
    python run_training.py --sentiment        # 只训练 Sentiment

================================================================================
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import subprocess
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger("HMATS_Training")


class TrainingOrchestrator:
    """训练流程编排器"""

    ASSETS = ["BTC", "ETH", "SOL"]

    def __init__(self, data_dir: str = './training_data', output_dir: str = './models'):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.script_dir = Path(__file__).parent
        # Root dir is one level up from training/
        self.root_dir = self.script_dir.parent

    def check_data(self) -> bool:
        """检查数据是否存在 (4H full parquets for DRL v7)"""
        drl_data_dir = self.root_dir / 'data' / 'drl_training'
        required = [drl_data_dir / f'{asset}_4H_full.parquet' for asset in self.ASSETS]

        missing = [f for f in required if not f.exists()]
        if missing:
            logger.warning("Missing DRL training data:")
            for f in missing:
                logger.warning(f"  - {f}")
            return False

        try:
            import pandas as pd
            for f in required:
                df = pd.read_parquet(f)
                status = "OK" if len(df) >= 4320 else "WARN"
                logger.info(f"  [{status}] {f.name}: {len(df)} rows ({len(df)/6:.0f} days @ 4H)")
        except Exception as e:
            logger.warning(f"Cannot read data: {e}")

        return len(missing) == 0
    
    def run_gmm(self):
        """GMM 6-Regime 预训练 (scripts/retrain_gmm.py)"""
        logger.info("="*60)
        logger.info("[Step 1] 6-Regime GMM Retrain")
        logger.info("="*60)

        cmd = [sys.executable, '-X', 'utf8',
               str(self.root_dir / 'scripts' / 'retrain_gmm.py')]
        return self._run(cmd, "GMM Retrain")
    
    def run_dt(self, epochs: int = 200, batch_size: int = 256, assets: list = None):
        """Decision Transformer v3.2 (per-asset, aligned with TQC 126-dim obs)"""
        logger.info("="*60)
        logger.info("[Step 2] Decision Transformer v3.2 (per-asset)")
        logger.info("="*60)

        assets = assets or self.ASSETS
        results = {}

        for asset in assets:
            logger.info(f"\n  >>> DT v3.2 {asset} Training Starting...")
            cmd = [sys.executable, '-X', 'utf8',
                   str(self.script_dir / 'drl' / 'train_decision_transformer_v32.py'),
                   '--asset', asset,
                   '--epochs', str(epochs), '--batch-size', str(batch_size)]
            results[asset] = self._run(cmd, f"DT v3.2 {asset}")

        success = all(results.values())
        if not success:
            failed = [a for a, ok in results.items() if not ok]
            logger.error(f"DT v3.2 failed for: {failed}")
        return success
    
    def run_drl(self, assets: list = None, quick: bool = False):
        """DRL v7 TQC RL (train_drl_full.py, ULTIMATE preset)"""
        logger.info("="*60)
        logger.info("[Step 3] DRL v7 TQC RL (ULTIMATE preset)")
        logger.info("="*60)

        assets = assets or self.ASSETS
        results = {}

        for asset in assets:
            logger.info(f"\n  >>> {asset} Training Starting...")
            cmd = [
                sys.executable, '-X', 'utf8', '-u',
                str(self.root_dir / 'train_drl_full.py'),
                '--asset', asset,
                '--folds', '3',
                '--no-progress-bar',
            ]
            if quick:
                cmd.extend(['--timesteps', '200000'])

            results[asset] = self._run(cmd, f"DRL v7 {asset}")

        success = all(results.values())
        if not success:
            failed = [a for a, ok in results.items() if not ok]
            logger.error(f"DRL v7 failed for: {failed}")
        return success
    
    def run_sentiment(self, epochs: int = 15, ensemble: int = 1):
        """Sentiment Agent v2.2"""
        logger.info("="*60)
        logger.info("[Step 4] Sentiment Agent v2.2")
        logger.info("="*60)
        
        cmd = [sys.executable, str(self.script_dir / 'sentiment' / 'train_sentiment_agent_v22.py'),
               '--epochs', str(epochs)]
        if ensemble > 1:
            cmd.extend(['--ensemble', str(ensemble)])
        
        return self._run(cmd, "Sentiment v2.2")
    
    def _run(self, cmd: list, name: str) -> bool:
        logger.info(f"运行: {' '.join(cmd)}")
        start = time.time()
        try:
            result = subprocess.run(cmd)
            elapsed = time.time() - start
            if result.returncode == 0:
                logger.info(f"ON {name} 完成 ({elapsed/60:.1f} min)")
                return True
            else:
                logger.error(f"OFF {name} 失败")
                return False
        except Exception as e:
            logger.error(f"OFF {name} 异常: {e}")
            return False
    
    def run_all(self, quick: bool = False):
        """完整流程: GMM -> DT v3.2 -> DRL v7 -> Sentiment v2.2"""
        logger.info("HMATS Full Training Pipeline")
        logger.info(f"  Mode: {'quick test' if quick else 'full training'}")

        if not self.check_data():
            logger.error("Data check failed")
            return False

        results = {}
        start = time.time()

        # Step 1: GMM
        results['gmm'] = self.run_gmm()

        # Step 2: DT v3.2 (depends on GMM for regime labels)
        if results['gmm']:
            results['dt'] = self.run_dt(30 if quick else 200, 128 if quick else 256)

        # Step 3: DRL v7 (depends on GMM for regime column in parquet)
        if results.get('gmm'):
            results['drl'] = self.run_drl(quick=quick)

        # Step 4: Sentiment v2.2 (independent)
        results['sentiment'] = self.run_sentiment(5 if quick else 15, 1 if quick else 3)

        # Summary
        total = time.time() - start
        logger.info("\nResults Summary")
        for k, v in results.items():
            status = "PASS" if v else "FAIL"
            logger.info(f"  [{status}] {k.upper()}")
        logger.info(f"Total time: {total/3600:.1f} hours")

        return all(results.values())


def main():
    parser = argparse.ArgumentParser(description='HMATS Training Orchestrator')
    parser.add_argument('--all', action='store_true', help='Full pipeline (GMM -> DT -> DRL -> Sentiment)')
    parser.add_argument('--quick', action='store_true', help='Quick test (reduced epochs/timesteps)')
    parser.add_argument('--gmm', action='store_true', help='GMM only')
    parser.add_argument('--dt', action='store_true', help='DT v3.2 only')
    parser.add_argument('--drl', action='store_true', help='DRL v7 only (all assets)')
    parser.add_argument('--sentiment', action='store_true', help='Sentiment v2.2 only')
    parser.add_argument('--asset', type=str, default=None,
                        choices=['BTC', 'ETH', 'SOL'],
                        help='Single asset for DT/DRL (default: all)')
    parser.add_argument('--data-dir', default='./training_data')
    parser.add_argument('--output-dir', default='./models')
    args = parser.parse_args()

    orch = TrainingOrchestrator(args.data_dir, args.output_dir)

    if args.all or not any([args.gmm, args.dt, args.drl, args.sentiment]):
        orch.run_all(args.quick)
    else:
        if args.gmm:
            orch.run_gmm()
        if args.dt:
            assets = [args.asset] if args.asset else None
            orch.run_dt(30 if args.quick else 200, assets=assets)
        if args.drl:
            assets = [args.asset] if args.asset else None
            orch.run_drl(assets=assets, quick=args.quick)
        if args.sentiment:
            orch.run_sentiment(5 if args.quick else 15)


if __name__ == "__main__":
    main()
