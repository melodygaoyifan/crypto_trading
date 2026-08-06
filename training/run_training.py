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

    def __init__(self, data_dir: str = './training_data', output_dir: str = './models',
                 venue: str = 'kraken', fee_side: str = 'taker'):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # [P189] Passed through to train_drl_full.py. Same defaults as
        # training/Makefile's DRL_VENUE/DRL_FEE_SIDE.
        self.venue = venue
        self.fee_side = fee_side

        self.script_dir = Path(__file__).parent
        # Root dir is one level up from training/
        self.root_dir = self.script_dir.parent

    # [P189] Every script this launcher can invoke, resolved once, in one place.
    #
    # Two of these were wrong and had been for as long as the file has existed:
    #   run_gmm -> root_dir/scripts/retrain_gmm.py   (only copy is in
    #              archive/gmm_research/; it trains the GLOBAL 6-component model
    #              that main.py:3552 treats as the legacy fallback)
    #   run_drl -> root_dir/train_drl_full.py        (off by one directory —
    #              the file is in training/, i.e. script_dir)
    # So `make all`, `make quick` and `make gmm` could not complete. The failure
    # was a subprocess returncode 2 and a shell "can't open file" line, arriving
    # after however many hours the preceding steps took. Same shape as [P186],
    # which found `make drl` pointing at a script that never existed.
    #
    # run_gmm now points at training/scripts/train_per_asset_gmm.py, which is
    # what main.py:3520-3541 loads FIRST ("Try per-asset models first (v7)") and
    # what [P164] established as the correct, leak-free trainer — it fits on
    # train folds only, per configs/split_manifest.json.
    SCRIPTS = {
        "gmm":       ("script_dir", "scripts/train_per_asset_gmm.py"),
        "dt":        ("script_dir", "drl/train_decision_transformer_v32.py"),
        "drl":       ("script_dir", "train_drl_full.py"),
        "sentiment": ("script_dir", "sentiment/train_sentiment_agent_v22.py"),
    }

    def _script(self, key: str) -> Path:
        base, rel = self.SCRIPTS[key]
        return (getattr(self, base) / rel).resolve()

    def preflight(self) -> bool:
        """[P189] Verify every script exists BEFORE running any of them.

        A pipeline that discovers a bad path in step 3 has already spent the
        cost of steps 1 and 2. This is a second's work and turns "can't open
        file" into a message that names the key, the resolved path, and the
        fact that nothing has run yet.
        """
        missing = [(k, self._script(k)) for k in self.SCRIPTS
                   if not self._script(k).exists()]
        if missing:
            logger.error("=" * 60)
            logger.error("[preflight] Refusing to start: %d training script(s) "
                         "do not exist.", len(missing))
            for k, p in missing:
                logger.error("  %-10s -> %s", k, p)
            logger.error("Nothing has been trained. Fix TrainingOrchestrator.SCRIPTS or "
                         "restore the file; do not comment out the step.")
            logger.error("=" * 60)
            return False
        return True

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
        """Per-asset GMM (training/scripts/train_per_asset_gmm.py).

        [P189] Was scripts/retrain_gmm.py, which is not in the tree. See
        TrainingOrchestrator.SCRIPTS for why this is the per-asset trainer and not the
        archived global one.
        """
        logger.info("="*60)
        logger.info("[Step 1] Per-asset GMM (BIC k=3-8, train folds only)")
        logger.info("="*60)

        cmd = [sys.executable, '-X', 'utf8', str(self._script("gmm"))]
        return self._run(cmd, "Per-asset GMM")
    
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
                   str(self._script("dt")),  # [P189] via SCRIPTS, so preflight
                                             # checks the path that is used
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
                str(self._script("drl")),   # [P189] was root_dir, file is in training/
                '--asset', asset,
                '--folds', '3',
                '--no-progress-bar',
                # [P189] Explicit, matching training/Makefile. After [P179] the
                # env charges real venue fees, so which venue was charged has to
                # be visible in the command that produced the model rather than
                # left to a default that can move underneath it.
                '--venue', self.venue,
                '--fee-side', self.fee_side,
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
        
        cmd = [sys.executable, str(self._script("sentiment")),  # [P189] see run_dt
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
    # [P189] Mirrors training/Makefile DRL_VENUE / DRL_FEE_SIDE.
    parser.add_argument('--venue', default='kraken',
                        help='Venue whose fees the DRL env charges (P179)')
    parser.add_argument('--fee-side', default='taker', choices=['maker', 'taker'])
    args = parser.parse_args()

    orch = TrainingOrchestrator(args.data_dir, args.output_dir,
                                venue=args.venue, fee_side=args.fee_side)

    # [P189] Before anything runs, not after step 3 has already burned hours.
    if not orch.preflight():
        return 1

    # [P189] These return values used to be discarded and main() returned None,
    # so the process exited 0 whether the pipeline succeeded or every stage
    # failed. `make all` reported success either way — a run that cannot fail
    # tells you nothing about the models it did or did not produce.
    ok = True
    if args.all or not any([args.gmm, args.dt, args.drl, args.sentiment]):
        ok = orch.run_all(args.quick)
    else:
        if args.gmm:
            ok = orch.run_gmm() and ok
        if args.dt:
            assets = [args.asset] if args.asset else None
            ok = orch.run_dt(30 if args.quick else 200, assets=assets) and ok
        if args.drl:
            assets = [args.asset] if args.asset else None
            ok = orch.run_drl(assets=assets, quick=args.quick) and ok
        if args.sentiment:
            ok = orch.run_sentiment(5 if args.quick else 15) and ok

    if not ok:
        logger.error("Training pipeline FAILED — see the per-stage results "
                     "above. Exiting 1.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
