"""Faithful ridge_16h across all 3 folds via the REAL TradingEnvFull env
(reuses FullDRLTrainer._evaluate_baselines). Settles era-stability with the
same dynamics/fees that produced the p381 fold_1 +0.50. No RL training."""
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from training.train_drl_full import FullDRLTrainer
from stable_baselines3.common.vec_env import DummyVecEnv

manifest = json.load(open("configs/feature_manifest.json"))
for asset in ("BTC", "ETH", "SOL"):
    df = pd.read_parquet(f"training/training_data/drl_training/{asset}_4H_full.parquet")
    fcols = [c for c in manifest["all_features"] if c in df.columns]
    tr = FullDRLTrainer(data=df, feature_cols=fcols, asset=asset,
                        decision_interval=4, venue="coinbase", fee_side="taker",
                        progress_bar=False, device="cpu")
    folds = tr._get_fold_splits()
    shs = []
    for i, (train_df, val_df) in enumerate(folds):
        eval_env = DummyVecEnv([lambda d=val_df: tr._create_env(d, augment_enabled=False)])
        base = tr._evaluate_baselines(eval_env, ridge_ctx={
            "train_df": train_df, "feature_cols": fcols, "decision_interval": 4})
        r = base.get("ridge_16h", {})
        sh = r.get("sharpe_after_cost")
        shs.append(sh)
        print(f"{asset} fold_{i+1} (recent->old): ridge_16h Sharpe {sh:+.2f} "
              f"pnl% {r.get('return_pct_after_cost',0):+.1f} "
              f"CI[{r.get('sharpe_ci_low',0):+.2f},{r.get('sharpe_ci_high',0):+.2f}] "
              f"excl0={r.get('sharpe_ci_excludes_zero')}", flush=True)
    signs = set(np.sign(s) for s in shs if s not in (None, 0))
    print(f"{asset}: fold Sharpes {[round(s,2) for s in shs]} -> era-stable={len(signs)<=1}\n", flush=True)
