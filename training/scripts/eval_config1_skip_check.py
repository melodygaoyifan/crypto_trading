#!/usr/bin/env python
"""
Stage 8 Phase B: Config 1 adaptive skip check.

Evaluates NAV% for all 3 folds of Config 1 (Optuna).
Decision rule:
  IF mean Final NAV > $800K (NAV% > +700%)
  AND all 3 folds NAV > initial ($100K)
  AND Mean/Std > 1.0
  -> SKIP Config 2 (lock Config 1)

  OTHERWISE -> must run Config 2
"""
import sys, json
from pathlib import Path
import numpy as np

_TRAINING_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = _TRAINING_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(_TRAINING_DIR))

from models.feature_extractors import get_extractor_class, get_extractor_kwargs

import pandas as pd
import joblib
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import TQC

from train_drl_full import TradingEnvFull, REGIME_NAMES, _load_regime_names
from core.regime_smoother import RegimeSmoother

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("SkipCheck")

CONFIG1_DIR = PROJECT_ROOT / "models" / "retrained" / "BTC" / "config1_optuna"
DATA_PATH = _TRAINING_DIR / "training_data" / "drl_training" / "BTC_4H_full.parquet"
INITIAL_BALANCE = 100_000
N_EPISODES = 10
SKIP_NAV_THRESHOLD = 800_000  # $800K


def load_data_and_splits():
    """Load data, apply smoother, return 3-fold val splits."""
    data = pd.read_parquet(DATA_PATH)
    if "regime" in data.columns:
        smoother = RegimeSmoother(min_persistence=2)
        data = smoother.smooth_column(data, "regime")

    manifest_path = PROJECT_ROOT / "configs" / "feature_manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    feature_cols = [c for c in manifest["all_features"] if c in data.columns]

    # Reproduce fold splits
    n = len(data)
    val_size = int(n * 0.15)
    gap = 42
    splits = []
    for fold_idx in range(3):
        val_end = n - fold_idx * val_size
        val_start = val_end - val_size
        val_df = data.iloc[val_start:val_end].copy()
        splits.append(val_df)

    return data, feature_cols, splits


def eval_nav(model, eval_env, n_episodes=10):
    """Run NAV% evaluation, return per-episode balances."""
    initial = eval_env.get_attr("initial_balance")[0]
    balances = []
    max_dds = []

    for ep in range(n_episodes):
        obs = eval_env.reset()
        done = False
        peak = initial
        max_dd = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, infos = eval_env.step(action)
            done = dones[0]
            bal = infos[0].get("balance", initial)
            if bal > peak:
                peak = bal
            dd = (peak - bal) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
        balances.append(infos[0].get("balance", initial))
        max_dds.append(max_dd * 100)

    return balances, max_dds


def main():
    # Load regime names
    global REGIME_NAMES
    REGIME_NAMES = _load_regime_names(asset="BTC")

    data, feature_cols, val_splits = load_data_and_splits()
    logger.info(f"Data: {len(data)} rows, {len(feature_cols)} features")

    fold_results = []

    for fold_idx in range(1, 4):
        fold_dir = CONFIG1_DIR / f"fold_{fold_idx}" / "logs" / "LSTM_FILM_A"
        best_path = fold_dir / "best_model" / "best_model.zip"
        scaler_path = fold_dir / "feature_scaler.pkl"

        if not best_path.exists():
            logger.warning(f"  Fold {fold_idx}: no best_model at {best_path}")
            fold_results.append(None)
            continue

        logger.info(f"\n--- Fold {fold_idx} ---")

        # Load scaler
        scaler_meta = joblib.load(scaler_path)
        scaler = scaler_meta["scaler"]
        scale_cols = scaler_meta["scale_cols"]

        val_df = val_splits[fold_idx - 1].copy()
        val_df[scale_cols] = scaler.transform(val_df[scale_cols])
        val_df[scale_cols] = val_df[scale_cols].replace(
            [np.inf, -np.inf], 0
        ).fillna(0).clip(-10.0, 10.0)

        def make_eval_env(df=val_df):
            env = TradingEnvFull(
                df=df, feature_cols=feature_cols, reward_mode="classic",
                augment_enabled=False,
                bear_reward_multiplier=1.2, bull_reward_multiplier=1.1,
                sol_vol_multiplier=1.0,
                dd_threshold=0.0866, dd_penalty_mult=2.8243,
                reward_clip=20.0,
                regime_weight_min=0.5962, regime_weight_max=1.5697,
            )
            return Monitor(env)

        eval_env = DummyVecEnv([make_eval_env])
        eval_env = VecFrameStack(eval_env, n_stack=8)

        model = TQC.load(str(best_path), env=eval_env, device="cuda")
        balances, max_dds = eval_nav(model, eval_env, n_episodes=N_EPISODES)

        mean_bal = np.mean(balances)
        nav_pct = (mean_bal - INITIAL_BALANCE) / INITIAL_BALANCE * 100
        mean_dd = np.mean(max_dds)

        fold_results.append({
            "fold": fold_idx,
            "mean_balance": float(mean_bal),
            "nav_pct": float(nav_pct),
            "max_dd_pct": float(mean_dd),
            "all_positive": all(b > INITIAL_BALANCE for b in balances),
        })

        logger.info(f"  NAV%: {nav_pct:+.2f}%  Final: ${mean_bal:,.0f}  MaxDD: {mean_dd:.2f}%")

        del model
        eval_env.close()
        import torch, gc
        torch.cuda.empty_cache()
        gc.collect()

    # Filter out None (missing folds)
    valid = [r for r in fold_results if r is not None]

    if len(valid) < 3:
        logger.error(f"\n  Only {len(valid)}/3 folds completed. Cannot evaluate.")
        print(f"\nDECISION: WAIT - only {len(valid)}/3 folds done")
        return

    # Compute aggregate metrics
    mean_nav = np.mean([r["mean_balance"] for r in valid])
    std_nav = np.std([r["mean_balance"] for r in valid])
    mean_nav_pct = np.mean([r["nav_pct"] for r in valid])
    all_positive = all(r["all_positive"] for r in valid)
    mean_std_ratio = mean_nav / std_nav if std_nav > 0 else float("inf")
    mean_dd = np.mean([r["max_dd_pct"] for r in valid])

    # Print summary
    print(f"\n{'=' * 70}")
    print(f"  CONFIG 1 (OPTUNA) - 3-FOLD NAV% EVALUATION")
    print(f"{'=' * 70}")
    print(f"{'Fold':<8} {'NAV%':>10} {'Final NAV':>14} {'MaxDD':>8} {'All>100K':>10}")
    print(f"{'-' * 70}")
    for r in valid:
        print(f"  {r['fold']:<6} {r['nav_pct']:>+9.2f}% ${r['mean_balance']:>12,.0f} "
              f"{r['max_dd_pct']:>7.2f}% {'YES' if r['all_positive'] else 'NO':>9}")
    print(f"{'-' * 70}")
    print(f"  {'MEAN':<6} {mean_nav_pct:>+9.2f}% ${mean_nav:>12,.0f} "
          f"{mean_dd:>7.2f}%")
    print(f"  STD:  ${std_nav:>12,.0f}")
    print(f"  Mean/Std ratio: {mean_std_ratio:.2f}")
    print(f"  All folds positive: {all_positive}")

    # Decision
    print(f"\n  --- SKIP CHECK ---")
    print(f"  Mean NAV: ${mean_nav:,.0f}  (threshold: ${SKIP_NAV_THRESHOLD:,.0f})")
    print(f"  All positive: {all_positive}  (required: True)")
    print(f"  Mean/Std: {mean_std_ratio:.2f}  (required: >1.0)")

    skip = (mean_nav > SKIP_NAV_THRESHOLD and all_positive and mean_std_ratio > 1.0)

    if skip:
        print(f"\n  ON DECISION: SKIP Config 2 - Lock Config 1 (Optuna)")
        print(f"  -> Kill background task to prevent Config 2 from running")
    else:
        reasons = []
        if mean_nav <= SKIP_NAV_THRESHOLD:
            reasons.append(f"Mean NAV ${mean_nav:,.0f} <= ${SKIP_NAV_THRESHOLD:,.0f}")
        if not all_positive:
            reasons.append("Some folds have NAV < $100K")
        if mean_std_ratio <= 1.0:
            reasons.append(f"Mean/Std {mean_std_ratio:.2f} <= 1.0")
        print(f"\n  OFF DECISION: MUST RUN Config 2")
        print(f"  Reasons: {'; '.join(reasons)}")

    # Save results
    out_path = PROJECT_ROOT / "reports" / "stage8_config1_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "config": "config1_optuna",
        "folds": valid,
        "mean_nav": float(mean_nav),
        "mean_nav_pct": float(mean_nav_pct),
        "std_nav": float(std_nav),
        "mean_std_ratio": float(mean_std_ratio),
        "mean_max_dd": float(mean_dd),
        "all_positive": bool(all_positive),
        "skip_config2": bool(skip),
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"\n  Results saved: {out_path}")


if __name__ == "__main__":
    main()