#!/usr/bin/env python3
"""
Stage 4: Architecture comparison - MLP vs LSTM vs GRU vs CNN.

Trains each architecture for N steps on fold_1 (latest data) and compares
eval rewards. Uses VecFrameStack for sequence models.

Usage:
    python scripts/compare_architectures.py --asset BTC --timesteps 200000
    python scripts/compare_architectures.py --asset BTC --timesteps 200000 --archs mlp lstm
    python scripts/compare_architectures.py --asset BTC --timesteps 200000 --reward-mode sortino

Output:
    models/retrained/{ASSET}/arch_comparison/results.json
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.preprocessing import RobustScaler

# ── Path setup ──────────────────────────────────────────────────────────
_TRAINING_DIR = Path(__file__).resolve().parent.parent   # training/
PROJECT_ROOT = _TRAINING_DIR.parent                      # project root
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(_TRAINING_DIR))

try:
    import gymnasium as gym
except ImportError:
    import gym

from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from sb3_contrib import TQC

from core.regime_smoother import RegimeSmoother
from models.feature_extractors import (
    get_extractor_class, get_extractor_kwargs,
    SEQUENCE_ARCHS, DEFAULT_N_STACK,
)

# Import env + constants from train_drl_full
from train_drl_full import (
    TradingEnvFull, ULTIMATE_PRESET, ASSET_OVERRIDES,
    REGIME_NAMES, _load_regime_names,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("ArchCompare")


def load_manifest():
    """Load feature manifest for no-scale info."""
    manifest_path = PROJECT_ROOT / "configs" / "feature_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            return json.load(f)
    return None


def make_env(df, feature_cols, reward_mode="classic",
             bear_reward_multiplier=1.2, bull_reward_multiplier=1.1,
             sol_vol_multiplier=1.0):
    """Create a TradingEnvFull wrapped in Monitor."""
    def _init():
        env = TradingEnvFull(
            df=df,
            feature_cols=feature_cols,
            reward_mode=reward_mode,
            bear_reward_multiplier=bear_reward_multiplier,
            bull_reward_multiplier=bull_reward_multiplier,
            sol_vol_multiplier=sol_vol_multiplier,
        )
        return Monitor(env)
    return _init


def run_architecture(
    arch: str,
    asset: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list,
    obs_dim: int,
    timesteps: int,
    eval_freq: int,
    output_dir: Path,
    device: str,
    reward_mode: str = "classic",
    bear_reward_multiplier: float = 1.2,
    bull_reward_multiplier: float = 1.1,
    sol_vol_multiplier: float = 1.0,
) -> dict:
    """Train one architecture and return results."""

    arch_dir = output_dir / arch
    arch_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*60}")
    logger.info(f"  Architecture: {arch.upper()}")
    logger.info(f"  Asset: {asset} | Steps: {timesteps:,}")
    logger.info(f"  Reward: {reward_mode}")
    logger.info(f"{'='*60}")

    use_stacking = arch in SEQUENCE_ARCHS
    n_stack = DEFAULT_N_STACK if use_stacking else 1

    # Create envs
    env_kwargs = dict(
        reward_mode=reward_mode,
        bear_reward_multiplier=bear_reward_multiplier,
        bull_reward_multiplier=bull_reward_multiplier,
        sol_vol_multiplier=sol_vol_multiplier,
    )
    train_env = DummyVecEnv([make_env(train_df, feature_cols, **env_kwargs)])
    eval_env = DummyVecEnv([make_env(val_df, feature_cols, **env_kwargs)])

    if use_stacking:
        train_env = VecFrameStack(train_env, n_stack=n_stack)
        eval_env = VecFrameStack(eval_env, n_stack=n_stack)
        logger.info(f"  VecFrameStack: n_stack={n_stack}")
        logger.info(f"  Stacked obs dim: {obs_dim * n_stack}")

    # Build policy kwargs
    extractor_cls = get_extractor_class(arch)
    extractor_kwargs = get_extractor_kwargs(arch, single_obs_dim=obs_dim)

    policy_kwargs = {
        "features_extractor_class": extractor_cls,
        "features_extractor_kwargs": extractor_kwargs,
        "net_arch": [256, 256],  # Smaller heads since extractor does heavy lifting
        "n_critics": 2,
        "n_quantiles": 32,
    }

    logger.info(f"  Extractor: {extractor_cls.__name__}")
    logger.info(f"  Extractor kwargs: {extractor_kwargs}")
    logger.info(f"  Net arch (heads): [256, 256]")

    # Create TQC model
    model = TQC(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=ULTIMATE_PRESET["learning_rate"],
        buffer_size=2_000_000,  # Full buffer (128GB RAM available)
        learning_starts=5_000,  # Faster warmup for comparison
        batch_size=ULTIMATE_PRESET["batch_size"],
        tau=ULTIMATE_PRESET["tau"],
        gamma=ULTIMATE_PRESET["gamma"],
        train_freq=1,
        gradient_steps=ULTIMATE_PRESET["gradient_steps"],
        ent_coef=0.1,  # FIXED
        top_quantiles_to_drop_per_net=ULTIMATE_PRESET["top_quantiles_to_drop_per_net"],
        policy_kwargs=policy_kwargs,
        device=device,
        verbose=0,
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.policy.parameters())
    trainable_params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    logger.info(f"  Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Eval callback
    eval_steps = len(val_df) - 10  # window_size=10
    eval_cb = EvalCallback(
        eval_env,
        n_eval_episodes=1,
        eval_freq=eval_freq,
        best_model_save_path=str(arch_dir),
        log_path=str(arch_dir),
        deterministic=True,
        verbose=0,
    )

    # Train
    t_start = time.time()
    model.learn(
        total_timesteps=timesteps,
        callback=eval_cb,
        progress_bar=False,
    )
    train_time = time.time() - t_start

    # Final eval
    rewards = []
    for _ in range(3):
        obs = eval_env.reset()
        done = False
        ep_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = eval_env.step(action)
            ep_reward += reward[0]
            done = dones[0]
        rewards.append(ep_reward)

    mean_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))

    result = {
        "arch": arch,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "train_time_sec": train_time,
        "timesteps": timesteps,
        "n_stack": n_stack,
        "reward_mode": reward_mode,
    }

    logger.info(f"  Result: reward={mean_reward:.2f} +/- {std_reward:.2f}")
    logger.info(f"  Time: {train_time:.0f}s | Params: {trainable_params:,}")

    # Save
    with open(arch_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    # Cleanup
    train_env.close()
    eval_env.close()

    return result


def main():
    parser = argparse.ArgumentParser(description="Stage 4: Architecture Comparison")
    parser.add_argument("--asset", required=True, choices=["BTC", "ETH", "SOL"])
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--archs", nargs="+", default=["mlp", "lstm", "gru", "cnn"],
                        choices=["mlp", "lstm", "gru", "cnn"])
    parser.add_argument("--reward-mode", default="classic", choices=["classic", "sortino"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "models" / "retrained"))
    args = parser.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    data_path = str(_TRAINING_DIR / "training_data" / "drl_training" / f"{args.asset}_4H_full.parquet")
    logger.info(f"Loading: {data_path}")
    data = pd.read_parquet(data_path)

    # Smooth regimes
    if "regime" in data.columns:
        smoother = RegimeSmoother(min_persistence=2)
        data = smoother.smooth_column(data, "regime")

    # Load features from manifest
    manifest = load_manifest()
    if manifest:
        feature_cols = [c for c in manifest["all_features"] if c in data.columns]
        no_scale = set(manifest.get("no_scale_features", []))
    else:
        feature_cols = [
            c for c in data.columns
            if data[c].dtype in ["float64", "float32"]
            and c not in ["timestamp", "open", "high", "low", "close", "volume",
                          "regime", "regime_confidence"]
        ]
        no_scale = set()

    obs_dim = len(feature_cols) + 4  # +4 env state

    # Override REGIME_NAMES per-asset
    import train_drl_full
    train_drl_full.REGIME_NAMES = _load_regime_names(asset=args.asset)
    logger.info(f"Regime names: k={len(train_drl_full.REGIME_NAMES)}")

    # Split: fold_1 (latest)
    n = len(data)
    val_size = int(n * 0.15)
    gap = 42
    train_end = n - val_size - gap
    val_start = train_end + gap

    train_df = data.iloc[:train_end].copy()
    val_df = data.iloc[val_start:].copy()

    # Scale features
    scale_cols = [c for c in feature_cols if c not in no_scale]
    scaler = RobustScaler()
    scaler.fit(train_df[scale_cols])
    scaler.scale_ = np.where(scaler.scale_ == 0, 1.0, scaler.scale_)
    train_df[scale_cols] = scaler.transform(train_df[scale_cols])
    val_df[scale_cols] = scaler.transform(val_df[scale_cols])
    for df_ in [train_df, val_df]:
        df_[scale_cols] = df_[scale_cols].replace([np.inf, -np.inf], 0).fillna(0)
        df_[scale_cols] = df_[scale_cols].clip(-10.0, 10.0)  # Post-scale outlier clip

    # Asset-specific overrides
    overrides = ASSET_OVERRIDES.get(args.asset, {})
    sol_vol_mult = overrides.get("sol_vol_multiplier", 1.0)

    logger.info(f"\nData: {len(train_df)} train, {len(val_df)} val")
    logger.info(f"Features: {len(feature_cols)} | Obs dim: {obs_dim}")
    logger.info(f"Architectures: {args.archs}")
    logger.info(f"Device: {args.device}")
    if sol_vol_mult != 1.0:
        logger.info(f"SOL vol multiplier: {sol_vol_mult}")

    output_dir = Path(args.output) / args.asset / "arch_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run each architecture
    results = []
    for arch in args.archs:
        result = run_architecture(
            arch=arch,
            asset=args.asset,
            train_df=train_df,
            val_df=val_df,
            feature_cols=feature_cols,
            obs_dim=obs_dim,
            timesteps=args.timesteps,
            eval_freq=args.eval_freq,
            output_dir=output_dir,
            device=args.device,
            reward_mode=args.reward_mode,
            sol_vol_multiplier=sol_vol_mult,
        )
        results.append(result)

    # Summary
    logger.info(f"\n{'='*70}")
    logger.info(f"Architecture Comparison Results - {args.asset}")
    logger.info(f"{'='*70}")
    logger.info(f"{'Arch':<8} {'Reward':>10} {'StdDev':>10} {'Params':>12} {'Time':>8}")
    logger.info(f"{'-'*50}")

    best = max(results, key=lambda r: r["mean_reward"])
    for r in sorted(results, key=lambda r: r["mean_reward"], reverse=True):
        marker = " *" if r["arch"] == best["arch"] else ""
        logger.info(f"{r['arch']:<8} {r['mean_reward']:>10.2f} {r['std_reward']:>10.2f} "
                     f"{r['trainable_params']:>12,} {r['train_time_sec']:>7.0f}s{marker}")

    logger.info(f"\nBest: {best['arch'].upper()} (reward={best['mean_reward']:.2f})")

    # Save combined results
    results_path = output_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "asset": args.asset,
            "timesteps": args.timesteps,
            "reward_mode": args.reward_mode,
            "results": results,
            "best_arch": best["arch"],
        }, f, indent=2)
    logger.info(f"Results saved: {results_path}")


if __name__ == "__main__":
    main()
