#!/usr/bin/env python3
"""
================================================================================
Stage 7 NAV% Retest - Scale-Invariant Reward Mode Comparison (Final Version)
================================================================================

Uses NAV% (final_balance - initial) / initial * 100 to fairly compare 5 reward
modes.  Raw SB3 eval reward is NOT comparable across modes because PnL weights
differ (classic≈1.0, sharpe≈0.4, sortino=multi-objective).

All experiments use identical model config (FiLM PosA, 200K steps, BTC fold_1).
Only reward_mode and augment vary.

Extra diagnostics per experiment:
  - trade_count:  direction-change count
  - avg_exposure: mean |position_ratio|
  - flat_pct:     % of steps with |position| < 0.01

Usage:
    python -X utf8 -u scripts/run_stage7_nav_retest.py --experiment classic
    python -X utf8 -u scripts/run_stage7_nav_retest.py --experiment all
    python -X utf8 -u scripts/run_stage7_nav_retest.py --analyze-only

================================================================================
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

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
# _TRAINING_DIR first: training/models/ has feature_extractors.py
# (root/models/ is a package but feature_extractors.py was moved to training/)

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import (
    BaseCallback, EvalCallback, CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import TQC

# Import models BEFORE train_drl_full - train_drl_full re-inserts PROJECT_ROOT
# at sys.path[0], which would shadow training/models/ with root/models/ (missing
# feature_extractors.py). By importing first, Python caches the correct module.
from models.feature_extractors import (
    get_extractor_class, get_extractor_kwargs,
    SEQUENCE_ARCHS, DEFAULT_N_STACK,
)

from train_drl_full import (
    TradingEnvFull,
    REGIME_NAMES,
    BEAR_REGIMES,
    BULL_REGIMES,
    ULTIMATE_PRESET,
    ASSET_OVERRIDES,
    _load_regime_names,
    GradientMonitorCallback,
    FPSLoggerCallback,
)
from core.regime_smoother import RegimeSmoother

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("Stage7_NAV")


# =============================================================================
# Constants
# =============================================================================

ASSET = "BTC"
DATA_PATH = _TRAINING_DIR / "training_data" / "drl_training" / "BTC_4H_full.parquet"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "stage7_nav_retest"

EXPERIMENTS = {
    "classic":     {"reward_mode": "classic",  "augment": False},
    "sharpe":      {"reward_mode": "sharpe",   "augment": False},
    "sortino":     {"reward_mode": "sortino",  "augment": False},
    "classic_aug": {"reward_mode": "classic",  "augment": True},
    "sharpe_aug":  {"reward_mode": "sharpe",   "augment": True},
}

# Fixed config - matches Stage 6 winner (FiLM PosA) + ULTIMATE base
FIXED_CONFIG = {
    "extractor":        "lstm_film_a",
    "n_stack":          8,              # DEFAULT_N_STACK
    "timesteps":        200_000,
    "eval_freq":        5_000,
    "n_eval_episodes":  10,             # double Optuna's 5 for lower variance
    "learning_rate":    3e-5,           # BTC default
    "buffer_size":      2_000_000,
    "learning_starts":  30_000,
    "batch_size":       256,
    "gradient_steps":   4,
    "ent_coef":         0.1,            # FIXED - NEVER "auto"
    "tau":              0.005,
    "gamma":            0.99,
    "n_quantiles":      32,
    "n_critics":        2,
    "top_quantiles_to_drop": 2,
    "net_arch":         [256, 256],     # Custom extractor path uses [256,256] (train_drl_full.py:971)
                                        # ULTIMATE [384,384,256] is for flat MLP only (no extractor)
    "bear_mult":        1.2,
    "bull_mult":        1.1,
    "sol_vol_mult":     1.0,            # BTC (not SOL)
    "smooth_regimes":   2,
}


# =============================================================================
# Data Loading (mirrors train_drl_full.py main())
# =============================================================================

def load_data() -> Tuple[pd.DataFrame, List[str], set]:
    """Load parquet, apply RegimeSmoother, resolve features."""
    logger.info(f"Loading data: {DATA_PATH}")
    data = pd.read_parquet(DATA_PATH)
    logger.info(f"  Loaded: {len(data)} rows, {len(data.columns)} columns")

    # RegimeSmoother (persistence=2, matches training)
    if "regime" in data.columns:
        smoother = RegimeSmoother(min_persistence=FIXED_CONFIG["smooth_regimes"])
        before = sum(1 for i in range(1, len(data))
                     if data["regime"].iloc[i] != data["regime"].iloc[i - 1])
        data = smoother.smooth_column(data, "regime")
        after = sum(1 for i in range(1, len(data))
                    if data["regime"].iloc[i] != data["regime"].iloc[i - 1])
        logger.info(f"  RegimeSmoother: flips {before} -> {after}")

    # Feature columns from manifest
    manifest_path = PROJECT_ROOT / "configs" / "feature_manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        feature_cols = [c for c in manifest["all_features"] if c in data.columns]
        no_scale = set(manifest.get("no_scale_features", []))
        logger.info(f"  Features: {len(feature_cols)} (manifest), "
                     f"{len(no_scale)} no-scale")
    else:
        raise FileNotFoundError(f"feature_manifest.json not found at {manifest_path}")

    return data, feature_cols, no_scale


def get_fold1_split(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Reproduce exact fold_1 split from train_drl_full.py."""
    n = len(data)
    val_size = int(n * 0.15)
    gap = 42  # 42 × 4H = 1 week
    # fold_idx=0 -> latest data
    val_end = n
    val_start = val_end - val_size
    train_end = val_start - gap
    train_df = data.iloc[:train_end].copy()
    val_df = data.iloc[val_start:val_end].copy()
    logger.info(f"  Fold 1: train={len(train_df)} rows, val={len(val_df)} rows, "
                f"gap={gap}")
    return train_df, val_df


def scale_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str],
    no_scale: set,
    save_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """RobustScaler: fit on train, transform both. Save scaler."""
    scale_cols = [c for c in feature_cols if c not in no_scale]
    pass_cols = [c for c in feature_cols if c in no_scale]

    scaler = RobustScaler()
    scaler.fit(train_df[scale_cols])
    scaler.scale_ = np.where(scaler.scale_ == 0, 1.0, scaler.scale_)

    train_scaled = train_df.copy()
    val_scaled = val_df.copy()
    train_scaled[scale_cols] = scaler.transform(train_df[scale_cols])
    val_scaled[scale_cols] = scaler.transform(val_df[scale_cols])

    for df in [train_scaled, val_scaled]:
        df[scale_cols] = df[scale_cols].replace([np.inf, -np.inf], 0).fillna(0)
        df[scale_cols] = df[scale_cols].clip(-10.0, 10.0)

    scaler_meta = {"scaler": scaler, "scale_cols": scale_cols, "pass_cols": pass_cols}
    joblib.dump(scaler_meta, save_dir / "feature_scaler.pkl")
    logger.info(f"  Scaler saved: {save_dir / 'feature_scaler.pkl'}")

    return train_scaled, val_scaled


# =============================================================================
# NAV% Evaluation
# =============================================================================

def eval_nav_pct(
    model, eval_env, n_episodes: int = 10,
) -> Dict:
    """
    NAV% = (mean_final_balance - initial_balance) / initial_balance * 100

    Uses VecEnv API (dones[0], infos[0]).
    """
    initial = eval_env.get_attr("initial_balance")[0]
    balances = []
    for ep in range(n_episodes):
        obs = eval_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, infos = eval_env.step(action)
            done = dones[0]
        # Validate info keys on first episode
        if ep == 0:
            info_keys = list(infos[0].keys())
            logger.info(f"  [NAV] info keys: {info_keys}")
            if "balance" not in infos[0]:
                logger.error("  [NAV] 'balance' key MISSING from info dict!")
        final_balance = infos[0].get("balance", initial)
        balances.append(final_balance)

    mean_bal = float(np.mean(balances))
    return {
        "nav_pct": (mean_bal - initial) / initial * 100,
        "nav_std": float(np.std(balances)) / initial * 100,
        "balances": [float(b) for b in balances],
    }


# =============================================================================
# Diagnostics (trade_count, avg_exposure, flat_pct)
# =============================================================================

def collect_diagnostics(
    model, eval_env, n_episodes: int = 10,
) -> Dict:
    """
    Collect behavioral stats from eval episodes (VecEnv API).
    - trade_count:  direction changes (excl. initial flat->position)
    - avg_exposure: mean |position_ratio|
    - flat_pct:     % of steps with |position| < 0.01
    """
    total_steps = 0
    total_trades = 0
    total_exposure = 0.0
    total_flat = 0

    for ep in range(n_episodes):
        obs = eval_env.reset()
        done = False
        prev_direction = 0
        _first_step = True
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, infos = eval_env.step(action)
            done = dones[0]
            info = infos[0]

            # Validate on very first step of first episode
            if ep == 0 and _first_step:
                if "position" not in info:
                    logger.error("  [DIAG] 'position' key MISSING from info dict!")
                _first_step = False

            pos = info.get("position", 0.0)
            direction = 1 if pos > 0.01 else (-1 if pos < -0.01 else 0)

            if direction != prev_direction and prev_direction != 0:
                total_trades += 1
            prev_direction = direction

            total_exposure += abs(pos)
            if abs(pos) < 0.01:
                total_flat += 1
            total_steps += 1

    return {
        "trade_count": total_trades,
        "avg_exposure": total_exposure / max(total_steps, 1),
        "flat_pct": total_flat / max(total_steps, 1) * 100,
        "total_steps": total_steps,
    }


# =============================================================================
# Single Experiment
# =============================================================================

def run_experiment(
    name: str,
    config: Dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str],
    no_scale: set,
    device: str,
) -> Dict:
    """Train one TQC model and evaluate with NAV% + diagnostics."""
    reward_mode = config["reward_mode"]
    augment = config["augment"]
    exp_dir = OUTPUT_DIR / name
    exp_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'=' * 70}")
    logger.info(f"  EXPERIMENT: {name}")
    logger.info(f"  reward_mode={reward_mode}, augment={augment}")
    logger.info(f"{'=' * 70}")

    # Scale features (separate scaler per experiment for reproducibility)
    train_scaled, val_scaled = scale_features(
        train_df, val_df, feature_cols, no_scale, exp_dir,
    )

    obs_dim = len(feature_cols) + 4  # 126

    # Create envs
    def make_train_env(df=train_scaled):
        env = TradingEnvFull(
            df=df,
            feature_cols=feature_cols,
            reward_mode=reward_mode,
            augment_enabled=augment,
            bear_reward_multiplier=FIXED_CONFIG["bear_mult"],
            bull_reward_multiplier=FIXED_CONFIG["bull_mult"],
            sol_vol_multiplier=FIXED_CONFIG["sol_vol_mult"],
        )
        return Monitor(env)

    def make_eval_env(df=val_scaled):
        env = TradingEnvFull(
            df=df,
            feature_cols=feature_cols,
            reward_mode=reward_mode,  # same as train (doesn't affect NAV)
            augment_enabled=False,    # NEVER augment eval
            bear_reward_multiplier=FIXED_CONFIG["bear_mult"],
            bull_reward_multiplier=FIXED_CONFIG["bull_mult"],
            sol_vol_multiplier=FIXED_CONFIG["sol_vol_mult"],
        )
        return Monitor(env)

    train_env = DummyVecEnv([make_train_env])
    eval_env = DummyVecEnv([make_eval_env])

    # VecFrameStack for sequence extractor
    n_stack = FIXED_CONFIG["n_stack"]
    train_env = VecFrameStack(train_env, n_stack=n_stack)
    eval_env = VecFrameStack(eval_env, n_stack=n_stack)
    logger.info(f"  VecFrameStack: n_stack={n_stack}, stacked obs={obs_dim * n_stack}")

    # Smoke test
    _obs = train_env.reset()
    expected_shape = (1, obs_dim * n_stack)
    assert _obs.shape == expected_shape, f"Obs shape {_obs.shape} != {expected_shape}"
    for _step in range(10):
        _act = np.array([[np.random.uniform(-1, 1)]], dtype=np.float32)
        _obs, _rew, _done, _info = train_env.step(_act)
        assert not np.isnan(_obs).any(), f"NaN at smoke step {_step}"
    train_env.reset()
    logger.info(f"  Smoke test: OK, obs={_obs.shape}")

    # Build TQC model with FiLM PosA extractor
    extractor_name = FIXED_CONFIG["extractor"]
    extractor_cls = get_extractor_class(extractor_name)
    extractor_kwargs = get_extractor_kwargs(extractor_name, single_obs_dim=obs_dim)

    # n_critics / n_quantiles are TQCPolicy params -> go in policy_kwargs
    # (top_quantiles_to_drop_per_net is the only TQC model-level param)
    policy_kwargs = {
        "features_extractor_class": extractor_cls,
        "features_extractor_kwargs": extractor_kwargs,
        "net_arch": FIXED_CONFIG["net_arch"],  # [256,256] - correct for custom extractor
        "n_critics": FIXED_CONFIG["n_critics"],
        "n_quantiles": FIXED_CONFIG["n_quantiles"],
    }

    model = TQC(
        "MlpPolicy",
        env=train_env,
        learning_rate=FIXED_CONFIG["learning_rate"],
        buffer_size=FIXED_CONFIG["buffer_size"],
        learning_starts=FIXED_CONFIG["learning_starts"],
        batch_size=FIXED_CONFIG["batch_size"],
        tau=FIXED_CONFIG["tau"],
        gamma=FIXED_CONFIG["gamma"],
        train_freq=1,
        gradient_steps=FIXED_CONFIG["gradient_steps"],
        ent_coef=FIXED_CONFIG["ent_coef"],
        target_update_interval=1,
        top_quantiles_to_drop_per_net=FIXED_CONFIG["top_quantiles_to_drop"],
        policy_kwargs=policy_kwargs,
        device=device,
        verbose=1,
        tensorboard_log=str(exp_dir / "tb"),
    )

    total_params = sum(p.numel() for p in model.policy.parameters())
    logger.info(f"  Parameters: {total_params:,}")

    # Callbacks - ensure dirs exist (guard against external deletion)
    best_model_dir = exp_dir / "best_model"
    best_model_dir.mkdir(parents=True, exist_ok=True)
    eval_log_dir = exp_dir / "eval"
    eval_log_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        GradientMonitorCallback(max_actor_loss=1e6, max_consecutive_explosions=3),
        FPSLoggerCallback(log_freq=50_000),
        EvalCallback(
            eval_env,
            best_model_save_path=str(best_model_dir),
            log_path=str(exp_dir / "eval"),
            eval_freq=FIXED_CONFIG["eval_freq"],
            deterministic=True,
            render=False,
        ),
    ]

    # Train
    start_time = time.time()
    try:
        model.learn(
            total_timesteps=FIXED_CONFIG["timesteps"],
            callback=callbacks,
            progress_bar=False,
        )
    except Exception as e:
        logger.error(f"  Training failed: {e}")
        import traceback
        traceback.print_exc()
        train_env.close()
        eval_env.close()
        return {"experiment": name, "error": str(e)}

    elapsed = time.time() - start_time
    logger.info(f"  Training done: {elapsed / 60:.1f} min")

    # Save final model
    model.save(str(exp_dir / "final_model"))

    # Load best_model (peak eval reward, less overfitted)
    best_path = best_model_dir / "best_model.zip"
    if best_path.exists():
        logger.info(f"  Loading best_model for evaluation")
        best_model = TQC.load(str(best_path), env=eval_env, device=device)
    else:
        logger.warning(f"  No best_model found, using final_model")
        best_model = model

    # Evaluate: NAV%
    nav_result = eval_nav_pct(best_model, eval_env,
                              n_episodes=FIXED_CONFIG["n_eval_episodes"])
    logger.info(f"  NAV%: {nav_result['nav_pct']:+.2f} +/- {nav_result['nav_std']:.2f}")

    # Evaluate: raw reward (for reference only)
    raw_rewards = []
    for _ in range(FIXED_CONFIG["n_eval_episodes"]):
        obs = eval_env.reset()
        done = False
        total_r = 0.0
        while not done:
            action, _ = best_model.predict(obs, deterministic=True)
            obs, rew, dones, infos = eval_env.step(action)
            total_r += rew[0]
            done = dones[0]
        raw_rewards.append(total_r)
    raw_mean = float(np.mean(raw_rewards))
    raw_std = float(np.std(raw_rewards))
    logger.info(f"  Raw reward: {raw_mean:+.2f} +/- {raw_std:.2f}")

    # Diagnostics
    diag = collect_diagnostics(best_model, eval_env,
                               n_episodes=FIXED_CONFIG["n_eval_episodes"])
    logger.info(f"  Trades: {diag['trade_count']}  "
                f"Exposure: {diag['avg_exposure']:.3f}  "
                f"Flat: {diag['flat_pct']:.1f}%")

    # Cleanup - best_model may or may not be a separate object from model
    _model_ref = model
    try:
        if best_model is not _model_ref:
            del best_model
    except NameError:
        pass
    del _model_ref, model
    train_env.close()
    eval_env.close()
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "experiment": name,
        "reward_mode": reward_mode,
        "augment": augment,
        "nav_pct": round(nav_result["nav_pct"], 4),
        "nav_std": round(nav_result["nav_std"], 4),
        "raw_reward": round(raw_mean, 2),
        "raw_reward_std": round(raw_std, 2),
        "trade_count": diag["trade_count"],
        "avg_exposure": round(diag["avg_exposure"], 4),
        "flat_pct": round(diag["flat_pct"], 2),
        "total_steps": diag["total_steps"],
        "elapsed_s": round(elapsed, 1),
        "balances": nav_result["balances"],
    }


# =============================================================================
# Analysis & Decision
# =============================================================================

def analyze_results(results: List[Dict]) -> Dict:
    """Apply decision matrix from the plan."""
    print(f"\n{'=' * 70}")
    print("  STAGE 7 NAV% RETEST - RESULTS")
    print(f"{'=' * 70}\n")

    # Header
    header = (f"{'Experiment':15s} {'NAV%':>9s} {'±std':>8s} {'RawRew':>9s} "
              f"{'Trades':>7s} {'Exposure':>9s} {'Flat%':>7s} {'Time':>7s}")
    print(header)
    print("-" * len(header))

    for r in results:
        if "error" in r:
            print(f"{r['experiment']:15s} ERROR: {r['error']}")
            continue
        print(f"{r['experiment']:15s} {r['nav_pct']:+9.2f} {r['nav_std']:8.2f} "
              f"{r['raw_reward']:+9.2f} {r['trade_count']:7d} "
              f"{r['avg_exposure']:9.4f} {r['flat_pct']:7.1f} "
              f"{r['elapsed_s']/60:6.1f}m")

    # Decision logic
    valid = [r for r in results if "error" not in r]
    if not valid:
        print("\n  No valid results - cannot decide")
        return {"winner": None, "decision": "No valid results"}

    classic = next((r for r in valid if r["experiment"] == "classic"), None)
    if not classic:
        print("\n  WARNING: classic baseline missing")
        winner = max(valid, key=lambda r: r["nav_pct"])
        return {"winner": winner["experiment"],
                "decision": "Classic missing - winner by NAV%"}

    c_nav = classic["nav_pct"]
    best_other = max((r for r in valid if r["experiment"] != "classic"),
                     key=lambda r: r["nav_pct"], default=None)

    print(f"\n  --- Decision Matrix ---")
    print(f"  Classic NAV%: {c_nav:+.2f}")

    if best_other:
        b_nav = best_other["nav_pct"]
        print(f"  Best other:   {best_other['experiment']} NAV%: {b_nav:+.2f}")

        if c_nav <= 0 and b_nav <= 0:
            # Both negative - pick least bad
            winner = max(valid, key=lambda r: r["nav_pct"])
            decision = "Both negative - least-bad wins, investigate further"
        elif c_nav > 0 and (b_nav / c_nav if c_nav != 0 else 0) < 0.95:
            winner_name = "classic"
            decision = "Stage 8 unchanged"
            print(f"  Ratio B/C = {b_nav/c_nav:.3f} < 0.95 -> classic WINS")
        elif c_nav > 0 and (b_nav / c_nav if c_nav != 0 else 0) >= 0.95 and \
             (b_nav / c_nav if c_nav != 0 else 0) <= 1.05:
            winner_name = "classic"
            decision = "Stage 8 unchanged (classic simpler, gap small)"
            print(f"  Ratio B/C = {b_nav/c_nav:.3f} ∈ [0.95, 1.05] -> classic wins (simpler)")
        else:
            winner_name = best_other["experiment"]
            ratio = b_nav / c_nav if c_nav != 0 else float("inf")
            decision = f"Stage 8 Tier 2 resarch needed (reward_mode={best_other['reward_mode']})"
            print(f"  Ratio B/C = {ratio:.3f} > 1.05 -> {winner_name} WINS")

        if c_nav > 0 or b_nav > 0:
            winner = next(r for r in valid if r["experiment"] == winner_name)
    else:
        winner = classic
        decision = "Only classic completed"

    # Flat% check - if sharpe/sortino has >50% flat, flag risk-shaping problem
    for r in valid:
        if r["experiment"] != "classic" and r["flat_pct"] > 50:
            print(f"\n  [WARN] {r['experiment']}: flat_pct={r['flat_pct']:.1f}% > 50% "
                  f"-> penalty-induced risk aversion (learned to do nothing)")

    print(f"\n  WINNER:   {winner['experiment']}")
    print(f"  DECISION: {decision}")

    return {
        "winner": winner["experiment"],
        "decision": decision,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Stage 7 NAV% Retest - Fair reward mode comparison"
    )
    parser.add_argument(
        "--experiment", type=str, default="all",
        choices=list(EXPERIMENTS.keys()) + ["all"],
        help="Which experiment to run (default: all)",
    )
    parser.add_argument(
        "--analyze-only", action="store_true",
        help="Skip training, analyze existing results",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device: auto, cuda, cpu",
    )
    args = parser.parse_args()

    # Resolve device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info(f"Device: {device}")
    if device == "cuda":
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / "stage7_nav_retest_results.json"

    if not args.analyze_only:
        # Load data once (shared across all experiments)
        data, feature_cols, no_scale = load_data()

        # Override global REGIME_NAMES
        global REGIME_NAMES
        import train_drl_full
        train_drl_full.REGIME_NAMES = _load_regime_names(asset=ASSET)
        logger.info(f"  Regime names ({ASSET}): k={len(train_drl_full.REGIME_NAMES)} "
                     f"{train_drl_full.REGIME_NAMES}")

        # Fold 1 split
        train_df, val_df = get_fold1_split(data)

        # Save feature_cols for reference
        fc_path = OUTPUT_DIR / "feature_cols.json"
        with open(fc_path, "w") as f:
            json.dump(feature_cols, f, indent=2)

        # Select experiments
        if args.experiment == "all":
            exp_names = list(EXPERIMENTS.keys())
        else:
            exp_names = [args.experiment]

        # Load existing results to merge (don't overwrite previous experiments)
        existing_results = []
        if results_path.exists():
            try:
                with open(results_path) as f:
                    saved = json.load(f)
                existing_results = saved.get("results", [])
                logger.info(f"  Loaded {len(existing_results)} existing results from {results_path}")
            except (json.JSONDecodeError, KeyError):
                logger.warning(f"  Could not load existing results, starting fresh")

        # Build lookup of existing experiments (by name)
        results_by_name = {r["experiment"]: r for r in existing_results}

        # Run experiments sequentially
        total_start = time.time()

        for name in exp_names:
            config = EXPERIMENTS[name]
            result = run_experiment(
                name=name,
                config=config,
                train_df=train_df,
                val_df=val_df,
                feature_cols=feature_cols,
                no_scale=no_scale,
                device=device,
            )
            results_by_name[name] = result

            # Save incrementally (crash-safe) - all experiments merged
            merged = list(results_by_name.values())
            _save_results(merged, results_path)

        results = list(results_by_name.values())

        total_time = time.time() - total_start
        logger.info(f"\n  Total time: {total_time / 3600:.1f} hours")

    else:
        # Analyze-only: load existing results
        if not results_path.exists():
            logger.error(f"No results file: {results_path}")
            sys.exit(1)
        with open(results_path) as f:
            saved = json.load(f)
        results = saved.get("results", [])

    # Analysis
    if results:
        analysis = analyze_results(results)
        _save_results(results, results_path, analysis=analysis)


def _save_results(
    results: List[Dict],
    path: Path,
    analysis: Dict = None,
):
    """Save results JSON (crash-safe: write to tmp then rename)."""
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": FIXED_CONFIG,
        "results": results,
    }
    if analysis:
        output["winner"] = analysis.get("winner")
        output["decision"] = analysis.get("decision")

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(output, f, indent=2, default=str)
    tmp.replace(path)
    logger.info(f"  Results saved: {path}")


if __name__ == "__main__":
    main()