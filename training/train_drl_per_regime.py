#!/usr/bin/env python3
"""
================================================================================
HMATS v7 - Per-Regime Specialist DRL Training
================================================================================

Trains per-semantic-group TQC specialist models + 1 universal fallback.
GMM regimes are mapped to 6 semantic groups; groups with insufficient data
automatically fall back to the universal model.

Key design:
  - Shared RobustScaler fit on FULL data (per-group scalers would fail for rare regimes)
  - Single chronological fold per group (too little data for 3-fold CV)
  - Quality gate: specialist must beat universal on same regime data
  - GMM posteriors are the natural router weights - no separate router training

Usage:
    python -X utf8 train_drl_per_regime.py --asset BTC
    python -X utf8 train_drl_per_regime.py --asset BTC --min-bars 1500 --steps-multiplier 1.5
    python -X utf8 train_drl_per_regime.py --asset BTC --universal-model models/retrained/BTC/fold_1/logs/ULTIMATE/best_model/best_model.zip

================================================================================
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import (
    BaseCallback, EvalCallback, CheckpointCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.monitor import Monitor

from sb3_contrib import TQC

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("DRL_PerRegime")

# ── Path setup: add project root for cross-package imports ──
_TRAINING_DIR = Path(__file__).resolve().parent          # training/
PROJECT_ROOT = _TRAINING_DIR.parent                      # project root
sys.path.insert(0, str(PROJECT_ROOT))

# Import from train_drl_full (same codebase)
from train_drl_full import (
    TradingEnvFull,
    ULTIMATE_PRESET,
    ASSET_OVERRIDES,
    _load_regime_names,
    GradientMonitorCallback,
    FPSLoggerCallback,
)
from core.regime_smoother import RegimeSmoother


# =============================================================================
# Semantic Regime Group Mapping
# =============================================================================

SEMANTIC_GROUPS = [
    "TRENDING_UP", "TRENDING_DOWN", "VOLATILE",
    "LOW_VOL_RANGE", "ACCUMULATION", "DISTRIBUTION",
]

# Per-asset mapping: GMM regime name -> semantic group
REGIME_GROUP_MAP = {
    "BTC": {
        "MOMENTUM_RALLY":       "TRENDING_UP",
        "PANIC_SELLOFF":        "TRENDING_DOWN",
        "VOLATILE_CHOP":        "VOLATILE",
        "EXTREME_VOLATILITY":   "VOLATILE",
        "QUIET_ACCUMULATION":   "ACCUMULATION",
        "WEAK_CONSOLIDATION":   "LOW_VOL_RANGE",
        "REGIME_0":             "LOW_VOL_RANGE",
        "REGIME_1":             "ACCUMULATION",
    },
    "ETH": {
        "MOMENTUM_RALLY":       "TRENDING_UP",
        "PANIC_SELLOFF":        "TRENDING_DOWN",
        "VOLATILE_CHOP":        "VOLATILE",
        "EXTREME_VOLATILITY":   "VOLATILE",
        "QUIET_ACCUMULATION":   "ACCUMULATION",
        "WEAK_CONSOLIDATION":   "LOW_VOL_RANGE",
        "REGIME_2":             "LOW_VOL_RANGE",
    },
    "SOL": {
        "MOMENTUM_RALLY":       "TRENDING_UP",
        "PANIC_SELLOFF":        "TRENDING_DOWN",
        "VOLATILE_CHOP":        "VOLATILE",
        "EXTREME_VOLATILITY":   "VOLATILE",
        "QUIET_ACCUMULATION":   "ACCUMULATION",
        "WEAK_CONSOLIDATION":   "LOW_VOL_RANGE",
        "REGIME_5":             "LOW_VOL_RANGE",
    },
}

# Default minimum bars per group to justify training a specialist
DEFAULT_MIN_BARS = 1500


# =============================================================================
# Data Splitting
# =============================================================================

def split_by_regime_group(
    df: pd.DataFrame,
    asset: str,
    regime_names: List[str],
) -> Dict[str, pd.DataFrame]:
    """
    Split training data into per-semantic-group subsets.

    Temporal order is preserved - each group's subset is the original rows
    where regime belongs to that group, in order.

    Returns:
        Dict mapping group_name -> DataFrame subset
    """
    group_map = REGIME_GROUP_MAP.get(asset, {})
    groups: Dict[str, pd.DataFrame] = {}

    # Build regime_id -> group mapping
    regime_id_to_group: Dict[int, str] = {}
    for idx, name in enumerate(regime_names):
        group = group_map.get(name, "LOW_VOL_RANGE")  # Unknown -> default
        regime_id_to_group[idx] = group

    for group_name in SEMANTIC_GROUPS:
        # Find all regime IDs that map to this group
        regime_ids = [
            idx for idx, g in regime_id_to_group.items()
            if g == group_name
        ]
        if not regime_ids:
            groups[group_name] = pd.DataFrame(columns=df.columns)
            continue

        mask = df["regime"].isin(regime_ids)
        group_df = df[mask].copy()
        groups[group_name] = group_df

    return groups


def build_regime_id_to_group(
    regime_names: List[str], asset: str,
) -> Dict[str, str]:
    """Build mapping from regime_id (str) -> semantic group name."""
    group_map = REGIME_GROUP_MAP.get(asset, {})
    result = {}
    for idx, name in enumerate(regime_names):
        group = group_map.get(name, "LOW_VOL_RANGE")
        result[str(idx)] = group
    return result


# =============================================================================
# Compute training steps proportionally
# =============================================================================

def compute_steps(group_bars: int, full_bars: int, full_steps: int) -> int:
    """Scale timesteps proportionally to data size. Floor = 500K."""
    ratio = group_bars / max(full_bars, 1)
    return max(500_000, int(full_steps * ratio))


def compute_buffer(group_bars: int) -> int:
    """Smaller replay buffer for smaller datasets."""
    return max(100_000, min(2_000_000, group_bars * 50))


# =============================================================================
# Per-Regime Trainer
# =============================================================================

class PerRegimeTrainer:
    """
    Trains per-semantic-group TQC specialists with quality gate.

    For each group:
      1. Check MIN_BARS threshold
      2. Single chronological split (80/20 + gap=42)
      3. Scale with SHARED scaler (fit on full data)
      4. Train TQC with proportional timesteps
      5. Quality gate: specialist eval > universal eval on same data
    """

    def __init__(
        self,
        data: pd.DataFrame,
        feature_cols: List[str],
        asset: str,
        regime_names: List[str],
        output_dir: str = "models/drl_per_regime",
        min_bars: int = DEFAULT_MIN_BARS,
        steps_multiplier: float = 1.0,
        universal_model_path: Optional[str] = None,
        device: str = "auto",
        progress_bar: bool = True,
        no_scale_features: Optional[set] = None,
        reward_mode: str = "classic",
    ):
        self.data = data
        self.feature_cols = feature_cols
        self.no_scale_features = no_scale_features or set()
        self.asset = asset
        self.regime_names = regime_names
        self.min_bars = min_bars
        self.steps_multiplier = steps_multiplier
        self.reward_mode = reward_mode
        self.progress_bar = progress_bar

        self.output_dir = Path(output_dir) / asset
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.universal_model_path = universal_model_path

        overrides = ASSET_OVERRIDES.get(asset, {})
        self.full_steps = overrides.get("total_timesteps", 2_500_000)
        self.learning_rate = overrides.get("learning_rate", ULTIMATE_PRESET["learning_rate"])
        self.sol_vol_multiplier = overrides.get("sol_vol_multiplier", 1.0)

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Will be set during run()
        self._shared_scaler_meta = None
        self._universal_model = None
        self._manifest = {
            "asset": asset,
            "regime_names": regime_names,
            "group_map": REGIME_GROUP_MAP.get(asset, {}),
            "regime_id_to_group": build_regime_id_to_group(regime_names, asset),
            "models": {},
            "scaler_path": str(self.output_dir / "feature_scaler.pkl"),
            "trained_at": None,
        }

    def run(self) -> Dict:
        """Main entry point - train universal + per-group specialists."""
        logger.info("=" * 70)
        logger.info(f"HMATS v7 - Per-Regime Specialist Training: {self.asset}")
        logger.info("=" * 70)
        logger.info(f"  Data rows:     {len(self.data)}")
        logger.info(f"  Features:      {len(self.feature_cols)}")
        logger.info(f"  Obs dim:       {len(self.feature_cols) + 4}")
        logger.info(f"  Regimes:       {len(self.regime_names)} - {self.regime_names}")
        logger.info(f"  Min bars:      {self.min_bars}")
        logger.info(f"  Steps mult:    {self.steps_multiplier}")
        logger.info(f"  Device:        {self.device}")
        logger.info(f"  Reward mode:   {self.reward_mode}")
        logger.info(f"  ent_coef:      {ULTIMATE_PRESET['ent_coef']} (FIXED)")
        logger.info("=" * 70)

        # Step 1: Fit shared scaler on full training data
        self._fit_shared_scaler()

        # Step 2: Load or train universal model
        self._prepare_universal_model()

        # Step 3: Split data by regime group
        groups = split_by_regime_group(self.data, self.asset, self.regime_names)

        logger.info("\n  Regime group data sizes:")
        for group_name in SEMANTIC_GROUPS:
            n = len(groups.get(group_name, []))
            trainable = "YES" if n >= self.min_bars else "NO (fallback)"
            logger.info(f"    {group_name:20s}: {n:6d} bars - {trainable}")

        # Step 4: Train per-group specialists
        results = {}
        for group_name in SEMANTIC_GROUPS:
            group_df = groups.get(group_name, pd.DataFrame())
            result = self._train_group(group_name, group_df)
            results[group_name] = result

        # Step 5: Save manifest
        self._manifest["trained_at"] = datetime.now(timezone.utc).isoformat()
        manifest_path = self.output_dir / "regime_router_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(self._manifest, f, indent=2)
        logger.info(f"\n  Manifest saved: {manifest_path}")

        # Summary
        specialists = sum(
            1 for v in self._manifest["models"].values()
            if not v.get("is_fallback", True) and v.get("path") != "universal"
        )
        fallbacks = sum(
            1 for v in self._manifest["models"].values()
            if v.get("is_fallback", True)
        )
        logger.info(f"\n  SUMMARY: {specialists} specialists + {fallbacks} fallbacks + 1 universal")
        logger.info("=" * 70)

        return results

    def _fit_shared_scaler(self):
        """Fit RobustScaler on full training data (80% chronological)."""
        logger.info("\n  Fitting shared RobustScaler on full data...")

        n = len(self.data)
        train_end = int(n * 0.80)
        train_df = self.data.iloc[:train_end]

        scale_cols = [c for c in self.feature_cols if c not in self.no_scale_features]
        pass_cols = [c for c in self.feature_cols if c in self.no_scale_features]

        scaler = RobustScaler()
        scaler.fit(train_df[scale_cols])

        # Handle zero IQR (sparse binary features)
        scaler.scale_ = np.where(scaler.scale_ == 0, 1.0, scaler.scale_)

        self._shared_scaler_meta = {
            "scaler": scaler,
            "scale_cols": scale_cols,
            "pass_cols": pass_cols,
        }

        # Save scaler
        scaler_path = self.output_dir / "feature_scaler.pkl"
        joblib.dump(self._shared_scaler_meta, scaler_path)
        logger.info(f"    Scaler saved: {scaler_path} "
                    f"({len(scale_cols)} scaled, {len(pass_cols)} pass-through)")

    def _scale_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply shared scaler to a DataFrame."""
        meta = self._shared_scaler_meta
        scaled = df.copy()
        scaled[meta["scale_cols"]] = meta["scaler"].transform(df[meta["scale_cols"]])

        # Clean up inf/NaN, clip outliers
        scaled[meta["scale_cols"]] = scaled[meta["scale_cols"]].replace(
            [np.inf, -np.inf], 0
        ).fillna(0)
        scaled[meta["scale_cols"]] = scaled[meta["scale_cols"]].clip(-10.0, 10.0)

        return scaled

    def _prepare_universal_model(self):
        """Load existing universal model or train a new one."""
        universal_dir = self.output_dir / "universal"
        universal_dir.mkdir(parents=True, exist_ok=True)

        if self.universal_model_path:
            # Load existing model
            load_path = self.universal_model_path.replace(".zip", "")
            logger.info(f"\n  Loading universal model: {load_path}")
            try:
                self._universal_model = TQC.load(load_path, device=self.device)
                # Copy to output dir
                self._universal_model.save(str(universal_dir / "best_model"))
                self._manifest["models"]["universal"] = {
                    "path": str(universal_dir / "best_model.zip"),
                    "source": self.universal_model_path,
                    "is_fallback": False,
                }
                logger.info("    Universal model loaded OK")
                return
            except Exception as e:
                logger.error(f"    Failed to load universal model: {e}")
                logger.info("    Training new universal model...")

        # Train universal model on full data
        logger.info("\n  Training universal model on full data...")
        self._universal_model, eval_reward = self._train_single(
            name="universal",
            df=self.data,
            output_subdir=universal_dir,
            timesteps=self.full_steps,
        )

        self._manifest["models"]["universal"] = {
            "path": str(universal_dir / "best_model.zip"),
            "eval_reward": eval_reward,
            "bars_trained": len(self.data),
            "is_fallback": False,
        }

    def _train_group(self, group_name: str, group_df: pd.DataFrame) -> Dict:
        """Train a specialist for one semantic group, or fall back to universal."""
        n_bars = len(group_df)
        group_dir = self.output_dir / group_name
        group_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\n{'─' * 60}")
        logger.info(f"  Group: {group_name} ({n_bars} bars)")

        # Check minimum data threshold
        if n_bars < self.min_bars:
            reason = f"Below MIN_BARS ({n_bars} < {self.min_bars})"
            logger.info(f"    FALLBACK -> universal: {reason}")

            self._manifest["models"][group_name] = {
                "path": str(self.output_dir / "universal" / "best_model.zip"),
                "eval_reward": None,
                "bars_trained": n_bars,
                "is_fallback": True,
                "fallback_reason": reason,
            }
            return {"group": group_name, "status": "fallback", "reason": reason}

        # Train specialist
        group_steps = int(compute_steps(
            n_bars, len(self.data), self.full_steps
        ) * self.steps_multiplier)

        logger.info(f"    Training specialist: {group_steps:,} steps "
                    f"(buffer={compute_buffer(n_bars):,})")

        specialist, eval_reward = self._train_single(
            name=group_name,
            df=group_df,
            output_subdir=group_dir,
            timesteps=group_steps,
            buffer_size=compute_buffer(n_bars),
        )

        if specialist is None:
            reason = "Training failed"
            logger.warning(f"    FALLBACK -> universal: {reason}")
            self._manifest["models"][group_name] = {
                "path": str(self.output_dir / "universal" / "best_model.zip"),
                "eval_reward": None,
                "bars_trained": n_bars,
                "is_fallback": True,
                "fallback_reason": reason,
            }
            return {"group": group_name, "status": "fallback", "reason": reason}

        # Quality gate: specialist must beat universal on this group's data
        universal_eval = self._evaluate_on_group(self._universal_model, group_df)
        logger.info(f"    Quality gate: specialist={eval_reward:.2f} vs "
                    f"universal={universal_eval:.2f}")

        if eval_reward <= universal_eval:
            reason = (
                f"Specialist ({eval_reward:.2f}) did not beat "
                f"universal ({universal_eval:.2f}) on group data"
            )
            logger.info(f"    FALLBACK -> universal: {reason}")
            self._manifest["models"][group_name] = {
                "path": str(self.output_dir / "universal" / "best_model.zip"),
                "eval_reward": eval_reward,
                "universal_eval_on_same_data": universal_eval,
                "bars_trained": n_bars,
                "is_fallback": True,
                "fallback_reason": reason,
            }
            return {
                "group": group_name,
                "status": "fallback",
                "reason": reason,
                "specialist_eval": eval_reward,
                "universal_eval": universal_eval,
            }

        # Specialist passed quality gate
        logger.info(f"    SPECIALIST PASSED: +{eval_reward - universal_eval:.2f} vs universal")

        self._manifest["models"][group_name] = {
            "path": str(group_dir / "best_model.zip"),
            "eval_reward": eval_reward,
            "universal_eval_on_same_data": universal_eval,
            "bars_trained": n_bars,
            "is_fallback": False,
        }
        return {
            "group": group_name,
            "status": "specialist",
            "eval_reward": eval_reward,
            "universal_eval": universal_eval,
        }

    def _train_single(
        self,
        name: str,
        df: pd.DataFrame,
        output_subdir: Path,
        timesteps: int,
        buffer_size: int = 2_000_000,
    ) -> Tuple[Optional[object], float]:
        """
        Train a single TQC model on given data.

        Uses single chronological split: train=80%, val=20%, gap=42.

        Returns:
            (model, eval_reward) or (None, 0.0) on failure
        """
        n = len(df)
        gap = 42
        val_ratio = 0.20
        val_size = max(int(n * val_ratio), 100)
        train_end = n - val_size - gap

        if train_end < 200:
            logger.warning(f"    [{name}] train_end={train_end} < 200, too small")
            return None, 0.0

        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end + gap:].copy()

        logger.info(f"    [{name}] Split: train={len(train_df)}, "
                    f"val={len(val_df)}, gap={gap}")

        # Scale with shared scaler
        train_scaled = self._scale_df(train_df)
        val_scaled = self._scale_df(val_df)

        # Create environments
        train_env = DummyVecEnv([lambda df=train_scaled: Monitor(TradingEnvFull(
            df=df,
            feature_cols=self.feature_cols,
            sol_vol_multiplier=self.sol_vol_multiplier,
            reward_mode=self.reward_mode,
        ))])
        eval_env = DummyVecEnv([lambda df=val_scaled: Monitor(TradingEnvFull(
            df=df,
            feature_cols=self.feature_cols,
            sol_vol_multiplier=self.sol_vol_multiplier,
            reward_mode=self.reward_mode,
        ))])

        # Smoke test
        obs = train_env.reset()
        expected_shape = (1, len(self.feature_cols) + 4)
        assert obs.shape == expected_shape, \
            f"[{name}] Obs shape mismatch: {obs.shape} vs {expected_shape}"
        for step_i in range(5):
            act = np.array([[np.random.uniform(-1, 1)]], dtype=np.float32)
            obs, rew, done, info = train_env.step(act)
            assert not np.isnan(obs).any(), f"[{name}] NaN in obs at smoke step {step_i}"
        train_env.reset()
        logger.info(f"    [{name}] Smoke test OK")

        # Build TQC model
        preset = dict(ULTIMATE_PRESET)
        preset["learning_rate"] = self.learning_rate
        preset["buffer_size"] = buffer_size
        policy_kwargs = preset.pop("policy_kwargs")
        policy = preset.pop("policy")

        log_dir = output_subdir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        model = TQC(
            policy,
            env=train_env,
            verbose=1,
            tensorboard_log=str(log_dir),
            device=self.device,
            policy_kwargs=policy_kwargs,
            **preset,
        )

        # Early stopping: fewer evals needed for smaller data
        min_evals = max(10, min(50, timesteps // 10_000))
        stop_callback = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=15,
            min_evals=min_evals,
            verbose=1,
        )

        eval_freq = max(2_000, min(5_000, timesteps // 100))
        checkpoint_freq = max(100_000, timesteps // 5)

        callbacks = [
            GradientMonitorCallback(max_actor_loss=1e6, max_consecutive_explosions=3),
            FPSLoggerCallback(log_freq=100_000),
            EvalCallback(
                eval_env,
                best_model_save_path=str(output_subdir),
                log_path=str(log_dir / "eval"),
                eval_freq=eval_freq,
                deterministic=True,
                render=False,
                callback_after_eval=stop_callback,
            ),
            CheckpointCallback(
                save_freq=checkpoint_freq,
                save_path=str(output_subdir / "checkpoints"),
                name_prefix=f"SPECIALIST_{name}",
            ),
        ]

        # Train
        start_time = time.time()
        try:
            model.learn(
                total_timesteps=timesteps,
                callback=callbacks,
                progress_bar=self.progress_bar,
            )
        except Exception as e:
            logger.error(f"    [{name}] Training failed: {e}")
            import traceback
            traceback.print_exc()
            train_env.close()
            eval_env.close()
            return None, 0.0

        train_time = time.time() - start_time
        logger.info(f"    [{name}] Training done in {train_time / 60:.1f} min")

        # Load best model (EvalCallback saved at peak eval reward)
        best_path = output_subdir / "best_model.zip"
        if best_path.exists():
            best_model = TQC.load(
                str(best_path).replace(".zip", ""),
                device=self.device,
            )
        else:
            # Save current as best
            model.save(str(output_subdir / "best_model"))
            best_model = model

        # Evaluate
        eval_reward = self._evaluate_model(best_model, eval_env)
        logger.info(f"    [{name}] Eval reward: {eval_reward:.2f}")

        train_env.close()
        eval_env.close()

        return best_model, eval_reward

    def _evaluate_model(
        self, model, env, n_episodes: int = 10,
    ) -> float:
        """Evaluate model on DummyVecEnv, return mean reward."""
        rewards = []
        for _ in range(n_episodes):
            obs = env.reset()
            done = False
            total = 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                total += reward[0]
            rewards.append(total)
        return float(np.mean(rewards))

    def _evaluate_on_group(
        self, model, group_df: pd.DataFrame, n_episodes: int = 10,
    ) -> float:
        """Evaluate a model specifically on a group's validation data."""
        n = len(group_df)
        gap = 42
        val_size = max(int(n * 0.20), 100)
        val_start = n - val_size

        if val_start < gap:
            return 0.0

        val_df = group_df.iloc[val_start:].copy()
        val_scaled = self._scale_df(val_df)

        eval_env = DummyVecEnv([lambda df=val_scaled: Monitor(TradingEnvFull(
            df=df,
            feature_cols=self.feature_cols,
            sol_vol_multiplier=self.sol_vol_multiplier,
            reward_mode=self.reward_mode,
        ))])

        result = self._evaluate_model(model, eval_env, n_episodes)
        eval_env.close()
        return result


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="HMATS v7 - Per-Regime Specialist DRL Training"
    )

    parser.add_argument("--asset", type=str, required=True,
                        choices=["BTC", "ETH", "SOL"])
    parser.add_argument("--data", type=str, default=None,
                        help="Parquet data file (default: training/training_data/drl_training/{ASSET}_4H_full.parquet)")
    parser.add_argument("--min-bars", type=int, default=DEFAULT_MIN_BARS,
                        help=f"Minimum bars per group to train specialist (default: {DEFAULT_MIN_BARS})")
    parser.add_argument("--steps-multiplier", type=float, default=1.0,
                        help="Multiply computed timesteps by this factor (default: 1.0)")
    parser.add_argument("--universal-model", type=str, default=None,
                        help="Path to existing universal model (skip universal training)")
    parser.add_argument("--output", type=str, default=str(PROJECT_ROOT / "models" / "drl_per_regime"),
                        help="Output directory for per-regime models")
    parser.add_argument("--smooth-regimes", type=int, default=2,
                        help="RegimeSmoother persistence (0=disable)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cuda, cpu")
    parser.add_argument("--no-progress-bar", action="store_true")
    parser.add_argument("--reward-mode", type=str, default="classic",
                        choices=["classic", "sortino", "sharpe"])

    args = parser.parse_args()

    # Resolve data path
    data_path = args.data or str(_TRAINING_DIR / "training_data" / "drl_training" / f"{args.asset}_4H_full.parquet")
    logger.info(f"Loading data: {data_path}")

    data = pd.read_parquet(data_path)
    logger.info(f"  Loaded: {len(data)} rows, {len(data.columns)} columns")

    # Apply RegimeSmoother (must match training pipeline)
    if args.smooth_regimes > 0 and "regime" in data.columns:
        smoother = RegimeSmoother(min_persistence=args.smooth_regimes)
        before = sum(1 for i in range(1, len(data))
                     if data["regime"].iloc[i] != data["regime"].iloc[i - 1])
        data = smoother.smooth_column(data, "regime")
        after = sum(1 for i in range(1, len(data))
                    if data["regime"].iloc[i] != data["regime"].iloc[i - 1])
        reduction = (1 - after / max(before, 1)) * 100
        logger.info(
            f"  RegimeSmoother(persistence={args.smooth_regimes}): "
            f"flips {before} -> {after} ({reduction:.0f}% reduction)"
        )

    # Load per-asset regime names
    regime_names = _load_regime_names(asset=args.asset)
    logger.info(f"  Regime names ({args.asset}): k={len(regime_names)} {regime_names}")

    # Feature columns from manifest
    manifest_path = PROJECT_ROOT / "configs" / "feature_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        feature_cols = [c for c in manifest["all_features"] if c in data.columns]
        no_scale = set(manifest.get("no_scale_features", []))
        logger.info(f"  Features: {len(feature_cols)} (from manifest, "
                    f"{len(no_scale)} no-scale)")
    else:
        feature_cols = [
            col for col in data.columns
            if data[col].dtype in ["float64", "float32"]
            and col not in [
                "timestamp", "open", "high", "low", "close", "volume",
                "regime", "regime_confidence",
            ]
        ]
        no_scale = set()
        logger.warning("  No feature_manifest.json - using auto-detect")
        logger.info(f"  Features: {len(feature_cols)} (auto-detected)")

    logger.info(f"  Obs dim: {len(feature_cols) + 4} (features + 4 env state)")

    # Regime group distribution
    group_map = REGIME_GROUP_MAP.get(args.asset, {})
    logger.info(f"\n  Semantic group mapping ({args.asset}):")
    for name in regime_names:
        group = group_map.get(name, "LOW_VOL_RANGE")
        logger.info(f"    {name:25s} -> {group}")

    # Device
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"\n  [CHECK] VecEnv = DummyVecEnv ONLY")
    logger.info(f"  [CHECK] ent_coef = {ULTIMATE_PRESET['ent_coef']} (FIXED)")
    logger.info(f"  [CHECK] Shared RobustScaler (fit on full data)")
    logger.info(f"  [CHECK] Quality gate: specialist must beat universal")
    logger.info(f"  [CHECK] Reward mode: {args.reward_mode}")

    # Train
    trainer = PerRegimeTrainer(
        data=data,
        feature_cols=feature_cols,
        asset=args.asset,
        regime_names=regime_names,
        output_dir=args.output,
        min_bars=args.min_bars,
        steps_multiplier=args.steps_multiplier,
        universal_model_path=args.universal_model,
        device=device,
        progress_bar=not args.no_progress_bar,
        no_scale_features=no_scale,
        reward_mode=args.reward_mode,
    )

    trainer.run()


if __name__ == "__main__":
    main()
