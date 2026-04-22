"""
TQC Teacher Oracle for DT Knowledge Distillation (KD-BeT inspired).

Replaces RegimeAwareExpert.get_action() with TQC.predict(obs) as the label
generator for DT supervised training.

TQC uses VecFrameStack(n_stack=8), so expected obs shape is (8×126,) = (1008,).
This oracle maintains a per-rollout deque of the last 8 obs and flattens it
before each predict() call, matching the exact training contract.

Rationale:
  The original RegimeAwareExpert uses hand-crafted future-return lookforward
  (SmartOracle) which is very noisy on BTC/ETH 4H bars. TQC has learned a
  smoothed optimal policy through 2.5M RL steps with friction-aware rewards:
    BTC fold_3:  891 mean reward  (bear-optimized)
    ETH fold_1: 1400 mean reward
    SOL fold_3:  744 mean reward

  Using TQC.predict(state) as the label gives DT a denoised consistent
  teacher — ~10× less noisy than sign(future_return_24h).

Paper: KD-BeT (Sensors 2025) validates "strong RL teacher → Transformer
student" distillation pattern, outperforms pure RL/IL in CARLA.

Interface matches RegimeAwareExpert.get_action(df, idx, regime) → (action, reward)
so it is a drop-in swap via --oracle-mode tqc_teacher.
"""
from __future__ import annotations

import logging
import os
from collections import deque
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger("TQCOracle")

# TQC training contract (see training/train_drl_full.py DEFAULT_N_STACK=8)
N_STACK = 8
SINGLE_OBS_DIM = 126   # 122 features + 4 env state
STACKED_OBS_DIM = N_STACK * SINGLE_OBS_DIM  # 1008

# Best fold per asset. [FIX 2026-04-22] ETH fold_1 -> fold_3:
# results.json reported fold_1 reward=1400 but train_rows=0/train_time=0 — stale
# checkpoint with corrupted metadata. fold_3 (reward=1029, train_rows=10028) is
# the real-trained best. DT val acc jumped 35.1% -> 58.8% after switching.
BEST_FOLDS = {
    "BTC": "fold_3",   # mean_reward=891  (v6 val acc 53.2%)
    "ETH": "fold_3",   # mean_reward=1029 (v6 val acc 58.8%)
    "SOL": "fold_3",   # mean_reward=744  (v6 val acc 65.2%)
}


def _default_tqc_path(asset: str) -> str:
    """Construct best-fold TQC checkpoint path for an asset."""
    fold = BEST_FOLDS[asset.upper()]
    return f"models/retrained/{asset.upper()}/{fold}/logs/LSTM_FILM_A/best_model/best_model.zip"


class TQCTeacherOracle:
    """Drop-in replacement for RegimeAwareExpert using a trained TQC model.

    Usage in DT training:
      oracle = TQCTeacherOracle(asset='BTC')
      oracle.set_context(features_scaled)  # MUST call before get_action
      # ... inside trajectory loop:
      action, reward = oracle.get_action(df, idx, regime)
    """

    def __init__(self, asset: str, tqc_model_path: Optional[str] = None):
        self.asset = asset.upper()
        self.tqc_model_path = tqc_model_path or _default_tqc_path(self.asset)
        self._model = None
        self._features_scaled: Optional[np.ndarray] = None  # (N, 122)
        self._prev_action = 0.0  # for rolling env_state simulation
        # Frame-stack buffer (latest N_STACK obs as 1D vectors)
        self._stack: deque = deque(maxlen=N_STACK)

        if not Path(self.tqc_model_path).exists():
            raise FileNotFoundError(
                f"[TQC-ORACLE] checkpoint not found: {self.tqc_model_path}"
            )

        # sb3_contrib.TQC import — only when oracle is actually used
        try:
            # Register training module aliases so the LSTM_FILM_A policy's
            # custom extractor can be unpickled (cloudpickle embeds
            # training.models.film_extractor paths)
            import importlib
            import sys as _sys
            for mod_name in ("training.models", "training.models.film_extractor",
                             "training.models.feature_extractors"):
                try:
                    importlib.import_module(mod_name)
                except Exception:
                    pass
            # Alias 'models' → 'training.models' for old checkpoints
            if "models" not in _sys.modules or _sys.modules.get("models") is None:
                try:
                    _sys.modules["models"] = importlib.import_module("training.models")
                    _sys.modules["models.film_extractor"] = importlib.import_module(
                        "training.models.film_extractor"
                    )
                except Exception:
                    pass

            from sb3_contrib import TQC
            self._model = TQC.load(self.tqc_model_path, device="auto")
            logger.info(
                f"[TQC-ORACLE] {self.asset}: loaded {self.tqc_model_path} "
                f"(obs_space={self._model.observation_space.shape})"
            )
        except Exception as e:
            raise RuntimeError(f"[TQC-ORACLE] failed to load TQC: {e}") from e

    def set_context(self, features_scaled: np.ndarray) -> None:
        """Provide scaled feature matrix (N, 122) for the entire dataset.

        Must be called ONCE per dataset build before get_action() is used.
        The oracle indexes into this to construct the 126-dim obs TQC expects.
        """
        if features_scaled.ndim != 2 or features_scaled.shape[1] != 122:
            raise ValueError(
                f"[TQC-ORACLE] expected features_scaled shape (N, 122), got {features_scaled.shape}"
            )
        self._features_scaled = features_scaled.astype(np.float32)
        self._prev_action = 0.0  # reset rolling state

    def reset_rollout(self) -> None:
        """Reset rolling env state + frame-stack (call at start of each trajectory)."""
        self._prev_action = 0.0
        self._stack.clear()

    def _build_obs(self, idx: int) -> np.ndarray:
        """Construct 126-dim observation matching TQC training contract.

        Layout: [122 scaled features | 4 env state]
        Env state: position_ratio, position_direction, pnl_ratio, drawdown.
        For teacher-label generation we use a stateless proxy:
          - position_ratio = |prev_action| (rough size)
          - position_direction = sign(prev_action)
          - pnl_ratio = 0.0 (unknown in labeling)
          - drawdown = 0.0 (unknown in labeling)
        """
        if self._features_scaled is None:
            raise RuntimeError(
                "[TQC-ORACLE] set_context() must be called before get_action()"
            )
        if idx < 0 or idx >= len(self._features_scaled):
            raise IndexError(f"[TQC-ORACLE] idx {idx} out of range (N={len(self._features_scaled)})")

        feat = self._features_scaled[idx]  # (122,)
        pos_ratio = abs(self._prev_action)
        pos_dir = float(np.sign(self._prev_action))
        env_state = np.array([pos_ratio, pos_dir, 0.0, 0.0], dtype=np.float32)
        obs = np.concatenate([feat, env_state]).astype(np.float32)
        return obs

    # --- RegimeAwareExpert-compatible interface ---

    def _stacked_obs(self, idx: int) -> np.ndarray:
        """Build (1008,) stacked observation matching TQC VecFrameStack contract.

        Pushes current obs onto rolling stack; when stack has < N_STACK frames,
        pads by repeating the oldest (same behavior as VecFrameStack on reset).
        """
        current = self._build_obs(idx)  # (126,)
        self._stack.append(current)
        # Pad with the first frame until we have N_STACK
        frames = list(self._stack)
        while len(frames) < N_STACK:
            frames.insert(0, frames[0])
        stacked = np.concatenate(frames).astype(np.float32)  # (1008,)
        assert stacked.shape == (STACKED_OBS_DIM,), \
            f"stacked obs shape {stacked.shape} != ({STACKED_OBS_DIM},)"
        return stacked

    def get_action(self, df, idx: int, regime: str) -> Tuple[float, float]:
        """Return (action, reward) with TQC as teacher.

        Matches RegimeAwareExpert.get_action signature but uses TQC.predict.
        `df` and `regime` are accepted for interface compatibility but not
        used directly — TQC observation is built from set_context() data.
        Frame-stacked to (1008,) to match TQC training VecFrameStack(n_stack=8).
        """
        try:
            obs = self._stacked_obs(idx)
            # TQC.predict returns (action, state). deterministic for stable labels.
            action, _ = self._model.predict(obs, deterministic=True)
            action_scalar = float(action[0]) if hasattr(action, "__len__") else float(action)
            # Clamp to [-1, 1] defensively
            action_scalar = max(-1.0, min(1.0, action_scalar))
            # Update rolling state for next call
            self._prev_action = action_scalar
            # Reward proxy: action magnitude × 10 (matches SmartOracle scale,
            # used by reward_head regression loss)
            reward = abs(action_scalar) * 10.0
            return action_scalar, reward
        except Exception as e:
            logger.warning(f"[TQC-ORACLE] predict failed at idx={idx}: {e} — fallback neutral")
            return 0.0, 0.0

    def get_action_batch(self, indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized prediction over a range of indices.

        Returns (actions, rewards) as float32 arrays.
        NOTE: this version does NOT update self._prev_action — caller must
        manage rolling state if using this path. Prefer get_action() for
        trajectory construction to preserve cumulative env_state semantics.
        """
        if self._features_scaled is None:
            raise RuntimeError("set_context() not called")
        feats = self._features_scaled[indices]  # (K, 122)
        K = len(feats)
        env_block = np.zeros((K, 4), dtype=np.float32)
        obs_batch = np.hstack([feats, env_block]).astype(np.float32)
        actions, _ = self._model.predict(obs_batch, deterministic=True)
        if actions.ndim > 1:
            actions = actions.squeeze(-1)
        actions = np.clip(actions.astype(np.float32), -1.0, 1.0)
        rewards = np.abs(actions) * 10.0
        return actions, rewards
