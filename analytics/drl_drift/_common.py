"""DRL drift validation — common utilities.

Read-only inference. Reuses the LIVE inference pipeline:
  - drl.ensemble.load_best_model -> handles cloudpickle module aliases
  - per-asset RobustScaler from models/retrained/{ASSET}/fold_3/.../feature_scaler.pkl
  - Pre-computed 122 features from training/training_data/drl_training/{ASSET}_4H_full.parquet

Why we read features directly from the parquet (not via build_obs):
  - The parquet IS the training data. Features are already computed and identical
    to what the model trained on.
  - Calling build_obs(asset, ohlcv_df, market_data, gmm_probs, env_state) per bar
    requires synthesizing a market_data dict + GMM probs from the parquet, which
    just inverts the rebuild pipeline. Slower and adds places to drift.
  - For drift validation we want to know: "given the SAME feature pipeline that
    produced training data, how does the model behave on recent data?" — reading
    pre-computed features is the cleanest answer.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# Silence noisy module loggers during diagnostic runs
logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(name)s | %(message)s")

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

# [P420] HOME TRIO BY DESIGN: TQC checkpoints exist only for BTC/ETH/SOL
# (BEST_FOLDS below); breadth assets have no DRL model to drift.
ASSETS = ["BTC", "ETH", "SOL"]

# Must match drl/ensemble.py BEST_FOLDS + drl/runtime_obs_builder.py BEST_FOLDS
BEST_FOLDS = {"BTC": "fold_3", "ETH": "fold_3", "SOL": "fold_3"}

SINGLE_OBS_DIM = 126   # 122 features + 4 env state
N_STACK = 8            # frame stack (matches training)
STACKED_OBS_DIM = SINGLE_OBS_DIM * N_STACK  # 1008

PARQUET_DIR = PROJ / "training" / "training_data" / "drl_training"
SCALER_DIR = PROJ / "models" / "retrained"
MANIFEST_PATH = PROJ / "configs" / "feature_manifest.json"


def load_manifest() -> dict:
    """Load the canonical feature manifest used by training + runtime."""
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_parquet(asset: str) -> pd.DataFrame:
    """Load the 4H full-feature parquet for an asset."""
    path = PARQUET_DIR / f"{asset}_4H_full.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Parquet not found: {path}")
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def load_scaler(asset: str) -> dict:
    """Load the per-asset feature scaler used at inference."""
    fold = BEST_FOLDS[asset]
    scaler_path = (
        SCALER_DIR / asset / fold / "logs" / "LSTM_FILM_A" / "feature_scaler.pkl"
    )
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found for {asset}: {scaler_path}")

    import joblib
    from infra.safe_torch_load import safe_joblib_load

    scaler_obj = safe_joblib_load(scaler_path, joblib_module=joblib)
    return scaler_obj if isinstance(scaler_obj, dict) else {"scaler": scaler_obj}


def build_features_122(df: pd.DataFrame, manifest: dict) -> Tuple[np.ndarray, list]:
    """Extract the 122 features from a parquet DataFrame in manifest order.

    Returns (features_matrix [n_rows, 122], feature_names_list).
    """
    all_features = manifest["all_features"]
    missing = [f for f in all_features if f not in df.columns]
    if missing:
        raise ValueError(
            f"Parquet missing {len(missing)} manifest features. First 5: {missing[:5]}"
        )
    feats = df[all_features].to_numpy(dtype=np.float32)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats, all_features


def scale_features(features_122: np.ndarray, scaler_meta: dict, manifest: dict) -> np.ndarray:
    """Apply per-asset RobustScaler to scalable cols, clip to [-10, 10].

    Mirrors RuntimeObsBuilder._scale_features.
    """
    no_scale = set(manifest.get("no_scale_features", []))
    all_features = manifest["all_features"]
    scaler = scaler_meta.get("scaler")
    if scaler is None:
        return np.clip(features_122, -10.0, 10.0)

    scale_cols = scaler_meta.get("scale_cols") or [f for f in all_features if f not in no_scale]
    scale_indices = [all_features.index(c) for c in scale_cols if c in all_features]

    if not scale_indices:
        return np.clip(features_122, -10.0, 10.0)

    out = features_122.copy()
    scalable = out[:, scale_indices]
    try:
        scaled = scaler.transform(scalable)
        out[:, scale_indices] = np.clip(scaled, -10.0, 10.0)
    except Exception as e:
        logging.warning(f"scaler.transform failed: {e}")
    return np.clip(out, -10.0, 10.0)


def build_obs_matrix(
    asset: str,
    days_back: Optional[int] = None,
    days_start: Optional[int] = None,
    days_end: Optional[int] = None,
) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """Build a [n_rows, 126] obs matrix for a date-window slice.

    Args:
        asset: BTC / ETH / SOL
        days_back: if set, take the last N days from the parquet end
        days_start, days_end: if both set, take parquet[max-days_start : max-days_end]
                              (e.g. days_start=540, days_end=180 => "older 1y window")

    Env state is set to zeros (no position) for all rows — drift validation
    asks how the policy responds to market state, not position state.
    """
    df = load_parquet(asset)
    manifest = load_manifest()
    scaler = load_scaler(asset)

    last_ts = df["timestamp"].max()
    if days_back is not None:
        df = df[df["timestamp"] >= last_ts - pd.Timedelta(days=days_back)]
    elif days_start is not None and days_end is not None:
        start = last_ts - pd.Timedelta(days=days_start)
        end = last_ts - pd.Timedelta(days=days_end)
        df = df[(df["timestamp"] >= start) & (df["timestamp"] < end)]

    feats_122, _ = build_features_122(df.reset_index(drop=True), manifest)
    feats_122 = scale_features(feats_122, scaler, manifest)

    # Append 4-dim zero env state
    env = np.zeros((feats_122.shape[0], 4), dtype=np.float32)
    obs_126 = np.concatenate([feats_122, env], axis=1)
    obs_126 = np.nan_to_num(obs_126, nan=0.0, posinf=0.0, neginf=0.0)
    obs_126 = np.clip(obs_126, -10.0, 10.0).astype(np.float32)
    return obs_126, df["timestamp"].reset_index(drop=True)


def stack_frames(obs_126: np.ndarray) -> np.ndarray:
    """Build the n_stack=8 stacked obs matrix [n_rows, 1008] used at inference.

    For row i, stack frames [max(0, i-7) .. i], zero-pad on the left if i<7.
    Mirrors VecFrameStack with channels-first concatenation as used by SB3.
    """
    n = obs_126.shape[0]
    out = np.zeros((n, STACKED_OBS_DIM), dtype=np.float32)
    for i in range(n):
        start = max(0, i - (N_STACK - 1))
        slab = obs_126[start : i + 1]  # (k, 126), k <= 8
        k = slab.shape[0]
        if k < N_STACK:
            pad = np.zeros((N_STACK - k, SINGLE_OBS_DIM), dtype=np.float32)
            slab = np.concatenate([pad, slab], axis=0)
        out[i] = slab.reshape(-1)
    return out


def load_drl_inference(asset: str):
    """Load TQCInference wrapper (handles cloudpickle aliases via _register_training_modules).

    Returns the TQCInference object. The underlying SB3 model is at .model._tqc_model
    if you need raw TQC API access (critic, policy, etc.).
    """
    from drl.ensemble import load_best_model

    tqc_inference = load_best_model(asset, output_dir=str(SCALER_DIR), device="cpu")
    if not tqc_inference.tqc_available:
        raise RuntimeError(f"DRL model failed to load for {asset}")
    return tqc_inference


def predict_actions(model_inference, stacked_obs: np.ndarray) -> np.ndarray:
    """Batch-predict deterministic actions over a stacked-obs matrix.

    stacked_obs shape (n, 1008). Returns (n,) float in [-1, 1].
    """
    sb3_model = model_inference._tqc_model
    actions, _ = sb3_model.predict(stacked_obs, deterministic=True)
    return np.asarray(actions).reshape(-1)
