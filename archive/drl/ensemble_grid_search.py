"""
TQC + DT Ensemble Grid Search (Stage 12)

Evaluates whether blending DT v3.2 direction signals with TQC actions
improves trading performance on validation data.

For each asset:
  1. Load val data (last 15% of parquet)
  2. Load TQC model (LSTM_FILM_A, best fold) + DT model
  3. Walk through val data step-by-step
  4. Test TQC weight grid: [0.7, 0.8, 0.85, 0.9, 0.95, 1.0]
  5. Apply gate: Ensemble Sharpe >= TQC×90%, MaxDD <= TQC×120%

Usage:
    python -X utf8 -u scripts/ensemble_grid_search.py
    python -X utf8 -u scripts/ensemble_grid_search.py --asset BTC
"""

import json
import logging
import os
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure project root is on sys.path (needed when running from scripts/)
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)  # Ensure relative paths resolve correctly

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("EnsembleGridSearch")

# --- Config ---

BEST_FOLDS = {"BTC": "fold_3", "ETH": "fold_1", "SOL": "fold_3"}

TQC_WEIGHT_GRID = [0.70, 0.80, 0.85, 0.90, 0.95, 1.00]

# Gate thresholds (from E2E guide)
SHARPE_GATE = 0.90   # Ensemble Sharpe >= TQC-only × 0.90
MAXDD_GATE = 1.20    # Ensemble MaxDD <= TQC-only × 1.20

N_STACK = 8
OBS_DIM = 126
DT_CONFIDENCE_THRESHOLD = 0.3

INITIAL_BALANCE = 10000.0


# --- Helpers ---

def load_feature_manifest() -> dict:
    with open("configs/feature_manifest.json") as f:
        return json.load(f)


def load_val_data(asset: str, manifest: dict) -> Tuple[pd.DataFrame, List[str]]:
    """Load last 15% of parquet as val data."""
    path = f"training/training_data/drl_training/{asset}_4H_full.parquet"
    df = pd.read_parquet(path)
    n = len(df)
    val_start = n - int(n * 0.15)
    val_df = df.iloc[val_start:].reset_index(drop=True)

    feature_cols = [c for c in manifest["all_features"] if c in val_df.columns]
    logger.info(f"  Val data: {len(val_df)} rows, {len(feature_cols)} features")
    return val_df, feature_cols


def load_and_apply_scaler(
    val_df: pd.DataFrame,
    feature_cols: List[str],
    scaler_path: str,
    no_scale_features: List[str],
) -> np.ndarray:
    """Load scaler and scale val features. Returns (n_rows, 122) array."""
    scaler_obj = joblib.load(scaler_path)

    # Handle both formats: dict (TQC) or raw scaler (DT)
    if isinstance(scaler_obj, dict):
        scaler = scaler_obj["scaler"]
        scale_cols = scaler_obj.get("scale_cols", [])
        pass_cols = scaler_obj.get("pass_cols", [])
    else:
        scaler = scaler_obj
        scale_cols = [c for c in feature_cols if c not in no_scale_features]
        pass_cols = [c for c in feature_cols if c in no_scale_features]

    features_raw = val_df[feature_cols].values.astype(np.float32)

    # Determine column indices
    scale_idx = [feature_cols.index(c) for c in scale_cols if c in feature_cols]
    pass_idx = [feature_cols.index(c) for c in pass_cols if c in feature_cols]

    features_scaled = features_raw.copy()
    if scale_idx:
        features_scaled[:, scale_idx] = scaler.transform(features_raw[:, scale_idx])
        features_scaled[:, scale_idx] = np.clip(features_scaled[:, scale_idx], -10, 10)

    # Clean NaN/inf
    features_scaled = np.nan_to_num(features_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    return features_scaled


def _register_training_modules():
    """Register training/models as 'models' in sys.modules for cloudpickle compat.

    TQC models were pickled from training/ dir with imports like
    'from models.film_extractor import LSTMFiLMPosAExtractor'.
    At load time from project root, we need 'models' -> 'training.models'.
    """
    import importlib
    if "models" not in sys.modules or sys.modules["models"] is None:
        try:
            training_models = importlib.import_module("training.models")
            sys.modules["models"] = training_models
            # Also register submodules cloudpickle may reference
            film = importlib.import_module("training.models.film_extractor")
            sys.modules["models.film_extractor"] = film
            feat = importlib.import_module("training.models.feature_extractors")
            sys.modules["models.feature_extractors"] = feat
            logger.info("  Registered training.models -> models alias")
        except Exception as e:
            logger.warning(f"  Failed to register models alias: {e}")


def load_tqc_model(asset: str):
    """Load TQC model with LSTM_FILM_A custom_objects."""
    from sb3_contrib import TQC

    fold = BEST_FOLDS[asset]
    model_path = (
        f"models/retrained/{asset}/{fold}/logs/LSTM_FILM_A/best_model/best_model"
    )

    # Register module aliases so cloudpickle can deserialize the FiLM extractor
    # class reference (models.film_extractor.LSTMFiLMPosAExtractor) natively.
    # NOTE: Do NOT pass custom_objects={"policy_kwargs": ...} because SB3's
    # json_to_data REPLACES (not updates) policy_kwargs, which drops n_quantiles,
    # net_arch, features_extractor_kwargs etc.
    _register_training_modules()

    model = TQC.load(model_path, device="cuda")
    logger.info(f"  TQC loaded: {model_path}")
    return model


def load_dt_model(asset: str):
    """Load DT v3.2 model via RealDecisionTransformer."""
    sys.path.insert(0, str(Path.cwd()))

    # The DT checkpoint was saved from __main__ (training script), so pickle
    # references __main__.DTConfigV32. Register it before torch.load().
    import __main__
    from training.drl.train_decision_transformer_v32 import (
        DTConfigV32, DecisionTransformerV32,
    )
    if not hasattr(__main__, "DTConfigV32"):
        __main__.DTConfigV32 = DTConfigV32
    if not hasattr(__main__, "DecisionTransformerV32"):
        __main__.DecisionTransformerV32 = DecisionTransformerV32

    from agents.model_alpha_agent import RealDecisionTransformer

    model_path = f"models/decision_transformer/{asset}/dt_v32_best.pt"
    dt = RealDecisionTransformer(model_path=model_path)
    logger.info(f"  DT loaded: ctx={dt._context_length}, state_dim={dt.state_dim}, "
                f"model={'OK' if dt.model is not None else 'FAILED'}")
    return dt


# --- Evaluation ---

@dataclass
class EvalMetrics:
    total_return: float
    sharpe: float
    max_dd: float
    win_rate: float
    n_steps: int


def compute_metrics(pnl_series: np.ndarray) -> EvalMetrics:
    """Compute trading metrics from per-step PnL."""
    if len(pnl_series) == 0:
        return EvalMetrics(0, 0, 0, 0, 0)

    total_return = float(np.sum(pnl_series))
    mean_ret = float(np.mean(pnl_series))
    std_ret = float(np.std(pnl_series))
    sharpe = mean_ret / max(std_ret, 1e-8)

    # Max drawdown from cumulative PnL
    cum_pnl = np.cumsum(pnl_series)
    peak = np.maximum.accumulate(cum_pnl)
    drawdown = peak - cum_pnl
    max_dd = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0

    wins = np.sum(pnl_series > 0)
    win_rate = float(wins / len(pnl_series)) if len(pnl_series) > 0 else 0.0

    return EvalMetrics(
        total_return=total_return,
        sharpe=sharpe,
        max_dd=max_dd,
        win_rate=win_rate,
        n_steps=len(pnl_series),
    )


def collect_predictions(
    scaled_features: np.ndarray,
    closes: np.ndarray,
    tqc_model,
    dt_model,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Walk through val data, collect TQC actions and DT predictions.

    Returns:
        tqc_actions: (n_steps,) TQC raw actions
        dt_preds: (n_steps,) DT predictions [-1, 1]
        dt_confs: (n_steps,) DT confidences [0, 1]
        price_returns: (n_steps,) next-bar returns
    """
    n = len(scaled_features)
    tqc_actions = np.zeros(n - 1)
    dt_preds = np.zeros(n - 1)
    dt_confs = np.zeros(n - 1)
    price_returns = np.zeros(n - 1)

    # TQC obs stacking buffer
    obs_buffer = deque(maxlen=N_STACK)

    # Simulated env state
    position = 0.0
    total_pnl = 0.0
    peak_balance = INITIAL_BALANCE

    # Reset DT context buffer
    dt_model._context_buffer = []

    for i in range(n - 1):
        # Build 126-dim observation
        features = scaled_features[i]  # (122,)
        cur_balance = INITIAL_BALANCE + total_pnl

        position_ratio = position
        position_direction = float(np.sign(position))
        pnl_ratio = total_pnl / INITIAL_BALANCE
        drawdown = max(0.0, (peak_balance - cur_balance) / peak_balance) if peak_balance > 0 else 0.0

        env_state = np.array(
            [position_ratio, position_direction, pnl_ratio, drawdown],
            dtype=np.float32,
        )
        obs = np.concatenate([features, env_state]).astype(np.float32)
        obs = np.clip(obs, -10.0, 10.0)

        # --- TQC prediction (stacked obs) ---
        obs_buffer.append(obs.copy())
        if len(obs_buffer) < N_STACK:
            # Zero-pad
            padded = [np.zeros(OBS_DIM, dtype=np.float32)] * (N_STACK - len(obs_buffer))
            padded.extend(obs_buffer)
        else:
            padded = list(obs_buffer)
        stacked = np.concatenate(padded).astype(np.float32)  # (1008,)

        tqc_action, _ = tqc_model.predict(
            stacked.reshape(1, -1), deterministic=True
        )
        tqc_act = float(np.clip(tqc_action[0], -1.0, 1.0))
        tqc_actions[i] = tqc_act

        # --- DT prediction (context buffer auto-managed) ---
        dt_pred, dt_conf = dt_model.predict(obs)
        dt_preds[i] = dt_pred
        dt_confs[i] = dt_conf

        # --- Price return ---
        if closes[i] > 0:
            price_returns[i] = (closes[i + 1] - closes[i]) / closes[i]

        # --- Update env state using TQC action (baseline position tracking) ---
        step_pnl = position * price_returns[i] * INITIAL_BALANCE
        total_pnl += step_pnl
        peak_balance = max(peak_balance, INITIAL_BALANCE + total_pnl)
        position = tqc_act  # Position = TQC action for next obs

    return tqc_actions, dt_preds, dt_confs, price_returns


def evaluate_weight(
    tqc_actions: np.ndarray,
    dt_preds: np.ndarray,
    dt_confs: np.ndarray,
    price_returns: np.ndarray,
    tqc_weight: float,
) -> EvalMetrics:
    """Evaluate a specific TQC/DT weight blend."""
    dt_weight = 1.0 - tqc_weight
    n = len(tqc_actions)

    pnl_series = np.zeros(n)
    position = 0.0

    for i in range(n):
        # Blend signals
        if dt_confs[i] >= DT_CONFIDENCE_THRESHOLD:
            action = tqc_weight * tqc_actions[i] + dt_weight * dt_preds[i]
        else:
            action = tqc_actions[i]  # DT not confident -> TQC only

        action = np.clip(action, -1.0, 1.0)

        # PnL from previous position
        pnl_series[i] = position * price_returns[i] * INITIAL_BALANCE

        # Update position for next step
        position = action

    return compute_metrics(pnl_series)


# --- Main ---

def run_grid_search(asset: str, manifest: dict):
    """Run ensemble grid search for one asset."""
    logger.info(f"\n{'='*60}")
    logger.info(f"  ENSEMBLE GRID SEARCH - {asset}")
    logger.info(f"{'='*60}")

    # 1. Load val data
    val_df, feature_cols = load_val_data(asset, manifest)

    # 2. Load scaler and scale features
    fold = BEST_FOLDS[asset]
    scaler_path = f"models/retrained/{asset}/{fold}/logs/LSTM_FILM_A/feature_scaler.pkl"
    no_scale = manifest.get("no_scale_features", [])

    if os.path.exists(scaler_path):
        logger.info(f"  Using TQC scaler: {scaler_path}")
        scaled_features = load_and_apply_scaler(val_df, feature_cols, scaler_path, no_scale)
    else:
        # Fallback to DT scaler
        dt_scaler_path = f"models/decision_transformer/{asset}/dt_scaler.pkl"
        logger.info(f"  TQC scaler not found, using DT scaler: {dt_scaler_path}")
        scaled_features = load_and_apply_scaler(val_df, feature_cols, dt_scaler_path, no_scale)

    closes = val_df["close"].values.astype(np.float64)

    # 3. Load models
    tqc_model = load_tqc_model(asset)
    dt_model = load_dt_model(asset)

    # 4. Collect predictions
    logger.info(f"  Collecting predictions on {len(scaled_features)} val rows...")
    tqc_actions, dt_preds, dt_confs, price_returns = collect_predictions(
        scaled_features, closes, tqc_model, dt_model
    )

    # 5. Evaluate weight grid
    results = {}
    for tqc_w in TQC_WEIGHT_GRID:
        metrics = evaluate_weight(tqc_actions, dt_preds, dt_confs, price_returns, tqc_w)
        results[tqc_w] = metrics

    # 6. Print results
    baseline = results[1.0]  # TQC-only

    logger.info(f"\n  {'TQC_W':>6} | {'DT_W':>5} | {'Return':>10} | {'Sharpe':>8} | "
                f"{'MaxDD':>8} | {'WinRate':>7} | {'Gate':>6}")
    logger.info(f"  {'-'*6}-+-{'-'*5}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}-+-{'-'*6}")

    best_passing = None
    best_sharpe = -999

    for tqc_w in TQC_WEIGHT_GRID:
        m = results[tqc_w]
        dt_w = 1.0 - tqc_w

        # Gate check
        if tqc_w == 1.0:
            gate = "BASE"
        else:
            sharpe_ok = m.sharpe >= baseline.sharpe * SHARPE_GATE
            dd_ok = (m.max_dd <= baseline.max_dd * MAXDD_GATE) if baseline.max_dd > 0 else True
            gate = "PASS" if (sharpe_ok and dd_ok) else "FAIL"

            if gate == "PASS" and m.sharpe > best_sharpe:
                best_sharpe = m.sharpe
                best_passing = tqc_w

        logger.info(
            f"  {tqc_w:>6.2f} | {dt_w:>5.2f} | "
            f"${m.total_return:>9.2f} | {m.sharpe:>8.4f} | "
            f"${m.max_dd:>7.2f} | {m.win_rate:>6.1%} | {gate:>6}"
        )

    # 7. Recommendation
    logger.info(f"\n  --- RECOMMENDATION for {asset} ---")

    # Signal agreement stats
    same_sign = np.mean((tqc_actions > 0) == (dt_preds > 0))
    mean_dt_conf = np.mean(dt_confs)
    high_conf_pct = np.mean(dt_confs >= DT_CONFIDENCE_THRESHOLD)

    logger.info(f"  Signal agreement: {same_sign:.1%}")
    logger.info(f"  Mean DT confidence: {mean_dt_conf:.3f}")
    logger.info(f"  DT high-conf rate (>={DT_CONFIDENCE_THRESHOLD}): {high_conf_pct:.1%}")

    if best_passing is not None:
        dt_w = 1.0 - best_passing
        logger.info(f"  >>> DEPLOY ENSEMBLE: TQC={best_passing:.2f}, DT={dt_w:.2f}")
        logger.info(f"  >>> Sharpe improvement: {results[best_passing].sharpe / baseline.sharpe:.2%} of TQC-only")
    else:
        logger.info(f"  >>> TQC ONLY - DT to SHADOW MODE")
        logger.info(f"  >>> No weight combination passed the gate criteria")

    return results, best_passing


def main():
    import argparse

    parser = argparse.ArgumentParser(description="TQC+DT Ensemble Grid Search")
    parser.add_argument("--asset", choices=["BTC", "ETH", "SOL"], default=None,
                        help="Single asset (default: all)")
    args = parser.parse_args()

    manifest = load_feature_manifest()
    assets = [args.asset] if args.asset else ["BTC", "ETH", "SOL"]

    all_results = {}
    for asset in assets:
        try:
            results, best_w = run_grid_search(asset, manifest)
            all_results[asset] = {"results": results, "best_weight": best_w}
        except Exception as e:
            logger.error(f"  {asset} FAILED: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    if len(all_results) > 1:
        logger.info(f"\n{'='*60}")
        logger.info(f"  SUMMARY")
        logger.info(f"{'='*60}")
        for asset, data in all_results.items():
            bw = data["best_weight"]
            if bw is not None:
                m = data["results"][bw]
                logger.info(f"  {asset}: ENSEMBLE TQC={bw:.2f} (Sharpe={m.sharpe:.4f})")
            else:
                m = data["results"][1.0]
                logger.info(f"  {asset}: TQC ONLY (Sharpe={m.sharpe:.4f})")


if __name__ == "__main__":
    main()
