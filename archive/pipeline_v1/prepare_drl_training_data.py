"""
HMATS v6.5.2 - Phase 2: DRL Training Data Preparation

Prepares per-asset parquet files with:
  1. 128-dim FeatureEngineer features (from OHLCV)
  2. GMM regime labels (integer 0-5)
  3. GMM regime confidence
  4. RegimeSmoother applied (persistence=2, matching runtime)

Usage:
    python scripts/prepare_drl_training_data.py                # Use cached OHLCV
    python scripts/prepare_drl_training_data.py --fetch         # Fetch fresh data (2000 bars)
    python scripts/prepare_drl_training_data.py --bars 1500     # Custom bar count

Output:
    data/drl_training/{BTC,ETH,SOL}_features.parquet
"""

import argparse
import importlib.util
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ── Project setup ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import data fetching from retrain_gmm.py
_spec = importlib.util.spec_from_file_location(
    "retrain_gmm", PROJECT_ROOT / "scripts" / "retrain_gmm.py"
)
_retrain_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_retrain_mod)

fetch_all_assets = _retrain_mod.fetch_all_assets
load_data_cache = _retrain_mod.load_data_cache
save_data_cache = _retrain_mod.save_data_cache

# Import FeatureEngineer from training code
_feat_spec = importlib.util.spec_from_file_location(
    "features", PROJECT_ROOT / "hmats_training_v5 (1)" / "drl" / "features.py"
)
_feat_mod = importlib.util.module_from_spec(_feat_spec)
_feat_spec.loader.exec_module(_feat_mod)
FeatureEngineer = _feat_mod.FeatureEngineer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DRL_DataPrep")

# ── Constants ─────────────────────────────────────────────────────────
GMM_DIR = PROJECT_ROOT / "models" / "regime_classifier"
OUTPUT_DIR = PROJECT_ROOT / "data" / "drl_training"
ASSETS = ["BTC", "ETH", "SOL"]


# ── RegimeSmoother ────────────────────────────────────────────────────
from core.regime_smoother import RegimeSmoother  # noqa: E402


def load_gmm():
    """Load pretrained GMM model, scaler params, and config."""
    config_path = GMM_DIR / "gmm_config.json"
    model_path = GMM_DIR / "gmm_model.pkl"

    if not config_path.exists() or not model_path.exists():
        raise FileNotFoundError(f"GMM files not found in {GMM_DIR}")

    with open(config_path) as f:
        config = json.load(f)

    model = joblib.load(model_path)
    scaler_mean = np.array(config["scaler_mean"])
    scaler_scale = np.array(config["scaler_scale"])
    feature_cols = config["feature_cols"]
    regime_names = config["regime_names"]

    logger.info(f"GMM loaded: {len(feature_cols)} features, {len(regime_names)} regimes")
    return model, scaler_mean, scaler_scale, feature_cols, regime_names


def compute_gmm_features(df: pd.DataFrame) -> np.ndarray:
    """Compute the 17 GMM features from OHLCV DataFrame.

    Mirrors main.py::_predict_gmm_regime() computation exactly.
    Returns (N, 17) array.
    """
    closes = df["close"].values
    volumes = df["volume"].values
    n = len(closes)

    # Pre-compute returns
    rets = np.zeros(n)
    rets[1:] = np.diff(closes) / np.where(closes[:-1] != 0, closes[:-1], 1.0)

    features_list = []
    for i in range(n):
        if i < 50:
            features_list.append(np.full(17, np.nan))
            continue

        ret_4h = rets[i]
        ret_1h = ret_4h / 4.0
        ret_24h = (closes[i] - closes[i - 6]) / closes[i - 6] if closes[i - 6] > 0 else 0.0
        ret_7d = (closes[i] - closes[i - 42]) / closes[i - 42] if i >= 42 and closes[i - 42] > 0 else 0.0

        vol_1h = float(np.std(rets[max(0, i - 1):i + 1])) if i >= 1 else 0.02
        vol_24h = float(np.std(rets[max(0, i - 5):i + 1])) if i >= 5 else 0.02
        vol_pct = float(np.searchsorted(np.sort(volumes[:i + 1]), volumes[i]) / (i + 1) * 100)

        if i >= 30:
            rolling_v = [float(np.std(rets[max(0, j - 5):j + 1])) for j in range(max(6, i - 19), i + 1)]
            vov = float(np.std(rolling_v)) if len(rolling_v) >= 2 else 0.0
        else:
            vov = 0.0

        mom_4h = ret_4h
        mom_24h = ret_24h

        if i >= 5:
            recent = rets[max(0, i - 5):i + 1]
            pos = int(np.sum(recent > 0))
            neg = int(np.sum(recent < 0))
            mom_con = (max(pos, neg) / len(recent)) * (1 if pos >= neg else -1)
        else:
            mom_con = 0.0

        # Cross-asset correlation: use training default (0.87)
        cross_corr = 0.87
        corr_regime = 1.0 if cross_corr > 0.7 else (0.0 if cross_corr < 0.3 else 0.5)

        # Fear index from RSI proxy
        # Simple RSI: Wilder's method over 14 bars
        if i >= 14:
            deltas = np.diff(closes[i - 14:i + 1])
            gains = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0.0
            losses = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0.0
            rs = gains / (losses + 1e-10)
            rsi = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi = 50.0
        fear_idx = 100.0 - rsi

        sent_mom = 0.0
        # Spread percentile: default to training mean
        spread_pct = 41.67
        depth_chg = 0.0

        features_list.append(np.array([
            ret_1h, ret_4h, ret_24h, ret_7d,
            vol_1h, vol_24h, vol_pct, vov,
            mom_4h, mom_24h, mom_con,
            cross_corr, corr_regime,
            fear_idx, sent_mom,
            spread_pct, depth_chg,
        ]))

    return np.array(features_list)


def predict_regimes(gmm_features: np.ndarray, model, scaler_mean, scaler_scale):
    """Run GMM prediction on all bars. Returns (regime_ids, confidences)."""
    n = len(gmm_features)
    regime_ids = np.full(n, -1, dtype=int)
    confidences = np.full(n, 0.0)

    for i in range(n):
        if np.any(np.isnan(gmm_features[i])):
            continue

        scaled = (gmm_features[i] - scaler_mean) / scaler_scale
        scaled = np.clip(scaled, -4.0, 4.0)

        probs = model.predict_proba(scaled.reshape(1, -1))[0]
        regime_ids[i] = int(np.argmax(probs))
        confidences[i] = float(probs[regime_ids[i]])

    return regime_ids, confidences


def verify_fold_coverage(df: pd.DataFrame, n_folds: int = 3, min_regimes: int = 3):
    """Verify regime coverage per fold. Returns list of issues."""
    n = len(df)
    val_size = int(n * 0.15)
    gap = 168
    issues = []

    for fold_idx in range(n_folds):
        val_end = n - fold_idx * val_size
        val_start = val_end - val_size
        train_end = val_start - gap

        if train_end <= 0:
            issues.append(f"Fold {fold_idx + 1}: train_end <= 0 (not enough data)")
            continue

        train_regimes = df["regime"].iloc[:train_end].unique()
        val_regimes = df["regime"].iloc[val_start:val_end].unique()

        # Filter out -1 (invalid bars)
        train_regimes = [r for r in train_regimes if r >= 0]
        val_regimes = [r for r in val_regimes if r >= 0]

        if len(train_regimes) < min_regimes:
            issues.append(
                f"Fold {fold_idx + 1}: train has only {len(train_regimes)} regimes "
                f"(need {min_regimes}): {sorted(train_regimes)}"
            )
        if len(val_regimes) < min_regimes:
            issues.append(
                f"Fold {fold_idx + 1}: val has only {len(val_regimes)} regimes "
                f"(need {min_regimes}): {sorted(val_regimes)}"
            )

        logger.info(
            f"  Fold {fold_idx + 1}: train={train_end} bars ({len(train_regimes)} regimes), "
            f"val={val_size} bars ({len(val_regimes)} regimes)"
        )

    return issues


def main():
    parser = argparse.ArgumentParser(description="Prepare DRL training data with GMM regime labels")
    parser.add_argument("--fetch", action="store_true", help="Fetch fresh data from Kraken")
    parser.add_argument("--bars", type=int, default=2000, help="Number of 4H bars per asset")
    parser.add_argument("--smooth", type=int, default=2, help="RegimeSmoother persistence (0=disable)")
    parser.add_argument("--folds", type=int, default=3, help="K-fold count for coverage check")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("DRL Training Data Preparation (v6.5.2)")
    logger.info("=" * 60)

    # Step 1: Load OHLCV data
    if args.fetch:
        logger.info(f"\nStep 1: Fetching {args.bars} 4H bars per asset from Kraken...")
        data = fetch_all_assets(limit=args.bars)
        save_data_cache(data)
    else:
        logger.info("\nStep 1: Loading cached OHLCV data...")
        data = load_data_cache()

    # Step 2: Load GMM
    logger.info("\nStep 2: Loading pretrained GMM model...")
    model, scaler_mean, scaler_scale, gmm_feat_cols, regime_names = load_gmm()

    # Step 3: Compute features + regimes per asset
    fe = FeatureEngineer(state_dim=128, cross_asset_dim=16)
    smoother = RegimeSmoother(min_persistence=args.smooth) if args.smooth > 0 else None
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for asset_name in ASSETS:
        logger.info(f"\n{'='*40}")
        logger.info(f"Processing {asset_name}...")
        logger.info(f"{'='*40}")

        df = data[asset_name].copy()
        logger.info(f"  Raw OHLCV: {len(df)} bars")

        # 3a. Compute 128-dim DRL features
        logger.info("  Computing 128-dim features...")
        df = fe.compute_features(df)
        drl_feature_cols = fe.get_feature_columns()
        logger.info(f"  Feature columns: {len(drl_feature_cols)}")

        # 3b. Compute GMM features and predict regimes
        logger.info("  Computing GMM features + regime labels...")
        gmm_features = compute_gmm_features(df)
        regime_ids, confidences = predict_regimes(gmm_features, model, scaler_mean, scaler_scale)

        df["regime"] = regime_ids
        df["regime_confidence"] = confidences

        # Stats before smoothing
        valid_mask = regime_ids >= 0
        n_valid = int(np.sum(valid_mask))
        regime_dist = pd.Series(regime_ids[valid_mask]).value_counts().sort_index()
        logger.info(f"  Valid regime labels: {n_valid}/{len(df)}")
        for idx, count in regime_dist.items():
            name = regime_names[idx] if idx < len(regime_names) else f"R{idx}"
            logger.info(f"    {name}: {count} ({count/n_valid*100:.1f}%)")

        # 3c. Apply RegimeSmoother
        if smoother is not None:
            before_flips = sum(
                1 for i in range(1, len(df))
                if df["regime"].iloc[i] != df["regime"].iloc[i - 1]
                and df["regime"].iloc[i] >= 0 and df["regime"].iloc[i - 1] >= 0
            )
            df = smoother.smooth_column(df, "regime")
            after_flips = sum(
                1 for i in range(1, len(df))
                if df["regime"].iloc[i] != df["regime"].iloc[i - 1]
                and df["regime"].iloc[i] >= 0 and df["regime"].iloc[i - 1] >= 0
            )
            logger.info(
                f"  RegimeSmoother(persistence={args.smooth}): "
                f"flips {before_flips} -> {after_flips}"
            )

        # 3d. Drop rows with invalid regimes (first 50 bars)
        df = df[df["regime"] >= 0].reset_index(drop=True)
        logger.info(f"  After filtering invalid bars: {len(df)} rows")

        # 3e. Verify fold regime coverage
        logger.info(f"  Fold regime coverage ({args.folds} folds):")
        issues = verify_fold_coverage(df, n_folds=args.folds)
        if issues:
            for issue in issues:
                logger.warning(f"    ISSUE: {issue}")
        else:
            logger.info("    All folds OK")

        # 3f. Save
        out_path = OUTPUT_DIR / f"{asset_name}_features.parquet"
        # Ensure we have all needed columns
        keep_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        keep_cols += [c for c in drl_feature_cols if c in df.columns]
        keep_cols += ["regime", "regime_confidence"]
        df[keep_cols].to_parquet(out_path, index=False)
        logger.info(f"  Saved: {out_path} ({len(df)} rows, {len(keep_cols)} cols)")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("DRL Training Data Preparation Complete")
    logger.info("=" * 60)
    for asset_name in ASSETS:
        out_path = OUTPUT_DIR / f"{asset_name}_features.parquet"
        if out_path.exists():
            df_check = pd.read_parquet(out_path)
            float_cols = [c for c in df_check.columns
                         if df_check[c].dtype in ["float64", "float32"]
                         and c not in ["timestamp", "open", "high", "low", "close", "volume"]]
            logger.info(
                f"  {asset_name}: {len(df_check)} rows, "
                f"{len(float_cols)} float features + regime"
            )

    logger.info(f"\nOutput directory: {OUTPUT_DIR}")
    logger.info("Next: python \"hmats_training_v5 (1)/train_tqc.py\" --data data/drl_training/<ASSET>_features.parquet ...")


if __name__ == "__main__":
    main()
