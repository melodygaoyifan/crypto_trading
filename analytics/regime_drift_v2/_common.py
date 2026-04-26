"""共用: GMM 加载 + feature pipeline + holdout 数据."""
from __future__ import annotations
import json
import sys
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ))

DEPLOYED_DIR = PROJ / 'models' / 'regime_classifier'
CANDIDATE_DIR = PROJ / 'training' / 'training_data' / 'gmm_models'

ASSETS = ['BTC', 'ETH', 'SOL']

# Reuse the canonical 12-feature builder used by training/scripts/train_per_asset_gmm.py.
from training.scripts.train_per_asset_gmm import (
    compute_gmm_features_batch as _compute_gmm_features_batch,
    GMM_FEATURE_COLS as _CANONICAL_FEATURE_COLS,
)


def load_gmm_bundle(directory: Path, asset: str) -> dict:
    """Load gmm_model.pkl + scaler.pkl + gmm_config.json from `directory/{asset}/`."""
    asset_dir = directory / asset
    model_path = asset_dir / 'gmm_model.pkl'
    scaler_path = asset_dir / 'scaler.pkl'
    config_path = asset_dir / 'gmm_config.json'
    if not model_path.exists():
        raise FileNotFoundError(f"GMM not found: {model_path}")

    gmm = joblib.load(model_path)
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    cfg = {}
    if config_path.exists():
        with config_path.open() as f:
            cfg = json.load(f)
    return {
        'gmm': gmm,
        'scaler': scaler,
        'feature_cols': cfg.get('feature_cols') or list(_CANONICAL_FEATURE_COLS),
        'regime_names': cfg.get('regime_names'),
        'regime_mapping': cfg.get('regime_mapping'),
        'config': cfg,
        'source': str(model_path),
        'mtime': model_path.stat().st_mtime,
    }


def get_ohlcv(asset: str, days: int | None = None) -> pd.DataFrame:
    candidates = [
        PROJ / f'training/training_data/drl_training/{asset}_4H_full.parquet',
        PROJ / f'training/training_data/drl_training/{asset.lower()}_4H_full.parquet',
    ]
    for f in candidates:
        if f.exists():
            df = pd.read_parquet(f)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            elif df.index.name == 'timestamp':
                df = df.reset_index()
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            if days:
                cutoff = df['timestamp'].max() - pd.Timedelta(days=days)
                df = df[df['timestamp'] >= cutoff].reset_index(drop=True)
            return df
    raise FileNotFoundError(f"No OHLCV found for {asset}")


def build_features_unified(df: pd.DataFrame, feature_cols: list[str], asset: str) -> pd.DataFrame:
    """Compute the 12 GMM features inline (mirrors main.py _predict_gmm_regime).

    `feature_cols` is the order the calling GMM expects; we reorder if needed.
    """
    raw = _compute_gmm_features_batch(df, asset=asset)  # (n, 12)
    canonical = list(_CANONICAL_FEATURE_COLS)
    feats = pd.DataFrame(raw, columns=canonical)
    # Drop rows with NaN (first ~50 bars of warm-up)
    feats = feats.dropna().reset_index(drop=True)
    missing = [c for c in feature_cols if c not in feats.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    return feats[list(feature_cols)]


def predict_labels_with_bundle(bundle: dict, raw_features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Use bundle's own scaler, then predict label + max_posterior."""
    gmm = bundle['gmm']
    scaler = bundle['scaler']
    if scaler is None:
        X = raw_features.values
    else:
        X = scaler.transform(raw_features.values)
    labels = gmm.predict(X)
    posteriors = gmm.predict_proba(X)
    return labels, posteriors.max(axis=1)


def get_aligned_features_and_df(asset: str, days: int | None, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (df_aligned_to_features, raw_features). df is trimmed to feature row count."""
    df = get_ohlcv(asset, days=days)
    feats = build_features_unified(df, feature_cols, asset)
    # Features drop ~50 warm-up bars; align tail
    df_aligned = df.iloc[-len(feats):].reset_index(drop=True)
    return df_aligned, feats
