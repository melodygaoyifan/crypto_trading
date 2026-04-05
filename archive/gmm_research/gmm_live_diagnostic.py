"""
================================================================================
HMATS v6.5.2 - GMM Live Diagnostic
================================================================================

Diagnoses why live GMM inference produces confidence=1.0 (degenerate posteriors).

Three checks:
  1. Print full predict_proba() 6-value vector for each asset
  2. Verify scaler file matches training (hash + stats comparison)
  3. Compare live feature values vs training feature distribution

Usage:
    python scripts/gmm_live_diagnostic.py

Requires: ccxt, joblib, numpy, pandas
================================================================================
"""

import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("gmm_diag")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_DIR = PROJECT_ROOT / "models" / "regime_classifier"
CACHE_DIR = PROJECT_ROOT / "data" / "gmm_training_cache"

FEATURE_COLS = [
    "return_1h", "return_4h", "return_24h", "return_7d",
    "volatility_1h", "volatility_24h", "vol_percentile", "vol_of_vol",
    "momentum_4h", "momentum_24h", "momentum_consistency",
    "cross_asset_correlation", "correlation_regime",
    "fear_index", "sentiment_momentum",
    "spread_percentile", "depth_change",
]

ASSETS = {"BTC/USD": "BTC", "ETH/USD": "ETH", "SOL/USD": "SOL"}
SPREAD_DEFAULTS = {"BTC": 5.0, "ETH": 8.0, "SOL": 12.0}


# ---------------------------------------------------------------------------
# Step 1: Fetch live OHLCV from Kraken (identical to main.py)
# ---------------------------------------------------------------------------
def fetch_live_ohlcv(symbol: str, limit: int = 250) -> pd.DataFrame:
    """Fetch 4H OHLCV bars from Kraken via ccxt."""
    import ccxt
    exchange = ccxt.kraken({"enableRateLimit": True})
    bars = exchange.fetch_ohlcv(symbol, "4h", limit=limit)
    df = pd.DataFrame(bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


# ---------------------------------------------------------------------------
# Step 2: Compute 17 GMM features - IDENTICAL to main.py:_predict_gmm_regime()
# ---------------------------------------------------------------------------
def compute_live_features(df: pd.DataFrame, asset: str, rsi_val: float = 50.0) -> np.ndarray:
    """
    Compute the 17-feature vector from OHLCV DataFrame.
    Matches main.py:3920-3974 exactly.
    """
    closes = df["close"].values
    vols = df["volume"].values
    n = len(closes)

    if n < 50:
        raise ValueError(f"Need >= 50 bars, got {n}")

    rets = np.diff(closes) / np.where(closes[:-1] != 0, closes[:-1], 1.0)

    ret_4h = float(rets[-1]) if len(rets) > 0 else 0.0
    ret_1h = ret_4h / 4.0
    ret_24h = float((closes[-1] - closes[-7]) / closes[-7]) if n >= 7 and closes[-7] > 0 else 0.0
    ret_7d = float((closes[-1] - closes[-43]) / closes[-43]) if n >= 43 and closes[-43] > 0 else 0.0

    vol_1h = float(np.std(rets[-2:])) if len(rets) >= 2 else 0.02
    vol_24h = float(np.std(rets[-6:])) if len(rets) >= 6 else 0.02
    vol_pct = float(np.searchsorted(np.sort(vols), vols[-1]) / n * 100) if n > 0 else 50.0

    if len(rets) >= 30:
        rolling_v = [float(np.std(rets[max(0, i - 6):i])) for i in range(6, len(rets))]
        vov = float(np.std(rolling_v[-20:])) if len(rolling_v) >= 20 else 0.0
    else:
        vov = 0.0

    mom_4h = ret_4h
    mom_24h = ret_24h

    if len(rets) >= 6:
        recent = rets[-6:]
        pos = int(np.sum(recent > 0))
        neg = int(np.sum(recent < 0))
        mom_con = (max(pos, neg) / 6.0) * (1 if pos >= neg else -1)
    else:
        mom_con = 0.0

    # CRITICAL: must match main.py defaults
    cross_corr = 0.87  # Default near training mean (0.872)
    corr_regime = 1.0 if cross_corr > 0.7 else (0.0 if cross_corr < 0.3 else 0.5)

    # Fear index from RSI
    fear_idx = 100.0 - rsi_val
    sent_mom = 0.0

    # Per-asset spread defaults matching training (retrain_gmm.py:370-371)
    spread_bps = SPREAD_DEFAULTS.get(asset, 10.0)
    spread_pct = min(spread_bps / 20.0 * 100, 100.0)
    depth_chg = 0.0

    features = np.array([
        ret_1h, ret_4h, ret_24h, ret_7d,
        vol_1h, vol_24h, vol_pct, vov,
        mom_4h, mom_24h, mom_con,
        cross_corr, corr_regime,
        fear_idx, sent_mom,
        spread_pct, depth_chg,
    ], dtype=np.float64)

    return features


def compute_rsi(closes: np.ndarray, period: int = 14) -> float:
    """Compute RSI(14) for the last bar."""
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


# ---------------------------------------------------------------------------
# Step 3: Load model + scaler
# ---------------------------------------------------------------------------
def load_gmm():
    gmm = joblib.load(MODEL_DIR / "gmm_model.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    with open(MODEL_DIR / "gmm_config.json") as f:
        config = json.load(f)
    return gmm, scaler, config


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Step 4: Load training data for comparison
# ---------------------------------------------------------------------------
def load_training_features():
    """Load cached training data and recompute features to get distribution."""
    from scripts.retrain_gmm import (
        compute_features_for_bar, compute_rsi as compute_rsi_full,
        compute_cross_asset_correlation, FEATURE_COLS as TRAIN_FEATURE_COLS,
    )

    data = {}
    for name in ["BTC", "ETH", "SOL"]:
        path = CACHE_DIR / f"{name}_4h.parquet"
        if path.exists():
            data[name] = pd.read_parquet(path)
            logger.info(f"  Loaded training cache: {name} ({len(data[name])} bars)")
        else:
            logger.warning(f"  Training cache not found: {path}")

    if not data:
        return None

    # Compute features for all bars (same as retrain_gmm.py:348-384)
    btc_closes = data.get("BTC", pd.DataFrame({"close": []}))["close"].values if "BTC" in data else np.array([])
    eth_closes = data.get("ETH", pd.DataFrame({"close": []}))["close"].values if "ETH" in data else np.array([])
    sol_closes = data.get("SOL", pd.DataFrame({"close": []}))["close"].values if "SOL" in data else np.array([])

    cross_corr_arr = compute_cross_asset_correlation(btc_closes, eth_closes, sol_closes)

    all_features = []
    for asset_name, df in data.items():
        closes = df["close"].values
        volumes = df["volume"].values
        n = len(closes)
        rsi_arr = compute_rsi_full(closes)
        spread_map = {"BTC": 5.0, "ETH": 8.0, "SOL": 12.0}
        spread_bps = spread_map.get(asset_name, 10.0)

        for idx in range(50, n):
            min_len = min(len(btc_closes), len(eth_closes), len(sol_closes))
            corr_idx = idx - (n - min_len + 1)
            if 0 <= corr_idx < len(cross_corr_arr):
                xcorr = cross_corr_arr[corr_idx]
            else:
                xcorr = 0.65

            feat = compute_features_for_bar(
                closes, volumes, idx,
                rsi_val=rsi_arr[idx],
                cross_corr=xcorr,
                spread_bps=spread_bps,
            )
            if feat is not None:
                all_features.append(feat)

    return np.array(all_features)


# ---------------------------------------------------------------------------
# Main diagnostic
# ---------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("HMATS GMM LIVE DIAGNOSTIC")
    print("=" * 80)

    # ------------------------------------------------------------------
    # CHECK 1: Scaler file verification
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("CHECK 1: SCALER FILE VERIFICATION")
    print("=" * 80)

    gmm, scaler, config = load_gmm()

    # Hash the files
    model_hash = file_hash(MODEL_DIR / "gmm_model.pkl")
    scaler_hash = file_hash(MODEL_DIR / "scaler.pkl")
    print(f"\nFile hashes:")
    print(f"  gmm_model.pkl:  {model_hash}")
    print(f"  scaler.pkl:     {scaler_hash}")

    # Compare scaler stats from pkl vs config
    config_mean = np.array(config["scaler_mean"])
    config_scale = np.array(config["scaler_scale"])
    pkl_mean = scaler.mean_
    pkl_scale = scaler.scale_

    print(f"\nScaler mean comparison (pkl vs config JSON):")
    mean_match = np.allclose(pkl_mean, config_mean, atol=1e-10)
    scale_match = np.allclose(pkl_scale, config_scale, atol=1e-10)
    print(f"  Mean match:  {'YES' if mean_match else 'MISMATCH!'}")
    print(f"  Scale match: {'YES' if scale_match else 'MISMATCH!'}")

    if not mean_match:
        for i in range(len(FEATURE_COLS)):
            if abs(pkl_mean[i] - config_mean[i]) > 1e-10:
                print(f"  DIFF {FEATURE_COLS[i]}: pkl={pkl_mean[i]:.10f} vs config={config_mean[i]:.10f}")
    if not scale_match:
        for i in range(len(FEATURE_COLS)):
            if abs(pkl_scale[i] - config_scale[i]) > 1e-10:
                print(f"  DIFF {FEATURE_COLS[i]}: pkl={pkl_scale[i]:.10f} vs config={config_scale[i]:.10f}")

    print(f"\nScaler statistics (from pkl):")
    print(f"  {'Feature':<30s} {'Mean':>12s} {'Scale':>12s}")
    print(f"  {'-'*30} {'-'*12} {'-'*12}")
    for i, name in enumerate(FEATURE_COLS):
        print(f"  {name:<30s} {pkl_mean[i]:>12.6f} {pkl_scale[i]:>12.6f}")

    print(f"\nGMM config:")
    print(f"  n_components:    {config['n_components']}")
    print(f"  covariance_type: {config['covariance_type']}")
    print(f"  regime_names:    {config['regime_names']}")
    print(f"  weights:         {[f'{w:.4f}' for w in config['weights']]}")
    print(f"  trained_at:      {config['trained_at']}")
    print(f"  training_samples:{config['training_samples']}")

    # ------------------------------------------------------------------
    # CHECK 2: Live predict_proba() for each asset
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("CHECK 2: LIVE predict_proba() VECTORS")
    print("=" * 80)

    regime_names = config["regime_names"]
    live_features = {}

    for symbol, asset in ASSETS.items():
        print(f"\n--- {asset} ({symbol}) ---")
        try:
            df = fetch_live_ohlcv(symbol, limit=250)
            print(f"  Fetched {len(df)} bars, latest: {df['timestamp'].iloc[-1]}")
            print(f"  Latest close: ${df['close'].iloc[-1]:,.2f}")

            rsi = compute_rsi(df["close"].values)
            features = compute_live_features(df, asset, rsi_val=rsi)
            live_features[asset] = features

            # Print raw features
            print(f"\n  RAW features (17):")
            for i, name in enumerate(FEATURE_COLS):
                print(f"    {name:<30s} = {features[i]:>14.8f}")

            # Scale
            scaled = (features - pkl_mean) / pkl_scale
            print(f"\n  Z-SCORES (after scaling):")
            for i, name in enumerate(FEATURE_COLS):
                z = scaled[i]
                flag = " *** EXTREME" if abs(z) > 3.0 else ""
                print(f"    {name:<30s} = {z:>10.4f}{flag}")

            n_extreme = int(np.sum(np.abs(scaled) > 3.0))
            print(f"\n  Extreme z-scores (|z|>3): {n_extreme}/{len(scaled)}")

            if n_extreme > len(scaled) * 0.3:
                print(f"  >>> DISTRIBUTION SHIFT DETECTED - main.py falls back to ADX <<<")

            # Clip z-scores (same as main.py)
            scaled_clipped = np.clip(scaled, -4.0, 4.0)

            # predict_proba
            probs = gmm.predict_proba(scaled_clipped.reshape(1, -1))[0]
            regime_idx = int(np.argmax(probs))
            confidence = float(probs[regime_idx])

            print(f"\n  predict_proba() FULL VECTOR:")
            for i, name in enumerate(regime_names):
                marker = " <-- WINNER" if i == regime_idx else ""
                print(f"    [{i}] {name:<25s} = {probs[i]:.15e}{marker}")

            print(f"\n  Predicted: {regime_names[regime_idx]} (conf={confidence:.10f})")

            if confidence > 0.9999:
                print(f"  >>> DEGENERATE POSTERIOR - main.py falls back to ADX <<<")
            elif confidence > 0.95:
                print(f"  >>> HIGH CONFIDENCE - within normal range for full-cov GMM <<<")

            # Also show un-clipped proba
            probs_unclipped = gmm.predict_proba(scaled.reshape(1, -1))[0]
            conf_unclipped = float(probs_unclipped[np.argmax(probs_unclipped)])
            if not np.allclose(probs, probs_unclipped, atol=1e-10):
                print(f"\n  predict_proba() WITHOUT z-clip:")
                for i, name in enumerate(regime_names):
                    print(f"    [{i}] {name:<25s} = {probs_unclipped[i]:.15e}")
                print(f"  Unclipped conf: {conf_unclipped:.10f}")

            time.sleep(1)  # Rate limit

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # ------------------------------------------------------------------
    # CHECK 3: Live vs Training feature distribution
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("CHECK 3: LIVE FEATURES vs TRAINING DISTRIBUTION")
    print("=" * 80)

    try:
        train_features = load_training_features()
        if train_features is not None:
            print(f"\nTraining data: {train_features.shape[0]} samples × {train_features.shape[1]} features")

            # Scale training features with the same scaler
            train_scaled = (train_features - pkl_mean) / pkl_scale

            print(f"\n  {'Feature':<25s} | {'Train Mean':>10s} {'Train Std':>10s} {'Train Min':>10s} {'Train Max':>10s} | ", end="")
            for asset in live_features:
                print(f"{'Live ' + asset:>10s} ", end="")
            print(f"| ", end="")
            for asset in live_features:
                print(f"{'z(' + asset + ')':>10s} ", end="")
            print()
            print(f"  {'-'*25}-+-{'-'*10}-{'-'*10}-{'-'*10}-{'-'*10}-+-", end="")
            for _ in live_features:
                print(f"{'-'*10}-", end="")
            print(f"+-", end="")
            for _ in live_features:
                print(f"{'-'*10}-", end="")
            print()

            for i, name in enumerate(FEATURE_COLS):
                col = train_features[:, i]
                print(f"  {name:<25s} | {np.mean(col):>10.6f} {np.std(col):>10.6f} {np.min(col):>10.6f} {np.max(col):>10.6f} | ", end="")
                for asset in live_features:
                    print(f"{live_features[asset][i]:>10.6f} ", end="")
                print(f"| ", end="")
                for asset in live_features:
                    z = (live_features[asset][i] - pkl_mean[i]) / pkl_scale[i]
                    flag = "*" if abs(z) > 3 else " "
                    print(f"{z:>9.4f}{flag}", end="")
                print()

            # Training z-score distribution
            print(f"\n  Training z-score percentiles:")
            print(f"  {'Feature':<25s} | {'1%':>8s} {'5%':>8s} {'25%':>8s} {'50%':>8s} {'75%':>8s} {'95%':>8s} {'99%':>8s} | {'|z|>3':>6s}")
            print(f"  {'-'*25}-+-{'-'*8}-{'-'*8}-{'-'*8}-{'-'*8}-{'-'*8}-{'-'*8}-{'-'*8}-+-{'-'*6}")
            for i, name in enumerate(FEATURE_COLS):
                col_z = train_scaled[:, i]
                pcts = np.percentile(col_z, [1, 5, 25, 50, 75, 95, 99])
                n_extreme = np.sum(np.abs(col_z) > 3)
                print(f"  {name:<25s} | {pcts[0]:>8.3f} {pcts[1]:>8.3f} {pcts[2]:>8.3f} {pcts[3]:>8.3f} {pcts[4]:>8.3f} {pcts[5]:>8.3f} {pcts[6]:>8.3f} | {n_extreme:>6d}")

            # Training predict_proba distribution
            print(f"\n  Training predict_proba confidence distribution:")
            train_clipped = np.clip(train_scaled, -4.0, 4.0)
            train_probs = gmm.predict_proba(train_clipped)
            train_conf = np.max(train_probs, axis=1)
            train_labels = np.argmax(train_probs, axis=1)

            print(f"    mean conf:   {np.mean(train_conf):.4f}")
            print(f"    std conf:    {np.std(train_conf):.4f}")
            print(f"    min conf:    {np.min(train_conf):.4f}")
            print(f"    max conf:    {np.max(train_conf):.4f}")
            print(f"    conf > 0.99: {np.sum(train_conf > 0.99)}/{len(train_conf)} ({np.sum(train_conf > 0.99)/len(train_conf)*100:.1f}%)")
            print(f"    conf > 0.999:{np.sum(train_conf > 0.999)}/{len(train_conf)} ({np.sum(train_conf > 0.999)/len(train_conf)*100:.1f}%)")
            print(f"    conf > 0.9999:{np.sum(train_conf > 0.9999)}/{len(train_conf)} ({np.sum(train_conf > 0.9999)/len(train_conf)*100:.1f}%)")
            print(f"    conf = 1.0:  {np.sum(train_conf >= 1.0 - 1e-15)}/{len(train_conf)} ({np.sum(train_conf >= 1.0 - 1e-15)/len(train_conf)*100:.1f}%)")

            # Regime distribution in training
            from collections import Counter
            regime_counts = Counter(train_labels)
            print(f"\n  Training regime distribution:")
            for idx in sorted(regime_counts.keys()):
                pct = regime_counts[idx] / len(train_labels) * 100
                print(f"    [{idx}] {regime_names[idx]:<25s}: {regime_counts[idx]:>5d} ({pct:.1f}%)")

        else:
            print("\n  WARNING: Could not load training data cache")

    except Exception as e:
        print(f"\n  ERROR loading training features: {e}")
        import traceback
        traceback.print_exc()

    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)

    for asset, feats in live_features.items():
        scaled = (feats - pkl_mean) / pkl_scale
        n_extreme = int(np.sum(np.abs(scaled) > 3.0))
        scaled_clipped = np.clip(scaled, -4.0, 4.0)
        probs = gmm.predict_proba(scaled_clipped.reshape(1, -1))[0]
        conf = float(np.max(probs))
        regime = regime_names[int(np.argmax(probs))]

        status = "OK" if conf < 0.9999 else "DEGENERATE (ADX fallback)"
        print(f"  {asset}: conf={conf:.10f} regime={regime} extreme_z={n_extreme}/17 -> {status}")


if __name__ == "__main__":
    main()
