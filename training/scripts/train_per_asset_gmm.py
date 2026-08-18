#!/usr/bin/env python3
"""
Stage 1: Train per-asset GMM with BIC search k=3-8.

Uses fold boundaries from configs/split_manifest.json to fit GMM only on
train data (Iron Rule #12: all layers share same splits).

Usage:
    python scripts/train_per_asset_gmm.py                  # All assets
    python scripts/train_per_asset_gmm.py --asset BTC      # Single asset
    python scripts/train_per_asset_gmm.py --fold 1         # Use fold_1 train range

Output:
    models/regime_classifier/{ASSET}/gmm_model.pkl
    models/regime_classifier/{ASSET}/scaler.pkl
    models/regime_classifier/{ASSET}/gmm_config.json
"""

import argparse
import json
import logging
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

_TRAINING_DIR = Path(__file__).resolve().parent.parent   # training/
PROJECT_ROOT = _TRAINING_DIR.parent                      # project root
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("PerAssetGMM")

ASSETS = ["BTC", "ETH", "SOL"]
DATA_DIR = _TRAINING_DIR / "training_data" / "drl_training"
OUTPUT_DIRS = [
    PROJECT_ROOT / "models" / "regime_classifier",
    _TRAINING_DIR / "training_data" / "gmm_models",
]

GMM_FEATURE_COLS = [
    "return_1h", "return_4h", "return_24h", "return_7d",
    "volatility_1h", "volatility_24h", "vol_percentile", "vol_of_vol",
    "momentum_consistency",
    "cross_asset_correlation",
    "fear_index",
    "spread_percentile",
]

GMM_BASE_CONFIG = {
    "covariance_type": "full",
    "reg_covar": 1e-2,
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42,
}

SPREAD_DEFAULTS = {"BTC": 5.0, "ETH": 8.0, "SOL": 12.0}


def compute_gmm_features_for_bar(closes, volumes, rets, i, asset="BTC"):
    """Compute 12 GMM features for a single bar. Mirrors main.py _predict_gmm_regime()."""
    if i < 50:
        return np.full(len(GMM_FEATURE_COLS), np.nan)

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

    if i >= 5:
        recent = rets[max(0, i - 5):i + 1]
        pos = int(np.sum(recent > 0))
        neg = int(np.sum(recent < 0))
        mom_con = (max(pos, neg) / len(recent)) * (1 if pos >= neg else -1)
    else:
        mom_con = 0.0

    cross_corr = 0.87

    if i >= 14:
        deltas = np.diff(closes[i - 14:i + 1])
        gains = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0.0
        losses = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0.0
        rs = gains / (losses + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    else:
        rsi = 50.0
    fear_idx = 100.0 - rsi

    spread = SPREAD_DEFAULTS.get(asset, 8.0)

    return np.array([
        ret_1h, ret_4h, ret_24h, ret_7d,
        vol_1h, vol_24h, vol_pct, vov,
        mom_con, cross_corr, fear_idx, spread,
    ])


def compute_gmm_features_batch(df: pd.DataFrame, asset: str = "BTC") -> np.ndarray:
    """Compute 12 GMM features for all bars."""
    closes = df["close"].values
    volumes = df["volume"].values
    n = len(closes)

    rets = np.zeros(n)
    rets[1:] = np.diff(closes) / np.where(closes[:-1] != 0, closes[:-1], 1.0)

    return np.array([
        compute_gmm_features_for_bar(closes, volumes, rets, i, asset)
        for i in range(n)
    ])


def name_clusters(gmm, scaler, X_scaled, labels):
    """Auto-name clusters based on centroid analysis. Works for k=3-8."""
    k = gmm.n_components
    centroids = gmm.means_

    scores = {}
    for c in range(k):
        mask = labels == c
        scores[c] = {
            "ret_24h": float(centroids[c, 2]),
            "ret_7d": float(centroids[c, 3]),
            "vol_24h": float(centroids[c, 5]),
            "mom_con": float(centroids[c, 8]),
            "vov": float(centroids[c, 7]),
            "count": int(mask.sum()),
        }

    assigned = {}
    remaining = list(range(k))

    # Progressively assign names by characteristic
    panic_score = {c: scores[c]["ret_24h"] - scores[c]["vol_24h"] for c in remaining}
    panic_id = min(panic_score, key=panic_score.get)
    assigned[panic_id] = "PANIC_SELLOFF"
    remaining.remove(panic_id)

    rally_score = {c: scores[c]["ret_24h"] + scores[c]["mom_con"] for c in remaining}
    rally_id = max(rally_score, key=rally_score.get)
    assigned[rally_id] = "MOMENTUM_RALLY"
    remaining.remove(rally_id)

    if remaining:
        vol_score = {c: scores[c]["vol_24h"] + scores[c]["vov"] for c in remaining}
        extreme_id = max(vol_score, key=vol_score.get)
        assigned[extreme_id] = "EXTREME_VOLATILITY"
        remaining.remove(extreme_id)

    if remaining:
        quiet_score = {c: -scores[c]["vol_24h"] - abs(scores[c]["ret_24h"]) for c in remaining}
        quiet_id = max(quiet_score, key=quiet_score.get)
        assigned[quiet_id] = "QUIET_ACCUMULATION"
        remaining.remove(quiet_id)

    if remaining:
        chop_id = max(remaining, key=lambda c: scores[c]["vol_24h"])
        assigned[chop_id] = "VOLATILE_CHOP"
        remaining.remove(chop_id)

    if remaining:
        weak_id = min(remaining, key=lambda c: scores[c]["vol_24h"])
        assigned[weak_id] = "WEAK_CONSOLIDATION"
        remaining.remove(weak_id)

    for c in remaining:
        assigned[c] = f"REGIME_{c}"

    regime_names = [assigned.get(i, f"REGIME_{i}") for i in range(k)]
    regime_mapping = {str(i): assigned.get(i, f"REGIME_{i}") for i in range(k)}

    logger.info("  Cluster naming:")
    for c in range(k):
        s = scores[c]
        logger.info(f"    {c} -> {regime_names[c]:25s} "
                     f"ret_24h={s['ret_24h']:+.3f} vol_24h={s['vol_24h']:+.3f} "
                     f"mom_con={s['mom_con']:+.3f} count={s['count']}")

    return regime_names, regime_mapping


def train_gmm_for_asset(asset: str, train_end: int = None, max_k: int = 8):
    """Train per-asset GMM with BIC search k=3-max_k.

    Args:
        asset: BTC/ETH/SOL
        train_end: Row index for end of training data. If None, use all data.
        max_k: Maximum number of components to search (default 8).
    """
    path = DATA_DIR / f"{asset}_4H_full.parquet"
    if not path.exists():
        logger.error(f"  {asset}: parquet not found at {path}")
        return None

    df = pd.read_parquet(path)
    logger.info(f"\n{'='*60}")
    logger.info(f"  {asset}: {len(df)} total bars")

    # Compute GMM features for all bars
    gmm_features = compute_gmm_features_batch(df, asset)

    # If train_end specified, only fit on train data (Iron Rule #12)
    if train_end is not None:
        logger.info(f"  Using train data only: rows [0:{train_end}] "
                     f"({train_end}/{len(df)} = {train_end/len(df):.0%})")
        gmm_features_fit = gmm_features[:train_end]
    else:
        gmm_features_fit = gmm_features
        logger.info(f"  Using ALL data for GMM fit (no split manifest)")

    # Filter valid features
    valid_mask = ~np.any(np.isnan(gmm_features_fit), axis=1)
    X = gmm_features_fit[valid_mask]
    logger.info(f"  Valid bars: {len(X)} (of {len(gmm_features_fit)}, "
                 f"dropped {len(gmm_features_fit) - len(X)} NaN rows)")

    # Fit scaler (on train data only)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # BIC search k=3 to max_k
    best_k = None
    best_bic = np.inf
    best_gmm = None
    bic_results = []

    for k in range(3, max_k + 1):
        config = dict(GMM_BASE_CONFIG, n_components=k)
        gmm = GaussianMixture(**config, verbose=0)
        gmm.fit(X_scaled)

        labels = gmm.predict(X_scaled)
        counts = Counter(labels)
        min_pct = min(counts.values()) / len(labels) if len(labels) > 0 else 0

        bic = gmm.bic(X_scaled)
        bic_results.append({"k": k, "bic": float(bic), "min_pct": float(min_pct)})

        if min_pct < 0.02:
            logger.info(f"    k={k}: BIC={bic:,.0f} - SKIP (min regime {min_pct:.1%} < 2%)")
            continue

        logger.info(f"    k={k}: BIC={bic:,.0f}, min regime={min_pct:.1%}")
        if bic < best_bic:
            best_bic = bic
            best_k = k
            best_gmm = gmm

    if best_gmm is None:
        logger.warning(f"  No k passed min regime check - falling back to k=6")
        config = dict(GMM_BASE_CONFIG, n_components=6)
        best_gmm = GaussianMixture(**config, verbose=0)
        best_gmm.fit(X_scaled)
        best_k = 6
        best_bic = best_gmm.bic(X_scaled)

    logger.info(f"  Selected k={best_k} (BIC={best_bic:,.0f})")

    # Final predict + name clusters
    labels = best_gmm.predict(X_scaled)
    probs = best_gmm.predict_proba(X_scaled)
    max_probs = probs.max(axis=1)

    regime_names, regime_mapping = name_clusters(best_gmm, scaler, X_scaled, labels)

    # Sanity checks
    conf_mean = float(max_probs.mean())
    conf_std = float(max_probs.std())
    transitions = sum(1 for i in range(1, len(labels)) if labels[i] != labels[i - 1])
    flip_rate = transitions / len(labels) if len(labels) > 0 else 0

    logger.info(f"\n  Sanity checks:")
    logger.info(f"    Confidence: mean={conf_mean:.3f}, std={conf_std:.3f}")
    logger.info(f"    Flip rate: {flip_rate:.3f} ({transitions} transitions)")
    logger.info(f"    Regime distribution:")
    for name, count in Counter([regime_names[l] for l in labels]).most_common():
        logger.info(f"      {name:25s}: {count:5d} ({count/len(labels):.1%})")

    # Post-prediction check: predict on FULL dataset, warn if any regime < 2%
    valid_all = ~np.any(np.isnan(gmm_features), axis=1)
    X_all = gmm_features[valid_all]
    X_all_scaled = scaler.transform(X_all)
    full_labels = best_gmm.predict(X_all_scaled)
    full_pcts = pd.Series(full_labels).value_counts(normalize=True)
    for regime_id in range(best_k):
        pct = full_pcts.get(regime_id, 0.0)
        if pct < 0.02:
            name = regime_names[regime_id] if regime_id < len(regime_names) else f"REGIME_{regime_id}"
            logger.warning(
                f"  POST-PREDICT: {name} (regime {regime_id}) = {pct:.1%} of full data "
                f"({int(pct * len(full_labels))} bars) - below 2% threshold"
            )

    # Save to both output dirs
    for out_base in OUTPUT_DIRS:
        out_dir = out_base / asset
        out_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(best_gmm, out_dir / "gmm_model.pkl")
        joblib.dump(scaler, out_dir / "scaler.pkl")

        config_data = {
            "n_components": best_k,
            "covariance_type": GMM_BASE_CONFIG["covariance_type"],
            "means": best_gmm.means_.tolist(),
            "weights": best_gmm.weights_.tolist(),
            "regime_names": regime_names,
            "regime_mapping": regime_mapping,
            "feature_cols": GMM_FEATURE_COLS,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "training_samples": int(len(labels)),
            "train_end_row": train_end,
            "flip_rate": float(flip_rate),
            "mean_confidence": conf_mean,
            "bic_search": bic_results,
        }
        with open(out_dir / "gmm_config.json", "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)

        logger.info(f"  Saved to {out_dir}/")

    return {
        "asset": asset,
        "k": best_k,
        "bic": best_bic,
        "mean_conf": conf_mean,
        "flip_rate": flip_rate,
        "train_samples": len(labels),
    }


def load_split_manifest(fold: int = 1) -> dict:
    """Load split manifest to get fold_N train_end per asset.

    [P164] This read was pointed at `config/` while `generate_split_manifest.py`
    writes `configs/` (and `config/` holds only optuna_winner.json). The lookup
    therefore ALWAYS missed, returned `{}`, and `train_end` came through as
    None — at which point `train_gmm_for_asset` logs "Using ALL data for GMM
    fit" and fits the regime model on 100% of history. Iron Rule #12 was a
    silent no-op inside the script written to enforce it, for every run this
    script has ever had.

    Now fail-closed. A missing or unusable manifest raises rather than quietly
    degrading to the leaky path; `--no-split` remains the explicit, visible way
    to ask for a full-sample fit.
    """
    manifest_path = PROJECT_ROOT / "configs" / "split_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Split manifest not found at {manifest_path}. Generate it with "
            f"`python training/scripts/generate_split_manifest.py`. Refusing to "
            f"fall back to a full-sample GMM fit, which leaks the validation "
            f"and test windows into the regime features (Iron Rule #12). Pass "
            f"--no-split if a full-sample fit is genuinely what you want."
        )

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    train_ends = {}
    for asset, info in manifest.get("assets", {}).items():
        for f_info in info.get("folds", []):
            if f_info["fold"] == fold:
                train_ends[asset] = f_info["train_end"]
                break

    if not train_ends:
        raise ValueError(
            f"Split manifest {manifest_path} contains no fold_{fold} boundaries "
            f"(assets present: {sorted(manifest.get('assets', {}))}). Refusing "
            f"to fit the GMM on all data — see Iron Rule #12."
        )

    logger.info(f"  Split manifest fold_{fold} train_ends: {train_ends}")
    return train_ends


def main():
    parser = argparse.ArgumentParser(description="Train per-asset GMM")
    parser.add_argument("--asset", type=str, default=None, choices=ASSETS,
                        help="Single asset (default: all)")
    parser.add_argument("--fold", type=int, default=1,
                        help="Which fold's train range to use for GMM fit (default: 1)")
    parser.add_argument("--no-split", action="store_true",
                        help="Use ALL data (ignore split manifest)")
    parser.add_argument("--max-k", type=int, default=8,
                        help="Maximum k for BIC search (default: 8)")
    args = parser.parse_args()

    assets = [args.asset] if args.asset else ASSETS

    # Load fold boundaries
    train_ends = {}
    if not args.no_split:
        train_ends = load_split_manifest(fold=args.fold)

    results = []
    for asset in assets:
        train_end = train_ends.get(asset) if not args.no_split else None
        result = train_gmm_for_asset(asset, train_end=train_end, max_k=args.max_k)
        if result:
            results.append(result)

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Per-Asset GMM Summary")
    logger.info(f"{'='*60}")
    for r in results:
        logger.info(f"  {r['asset']}: k={r['k']}, BIC={r['bic']:,.0f}, "
                     f"conf={r['mean_conf']:.3f}, flip={r['flip_rate']:.3f}, "
                     f"train_n={r['train_samples']}")

    # Verification
    all_ok = True
    for r in results:
        if not (3 <= r["k"] <= 8):
            logger.error(f"  FAIL: {r['asset']} k={r['k']} not in [3,8]")
            all_ok = False
        if r["flip_rate"] < 0.05 or r["flip_rate"] > 0.50:
            logger.warning(f"  WARN: {r['asset']} flip_rate={r['flip_rate']:.3f} outside [0.05, 0.50]")

    if all_ok:
        logger.info(f"\n  ALL CHECKS PASSED")
    else:
        logger.error(f"\n  SOME CHECKS FAILED")


if __name__ == "__main__":
    main()
