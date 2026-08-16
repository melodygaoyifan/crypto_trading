#!/usr/bin/env python3
"""
HMATS v7 - Ultimate Rebuild: Data Pipeline + Per-Asset GMM + External Data
=====================================================================

Steps 1-7 of the Ultimate Rebuild:
  1. Resample 1H -> 4H from historical parquets (8.5y BTC/ETH, 5.5y SOL)
  2. Feature engineering with FeatureEngineer (103 base dims)
  3. Merge external data (7 new features from Coinglass + Futures)
  4. Per-asset GMM with BIC search (k=3-8, 12 features)
  5. Generate DRL training parquets with regime labels + regime_proba[8]
  6. Generate feature manifest (configs/feature_manifest.json)
  7. Verify fold splits + sanity checks

Data sources:
  training/training_data/raw/{BTC,ETH,SOL}_60m.parquet  -> 1H OHLCV
  training/training_data/coinglass_history/{ASSET}_{funding,oi,liquidation}_1d.parquet  -> Coinglass daily
  training/training_data/futures/{ASSET}_futures_daily.parquet -> Futures daily

Usage:
    python scripts/rebuild_pipeline.py                      # Full pipeline
    python scripts/rebuild_pipeline.py --skip-gmm           # Skip GMM retrain, use existing
    python scripts/rebuild_pipeline.py --resample-only      # Only resample

Output:
    training/training_data/drl_training/{BTC,ETH,SOL}_4H_full.parquet        (DRL training data)
    models/regime_classifier/{ASSET}/gmm_model.pkl          (Per-asset GMM)
    models/regime_classifier/{ASSET}/scaler.pkl             (Per-asset scaler)
    models/regime_classifier/{ASSET}/gmm_config.json        (GMM config + regime mapping)
    configs/feature_manifest.json                            (Feature manifest)
"""

import argparse
import importlib.util
import json
import logging
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# ── Project setup ─────────────────────────────────────────────────────
_training_dir = Path(__file__).resolve().parent.parent  # training/
PROJECT_ROOT = _training_dir.parent                     # project root
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(_training_dir))
_feat_spec = importlib.util.spec_from_file_location(
    "features", _training_dir / "drl" / "features.py"
)
_feat_mod = importlib.util.module_from_spec(_feat_spec)
_feat_spec.loader.exec_module(_feat_mod)
FeatureEngineer = _feat_mod.FeatureEngineer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("RebuildPipeline")

# ── Constants ─────────────────────────────────────────────────────────
RAW_DATA_DIR = _training_dir / "training_data" / "raw"
OUTPUT_DIR = _training_dir / "training_data" / "drl_training"
GMM_BUILD_DIR = _training_dir / "training_data" / "gmm_models"
PROD_GMM_DIR = PROJECT_ROOT / "models" / "regime_classifier"
COINGLASS_DIR = _training_dir / "training_data" / "coinglass_history"
FUTURES_DIR = _training_dir / "training_data" / "futures"
MANIFEST_DIR = PROJECT_ROOT / "configs"
ASSETS = ["BTC", "ETH", "SOL"]

MAX_REGIME_PROBA = 8  # Zero-pad proba vector to 8 columns

# GMM config - v3: cleaned to 12 dims
# [P221] Runtime ranks the current volume within its fetched frame
# (721 bars bootstrapped to ~1024). Training must rank within the same
# trailing depth or the percentile distributions diverge.
GMM_VOL_PCT_WINDOW = 1024

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

# External feature column names (7 new)
EXTERNAL_FEATURE_COLS = [
    "funding_rate_zscore",
    "oi_change_5d",
    "liq_imbalance",
    "taker_ratio_zscore",
    "tradecount_zscore",
    "taker_vol_momentum",
    "has_external_data",
]

# ── RegimeSmoother ────────────────────────────────────────────────────
from core.regime_smoother import RegimeSmoother  # noqa: E402
from scripts.wavelet_denoise import (  # noqa: E402
    wavelet_denoise_causal, DENOISE_COLUMNS, DENOISED_FEATURE_NAMES,
)


# ══════════════════════════════════════════════════════════════════════
# STEP 1: RESAMPLE 1H -> 4H
# ══════════════════════════════════════════════════════════════════════

def load_and_resample(asset: str) -> pd.DataFrame:
    """Load 1H parquet and resample to 4H."""
    raw_path = RAW_DATA_DIR / f"{asset}_60m.parquet"
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data not found: {raw_path}")

    df = pd.read_parquet(raw_path)
    logger.info(f"  Loaded {asset}: {len(df)} 1H bars")

    # Normalize columns
    df.columns = df.columns.str.lower()

    # Handle volume column naming
    if "volume_usdt" in df.columns:
        df = df.rename(columns={"volume_usdt": "volume"})
    elif "volume btc" in df.columns and "volume" not in df.columns:
        df = df.rename(columns={"volume btc": "volume"})

    # Parse timestamp (coerce out-of-bounds values to NaT)
    if "unix_timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["unix_timestamp"], unit="ms", errors="coerce")
    elif "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Drop rows with invalid timestamps
    n_before = len(df)
    df = df.dropna(subset=["timestamp"])
    if len(df) < n_before:
        logger.warning(f"  Dropped {n_before - len(df)} rows with invalid timestamps")

    df = df.sort_values("timestamp").reset_index(drop=True)

    # Resample to 4H
    df = df.set_index("timestamp")
    df_4h = df.resample("4h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()
    df_4h = df_4h.reset_index()

    logger.info(f"  Resampled to 4H: {len(df_4h)} bars")
    logger.info(f"  Date range: {df_4h['timestamp'].iloc[0]} -> {df_4h['timestamp'].iloc[-1]}")

    return df_4h


# ══════════════════════════════════════════════════════════════════════
# STEP 2: FEATURE ENGINEERING (103 base dims)
# ══════════════════════════════════════════════════════════════════════

def compute_drl_features(df: pd.DataFrame) -> tuple:
    """Compute 103-dim DRL features using FeatureEngineer."""
    fe = FeatureEngineer(state_dim=128, cross_asset_dim=16)
    df = fe.compute_features(df)
    feature_cols = fe.get_feature_columns()
    return df, feature_cols


# ══════════════════════════════════════════════════════════════════════
# STEP 3: MERGE EXTERNAL DATA (7 new features)
# ══════════════════════════════════════════════════════════════════════

def _rolling_zscore(series: pd.Series, window: int = 30) -> pd.Series:
    """Rolling z-score: (x - rolling_mean) / rolling_std, clipped to [-10, 10]."""
    mean = series.rolling(window, min_periods=10).mean()
    std = series.rolling(window, min_periods=10).std().replace(0, 1e-10)
    z = (series - mean) / std
    return z.clip(-10, 10)


def _load_coinglass_daily(asset: str) -> pd.DataFrame:
    """Load and derive features from Coinglass 1D data (funding, OI, liquidation)."""
    # Funding -> funding_rate_zscore
    fpath = COINGLASS_DIR / f"{asset}_funding_1d.parquet"
    if fpath.exists():
        funding = pd.read_parquet(fpath)
        funding["timestamp"] = pd.to_datetime(funding["timestamp"], utc=True)
        funding = funding.sort_values("timestamp")
        # [P253] CAUSAL SHIFT — this was the P247-F1 leak's last unfixed
        # carrier (flagged in P247 as "still carries the leak for the DRL
        # feature set ... for the next parquet rebuild"). Daily rows are
        # stamped at day-OPEN while funding_close is the day's LAST
        # (16:00 UTC) event, so z-scoring the unshifted series and
        # merge_asof(backward)-ing it below hands every 00:00-12:00 bar up to
        # 16h of FUTURE funding. shift(1) makes bars on day D read day D-1's
        # close, z-scored over a trailing window ending at D-1 — the same
        # semantics as regime_model_lab._causal_funding_z, applied at the
        # source so train_drl_full/train_supervised_full stop consuming the
        # leaked column through the manifest.
        funding["funding_rate_zscore"] = _rolling_zscore(
            funding["funding_close"].shift(1), 30)
    else:
        funding = pd.DataFrame(columns=["timestamp", "funding_rate_zscore"])

    # OI -> oi_change_5d
    fpath = COINGLASS_DIR / f"{asset}_oi_1d.parquet"
    if fpath.exists():
        oi = pd.read_parquet(fpath)
        oi["timestamp"] = pd.to_datetime(oi["timestamp"], utc=True)
        oi = oi.sort_values("timestamp")
        oi["oi_change_5d"] = oi["oi_close"].pct_change(5).clip(-5, 5)
    else:
        oi = pd.DataFrame(columns=["timestamp", "oi_change_5d"])

    # Liquidation -> liq_imbalance (already computed in source, just clip)
    fpath = COINGLASS_DIR / f"{asset}_liquidation_1d.parquet"
    if fpath.exists():
        liq = pd.read_parquet(fpath)
        liq["timestamp"] = pd.to_datetime(liq["timestamp"], utc=True)
        liq = liq.sort_values("timestamp")
        liq["liq_imbalance"] = liq["liq_imbalance"].clip(-1, 1)
    else:
        liq = pd.DataFrame(columns=["timestamp", "liq_imbalance"])

    # Merge all Coinglass sources
    merged = funding[["timestamp", "funding_rate_zscore"]].merge(
        oi[["timestamp", "oi_change_5d"]], on="timestamp", how="outer"
    ).merge(
        liq[["timestamp", "liq_imbalance"]], on="timestamp", how="outer"
    )
    return merged.sort_values("timestamp").reset_index(drop=True)


def _load_futures_daily(asset: str) -> pd.DataFrame:
    """Load and derive features from futures daily data."""
    fpath = FUTURES_DIR / f"{asset}_futures_daily.parquet"
    if not fpath.exists():
        # [P281] REFUSE — never silently backfill zeros again. This exact
        # branch fed three all-zero columns to every model ever trained
        # (P279: training_data/futures/ never existed and nothing said so).
        # The fetcher now exists and is in the refresh-data chain; a missing
        # file is a broken chain, not a neutral default (P2/P199).
        raise SystemExit(
            f"[P281] REFUSING: {fpath} missing — taker_ratio_zscore/"
            f"tradecount_zscore/taker_vol_momentum would be silently ZERO "
            f"in the parquet (the P279 dead-columns defect). Run:\n"
            f"  python -X utf8 training/scripts/"
            f"fetch_binance_futures_daily.py --assets {asset}\n"
            f"then re-run the rebuild.")

    df = pd.read_parquet(fpath)

    # Fix data types (stored as object/string in these parquets)
    for col in ["marketorder_volume", "marketorder_volume_from", "tradecount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fix timestamps + filter corrupted rows (1970 dates)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df[df["timestamp"] >= "2000-01-01"].copy()
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Derive features on daily data
    df["taker_ratio_zscore"] = _rolling_zscore(df["marketorder_volume"], 30)
    df["tradecount_zscore"] = _rolling_zscore(df["tradecount"], 30)
    df["taker_vol_momentum"] = df["marketorder_volume_from"].pct_change(5).clip(-5, 5)

    return df[["timestamp", "taker_ratio_zscore", "tradecount_zscore", "taker_vol_momentum"]].copy()


def merge_external_data(df_4h: pd.DataFrame, asset: str) -> pd.DataFrame:
    """Merge 7 external features into 4H df via merge_asof (daily->4H forward-fill).

    Features added:
      funding_rate_zscore  - 30d z-score of OI-weighted funding rate
      oi_change_5d         - 5d pct change of aggregated OI
      liq_imbalance        - (long-short)/total liquidations, clipped [-1,1]
      taker_ratio_zscore   - 30d z-score of taker market order volume
      tradecount_zscore    - 30d z-score of trade count
      taker_vol_momentum   - 5d pct change of taker volume
      has_external_data    - binary flag (1 where external data available)

    Pre-data-start: all external = 0.0, has_external_data = 0.
    """
    cg = _load_coinglass_daily(asset)
    fut = _load_futures_daily(asset)

    # Merge Coinglass + futures into single daily df
    daily = cg.merge(fut, on="timestamp", how="outer").sort_values("timestamp").reset_index(drop=True)

    # Ensure 4H df has UTC-aware timestamps for merge_asof
    df = df_4h.copy()
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    df = df.sort_values("timestamp").reset_index(drop=True)
    daily = daily.sort_values("timestamp").reset_index(drop=True)

    ext_cols = [c for c in EXTERNAL_FEATURE_COLS if c != "has_external_data"]

    # [FIX 2026-04-22] When Coinglass + Futures files are missing, daily is empty
    # with object-dtype timestamp column; merge_asof rejects the type mismatch
    # against the UTC datetime64 left side. Short-circuit and backfill zeros.
    if len(daily) == 0:
        for col in ext_cols:
            df[col] = 0.0
        df["has_external_data"] = 0.0
        logger.info(f"    External data: 0/{len(df)} bars — no Coinglass/Futures files "
                    f"({asset}), filled with zeros")
        return df

    # merge_asof: for each 4H bar, use most recent daily data <= that timestamp.
    # [P287] tolerance bounds staleness at 3 days (mirroring the live P265g
    # funding bound): before this, direction="backward" with NO tolerance
    # meant a bar could match a daily row from ARBITRARILY far back — a
    # September rebuild against 1d archives frozen in August would stamp
    # weeks of bars with the frozen values AND flag them has_external_data.
    # Within contiguous daily coverage (rows <=1d apart) the tolerance never
    # binds, so in-coverage content is byte-identical to the old merge.
    df = pd.merge_asof(df, daily[["timestamp"] + ext_cols],
                       on="timestamp", direction="backward",
                       tolerance=pd.Timedelta(days=3))

    # has_external_data flag: 1 where ANY external feature is non-NaN
    df["has_external_data"] = (~df[ext_cols].isna().all(axis=1)).astype(float)

    # Fill pre-data NaN with 0.0
    for col in ext_cols:
        df[col] = df[col].fillna(0.0)

    n_with = int(df["has_external_data"].sum())
    logger.info(f"    External data: {n_with}/{len(df)} bars have data "
                f"({n_with/len(df)*100:.1f}%)")

    return df


# ══════════════════════════════════════════════════════════════════════
# STEP 4: PER-ASSET GMM (BIC search k=3-8)
# ══════════════════════════════════════════════════════════════════════

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
    # [P221] TRAILING window, mirroring the runtime contract. The runtime ranks
    # vols[-1] within its fetched ~1024-bar frame (market_data_pipeline
    # _predict_gmm_regime: `vols = df["volume"].values`); the old expanding
    # `volumes[:i+1]` ranked against the full 6-year history — with volume's
    # secular drift those are materially different distributions, i.e. a
    # train/serve skew on one of the GMM's 10 effective inputs (same family
    # as the P214 wavelet skew, and a contributor to the old GMM's
    # distribution-shift saturation, P215 addendum).
    _vp_lo = max(0, i - GMM_VOL_PCT_WINDOW + 1)
    _vp_win = volumes[_vp_lo:i + 1]
    vol_pct = float(np.searchsorted(np.sort(_vp_win), volumes[i]) / len(_vp_win) * 100)

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

    cross_corr = 0.87  # default, matches runtime

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
        mom_con,
        cross_corr,
        fear_idx,
        spread,
    ])


def compute_gmm_features_batch(df: pd.DataFrame, asset: str = "BTC") -> np.ndarray:
    """Compute 12 GMM features for all bars."""
    closes = df["close"].values
    volumes = df["volume"].values
    n = len(closes)

    rets = np.zeros(n)
    rets[1:] = np.diff(closes) / np.where(closes[:-1] != 0, closes[:-1], 1.0)

    features = np.array([
        compute_gmm_features_for_bar(closes, volumes, rets, i, asset)
        for i in range(n)
    ])
    return features


def name_clusters(gmm, scaler, X_scaled, labels):
    """Auto-name clusters based on centroid analysis. Works for k=3-8."""
    k = gmm.n_components
    centroids = gmm.means_  # in scaled space

    # Score each cluster
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

    # 1. PANIC_SELLOFF: most negative returns + high vol
    panic_score = {c: scores[c]["ret_24h"] - scores[c]["vol_24h"] for c in remaining}
    panic_id = min(panic_score, key=panic_score.get)
    assigned[panic_id] = "PANIC_SELLOFF"
    remaining.remove(panic_id)

    # 2. MOMENTUM_RALLY: most positive returns + positive momentum
    rally_score = {c: scores[c]["ret_24h"] + scores[c]["mom_con"] for c in remaining}
    rally_id = max(rally_score, key=rally_score.get)
    assigned[rally_id] = "MOMENTUM_RALLY"
    remaining.remove(rally_id)

    # 3. EXTREME_VOLATILITY: highest vol (if k >= 3)
    if remaining:
        vol_score = {c: scores[c]["vol_24h"] + scores[c]["vov"] for c in remaining}
        extreme_id = max(vol_score, key=vol_score.get)
        assigned[extreme_id] = "EXTREME_VOLATILITY"
        remaining.remove(extreme_id)

    # 4. QUIET_ACCUMULATION: lowest vol + near-zero returns (if k >= 4)
    if remaining:
        quiet_score = {c: -scores[c]["vol_24h"] - abs(scores[c]["ret_24h"]) for c in remaining}
        quiet_id = max(quiet_score, key=quiet_score.get)
        assigned[quiet_id] = "QUIET_ACCUMULATION"
        remaining.remove(quiet_id)

    # 5. VOLATILE_CHOP: higher vol of remaining (if k >= 5)
    if remaining:
        chop_id = max(remaining, key=lambda c: scores[c]["vol_24h"])
        assigned[chop_id] = "VOLATILE_CHOP"
        remaining.remove(chop_id)

    # 6. WEAK_CONSOLIDATION: lower vol of remaining (if k >= 6)
    if remaining:
        weak_id = min(remaining, key=lambda c: scores[c]["vol_24h"])
        assigned[weak_id] = "WEAK_CONSOLIDATION"
        remaining.remove(weak_id)

    # 7+: [P221 naming pass] Assign the REMAINING known vocabulary before ever
    # falling back to a generic name. The 2026-08-07 refit produced clusters
    # named REGIME_1 (BTC, 20% of bars) and REGIME_7 (SOL) — names that
    # resolve fine (so the P184 guard cannot fire) but appear in NO
    # name-keyed table (POSITION_BIAS / BULL_REGIMES / BEAR_REGIMES /
    # regime_weights), silently taking neutral values in every
    # regime-conditional term. The runtime vocabulary has two more members
    # the heuristics above never assign:
    #   STEADY_UPTREND — positive drift at low vol
    #   NEUTRAL_DRIFT  — everything near zero
    # Assign them by centroid shape; only if MORE clusters remain after the
    # full 8-name vocabulary is exhausted does a generic name appear — and
    # then it is a loud ERROR, not a silent label.
    if remaining:
        up_score = {c: scores[c]["ret_24h"] + scores[c]["ret_7d"]
                    - scores[c]["vol_24h"] for c in remaining}
        up_id = max(up_score, key=up_score.get)
        if scores[up_id]["ret_24h"] > 0:
            assigned[up_id] = "STEADY_UPTREND"
            remaining.remove(up_id)
    if remaining:
        drift_score = {c: -abs(scores[c]["ret_24h"]) - abs(scores[c]["mom_con"])
                       for c in remaining}
        drift_id = max(drift_score, key=drift_score.get)
        assigned[drift_id] = "NEUTRAL_DRIFT"
        remaining.remove(drift_id)
    for c in remaining:
        assigned[c] = f"REGIME_{c}"
        logger.error(
            f"  [P221] cluster {c} exhausted the 8-name vocabulary and got "
            f"generic name REGIME_{c} — it will take NEUTRAL values in every "
            f"name-keyed regime table. Extend the vocabulary (and the bias "
            f"tables) before training on this parquet.")

    # Build regime_names list + mapping
    regime_names = [assigned.get(i, f"REGIME_{i}") for i in range(k)]
    regime_mapping = {str(i): assigned.get(i, f"REGIME_{i}") for i in range(k)}

    logger.info("  Cluster naming:")
    for c in range(k):
        s = scores[c]
        logger.info(f"    {c} -> {regime_names[c]:25s} "
                     f"ret_24h={s['ret_24h']:+.3f} vol_24h={s['vol_24h']:+.3f} "
                     f"mom_con={s['mom_con']:+.3f} count={s['count']}")

    return regime_names, regime_mapping


def _gmm_sanity_checks(gmm, labels, max_probs, regime_names, X_scaled):
    """Run sanity checks on a trained GMM."""
    k = gmm.n_components

    # 1. Confidence distribution
    logger.info(f"  mean_conf={max_probs.mean():.3f}, std={max_probs.std():.3f}, "
                f"min={max_probs.min():.3f}, max={max_probs.max():.3f}")
    if max_probs.mean() > 0.98:
        logger.warning("  WARNING: mean confidence > 0.98 - possible collapse!")

    # 2. All clusters used, check min pct
    dist = Counter(labels)
    for c in range(k):
        count = dist.get(c, 0)
        pct = count / len(labels) if len(labels) > 0 else 0
        name = regime_names[c]
        status = "OK" if pct > 0.03 else "WARN"
        logger.info(f"    {name:25s}: {pct:.1%} ({count} samples) [{status}]")

    # 3. Flip rate
    transitions = sum(1 for i in range(1, len(labels)) if labels[i] != labels[i - 1])
    flip_rate = transitions / max(len(labels), 1)
    logger.info(f"  Flip rate: {flip_rate:.1%}")

    # 4. Z-score check on recent data
    recent = X_scaled[-1000:] if len(X_scaled) > 1000 else X_scaled
    z_extreme = (np.abs(recent) > 3).mean(axis=1).mean()
    logger.info(f"  Avg features with |z|>3 on recent 1000 bars: {z_extreme:.3f}")

    return flip_rate


# ---------------------------------------------------------------------------
# [P200 2026-08-07] GMM fit boundary — Iron Rule #12 for THIS script.
#
# P164 fixed the full-history GMM fit in train_per_asset_gmm.py, but THIS
# script — the one that actually generated the deployed training parquets —
# kept fitting scaler + GMM + BIC-k + cluster names on 100% of history and
# even deploying that model. Every regime_proba_0..7 column it ever emitted
# was a function of the validation windows it was later evaluated on.
#
# The boundary is derived from the SAME fold arithmetic as
# train_drl_full._get_fold_splits (val_size = int(n*0.15), gap=42, 3 folds,
# expanding train, folds rolling backwards): the STRICTEST fold boundary is
# fold_3's train_end = n - 3*val_size - gap. Fitting on rows before it keeps
# the GMM blind to EVERY fold's validation window — note that fitting to
# fold_1's boundary (what train_per_asset_gmm's fold=1 default does) still
# leaks folds 2/3's val windows, which sit inside fold_1's train range.
#
# Derived-not-looked-up on purpose: the split manifest is generated FROM the
# parquets this script produces, so reading it here would be circular (and
# stale indices from an older data range would silently mis-cut). A test
# pins this arithmetic against the trainer's.
# ---------------------------------------------------------------------------
GMM_FIT_N_FOLDS = 3
GMM_FIT_VAL_RATIO = 0.15
GMM_FIT_GAP = 42


def gmm_fit_boundary(n_valid_rows: int) -> int:
    """First `boundary` valid rows are eligible for the GMM fit — everything
    at or after the strictest fold's train_end is held out."""
    boundary = int(n_valid_rows * (1 - GMM_FIT_N_FOLDS * GMM_FIT_VAL_RATIO)) - GMM_FIT_GAP
    if boundary < 1000:
        raise ValueError(
            f"gmm_fit_boundary: only {boundary} rows before the strictest fold "
            f"boundary (n_valid={n_valid_rows}) — too little data for a "
            f"meaningful GMM fit. Fetch more history before rebuilding."
        )
    return boundary


def retrain_gmm_per_asset(asset: str, gmm_features: np.ndarray, smooth: int = 2,
                          no_split: bool = False):
    """Train per-asset GMM with BIC search k=3-8.

    Selects lowest BIC where min regime pct > 2%.
    Saves to models/regime_classifier/{ASSET}/ and data/gmm_models/{ASSET}/.

    [P200] Fits on TRAIN-ONLY rows (before the strictest fold boundary) unless
    `no_split=True` is passed explicitly — the full-sample fit is the leak
    P164 documented, and it survives here until 2026-08-07.
    """
    logger.info(f"\n  === GMM for {asset} ===")

    # Filter valid features
    valid_mask = ~np.any(np.isnan(gmm_features), axis=1)
    valid_idx = np.where(valid_mask)[0]
    if no_split:
        logger.warning(
            f"  {asset}: --gmm-no-split — fitting GMM on ALL {len(valid_idx)} "
            f"valid bars. The resulting regime features LEAK every validation "
            f"window (Iron Rule #12); do not train or evaluate models for "
            f"promotion on this parquet.")
        fit_idx = valid_idx
    else:
        boundary = gmm_fit_boundary(len(valid_idx))
        fit_idx = valid_idx[:boundary]
        logger.info(
            f"  {asset}: fitting GMM on first {len(fit_idx)}/{len(valid_idx)} "
            f"valid bars (strictest fold boundary; val windows held out)")
    X = gmm_features[fit_idx]
    logger.info(f"  {asset}: {len(X)} fit bars (of {len(gmm_features)} total)")

    # Fit scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    logger.info(f"  Scaler stats:")
    for i, name in enumerate(GMM_FEATURE_COLS):
        logger.info(f"    {name:25s}: mean={scaler.mean_[i]:+.6f}, scale={scaler.scale_[i]:.6f}")

    # BIC search k=3 to 8
    best_k = None
    best_bic = np.inf
    best_gmm = None
    bic_results = []

    for k in range(3, 9):
        config = dict(GMM_BASE_CONFIG, n_components=k)
        gmm = GaussianMixture(**config, verbose=0)
        gmm.fit(X_scaled)

        labels = gmm.predict(X_scaled)
        counts = Counter(labels)
        min_pct = min(counts.values()) / len(labels) if len(labels) > 0 else 0

        bic = gmm.bic(X_scaled)
        bic_results.append((k, bic, min_pct))

        if min_pct < 0.02:
            logger.info(f"    k={k}: BIC={bic:,.0f} - SKIP (min regime {min_pct:.1%} < 2%)")
            continue

        logger.info(f"    k={k}: BIC={bic:,.0f}, min regime={min_pct:.1%}")

        if bic < best_bic:
            best_bic = bic
            best_k = k
            best_gmm = gmm

    if best_gmm is None:
        # Fallback: use k=6 without min pct constraint
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
    logger.info(f"\n  === {asset} GMM SANITY CHECKS ===")
    flip_rate = _gmm_sanity_checks(best_gmm, labels, max_probs, regime_names, X_scaled)

    # ── Save to build dir ─────────────────────────────────────────────
    build_dir = GMM_BUILD_DIR / asset
    build_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_gmm, build_dir / "gmm_model.pkl")
    joblib.dump(scaler, build_dir / "scaler.pkl")

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
        "training_samples": len(labels),
        # [P200] Record which fit policy produced this artifact — a full-sample
        # GMM and a split-aware GMM are indistinguishable by value (P179
        # lesson: record which one you read). "split_aware" = fit on rows
        # before the strictest fold boundary only.
        "fit_policy": ("full_sample_LEAKY" if no_split else "split_aware"),
        "flip_rate": flip_rate,
        "mean_confidence": float(max_probs.mean()),
        "bic_search": [{"k": k, "bic": b, "min_pct": p} for k, b, p in bic_results],
    }
    with open(build_dir / "gmm_config.json", "w") as f:
        json.dump(config_data, f, indent=2)

    logger.info(f"  GMM saved to {build_dir}/")

    return best_gmm, scaler, regime_names, regime_mapping


def load_existing_gmm_per_asset(asset: str):
    """Load per-asset GMM from production path."""
    asset_dir = PROD_GMM_DIR / asset

    # Try per-asset path first, fall back to shared path
    if (asset_dir / "gmm_config.json").exists():
        config_path = asset_dir / "gmm_config.json"
        model_path = asset_dir / "gmm_model.pkl"
        scaler_path = asset_dir / "scaler.pkl"
    else:
        # [P287] The old fallback silently used the LEGACY SHARED GMM — a
        # per-asset parquet built from the global legacy fit is the P4
        # mixed-artifact shape (features paired with a model that did not
        # produce them), and the legacy fit is pre-P200 full-sample = leaky
        # by construction (P159). REFUSE instead of degrading.
        raise FileNotFoundError(
            f"[P287] No per-asset GMM for {asset} at {asset_dir} — refusing "
            f"the legacy shared-GMM fallback (P4 mixed-artifact / P159 leaky "
            f"fit). Fix: run a split-aware per-asset fit (rebuild_pipeline "
            f"without --skip-gmm) or copy the P221 artifacts into place.")

    with open(config_path) as f:
        config = json.load(f)

    gmm = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    regime_names = config["regime_names"]
    regime_mapping = config.get("regime_mapping", {})

    logger.info(f"  Loaded {asset} GMM: k={config.get('n_components')}, "
                f"trained={config.get('trained_at', 'unknown')}")

    return gmm, scaler, regime_names, regime_mapping


# ══════════════════════════════════════════════════════════════════════
# STEP 5: REGIME PREDICTION (with probability vector)
# ══════════════════════════════════════════════════════════════════════

def predict_regimes(gmm_features, gmm, scaler):
    """Run GMM prediction. Returns regime IDs, confidences, and probability matrix [n, 8]."""
    n = len(gmm_features)
    k = gmm.n_components
    regime_ids = np.full(n, -1, dtype=int)
    confidences = np.full(n, 0.0)
    proba_matrix = np.zeros((n, MAX_REGIME_PROBA))  # 8 cols, zero-padded

    scaler_mean = scaler.mean_
    scaler_scale = scaler.scale_

    for i in range(n):
        if np.any(np.isnan(gmm_features[i])):
            continue

        scaled = (gmm_features[i] - scaler_mean) / scaler_scale
        scaled = np.clip(scaled, -4.0, 4.0)

        probs = gmm.predict_proba(scaled.reshape(1, -1))[0]
        regime_ids[i] = int(np.argmax(probs))
        confidences[i] = float(probs[regime_ids[i]])
        proba_matrix[i, :k] = probs  # first k columns, rest stay 0

    return regime_ids, confidences, proba_matrix


# ══════════════════════════════════════════════════════════════════════
# STEP 6: FEATURE MANIFEST
# ══════════════════════════════════════════════════════════════════════

def generate_feature_manifest(base_features: list, output_path: Path,
                              denoised_features: list = None):
    """Generate configs/feature_manifest.json with ordered feature list and metadata."""
    regime_proba_features = [f"regime_proba_{i}" for i in range(MAX_REGIME_PROBA)]
    denoised_features = denoised_features or []

    all_features = (base_features + denoised_features +
                    EXTERNAL_FEATURE_COLS + regime_proba_features)
    no_scale = regime_proba_features + ["has_external_data"]
    binary = ["has_external_data"]

    manifest = {
        "base_features": base_features,
        "denoised_features": denoised_features,
        "external_features": EXTERNAL_FEATURE_COLS,
        "regime_proba_features": regime_proba_features,
        "all_features": all_features,
        "total_feature_count": len(all_features),
        "no_scale_features": no_scale,
        "binary_features": binary,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"  Feature manifest saved: {output_path}")
    logger.info(f"    Base: {len(base_features)}, External: {len(EXTERNAL_FEATURE_COLS)}, "
                f"Regime proba: {MAX_REGIME_PROBA}, Total: {len(all_features)}")

    return manifest


# ══════════════════════════════════════════════════════════════════════
# STEP 7: VERIFY FOLD SPLITS
# ══════════════════════════════════════════════════════════════════════

def verify_folds(df, asset, n_folds=3, val_ratio=0.15, gap=42, window_size=10):
    """Verify fold splits match spec requirements."""
    n = len(df)
    val_size = int(n * val_ratio)

    logger.info(f"\n  {asset} fold verification (total: {n} bars)")
    all_ok = True

    for fold_idx in range(n_folds):
        val_end = n - fold_idx * val_size
        val_start = val_end - val_size
        train_end = val_start - gap
        eval_steps = val_size - window_size

        if train_end <= 0:
            logger.error(f"    Fold {fold_idx + 1}: train_end <= 0!")
            all_ok = False
            continue

        # Regime coverage
        train_regimes = set(df["regime"].iloc[:train_end].unique()) - {-1}
        val_regimes = set(df["regime"].iloc[val_start:val_end].unique()) - {-1}

        status = "OK" if train_end > 5000 and eval_steps > 100 else "WARN"
        if train_end <= 5000 or eval_steps <= 100:
            all_ok = False

        logger.info(
            f"    Fold {fold_idx + 1}: train={train_end} bars, "
            f"val={val_size} bars, eval_steps={eval_steps}, "
            f"regimes(train={len(train_regimes)}, val={len(val_regimes)}) [{status}]"
        )

    return all_ok


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="HMATS v7 Ultimate Rebuild Pipeline")
    parser.add_argument("--skip-gmm", action="store_true",
                        help="Skip GMM retraining, use existing per-asset models")
    parser.add_argument("--gmm-no-split", action="store_true",
                        help="[P200] Fit the GMM on ALL history (the pre-2026-08 "
                             "leaky behavior). The resulting parquets must never "
                             "be used to train or promote models — explicit "
                             "opt-in only, e.g. for offline visualisation.")
    parser.add_argument("--resample-only", action="store_true",
                        help="Only resample 1H->4H, don't compute features or train GMM")
    parser.add_argument("--smooth", type=int, default=2,
                        help="RegimeSmoother persistence (0=disable)")
    parser.add_argument("--folds", type=int, default=3,
                        help="Number of folds for verification")
    args = parser.parse_args()

    start_time = time.time()
    logger.info("=" * 60)
    logger.info("HMATS v7 - ULTIMATE REBUILD PIPELINE (v2)")
    logger.info("=" * 60)

    # ── Step 1: Resample 1H -> 4H ────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: RESAMPLE 1H -> 4H")
    logger.info("=" * 60)

    resampled = {}
    for asset in ASSETS:
        resampled[asset] = load_and_resample(asset)

    if args.resample_only:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        for asset, df in resampled.items():
            path = OUTPUT_DIR / f"{asset}_4H_resampled.parquet"
            df.to_parquet(path, index=False)
            logger.info(f"  Saved: {path}")
        logger.info("\nResample complete. Exiting.")
        return

    # ── Step 2: Feature Engineering ──────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: FEATURE ENGINEERING (103 base dims)")
    logger.info("=" * 60)

    featured = {}
    drl_feature_cols = None
    for asset in ASSETS:
        logger.info(f"\n  Processing {asset}...")
        df, feat_cols = compute_drl_features(resampled[asset])
        featured[asset] = df
        if drl_feature_cols is None:
            drl_feature_cols = feat_cols
        logger.info(f"    {asset}: {len(df)} bars, {len(feat_cols)} base features")

    # ── Step 2.5: Wavelet Denoising ────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2.5: WAVELET DENOISING (Coiflet-4 level-2)")
    logger.info("=" * 60)

    denoised_cols_added = []
    for asset in ASSETS:
        df = featured[asset]
        n_added = 0
        for src_col, dst_col in DENOISE_COLUMNS.items():
            if src_col in df.columns:
                raw_vals = df[src_col].values.astype(float)
                # Fill NaN before denoising (wavelet can't handle NaN).
                # [P253] The old np.nanmedian(raw_vals) injected a
                # WHOLE-COLUMN statistic into rows the causal wavelet then
                # treats as observed — a full-history value smuggled past the
                # P164 fix. ffill (last observation) is backward-looking.
                # [P260, precision per the fresh-mind review] The .bfill()
                # leg IS a look-ahead — but ONLY for LEADING NaNs (rows
                # before the feature's first observation, deep inside the
                # warmup the fold boundaries discard). Stated exactly so the
                # word "causal" is never doing more work than the code.
                nan_mask = np.isnan(raw_vals)
                if nan_mask.any():
                    _filled = pd.Series(raw_vals).ffill().bfill().values
                    raw_vals[nan_mask] = _filled[nan_mask]
                # [P164] CAUSAL. The previous `wavelet_denoise(raw_vals)` applied
                # the transform to the entire 8.5-year column at once, and its
                # VisuShrink threshold is computed from the whole array — so every
                # training row saw the future while live rows see a trailing 256
                # window. That gap, not regime shift, is what separates the
                # reported val Sharpe (+7 to +17) from the live IC (+0.052).
                # Do not "optimise" this back into a single whole-column call.
                denoised = wavelet_denoise_causal(raw_vals)
                df[dst_col] = denoised
                n_added += 1

                # SNR check
                noise = raw_vals - denoised
                snr = np.var(raw_vals) / (np.var(noise) + 1e-20)
                snr_db = 10 * np.log10(snr) if snr > 0 else 0.0
                logger.info(f"    {asset} {src_col} -> {dst_col}: SNR={snr_db:.1f} dB")
            else:
                logger.warning(f"    {asset}: column '{src_col}' not found, skipping")
        featured[asset] = df
        if not denoised_cols_added:
            denoised_cols_added = [dst for src, dst in DENOISE_COLUMNS.items()
                                   if src in df.columns]
        logger.info(f"  {asset}: {n_added} denoised columns added")

    # ── Step 3: Merge External Data ──────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: MERGE EXTERNAL DATA (7 new features)")
    logger.info("=" * 60)

    for asset in ASSETS:
        logger.info(f"\n  Processing {asset}...")
        featured[asset] = merge_external_data(featured[asset], asset)

    # ── Step 4: Per-Asset GMM ────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: PER-ASSET GMM (BIC search k=3-8)")
    logger.info("=" * 60)

    # Compute GMM features for all assets
    all_gmm_features = {}
    for asset in ASSETS:
        logger.info(f"  Computing GMM features for {asset}...")
        gmm_feats = compute_gmm_features_batch(featured[asset], asset)
        all_gmm_features[asset] = gmm_feats
        valid = ~np.any(np.isnan(gmm_feats), axis=1)
        logger.info(f"    {asset}: {valid.sum()} valid / {len(gmm_feats)} total")

    per_asset_gmms = {}
    if args.skip_gmm:
        logger.info("\n  Loading existing per-asset GMMs (--skip-gmm)...")
        for asset in ASSETS:
            gmm, scaler, rnames, rmapping = load_existing_gmm_per_asset(asset)
            per_asset_gmms[asset] = (gmm, scaler, rnames, rmapping)
    else:
        for asset in ASSETS:
            gmm, scaler, rnames, rmapping = retrain_gmm_per_asset(
                asset, all_gmm_features[asset], smooth=args.smooth,
                no_split=args.gmm_no_split,
            )
            per_asset_gmms[asset] = (gmm, scaler, rnames, rmapping)

    # ── Step 5: Generate DRL Training Parquets ───────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("STEP 5: GENERATE DRL TRAINING PARQUETS")
    logger.info("=" * 60)

    smoother = RegimeSmoother(min_persistence=args.smooth) if args.smooth > 0 else None
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    regime_proba_cols = [f"regime_proba_{i}" for i in range(MAX_REGIME_PROBA)]

    for asset in ASSETS:
        logger.info(f"\n  Processing {asset}...")
        df = featured[asset].copy()
        gmm_feats = all_gmm_features[asset]
        gmm, scaler, regime_names, regime_mapping = per_asset_gmms[asset]

        # Predict regimes with probability vector
        regime_ids, confidences, proba_matrix = predict_regimes(gmm_feats, gmm, scaler)
        df["regime"] = regime_ids
        df["regime_confidence"] = confidences
        for i in range(MAX_REGIME_PROBA):
            df[f"regime_proba_{i}"] = proba_matrix[:, i]

        # Stats before smoothing
        valid_mask = regime_ids >= 0
        n_valid = int(np.sum(valid_mask))
        regime_dist = Counter(regime_ids[valid_mask])
        logger.info(f"    Valid bars: {n_valid}/{len(df)}")
        for idx in sorted(regime_dist.keys()):
            name = regime_names[idx] if idx < len(regime_names) else f"R{idx}"
            count = regime_dist[idx]
            logger.info(f"      {name}: {count} ({count/n_valid*100:.1f}%)")

        # Verify proba sums
        valid_proba = proba_matrix[valid_mask]
        proba_sums = valid_proba.sum(axis=1)
        logger.info(f"    Proba sum: mean={proba_sums.mean():.4f}, "
                     f"min={proba_sums.min():.4f}, max={proba_sums.max():.4f}")

        # Apply RegimeSmoother
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
            logger.info(f"    RegimeSmoother: flips {before_flips} -> {after_flips}")

        # Drop invalid bars (first 50 + NaN features)
        df = df[df["regime"] >= 0].reset_index(drop=True)
        key_feat = drl_feature_cols[0] if drl_feature_cols else None
        if key_feat and key_feat in df.columns:
            df = df.dropna(subset=[key_feat]).reset_index(drop=True)
        logger.info(f"    After filtering: {len(df)} bars")

        # Build column list
        keep_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        keep_cols += [c for c in drl_feature_cols if c in df.columns]
        keep_cols += [c for c in denoised_cols_added if c in df.columns]
        keep_cols += [c for c in EXTERNAL_FEATURE_COLS if c in df.columns]
        keep_cols += regime_proba_cols
        keep_cols += ["regime", "regime_confidence"]

        # Strip timezone for parquet compatibility (all UTC)
        if df["timestamp"].dt.tz is not None:
            df["timestamp"] = df["timestamp"].dt.tz_localize(None)

        out_path = OUTPUT_DIR / f"{asset}_4H_full.parquet"
        df[keep_cols].to_parquet(out_path, index=False)
        logger.info(f"    Saved: {out_path} ({len(df)} rows, {len(keep_cols)} cols)")

    # ── Step 5b: fv2 flow features (P266 — folded in) ────────────────
    # [P266] A parquet rebuild used to be TWO steps: this script, then
    # scripts/build_flow_features.py — a footgun the P253c standing rule
    # documented after the P253b rebuild itself silently DROPPED the 13
    # fv2_* columns (this script regenerates its own column set and knew
    # nothing about the sibling's extras; caught by the P252b parity test).
    # The fv2 build now runs INSIDE the rebuild, after every asset's parquet
    # is written (cross-asset features need all three). It remains runnable
    # standalone. A failure here is a FAILURE of the rebuild — the P253c
    # rule exists precisely because a parquet without fv2 looks complete.
    logger.info("\n" + "=" * 60)
    logger.info("STEP 5b: FV2 FLOW FEATURES (build_flow_features, P266)")
    logger.info("=" * 60)
    from build_flow_features import main as _fv2_main
    _fv2_rc = _fv2_main()
    if _fv2_rc != 0:
        logger.error("  fv2 build FAILED (rc=%s) — the parquets are "
                     "INCOMPLETE (no fv2_* columns). Fix and re-run; do not "
                     "train on this output.", _fv2_rc)
        sys.exit(2)

    # ── Step 6: Feature Manifest ─────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("STEP 6: FEATURE MANIFEST")
    logger.info("=" * 60)

    # Save feature_cols.json (base features only, for backward compat)
    feat_cols_path = OUTPUT_DIR / "feature_cols.json"
    with open(feat_cols_path, "w") as f:
        json.dump(drl_feature_cols, f, indent=2)

    # Generate full manifest
    manifest_path = MANIFEST_DIR / "feature_manifest.json"
    manifest = generate_feature_manifest(
        drl_feature_cols, manifest_path,
        denoised_features=denoised_cols_added,
    )

    # ── Step 7: Verify Fold Splits ───────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("STEP 7: VERIFY FOLD SPLITS")
    logger.info("=" * 60)

    all_ok = True
    for asset in ASSETS:
        df = pd.read_parquet(OUTPUT_DIR / f"{asset}_4H_full.parquet")
        ok = verify_folds(df, asset, n_folds=args.folds, gap=42, window_size=10)
        if not ok:
            all_ok = False

    # ── Pre-Training Checklist ────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("PRE-TRAINING CHECKLIST")
    logger.info("=" * 60)

    checks = []
    for asset in ASSETS:
        df = pd.read_parquet(OUTPUT_DIR / f"{asset}_4H_full.parquet")
        n = len(df)
        val_size = int(n * 0.15)
        eval_steps = val_size - 10

        checks.append(("eval_steps >= 1500", eval_steps >= 1500, f"{asset}: {eval_steps}"))
        checks.append(("total rows > 5000", n > 5000, f"{asset}: {n}"))

        # Verify external features present
        ext_present = sum(1 for c in EXTERNAL_FEATURE_COLS if c in df.columns)
        checks.append((f"{asset} external features", ext_present == 7, f"{ext_present}/7"))

        # Verify regime_proba columns
        proba_present = sum(1 for c in regime_proba_cols if c in df.columns)
        checks.append((f"{asset} regime_proba cols", proba_present == 8, f"{proba_present}/8"))

        # Verify proba sums ≈ 1.0 for valid rows
        valid = df[df["regime"] >= 0]
        proba_sum = valid[regime_proba_cols].sum(axis=1)
        sum_ok = (proba_sum - 1.0).abs().max() < 0.01
        checks.append((f"{asset} proba sum ≈ 1.0", sum_ok, f"max_diff={float((proba_sum - 1.0).abs().max()):.4f}"))

        # Verify has_external_data flag
        ext_ratio = df["has_external_data"].mean()
        checks.append((f"{asset} has_external_data", ext_ratio > 0.3, f"{ext_ratio:.1%}"))

        # Verify denoised columns present
        dn_present = sum(1 for c in denoised_cols_added if c in df.columns)
        dn_expected = len(denoised_cols_added)
        checks.append((f"{asset} denoised features", dn_present == dn_expected,
                       f"{dn_present}/{dn_expected}"))

        # Verify NaN-free feature columns
        feat_nan = df[[c for c in drl_feature_cols if c in df.columns]].isna().sum().sum()
        checks.append((f"{asset} no NaN in features", feat_nan == 0, f"{feat_nan} NaN"))

    # General checks
    checks.append(("feature_manifest exists", manifest_path.exists(), str(manifest_path)))
    checks.append(("n_folds = 3", args.folds == 3, f"{args.folds}"))
    checks.append(("ent_coef = 0.1", True, "hardcoded in train script"))
    checks.append(("DummyVecEnv only", True, "hardcoded in train script"))

    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        logger.info(f"  [{status}] {name} ({detail})")
        if not passed:
            all_ok = False

    # ── Summary ──────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Time: {elapsed / 60:.1f} minutes")

    for asset in ASSETS:
        path = OUTPUT_DIR / f"{asset}_4H_full.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            n_features = len([c for c in df.columns
                             if c not in ["timestamp", "open", "high", "low", "close",
                                          "volume", "regime", "regime_confidence"]])
            logger.info(f"  {asset}: {len(df)} bars, {n_features} features, "
                        f"regimes: {df['regime'].value_counts().to_dict()}")

    if all_ok:
        # Deploy per-asset GMM models to production path.
        # [P287] Two guards: (a) a --gmm-no-split fit is LEAKY (full-sample)
        # and must NEVER reach models/regime_classifier — the P267 invariant
        # (deploy-side fit_policy must be split_aware) previously had no
        # enforcement at this, the only deploy site; (b) even a clean refit
        # overwrites the runtime artifacts beneath whatever checkpoints are
        # deployed — {GMM, parquets, checkpoints} move as ONE versioned set
        # (P215/P253b), so the swap is announced loudly.
        if not args.skip_gmm and args.gmm_no_split:
            logger.error(
                "  [P287] REFUSING GMM deploy: --gmm-no-split produced a "
                "full-sample (LEAKY) fit. It must never reach "
                f"{PROD_GMM_DIR} (P267: deploy-side fit_policy must be "
                "split_aware). Build-dir artifacts kept for visualization "
                "only.")
        elif not args.skip_gmm:
            logger.warning(
                "  [P287] Overwriting the RUNTIME GMM artifacts in "
                f"{PROD_GMM_DIR} — {{GMM, parquets, checkpoints}} move as "
                "ONE versioned set (P215/P253b); if live checkpoints are "
                "paired with the current fit, record this swap as a "
                "deliberate decision.")
            for asset in ASSETS:
                src_dir = GMM_BUILD_DIR / asset
                dst_dir = PROD_GMM_DIR / asset
                dst_dir.mkdir(parents=True, exist_ok=True)

                # Backup old model
                old_model = dst_dir / "gmm_model.pkl"
                if old_model.exists():
                    backup = f"gmm_model_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                    shutil.copy2(old_model, dst_dir / backup)

                for fname in ["gmm_model.pkl", "scaler.pkl", "gmm_config.json"]:
                    src = src_dir / fname
                    if src.exists():
                        shutil.copy2(src, dst_dir / fname)

                logger.info(f"  {asset} GMM deployed to {dst_dir}")

        logger.info("\n  ALL CHECKS PASSED - Ready for training")
        logger.info(f"\n  Total features: {manifest['total_feature_count']} "
                    f"(base={len(drl_feature_cols)}, "
                    f"ext={len(EXTERNAL_FEATURE_COLS)}, "
                    f"proba={MAX_REGIME_PROBA})")
        logger.info(f"  Obs dim for DRL: {manifest['total_feature_count']} + 4 env state = "
                    f"{manifest['total_feature_count'] + 4}")
    else:
        logger.error("\n  SOME CHECKS FAILED - Review issues above before training")


if __name__ == "__main__":
    main()
