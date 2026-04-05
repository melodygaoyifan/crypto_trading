"""
================================================================================
HMATS v6.5.2 - GMM Regime Classifier Retraining
================================================================================

Retrains the 6-component GMM with a NEW StandardScaler fitted on fresh Kraken
OHLCV data.  Feature computation is IDENTICAL to main.py:_predict_gmm_regime()
so the scaler never drifts from runtime expectations.

Usage:
    python scripts/retrain_gmm.py                  # Full pipeline
    python scripts/retrain_gmm.py --fetch-only      # Just download data
    python scripts/retrain_gmm.py --skip-fetch       # Use cached data
    python scripts/retrain_gmm.py --dry-run          # Train but don't deploy

Outputs:
    models/regime_classifier/gmm_model_NEW.pkl
    models/regime_classifier/gmm_config_NEW.json
    models/regime_classifier/agreement_report.json
================================================================================
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("retrain_gmm")

# ---------------------------------------------------------------------------
# Constants (match runtime exactly)
# ---------------------------------------------------------------------------
ASSETS = ["BTC/USD", "ETH/USD", "SOL/USD"]
ASSET_NAMES = {"BTC/USD": "BTC", "ETH/USD": "ETH", "SOL/USD": "SOL"}
TIMEFRAME = "4h"
BARS_NEEDED = 2000  # ~333 days of 4H data per asset
GMM_DIR = PROJECT_ROOT / "models" / "regime_classifier"
DATA_CACHE_DIR = PROJECT_ROOT / "data" / "gmm_training_cache"

FEATURE_COLS = [
    "return_1h", "return_4h", "return_24h", "return_7d",
    "volatility_1h", "volatility_24h", "vol_percentile", "vol_of_vol",
    "momentum_4h", "momentum_24h", "momentum_consistency",
    "cross_asset_correlation", "correlation_regime",
    "fear_index", "sentiment_momentum",
    "spread_percentile", "depth_change",
]

REGIME_NAMES_TEMPLATE = [
    "MOMENTUM_RALLY",
    "QUIET_ACCUMULATION",
    "PANIC_SELLOFF",
    "VOLATILE_CHOP",
    "EXTREME_VOLATILITY",
    "WEAK_CONSOLIDATION",
]


# ============================================================================
# STEP 1: DATA FETCHING
# ============================================================================

def fetch_ohlcv_kraken(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    """
    Fetch historical OHLCV from Kraken via CCXT with pagination.
    Returns DataFrame with columns: timestamp, open, high, low, close, volume.
    """
    import ccxt

    exchange = ccxt.kraken({
        "enableRateLimit": True,
        "timeout": 30000,
    })

    all_bars = []
    since = None

    # Calculate starting timestamp (go back enough history)
    end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
    # 4h = 4*3600*1000 ms
    bar_ms = 4 * 3600 * 1000
    since = end_time - (limit * bar_ms)

    logger.info(f"  Fetching {symbol} {timeframe} from {datetime.fromtimestamp(since/1000, tz=timezone.utc).strftime('%Y-%m-%d')} ...")

    while len(all_bars) < limit:
        try:
            batch = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=720)
        except Exception as e:
            logger.warning(f"  Fetch error for {symbol}: {e}, retrying in 5s...")
            time.sleep(5)
            continue

        if not batch:
            break

        all_bars.extend(batch)
        logger.info(f"  {symbol}: {len(all_bars)} bars fetched")

        # Move forward past the last bar
        since = batch[-1][0] + bar_ms

        if len(batch) < 720:
            break  # No more data available

        time.sleep(exchange.rateLimit / 1000)  # Respect rate limits

    # Deduplicate by timestamp
    seen = set()
    unique_bars = []
    for bar in all_bars:
        if bar[0] not in seen:
            seen.add(bar[0])
            unique_bars.append(bar)

    df = pd.DataFrame(unique_bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Trim to requested limit
    if len(df) > limit:
        df = df.tail(limit).reset_index(drop=True)

    logger.info(f"  {symbol}: {len(df)} unique bars from {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
    return df


def fetch_all_assets(limit: int = BARS_NEEDED) -> dict:
    """Fetch OHLCV for all 3 assets, return {asset_name: DataFrame}."""
    data = {}
    for symbol in ASSETS:
        name = ASSET_NAMES[symbol]
        logger.info(f"Fetching {name} ({symbol})...")
        df = fetch_ohlcv_kraken(symbol, TIMEFRAME, limit)
        data[name] = df
        logger.info(f"  {name}: {len(df)} bars OK")
    return data


def save_data_cache(data: dict):
    """Save fetched data to parquet for reuse."""
    DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in data.items():
        path = DATA_CACHE_DIR / f"{name}_4h.parquet"
        df.to_parquet(path, index=False)
        logger.info(f"  Cached {name} -> {path}")


def load_data_cache() -> dict:
    """Load previously cached data."""
    data = {}
    for symbol in ASSETS:
        name = ASSET_NAMES[symbol]
        path = DATA_CACHE_DIR / f"{name}_4h.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Cache not found: {path}. Run with --fetch-only first.")
        df = pd.read_parquet(path)
        data[name] = df
        logger.info(f"  Loaded {name}: {len(df)} bars from cache")
    return data


# ============================================================================
# STEP 2: FEATURE ENGINEERING
# ============================================================================
# CRITICAL: This must be IDENTICAL to main.py _predict_gmm_regime() lines 3609-3659

def compute_features_for_bar(
    closes: np.ndarray,
    volumes: np.ndarray,
    idx: int,
    rsi_val: float = 50.0,
    cross_corr: float = 0.65,
    spread_bps: float = 10.0,
) -> np.ndarray:
    """
    Compute the 17-feature vector for a single bar, matching runtime exactly.

    Args:
        closes: Full close price array up to and including bar at idx
        volumes: Full volume array up to and including bar at idx
        idx: Current bar index (we use data up to closes[:idx+1])
        rsi_val: RSI(14) value for this bar
        cross_corr: Cross-asset correlation estimate
        spread_bps: Spread in basis points

    Returns:
        17-element numpy array matching FEATURE_COLS order
    """
    c = closes[:idx + 1]
    v = volumes[:idx + 1]
    n = len(c)

    if n < 50:
        return None

    # Returns
    rets = np.diff(c) / np.where(c[:-1] != 0, c[:-1], 1.0)

    ret_4h = float(rets[-1]) if len(rets) > 0 else 0.0
    ret_1h = ret_4h / 4.0
    ret_24h = float((c[-1] - c[-7]) / c[-7]) if n >= 7 and c[-7] > 0 else 0.0
    ret_7d = float((c[-1] - c[-43]) / c[-43]) if n >= 43 and c[-43] > 0 else 0.0

    # Volatility
    vol_1h = float(np.std(rets[-2:])) if len(rets) >= 2 else 0.02
    vol_24h = float(np.std(rets[-6:])) if len(rets) >= 6 else 0.02
    vol_pct = float(np.searchsorted(np.sort(v), v[-1]) / n * 100) if n > 0 else 50.0

    # Vol-of-vol
    if len(rets) >= 30:
        rolling_v = [float(np.std(rets[max(0, i - 6):i])) for i in range(6, len(rets))]
        vov = float(np.std(rolling_v[-20:])) if len(rolling_v) >= 20 else 0.0
    else:
        vov = 0.0

    # Momentum
    mom_4h = ret_4h
    mom_24h = ret_24h

    if len(rets) >= 6:
        recent = rets[-6:]
        pos = int(np.sum(recent > 0))
        neg = int(np.sum(recent < 0))
        mom_con = (max(pos, neg) / 6.0) * (1 if pos >= neg else -1)
    else:
        mom_con = 0.0

    # Cross-asset correlation (passed in)
    corr_regime = 1.0 if cross_corr > 0.7 else (0.0 if cross_corr < 0.3 else 0.5)

    # Sentiment / market structure
    fear_idx = 100.0 - rsi_val
    sent_mom = 0.0  # Always 0 at runtime
    spread_pct = min(spread_bps / 20.0 * 100, 100.0)
    depth_chg = 0.0  # Always 0 at runtime

    return np.array([
        ret_1h, ret_4h, ret_24h, ret_7d,
        vol_1h, vol_24h, vol_pct, vov,
        mom_4h, mom_24h, mom_con,
        cross_corr, corr_regime,
        fear_idx, sent_mom,
        spread_pct, depth_chg,
    ], dtype=np.float64)


def compute_rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """Compute RSI for full close array. Returns array same length as closes."""
    rsi = np.full(len(closes), 50.0)
    if len(closes) < period + 1:
        return rsi

    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rsi[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i + 1] = 100.0 - 100.0 / (1.0 + rs)

    return rsi


def compute_cross_asset_correlation(
    btc_closes: np.ndarray,
    eth_closes: np.ndarray,
    sol_closes: np.ndarray,
    window: int = 30,
) -> np.ndarray:
    """
    Compute rolling cross-asset correlation (avg of BTC-ETH, BTC-SOL, ETH-SOL).
    Returns array of length = min(len(btc), len(eth), len(sol)).
    """
    min_len = min(len(btc_closes), len(eth_closes), len(sol_closes))
    btc_r = np.diff(btc_closes[-min_len:]) / np.where(btc_closes[-min_len:-1] != 0, btc_closes[-min_len:-1], 1.0)
    eth_r = np.diff(eth_closes[-min_len:]) / np.where(eth_closes[-min_len:-1] != 0, eth_closes[-min_len:-1], 1.0)
    sol_r = np.diff(sol_closes[-min_len:]) / np.where(sol_closes[-min_len:-1] != 0, sol_closes[-min_len:-1], 1.0)

    corr = np.full(len(btc_r), 0.65)
    for i in range(window, len(btc_r)):
        b = btc_r[i - window:i]
        e = eth_r[i - window:i]
        s = sol_r[i - window:i]
        c_be = np.corrcoef(b, e)[0, 1] if np.std(b) > 0 and np.std(e) > 0 else 0.65
        c_bs = np.corrcoef(b, s)[0, 1] if np.std(b) > 0 and np.std(s) > 0 else 0.65
        c_es = np.corrcoef(e, s)[0, 1] if np.std(e) > 0 and np.std(s) > 0 else 0.65
        avg = (c_be + c_bs + c_es) / 3.0
        corr[i] = float(np.clip(avg, -1.0, 1.0))

    return corr


def build_feature_matrix(data: dict) -> tuple:
    """
    Build full feature matrix from all 3 assets' OHLCV data.

    Returns:
        (features: np.ndarray of shape [N, 17],
         timestamps: list of datetime,
         asset_labels: list of str)
    """
    # Align timestamps across assets
    btc_df = data["BTC"]
    eth_df = data["ETH"]
    sol_df = data["SOL"]

    btc_closes = btc_df["close"].values
    eth_closes = eth_df["close"].values
    sol_closes = sol_df["close"].values

    # Compute cross-asset correlation
    logger.info("Computing cross-asset correlations...")
    cross_corr_arr = compute_cross_asset_correlation(btc_closes, eth_closes, sol_closes)

    all_features = []
    all_timestamps = []
    all_assets = []

    for asset_name, df in data.items():
        closes = df["close"].values
        volumes = df["volume"].values
        n = len(closes)

        # Compute RSI for all bars
        rsi_arr = compute_rsi(closes)

        logger.info(f"Computing features for {asset_name} ({n} bars)...")

        # Start from bar 50 (need history for indicators)
        for idx in range(50, n):
            # Cross-asset correlation: align by position from end
            # cross_corr_arr has length = min_len-1 (from diff)
            min_len = min(len(btc_closes), len(eth_closes), len(sol_closes))
            corr_idx = idx - (n - min_len + 1)
            if 0 <= corr_idx < len(cross_corr_arr):
                xcorr = cross_corr_arr[corr_idx]
            else:
                xcorr = 0.65

            # Approximate spread_bps: typical for each asset on Kraken
            spread_map = {"BTC": 5.0, "ETH": 8.0, "SOL": 12.0}
            spread_bps = spread_map.get(asset_name, 10.0)

            feat = compute_features_for_bar(
                closes, volumes, idx,
                rsi_val=rsi_arr[idx],
                cross_corr=xcorr,
                spread_bps=spread_bps,
            )

            if feat is not None:
                all_features.append(feat)
                all_timestamps.append(df["timestamp"].iloc[idx])
                all_assets.append(asset_name)

    features = np.array(all_features)
    logger.info(f"Feature matrix: {features.shape[0]} samples × {features.shape[1]} features")

    # Sanity: check for NaN/inf
    nan_count = np.sum(~np.isfinite(features))
    if nan_count > 0:
        logger.warning(f"Found {nan_count} NaN/inf values in features, replacing with 0")
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return features, all_timestamps, all_assets


# ============================================================================
# STEP 3: GMM TRAINING
# ============================================================================

def train_gmm(features: np.ndarray, n_components: int = 6) -> tuple:
    """
    Train GMM on feature matrix.

    Returns:
        (gmm_model, scaler, labels, probs)
    """
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler

    logger.info(f"Training GMM: {features.shape[0]} samples, {n_components} components...")

    # Fit scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # Log scaler stats for key features
    logger.info("Scaler statistics (runtime-matching feature computation):")
    for i, name in enumerate(FEATURE_COLS):
        logger.info(f"  {name:25s}: mean={scaler.mean_[i]:+.6f}, scale={scaler.scale_[i]:.6f}")

    # Fit GMM
    # reg_covar=1e-2 adds regularization to covariance matrices, producing
    # softer posteriors (prevents near-1.0 confidence for well-separated clusters).
    # This is important because 2 features (sentiment_momentum, depth_change) are
    # always 0, and correlation_regime has near-zero variance - without regularization
    # these low-variance dimensions cause near-degenerate posteriors.
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        n_init=10,
        max_iter=300,
        random_state=42,
        reg_covar=1e-2,
        verbose=1,
    )
    gmm.fit(X_scaled)

    logger.info(f"GMM converged in {gmm.n_iter_} iterations")
    logger.info(f"Log-likelihood: {gmm.score(X_scaled):.4f}")

    # Get predictions
    labels = gmm.predict(X_scaled)
    probs = gmm.predict_proba(X_scaled)

    return gmm, scaler, labels, probs


def map_clusters_to_regimes(
    gmm,
    scaler,
    features: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """
    Map GMM cluster indices to interpretable regime names.

    Strategy: use cluster centroids in ORIGINAL feature space to determine:
    - High positive returns + high momentum -> MOMENTUM_RALLY
    - High negative returns + high vol -> PANIC_SELLOFF
    - High vol + low directional momentum -> VOLATILE_CHOP
    - Extreme vol -> EXTREME_VOLATILITY
    - Low vol + positive bias -> QUIET_ACCUMULATION
    - Low vol + slight negative bias -> WEAK_CONSOLIDATION
    """
    n_clusters = gmm.n_components

    # Compute cluster statistics in original feature space
    cluster_stats = {}
    for c in range(n_clusters):
        mask = labels == c
        if mask.sum() == 0:
            cluster_stats[c] = {k: 0.0 for k in FEATURE_COLS}
            continue
        means = features[mask].mean(axis=0)
        cluster_stats[c] = {FEATURE_COLS[i]: means[i] for i in range(len(FEATURE_COLS))}

    # Score each cluster on key dimensions
    scores = {}
    for c, stats in cluster_stats.items():
        ret = stats["return_24h"]
        vol = stats["volatility_24h"]
        mom = stats["momentum_24h"]
        mom_con = stats["momentum_consistency"]
        vov = stats["vol_of_vol"]

        scores[c] = {
            "returns": ret,
            "volatility": vol,
            "momentum": mom,
            "consistency": mom_con,
            "vol_of_vol": vov,
            "count": int((labels == c).sum()),
        }

    logger.info("\nCluster analysis:")
    for c in range(n_clusters):
        s = scores[c]
        logger.info(
            f"  Cluster {c}: ret24h={s['returns']:+.5f} vol24h={s['volatility']:.5f} "
            f"mom={s['momentum']:+.5f} con={s['consistency']:+.3f} "
            f"vov={s['vol_of_vol']:.5f} n={s['count']}"
        )

    # Assign regimes by ranking clusters on each dimension
    available_names = list(REGIME_NAMES_TEMPLATE)
    mapping = {}

    # 1. EXTREME_VOLATILITY: highest vol_of_vol
    sorted_by_vov = sorted(range(n_clusters), key=lambda c: scores[c]["vol_of_vol"], reverse=True)
    extreme_vol_cluster = sorted_by_vov[0]
    if scores[extreme_vol_cluster]["vol_of_vol"] > scores[sorted_by_vov[-1]]["vol_of_vol"] * 1.5:
        mapping[extreme_vol_cluster] = "EXTREME_VOLATILITY"
        available_names.remove("EXTREME_VOLATILITY")
    else:
        # If vol_of_vol doesn't differentiate well, use raw volatility
        sorted_by_vol = sorted(range(n_clusters), key=lambda c: scores[c]["volatility"], reverse=True)
        mapping[sorted_by_vol[0]] = "EXTREME_VOLATILITY"
        available_names.remove("EXTREME_VOLATILITY")

    # 2. PANIC_SELLOFF: most negative returns + high vol
    remaining = [c for c in range(n_clusters) if c not in mapping]
    panic_score = {c: scores[c]["returns"] - scores[c]["volatility"] for c in remaining}
    panic_cluster = min(remaining, key=lambda c: panic_score[c])
    if scores[panic_cluster]["returns"] < 0:
        mapping[panic_cluster] = "PANIC_SELLOFF"
        available_names.remove("PANIC_SELLOFF")

    # 3. MOMENTUM_RALLY: most positive returns + high momentum
    remaining = [c for c in range(n_clusters) if c not in mapping]
    rally_score = {c: scores[c]["returns"] + scores[c]["momentum"] for c in remaining}
    rally_cluster = max(remaining, key=lambda c: rally_score[c])
    if scores[rally_cluster]["returns"] > 0:
        mapping[rally_cluster] = "MOMENTUM_RALLY"
        available_names.remove("MOMENTUM_RALLY")

    # 4. VOLATILE_CHOP: high vol + low directional consistency
    remaining = [c for c in range(n_clusters) if c not in mapping]
    if remaining and "VOLATILE_CHOP" in available_names:
        chop_score = {c: scores[c]["volatility"] - abs(scores[c]["consistency"]) for c in remaining}
        chop_cluster = max(remaining, key=lambda c: chop_score[c])
        mapping[chop_cluster] = "VOLATILE_CHOP"
        available_names.remove("VOLATILE_CHOP")

    # 5. QUIET_ACCUMULATION: low vol + neutral/positive bias
    remaining = [c for c in range(n_clusters) if c not in mapping]
    if remaining and "QUIET_ACCUMULATION" in available_names:
        quiet_score = {c: -scores[c]["volatility"] + scores[c]["returns"] * 0.5 for c in remaining}
        quiet_cluster = max(remaining, key=lambda c: quiet_score[c])
        mapping[quiet_cluster] = "QUIET_ACCUMULATION"
        available_names.remove("QUIET_ACCUMULATION")

    # 6. WEAK_CONSOLIDATION: whatever remains
    remaining = [c for c in range(n_clusters) if c not in mapping]
    for c in remaining:
        if available_names:
            mapping[c] = available_names.pop(0)
        else:
            mapping[c] = f"REGIME_{c}"

    logger.info("\nRegime mapping:")
    for c in range(n_clusters):
        name = mapping.get(c, f"UNMAPPED_{c}")
        pct = scores[c]["count"] / len(labels) * 100
        logger.info(f"  Cluster {c} -> {name} ({pct:.1f}%)")

    return mapping


# ============================================================================
# STEP 4: SANITY CHECKS
# ============================================================================

def run_sanity_checks(gmm, scaler, features: np.ndarray, labels: np.ndarray) -> bool:
    """Run all sanity checks from the prompt. Returns True if all pass."""
    X_scaled = scaler.transform(features)
    probs = gmm.predict_proba(X_scaled)
    max_probs = probs.max(axis=1)

    checks_passed = True

    # Check 1: No confidence=1.0 collapse
    mean_conf = max_probs.mean()
    logger.info(f"\n{'='*60}")
    logger.info(f"SANITY CHECKS")
    logger.info(f"{'='*60}")

    logger.info(f"\n1. Confidence distribution:")
    logger.info(f"   Mean max_prob:  {mean_conf:.4f}")
    logger.info(f"   Std max_prob:   {max_probs.std():.4f}")
    logger.info(f"   Min max_prob:   {max_probs.min():.4f}")
    logger.info(f"   Max max_prob:   {max_probs.max():.4f}")
    logger.info(f"   % with conf>0.95: {(max_probs > 0.95).mean():.1%}")

    if mean_conf > 0.99:
        logger.error("   FAIL: Mean confidence > 0.99 - collapsing to single cluster!")
        checks_passed = False
    elif mean_conf > 0.95:
        logger.warning(f"   WARN: Mean confidence {mean_conf:.4f} is high but may be acceptable")
        logger.warning(f"   (Old broken model was 1.0000 with 0 variance)")
        # Not a hard fail - check variance instead
        if max_probs.std() < 0.02:
            logger.error("   FAIL: High mean + negligible variance = degenerate")
            checks_passed = False
    else:
        logger.info("   PASS")

    if max_probs.std() < 0.05:
        logger.error("   FAIL: Confidence std < 0.05 - no variance!")
        checks_passed = False
    else:
        logger.info("   PASS: Confidence has healthy variance")

    # Check 2: All clusters used
    unique_labels = np.unique(labels)
    logger.info(f"\n2. Cluster usage: {len(unique_labels)} of {gmm.n_components} clusters used")
    dist = Counter(labels.tolist())
    for label in sorted(dist.keys()):
        pct = dist[label] / len(labels)
        status = "WARN (< 5%)" if pct < 0.05 else "OK"
        logger.info(f"   Cluster {label}: {pct:.1%} ({dist[label]} samples) [{status}]")

    if len(unique_labels) < 3:
        logger.error(f"   FAIL: Only {len(unique_labels)} clusters used!")
        checks_passed = False
    else:
        logger.info("   PASS")

    # Check 3: No cluster too rare
    for label, count in dist.items():
        pct = count / len(labels)
        if pct < 0.01:  # Less than 1% is problematic
            logger.warning(f"   WARNING: Cluster {label} has only {pct:.1%} - very rare")

    # Check 4: Temporal coherence (flip rate)
    transitions = sum(1 for i in range(1, len(labels)) if labels[i] != labels[i - 1])
    flip_rate = transitions / len(labels)
    logger.info(f"\n3. Temporal coherence:")
    logger.info(f"   Transitions: {transitions}")
    logger.info(f"   Flip rate:   {flip_rate:.1%}")

    if flip_rate > 0.4:
        logger.warning(f"   WARNING: Flip rate {flip_rate:.1%} is high - regimes may be unstable")
    else:
        logger.info("   PASS")

    # Check 5: Feature z-score distribution on recent data
    recent = X_scaled[-100:]  # Last ~100 bars
    extreme_per_sample = (np.abs(recent) > 3).sum(axis=1)
    avg_extreme = extreme_per_sample.mean()
    logger.info(f"\n4. Feature z-scores on recent data (last 100 bars):")
    logger.info(f"   Avg features with |z|>3: {avg_extreme:.2f}")

    if avg_extreme > 3:
        logger.warning(f"   WARNING: Still many extreme z-scores on recent data")
    else:
        logger.info("   PASS")

    logger.info(f"\n{'='*60}")
    logger.info(f"OVERALL: {'ALL CHECKS PASSED' if checks_passed else 'SOME CHECKS FAILED'}")
    logger.info(f"{'='*60}")

    return checks_passed


# ============================================================================
# STEP 5: REGIME AGREEMENT BACKTEST
# ============================================================================

def regime_agreement_backtest(
    old_config_path: Path,
    old_model_path: Path,
    new_gmm,
    new_scaler,
    new_mapping: dict,
    features: np.ndarray,
) -> dict:
    """Compare old vs new GMM regime classifications."""

    logger.info(f"\n{'='*60}")
    logger.info(f"REGIME AGREEMENT BACKTEST")
    logger.info(f"{'='*60}")

    # Load old model + scaler
    with open(old_config_path) as f:
        old_config = json.load(f)

    old_gmm = joblib.load(old_model_path)
    old_scaler_mean = np.array(old_config["scaler_mean"])
    old_scaler_scale = np.array(old_config["scaler_scale"])
    old_regime_names = old_config["regime_names"]

    # Old model predictions
    old_scaled = (features - old_scaler_mean) / old_scaler_scale
    old_scaled = np.clip(old_scaled, -4.0, 4.0)  # Same clipping as runtime
    old_labels_raw = old_gmm.predict(old_scaled)
    old_probs = old_gmm.predict_proba(old_scaled)
    old_regimes = [old_regime_names[l] for l in old_labels_raw]

    # New model predictions
    new_scaled = new_scaler.transform(features)
    new_labels_raw = new_gmm.predict(new_scaled)
    new_probs = new_gmm.predict_proba(new_scaled)
    new_regimes = [new_mapping[l] for l in new_labels_raw]

    # --- Metric 1: Raw agreement ---
    agreement = sum(o == n for o, n in zip(old_regimes, new_regimes)) / len(old_regimes)

    # --- Metric 2: Agreement by regime ---
    regime_agreement = {}
    for regime in set(old_regimes):
        indices = [i for i, r in enumerate(old_regimes) if r == regime]
        if indices:
            matches = sum(1 for i in indices if new_regimes[i] == regime)
            regime_agreement[regime] = matches / len(indices)

    # --- Metric 3: Transition agreement ---
    old_transitions = [i for i in range(1, len(old_regimes)) if old_regimes[i] != old_regimes[i - 1]]
    new_transitions = [i for i in range(1, len(new_regimes)) if new_regimes[i] != new_regimes[i - 1]]

    matched = 0
    for ot in old_transitions:
        if any(abs(ot - nt) <= 2 for nt in new_transitions):
            matched += 1
    transition_agreement = matched / max(len(old_transitions), 1)

    # --- Metric 4: Confidence comparison ---
    old_conf_mean = old_probs.max(axis=1).mean()
    new_conf_mean = new_probs.max(axis=1).mean()

    # --- Metric 5: Distribution ---
    old_dist = Counter(old_regimes)
    new_dist = Counter(new_regimes)

    # Print report
    logger.info(f"\nOverall Agreement: {agreement:.1%}")

    logger.info(f"\nPer-Regime Agreement:")
    for regime, agr in sorted(regime_agreement.items()):
        logger.info(f"  {regime:25s}: {agr:.1%}")

    logger.info(f"\nTransition Agreement: {transition_agreement:.1%}")
    logger.info(f"\nOld Model Avg Confidence: {old_conf_mean:.3f}")
    logger.info(f"New Model Avg Confidence: {new_conf_mean:.3f}")

    logger.info(f"\nRegime Distribution:")
    logger.info(f"  {'Regime':25s} {'Old':>8s} {'New':>8s} {'Delta':>8s}")
    all_regime_names = sorted(set(list(old_dist.keys()) + list(new_dist.keys())))
    for regime in all_regime_names:
        old_pct = old_dist.get(regime, 0) / len(old_regimes)
        new_pct = new_dist.get(regime, 0) / len(new_regimes)
        logger.info(f"  {regime:25s} {old_pct:>7.1%} {new_pct:>7.1%} {new_pct - old_pct:>+7.1%}")

    logger.info(f"\n{'='*60}")
    if agreement >= 0.85:
        verdict = "HIGH AGREEMENT - DRL retraining likely NOT needed"
        drl_retrain = False
    elif agreement >= 0.70:
        verdict = "MODERATE AGREEMENT - DRL retraining RECOMMENDED"
        drl_retrain = True
    else:
        verdict = "LOW AGREEMENT - DRL retraining REQUIRED"
        drl_retrain = True
    logger.info(f"VERDICT: {verdict}")
    logger.info(f"{'='*60}")

    report = {
        "overall_agreement": round(agreement, 4),
        "regime_agreement": {k: round(v, 4) for k, v in regime_agreement.items()},
        "transition_agreement": round(transition_agreement, 4),
        "old_confidence_mean": round(old_conf_mean, 4),
        "new_confidence_mean": round(new_conf_mean, 4),
        "old_distribution": {k: round(v / len(old_regimes), 4) for k, v in old_dist.items()},
        "new_distribution": {k: round(v / len(new_regimes), 4) for k, v in new_dist.items()},
        "total_samples": len(old_regimes),
        "drl_retrain_needed": drl_retrain,
        "verdict": verdict,
    }
    return report


# ============================================================================
# STEP 6: SAVE / DEPLOY
# ============================================================================

def save_new_model(
    gmm,
    scaler,
    mapping: dict,
    features: np.ndarray,
    labels: np.ndarray,
    agreement_report: dict,
    deploy: bool = False,
):
    """Save the new GMM model and config."""
    suffix = "" if deploy else "_NEW"
    now = datetime.now(timezone.utc)

    # Build config
    config = {
        "n_components": gmm.n_components,
        "covariance_type": gmm.covariance_type,
        "means": gmm.means_.tolist(),
        "weights": gmm.weights_.tolist(),
        "regime_names": [mapping[i] for i in range(gmm.n_components)],
        "regime_mapping": {str(i): mapping[i] for i in range(gmm.n_components)},
        "feature_cols": FEATURE_COLS,
        "trained_at": now.isoformat(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "training_samples": len(labels),
        "agreement_with_old": agreement_report.get("overall_agreement"),
    }

    # Backup old model before deploying
    if deploy:
        backup_suffix = now.strftime("%Y%m%d_%H%M%S")
        for fname in ["gmm_model.pkl", "gmm_config.json", "scaler.pkl"]:
            old_path = GMM_DIR / fname
            if old_path.exists():
                backup_path = GMM_DIR / f"{fname}.backup_{backup_suffix}"
                old_path.rename(backup_path)
                logger.info(f"  Backed up {fname} -> {backup_path.name}")

    # Save model
    model_path = GMM_DIR / f"gmm_model{suffix}.pkl"
    joblib.dump(gmm, model_path)
    logger.info(f"Saved GMM model -> {model_path}")

    # Save config
    config_path = GMM_DIR / f"gmm_config{suffix}.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Saved config -> {config_path}")

    # Save scaler separately
    scaler_path = GMM_DIR / f"scaler{suffix}.pkl"
    joblib.dump(scaler, scaler_path)
    logger.info(f"Saved scaler -> {scaler_path}")

    # Save numpy artifacts
    np.save(GMM_DIR / f"gmm_means{suffix}.npy", gmm.means_)
    np.save(GMM_DIR / f"gmm_covariances{suffix}.npy", gmm.covariances_)

    # Save agreement report
    report_path = GMM_DIR / f"agreement_report{suffix}.json"
    with open(report_path, "w") as f:
        json.dump(agreement_report, f, indent=2)
    logger.info(f"Saved agreement report -> {report_path}")

    # Save training report
    report_md_path = GMM_DIR / f"training_report{suffix}.md"
    dist = Counter(labels.tolist())
    with open(report_md_path, "w") as f:
        f.write(f"# GMM Regime Classifier Training Report\n\n")
        f.write(f"## Training Info\n")
        f.write(f"- Date: {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- Samples: {len(labels)}\n")
        f.write(f"- Agreement with old model: {agreement_report.get('overall_agreement', 'N/A')}\n\n")
        f.write(f"## Model Config\n")
        f.write(f"- Components: {gmm.n_components}\n")
        f.write(f"- Covariance Type: {gmm.covariance_type}\n")
        f.write(f"- Log-Likelihood: {gmm.score(scaler.transform(features)):.4f}\n\n")
        f.write(f"## Regime Mapping\n")
        f.write(f"| Cluster | Regime Name | % |\n")
        f.write(f"|---------|-------------|---|\n")
        for c in range(gmm.n_components):
            pct = dist.get(c, 0) / len(labels) * 100
            f.write(f"| {c} | {mapping[c]} | {pct:.1f}% |\n")
        f.write(f"\n## Scaler Statistics\n")
        f.write(f"| Feature | Mean | Scale |\n")
        f.write(f"|---------|------|-------|\n")
        for i, name in enumerate(FEATURE_COLS):
            f.write(f"| {name} | {scaler.mean_[i]:.6f} | {scaler.scale_[i]:.6f} |\n")

    logger.info(f"Saved training report -> {report_md_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Retrain GMM Regime Classifier")
    parser.add_argument("--fetch-only", action="store_true", help="Only fetch data, don't train")
    parser.add_argument("--skip-fetch", action="store_true", help="Use cached data")
    parser.add_argument("--dry-run", action="store_true", help="Train but don't deploy to production path")
    parser.add_argument("--deploy", action="store_true", help="Deploy directly (backup old model)")
    parser.add_argument("--bars", type=int, default=BARS_NEEDED, help="Number of 4H bars per asset")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("HMATS GMM RETRAINING PIPELINE")
    logger.info("=" * 60)

    # --- Step 1: Data ---
    if args.skip_fetch:
        logger.info("\nStep 1: Loading cached data...")
        data = load_data_cache()
    else:
        logger.info("\nStep 1: Fetching fresh data from Kraken...")
        data = fetch_all_assets(limit=args.bars)
        save_data_cache(data)
        if args.fetch_only:
            logger.info("Data fetch complete. Use --skip-fetch to train.")
            return

    # --- Step 2: Feature engineering ---
    logger.info("\nStep 2: Computing features (matching runtime exactly)...")
    features, timestamps, asset_labels = build_feature_matrix(data)

    # --- Step 3: Train ---
    logger.info("\nStep 3: Training GMM...")
    gmm, scaler, labels, probs = train_gmm(features, n_components=6)

    # Map clusters to regimes
    mapping = map_clusters_to_regimes(gmm, scaler, features, labels)

    # --- Step 4: Sanity checks ---
    logger.info("\nStep 4: Running sanity checks...")
    passed = run_sanity_checks(gmm, scaler, features, labels)

    if not passed:
        logger.error("\nSanity checks FAILED. Aborting deployment.")
        logger.error("Review the output above and adjust parameters.")
        # Still save for inspection
        save_new_model(gmm, scaler, mapping, features, labels, {}, deploy=False)
        return

    # --- Step 5: Agreement backtest ---
    logger.info("\nStep 5: Running regime agreement backtest...")
    old_config_path = GMM_DIR / "gmm_config.json"
    old_model_path = GMM_DIR / "gmm_model.pkl"

    if old_config_path.exists() and old_model_path.exists():
        agreement_report = regime_agreement_backtest(
            old_config_path, old_model_path,
            gmm, scaler, mapping, features,
        )
    else:
        logger.warning("Old model not found, skipping agreement backtest")
        agreement_report = {"overall_agreement": None, "verdict": "No old model for comparison"}

    # --- Step 6: Save ---
    logger.info("\nStep 6: Saving model...")
    deploy = args.deploy and not args.dry_run
    save_new_model(gmm, scaler, mapping, features, labels, agreement_report, deploy=deploy)

    if deploy:
        logger.info("\nModel DEPLOYED to production path.")
        logger.info("Run 'python main.py --mode paper' to verify.")
    else:
        logger.info("\nModel saved as *_NEW files. To deploy:")
        logger.info("  python scripts/retrain_gmm.py --skip-fetch --deploy")

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
