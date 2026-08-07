"""[P200-FEATURES] Flow-v2 + seasonality + cross-asset features — the feature
build the edge probe asked for.

WHY THESE
---------
Three independent lines of evidence point at the same family:
  1. The edge probe's strongest group was external flow/positioning
     (funding/OI/liq) — but it has only ~180 days of Coinglass API depth.
  2. Live agent ICs: whale + funding were the only agents positive in both
     measured windows (Apr-Jun and Jun-Aug 2026).
  3. Literature: order flow substantially improves daily crypto return
     prediction over 40 price-derived characteristics without it
     (Anastasopoulos & Gradojevic 2025/2026); microstructure feature
     importance is stable across assets.
The Binance kline archive carries taker_buy_base/quote, trade count and
quote_volume for the FULL history at 1H — the same signal family, free.
fetch_binance_full.py now keeps those columns; this script turns them into
causal 4H features and appends them to the training parquets.

CONTRACT
--------
* Appends EXTRA columns to training/training_data/drl_training/{A}_4H_full.parquet.
  It does NOT touch configs/feature_manifest.json — obs_dim stays 126 (Iron
  Law). New features are probe/ridge candidates first; admitting any into the
  DRL manifest is a separate, deliberate obs-contract change.
* Every rolling statistic is CAUSAL (trailing windows only, min_periods
  enforced; early bars are NaN, never filled from the future). P164 rule:
  a feature transform is verified causal by construction, not by inspection —
  tests/test_flow_features_causal.py perturbs the future and asserts the past
  does not move.

FEATURES (13)
-------------
flow (7, from kline flow columns, 4H-aggregated then rolling-z 30d=180 bars):
  fv2_taker_ratio_z      taker_buy_base / volume, z
  fv2_taker_ratio_mom    5d change of the taker ratio
  fv2_count_z            trade count, z (activity)
  fv2_avg_trade_size_z   quote_volume / count, z (large-trader proxy)
  fv2_amihud_z           |ret_4h| / quote_volume, z (illiquidity)
  fv2_quote_vol_z        quote_volume, z
  fv2_taker_quote_share_z taker_buy_quote / quote_volume, z
seasonality (4, deterministic — documented weak but orthogonal and free):
  fv2_hour_sin/cos       position of the 4H bar inside the UTC day
  fv2_dow_sin/cos        day of week
cross-asset (2, from the sibling parquets; BTC gets ETH as reference):
  fv2_rel_strength_24h   ret_24h(asset) - ret_24h(ref)
  fv2_ref_lag_ret_4h     the reference asset's PREVIOUS bar return (lead-lag)

Usage:  python -X utf8 training/scripts/build_flow_features.py
Idempotent: recomputes and overwrites only fv2_* columns.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_TRAINING_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = _TRAINING_DIR / "training_data" / "raw"
OUT_DIR = _TRAINING_DIR / "training_data" / "drl_training"

Z_WINDOW = 180        # 30 days of 4H bars
Z_MIN = 42            # 1 week minimum before a z-score is emitted
MOM_BARS = 30         # 5 days
ASSETS = ("BTC", "ETH", "SOL")
REF = {"BTC": "ETH", "ETH": "BTC", "SOL": "BTC"}
FLOW_COLS = ["quote_volume", "count", "taker_buy_base", "taker_buy_quote"]


def _roll_z(s: pd.Series, window: int = Z_WINDOW, min_periods: int = Z_MIN) -> pd.Series:
    """Trailing z-score. shift(0) semantics: the bar's own value enters its
    window — acceptable because the STATISTIC uses only bars <= t."""
    m = s.rolling(window, min_periods=min_periods).mean()
    sd = s.rolling(window, min_periods=min_periods).std()
    return ((s - m) / sd.replace(0, np.nan)).clip(-6, 6)


def _agg_4h(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    agg = {"volume": "sum", "close": "last", "open": "first"}
    for c in FLOW_COLS:
        agg[c] = "sum"
    out = (df.set_index("timestamp").resample("4h").agg(agg)).dropna(subset=["close"])
    return out.reset_index()


def flow_features_4h(raw: pd.DataFrame) -> pd.DataFrame:
    """4H flow feature frame from a 1H raw frame carrying the flow columns."""
    g = _agg_4h(raw)
    out = pd.DataFrame({"timestamp": g["timestamp"]})
    ret_4h = g["close"].pct_change()

    vol = g["volume"].replace(0, np.nan)
    qv = g["quote_volume"].replace(0, np.nan)
    cnt = g["count"].replace(0, np.nan)

    taker_ratio = (g["taker_buy_base"] / vol).clip(0, 1)
    out["fv2_taker_ratio_z"] = _roll_z(taker_ratio)
    out["fv2_taker_ratio_mom"] = (taker_ratio
                                  - taker_ratio.rolling(MOM_BARS, min_periods=MOM_BARS).mean())
    out["fv2_count_z"] = _roll_z(np.log1p(cnt))
    out["fv2_avg_trade_size_z"] = _roll_z(np.log1p(qv / cnt))
    out["fv2_amihud_z"] = _roll_z(np.log1p(ret_4h.abs() / qv * 1e9))
    out["fv2_quote_vol_z"] = _roll_z(np.log1p(qv))
    out["fv2_taker_quote_share_z"] = _roll_z((g["taker_buy_quote"] / qv).clip(0, 1))

    ts = pd.to_datetime(out["timestamp"])
    hour_phase = ts.dt.hour / 24.0 * 2 * np.pi
    dow_phase = ts.dt.dayofweek / 7.0 * 2 * np.pi
    out["fv2_hour_sin"] = np.sin(hour_phase)
    out["fv2_hour_cos"] = np.cos(hour_phase)
    out["fv2_dow_sin"] = np.sin(dow_phase)
    out["fv2_dow_cos"] = np.cos(dow_phase)
    return out


def cross_asset_features(asset: str, closes: dict) -> pd.DataFrame:
    """Relative strength vs the reference asset + the reference's LAGGED
    return (lead-lag). The lag is what keeps it causal: the reference's
    CURRENT bar is contemporaneous, its previous bar is information."""
    ref = REF[asset]
    a = closes[asset]
    r = closes[ref]
    m = a.merge(r, on="timestamp", how="left", suffixes=("", "_ref"))
    out = pd.DataFrame({"timestamp": m["timestamp"]})
    ret24_a = m["close"].pct_change(6)
    ret24_r = m["close_ref"].pct_change(6)
    out["fv2_rel_strength_24h"] = (ret24_a - ret24_r).clip(-1, 1)
    out["fv2_ref_lag_ret_4h"] = m["close_ref"].pct_change().shift(1).clip(-0.5, 0.5)
    return out


def main() -> int:
    closes = {}
    flow = {}
    for a in ASSETS:
        raw = pd.read_parquet(RAW_DIR / f"{a}_60m.parquet")
        missing = [c for c in FLOW_COLS if c not in raw.columns]
        if missing:
            print(f"ERROR: {a}_60m.parquet lacks flow columns {missing} — "
                  f"re-run training/fetch_binance_full.py (post-P200-FEATURES) first.")
            return 1
        n_nan = int(raw[FLOW_COLS].isna().all(axis=1).sum())
        if n_nan:
            print(f"  {a}: {n_nan} raw rows lack flow data (pre-refetch rows) "
                  f"— their features will be NaN, which is honest")
        f = flow_features_4h(raw)
        flow[a] = f
        g = _agg_4h(raw)
        closes[a] = g[["timestamp", "close"]]

    for a in ASSETS:
        pq = OUT_DIR / f"{a}_4H_full.parquet"
        df = pd.read_parquet(pq)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.drop(columns=[c for c in df.columns if c.startswith("fv2_")])
        feats = flow[a].merge(cross_asset_features(a, closes), on="timestamp", how="left")
        feats["timestamp"] = pd.to_datetime(feats["timestamp"])
        merged = df.merge(feats, on="timestamp", how="left")
        assert len(merged) == len(df), "flow merge changed row count"
        fv2 = [c for c in merged.columns if c.startswith("fv2_")]
        cov = 1.0 - float(merged[fv2].isna().all(axis=1).mean())
        merged.to_parquet(pq)
        print(f"  {a}: +{len(fv2)} fv2 features, coverage {cov:.0%}, "
              f"rows {len(merged)} -> {pq.name}")
    print("NOTE: fv2_* are EXTRA columns — NOT in configs/feature_manifest.json; "
          "obs_dim stays 126. Probe them with edge_probe.py before any "
          "manifest change.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
