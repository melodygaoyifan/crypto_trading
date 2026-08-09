"""[P1a / P221-followup] fv2 flow features — the ONE implementation.

This module is the single source of truth for the 13 fv2_* features, and it
lives in data_mgmt/ (a runtime-shipped package) ON PURPOSE:

  * training/scripts/build_flow_features.py imports FROM here (training ->
    runtime is the safe import direction; the reverse is the P214 class —
    runtime importing training/ code that is not in the engine image).
  * the live feed (data_mgmt/feeds/binance_flow_feed.py) imports from here,
    so train/serve parity is IDENTITY, not a claim: same function object
    computes both sides. tests/test_flow_feature_parity_runtime.py pins it.

Every rolling statistic is CAUSAL (trailing windows, min_periods enforced;
warmup bars are NaN, never fabricated). tests/test_flow_features_causal.py
perturbs the future and asserts the past does not move (the P164 rule).

FEATURES (13)
-------------
flow (7, from kline flow columns, 4H-aggregated then rolling-z 30d=180 bars):
  fv2_taker_ratio_z       taker_buy_base / volume, z
  fv2_taker_ratio_mom     5d change of the taker ratio
  fv2_count_z             trade count, z (activity)
  fv2_avg_trade_size_z    quote_volume / count, z (large-trader proxy)
  fv2_amihud_z            |ret_4h| / quote_volume, z (illiquidity)
  fv2_quote_vol_z         quote_volume, z
  fv2_taker_quote_share_z taker_buy_quote / quote_volume, z
seasonality (4, deterministic): fv2_hour_sin/cos, fv2_dow_sin/cos
cross-asset (2): fv2_rel_strength_24h (vs reference), fv2_ref_lag_ret_4h
  (the reference's PREVIOUS bar return — the lag is what keeps it causal)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

Z_WINDOW = 180        # 30 days of 4H bars
Z_MIN = 42            # 1 week minimum before a z-score is emitted
MOM_BARS = 30         # 5 days
REF = {"BTC": "ETH", "ETH": "BTC", "SOL": "BTC"}
FLOW_COLS = ["quote_volume", "count", "taker_buy_base", "taker_buy_quote"]
FV2_COLUMNS = [
    "fv2_taker_ratio_z", "fv2_taker_ratio_mom", "fv2_count_z",
    "fv2_avg_trade_size_z", "fv2_amihud_z", "fv2_quote_vol_z",
    "fv2_taker_quote_share_z",
    "fv2_hour_sin", "fv2_hour_cos", "fv2_dow_sin", "fv2_dow_cos",
    "fv2_rel_strength_24h", "fv2_ref_lag_ret_4h",
]


def _roll_z(s: pd.Series, window: int = Z_WINDOW, min_periods: int = Z_MIN) -> pd.Series:
    """Trailing z-score: the statistic at t uses only bars <= t."""
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


def latest_fv2_vector(raw_1h_self: pd.DataFrame, raw_1h_ref: pd.DataFrame,
                      asset: str) -> "pd.Series | None":
    """The runtime entry point: the CURRENT bar's 13 fv2 values from trailing
    1H frames (self + reference asset). Returns None when warmup is not met —
    absence must be explicit, never fabricated (P160/P170)."""
    f = flow_features_4h(raw_1h_self)
    if f.empty:
        return None
    g_self = _agg_4h(raw_1h_self)[["timestamp", "close"]]
    g_ref = _agg_4h(raw_1h_ref)[["timestamp", "close"]]
    ref_name = REF[asset]
    x = cross_asset_features(asset, {asset: g_self, ref_name: g_ref})
    row = f.merge(x, on="timestamp", how="left").iloc[-1]
    vec = row[FV2_COLUMNS]
    if vec.isna().any():
        return None
    return vec
