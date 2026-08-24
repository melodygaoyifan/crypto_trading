"""[P390] Backfill multi-year derivatives-positioning history from Binance Vision
futures METRICS — resolving the two blockers on the OI-level lead (P388):
  (1) CoinGlass OI/LSR is a ~186d rolling window, so the feature was missing for
      ~91% of the 6y training window -> unlearnable.
  (2) With only 186d (all in-sample), there was no OUT-OF-SAMPLE data to run the
      hold-aware gate on.
Binance Vision `futures/um/daily/metrics` publishes daily CSVs back to ~2020-10 with
sum_open_interest, top-trader & global long/short account ratios, and taker
long/short volume ratio — the full positioning bundle, at 5-min granularity, ~11KB
per file. Backfilling it gives ~5.5y spanning the folds: the feature becomes
learnable AND the OOS probe becomes possible NOW.

Output: training/training_data/coinglass_history/{ASSET}_metrics_4h.parquet (joins
the same dir the P389 accumulation writes; distinct name so it never collides).
Resample 5min -> 4h: OI = last (level), ratios = mean, taker ratio = mean. Causal by
construction (each 4h bar aggregates only its own 5-min rows).
"""
from __future__ import annotations
import io
import sys
import zipfile
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "training" / "training_data" / "coinglass_history"
CACHE = Path(__file__).resolve().parent / "_metricscache"
CACHE.mkdir(exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
SYMS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"}
URL = ("https://data.binance.vision/data/futures/um/daily/metrics/"
       "{sym}/{sym}-metrics-{d}.zip")


def fetch_day(sym, d):
    f = CACHE / f"{sym}-{d}.parquet"
    if f.exists():
        try:
            return pd.read_parquet(f)
        except Exception:
            pass
    url = URL.format(sym=sym, d=d)
    try:
        raw = urllib.request.urlopen(url, timeout=60).read()
    except Exception:
        return None
    try:
        zf = zipfile.ZipFile(io.BytesIO(raw))
        df = pd.read_csv(zf.open(zf.namelist()[0]))
    except Exception:
        return None
    if "create_time" not in df.columns:
        return None
    df["ts"] = pd.to_datetime(df["create_time"], utc=True)
    keep = df[["ts", "sum_open_interest", "sum_open_interest_value",
               "sum_toptrader_long_short_ratio", "count_long_short_ratio",
               "sum_taker_long_short_vol_ratio"]].copy()
    for c in keep.columns:
        if c != "ts":
            keep[c] = pd.to_numeric(keep[c], errors="coerce")
    keep.to_parquet(f)
    return keep


def build_4h(asset, start, end):
    sym = SYMS[asset]
    days = []
    d = start
    while d <= end:
        days.append(d); d += timedelta(days=1)
    frames = []
    missing = 0
    for dd in days:
        x = fetch_day(sym, dd.isoformat())
        if x is None or not len(x):
            missing += 1
            continue
        frames.append(x)
    if not frames:
        return None, len(days), missing
    m = pd.concat(frames).sort_values("ts").set_index("ts")
    r = m.resample("4h")
    out = pd.DataFrame({
        "timestamp": r["sum_open_interest"].last().index,
        "oi_close": r["sum_open_interest"].last().to_numpy(),
        "oi_value": r["sum_open_interest_value"].last().to_numpy(),
        "toptrader_ls_ratio": r["sum_toptrader_long_short_ratio"].mean().to_numpy(),
        "global_ls_ratio": r["count_long_short_ratio"].mean().to_numpy(),
        "taker_ls_ratio": r["sum_taker_long_short_vol_ratio"].mean().to_numpy(),
    }).dropna(subset=["oi_close"])
    out["asset"] = asset
    return out, len(days), missing


def main():
    start = date(*map(int, (sys.argv[1] if len(sys.argv) > 1 else "2020-10-01").split("-")))
    end = date(*map(int, (sys.argv[2] if len(sys.argv) > 2 else "2026-07-31").split("-")))
    assets = sys.argv[3].split(",") if len(sys.argv) > 3 else list(SYMS)
    print("=" * 84)
    print(f"  BINANCE VISION METRICS BACKFILL {start} -> {end}  (OI + long/short + taker)")
    print("=" * 84)
    for a in assets:
        df, ndays, missing = build_4h(a, start, end)
        if df is None:
            print(f"{a}: NO DATA ({missing}/{ndays} days missing)"); continue
        p = OUT / f"{a}_metrics_4h.parquet"
        df.to_parquet(p)
        print(f"{a}: {len(df)} 4H bars, {str(df['timestamp'].min())[:10]} -> "
              f"{str(df['timestamp'].max())[:10]}, {missing}/{ndays} days missing "
              f"-> {p.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
