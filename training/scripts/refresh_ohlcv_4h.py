#!/usr/bin/env python3
"""
[P199] Refresh the 4H OHLCV series used by ANALYTICS (shadow IC, etc.).

WHY THIS EXISTS
---------------
`analytics/shadow_ic/compute_shadow_ic.py` needs 4H closes to compute forward
returns. It read `training/training_data/drl_training/{ASSET}_4H_full.parquet`,
which is the **DRL training set** — 130 columns, regenerated only by a full
`rebuild_pipeline` run, and frozen at **2026-03-31**. The shadow ledgers start
**2026-04-30**. Zero overlap, so every shadow record scored `N=0` and the gate
reported INSUFFICIENT_SAMPLES for months — indistinguishable from "the
strategies have no signal". The v5.1 promotion criterion could never be
evaluated, on the server (parquets are dockerignored) or off it.

The fix is to stop coupling an analytics price series to a training artifact.
This writes `{ASSET}_4H_ohlcv.parquet` — OHLCV only, no features — which
compute_shadow_ic prefers over the training parquet.

WHY NOT JUST APPEND TO THE TRAINING PARQUET
-------------------------------------------
It has 122 feature columns. Appending OHLCV-only rows leaves every feature NaN
for the new bars, silently corrupting the DRL training input. A price series and
a feature matrix have different refresh cadences and different owners; conflating
them is what created this.

SOURCE
------
Resampled from `training/training_data/raw/{ASSET}_60m.parquet`, which
`training/fetch_binance_full.py` refreshes (it merges rather than overwrites, so
re-running is safe and cheap). Validated 2026-08-07: the resample reproduces all
**6,525** overlapping bars of the existing 4H parquet to **0.000000%** — this is
exactly the transform that built it.

Coverage is bounded by Binance's MONTHLY archive, so the current partial month is
absent until published. That is a lag, not a gap, and it is reported explicitly
below rather than left for the caller to infer.

USAGE
-----
    python -X utf8 training/fetch_binance_full.py        # refresh raw 60m first
    python -X utf8 training/scripts/refresh_ohlcv_4h.py  # then rebuild 4H OHLCV
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RAW_DIR = REPO / "training" / "training_data" / "raw"
OUT_DIR = REPO / "training" / "training_data" / "drl_training"
ASSETS = ("BTC", "ETH", "SOL")
COLS = ["open", "high", "low", "close", "volume"]


def build(asset: str):
    import pandas as pd

    src = RAW_DIR / f"{asset}_60m.parquet"
    if not src.exists():
        print(f"  {asset}: MISSING {src} — run training/fetch_binance_full.py first")
        return None

    raw = pd.read_parquet(src)
    if "timestamp" not in raw.columns:
        print(f"  {asset}: {src.name} has no `timestamp` column; got "
              f"{list(raw.columns)[:6]}")
        return None
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])

    # origin='start_day' pins bins to 00/04/08/12/16/20 UTC, which is the
    # convention the existing 4H parquet uses. Any other origin silently
    # shifts every bar and the 0.000000% validation above would not hold.
    df = (raw.set_index("timestamp")
             .resample("4h", origin="start_day")
             .agg({"open": "first", "high": "max", "low": "min",
                   "close": "last", "volume": "sum"})
             .dropna(subset=["close"])
             .reset_index())

    out = OUT_DIR / f"{asset}_4H_ohlcv.parquet"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"  {asset}: {len(df):6} bars  {df.timestamp.min()} -> {df.timestamp.max()}"
          f"  -> {out.name}")
    return df


def main() -> int:
    print("=" * 78)
    print("[P199] Rebuilding 4H OHLCV for analytics (training parquets untouched)")
    print("=" * 78)
    built = {a: build(a) for a in ASSETS}
    ok = [a for a, d in built.items() if d is not None]
    if not ok:
        print("\nNothing built. Run training/fetch_binance_full.py first.")
        return 2

    # State the coverage lag outright. A silently-stale price series is the
    # exact failure this script exists to end.
    import pandas as pd
    latest = min(pd.to_datetime(built[a].timestamp.max()) for a in ok)
    now = pd.Timestamp.utcnow().tz_localize(None)
    lag_days = (now - latest).total_seconds() / 86400.0
    print(f"\nCoverage ends {latest} ({lag_days:.1f} days behind now).")
    if lag_days > 40:
        print("  WARNING: more than ~40 days stale. Binance publishes MONTHLY "
              "archives, so a lag up to ~31 days is expected; beyond that, "
              "training/fetch_binance_full.py has probably not been re-run.")
    print("\nNext: python -X utf8 -m analytics.shadow_ic.compute_shadow_ic \\")
    print("        --prefixes microstructure,cascade,funding,ml_factor --window-days 30")
    return 0


if __name__ == "__main__":
    sys.exit(main())
