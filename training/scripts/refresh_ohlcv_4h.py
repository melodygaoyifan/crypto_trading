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

[P266] DAILY-ARCHIVE EXTENSION
------------------------------
The raw parquet is bounded by Binance's MONTHLY archive (up to ~31 days
behind). P264 recorded why that is not good enough for this series: the
~2026-09-07..09 P166 forward reads (regime books, derivflow, ma_filter — the
September exams) need prices through early September, and the September
monthly archive lands ~October. This script therefore extends PAST the raw
parquet's end using Binance's DAILY vision archives (published T+1), fetched
here and merged in-memory — the TRAINING parquet is deliberately never
touched (see the section above; a price series and a feature matrix have
different owners). Only fully COMPLETED days are fetched, so every appended
4H bar is complete by construction. A daily-fetch failure degrades to the
monthly-only behavior, loudly.

USAGE
-----
    python -X utf8 training/fetch_binance_full.py        # refresh raw 60m first
    python -X utf8 training/scripts/refresh_ohlcv_4h.py  # then rebuild 4H OHLCV

[P420] BREADTH ASSETS + THE CALIBRATION-INPUT CONVENTION
--------------------------------------------------------
`--assets XRP,BNB` (any subset of the fetch_binance_full universe; an unknown
asset REFUSES, exit 2 — a fetcher that accepts any string writes a file for an
asset that does not exist and reports success, P291). The default stays the
home trio. This is the ONLY writer of `{ASSET}_4H_ohlcv.parquet`; the breadth
Kraken window that scripts/september_check.py fetches goes to a SEPARATE
`{ASSET}_4H_ohlcv_kraken.parquet` (the scorer unions them, primary wins).
Before P420 september_check merged Kraken prints INTO this file and the
seat-alpha producer (training/seat_alpha_calibration.py, the source of the
LIVE XRP/BNB gate constants) drifted from its own shipped table.

`--no-daily-extension` builds the raw-archive resample only (through the last
monthly archive). That is the exact input the shipped XRP/BNB seat-alpha table
was measured on (P412b/c, raw through 2026-07-31); the daily extension appends
completed August bars, which legitimately MOVES the open-ended validation cell.
The calibration report stamps the input's sha256 + row count so either build
is attributable (P420).
"""
from __future__ import annotations

import io
import sys
import urllib.request
import zipfile
from datetime import date, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RAW_DIR = REPO / "training" / "training_data" / "raw"
OUT_DIR = REPO / "training" / "training_data" / "drl_training"
ASSETS = ("BTC", "ETH", "SOL")          # the DEFAULT roster (home trio)
COLS = ["open", "high", "low", "close", "volume"]

# [P266] Same symbol map + archive host as fetch_binance_full.py.
# [P420] Extended to the P262/P281 breadth assets so their PRIMARY series can
# be rebuilt from raw/{ASSET}_60m.parquet with the SAME resample convention;
# pinned equal to fetch_binance_full.SYMBOLS by test (two maps of one universe
# drift — P172).
SYMBOLS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT",
           "XRP": "XRPUSDT", "ADA": "ADAUSDT", "LTC": "LTCUSDT",
           "DOGE": "DOGEUSDT", "BNB": "BNBUSDT"}
PRIMARY_SUFFIX = "_4H_ohlcv.parquet"


def primary_ohlcv_path(asset: str) -> Path:
    """[P420] The one file this script owns for `asset` (the calibration input)."""
    return OUT_DIR / f"{asset}{PRIMARY_SUFFIX}"


def parse_assets(spec) -> list:
    """[P420] Comma list -> upper-cased roster; REFUSES an unknown asset with a
    ValueError naming the universe (never silently builds nothing / something
    else)."""
    if not spec:
        return list(ASSETS)
    out = []
    for a in str(spec).split(","):
        a = a.strip().upper()
        if not a:
            continue
        if a not in SYMBOLS:
            raise ValueError(
                f"unknown asset {a!r}; known: {sorted(SYMBOLS)} (P291: a "
                f"fetcher must refuse, not fabricate)")
        if a not in out:
            out.append(a)
    if not out:
        raise ValueError("--assets resolved to an empty roster")
    return out


DAILY_BASE = "https://data.binance.vision/data/spot/daily/klines"
# A gap this large means the monthly fetcher was skipped for 2+ months —
# use it instead of hammering ~60+ daily zips.
MAX_DAILY_GAP_DAYS = 62


def daily_dates_needed(last_covered, today_utc) -> list:
    """ISO dates to fetch from the DAILY archives: the raw parquet's last
    covered day (re-fetched so a partial boundary day cannot leave a seam —
    dedup keep='last' makes the re-fetch harmless) through YESTERDAY. Today
    is never fetched: its file does not exist yet and its day is incomplete
    (an in-progress day would append partial 4H bars — the exact class of
    bug P253c/P265 removed elsewhere)."""
    start = last_covered
    end = today_utc - timedelta(days=1)
    if start > end:
        return []
    n = (end - start).days + 1
    if n > MAX_DAILY_GAP_DAYS:
        return None  # caller reports: run the monthly fetcher first
    return [(start + timedelta(days=i)).isoformat() for i in range(n)]


def parse_kline_frame(df):
    """Binance kline CSV -> [timestamp, o/h/l/c/v]. Same conventions as
    fetch_binance_full.py: 12 unnamed columns; ms vs µs timestamps detected
    by magnitude (Binance switched to µs ~2025-01)."""
    import pandas as pd
    df = df.copy()
    df.columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "count", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ]
    _first = int(df["open_time"].iloc[0])
    _unit = "us" if _first > 1e14 else "ms"
    df["timestamp"] = pd.to_datetime(df["open_time"], unit=_unit)
    out = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    for c in COLS:
        out[c] = out[c].astype(float)
    return out


def fetch_daily_1h(asset: str, iso_date: str):
    """One day's 1h klines from the daily archive, or None (404 = not yet
    published; any other failure logged by the caller)."""
    import pandas as pd
    sym = SYMBOLS[asset]
    url = f"{DAILY_BASE}/{sym}/1h/{sym}-1h-{iso_date}.zip"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            blob = r.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        with z.open(z.namelist()[0]) as f:
            raw = pd.read_csv(f, header=None)
    return parse_kline_frame(raw)


def extend_with_daily(asset: str, raw):
    """Merge daily-archive 1h bars past the raw parquet's end. Returns
    (frame, note). Fail-soft: any trouble returns the input unchanged with
    the reason — never worse than the monthly-only behavior."""
    import pandas as pd
    last_ts = pd.Timestamp(raw["timestamp"].max())
    today = pd.Timestamp.utcnow().date()
    dates = daily_dates_needed(last_ts.date(), today)
    if dates is None:
        return raw, (f"gap > {MAX_DAILY_GAP_DAYS}d — run "
                     f"training/fetch_binance_full.py first")
    if not dates:
        return raw, "already current through yesterday"
    frames = []
    stopped = None
    for d in dates:
        try:
            f = fetch_daily_1h(asset, d)
        except Exception as e:  # noqa: silent-swallow — degrade to monthly-only, reason surfaced in the note
            stopped = f"{d}: {type(e).__name__}: {e}"
            break
        if f is None:
            stopped = f"{d}: not yet published (404)"
            break
        frames.append(f)
    if not frames:
        return raw, f"no daily data added ({stopped or 'nothing to fetch'})"
    ext = pd.concat(frames, ignore_index=True)
    merged = (pd.concat([raw[["timestamp"] + COLS], ext], ignore_index=True)
                .drop_duplicates("timestamp", keep="last")
                .sort_values("timestamp").reset_index(drop=True))
    note = (f"+{len(frames)} day(s) via daily archives -> "
            f"{pd.Timestamp(merged['timestamp'].max())}")
    if stopped:
        note += f" (stopped at {stopped})"
    return merged, note


def build(asset: str, daily_extension: bool = True):
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

    # [P266] Extend past the monthly-archive boundary via DAILY archives
    # (T+1, completed days only). In-memory only — the raw parquet is
    # deliberately untouched.
    if daily_extension:
        raw, _ext_note = extend_with_daily(asset, raw)
    else:
        # [P420] raw-archive only: the seat-alpha calibration input convention
        _ext_note = "SKIPPED (--no-daily-extension; raw archive only)"
    print(f"  {asset}: daily extension: {_ext_note}")

    # origin='start_day' pins bins to 00/04/08/12/16/20 UTC, which is the
    # convention the existing 4H parquet uses. Any other origin silently
    # shifts every bar and the 0.000000% validation above would not hold.
    df = (raw.set_index("timestamp")
             .resample("4h", origin="start_day")
             .agg({"open": "first", "high": "max", "low": "min",
                   "close": "last", "volume": "sum"})
             .dropna(subset=["close"])
             .reset_index())

    out = primary_ohlcv_path(asset)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"  {asset}: {len(df):6} bars  {df.timestamp.min()} -> {df.timestamp.max()}"
          f"  -> {out.name}")
    return df


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="[P199/P420] rebuild {ASSET}_4H_ohlcv.parquet")
    ap.add_argument("--assets", default=None,
                    help="comma list (default: BTC,ETH,SOL); unknown -> refuse")
    ap.add_argument("--no-daily-extension", action="store_true",
                    help="raw monthly archive only — the seat-alpha calibration "
                         "input convention (P420)")
    args = ap.parse_args(argv)
    try:
        assets = parse_assets(args.assets)
    except ValueError as e:
        print(f"REFUSING: {e}")
        return 2
    print("=" * 78)
    print("[P199] Rebuilding 4H OHLCV for analytics (training parquets untouched)")
    print(f"       assets={assets} daily_extension={not args.no_daily_extension}")
    print("=" * 78)
    built = {a: build(a, daily_extension=not args.no_daily_extension)
             for a in assets}
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
