#!/usr/bin/env python3
"""[P281] Binance FUTURES daily aggregates — fills the three feature columns
that have been ALL-ZERO for the project's entire history.

WHY
---
The P279 data audit found `taker_ratio_zscore`, `tradecount_zscore` and
`taker_vol_momentum` in the feature manifest with NO source: the loader
(`rebuild_pipeline._load_futures_daily`) expects
`training_data/futures/{ASSET}_futures_daily.parquet`, the directory never
existed, and the merge silently backfilled 0.0 — every model ever trained
consumed three dead inputs. This fetcher builds those files at FULL history
from the same Binance Vision archive API `fetch_binance_full` already uses
(futures/um monthly 1d klines: taker-buy volumes + trade count, published
since each perp's 2019/2020 listing).

COLUMN MAPPING (documented because the loader's names predate this source):
  marketorder_volume      := taker_buy_quote  (USD taker buy volume/day)
  marketorder_volume_from := taker_buy_base   (base-asset taker buy/day)
  tradecount              := count            (trades/day)
The downstream features are a 30d z-score and a 5d pct-change — both
UNIT-FREE, so the mapping choice cannot skew scale-calibrated consumers
(the P219/P224 magnitude-swap trap does not apply).

CAUSALITY (the P247 lesson, applied at BUILD time not patch time):
each row is stamped at the day's CLOSE boundary (day open + 1 day), so the
merge_asof in rebuild_pipeline hands a 4H bar only COMPLETED days — day-D
data is never visible to day-D bars. `data_date` carries the actual day for
audit. Merge-not-overwrite (P266): re-runs union by timestamp, new wins.

Run (operator-local):
    python -X utf8 training/scripts/fetch_binance_futures_daily.py
    python -X utf8 training/scripts/fetch_binance_futures_daily.py \
        --assets BTC,ETH,SOL,XRP,ADA,LTC,DOGE,BNB
"""
from __future__ import annotations

import argparse
import logging
import time
import zipfile
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("fetch_futures_daily")

BASE_URL = "https://data.binance.vision/data/futures/um/monthly/klines"
SYMBOLS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT",
           "XRP": "XRPUSDT", "ADA": "ADAUSDT", "LTC": "LTCUSDT",
           "DOGE": "DOGEUSDT", "BNB": "BNBUSDT"}
OUT_DIR = Path(__file__).resolve().parent.parent / "training_data" / "futures"

KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume",
              "close_time", "quote_volume", "count", "taker_buy_base",
              "taker_buy_quote", "ignore"]


def download_month(asset: str, year: int, month: int,
                   retries: int = 3) -> pd.DataFrame | None:
    sym = SYMBOLS[asset]
    url = f"{BASE_URL}/{sym}/1d/{sym}-1d-{year}-{month:02d}.zip"
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 404:
                return None          # pre-listing / not yet published
            if r.status_code != 200:
                time.sleep(2)
                continue
            with zipfile.ZipFile(BytesIO(r.content)) as zf:
                with zf.open(zf.namelist()[0]) as f:
                    df = pd.read_csv(f, header=None)
            if len(df.columns) != 12:
                return None
            df.columns = KLINE_COLS
            # header row guard (some archives ship one) + ms->us switch
            df = df[pd.to_numeric(df["open_time"], errors="coerce").notna()]
            df["open_time"] = pd.to_numeric(df["open_time"])
            unit = "us" if df["open_time"].iloc[0] > 1e14 else "ms"
            df["open_time"] = pd.to_datetime(df["open_time"], unit=unit,
                                             utc=True)
            for c in ("taker_buy_base", "taker_buy_quote", "count"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
            return df
        except Exception as e:
            logger.warning(f"  {asset} {year}-{month:02d}: "
                           f"{type(e).__name__} (attempt {attempt})")
            time.sleep(2)
    return None


def fetch_asset(asset: str, years: int) -> pd.DataFrame:
    end = datetime.now(timezone.utc)
    frames = []
    y, m = end.year - years, end.month
    while (y, m) <= (end.year, end.month):
        df = download_month(asset, y, m)
        if df is not None and len(df):
            frames.append(df)
        m += 1
        if m > 12:
            y, m = y + 1, 1
    if not frames:
        return pd.DataFrame()
    raw = (pd.concat(frames)
           .drop_duplicates(subset="open_time")
           .sort_values("open_time").reset_index(drop=True))
    out = pd.DataFrame({
        # availability stamp = day CLOSE boundary (P247 causality at build)
        "timestamp": raw["open_time"] + timedelta(days=1),
        "data_date": raw["open_time"].dt.date.astype(str),
        "marketorder_volume": raw["taker_buy_quote"],
        "marketorder_volume_from": raw["taker_buy_base"],
        "tradecount": raw["count"],
    })
    # drop the LAST row if its data_date is TODAY (in-progress day, P253c —
    # monthly archives only publish completed days, but belt over suspenders)
    today = end.date().isoformat()
    out = out[out["data_date"] < today].reset_index(drop=True)
    return out


def merge_write(asset: str, new: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{asset}_futures_daily.parquet"
    if path.exists():
        try:
            old = pd.read_parquet(path)
            new = (pd.concat([old, new])
                   .drop_duplicates(subset="timestamp", keep="last")
                   .sort_values("timestamp").reset_index(drop=True))
        except Exception:
            path.rename(path.with_suffix(".parquet.bak"))
    tmp = path.with_suffix(".tmp")
    new.to_parquet(tmp, index=False)
    tmp.replace(path)
    logger.info(f"  {asset}: {len(new)} days "
                f"({new['data_date'].iloc[0]} -> {new['data_date'].iloc[-1]})"
                f" -> {path.name}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default="BTC,ETH,SOL")
    ap.add_argument("--years", type=int, default=6)
    args = ap.parse_args()
    for asset in [a.strip().upper() for a in args.assets.split(",") if a]:
        if asset not in SYMBOLS:
            logger.warning(f"  {asset}: no perp symbol mapped — skipped")
            continue
        logger.info(f"[{asset}] fetching futures 1d klines "
                    f"({args.years}y)...")
        df = fetch_asset(asset, args.years)
        if df.empty:
            logger.warning(f"  {asset}: nothing fetched")
            continue
        merge_write(asset, df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
