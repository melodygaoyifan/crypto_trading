"""Robust Binance monthly OHLCV downloader — fills gaps in {ASSET}_60m.parquet.

Unlike get_data.py, this:
  - Logs every month (success OR failure)
  - Retries transient failures (up to 3 times)
  - Merges with existing parquet instead of overwriting
  - Supports extended year range (default 3 years)
"""
from __future__ import annotations

import logging
import time
import zipfile
from datetime import datetime
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("fetch_binance_full")

BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"
SYMBOLS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"}
OUTPUT_DIR = Path("training/training_data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _month_url(asset: str, year: int, month: int) -> str:
    symbol = SYMBOLS[asset]
    return f"{BASE_URL}/{symbol}/1h/{symbol}-1h-{year}-{month:02d}.zip"


def download_month(asset: str, year: int, month: int, retries: int = 3) -> pd.DataFrame | None:
    url = _month_url(asset, year, month)
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 404:
                logger.warning(f"  {year}-{month:02d}: 404 (not published yet)")
                return None
            if r.status_code != 200:
                logger.warning(f"  {year}-{month:02d}: HTTP {r.status_code} (attempt {attempt})")
                time.sleep(2)
                continue
            with zipfile.ZipFile(BytesIO(r.content)) as zf:
                with zf.open(zf.namelist()[0]) as f:
                    df = pd.read_csv(f, header=None)
            df.columns = [
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "count", "taker_buy_base",
                "taker_buy_quote", "ignore",
            ]
            # Binance switched from ms → µs timestamps starting ~2025-01.
            # Detect by magnitude: ms timestamps ~1.7e12, µs ~1.7e15.
            _first = int(df["open_time"].iloc[0])
            _unit = "us" if _first > 1e14 else "ms"
            df["timestamp"] = pd.to_datetime(df["open_time"], unit=_unit)
            df = df[["timestamp", "open", "high", "low", "close", "volume"]]
            for c in ("open", "high", "low", "close", "volume"):
                df[c] = df[c].astype(float)
            return df
        except Exception as exc:
            logger.warning(f"  {year}-{month:02d}: {type(exc).__name__}: {str(exc)[:80]} (attempt {attempt})")
            time.sleep(2)
    return None


def fetch_asset(asset: str, start: datetime, end: datetime) -> pd.DataFrame:
    existing_path = OUTPUT_DIR / f"{asset}_60m.parquet"
    if existing_path.exists():
        existing = pd.read_parquet(existing_path)
        logger.info(f"{asset}: existing file has {len(existing)} rows "
                    f"({existing['timestamp'].min()} -> {existing['timestamp'].max()})")
    else:
        existing = None

    all_frames = []
    if existing is not None:
        all_frames.append(existing)

    current = start.replace(day=1)
    hit = miss = 0
    while current <= end:
        df = download_month(asset, current.year, current.month)
        if df is not None:
            all_frames.append(df)
            logger.info(f"  {asset} {current.year}-{current.month:02d}: {len(df)} rows")
            hit += 1
        else:
            miss += 1
        # advance
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)
        time.sleep(0.3)

    merged = pd.concat(all_frames, ignore_index=True)
    merged = merged.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    merged.to_parquet(existing_path)
    logger.info(f"{asset}: saved {len(merged)} rows to {existing_path}  "
                f"(hits={hit}, misses={miss})")
    return merged


def main() -> None:
    # [P199] The old hardcoded 3 years (~6,570 4H bars) is NOT enough for the
    # DRL's 3-fold walk-forward: train_drl_full's fold arithmetic needs
    # train_end = n - 3*int(0.15n) - 42 >= 5000, i.e. n >= ~9,200 4H bars
    # (~4.2 years). A 3-year fetch is exactly what produced the 2026-04-22
    # parquets on which folds 2 and 3 silently skip and fold_3 — the deployed
    # fold — is unreachable. Default is now 6 years (SOLUSDT listed on
    # Binance ~2020-08, so ~6y is also the maximum consistent range across
    # all three assets).
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=6,
                    help="Years of 1H history to fetch (default 6; "
                         "minimum ~5 for the 3-fold DRL walk-forward)")
    args = ap.parse_args()
    end = datetime.now()
    start = datetime(end.year - args.years, end.month, 1)
    n_4h_est = int((end - start).days * 6)
    if n_4h_est < 9200:
        logger.warning(
            f"--years {args.years} yields ~{n_4h_est} 4H bars < ~9,200 — "
            f"folds 2/3 of the DRL walk-forward will be SKIPPED on this data.")
    logger.info(f"Fetching 1H OHLCV: {start:%Y-%m} -> {end:%Y-%m}")
    for asset in ("BTC", "ETH", "SOL"):
        logger.info(f"\n===== {asset} =====")
        fetch_asset(asset, start, end)
    logger.info("\nAll done.")


if __name__ == "__main__":
    main()
