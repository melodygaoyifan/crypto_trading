"""[P221-followup] Full-history funding rates from Binance fapi — FREE.

The external feature family was limited to ~180 days (8% of bars) because
Coinglass was the assumed source and its API depth is paid. But funding —
the single most evidence-backed external feature (live-positive in both
measured IC windows, and the strongest probe group) — has a free,
full-history source: Binance's /fapi/v1/fundingRate endpoint (8h records
back to each perp's listing).

Writes {ASSET}_funding_1d.parquet into the coinglass_history dir in exactly
the shape rebuild_pipeline._load_coinglass_daily expects (timestamp,
funding_close daily last), so coverage of `funding_rate_zscore` goes from
~8% to ~full-history with zero loader changes. OI and liquidations remain
Coinglass-limited (genuinely paid) — `has_external_data` handles partials.

Usage: python -X utf8 training/scripts/fetch_binance_funding.py
"""

from __future__ import annotations

import sys
import time
import zipfile
from datetime import datetime
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests

OUT_DIR = Path(__file__).resolve().parent.parent / "training_data" / "coinglass_history"
SYMBOLS = {
    "BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT",
    # [P301] The five P262-certified breadth assets. Their trend/hold books
    # beat flat 5/5 on data no selection ever touched, but the derivatives
    # exam could not price their CARRY - and carry is the dominant drag on a
    # long-biased perp book (P296 measured -59.7% over six years for a
    # permanent long). Without funding they could only be judged on a
    # quantity that ignores the biggest cost they would actually pay.
    "XRP": "XRPUSDT", "ADA": "ADAUSDT", "LTC": "LTCUSDT",
    "DOGE": "DOGEUSDT", "BNB": "BNBUSDT",
}
# The fapi REST endpoint returns HTTP 451 (region-blocked) from this machine;
# the Vision ARCHIVE serves the same data un-blocked — the same CDN the kline
# fetch already uses.
BASE = "https://data.binance.vision/data/futures/um/monthly/fundingRate"
START = (2020, 8)  # earliest consistent history across the three perps


def fetch_all(symbol: str) -> pd.DataFrame:
    frames = []
    now = datetime.utcnow()
    y, m = START
    while (y, m) <= (now.year, now.month):
        url = f"{BASE}/{symbol}/{symbol}-fundingRate-{y}-{m:02d}.zip"
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                with zipfile.ZipFile(BytesIO(r.content)) as zf:
                    with zf.open(zf.namelist()[0]) as f:
                        df = pd.read_csv(f)
                df.columns = [c.strip().lower() for c in df.columns]
                tcol = "calc_time" if "calc_time" in df.columns else df.columns[0]
                rcol = ("last_funding_rate" if "last_funding_rate" in df.columns
                        else df.columns[-1])
                out = pd.DataFrame({
                    "timestamp": pd.to_datetime(df[tcol].astype("int64"),
                                                unit="ms", utc=True),
                    "fundingRate": df[rcol].astype(float),
                })
                frames.append(out)
            elif r.status_code != 404:
                print(f"  {symbol} {y}-{m:02d}: HTTP {r.status_code}")
        except Exception as e:
            print(f"  {symbol} {y}-{m:02d}: {type(e).__name__}: {e}")
        time.sleep(0.2)
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates("timestamp").sort_values("timestamp")


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--assets", default=None,
                    help="comma-separated subset, e.g. XRP,ADA (default: all)")
    args = ap.parse_args(argv)
    wanted = ([a.strip().upper() for a in args.assets.split(",") if a.strip()]
              if args.assets else list(SYMBOLS))
    unknown = [a for a in wanted if a not in SYMBOLS]
    if unknown:
        # [P291] a fetcher that accepts any string writes a file for an asset
        # that does not exist and reports success.
        print(f"ERROR: unknown asset(s) {unknown}; known: {sorted(SYMBOLS)}")
        return 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for asset in wanted:
        sym = SYMBOLS[asset]
        df = fetch_all(sym)
        daily = (df.set_index("timestamp")["fundingRate"]
                 .resample("1D").agg(["first", "max", "min", "last"])
                 .dropna(how="all").reset_index())
        daily.columns = ["timestamp", "funding_open", "funding_high",
                         "funding_low", "funding_close"]
        out = OUT_DIR / f"{asset}_funding_1d.parquet"
        # [P266] MERGE, never overwrite. The rule was written for the
        # CoinGlass archive (bounded API depth, so the file is the only copy
        # of that history) and applies here for a different reason: a partial
        # Vision outage would otherwise replace a complete series with a
        # truncated one, silently.
        if out.exists():
            try:
                prev = pd.read_parquet(out)
                daily = (pd.concat([prev, daily], ignore_index=True)
                         .drop_duplicates("timestamp", keep="last")
                         .sort_values("timestamp").reset_index(drop=True))
            except Exception as e:
                print(f"  {asset}: existing parquet unreadable ({e}); "
                      f"writing fresh")
        daily.to_parquet(out)
        print(f"{asset}: {len(df)} 8h records -> {len(daily)} daily rows "
              f"({daily['timestamp'].min().date()} -> {daily['timestamp'].max().date()}) -> {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
