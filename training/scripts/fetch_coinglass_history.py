"""
Fetch historical Coinglass data (funding rate, OI, liquidation)
for DRL training integration.

Usage:
    python -X utf8 scripts/fetch_coinglass_history.py [--interval {4h,1d}]

Outputs (per --interval; default 4h):
    training/training_data/coinglass_history/{BTC,ETH,SOL}_funding_{iv}.parquet
    training/training_data/coinglass_history/{BTC,ETH,SOL}_oi_{iv}.parquet
    training/training_data/coinglass_history/{BTC,ETH,SOL}_liquidation_{iv}.parquet

[P287] BOTH intervals must be kept fresh, and the 1d files are the ones
that matter most: `rebuild_pipeline._load_coinglass_daily` reads ONLY the
`*_oi_1d.parquet` / `*_liquidation_1d.parquet` archives — the 4h files
feed nothing on the training path. Before P287 this script could only
write 4h (INTERVAL was an un-parameterized constant), so the consumed 1d
archives were a one-shot window no command could extend. `make
refresh-data` now runs both intervals.

API: Coinglass v3 (https://open-api-v3.coinglass.com)
Auth: CG-API-KEY header

[P266] CADENCE RULE — re-run this AT LEAST every ~5 months (150 days).
The API serves only ~180 days of depth for the liquidation/OI endpoints
(measured 2026-08: 1080 4h rows regardless of the lookback requested), and
this archive is the ONLY store of that history — `liq_imbalance`, the
strict-window carrier of the external feature group (P256), is trainable
only as far back as this file reaches. Since P266 the script MERGES into the
existing archive (never overwrites), so each re-run within the 180-day
window grows the history losslessly; a gap longer than the API window
permanently loses the middle. `make check` (training/Makefile) reports the
archive's age.
"""

import os
import sys
import time
import json
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

# ─── Config ──────────────────────────────────────────────────────────────────

BASE_URL = "https://open-api-v3.coinglass.com"
ASSETS = ["BTC", "ETH", "SOL"]
# Default interval; override with --interval. "1d" writes the archives the
# rebuild actually consumes (see module docstring, [P287]).
INTERVAL = "4h"
VALID_INTERVALS = ("4h", "1d")
MAX_LIMIT = 4500  # max per request
RATE_LIMIT_SLEEP = 0.5  # seconds between requests

# How far back to fetch (in days). 2+ years covers DRL training window.
LOOKBACK_DAYS = 900

_TRAINING_DIR = Path(__file__).resolve().parent.parent   # training/
OUTPUT_DIR = _TRAINING_DIR / "training_data" / "coinglass_history"

# Verified working endpoints on v3 API (probed 2026-02-14)
ENDPOINTS = {
    "funding": {
        "path": "/api/futures/fundingRate/oi-weight-ohlc-history",
        "desc": "OI-weighted funding rate OHLC (aggregated across exchanges)",
        "symbol_type": "coin",  # uses BTC, ETH, SOL
        "exchange": None,
        "parser": "ohlc",
    },
    "oi": {
        "path": "/api/futures/openInterest/ohlc-aggregated-history",
        "desc": "Aggregated open interest OHLC (across exchanges)",
        "symbol_type": "coin",
        "exchange": None,
        "parser": "ohlc",
    },
    "liquidation": {
        "path": "/api/futures/liquidation/aggregated-history",
        "desc": "Aggregated liquidation data",
        "symbol_type": "coin",
        "exchange": None,
        "parser": "liquidation",
    },
}

# Symbol mapping
SYMBOL_MAP = {
    "BTC": {"pair": "BTCUSDT", "coin": "BTC"},
    "ETH": {"pair": "ETHUSDT", "coin": "ETH"},
    "SOL": {"pair": "SOLUSDT", "coin": "SOL"},
}


def get_api_key():
    load_dotenv()
    key = os.environ.get("COINGLASS_API_KEY", "")
    if not key:
        print("[ERROR] COINGLASS_API_KEY not found in .env")
        sys.exit(1)
    return key


def fetch_endpoint(api_key: str, endpoint_path: str, params: dict) -> dict:
    """Make a single API request to Coinglass v3."""
    url = f"{BASE_URL}{endpoint_path}"
    headers = {
        "CG-API-KEY": api_key,
        "Accept": "application/json",
    }

    resp = requests.get(url, headers=headers, params=params, timeout=30)

    if resp.status_code == 429:
        print(f"  [RATE LIMIT] Sleeping 10s...")
        time.sleep(10)
        resp = requests.get(url, headers=headers, params=params, timeout=30)

    if resp.status_code != 200:
        print(f"  [HTTP {resp.status_code}] {url}")
        print(f"  Response: {resp.text[:500]}")
        return {}

    result = resp.json()

    code = result.get("code")
    if code not in ("0", 0):
        print(f"  [API ERROR] code={code}, msg={result.get('msg')}")
        return {}

    return result


def fetch_paginated(
    api_key: str,
    endpoint_path: str,
    symbol: str,
    interval: str = "4h",
    lookback_days: int = LOOKBACK_DAYS,
    exchange: str = None,
    symbol_type: str = "coin",
) -> list:
    """Fetch data with pagination (oldest first)."""

    now = int(datetime.now(timezone.utc).timestamp())
    start = int((datetime.now(timezone.utc) - timedelta(days=lookback_days)).timestamp())

    sym = SYMBOL_MAP[symbol]["pair"] if symbol_type == "pair" else SYMBOL_MAP[symbol]["coin"]

    all_data = []
    current_start = start
    page = 0

    while current_start < now:
        params = {
            "symbol": sym,
            "interval": interval,
            "limit": MAX_LIMIT,
            "startTime": current_start,
            "endTime": now,
        }
        if exchange:
            params["exchange"] = exchange

        result = fetch_endpoint(api_key, endpoint_path, params)
        data = result.get("data", [])

        if not data:
            break

        # Handle different response formats
        if isinstance(data, list):
            all_data.extend(data)
        elif isinstance(data, dict):
            for key in ("dataMap", "list", "items"):
                if key in data and isinstance(data[key], list):
                    all_data.extend(data[key])
                    break
            else:
                all_data.append(data)

        page += 1
        batch_size = len(data) if isinstance(data, list) else 1

        if batch_size < MAX_LIMIT:
            break

        # Move start time forward to last timestamp + 1
        last = data[-1] if isinstance(data, list) else data
        if isinstance(last, dict) and "t" in last:
            current_start = int(last["t"]) + 1
        else:
            break

        print(f"    Page {page}: {batch_size} rows, total {len(all_data)}")
        time.sleep(RATE_LIMIT_SLEEP)

    return all_data


def parse_ohlc_data(raw_data: list, symbol: str, data_type: str) -> pd.DataFrame:
    """Parse OHLC response (funding rate or OI) into DataFrame."""
    if not raw_data:
        return pd.DataFrame()

    rows = []
    for item in raw_data:
        if not isinstance(item, dict):
            continue
        ts = item.get("t", 0)
        if ts == 0:
            continue
        # Detect seconds vs milliseconds
        if ts > 1e12:
            ts_pd = pd.Timestamp(ts, unit="ms", tz="UTC")
        else:
            ts_pd = pd.Timestamp(ts, unit="s", tz="UTC")

        row = {
            "timestamp": ts_pd,
            f"{data_type}_open": _to_float(item.get("o", 0)),
            f"{data_type}_high": _to_float(item.get("h", 0)),
            f"{data_type}_low": _to_float(item.get("l", 0)),
            f"{data_type}_close": _to_float(item.get("c", 0)),
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    df["asset"] = symbol
    return df


def parse_liquidation_data(raw_data: list, symbol: str) -> pd.DataFrame:
    """Parse liquidation response into DataFrame."""
    if not raw_data:
        return pd.DataFrame()

    rows = []
    for item in raw_data:
        if not isinstance(item, dict):
            continue
        ts = item.get("t", 0)
        if ts == 0:
            continue
        if ts > 1e12:
            ts_pd = pd.Timestamp(ts, unit="ms", tz="UTC")
        else:
            ts_pd = pd.Timestamp(ts, unit="s", tz="UTC")

        long_liq = _to_float(item.get("longLiquidationUsd", 0))
        short_liq = _to_float(item.get("shortLiquidationUsd", 0))
        total_liq = long_liq + short_liq

        row = {
            "timestamp": ts_pd,
            "long_liq_usd": long_liq,
            "short_liq_usd": short_liq,
            "total_liq_usd": total_liq,
            "liq_imbalance": (long_liq - short_liq) / total_liq if total_liq > 0 else 0.0,
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    df["asset"] = symbol
    return df


def merge_history(existing: "pd.DataFrame | None", new: "pd.DataFrame") -> "pd.DataFrame":
    """[P266] Union-by-timestamp merge — the archive must GROW.

    The CoinGlass API serves only ~180 days of depth for the liquidation/OI
    endpoints (measured 2026-08: 1080 4h rows whatever lookback is requested),
    and this script used to bare-overwrite the output parquet. Two
    consequences: the trainable history of `liq_imbalance` — the strict-window
    carrier of the external feature group (P256) and the basis derivflow is
    forward-testing — was capped at 180 rolling days FOREVER; and a re-fetch
    after a >180-day gap would have permanently lost the un-overlapping
    middle. Merging keeps every previously-captured row and lets the archive
    grow past the API's window with each re-fetch.

    New rows win on timestamp collision (the API may restate the most recent
    bars). Sorted, deduplicated, timestamp-typed.
    """
    if existing is None or existing.empty:
        merged = new.copy()
    else:
        existing = existing.copy()
        existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True)
        new = new.copy()
        new["timestamp"] = pd.to_datetime(new["timestamp"], utc=True)
        merged = pd.concat([existing, new], ignore_index=True)
    merged = (merged.drop_duplicates("timestamp", keep="last")
                    .sort_values("timestamp").reset_index(drop=True))
    return merged


def _to_float(val) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def main(interval: str = INTERVAL):
    if interval not in VALID_INTERVALS:
        print(f"[ERROR] interval must be one of {VALID_INTERVALS}, got {interval!r}")
        sys.exit(2)
    api_key = get_api_key()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("COINGLASS HISTORICAL DATA FETCHER (v3 API)")
    print(f"API Key: {api_key[:8]}...{api_key[-4:]}")
    print(f"Assets: {ASSETS}")
    print(f"Interval: {interval}")
    print(f"Lookback: {LOOKBACK_DAYS} days")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 70)

    # ─── Phase 1: Quick probe ─────────────────────────────────────────────────

    print("\n[PHASE 1] Probing endpoints...")
    for name, ep in ENDPOINTS.items():
        sym = SYMBOL_MAP["BTC"][ep["symbol_type"]]
        params = {"symbol": sym, "interval": interval, "limit": 2}
        if ep["exchange"]:
            params["exchange"] = ep["exchange"]
        result = fetch_endpoint(api_key, ep["path"], params)
        data = result.get("data", [])
        n = len(data) if isinstance(data, list) else "dict"
        sample = ""
        if isinstance(data, list) and data:
            sample = json.dumps(data[0])[:150]
        print(f"  {name:15s} [{n} items] {sample}")
        time.sleep(RATE_LIMIT_SLEEP)

    # ─── Phase 2: Fetch all data ──────────────────────────────────────────────

    print("\n[PHASE 2] Fetching historical data...")

    summary = {}

    for name, ep in ENDPOINTS.items():
        for asset in ASSETS:
            label = f"{asset}_{name}"
            print(f"\n  Fetching {label}...")

            raw = fetch_paginated(
                api_key=api_key,
                endpoint_path=ep["path"],
                symbol=asset,
                interval=interval,
                lookback_days=LOOKBACK_DAYS,
                exchange=ep.get("exchange"),
                symbol_type=ep.get("symbol_type", "coin"),
            )

            if not raw:
                print(f"    [EMPTY] No data for {label}")
                continue

            print(f"    Raw rows: {len(raw)}")

            # Parse
            if ep["parser"] == "ohlc":
                df = parse_ohlc_data(raw, asset, name)
            elif ep["parser"] == "liquidation":
                df = parse_liquidation_data(raw, asset)
            else:
                df = parse_ohlc_data(raw, asset, name)

            if df.empty:
                print(f"    [PARSE FAILED] Could not parse {label}")
                continue

            # Save — [P266] MERGE with the existing archive (never overwrite;
            # see merge_history), atomic write so a crash cannot truncate it.
            out_path = OUTPUT_DIR / f"{asset}_{name}_{interval}.parquet"
            existing = None
            if out_path.exists():
                try:
                    existing = pd.read_parquet(out_path)
                except Exception as e:
                    print(f"    [WARN] existing archive unreadable "
                          f"({type(e).__name__}: {e}) — keeping a .bak copy")
                    out_path.rename(out_path.with_suffix(".parquet.bak"))
            n_old = 0 if existing is None else len(existing)
            df = merge_history(existing, df)
            tmp_path = out_path.with_suffix(".parquet.tmp")
            df.to_parquet(tmp_path, index=False)
            os.replace(tmp_path, out_path)

            date_range = f"{df['timestamp'].min()} -> {df['timestamp'].max()}"
            print(f"    Saved: {out_path} ({n_old} existing + fetch -> "
                  f"{len(df)} rows, {date_range})")

            summary[label] = {
                "rows": len(df),
                "start": str(df["timestamp"].min()),
                "end": str(df["timestamp"].max()),
                "columns": list(df.columns),
            }

            time.sleep(RATE_LIMIT_SLEEP)

    # ─── Phase 3: Summary ────────────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for label, info in summary.items():
        print(f"  {label:25s}  {info['rows']:>6} rows  {info['start']} -> {info['end']}")

    # Total data points
    total_rows = sum(int(info["rows"]) for info in summary.values())
    print(f"\n  Total: {total_rows} data points across {len(summary)} files")

    # Save summary (interval-suffixed so a 1d run cannot clobber the 4h
    # record; the legacy un-suffixed name stays for 4h)
    summary_path = OUTPUT_DIR / (
        "fetch_summary.json" if interval == "4h"
        else f"fetch_summary_{interval}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary: {summary_path}")

    # ─── Phase 4: Sanity check ───────────────────────────────────────────────

    print("\n[SANITY CHECK]")
    for asset in ASSETS:
        files = list(OUTPUT_DIR.glob(f"{asset}_*.parquet"))
        if files:
            print(f"  {asset}: {len(files)} files")
            for f in files:
                df = pd.read_parquet(f)
                non_null = df.notna().sum()
                print(f"    {f.name}: {len(df)} rows, cols={list(df.columns)}")
                # Check for NaN
                nan_pct = df.isna().mean() * 100
                nan_cols = {c: f"{v:.1f}%" for c, v in nan_pct.items() if v > 0}
                if nan_cols:
                    print(f"      NaN: {nan_cols}")
        else:
            print(f"  {asset}: NO FILES")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--interval", choices=list(VALID_INTERVALS),
                    default=INTERVAL,
                    help="candle interval to fetch/merge; '1d' writes the "
                         "archives rebuild_pipeline consumes ([P287])")
    args = ap.parse_args()
    main(interval=args.interval)
