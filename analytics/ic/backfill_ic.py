#!/usr/bin/env python3
"""
Backfill IC for price-driven agents using historical OHLCV.

Bootstraps an initial IC dataset from historical 4H bars without needing live
runtime. Limitations:
  - Only price-derivable signals can be backfilled (RSI, MACD, BB, momentum,
    mean-reversion z-score, simple regime classifier).
  - Sentiment/CRACK/Lead-Lag/funding-rate signals require live data → cannot
    be backfilled. They populate after the agent runs in shadow mode.

Usage:
    python analytics/ic/backfill_ic.py --asset BTC --days 180
    python analytics/ic/backfill_ic.py --all-assets --days 365

Output:
    logs/ic_signals/ic_signals_BACKFILL_{asset}.jsonl
    Then run:
        python analytics/ic/compute_ic.py --all-assets
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OHLCV_DIR = REPO / "training" / "training_data" / "drl_training"
IC_LOG_DIR = REPO / "logs" / "ic_signals"
IC_LOG_DIR.mkdir(parents=True, exist_ok=True)


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _zscore(series: pd.Series, window: int = 20) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std().replace(0, np.nan)
    return (series - mean) / std


def backfill_signals(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Compute price-derivable signals on historical OHLCV."""
    df = ohlcv.copy()
    df["rsi_14"] = _rsi(df["close"], 14)

    # Direction inferred from RSI (mean-reversion contrarian)
    df["quant_direction"] = 0.0
    df.loc[df["rsi_14"] < 30, "quant_direction"] = 0.7
    df.loc[df["rsi_14"] > 70, "quant_direction"] = -0.7
    df.loc[(df["rsi_14"] >= 30) & (df["rsi_14"] < 45), "quant_direction"] = 0.3
    df.loc[(df["rsi_14"] > 55) & (df["rsi_14"] <= 70), "quant_direction"] = -0.3

    # Momentum-based direction (trend follow)
    df["ret_24"] = df["close"].pct_change(6)  # 6 × 4H = 24h
    df["mom_z"] = _zscore(df["ret_24"], window=20)
    df["momentum_direction"] = np.tanh(df["mom_z"] * 0.5)

    # Mean-reversion direction (z-score of deviation)
    df["price_z"] = _zscore(df["close"], window=20)
    df["meanrev_direction"] = -np.tanh(df["price_z"] * 0.5)

    # BB position
    bb_mid = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    df["bb_position"] = (df["close"] - bb_mid) / (2 * bb_std.replace(0, np.nan))
    df["bb_direction"] = -np.tanh(df["bb_position"] * 1.5)

    return df


def asset_to_records(df: pd.DataFrame, asset: str) -> List[dict]:
    """Convert backfill rows to ic_signals JSONL records."""
    out: List[dict] = []
    for _, row in df.iterrows():
        if pd.isna(row.get("rsi_14")):
            continue
        ts = row["timestamp"]
        if not isinstance(ts, datetime):
            try:
                ts = pd.to_datetime(ts).to_pydatetime()
            except Exception:
                continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        record = {
            "timestamp": ts.isoformat(),
            "asset": asset,
            "bar_id": f"{asset}_{ts.strftime('%Y-%m-%dT%H:%M')}",
            "current_price": float(row["close"]),
            "agent_signals": {
                "quant_backfill": {
                    "direction": float(row["quant_direction"])
                    if not pd.isna(row["quant_direction"]) else 0.0,
                    "rsi": float(row["rsi_14"])
                    if not pd.isna(row["rsi_14"]) else 0.0,
                },
                "momentum_backfill": {
                    "direction": float(row["momentum_direction"])
                    if not pd.isna(row["momentum_direction"]) else 0.0,
                    "z_score": float(row["mom_z"])
                    if not pd.isna(row["mom_z"]) else 0.0,
                },
                "meanrev_backfill": {
                    "direction": float(row["meanrev_direction"])
                    if not pd.isna(row["meanrev_direction"]) else 0.0,
                    "z_score": float(row["price_z"])
                    if not pd.isna(row["price_z"]) else 0.0,
                },
                "bb_backfill": {
                    "direction": float(row["bb_direction"])
                    if not pd.isna(row["bb_direction"]) else 0.0,
                    "position": float(row["bb_position"])
                    if not pd.isna(row["bb_position"]) else 0.0,
                },
            },
            "regime": {"phase": "BACKFILL"},
            "fusion_output": {
                "final_direction": float(row["quant_direction"])
                if not pd.isna(row["quant_direction"]) else 0.0
            },
            "future_returns": None,
            "_backfill": True,
        }
        out.append(record)
    return out


def backfill_one(asset: str, days: int) -> int:
    parquet_path = OHLCV_DIR / f"{asset}_4H_full.parquet"
    if not parquet_path.exists():
        print(f"[{asset}] OHLCV missing: {parquet_path}", file=sys.stderr)
        return 0
    df = pd.read_parquet(parquet_path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    elif df.index.name == "timestamp":
        df = df.reset_index()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    cutoff = df["timestamp"].max() - pd.Timedelta(days=days)
    df = df[df["timestamp"] >= cutoff].sort_values("timestamp")
    print(f"[{asset}] backfilling {len(df)} bars ({days} days)")
    df = backfill_signals(df)
    records = asset_to_records(df, asset)
    out = IC_LOG_DIR / f"ic_signals_BACKFILL_{asset}.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, default=str) + "\n")
        # [P103 2026-04-27] P37 family: flush before close so SIGKILL
        # mid-backfill doesn't truncate the JSONL. The `with` block does
        # close cleanly, but explicit flush() inside the loop ensures
        # OS buffers are emptied and downstream readers see complete
        # lines, not half-written ones, even on partial-completion.
        fh.flush()
    print(f"[{asset}] wrote {len(records)} records to {out}")
    return len(records)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--asset", default=None)
    p.add_argument("--all-assets", action="store_true")
    p.add_argument("--days", type=int, default=180)
    args = p.parse_args()

    assets = ["BTC", "ETH", "SOL"] if args.all_assets else [args.asset or "BTC"]
    total = 0
    for a in assets:
        total += backfill_one(a, args.days)
    print(f"\nTotal backfill records: {total}")
    print("\nNext: python analytics/ic/compute_ic.py --all-assets")


if __name__ == "__main__":
    main()
