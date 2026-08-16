#!/usr/bin/env python3
"""[P266] Report the AGE of every training-data archive, with thresholds.

Invoked by `make check` (training/Makefile). Exists because the CoinGlass
liquidation/OI archive is the ONLY store of a series the API serves just
~180 days of — a stale archive silently caps the trainable history of
`liq_imbalance` (the external group's strict-window carrier, P256), and a
gap longer than the API window loses the middle permanently. The other
archives merely lag their public sources (Binance monthly archives), which
is recoverable; CoinGlass staleness is not.

Exit codes: 0 = all fresh; 1 = warnings (stale but recoverable or
approaching a loss window). Never blocks — this is a reporter.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TD = REPO / "training" / "training_data"

# (glob, warn_age_days, why)
CHECKS = [
    ("raw/*_60m.parquet", 45,
     "Binance MONTHLY archives: ~31d lag is structural; beyond ~45d "
     "fetch_binance_full.py has not been re-run (recoverable)"),
    ("coinglass_history/*_liquidation_4h.parquet", 150,
     "CoinGlass API depth is ~180d; beyond 150d since last fetch you are "
     "approaching PERMANENT history loss — run "
     "scripts/fetch_coinglass_history.py (it merges, P266)"),
    # [P287] The 1d oi/liquidation archives are what rebuild_pipeline
    # actually READS (_load_coinglass_daily); the 4h files above feed
    # nothing on the training path. Before P287 no command could even
    # refresh these — watch them with the same permanent-loss urgency.
    ("coinglass_history/*_liquidation_1d.parquet", 150,
     "the archives rebuild_pipeline CONSUMES; CoinGlass depth ~180d — run "
     "scripts/fetch_coinglass_history.py --interval 1d (merges, P287)"),
    ("coinglass_history/*_oi_1d.parquet", 150,
     "the archives rebuild_pipeline CONSUMES; CoinGlass depth ~180d — run "
     "scripts/fetch_coinglass_history.py --interval 1d (merges, P287)"),
    ("coinglass_history/*_funding_1d.parquet", 60,
     "Binance funding via vision archives — re-run "
     "scripts/fetch_binance_funding.py with the monthly refresh"),
    ("drl_training/*_4H_full.parquet", 60,
     "feature parquets older than the raw data they derive from — re-run "
     "scripts/rebuild_pipeline.py (fv2 included since P266)"),
    ("drl_training/*_4H_ohlcv.parquet", 21,
     "the analytics/scorer price series — scripts/refresh_ohlcv_4h.py "
     "extends past the monthly archives via DAILY archives since P266, so "
     "this should never be more than ~2 days behind after a run"),
]


def newest_data_ts(path: Path):
    """Newest timestamp INSIDE the file (coverage end), not file mtime —
    mtime says when we fetched, coverage says what we have."""
    try:
        import pandas as pd
        df = pd.read_parquet(path, columns=None)
        col = "timestamp" if "timestamp" in df.columns else None
        s = df[col] if col else df.index.to_series()
        ts = pd.to_datetime(pd.Series(s).max(), utc=True)
        return ts.to_pydatetime()
    except Exception as e:  # noqa: silent-swallow — a reporter must report, not crash; unreadable is its own finding
        return e


def main() -> int:
    now = datetime.now(timezone.utc)
    warned = False
    print("--- data freshness (coverage end vs now) ---")
    for pattern, warn_days, why in CHECKS:
        files = sorted(TD.glob(pattern))
        if not files:
            print(f"  !! {pattern}: NO FILES — {why}")
            warned = True
            continue
        worst = None
        for f in files:
            ts = newest_data_ts(f)
            if isinstance(ts, Exception):
                print(f"  !! {f.name}: unreadable ({type(ts).__name__})")
                warned = True
                continue
            age = (now - ts).total_seconds() / 86400.0
            if worst is None or age > worst[1]:
                worst = (f.name, age)
        if worst is None:
            continue
        name, age = worst
        mark = "!!" if age > warn_days else "ok"
        if age > warn_days:
            warned = True
        print(f"  {mark} {pattern}: oldest coverage end {age:5.1f}d ago "
              f"(warn at {warn_days}d) [{name}]")
        if age > warn_days:
            print(f"       -> {why}")
    return 1 if warned else 0


if __name__ == "__main__":
    sys.exit(main())
