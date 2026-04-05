#!/usr/bin/env python3
"""
Validate external data (Coinglass + Futures) before DRL pipeline integration.

Checks:
  - File existence
  - Timestamp sanity (2015-2027)
  - Missing values < 10% per column
  - No extreme outliers (z > 20)
  - Data type correctness
  - Corrupted rows

Usage:
    python scripts/validate_external_data.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

COINGLASS_DIR = PROJECT_ROOT / "data" / "coinglass_history"
FUTURES_DIR = PROJECT_ROOT / "training" / "training_data" / "futures"

ASSETS = ["BTC", "ETH", "SOL"]
COINGLASS_TYPES = ["funding", "oi", "liquidation"]

errors = []
warnings = []


def check(condition: bool, msg: str, level: str = "ERROR"):
    if not condition:
        if level == "ERROR":
            errors.append(msg)
            print(f"  [FAIL] {msg}")
        else:
            warnings.append(msg)
            print(f"  [WARN] {msg}")
    else:
        print(f"  [ OK ] {msg}")


def validate_coinglass():
    print("\n=== COINGLASS 1D DATA ===")
    for asset in ASSETS:
        for dtype in COINGLASS_TYPES:
            fname = f"{asset}_{dtype}_1d.parquet"
            fpath = COINGLASS_DIR / fname
            check(fpath.exists(), f"{fname} exists")
            if not fpath.exists():
                continue

            df = pd.read_parquet(fpath)
            check(len(df) > 1000, f"{fname}: {len(df)} rows (need >1000)")

            # Timestamp sanity
            ts = pd.to_datetime(df["timestamp"])
            check(ts.min().year >= 2015, f"{fname}: min year={ts.min().year} (need >=2015)")
            check(ts.max().year <= 2027, f"{fname}: max year={ts.max().year} (need <=2027)")

            # Missing values
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                nan_pct = df[col].isna().mean()
                check(nan_pct < 0.10, f"{fname}.{col}: NaN={nan_pct:.1%} (need <10%)")

            # Outliers (z > 20)
            for col in numeric_cols:
                vals = df[col].dropna()
                if len(vals) < 10:
                    continue
                std = vals.std()
                if std > 0:
                    z = ((vals - vals.mean()) / std).abs()
                    n_extreme = (z > 20).sum()
                    check(n_extreme == 0,
                          f"{fname}.{col}: {n_extreme} rows with |z|>20",
                          level="WARN" if n_extreme < 5 else "ERROR")


def validate_futures():
    print("\n=== FUTURES DAILY DATA ===")
    expected_numeric = [
        "open", "high", "low", "close", "volume",
        "volume_from", "marketorder_volume", "marketorder_volume_from", "tradecount",
    ]

    for asset in ASSETS:
        fname = f"{asset}_futures_daily.parquet"
        fpath = FUTURES_DIR / fname
        check(fpath.exists(), f"{fname} exists")
        if not fpath.exists():
            continue

        df = pd.read_parquet(fpath)
        check(len(df) > 1000, f"{fname}: {len(df)} rows (need >1000)")

        # Data type check (known issue: stored as object)
        for col in expected_numeric:
            if col in df.columns:
                is_numeric = df[col].dtype in [np.float64, np.float32, np.int64]
                if not is_numeric:
                    check(False,
                          f"{fname}.{col}: dtype={df[col].dtype} (need numeric). "
                          "Pipeline will auto-convert with pd.to_numeric().",
                          level="WARN")
                    # Try conversion
                    converted = pd.to_numeric(df[col], errors="coerce")
                    nan_after = converted.isna().sum() - df[col].isna().sum()
                    check(nan_after < 10,
                          f"{fname}.{col}: {nan_after} rows fail numeric conversion",
                          level="WARN" if nan_after < 10 else "ERROR")

        # Timestamp + corrupted rows
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        bad_rows = (ts < "2000-01-01") | ts.isna()
        n_bad = bad_rows.sum()
        check(n_bad < 10,
              f"{fname}: {n_bad} corrupted timestamp rows (will be filtered)",
              level="WARN" if n_bad < 10 else "ERROR")

        ts_clean = ts[~bad_rows]
        if len(ts_clean) > 0:
            check(ts_clean.min().year >= 2015,
                  f"{fname}: min year={ts_clean.min().year}")
            check(ts_clean.max().year <= 2027,
                  f"{fname}: max year={ts_clean.max().year}")


def main():
    print("HMATS External Data Validation")
    print("=" * 50)

    validate_coinglass()
    validate_futures()

    print("\n" + "=" * 50)
    print(f"RESULTS: {len(errors)} errors, {len(warnings)} warnings")

    if errors:
        print("\nBLOCKING ERRORS:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    elif warnings:
        print("\nNon-blocking warnings (handled by pipeline):")
        for w in warnings:
            print(f"  - {w}")
        print("\nDATA VALIDATED - safe to proceed with pipeline")
    else:
        print("\nALL CHECKS PASSED - data is clean")


if __name__ == "__main__":
    main()
