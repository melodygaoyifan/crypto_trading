"""Comprehensive validation: raw parquets + Coinglass source + futures source + derived features."""
import pandas as pd
import numpy as np
from pathlib import Path
import json

PROJECT = Path(r"c:\Users\melod\Downloads\hmats")
DRL_DIR = PROJECT / "data" / "drl_training"
CG_DIR = PROJECT / "data" / "coinglass_history"
FUTURES_DIR = PROJECT / "hmats_training_v5 (1)" / "training_data" / "futures"

ASSETS = ["BTC", "ETH", "SOL"]
SEP = "=" * 70

# ═══════════════════════════════════════════════════════════
# SECTION 1: Raw DRL Parquet Validation
# ═══════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("SECTION 1: RAW DRL PARQUET VALIDATION")
print(SEP)

for asset in ASSETS:
    path = DRL_DIR / f"{asset}_4H_full.parquet"
    if not path.exists():
        print(f"\n[FAIL] {asset}: FILE MISSING")
        continue

    df = pd.read_parquet(path)
    n = len(df)
    ts = pd.to_datetime(df["timestamp"])
    date_start = ts.iloc[0]
    date_end = ts.iloc[-1]

    # Row count
    if asset == "SOL":
        row_ok = 10000 <= n <= 15000
        expected = "~12K"
    else:
        row_ok = 15000 <= n <= 22000
        expected = "~18K"

    # NaN
    nan_total = df.select_dtypes(include="number").isna().sum().sum()

    # Year range
    years = ts.dt.year.unique()
    year_min, year_max = years.min(), years.max()
    year_ok = 2017 <= year_min and year_max <= 2026
    bad_years = [y for y in years if y < 2015 or y > 2027]

    # SOL start
    if asset == "SOL":
        sol_start_ok = year_min >= 2020
    else:
        sol_start_ok = True

    # Scale check
    numeric_cols = df.select_dtypes(include="number").columns
    big_cols = {}
    for col in numeric_cols:
        max_abs = df[col].abs().max()
        if max_abs > 1e6:
            big_cols[col] = max_abs

    # Timestamp gaps
    diffs = ts.diff().dropna()
    expected_delta = pd.Timedelta(hours=4)
    gaps = diffs[diffs != expected_delta]
    n_gaps = len(gaps)
    max_gap = gaps.max() if n_gaps > 0 else pd.Timedelta(0)

    print(f"\n{'─' * 50}")
    print(f"  {asset}_4H_full.parquet")
    print(f"{'─' * 50}")
    print(f"  Rows:       {n:,} (expected {expected})  {'✓' if row_ok else '✗'}")
    print(f"  Date range: {date_start.strftime('%Y-%m-%d')} -> {date_end.strftime('%Y-%m-%d')}")
    print(f"  Year range: {year_min}-{year_max}  {'✓' if year_ok else '✗ BAD YEARS: ' + str(bad_years)}")
    if asset == "SOL":
        print(f"  SOL start:  {year_min} >= 2020  {'✓' if sol_start_ok else '✗'}")
    print(f"  NaN total:  {nan_total}  {'✓' if nan_total == 0 else '✗'}")
    print(f"  Columns:    {len(df.columns)}")
    print(f"  4H gaps:    {n_gaps} non-4H intervals (max gap: {max_gap})")
    if big_cols:
        print(f"  |val| > 1e6: {len(big_cols)} columns:")
        for col, val in sorted(big_cols.items(), key=lambda x: -x[1])[:5]:
            print(f"    {col}: max={val:.2e}")
    else:
        print(f"  |val| > 1e6: None ✓")

# ═══════════════════════════════════════════════════════════
# SECTION 2: COINGLASS SOURCE DATA
# ═══════════════════════════════════════════════════════════
print(f"\n\n{SEP}")
print("SECTION 2: COINGLASS SOURCE DATA")
print(SEP)

cg_files = {
    "funding": {"key_col": "funding_close", "derives": "funding_rate_zscore"},
    "oi": {"key_col": "oi_close", "derives": "oi_change_5d"},
    "liquidation": {"key_col": None, "derives": "liq_imbalance"},
}

for asset in ASSETS:
    print(f"\n{'─' * 50}")
    print(f"  {asset} Coinglass Data")
    print(f"{'─' * 50}")

    for dtype, info in cg_files.items():
        # Check both 4h and 1d
        for interval in ["4h", "1d"]:
            path = CG_DIR / f"{asset}_{dtype}_{interval}.parquet"
            if not path.exists():
                print(f"  {dtype}_{interval}: MISSING")
                continue

            df = pd.read_parquet(path)
            cols = list(df.columns)
            n = len(df)

            # Find timestamp column
            ts_col = None
            for c in ["timestamp", "time", "date", "t"]:
                if c in df.columns:
                    ts_col = c
                    break

            if ts_col:
                ts = pd.to_datetime(df[ts_col], errors="coerce")
                ts_valid = ts.dropna()
                if len(ts_valid) > 0:
                    date_range = f"{ts_valid.iloc[0].strftime('%Y-%m-%d')} -> {ts_valid.iloc[-1].strftime('%Y-%m-%d')}"
                else:
                    date_range = "NO VALID TIMESTAMPS"
            else:
                date_range = f"NO ts col (cols: {cols[:5]})"

            # NaN check
            nan_pct = df.isna().mean()
            high_nan = nan_pct[nan_pct > 0.1]

            # Key column check
            key_col = info["key_col"]
            if key_col and key_col in df.columns:
                key_stats = f"min={df[key_col].min():.4f}, max={df[key_col].max():.4f}, NaN={df[key_col].isna().sum()}"
            elif key_col:
                # Try to find similar column
                similar = [c for c in df.columns if key_col.split("_")[0] in c.lower()]
                key_stats = f"COLUMN MISSING (similar: {similar[:3]})"
            else:
                key_stats = "N/A (multiple columns)"

            print(f"  {dtype}_{interval}: {n:,} rows | {date_range}")
            print(f"    Columns: {cols}")
            if key_col:
                print(f"    {key_col}: {key_stats}")
            if len(high_nan) > 0:
                print(f"    High NaN (>10%): {dict(high_nan.round(3))}")

            # For liquidation, show long/short columns
            if dtype == "liquidation":
                for lc in df.columns:
                    if "long" in lc.lower() or "short" in lc.lower() or "liq" in lc.lower():
                        if df[lc].dtype in ["float64", "float32", "int64"]:
                            print(f"    {lc}: min={df[lc].min():.2f}, max={df[lc].max():.2f}")

# ═══════════════════════════════════════════════════════════
# SECTION 3: FUTURES DAILY SOURCE DATA
# ═══════════════════════════════════════════════════════════
print(f"\n\n{SEP}")
print("SECTION 3: FUTURES DAILY SOURCE DATA")
print(SEP)

for asset in ASSETS:
    path = FUTURES_DIR / f"{asset}_futures_daily.parquet"
    if not path.exists():
        print(f"\n  {asset}: MISSING at {path}")
        continue

    df = pd.read_parquet(path)
    cols = list(df.columns)
    n = len(df)

    # Find timestamp
    ts_col = None
    for c in ["timestamp", "time", "date", "t", "createTime"]:
        if c in df.columns:
            ts_col = c
            break

    if ts_col:
        ts = pd.to_datetime(df[ts_col], errors="coerce")
        ts_valid = ts.dropna()
        date_range = f"{ts_valid.iloc[0].strftime('%Y-%m-%d')} -> {ts_valid.iloc[-1].strftime('%Y-%m-%d')}" if len(ts_valid) > 0 else "NO VALID"
    else:
        date_range = f"NO ts col (cols: {cols[:8]})"

    print(f"\n{'─' * 50}")
    print(f"  {asset}_futures_daily.parquet: {n:,} rows | {date_range}")
    print(f"  Columns: {cols}")

    # Check for the key columns we need
    for key in ["marketorder_volume", "tradecount", "marketorder_volume_from"]:
        if key in df.columns:
            col_vals = pd.to_numeric(df[key], errors="coerce")
            n_nan = col_vals.isna().sum()
            dtype_str = str(df[key].dtype)
            print(f"    {key} (dtype={dtype_str}): min={col_vals.min():.2f}, max={col_vals.max():.2e}, NaN={n_nan}")
        else:
            similar = [c for c in df.columns if any(k in c.lower() for k in key.split("_")[:2])]
            print(f"    {key}: MISSING (similar: {similar[:5]})")

# ═══════════════════════════════════════════════════════════
# SECTION 4: DERIVED EXTERNAL FEATURES IN FINAL PARQUETS
# ═══════════════════════════════════════════════════════════
print(f"\n\n{SEP}")
print("SECTION 4: DERIVED EXTERNAL FEATURES IN FINAL PARQUETS")
print(SEP)

EXT_COLS = [
    "funding_rate_zscore", "oi_change_5d", "liq_imbalance",
    "taker_ratio_zscore", "tradecount_zscore", "taker_vol_momentum",
    "has_external_data",
]

for asset in ASSETS:
    path = DRL_DIR / f"{asset}_4H_full.parquet"
    df = pd.read_parquet(path)
    ts = pd.to_datetime(df["timestamp"])

    print(f"\n{'─' * 50}")
    print(f"  {asset} External Features")
    print(f"{'─' * 50}")

    for col in EXT_COLS:
        if col not in df.columns:
            print(f"  {col}: MISSING ✗")
            continue

        vals = df[col]
        n_nan = vals.isna().sum()
        n_zero = (vals == 0).sum()
        pct_zero = n_zero / len(vals) * 100

        if col == "has_external_data":
            n_ones = (vals == 1).sum()
            # Find transition point
            transitions = df[vals.diff().abs() > 0].index
            if len(transitions) > 0:
                trans_idx = transitions[0]
                trans_ts = ts.iloc[trans_idx]
                print(f"  {col}: 0->1 at row {trans_idx} ({trans_ts.strftime('%Y-%m-%d')})")
                print(f"    {n_zero} zeros ({pct_zero:.1f}%), {n_ones} ones ({n_ones/len(vals)*100:.1f}%)")
            else:
                print(f"  {col}: ALL SAME VALUE ({vals.iloc[0]})")
        else:
            print(f"  {col}: min={vals.min():.4f}, max={vals.max():.4f}, "
                  f"NaN={n_nan}, zeros={n_zero} ({pct_zero:.1f}%)")

            # Check pre-external region (has_external_data=0)
            if "has_external_data" in df.columns:
                pre_mask = df["has_external_data"] == 0
                pre_vals = vals[pre_mask]
                if len(pre_vals) > 0:
                    non_zero_pre = (pre_vals != 0).sum()
                    print(f"    Pre-external ({pre_mask.sum()} rows): non-zero={non_zero_pre} "
                          f"{'✓ all zero' if non_zero_pre == 0 else '✗ SHOULD BE ZERO'}")

            # Check clip ranges
            if "zscore" in col:
                out_of_range = ((vals < -10) | (vals > 10)).sum()
                print(f"    Clip [-10,10]: {out_of_range} out of range {'✓' if out_of_range == 0 else '✗'}")
            elif col in ["oi_change_5d", "taker_vol_momentum"]:
                out_of_range = ((vals < -5) | (vals > 5)).sum()
                print(f"    Clip [-5,5]: {out_of_range} out of range {'✓' if out_of_range == 0 else '✗'}")
            elif col == "liq_imbalance":
                out_of_range = ((vals < -1) | (vals > 1)).sum()
                print(f"    Clip [-1,1]: {out_of_range} out of range {'✓' if out_of_range == 0 else '✗'}")

print(f"\n{SEP}")
print("VALIDATION COMPLETE")
print(SEP)
