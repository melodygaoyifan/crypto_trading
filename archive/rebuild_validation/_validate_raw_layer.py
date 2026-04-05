"""① Raw Data Layer Validation - OHLCV + External + NaN audit."""
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT = Path(r"c:\Users\melod\Downloads\hmats")
DRL_DIR = PROJECT / "data" / "drl_training"
CG_DIR = PROJECT / "data" / "coinglass_history"
FUTURES_DIR = PROJECT / "hmats_training_v5 (1)" / "training_data" / "futures"

ASSETS = ["BTC", "ETH", "SOL"]
SEP = "=" * 70
PASS = "PASS"
FAIL = "FAIL"
results = []  # (section, check, status, detail)

def log(section, check, ok, detail=""):
    status = PASS if ok else FAIL
    results.append((section, check, status, detail))
    marker = "✓" if ok else "✗"
    print(f"  {marker} {check}" + (f"  - {detail}" if detail else ""))

# ═══════════════════════════════════════════════════════════
# SECTION A: OHLCV TIMESTAMPS
# ═══════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("SECTION A: OHLCV TIMESTAMPS")
print(SEP)

for asset in ASSETS:
    path = DRL_DIR / f"{asset}_4H_full.parquet"
    df = pd.read_parquet(path)
    ts = pd.to_datetime(df["timestamp"])
    n = len(df)

    print(f"\n  {asset} ({n:,} rows)")
    print(f"  {'─' * 50}")

    # Year range 2017-2026 (no corrupt timestamps)
    years = ts.dt.year.unique()
    bad_years = [y for y in years if y < 2015 or y > 2027]
    log("A", f"{asset} year range {years.min()}-{years.max()}", len(bad_years) == 0,
        f"BAD YEARS: {bad_years}" if bad_years else "")

    # Corrupt timestamp check (year=57086 etc.)
    corrupt = ts[ts.dt.year > 2030]
    log("A", f"{asset} no corrupt timestamps", len(corrupt) == 0,
        f"{len(corrupt)} corrupt rows" if len(corrupt) > 0 else "")

    # 4H interval consistency
    diffs = ts.diff().dropna()
    expected_4h = pd.Timedelta(hours=4)
    non_4h = diffs[diffs != expected_4h]
    # Allow small number of gaps (exchange downtime etc.) but no duplicates
    duplicates = diffs[diffs == pd.Timedelta(0)]
    log("A", f"{asset} no duplicate bars", len(duplicates) == 0,
        f"{len(duplicates)} duplicates" if len(duplicates) > 0 else "")

    gaps = non_4h[non_4h > expected_4h]
    # Count gaps > 8h (missed bars)
    big_gaps = gaps[gaps > pd.Timedelta(hours=8)]
    log("A", f"{asset} 4H interval (gaps>8h: {len(big_gaps)})", len(big_gaps) < 50,
        f"max gap: {gaps.max()}" if len(gaps) > 0 else "no gaps")

    # SOL start check
    if asset == "SOL":
        pre_2020 = ts[ts.dt.year < 2020]
        log("A", f"SOL starts 2020-08 (pre-2020 rows: {len(pre_2020)})", len(pre_2020) == 0)

    # Row count
    if asset == "SOL":
        log("A", f"{asset} row count: {n:,}", 10000 <= n <= 15000, "expected ~12K")
    else:
        log("A", f"{asset} row count: {n:,}", 15000 <= n <= 22000, "expected ~18K")

# ═══════════════════════════════════════════════════════════
# SECTION B: OHLCV VALUES
# ═══════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("SECTION B: OHLCV VALUES")
print(SEP)

for asset in ASSETS:
    path = DRL_DIR / f"{asset}_4H_full.parquet"
    df = pd.read_parquet(path)

    print(f"\n  {asset}")
    print(f"  {'─' * 50}")

    # Non-zero, non-negative OHLC
    for col in ["open", "high", "low", "close"]:
        vals = df[col]
        n_zero = (vals == 0).sum()
        n_neg = (vals < 0).sum()
        n_nan = vals.isna().sum()
        log("B", f"{asset} {col}: non-zero, non-neg",
            n_zero == 0 and n_neg == 0 and n_nan == 0,
            f"zero={n_zero}, neg={n_neg}, NaN={n_nan}")

    # high >= max(open, close)
    h_violation = (df["high"] < df[["open", "close"]].max(axis=1)).sum()
    log("B", f"{asset} high >= max(open,close)", h_violation == 0,
        f"{h_violation} violations" if h_violation > 0 else "")

    # low <= min(open, close)
    l_violation = (df["low"] > df[["open", "close"]].min(axis=1)).sum()
    log("B", f"{asset} low <= min(open,close)", l_violation == 0,
        f"{l_violation} violations" if l_violation > 0 else "")

    # volume > 0
    v_zero = (df["volume"] <= 0).sum()
    log("B", f"{asset} volume > 0", v_zero == 0,
        f"{v_zero} zero/neg volume bars" if v_zero > 0 else "")

    # No inf
    numeric = df.select_dtypes(include="number")
    n_inf = np.isinf(numeric.values).sum()
    log("B", f"{asset} no inf values", n_inf == 0,
        f"{n_inf} inf values" if n_inf > 0 else "")

# ═══════════════════════════════════════════════════════════
# SECTION C: EXTERNAL DATA FILE EXISTENCE
# ═══════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("SECTION C: EXTERNAL DATA FILE EXISTENCE")
print(SEP)

# 9 Coinglass daily files
print("\n  Coinglass Daily (9 files):")
cg_count = 0
for asset in ASSETS:
    for dtype in ["funding", "oi", "liquidation"]:
        path = CG_DIR / f"{asset}_{dtype}_1d.parquet"
        exists = path.exists()
        if exists:
            cg_count += 1
            df_cg = pd.read_parquet(path)
            log("C", f"{asset}_{dtype}_1d.parquet ({len(df_cg)} rows)", True)
        else:
            log("C", f"{asset}_{dtype}_1d.parquet", False, "MISSING")

log("C", f"Coinglass daily total: {cg_count}/9", cg_count == 9)

# 3 Futures daily files
print("\n  Futures Daily (3 files):")
fut_count = 0
for asset in ASSETS:
    path = FUTURES_DIR / f"{asset}_futures_daily.parquet"
    exists = path.exists()
    if exists:
        fut_count += 1
        df_f = pd.read_parquet(path)
        log("C", f"{asset}_futures_daily.parquet ({len(df_f)} rows)", True)
    else:
        log("C", f"{asset}_futures_daily.parquet", False, "MISSING")

log("C", f"Futures daily total: {fut_count}/3", fut_count == 3)

# ═══════════════════════════════════════════════════════════
# SECTION D: EXTERNAL DATA QUALITY
# ═══════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("SECTION D: EXTERNAL DATA QUALITY")
print(SEP)

# Coinglass timestamp + NaN + outlier checks
for asset in ASSETS:
    print(f"\n  {asset} Coinglass:")
    for dtype in ["funding", "oi", "liquidation"]:
        path = CG_DIR / f"{asset}_{dtype}_1d.parquet"
        if not path.exists():
            continue
        df_cg = pd.read_parquet(path)

        # Find timestamp col
        ts_col = None
        for c in ["timestamp", "time", "date", "t", "createTime"]:
            if c in df_cg.columns:
                ts_col = c
                break

        if ts_col:
            ts = pd.to_datetime(df_cg[ts_col], errors="coerce")
            ts_valid = ts.dropna()
            if len(ts_valid) > 0:
                y_min, y_max = ts_valid.dt.year.min(), ts_valid.dt.year.max()
                in_range = 2015 <= y_min and y_max <= 2027
                log("D", f"{asset}_{dtype}_1d ts: {y_min}-{y_max}", in_range)
            else:
                log("D", f"{asset}_{dtype}_1d timestamps", False, "NO VALID TIMESTAMPS")

        # NaN per column
        nan_pct = df_cg.select_dtypes(include="number").isna().mean()
        high_nan = nan_pct[nan_pct > 0.10]
        log("D", f"{asset}_{dtype}_1d NaN<10%", len(high_nan) == 0,
            f"high NaN cols: {dict(high_nan.round(3))}" if len(high_nan) > 0 else "")

    # Futures
    print(f"\n  {asset} Futures:")
    path = FUTURES_DIR / f"{asset}_futures_daily.parquet"
    if path.exists():
        df_f = pd.read_parquet(path)

        # Timestamp
        ts_col = None
        for c in ["timestamp", "time", "date", "t", "createTime"]:
            if c in df_f.columns:
                ts_col = c
                break

        if ts_col:
            ts = pd.to_datetime(df_f[ts_col], errors="coerce")
            ts_valid = ts.dropna()
            if len(ts_valid) > 0:
                y_min, y_max = ts_valid.dt.year.min(), ts_valid.dt.year.max()
                # Allow 1970 dates (known issue, filtered by pipeline)
                real_ts = ts_valid[ts_valid.dt.year >= 2000]
                if len(real_ts) > 0:
                    y_min_real = real_ts.dt.year.min()
                    y_max_real = real_ts.dt.year.max()
                    n_1970 = (ts_valid.dt.year < 2000).sum()
                    in_range = 2015 <= y_min_real and y_max_real <= 2027
                    log("D", f"{asset}_futures ts: {y_min_real}-{y_max_real}", in_range,
                        f"({n_1970} rows with year<2000, filtered by pipeline)" if n_1970 > 0 else "")

        # NaN
        for key in ["marketorder_volume", "tradecount", "marketorder_volume_from"]:
            if key in df_f.columns:
                vals = pd.to_numeric(df_f[key], errors="coerce")
                nan_pct_val = vals.isna().mean()
                log("D", f"{asset}_futures {key} NaN={nan_pct_val:.1%}", nan_pct_val < 0.10)

# Outlier check on derived features (z > 20)
print(f"\n  Derived Feature Outlier Check:")
EXT_ZSCORE_COLS = ["funding_rate_zscore", "taker_ratio_zscore", "tradecount_zscore"]
EXT_CLIP_COLS = ["oi_change_5d", "taker_vol_momentum", "liq_imbalance"]
for asset in ASSETS:
    df = pd.read_parquet(DRL_DIR / f"{asset}_4H_full.parquet")
    for col in EXT_ZSCORE_COLS:
        if col in df.columns:
            out = ((df[col].abs() > 20)).sum()
            log("D", f"{asset} {col} |z|>20: {out}", out == 0)
    for col in EXT_CLIP_COLS:
        if col in df.columns:
            out = ((df[col].abs() > 20)).sum()
            log("D", f"{asset} {col} |val|>20: {out}", out == 0)

# ═══════════════════════════════════════════════════════════
# SECTION E: NaN AUDIT
# ═══════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("SECTION E: NaN AUDIT (Final DRL Parquets)")
print(SEP)

for asset in ASSETS:
    path = DRL_DIR / f"{asset}_4H_full.parquet"
    df = pd.read_parquet(path)
    numeric = df.select_dtypes(include="number")

    print(f"\n  {asset}")
    print(f"  {'─' * 50}")

    # Total NaN
    nan_total = numeric.isna().sum().sum()
    log("E", f"{asset} total NaN across all numeric cols", nan_total == 0,
        f"{nan_total} NaN" if nan_total > 0 else "")

    # Per-column NaN
    nan_per_col = numeric.isna().sum()
    nan_cols = nan_per_col[nan_per_col > 0]
    if len(nan_cols) > 0:
        for col, count in nan_cols.items():
            log("E", f"  {col}: {count} NaN", False)

    # First row check (rolling edge case)
    first_row_nan = numeric.iloc[0].isna().sum()
    log("E", f"{asset} first row NaN count", first_row_nan == 0,
        f"{first_row_nan} NaN in row 0" if first_row_nan > 0 else "")

    # Last row check
    last_row_nan = numeric.iloc[-1].isna().sum()
    log("E", f"{asset} last row NaN count", last_row_nan == 0,
        f"{last_row_nan} NaN in last row" if last_row_nan > 0 else "")

    # z-score columns: check first 30 rows (rolling warmup)
    zscore_cols = [c for c in df.columns if "zscore" in c.lower()]
    for zc in zscore_cols:
        first_30_nan = df[zc].iloc[:30].isna().sum()
        if first_30_nan > 0:
            log("E", f"{asset} {zc} first-30-rows NaN: {first_30_nan}", False,
                "rolling warmup not filled")
    if zscore_cols:
        all_z_first30_ok = all(df[zc].iloc[:30].isna().sum() == 0 for zc in zscore_cols)
        log("E", f"{asset} all z-score cols: first 30 rows filled", all_z_first30_ok)

    # Inf check
    n_inf = np.isinf(numeric.values).sum()
    log("E", f"{asset} no inf in final parquet", n_inf == 0)

# ═══════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("SUMMARY")
print(SEP)

n_pass = sum(1 for r in results if r[2] == PASS)
n_fail = sum(1 for r in results if r[2] == FAIL)
print(f"\n  Total checks: {len(results)}")
print(f"  PASS: {n_pass}")
print(f"  FAIL: {n_fail}")

if n_fail > 0:
    print(f"\n  FAILURES:")
    for section, check, status, detail in results:
        if status == FAIL:
            print(f"    [{section}] {check}" + (f"  - {detail}" if detail else ""))

print(f"\n{SEP}")
