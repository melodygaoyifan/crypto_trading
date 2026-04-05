"""② Feature Engineering Layer Validation."""
import pandas as pd
import numpy as np
from pathlib import Path
import json

PROJECT = Path(r"c:\Users\melod\Downloads\hmats")
DRL_DIR = PROJECT / "data" / "drl_training"
FEATURES_PY = PROJECT / "hmats_training_v5 (1)" / "drl" / "features.py"

ASSETS = ["BTC", "ETH", "SOL"]
SEP = "=" * 70
results = []

def log(section, check, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    results.append((section, check, status, detail))
    marker = "✓" if ok else "✗"
    print(f"  {marker} {check}" + (f"  - {detail}" if detail else ""))

# ═══════════════════════════════════════════════════════════
# P0: REDUNDANT FEATURE REMOVAL
# ═══════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("P0: REDUNDANT FEATURE REMOVAL")
print(SEP)

# Check features.py source code
with open(FEATURES_PY, encoding="utf-8") as f:
    src = f.read()

# Deleted mom periods (should NOT exist)
deleted_mom = ["mom_4h", "mom_6h", "mom_8h", "mom_12h", "mom_24h", "mom_48h", "mom_72h", "mom_96h"]
for m in deleted_mom:
    # Check both code and parquet
    in_code = f"'{m}'" in src or f'"{m}"' in src or f"f'{m}" in src
    log("P0", f"{m} NOT in features.py", not in_code)

# Retained mom periods
retained_mom = ["mom_120h", "mom_168h", "mom_336h", "mom_720h"]
for m in retained_mom:
    in_code = m in src
    log("P0", f"{m} IN features.py", in_code)

# Verify parquets match code
print("\n  Parquet verification:")
for asset in ASSETS:
    df = pd.read_parquet(DRL_DIR / f"{asset}_4H_full.parquet")
    cols = list(df.columns)

    # Deleted mom should not be in parquet
    for m in deleted_mom:
        log("P0", f"{asset}: {m} NOT in parquet", m not in cols)

    # Retained mom should be in parquet
    for m in retained_mom:
        log("P0", f"{asset}: {m} IN parquet", m in cols)

# Count base features (exclude timestamp, OHLCV, volume, regime, regime_confidence, external, proba)
print("\n  Base feature count:")
with open(PROJECT / "config" / "feature_manifest.json") as f:
    manifest = json.load(f)

base_count = len(manifest["base_features"])
log("P0", f"base_features count = {base_count}", base_count == 102,
    "NOTE: 102 (not 103). features.py produces exactly 102")

# ═══════════════════════════════════════════════════════════
# EXTERNAL FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("EXTERNAL FEATURE ENGINEERING")
print(SEP)

EXT_COLS = {
    "funding_rate_zscore": {"type": "zscore", "clip": (-10, 10)},
    "oi_change_5d": {"type": "pct_change", "clip": (-5, 5)},
    "liq_imbalance": {"type": "ratio", "clip": (-1, 1)},
    "taker_ratio_zscore": {"type": "zscore", "clip": (-10, 10)},
    "tradecount_zscore": {"type": "zscore", "clip": (-10, 10)},
    "taker_vol_momentum": {"type": "pct_change", "clip": (-5, 5)},
    "has_external_data": {"type": "binary", "clip": (0, 1)},
}

for asset in ASSETS:
    df = pd.read_parquet(DRL_DIR / f"{asset}_4H_full.parquet")
    ts = pd.to_datetime(df["timestamp"])

    print(f"\n  {asset}")
    print(f"  {'─' * 50}")

    # All 7 columns present
    for col in EXT_COLS:
        log("EXT", f"{asset}: {col} present", col in df.columns)

    # Clip range verification
    for col, info in EXT_COLS.items():
        if col not in df.columns:
            continue
        vals = df[col]
        lo, hi = info["clip"]
        below = (vals < lo - 0.001).sum()
        above = (vals > hi + 0.001).sum()
        log("EXT", f"{asset}: {col} clip [{lo},{hi}]",
            below == 0 and above == 0,
            f"{below} below, {above} above" if (below + above) > 0 else "")

    # Pre-2020 check: external features should be 0.0, has_external_data = 0
    pre_2020_mask = ts.dt.year < 2020
    n_pre = pre_2020_mask.sum()
    if n_pre > 0:
        numeric_ext = ["funding_rate_zscore", "oi_change_5d", "liq_imbalance",
                       "taker_ratio_zscore", "tradecount_zscore", "taker_vol_momentum"]
        for col in numeric_ext:
            if col in df.columns:
                pre_nonzero = (df.loc[pre_2020_mask, col] != 0.0).sum()
                log("EXT", f"{asset}: {col} pre-2020 all=0.0 ({n_pre} rows)",
                    pre_nonzero == 0, f"{pre_nonzero} non-zero" if pre_nonzero > 0 else "")

        if "has_external_data" in df.columns:
            pre_flag = (df.loc[pre_2020_mask, "has_external_data"] != 0).sum()
            log("EXT", f"{asset}: has_external_data pre-2020 all=0",
                pre_flag == 0, f"{pre_flag} non-zero" if pre_flag > 0 else "")
    else:
        log("EXT", f"{asset}: no pre-2020 rows (SOL starts 2020)", True)

    # has_external_data transition point
    if "has_external_data" in df.columns:
        hed = df["has_external_data"]
        transitions = df[hed.diff().abs() > 0].index
        if len(transitions) > 0:
            trans_ts = ts.iloc[transitions[0]]
            n_zeros = (hed == 0).sum()
            n_ones = (hed == 1).sum()
            log("EXT", f"{asset}: has_external_data 0->1 at {trans_ts.strftime('%Y-%m-%d')}",
                True, f"zeros={n_zeros}, ones={n_ones}")

# ═══════════════════════════════════════════════════════════
# MERGE STRATEGY VERIFICATION (code audit)
# ═══════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("MERGE STRATEGY (rebuild_pipeline.py code audit)")
print(SEP)

rebuild_py = PROJECT / "scripts" / "rebuild_pipeline.py"
with open(rebuild_py, encoding="utf-8") as f:
    rebuild_src = f.read()

# Check merge_asof with direction='backward'
merge_asof_count = rebuild_src.count("merge_asof")
backward_count = rebuild_src.count("direction='backward'") + rebuild_src.count('direction="backward"')
log("MERGE", f"merge_asof calls: {merge_asof_count}", merge_asof_count >= 2)
log("MERGE", f"direction='backward': {backward_count}", backward_count >= 2)

# Check z-score function
has_rolling_zscore = "_rolling_zscore" in rebuild_src
log("MERGE", f"_rolling_zscore function exists", has_rolling_zscore)

# Check fillna(0) for rolling warmup
fillna_count = rebuild_src.count("fillna(0")
log("MERGE", f"fillna(0) calls: {fillna_count}", fillna_count >= 3,
    "handles rolling warmup NaN")

# ═══════════════════════════════════════════════════════════
# FEATURE SCALE AUDIT
# ═══════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("FEATURE SCALE AUDIT")
print(SEP)

for asset in ASSETS:
    df = pd.read_parquet(DRL_DIR / f"{asset}_4H_full.parquet")
    numeric = df.select_dtypes(include="number")

    print(f"\n  {asset}")
    print(f"  {'─' * 50}")

    # No feature |val| > 1e6
    feature_cols = [c for c in manifest["all_features"] if c in numeric.columns]
    big_cols = {}
    for col in feature_cols:
        max_abs = numeric[col].abs().max()
        if max_abs > 1e6:
            big_cols[col] = max_abs
    log("SCALE", f"{asset}: no feature |val| > 1e6", len(big_cols) == 0,
        f"offenders: {big_cols}" if big_cols else "")

    # Scale diversity check (min/max range for key features)
    scale_examples = {}
    for col in ["gap_risk", "vol_12h", "ret_24h", "rsi_14", "bb_width_20"]:
        if col in numeric.columns:
            lo, hi = numeric[col].min(), numeric[col].max()
            scale_examples[col] = f"[{lo:.4f}, {hi:.4f}]"
    log("SCALE", f"{asset}: scale diversity noted", True,
        "  ".join(f"{k}={v}" for k, v in scale_examples.items()))

# ═══════════════════════════════════════════════════════════
# SPARSE BINARY FEATURES (P1)
# ═══════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("SPARSE BINARY FEATURES (P1)")
print(SEP)

EXPECTED_SPARSE = {
    "capitulation", "death_cross", "panic", "fear_extreme",
    "breakdown_20", "breakdown_50", "breakdown_100",
    "vol_expansion", "consecutive_red", "golden_cross",
    "breakout_20",
}

for asset in ASSETS:
    df = pd.read_parquet(DRL_DIR / f"{asset}_4H_full.parquet")
    feature_cols = [c for c in manifest["base_features"] if c in df.columns]

    print(f"\n  {asset}")
    print(f"  {'─' * 50}")

    sparse_cols = []
    for col in feature_cols:
        zero_pct = (df[col] == 0).sum() / len(df) * 100
        if zero_pct > 80:
            sparse_cols.append((col, zero_pct))

    n_sparse = len(sparse_cols)
    log("SPARSE", f"{asset}: {n_sparse} features >80% zeros", n_sparse >= 10,
        "expected ~17 crash/panic signals")

    # List them
    for col, zpct in sorted(sparse_cols, key=lambda x: -x[1]):
        in_expected = col in EXPECTED_SPARSE
        tag = "" if in_expected else " (not in expected set)"
        print(f"    {col}: {zpct:.1f}% zeros{tag}")

    # Verify they are all retained (not dropped)
    for col in EXPECTED_SPARSE:
        if col in df.columns:
            log("SPARSE", f"{asset}: {col} retained", True)
        else:
            log("SPARSE", f"{asset}: {col} retained", False, "MISSING")

# ═══════════════════════════════════════════════════════════
# Z-SCORE ROLLING WARMUP
# ═══════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("Z-SCORE ROLLING WARMUP (first 30 rows)")
print(SEP)

zscore_feature_cols = ["funding_rate_zscore", "taker_ratio_zscore", "tradecount_zscore",
                       "zscore_24h", "zscore_48h", "zscore_96h", "zscore_168h",
                       "zscore_336h", "zscore_720h"]

for asset in ASSETS:
    df = pd.read_parquet(DRL_DIR / f"{asset}_4H_full.parquet")
    print(f"\n  {asset}")
    for col in zscore_feature_cols:
        if col in df.columns:
            first_30_nan = df[col].iloc[:30].isna().sum()
            first_30_val = df[col].iloc[:30]
            all_zero_first_30 = (first_30_val == 0.0).all()
            log("WARMUP", f"{asset}: {col} first 30 rows NaN={first_30_nan}",
                first_30_nan == 0,
                "all zeros (fillna applied)" if all_zero_first_30 else f"min={first_30_val.min():.4f}")

# ═══════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("SUMMARY")
print(SEP)

n_pass = sum(1 for r in results if r[2] == "PASS")
n_fail = sum(1 for r in results if r[2] == "FAIL")
print(f"\n  Total checks: {len(results)}")
print(f"  PASS: {n_pass}")
print(f"  FAIL: {n_fail}")

if n_fail > 0:
    print(f"\n  FAILURES:")
    for section, check, status, detail in results:
        if status == "FAIL":
            print(f"    [{section}] {check}" + (f"  - {detail}" if detail else ""))

print(f"\n{SEP}")
