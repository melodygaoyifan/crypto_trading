"""④ Scaler Layer Validation."""
import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib

PROJECT = Path(r"c:\Users\melod\Downloads\hmats")
DRL_DIR = PROJECT / "data" / "drl_training"
MANIFEST = PROJECT / "config" / "feature_manifest.json"
SPLIT_MANIFEST = PROJECT / "config" / "split_manifest.json"

ASSETS = ["BTC", "ETH", "SOL"]
SEP = "=" * 70
results = []

def log(section, check, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    results.append((section, check, status, detail))
    marker = "✓" if ok else "✗"
    print(f"  {marker} {check}" + (f"  - {detail}" if detail else ""))

# Load manifests
with open(MANIFEST) as f:
    manifest = json.load(f)
with open(SPLIT_MANIFEST) as f:
    split_manifest = json.load(f)

all_features = manifest["all_features"]
no_scale = set(manifest.get("no_scale_features", []))
binary_features = set(manifest.get("binary_features", []))

# ═══════════════════════════════════════════════════════════
# CODE AUDIT: _scale_features() in train_drl_full.py
# ═══════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("CODE AUDIT: _scale_features()")
print(SEP)

# Read the source code to verify logic
src_path = PROJECT / "train_drl_full.py"
src = src_path.read_text(encoding="utf-8")

# Check 1: RobustScaler fit only on train_df
log("CODE", "scaler.fit(train_df[scale_cols]) - fit on TRAIN only",
    "scaler.fit(train_df[scale_cols])" in src)

# Check 2: Transform both train and val
log("CODE", "transform train_df AND val_df",
    "scaler.transform(train_df[scale_cols])" in src and
    "scaler.transform(val_df[scale_cols])" in src)

# Check 3: no_scale_features exclusion
log("CODE", "scale_cols excludes no_scale_features",
    "c not in self.no_scale_features" in src)

# Check 4: Post-scale clip
log("CODE", "post-scale clip [-10, 10]",
    "clip(-10.0, 10.0)" in src)

# Check 5: inf -> 0 handling
log("CODE", "inf -> 0 replacement",
    "[np.inf, -np.inf], 0" in src)

# Check 6: NaN -> 0 handling
log("CODE", "NaN -> 0 fill",
    "fillna(0)" in src)

# Check 7: scale_==0 -> 1.0 guard (zero-IQR features)
log("CODE", "zero IQR guard: scale_ == 0 -> 1.0",
    "np.where(scaler.scale_ == 0, 1.0, scaler.scale_)" in src)

# ═══════════════════════════════════════════════════════════
# NO_SCALE FEATURES CORRECTNESS
# ═══════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("NO_SCALE FEATURES")
print(SEP)

expected_no_scale = {f"regime_proba_{i}" for i in range(8)} | {"has_external_data"}
log("NOSCALE", f"no_scale_features = {len(no_scale)} cols", no_scale == expected_no_scale,
    str(sorted(no_scale)))

# Verify these ARE in all_features (they ARE obs inputs, just unscaled)
for col in sorted(expected_no_scale):
    log("NOSCALE", f"  {col} ∈ all_features", col in all_features)

# ═══════════════════════════════════════════════════════════
# SCALER ARTIFACT VALIDATION (from BTC fold_1)
# ═══════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("SCALER ARTIFACT (BTC fold_1)")
print(SEP)

btc_scaler_path = (PROJECT / "models" / "retrained" / "BTC" / "fold_1" /
                   "logs" / "ULTIMATE" / "feature_scaler.pkl")

if btc_scaler_path.exists():
    scaler_data = joblib.load(btc_scaler_path)

    # Must be a dict with scaler, scale_cols, pass_cols
    log("ARTIFACT", "scaler_data is dict", isinstance(scaler_data, dict))

    scaler = scaler_data["scaler"]
    scale_cols = scaler_data["scale_cols"]
    pass_cols = scaler_data["pass_cols"]

    log("ARTIFACT", f"scaler type = {type(scaler).__name__}",
        type(scaler).__name__ == "RobustScaler")
    log("ARTIFACT", f"scale_cols count = {len(scale_cols)}",
        len(scale_cols) == len(all_features) - len(no_scale),
        f"expected {len(all_features) - len(no_scale)}")
    log("ARTIFACT", f"pass_cols count = {len(pass_cols)}",
        len(pass_cols) == len(no_scale),
        f"expected {len(no_scale)}")

    # pass_cols should match no_scale
    pass_set = set(pass_cols)
    log("ARTIFACT", "pass_cols == no_scale_features", pass_set == expected_no_scale)

    # No pass_col appears in scale_cols (mutual exclusion)
    overlap = set(scale_cols) & pass_set
    log("ARTIFACT", "scale_cols ∩ pass_cols = ∅", len(overlap) == 0,
        str(overlap) if overlap else "")

    # scale_ array: no zeros (all handled)
    n_zero_scale = (scaler.scale_ == 0).sum()
    log("ARTIFACT", f"scale_ has no zeros (zero IQR handled)",
        n_zero_scale == 0, f"{n_zero_scale} zeros" if n_zero_scale > 0 else "")

    # scale_ array: no inf or NaN
    n_bad = np.isnan(scaler.scale_).sum() + np.isinf(scaler.scale_).sum()
    log("ARTIFACT", f"scale_ no NaN/inf", n_bad == 0)

    # center_ (median) sanity
    log("ARTIFACT", f"center_ shape = ({scaler.n_features_in_},)",
        scaler.center_.shape == (scaler.n_features_in_,))
else:
    log("ARTIFACT", "BTC fold_1 scaler exists", False, "MISSING - run BTC training first")

# ═══════════════════════════════════════════════════════════
# SIMULATION: Apply scaler to actual data, check distribution
# ═══════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("SIMULATION: Scale BTC fold_1 train/val")
print(SEP)

if btc_scaler_path.exists():
    df = pd.read_parquet(DRL_DIR / "BTC_4H_full.parquet")
    sm = split_manifest["assets"]["BTC"]
    fold_1 = sm["folds"][0]
    train_end = fold_1["train_end"]
    val_start = fold_1["val_start"]
    val_end = fold_1["val_end"]

    train_df = df.iloc[:train_end]
    val_df = df.iloc[val_start:val_end]

    scaler_data = joblib.load(btc_scaler_path)
    scaler = scaler_data["scaler"]
    scale_cols = scaler_data["scale_cols"]
    pass_cols = scaler_data["pass_cols"]

    # Verify fit is on train only: re-fit on train, compare centers
    from sklearn.preprocessing import RobustScaler as RS
    fresh_scaler = RS()
    fresh_scaler.fit(train_df[scale_cols])
    center_match = np.allclose(fresh_scaler.center_, scaler.center_, atol=1e-10)
    log("FIT", "scaler.center_ matches fresh fit on train_df",
        center_match, "proves fit was train-only")

    scale_match = True
    # Check scale_ matches (after zero-IQR guard)
    fresh_scale = np.where(fresh_scaler.scale_ == 0, 1.0, fresh_scaler.scale_)
    saved_scale = scaler.scale_
    scale_close = np.allclose(fresh_scale, saved_scale, atol=1e-10)
    log("FIT", "scaler.scale_ matches (with zero-IQR guard)",
        scale_close)

    # Transform both
    train_scaled = scaler.transform(train_df[scale_cols])
    val_scaled = scaler.transform(val_df[scale_cols])

    # Clip (as the code does)
    train_scaled = np.clip(train_scaled, -10, 10)
    val_scaled = np.clip(val_scaled, -10, 10)

    print(f"\n  TRAIN ({train_scaled.shape[0]} rows × {train_scaled.shape[1]} cols):")

    # >95% of values in [-5, 5]
    train_in_5 = (np.abs(train_scaled) <= 5).mean()
    log("DIST", f"train: {train_in_5*100:.1f}% in [-5, 5]",
        train_in_5 > 0.95, ">95% required")

    # No values > 100 (clip should prevent)
    train_max = np.abs(train_scaled).max()
    log("DIST", f"train: max |value| = {train_max:.2f}",
        train_max <= 10.0, "clip bound = 10")

    # No NaN/inf after scaling
    train_nan = np.isnan(train_scaled).sum()
    train_inf = np.isinf(train_scaled).sum()
    log("DIST", f"train: NaN={train_nan}, inf={train_inf}",
        train_nan == 0 and train_inf == 0)

    print(f"\n  VAL ({val_scaled.shape[0]} rows × {val_scaled.shape[1]} cols):")

    val_in_5 = (np.abs(val_scaled) <= 5).mean()
    log("DIST", f"val: {val_in_5*100:.1f}% in [-5, 5]",
        val_in_5 > 0.95, ">95% required")

    val_max = np.abs(val_scaled).max()
    log("DIST", f"val: max |value| = {val_max:.2f}",
        val_max <= 10.0, "clip bound = 10")

    val_nan = np.isnan(val_scaled).sum()
    val_inf = np.isinf(val_scaled).sum()
    log("DIST", f"val: NaN={val_nan}, inf={val_inf}",
        val_nan == 0 and val_inf == 0)

    # Per-column: flag any column where >10% of values > 5 (outlier-heavy)
    print(f"\n  Per-column outlier check (|scaled| > 5 in >10% of train):")
    col_outlier_pct = (np.abs(train_scaled) > 5).mean(axis=0)
    outlier_cols = [(scale_cols[i], col_outlier_pct[i])
                    for i in range(len(scale_cols)) if col_outlier_pct[i] > 0.10]
    if outlier_cols:
        for col, pct in sorted(outlier_cols, key=lambda x: -x[1]):
            log("DIST", f"  {col}: {pct*100:.1f}% > 5", False, "heavy-tailed")
    else:
        log("DIST", "all columns <10% outliers", True)

    # Verify pass_cols are UNCHANGED
    print(f"\n  Pass-through columns (unscaled):")
    for col in pass_cols:
        if col in train_df.columns:
            raw_vals = train_df[col].values
            if col.startswith("regime_proba_"):
                in_01 = ((raw_vals >= -0.001) & (raw_vals <= 1.001)).all()
                log("PASS", f"{col} in [0,1]", in_01)
            elif col == "has_external_data":
                unique = set(raw_vals)
                log("PASS", f"{col} is binary", unique <= {0.0, 1.0},
                    f"unique={unique}")
else:
    print("  SKIPPED (no BTC fold_1 scaler)")

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
