"""③ GMM Regime Layer Validation."""
import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib

PROJECT = Path(r"c:\Users\melod\Downloads\hmats")
DRL_DIR = PROJECT / "data" / "drl_training"
GMM_DIR = PROJECT / "models" / "regime_classifier"
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

# ═══════════════════════════════════════════════════════════
# PER-ASSET GMM
# ═══════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("PER-ASSET GMM")
print(SEP)

# Load split manifest for train_end verification
with open(SPLIT_MANIFEST) as f:
    split_manifest = json.load(f)

for asset in ASSETS:
    asset_dir = GMM_DIR / asset
    config_path = asset_dir / "gmm_config.json"
    model_path = asset_dir / "gmm_model.pkl"
    scaler_path = asset_dir / "scaler.pkl"

    print(f"\n  {asset}")
    print(f"  {'─' * 50}")

    # Files exist
    log("GMM", f"{asset}: gmm_config.json exists", config_path.exists())
    log("GMM", f"{asset}: gmm_model.pkl exists", model_path.exists())
    log("GMM", f"{asset}: scaler.pkl exists", scaler_path.exists())

    if not config_path.exists():
        continue

    with open(config_path) as f:
        cfg = json.load(f)

    # k ∈ [3, 8]
    k = cfg.get("n_components", cfg.get("k"))
    log("GMM", f"{asset}: k={k} ∈ [3,8]", 3 <= k <= 8)

    # 12-dim input features
    feature_cols = cfg.get("feature_cols", [])
    log("GMM", f"{asset}: GMM input dim = {len(feature_cols)}", len(feature_cols) == 12)

    # Min regime pct > 2% (from GMM weights)
    weights = cfg.get("weights", [])
    if weights:
        regime_names = cfg.get("regime_names", [f"R{i}" for i in range(len(weights))])
        pcts = {regime_names[i]: w * 100 for i, w in enumerate(weights)}
        min_pct = min(pcts.values())
        min_name = min(pcts, key=pcts.get)
        log("GMM", f"{asset}: min regime pct = {min_pct:.1f}% ({min_name})", min_pct > 2.0,
            " | ".join(f"{n}={p:.1f}%" for n, p in sorted(pcts.items(), key=lambda x: x[1])))
    else:
        log("GMM", f"{asset}: min regime pct", False, "no weights in config")

    # Transition frequency
    trans_freq = cfg.get("transition_freq", cfg.get("flip_rate"))
    if trans_freq is not None:
        log("GMM", f"{asset}: transition freq = {trans_freq:.3f}",
            0.05 <= trans_freq <= 0.20,
            "in range [0.05, 0.20]" if 0.05 <= trans_freq <= 0.20
            else f"OUT OF RANGE {'too low (stuck)' if trans_freq < 0.05 else 'too high (noise)'}")
    else:
        log("GMM", f"{asset}: transition freq", False, "not in config")

    # Train-only fit
    train_end = cfg.get("train_end_row")
    training_samples = cfg.get("training_samples")
    parquet_rows = split_manifest["assets"][asset]["total_rows"]

    log("GMM", f"{asset}: train_end_row = {train_end}", train_end is not None,
        f"(full parquet = {parquet_rows})")
    if train_end is not None:
        fold1_end = split_manifest["assets"][asset]["folds"][0]["train_end"]
        log("GMM", f"{asset}: train_end matches fold_1 ({fold1_end})",
            train_end == fold1_end)
        log("GMM", f"{asset}: training_samples < total_rows",
            training_samples is not None and training_samples < parquet_rows,
            f"samples={training_samples} vs total={parquet_rows}")

    # BIC search results
    bic_results = cfg.get("bic_results", cfg.get("bic_search", {}))
    if bic_results:
        log("GMM", f"{asset}: BIC search performed", True,
            f"k values tested: {list(bic_results.keys()) if isinstance(bic_results, dict) else 'present'}")

    # Regime names
    regime_names = cfg.get("regime_names", [])
    log("GMM", f"{asset}: regime_names count = {len(regime_names)}",
        len(regime_names) == k, str(regime_names))

# ═══════════════════════════════════════════════════════════
# 3 INDEPENDENT GMMs (not shared)
# ═══════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("INDEPENDENT GMMs CHECK")
print(SEP)

# Verify models are different
models = {}
for asset in ASSETS:
    model_path = GMM_DIR / asset / "gmm_model.pkl"
    if model_path.exists():
        gmm = joblib.load(model_path)
        models[asset] = gmm

if len(models) == 3:
    # Check means are different (proves independent training)
    btc_means = models["BTC"].means_
    eth_means = models["ETH"].means_
    sol_means = models["SOL"].means_

    btc_eth_same = np.allclose(btc_means, eth_means, atol=1e-6) if btc_means.shape == eth_means.shape else False
    btc_sol_same = np.allclose(btc_means, sol_means, atol=1e-6) if btc_means.shape == sol_means.shape else False

    log("INDEP", "BTC ≠ ETH (different means)", not btc_eth_same)
    log("INDEP", "BTC ≠ SOL (different means)", not btc_sol_same)

    for asset, gmm in models.items():
        log("INDEP", f"{asset}: n_components = {gmm.n_components}", True)
else:
    log("INDEP", "3 independent models loaded", False, f"only {len(models)} loaded")

# ═══════════════════════════════════════════════════════════
# REGIME_PROBA VECTOR
# ═══════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("REGIME_PROBA VECTOR")
print(SEP)

with open(MANIFEST) as f:
    manifest = json.load(f)

for asset in ASSETS:
    df = pd.read_parquet(DRL_DIR / f"{asset}_4H_full.parquet")
    cfg_path = GMM_DIR / asset / "gmm_config.json"
    with open(cfg_path) as f:
        cfg = json.load(f)
    k = cfg.get("n_components", cfg.get("k"))

    print(f"\n  {asset} (k={k})")
    print(f"  {'─' * 50}")

    # 8 proba columns present
    proba_cols = [f"regime_proba_{i}" for i in range(8)]
    for pc in proba_cols:
        log("PROBA", f"{asset}: {pc} present", pc in df.columns)

    # Zero-padding for k < 8
    if k < 8:
        for i in range(k, 8):
            col = f"regime_proba_{i}"
            if col in df.columns:
                all_zero = (df[col] == 0.0).all()
                log("PROBA", f"{asset}: {col} zero-padded (k={k})", all_zero)
    else:
        log("PROBA", f"{asset}: k=8, no zero-padding needed", True)

    # Row sum ≈ 1.0
    existing_proba = [pc for pc in proba_cols if pc in df.columns]
    row_sums = df[existing_proba].sum(axis=1)
    max_deviation = (row_sums - 1.0).abs().max()
    mean_deviation = (row_sums - 1.0).abs().mean()
    log("PROBA", f"{asset}: row sum ≈ 1.0 (max dev={max_deviation:.6f})",
        max_deviation < 0.01)

    # regime (int) NOT in obs
    all_features = manifest.get("all_features", [])
    log("PROBA", f"{asset}: 'regime' NOT in all_features", "regime" not in all_features)

    # regime_confidence NOT in obs
    log("PROBA", f"{asset}: 'regime_confidence' NOT in all_features",
        "regime_confidence" not in all_features)

    # regime and regime_confidence still exist in parquet (backward compat)
    log("PROBA", f"{asset}: 'regime' column exists in parquet (compat)",
        "regime" in df.columns)
    log("PROBA", f"{asset}: 'regime_confidence' exists in parquet (compat)",
        "regime_confidence" in df.columns)

# ═══════════════════════════════════════════════════════════
# TWO INDEPENDENT SCALER CHAINS (P8)
# ═══════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("TWO INDEPENDENT SCALER CHAINS (P8)")
print(SEP)

for asset in ASSETS:
    print(f"\n  {asset}")
    print(f"  {'─' * 50}")

    # GMM scaler
    gmm_scaler_path = GMM_DIR / asset / "scaler.pkl"
    if gmm_scaler_path.exists():
        gmm_scaler = joblib.load(gmm_scaler_path)
        scaler_type = type(gmm_scaler).__name__
        n_features = gmm_scaler.n_features_in_
        log("SCALER", f"{asset} GMM scaler type = {scaler_type}", scaler_type == "StandardScaler")
        log("SCALER", f"{asset} GMM scaler n_features_in_ = {n_features}", n_features == 12)
    else:
        log("SCALER", f"{asset} GMM scaler exists", False, "MISSING")

    # Feature scaler (from most recent fold)
    feat_scaler_path = PROJECT / "models" / "retrained" / asset / "fold_1" / "logs" / "ULTIMATE" / "feature_scaler.pkl"
    if feat_scaler_path.exists():
        feat_scaler_data = joblib.load(feat_scaler_path)
        # Saved as dict with 'scaler', 'scale_cols', 'pass_cols'
        if isinstance(feat_scaler_data, dict):
            feat_scaler = feat_scaler_data["scaler"]
        else:
            feat_scaler = feat_scaler_data
        scaler_type = type(feat_scaler).__name__
        n_features = feat_scaler.n_features_in_
        log("SCALER", f"{asset} Feature scaler type = {scaler_type}", scaler_type == "RobustScaler")
        log("SCALER", f"{asset} Feature scaler n_features_in_ = {n_features}", n_features > 50,
            f"dim={n_features}")
        if isinstance(feat_scaler_data, dict):
            n_pass = len(feat_scaler_data.get("pass_cols", []))
            log("SCALER", f"{asset} Feature scaler pass_cols = {n_pass}", n_pass == 9,
                "regime_proba_0..7 + has_external_data")
    else:
        log("SCALER", f"{asset} Feature scaler exists", False,
            f"MISSING at {feat_scaler_path}")

    # Verify they are different objects/types
    if gmm_scaler_path.exists() and feat_scaler_path.exists():
        gmm_s = joblib.load(gmm_scaler_path)
        feat_s_data = joblib.load(feat_scaler_path)
        feat_s = feat_s_data["scaler"] if isinstance(feat_s_data, dict) else feat_s_data
        log("SCALER", f"{asset} scalers are different types",
            type(gmm_s).__name__ != type(feat_s).__name__)
        log("SCALER", f"{asset} scalers have different dims",
            gmm_s.n_features_in_ != feat_s.n_features_in_)

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
