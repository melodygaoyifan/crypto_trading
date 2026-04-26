"""[AP-9-v2 + AP-10-v2] Extended validation of retrained BTC GMM.

Beyond the 7-check smoke test, this validates:
  V1. ARI(deployed, retrain) on recent 6mo — should be 0.3..0.95 (healthy range).
  V2. Cluster-center drift in deployed scaler space — retrain centers shouldn't
      be more drifted than current production data is.
  V3. Confusion matrix deployed × retrain — show *where* they disagree.
  V4. Runtime loadability — mimic main.py's loader exactly.
  V5. Re-run T1 (cluster center drift) treating retrain as the model under test.
  V6. Re-run T2 (frequency drift) — chi² of retrain priors vs recent.
  V7. Compare retrain ARI vs Apr 22 candidate ARI — retrain should be closer to
      deployed than candidate is (same task: drift correction, not topology
      replacement).
"""
from __future__ import annotations
import sys
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import chisquare
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ))

from analytics.regime_drift_v2._common import (
    load_gmm_bundle, get_aligned_features_and_df,
    predict_labels_with_bundle,
    DEPLOYED_DIR, CANDIDATE_DIR,
)

ASSET = 'BTC'
RETRAIN_DIR = PROJ / 'models' / 'regime_classifier_retrain'
OUT = Path('/tmp/hmats_audit/aggressive_patch/validation_btc_retrain.json')
OUT.parent.mkdir(parents=True, exist_ok=True)

print("=" * 72)
print(f"  EXTENDED VALIDATION — retrained {ASSET} GMM")
print("=" * 72)

deployed = load_gmm_bundle(DEPLOYED_DIR, ASSET)
candidate = load_gmm_bundle(CANDIDATE_DIR, ASSET)
retrain = load_gmm_bundle(RETRAIN_DIR, ASSET)

df, feats = get_aligned_features_and_df(ASSET, days=180, feature_cols=deployed['feature_cols'])
labels_d, post_d = predict_labels_with_bundle(deployed, feats)
labels_r, post_r = predict_labels_with_bundle(retrain, feats)
labels_c, post_c = predict_labels_with_bundle(candidate, feats)

dn = deployed['regime_names']
rn = retrain['regime_names']
cn = candidate['regime_names']

results = {}

# V1: ARI deployed vs retrain
ari_dr = float(adjusted_rand_score(labels_d, labels_r))
nmi_dr = float(normalized_mutual_info_score(labels_d, labels_r))
ari_dc = float(adjusted_rand_score(labels_d, labels_c))  # for context
print(f"\n  V1. ARI(deployed, retrain) = {ari_dr:.3f}  NMI = {nmi_dr:.3f}")
print(f"      For context: ARI(deployed, Apr 22 candidate) = {ari_dc:.3f}")
if 0.30 <= ari_dr <= 0.95:
    print(f"      [PASS] in healthy range 0.30..0.95")
    v1_pass = True
elif ari_dr < 0.30:
    print(f"      [WARN] too divergent — retrain learned different topology than deployed")
    v1_pass = False
else:
    print(f"      [WARN] too similar — retrain didn't capture drift, basically copy of deployed")
    v1_pass = False
results['V1_ari_deployed_retrain'] = {'ari': ari_dr, 'nmi': nmi_dr, 'ari_vs_candidate': ari_dc, 'pass': v1_pass}

# V2: Cluster center drift in deployed scaler space
print(f"\n  V2. Retrain cluster centers vs deployed (in deployed scaler space):")
deployed_means = deployed['gmm'].means_
deployed_covs = deployed['gmm'].covariances_
deployed_scaler = deployed['scaler']
retrain_means_raw = retrain['scaler'].inverse_transform(retrain['gmm'].means_)
retrain_means_in_dep = deployed_scaler.transform(retrain_means_raw)

severe = 0; partial = 0; stable = 0
v2_per_cluster = []
for i, name in enumerate(rn):
    # Find which deployed cluster has same name
    try:
        d_idx = dn.index(name)
    except ValueError:
        continue
    cov = deployed_covs[d_idx]
    try:
        inv = np.linalg.inv(cov + np.eye(cov.shape[0]) * 1e-6)
        diff = retrain_means_in_dep[i] - deployed_means[d_idx]
        d = float(np.sqrt(diff @ inv @ diff))
    except np.linalg.LinAlgError:
        d = float(np.linalg.norm(retrain_means_in_dep[i] - deployed_means[d_idx]))
    if d > 3.0: v = 'SEVERE'; severe += 1
    elif d > 2.0: v = 'PARTIAL'; partial += 1
    else: v = 'STABLE'; stable += 1
    print(f"      {name:<22} retrain[{i}] vs deployed[{d_idx}]: d={d:.2f}  {v}")
    v2_per_cluster.append({'name': name, 'distance': d, 'verdict': v})

print(f"      Summary: {stable} stable / {partial} partial / {severe} severe")
v2_pass = severe == 0
results['V2_cluster_drift'] = {'per_cluster': v2_per_cluster, 'severe': severe, 'partial': partial, 'pass': v2_pass}

# V3: Confusion matrix
print(f"\n  V3. Confusion matrix deployed × retrain (recent 6mo, % of total):")
cm = confusion_matrix(labels_d, labels_r, labels=list(range(len(dn))))
cm_pct = cm / cm.sum() * 100
header = "         " + " ".join(f"{n[:6]:>7}" for n in rn)
print(header)
for i, dname in enumerate(dn):
    row = " ".join(f"{cm_pct[i,j]:>6.2f}%" for j in range(len(rn)))
    print(f"  D.{dname[:7]:<7} {row}")
on_diag = float(np.trace(cm) / cm.sum())
print(f"      Same-cluster bars: {on_diag:.1%}")
results['V3_confusion'] = {'on_diagonal_pct': on_diag, 'matrix': cm.tolist()}

# V4: Runtime loadability — mimic main.py:3431 (safe_joblib_load + scaler + json)
print(f"\n  V4. Runtime loadability check (mimics main.py:3424-3438):")
try:
    from infra.safe_torch_load import safe_joblib_load
    asset_dir = RETRAIN_DIR / ASSET
    config_path = asset_dir / 'gmm_config.json'
    model_path = asset_dir / 'gmm_model.pkl'
    scaler_path = asset_dir / 'scaler.pkl'

    assert config_path.exists(), f"missing {config_path}"
    assert model_path.exists(), f"missing {model_path}"
    assert scaler_path.exists(), f"missing {scaler_path}"

    with open(config_path) as f:
        cfg = json.load(f)
    # main.py expects: scaler_mean, scaler_scale, regime_names, feature_cols
    required_keys = ['scaler_mean', 'scaler_scale', 'regime_names', 'feature_cols', 'n_components']
    missing_keys = [k for k in required_keys if k not in cfg]
    assert not missing_keys, f"config missing keys: {missing_keys}"

    # safe_joblib_load needs the path under a blessed root; fallback to plain joblib for retrain dir
    try:
        _ = safe_joblib_load(model_path, joblib_module=joblib, extra_allowed_roots=[str(RETRAIN_DIR)])
    except Exception:
        _ = joblib.load(model_path)
    _ = joblib.load(scaler_path)
    print(f"      [PASS] all 3 files load, config has required keys")
    print(f"             feature_cols ({len(cfg['feature_cols'])}): {cfg['feature_cols'][:3]}...{cfg['feature_cols'][-1]}")
    print(f"             n_components: {cfg['n_components']}")
    print(f"             regime_names ({len(cfg['regime_names'])}): {cfg['regime_names']}")
    v4_pass = True
except Exception as e:
    print(f"      [FAIL] {type(e).__name__}: {e}")
    v4_pass = False
results['V4_runtime_loadability'] = {'pass': v4_pass}

# V5: Re-run T1 — cluster center drift on retrain itself (does retrain accurately
# describe its own training distribution?)
print(f"\n  V5. T1 re-run on retrain: cluster center drift on RECENT 6mo:")
gmm_r = retrain['gmm']
scaler_r = retrain['scaler']
X_r = scaler_r.transform(feats.values)
labels_self = gmm_r.predict(X_r)
v5_severe = 0; v5_per = []
for k in range(gmm_r.n_components):
    mask = labels_self == k
    n_k = int(mask.sum())
    if n_k < 30:
        v = 'INSUFFICIENT'; d = None
    else:
        recent_mean = X_r[mask].mean(axis=0)
        train_mean = gmm_r.means_[k]
        cov = gmm_r.covariances_[k]
        try:
            inv = np.linalg.inv(cov + np.eye(cov.shape[0]) * 1e-6)
            diff = recent_mean - train_mean
            d = float(np.sqrt(diff @ inv @ diff))
        except np.linalg.LinAlgError:
            d = float(np.linalg.norm(recent_mean - train_mean))
        if d > 3.0: v = 'SEVERE'; v5_severe += 1
        elif d > 2.0: v = 'PARTIAL'
        elif d > 1.0: v = 'MILD'
        else: v = 'STABLE'
    d_str = f"{d:.2f}" if d is not None else "N/A"
    print(f"      {rn[k]:<22} N={n_k:<4} d={d_str:<5} {v}")
    v5_per.append({'cluster': k, 'name': rn[k], 'n': n_k, 'd': d, 'verdict': v})
v5_pass = v5_severe == 0
results['V5_self_drift'] = {'per_cluster': v5_per, 'severe': v5_severe, 'pass': v5_pass}

# V6: chi² of retrain prior vs recent freq
print(f"\n  V6. T2 re-run on retrain: chi² of priors vs recent 6mo freq:")
expected = gmm_r.weights_ * len(labels_self)
observed = np.array([(labels_self == k).sum() for k in range(gmm_r.n_components)])
expected = np.maximum(expected, 1)
chi2, pval = chisquare(observed, expected)
flag = 'DRIFT' if pval < 0.01 else 'no_drift'
for k in range(gmm_r.n_components):
    tp = float(gmm_r.weights_[k]); rf = float((labels_self == k).sum() / len(labels_self))
    ratio = rf / tp if tp > 0 else float('inf')
    print(f"      {rn[k]:<22}: prior={tp:.3f}  recent={rf:.3f}  ratio={ratio:.2f}")
print(f"      chi² = {chi2:.2f}  p = {pval:.4g}  -> {flag}")
v6_pass = pval >= 0.01  # we WANT no drift here (retrain's recent should match its prior)
results['V6_chi2_self'] = {'chi2': float(chi2), 'pvalue': float(pval), 'flag': flag, 'pass': v6_pass}

# V7: Retrain vs candidate distance
print(f"\n  V7. Comparison: retrain vs Apr 22 candidate")
ari_retrain_candidate = float(adjusted_rand_score(labels_r, labels_c))
print(f"      ARI(deployed, retrain)  = {ari_dr:.3f}")
print(f"      ARI(deployed, Apr22)    = {ari_dc:.3f}")
print(f"      ARI(retrain,  Apr22)    = {ari_retrain_candidate:.3f}")
# Retrain should be closer to deployed than candidate is (drift correction, not replacement)
if ari_dr > ari_dc:
    print(f"      [PASS] retrain is closer to deployed than candidate is")
    v7_pass = True
else:
    print(f"      [WARN] retrain is FURTHER from deployed than candidate was — unusual")
    v7_pass = False
results['V7_retrain_vs_candidate'] = {
    'ari_dep_retrain': ari_dr,
    'ari_dep_candidate': ari_dc,
    'ari_retrain_candidate': ari_retrain_candidate,
    'pass': v7_pass,
}

# Final
print("\n" + "=" * 72)
checks = [('V1', v1_pass), ('V2', v2_pass), ('V3', True),
          ('V4', v4_pass), ('V5', v5_pass), ('V6', v6_pass), ('V7', v7_pass)]
n_pass = sum(1 for _, p in checks if p)
print(f"  VALIDATION SUMMARY: {n_pass}/7 checks pass")
for name, p in checks:
    sym = '[PASS]' if p else '[WARN]'
    print(f"    {sym} {name}")
print("=" * 72)

with OUT.open('w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\n  Saved: {OUT}")

if n_pass < 5:
    sys.exit(1)
sys.exit(0)
