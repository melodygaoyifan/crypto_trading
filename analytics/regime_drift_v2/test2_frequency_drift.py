"""TEST 2: Cluster frequency drift (recent prior vs train weights_)."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import chisquare

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analytics.regime_drift_v2._common import (
    load_gmm_bundle, get_aligned_features_and_df,
    DEPLOYED_DIR, ASSETS,
)

results = {}
for asset in ASSETS:
    print(f"\n  === {asset} ===")
    try:
        bundle = load_gmm_bundle(DEPLOYED_DIR, asset)
    except Exception as e:
        results[asset] = {'error': str(e)}; continue
    gmm = bundle['gmm']; scaler = bundle['scaler']
    regime_names = bundle.get('regime_names') or [f'c{i}' for i in range(gmm.n_components)]

    df, feats = get_aligned_features_and_df(asset, days=180, feature_cols=bundle['feature_cols'])
    X = scaler.transform(feats.values)
    labels = gmm.predict(X)

    train_priors = gmm.weights_
    recent_freqs = np.array([(labels == k).sum() / len(labels) for k in range(gmm.n_components)])
    asset_results = []
    for k in range(gmm.n_components):
        tp = float(train_priors[k]); rf = float(recent_freqs[k])
        ratio = rf / tp if tp > 0 else float('inf')
        if rf < 0.005: verdict = 'NEAR_EXTINCT'
        elif ratio < 0.3: verdict = 'SHRUNK'
        elif ratio > 3.0: verdict = 'EXPANDED'
        elif 0.5 < ratio < 2.0: verdict = 'STABLE'
        else: verdict = 'DRIFTED'
        asset_results.append({
            'cluster': k, 'name': regime_names[k],
            'train_prior': tp, 'recent_freq': rf,
            'ratio': float(ratio) if not np.isinf(ratio) else None,
            'verdict': verdict,
        })
        ratio_str = f"{ratio:.2f}" if not np.isinf(ratio) else "  inf"
        print(f"    {regime_names[k]:<25}: train={tp:.3f}  recent={rf:.3f}  ratio={ratio_str}  {verdict}")

    expected = train_priors * len(labels)
    observed = np.array([(labels == k).sum() for k in range(gmm.n_components)])
    expected = np.maximum(expected, 1)
    chi2, pval = chisquare(observed, expected)
    flag = 'DRIFT' if pval < 0.01 else 'NO_DRIFT'
    print(f"    Chi-square: chi2={chi2:.1f}  p={pval:.4g}  → {flag}")
    results[asset] = {
        'clusters': asset_results,
        'chi2': float(chi2), 'pvalue': float(pval), 'flag': flag,
    }

OUT = Path('/tmp/hmats_audit/regime_drift_v2/test2_frequency_drift.json')
OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open('w') as f: json.dump(results, f, indent=2, default=str)
print(f"\n  Saved: {OUT}")
