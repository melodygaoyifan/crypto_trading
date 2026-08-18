"""TEST 1: Cluster center drift (deployed scaler space, recent 6mo)."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

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
        results[asset] = {'error': str(e)}
        print(f"    FAIL — {e}")
        continue
    gmm = bundle['gmm']; scaler = bundle['scaler']

    df, feats = get_aligned_features_and_df(asset, days=180, feature_cols=bundle['feature_cols'])
    if len(feats) < 200:
        results[asset] = {'error': f'insufficient data: {len(feats)}'}
        continue

    X = scaler.transform(feats.values)
    labels = gmm.predict(X)
    means = gmm.means_
    covs = gmm.covariances_
    regime_names = bundle.get('regime_names') or [f'cluster_{i}' for i in range(gmm.n_components)]

    asset_results = []
    n_drifted = 0
    for k in range(gmm.n_components):
        mask = labels == k
        n_k = int(mask.sum())
        if n_k < 30:
            verdict = 'INSUFFICIENT'
            d = None
        else:
            recent_mean = X[mask].mean(axis=0)
            train_mean = means[k]
            cov = covs[k] if gmm.covariance_type == 'full' else np.eye(len(train_mean))
            try:
                inv = np.linalg.inv(cov + np.eye(cov.shape[0]) * 1e-6)
                diff = recent_mean - train_mean
                d = float(np.sqrt(diff @ inv @ diff))
            except np.linalg.LinAlgError:
                d = float(np.linalg.norm(recent_mean - train_mean))
            if d > 3.0: verdict = 'SEVERE_DRIFT'
            elif d > 2.0: verdict = 'DRIFT_DETECTED'
            elif d > 1.0: verdict = 'MILD_DRIFT'
            else: verdict = 'STABLE'
            if verdict in ('SEVERE_DRIFT', 'DRIFT_DETECTED'):
                n_drifted += 1
        asset_results.append({
            'cluster': k, 'name': regime_names[k],
            'n_recent': n_k, 'drift_distance_mahalanobis': d, 'verdict': verdict,
        })
        d_str = f"{d:5.2f}" if d is not None else "  N/A"
        print(f"    {k}: {regime_names[k]:<25} N={n_k:>4}  d={d_str}  {verdict}")
    print(f"    → {n_drifted}/{gmm.n_components} clusters drifted")
    results[asset] = {'clusters': asset_results, 'n_drifted': n_drifted, 'k': gmm.n_components}

OUT = Path('/tmp/hmats_audit/regime_drift_v2/test1_cluster_drift.json')
OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open('w', encoding="utf-8") as f: json.dump(results, f, indent=2, default=str)
print(f"\n  Saved: {OUT}")
