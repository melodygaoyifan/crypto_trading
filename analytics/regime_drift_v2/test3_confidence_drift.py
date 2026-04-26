"""TEST 3: Posterior confidence drift (max posterior early vs recent)."""
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
        results[asset] = {'error': str(e)}; continue
    gmm = bundle['gmm']; scaler = bundle['scaler']

    df, feats = get_aligned_features_and_df(asset, days=None, feature_cols=bundle['feature_cols'])
    X = scaler.transform(feats.values)
    posteriors = gmm.predict_proba(X)
    max_post = posteriors.max(axis=1)

    n = len(feats)
    n_recent = min(180 * 6, n // 4)  # 6 months OR last 25%
    early = max_post[:n // 2]
    recent = max_post[-n_recent:]
    early_conf = float(early.mean())
    recent_conf = float(recent.mean())
    delta = recent_conf - early_conf

    if delta < -0.10: verdict = 'SEVERE_BLUR'
    elif delta < -0.05: verdict = 'MODERATE_BLUR'
    elif abs(delta) < 0.03: verdict = 'STABLE'
    else: verdict = 'MILD_DRIFT'

    print(f"    early_conf:  {early_conf:.4f}  (first half, n={len(early)})")
    print(f"    recent_conf: {recent_conf:.4f}  (last 6mo, n={len(recent)})")
    print(f"    delta:       {delta:+.4f}  → {verdict}")
    results[asset] = {
        'early_conf': early_conf, 'recent_conf': recent_conf,
        'delta': delta, 'verdict': verdict,
        'n_early': int(len(early)), 'n_recent': int(len(recent)),
    }

OUT = Path('/tmp/hmats_audit/regime_drift_v2/test3_confidence_drift.json')
OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open('w') as f: json.dump(results, f, indent=2, default=str)
print(f"\n  Saved: {OUT}")
