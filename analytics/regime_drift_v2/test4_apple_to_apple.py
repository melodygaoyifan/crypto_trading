"""TEST 4: Apple-to-apple — same holdout, each model's own scaler, ARI/NMI."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analytics.regime_drift_v2._common import (
    load_gmm_bundle, get_aligned_features_and_df,
    predict_labels_with_bundle,
    DEPLOYED_DIR, CANDIDATE_DIR, ASSETS,
)

results = {}
for asset in ASSETS:
    print(f"\n  === {asset} ===")
    try:
        deployed = load_gmm_bundle(DEPLOYED_DIR, asset)
        candidate = load_gmm_bundle(CANDIDATE_DIR, asset)
    except Exception as e:
        results[asset] = {'error': str(e)}; print(f"    FAIL — {e}"); continue

    fd = list(deployed['feature_cols']); fc = list(candidate['feature_cols'])
    if fd != fc:
        results[asset] = {'error': 'feature_cols mismatch'}; continue

    df, feats = get_aligned_features_and_df(asset, days=180, feature_cols=fd)
    labels_d, post_d = predict_labels_with_bundle(deployed, feats)
    labels_c, post_c = predict_labels_with_bundle(candidate, feats)

    ari = float(adjusted_rand_score(labels_d, labels_c))
    nmi = float(normalized_mutual_info_score(labels_d, labels_c))
    if ari > 0.7: verdict = 'GMM_HEALTHY'
    elif ari > 0.3: verdict = 'PARTIAL_DRIFT'
    else: verdict = 'SEVERE_DRIFT'

    print(f"    Holdout: {len(feats)} bars (recent 6 months)")
    print(f"    Deployed clusters used:  {sorted(set(int(x) for x in labels_d))}")
    print(f"    Candidate clusters used: {sorted(set(int(x) for x in labels_c))}")
    print(f"    ARI: {ari:.3f}    NMI: {nmi:.3f}")
    print(f"    Mean post — deployed: {post_d.mean():.3f}    candidate: {post_c.mean():.3f}")
    print(f"    → {verdict}")
    results[asset] = {
        'ari': ari, 'nmi': nmi,
        'mean_post_deployed': float(post_d.mean()),
        'mean_post_candidate': float(post_c.mean()),
        'n_holdout': int(len(feats)),
        'verdict': verdict,
    }

OUT = Path('/tmp/hmats_audit/regime_drift_v2/test4_apple_to_apple.json')
OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open('w') as f: json.dump(results, f, indent=2, default=str)
print(f"\n  Saved: {OUT}")
