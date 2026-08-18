"""TEST 0: Artifact metadata audit (no ML)."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analytics.regime_drift_v2._common import (
    load_gmm_bundle, ASSETS, DEPLOYED_DIR, CANDIDATE_DIR,
)

OUT = Path('/tmp/hmats_audit/regime_drift_v2/test0_metadata.json')
OUT.parent.mkdir(parents=True, exist_ok=True)

results = {}
for asset in ASSETS:
    try:
        deployed = load_gmm_bundle(DEPLOYED_DIR, asset)
        candidate = load_gmm_bundle(CANDIDATE_DIR, asset)
    except Exception as e:
        print(f"  {asset}: FAIL — {e}")
        results[asset] = {'error': str(e)}
        continue

    asset_result = {
        'deployed': {
            'source': deployed['source'], 'mtime': deployed['mtime'],
            'n_components': int(deployed['gmm'].n_components),
            'feature_cols': list(deployed['feature_cols']),
            'regime_names': list(deployed['regime_names'] or []),
            'training_samples': deployed['config'].get('training_samples'),
            'mean_confidence': deployed['config'].get('mean_confidence'),
        },
        'candidate': {
            'source': candidate['source'], 'mtime': candidate['mtime'],
            'n_components': int(candidate['gmm'].n_components),
            'feature_cols': list(candidate['feature_cols']),
            'regime_names': list(candidate['regime_names'] or []),
            'training_samples': candidate['config'].get('training_samples'),
            'mean_confidence': candidate['config'].get('mean_confidence'),
        },
    }

    d_mean = np.asarray(deployed['scaler'].mean_)
    c_mean = np.asarray(candidate['scaler'].mean_)
    d_scale = np.asarray(deployed['scaler'].scale_)
    c_scale = np.asarray(candidate['scaler'].scale_)
    fnames = deployed['feature_cols']
    drift = []
    for i, name in enumerate(fnames):
        mp = (c_mean[i]-d_mean[i])/d_mean[i]*100 if d_mean[i] != 0 else None
        sp = (c_scale[i]-d_scale[i])/d_scale[i]*100 if d_scale[i] != 0 else None
        drift.append({
            'feature': name,
            'deployed_mean': float(d_mean[i]), 'candidate_mean': float(c_mean[i]),
            'mean_pct_change': float(mp) if mp is not None else None,
            'deployed_scale': float(d_scale[i]), 'candidate_scale': float(c_scale[i]),
            'scale_pct_change': float(sp) if sp is not None else None,
        })
    asset_result['scaler_drift'] = drift

    dep_r = set(asset_result['deployed']['regime_names'])
    cand_r = set(asset_result['candidate']['regime_names'])
    asset_result['regime_changes'] = {
        'removed_in_candidate': sorted(dep_r - cand_r),
        'added_in_candidate': sorted(cand_r - dep_r),
        'k_change': asset_result['candidate']['n_components'] - asset_result['deployed']['n_components'],
    }
    results[asset] = asset_result

    print(f"\n  === {asset} ===")
    print(f"    Deployed:  k={asset_result['deployed']['n_components']:<2} train_n={asset_result['deployed']['training_samples']:<6} mean_conf={asset_result['deployed']['mean_confidence']:.3f}")
    print(f"    Candidate: k={asset_result['candidate']['n_components']:<2} train_n={asset_result['candidate']['training_samples']:<6} mean_conf={asset_result['candidate']['mean_confidence']:.3f}")
    print(f"    Δk = {asset_result['regime_changes']['k_change']:+d}")
    print(f"    Removed: {asset_result['regime_changes']['removed_in_candidate']}")
    print(f"    Added:   {asset_result['regime_changes']['added_in_candidate']}")
    sd = sorted(drift, key=lambda x: abs(x['mean_pct_change'] or 0), reverse=True)[:4]
    print(f"    Top 4 mean drifts:")
    for s in sd:
        m = s['mean_pct_change']; sc = s['scale_pct_change']
        m_str = f"{m:+6.1f}%" if m is not None else "  N/A"
        s_str = f"{sc:+6.1f}%" if sc is not None else "  N/A"
        print(f"      {s['feature']:<30} mean={m_str}  scale={s_str}")

with OUT.open('w', encoding="utf-8") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\n  Saved: {OUT}")
