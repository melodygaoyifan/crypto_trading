"""TEST 5: Temporal localization of deployed-vs-candidate disagreement."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

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
        results[asset] = {'error': str(e)}; continue

    df, feats = get_aligned_features_and_df(asset, days=365, feature_cols=deployed['feature_cols'])
    labels_d, _ = predict_labels_with_bundle(deployed, feats)
    labels_c, _ = predict_labels_with_bundle(candidate, feats)
    df = df.iloc[:len(labels_d)].copy()
    df['label_deployed'] = labels_d
    df['label_candidate'] = labels_c
    df['disagree'] = (labels_d != labels_c).astype(int)

    df['month'] = df['timestamp'].dt.to_period('M').astype(str)
    monthly = df.groupby('month')['disagree'].agg(['mean', 'count']).reset_index()
    monthly.columns = ['month', 'disagreement_rate', 'n_bars']

    print(f"    Per-month disagreement (12mo, ~{len(df)} bars):")
    for _, row in monthly.iterrows():
        bar = '#' * int(round(row['disagreement_rate'] * 30))
        print(f"      {row['month']}: {row['disagreement_rate']:>5.1%} ({int(row['n_bars']):>4} bars) {bar}")

    pair_df = df[df['disagree'] == 1].groupby(['label_deployed', 'label_candidate']).size().reset_index(name='count')
    pair_df = pair_df.sort_values('count', ascending=False).head(5)
    dn = deployed.get('regime_names') or [f'd{i}' for i in range(deployed['gmm'].n_components)]
    cn = candidate.get('regime_names') or [f'c{i}' for i in range(candidate['gmm'].n_components)]
    print(f"    Top 5 disagreement pairs (deployed → candidate):")
    pairs_out = []
    for _, row in pair_df.iterrows():
        d_name = dn[int(row['label_deployed'])]; c_name = cn[int(row['label_candidate'])]
        print(f"      {d_name:<25} → {c_name:<25}: {int(row['count']):>4} bars")
        pairs_out.append({'deployed': d_name, 'candidate': c_name, 'count': int(row['count'])})

    chunk34_mask = (df['timestamp'] >= '2025-08-01') & (df['timestamp'] <= '2026-01-31')
    other_mask = ~chunk34_mask
    chunk34 = float(df.loc[chunk34_mask, 'disagree'].mean()) if chunk34_mask.sum() > 0 else None
    other = float(df.loc[other_mask, 'disagree'].mean()) if other_mask.sum() > 0 else None
    print(f"    Chunks 3-4 zone (Aug 2025 - Jan 2026):")
    if chunk34 is not None:
        print(f"      Disagreement: {chunk34:.1%} (N={int(chunk34_mask.sum())})")
    if other is not None:
        print(f"    Other periods:")
        print(f"      Disagreement: {other:.1%} (N={int(other_mask.sum())})")
    if chunk34 is not None and other is not None:
        elev = chunk34 - other
        flag = ' ** HIGH **' if elev > 0.15 else ''
        print(f"    Chunk 3-4 elevation: {elev:+.1%}{flag}")

    results[asset] = {
        'monthly_disagreement': monthly.to_dict(orient='records'),
        'top_pairs': pairs_out,
        'chunk34_disagreement': chunk34,
        'other_disagreement': other,
    }

OUT = Path('/tmp/hmats_audit/regime_drift_v2/test5_disagreement_timeline.json')
OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open('w', encoding="utf-8") as f: json.dump(results, f, indent=2, default=str)
print(f"\n  Saved: {OUT}")
