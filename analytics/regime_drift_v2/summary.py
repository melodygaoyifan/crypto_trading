"""Aggregate all 5 tests into a single severity score + per-asset verdict."""
from __future__ import annotations
import json
from pathlib import Path

OUT_DIR = Path('/tmp/hmats_audit/regime_drift_v2')
files = {
    'test0': OUT_DIR / 'test0_metadata.json',
    'test1': OUT_DIR / 'test1_cluster_drift.json',
    'test2': OUT_DIR / 'test2_frequency_drift.json',
    'test3': OUT_DIR / 'test3_confidence_drift.json',
    'test4': OUT_DIR / 'test4_apple_to_apple.json',
    'test5': OUT_DIR / 'test5_disagreement_timeline.json',
}
data = {k: json.load(p.open()) for k, p in files.items()}

ASSETS = ['BTC', 'ETH', 'SOL']
print("=" * 84)
print(f"  REGIME DRIFT DIAGNOSTIC SUMMARY (Feb 15 deployed GMM, recent 6mo holdout)")
print("=" * 84)

aggregate = {}
for asset in ASSETS:
    print(f"\n  === {asset} ===")
    score = 0
    notes = []

    # Test 1: cluster center drift
    t1 = data['test1'].get(asset, {})
    if 'clusters' in t1:
        n_severe = sum(1 for c in t1['clusters'] if c['verdict'] == 'SEVERE_DRIFT')
        n_drift = sum(1 for c in t1['clusters'] if c['verdict'] == 'DRIFT_DETECTED')
        n_mild = sum(1 for c in t1['clusters'] if c['verdict'] == 'MILD_DRIFT')
        n_extinct = sum(1 for c in t1['clusters'] if c['verdict'] == 'INSUFFICIENT')
        score += n_severe * 30 + n_drift * 15 + n_mild * 5 + n_extinct * 8
        print(f"    Test 1 (Mahalanobis): {n_severe}severe / {n_drift}drift / {n_mild}mild / {n_extinct}extinct")
        notes.append(f"T1: {n_extinct} clusters near-extinct (stable geometry, narrow exercise)")

    # Test 2: frequency
    t2 = data['test2'].get(asset, {})
    if 'clusters' in t2:
        ne = sum(1 for c in t2['clusters'] if c['verdict'] == 'NEAR_EXTINCT')
        sh = sum(1 for c in t2['clusters'] if c['verdict'] == 'SHRUNK')
        ex = sum(1 for c in t2['clusters'] if c['verdict'] == 'EXPANDED')
        score += ne * 20 + sh * 10 + ex * 8
        chi_p = t2.get('pvalue', 1.0)
        flag = 'DRIFT' if chi_p < 0.01 else 'no'
        print(f"    Test 2 (frequency): {ne}extinct / {sh}shrunk / {ex}expanded  chi2 p={chi_p:.2e} ({flag})")
        notes.append(f"T2: chi2 p={chi_p:.0e}")

    # Test 3: confidence
    t3 = data['test3'].get(asset, {})
    if 'verdict' in t3:
        v = t3['verdict']; d = t3.get('delta', 0.0)
        if v == 'SEVERE_BLUR': score += 25
        elif v == 'MODERATE_BLUR': score += 12
        print(f"    Test 3 (confidence):  delta={d:+.4f}  → {v}")

    # Test 4: ARI
    t4 = data['test4'].get(asset, {})
    if 'ari' in t4:
        ari = t4['ari']; v = t4['verdict']
        if v == 'SEVERE_DRIFT': score += 35
        elif v == 'PARTIAL_DRIFT': score += 18
        print(f"    Test 4 (ARI vs Apr22 candidate): ARI={ari:.3f} → {v}")
        notes.append(f"T4 ARI={ari:.2f}: candidate has different cluster topology")

    # Test 5: temporal
    t5 = data['test5'].get(asset, {})
    if 'monthly_disagreement' in t5:
        rates = [m['disagreement_rate'] for m in t5['monthly_disagreement']]
        avg = sum(rates) / len(rates) if rates else 0
        score += int(avg * 20)
        chunk34 = t5.get('chunk34_disagreement')
        elev = (chunk34 - t5.get('other_disagreement', 0)) if chunk34 is not None else 0
        print(f"    Test 5 (timeline): avg disagree={avg:.1%}  chunks3-4 elevation={elev:+.1%}")
        if elev > 0.15:
            score += 10
            notes.append("T5: chunks 3-4 elevated → regime shift zone confirmed")

    # Verdict
    if score < 30: severity = 'HEALTHY'
    elif score < 60: severity = 'DRIFTED_BUT_USABLE'
    elif score < 100: severity = 'DRIFTED_RETRAIN_RECOMMENDED'
    else: severity = 'SEVERELY_DRIFTED'

    print(f"    SCORE: {score:>3d} / 200 → {severity}")
    aggregate[asset] = {'score': score, 'severity': severity, 'notes': notes}

print("\n" + "=" * 84)
print("  CROSS-ASSET VERDICT")
print("=" * 84)
for asset, agg in aggregate.items():
    print(f"  {asset}: score={agg['score']:>3d}  {agg['severity']}")

(OUT_DIR / 'summary.json').write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
print(f"\n  Saved: {OUT_DIR / 'summary.json'}")
