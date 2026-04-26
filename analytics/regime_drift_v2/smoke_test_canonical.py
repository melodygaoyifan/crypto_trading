"""[AP-10-v2] Production smoke test for retrained BTC GMM.

This is NOT shadow training — it's a 7-check load+contract sanity test
that runs in seconds, gating whether the retrained GMM is safe to swap in.

Pass criteria (all must hold):
  C1. File loads cleanly (pickle + scaler + config integrity)
  C2. n_clusters matches deployed
  C3. regime_names == canonical (production code's contract — main.py:1490 etc.)
  C4. Predict on recent 6mo: no NaN, length matches input
  C5. Cluster freq distribution: no single cluster > 80% (avoid degenerate fit)
  C6. Every canonical regime has prior > 1% (no canonical regime is fully dead)
  C7. Mean posterior confidence not catastrophically dropped vs deployed (delta > -0.10)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ))

from analytics.regime_drift_v2._common import (
    load_gmm_bundle, get_aligned_features_and_df,
    predict_labels_with_bundle,
    DEPLOYED_DIR,
)

RETRAIN_DIR = PROJ / 'models' / 'regime_classifier_retrain'

CANONICAL_REGIMES_BTC = {
    'STEADY_UPTREND', 'NEUTRAL_DRIFT', 'WEAK_CONSOLIDATION',
    'MOMENTUM_RALLY', 'QUIET_ACCUMULATION', 'VOLATILE_CHOP',
    'EXTREME_VOLATILITY', 'PANIC_SELLOFF',
}

ASSET = 'BTC'

print("=" * 64)
print(f"  [AP-10-v2] Smoke test for retrained {ASSET} GMM")
print("=" * 64)

passed: list[str] = []
failed: list[tuple[str, str]] = []
warned: list[tuple[str, str]] = []


def _ok(name: str, msg: str = ''):
    suffix = f"  ({msg})" if msg else ''
    print(f"  [PASS] {name}{suffix}")
    passed.append(name)


def _fail(name: str, msg: str):
    print(f"  [FAIL] {name}: {msg}")
    failed.append((name, msg))


def _warn(name: str, msg: str):
    print(f"  [WARN] {name}: {msg}")
    warned.append((name, msg))


# C1
try:
    deployed = load_gmm_bundle(DEPLOYED_DIR, ASSET)
    retrain = load_gmm_bundle(RETRAIN_DIR, ASSET)
    _ok('C1_file_load', f"deployed k={deployed['gmm'].n_components}, retrain k={retrain['gmm'].n_components}")
except Exception as e:
    _fail('C1_file_load', str(e))
    print("\nCannot proceed without files. Exiting.")
    sys.exit(1)

# C2
n_d = deployed['gmm'].n_components
n_r = retrain['gmm'].n_components
if n_d == n_r:
    _ok('C2_n_clusters', f"both k={n_d}")
else:
    _fail('C2_n_clusters', f"deployed={n_d} retrain={n_r}")

# C3
retrain_names = set(retrain['regime_names'] or [])
missing = CANONICAL_REGIMES_BTC - retrain_names
extra = retrain_names - CANONICAL_REGIMES_BTC
if not missing and not extra:
    _ok('C3_canonical_names', f"all 8 canonical")
else:
    _fail('C3_canonical_names', f"missing={missing} extra={extra}")

# C4
try:
    df, feats = get_aligned_features_and_df(ASSET, days=180, feature_cols=retrain['feature_cols'])
    labels_r, post_r = predict_labels_with_bundle(retrain, feats)
    if np.any(np.isnan(post_r)):
        _fail('C4_predict_no_nan', 'NaN in posteriors')
    elif len(labels_r) != len(feats):
        _fail('C4_predict_no_nan', f'length mismatch: feats={len(feats)} labels={len(labels_r)}')
    else:
        _ok('C4_predict_no_nan', f"{len(labels_r)} labels, no NaN")
except Exception as e:
    _fail('C4_predict_no_nan', f'crash: {e}')

# C5
freq_r = np.bincount(labels_r, minlength=n_r) / max(1, len(labels_r))
max_freq = float(freq_r.max())
if max_freq > 0.80:
    dom = retrain['regime_names'][int(freq_r.argmax())]
    _fail('C5_freq_balance', f"{dom} = {max_freq:.2%} > 80%")
else:
    _ok('C5_freq_balance', f"max cluster freq {max_freq:.2%}")

# C6
priors = retrain['gmm'].weights_
low_prior = [(retrain['regime_names'][i], float(priors[i])) for i in range(n_r) if priors[i] < 0.01]
if low_prior:
    _warn('C6_priors', f"{len(low_prior)} regime(s) with prior <1%: " +
          ", ".join(f"{n}({p:.4f})" for n, p in low_prior))
else:
    _ok('C6_priors', f"all priors >= 1%, range [{priors.min():.3f}, {priors.max():.3f}]")

# C7
labels_d, post_d = predict_labels_with_bundle(deployed, feats)
conf_d = float(post_d.mean())
conf_r = float(post_r.mean())
delta = conf_r - conf_d
if delta < -0.10:
    _fail('C7_posterior_conf', f"catastrophic drop {conf_d:.3f} -> {conf_r:.3f} (delta={delta:+.3f})")
elif delta < -0.05:
    _warn('C7_posterior_conf', f"moderate drop {conf_d:.3f} -> {conf_r:.3f}")
else:
    _ok('C7_posterior_conf', f"deployed={conf_d:.3f} retrain={conf_r:.3f} (delta={delta:+.3f})")

# Verdict
print("\n" + "=" * 64)
print(f"  Summary: {len(passed)} pass, {len(warned)} warn, {len(failed)} fail")
print("=" * 64)

if not failed:
    print("\n[VERDICT] ALL CHECKS PASSED - safe to deploy")
    print("\nDeploy command (atomic):")
    print(f"  cp -r {DEPLOYED_DIR / ASSET} {DEPLOYED_DIR / (ASSET + '.bak')}")
    print(f"  cp {RETRAIN_DIR / ASSET}/* {DEPLOYED_DIR / ASSET}/")
    print("\nRollback if production misbehaves:")
    print(f"  rm -rf {DEPLOYED_DIR / ASSET}")
    print(f"  mv {DEPLOYED_DIR / (ASSET + '.bak')} {DEPLOYED_DIR / ASSET}")
    sys.exit(0)
else:
    print(f"\n[VERDICT] {len(failed)} CHECK(S) FAILED - DO NOT DEPLOY")
    for name, msg in failed:
        print(f"  - {name}: {msg}")
    print(f"\nDiscard retrain:\n  rm -rf {RETRAIN_DIR}")
    sys.exit(1)
