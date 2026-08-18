"""[P307] What do the GMM's three non-informative inputs actually cost?

P306 recorded that `return_1h` is exactly `return_4h / 4`, so after the
StandardScaler the two columns are IDENTICAL. P221 recorded that
`cross_asset_correlation` and `spread_percentile` are per-asset CONSTANTS,
so after scaling they are all-zero columns. That leaves the classifier with
9 informative inputs out of 12.

"Carries no information" is the easy half of the claim and it is not the
part that matters. In a full-covariance GMM a duplicated column is not
neutral: the covariance matrix is singular in that plane (regularised here
by reg_covar=1e-2), and the duplicated direction is counted TWICE in the
Mahalanobis distance that assigns every bar to a cluster. So the honest
question is whether `return_4h` has been silently double-weighted in the
live regime classifier, and by how much.

This probe answers it by refitting, per asset, with the EXACT production
recipe (same scaler, same GMM_BASE_CONFIG, same k=3..8 BIC search with the
2% minimum-regime rule, same split-aware fit boundary) under three feature
sets:

    full     the 12 production columns
    no_dup   drop return_1h  (11 columns)
    no_dead  drop return_1h + the two constants  (9 columns)

and comparing k, the Adjusted Rand Index of the bar-by-bar assignments, the
regime census and the mean max-posterior.

It CHANGES NOTHING. A feature-set change is a P215 atomic operation on
{GMM, parquets, checkpoints} and needs this measurement first, not after.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from sklearn.metrics import adjusted_rand_score          # noqa: E402
from sklearn.mixture import GaussianMixture              # noqa: E402
from sklearn.preprocessing import StandardScaler         # noqa: E402

from training.scripts.rebuild_pipeline import (          # noqa: E402
    GMM_BASE_CONFIG, GMM_FEATURE_COLS, compute_gmm_features_batch,
    gmm_fit_boundary)

REPORT = REPO / "training" / "reports" / "gmm_redundancy_probe_p307.json"
DATA = REPO / "training" / "training_data" / "drl_training"


def _fit(X_scaled):
    """The production BIC search, verbatim in behaviour."""
    best = (None, np.inf, None)
    for k in range(3, 9):
        g = GaussianMixture(**dict(GMM_BASE_CONFIG, n_components=k), verbose=0)
        g.fit(X_scaled)
        counts = Counter(g.predict(X_scaled))
        if min(counts.values()) / len(X_scaled) < 0.02:
            continue
        bic = g.bic(X_scaled)
        if bic < best[1]:
            best = (k, bic, g)
    if best[2] is None:
        g = GaussianMixture(**dict(GMM_BASE_CONFIG, n_components=6), verbose=0)
        g.fit(X_scaled)
        best = (6, g.bic(X_scaled), g)
    return best


def run(assets=("BTC", "ETH", "SOL")) -> dict:
    import pandas as pd
    out = {"generated": datetime.now(timezone.utc).isoformat(),
           "features": list(GMM_FEATURE_COLS), "assets": {}}
    for a in assets:
        df = pd.read_parquet(DATA / f"{a}_4H_full.parquet")
        # The 12 GMM inputs are NOT parquet columns - they are recomputed by
        # the pipeline's own builder, which is the single source of truth for
        # what the classifier actually sees (P172).
        M = compute_gmm_features_batch(df, asset=a)
        valid = np.where(~np.isnan(M).any(axis=1))[0]
        fit_idx = valid[:gmm_fit_boundary(len(valid))]
        base = M[fit_idx]

        # the collinearity claim, verified rather than assumed
        i1 = GMM_FEATURE_COLS.index("return_1h")
        i4 = GMM_FEATURE_COLS.index("return_4h")
        ratio = base[:, i4] / np.where(base[:, i1] != 0, base[:, i1], np.nan)
        rec = {
            "fit_bars": int(len(base)),
            "return_1h_is_return_4h_over_4": bool(
                np.nanmax(np.abs(ratio - 4.0)) < 1e-6),
            "corr_ret1h_ret4h": round(float(
                np.corrcoef(base[:, i1], base[:, i4])[0, 1]), 12),
            "constant_columns": [c for j, c in enumerate(GMM_FEATURE_COLS)
                                 if float(np.std(base[:, j])) < 1e-12],
            "variants": {},
        }
        drop = {
            "full": [],
            "no_dup": ["return_1h"],
            "no_dead": ["return_1h"] + rec["constant_columns"],
        }
        labels = {}
        for name, dropped in drop.items():
            keep = [j for j, c in enumerate(GMM_FEATURE_COLS)
                    if c not in dropped]
            Xs = StandardScaler().fit_transform(base[:, keep])
            k, bic, g = _fit(Xs)
            lab = g.predict(Xs)
            proba = g.predict_proba(Xs)
            labels[name] = lab
            cnt = Counter(lab)
            rec["variants"][name] = {
                "n_features": len(keep),
                "dropped": dropped,
                "k": int(k),
                "bic": round(float(bic), 1),
                "mean_max_posterior": round(float(proba.max(axis=1).mean()), 4),
                "census_pct": {str(c): round(100.0 * n / len(lab), 2)
                               for c, n in sorted(cnt.items())},
            }
        rec["ari_full_vs_no_dup"] = round(
            float(adjusted_rand_score(labels["full"], labels["no_dup"])), 4)
        rec["ari_full_vs_no_dead"] = round(
            float(adjusted_rand_score(labels["full"], labels["no_dead"])), 4)
        out["assets"][a] = rec
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default="BTC,ETH,SOL")
    args = ap.parse_args()
    rep = run(tuple(x.strip().upper() for x in args.assets.split(",")))
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    for a, r in rep["assets"].items():
        if "error" in r:
            print(f"{a}: {r['error']}")
            continue
        print(f"\n=== {a}  fit_bars={r['fit_bars']}  "
              f"ret1h==ret4h/4: {r['return_1h_is_return_4h_over_4']}  "
              f"corr={r['corr_ret1h_ret4h']}  "
              f"constants={r['constant_columns']}")
        for n, v in r["variants"].items():
            print(f"  {n:<9} nfeat={v['n_features']:>2} k={v['k']} "
                  f"BIC={v['bic']:>12,.0f} maxpost={v['mean_max_posterior']:.4f}")
        print(f"  ARI full vs no_dup  = {r['ari_full_vs_no_dup']}")
        print(f"  ARI full vs no_dead = {r['ari_full_vs_no_dead']}")
    print(f"\nreport -> {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
