#!/usr/bin/env python3
"""[research 2026-08-26] Jump model vs GMM regime CHURN, on OUR own features.

The RL/signal literature sweep (2026-08-26) found jump models beat GMM/HMM on
TURNOVER (Shu/Kolm/Mulvey, arXiv 2402.05272: ~69% fewer regime switches, Sharpe
0.68 vs 0.54) because a jump PENALTY enforces state persistence -- structurally
the same idea as our RegimeSmoother, done inside the model. The win is "fewer
flips / lower churn, NOT higher IC" -- and lower churn is exactly the fee-floor
lever we care about (fewer regime flips -> the regime-conditional controls fire
less, less whipsaw).

This is the DISCIPLINED FIRST STEP before any live swap: a live GMM->jump swap is
a P215 atomic-artifact campaign (parquets+checkpoints+revalidate the whole
regime-conditional stack). Measure the churn win on our historical features
FIRST; only a real win justifies that campaign. Observation-only, no runtime
change, no new live dependency.

HONESTY: both the GMM labels (stored) and this jump fit are BATCH/whole-series
fits, so this measures the jump-penalty's churn reduction on identical features,
NOT the live filtered-online churn (the research's filtering-lag caveat applies
to both equally). A live jump model would use the online/filtered variant.

    python -X utf8 training/scripts/jump_vs_gmm_churn_probe.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "training"))
from scripts.rebuild_pipeline import compute_gmm_features_batch  # noqa: E402

DRL = REPO / "training" / "training_data" / "drl_training"
ASSETS = ["BTC", "ETH", "SOL"]
LAMBDAS = [0.0, 5.0, 20.0, 50.0, 100.0]   # jump penalty; 0 ~= no persistence


def _switches(labels: np.ndarray) -> int:
    return int((labels[1:] != labels[:-1]).sum())


def _fit_jump(X: np.ndarray, k: int, lam: float, iters: int = 30, seed: int = 7):
    """Minimal jump model: coordinate descent between centroids and a DP over
    states with a jump penalty (Nystrup/Lindstrom/Madsen 2020 form)."""
    from sklearn.cluster import KMeans
    n = len(X)
    s = KMeans(n_clusters=k, n_init=5, random_state=seed).fit_predict(X)
    for _ in range(iters):
        mu = np.array([X[s == j].mean(axis=0) if (s == j).any() else X[np.random.randint(n)]
                       for j in range(k)])
        d = ((X[:, None, :] - mu[None, :, :]) ** 2).sum(axis=2)  # (n,k) sq-dist
        # DP with jump penalty lam
        cost = np.empty((n, k)); back = np.empty((n, k), dtype=int)
        cost[0] = d[0]
        for t in range(1, n):
            prev = cost[t - 1]
            for j in range(k):
                trans = prev + lam * (np.arange(k) != j)
                b = int(np.argmin(trans))
                back[t, j] = b
                cost[t, j] = d[t, j] + trans[b]
        s_new = np.empty(n, dtype=int)
        s_new[-1] = int(np.argmin(cost[-1]))
        for t in range(n - 1, 0, -1):
            s_new[t - 1] = back[t, s_new[t]]
        if (s_new == s).all():
            s = s_new; break
        s = s_new
    return s


def main() -> int:
    print(f"{'asset':5} {'k':>2} {'GMM sw':>7} {'GMM %':>6} {'GMM run':>7}  "
          f"| jump switches by lambda {LAMBDAS}")
    for a in ASSETS:
        f = DRL / f"{a}_4H_full.parquet"
        if not f.exists():
            print(f"{a}: MISSING {f}"); continue
        d = pd.read_parquet(f)
        if "regime" not in d.columns:
            print(f"{a}: no stored regime column"); continue
        gmm = d["regime"].to_numpy()
        k = int(len(np.unique(gmm)))
        gsw = _switches(gmm)
        X = compute_gmm_features_batch(d, a).astype(float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X = (X - X.mean(0)) / (X.std(0) + 1e-9)
        row = []
        for lam in LAMBDAS:
            s = _fit_jump(X, k, lam)
            jsw = _switches(s)
            run = len(s) / (jsw + 1)
            cut = (1 - jsw / gsw) * 100 if gsw else 0.0
            row.append(f"{jsw}({cut:+.0f}%,run{run:.0f})")
        n = len(d)
        print(f"{a:5} {k:>2} {gsw:>7} {gsw/n*100:>5.1f}% {n/(gsw+1):>6.1f}  | "
              + "  ".join(row))
    print("\nRead: GMM sw = stored regime switches (per-bar independent classify).")
    print("jump = same features, jump penalty lambda. cut% = fewer switches vs GMM;")
    print("run = avg bars per regime. A large cut at a lambda where run is ~days,")
    print("not ~weeks, is the churn win the research predicts (P221 addendum candidate).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
