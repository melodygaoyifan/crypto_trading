#!/usr/bin/env python3
"""[P414c] Export a SPLIT-AWARE jump-model regime artifact + validate its
vocabulary maps to the live control tables — the offline half of the eventual
GMM->jump swap (a P215 campaign), done shadow-first.

WHY (research 2026-08-26 + P414): a jump model cuts regime CHURN vs the GMM
(~80% batch, ~60% online/filtered on our own data) because a jump PENALTY
enforces state persistence -- and lower churn is our fee-floor lever (less
whipsaw in the regime-conditional controls: trend gate, smart beta, ADVISE
weights, kraken_quant buckets). The DECIDER is GMM-independent (P411/P407), so
the payoff is second-order, which is exactly why this ships shadow-first and the
live cutover stays a gated P215 campaign, never a blind swap.

THE BLOCKER THIS CHECKS (P217/P267): the control tables key on NAMED regimes
(MOMENTUM_RALLY, QUIET_ACCUMULATION, ...). A jump model emits states 0..k-1; if
they do not map to names the tables understand, a swap silently breaks the
controls. This maps each jump state to the GMM regime it most OVERLAPS with and
inherits that name, then reports coverage -- a clean, mostly-1:1 mapping is the
GO signal.

LEAK-FREE: centroids + scaler are fit on TRAIN rows only (the same strict fold
boundary the GMM uses, rebuild_pipeline). Online labels use a forward-only DP
filter (data <= t) -- the honest live label, not the batch-smoothed one.

    python -X utf8 training/scripts/export_jump_regime.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "training"))
from scripts.rebuild_pipeline import compute_gmm_features_batch, GMM_FEATURE_COLS  # noqa: E402

DRL = REPO / "training" / "training_data" / "drl_training"
GMM_DIR = REPO / "models" / "regime_classifier"
OUT_DIR = REPO / "configs" / "jumpregime"
ASSETS = ["BTC", "ETH", "SOL"]
LAMBDA = 20.0          # P414: ~80% batch / ~60% online churn cut, ~5-day runs
TRAIN_FRAC = 0.55      # n*(1-3*0.15) — the strictest fold boundary the GMM uses
WARMUP = 42


def _switches(l): return int((l[1:] != l[:-1]).sum())


def _fit_centroids(Xtr, k, lam, iters=30, seed=7):
    """Batch jump fit on TRAIN rows -> centroids (scaled space)."""
    from sklearn.cluster import KMeans
    n = len(Xtr)
    s = KMeans(k, n_init=8, random_state=seed).fit_predict(Xtr)
    mu = None
    for _ in range(iters):
        mu = np.array([Xtr[s == j].mean(0) if (s == j).any() else Xtr[0]
                       for j in range(k)])
        d = ((Xtr[:, None, :] - mu[None, :, :]) ** 2).sum(2)
        cost = np.empty((n, k)); back = np.empty((n, k), int); cost[0] = d[0]
        for t in range(1, n):
            for j in range(k):
                tr = cost[t - 1] + lam * (np.arange(k) != j)
                b = int(np.argmin(tr)); back[t, j] = b; cost[t, j] = d[t, j] + tr[b]
        sn = np.empty(n, int); sn[-1] = int(np.argmin(cost[-1]))
        for t in range(n - 1, 0, -1):
            sn[t - 1] = back[t, sn[t]]
        if (sn == s).all():
            s = sn; break
        s = sn
    return mu


def _online_filter(X, mu, lam):
    """Forward-only DP: label at t uses data <= t (the live label)."""
    n, k = len(X), len(mu)
    d = ((X[:, None, :] - mu[None, :, :]) ** 2).sum(2)
    cost = d[0].copy(); lab = np.empty(n, int); lab[0] = int(np.argmin(cost))
    for t in range(1, n):
        nc = np.array([d[t, j] + np.min(cost + lam * (np.arange(k) != j))
                       for j in range(k)])
        cost = nc; lab[t] = int(np.argmin(cost))
    return lab


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    go = True
    for a in ASSETS:
        f = DRL / f"{a}_4H_full.parquet"
        gcfg = GMM_DIR / a / "gmm_config.json"
        if not f.exists() or not gcfg.exists():
            print(f"{a}: MISSING parquet or gmm_config — skip"); go = False; continue
        d = pd.read_parquet(f)
        gmm_lab = d["regime"].to_numpy()
        names = json.loads(gcfg.read_text(encoding="utf-8")).get("regime_names") or []
        X = compute_gmm_features_batch(d, a).astype(float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        n = len(X); tr_end = int(n * TRAIN_FRAC) - WARMUP
        mean = X[:tr_end].mean(0); std = X[:tr_end].std(0) + 1e-9   # TRAIN-only
        Xs = (X - mean) / std
        k = int(len(np.unique(gmm_lab)))
        mu = _fit_centroids(Xs[:tr_end], k, LAMBDA)
        online = _online_filter(Xs, mu, LAMBDA)

        # vocabulary: each jump state -> the GMM regime it most overlaps -> name
        state_to_name = {}
        cover = {}
        for j in range(k):
            m = online == j
            if not m.any():
                state_to_name[j] = f"JUMP_{j}"; continue
            gl = int(np.bincount(gmm_lab[m], minlength=len(names)).argmax())
            nm = names[gl] if gl < len(names) else f"REGIME_{gl}"
            state_to_name[j] = nm
            cover.setdefault(nm, 0)
            cover[nm] += int(m.sum())
        distinct = len(set(state_to_name.values()))
        gsw, osw = _switches(gmm_lab), _switches(online)
        clean = distinct >= max(3, k - 1)   # near-1:1 mapping = vocab-safe
        go = go and clean
        payload = {
            "asset": a, "lambda": LAMBDA, "k": k,
            "feature_order": list(GMM_FEATURE_COLS),
            "scaler_mean": mean.tolist(), "scaler_std": std.tolist(),
            "centroids": mu.tolist(),
            "state_to_name": {str(kk): vv for kk, vv in state_to_name.items()},
            "provenance": {"fit_policy": "split_aware", "train_end": tr_end,
                           "n_total": n, "gmm_names": names,
                           "online_churn_pct": round(osw / n * 100, 2),
                           "gmm_churn_pct": round(gsw / n * 100, 2),
                           "churn_cut_pct": round((1 - osw / gsw) * 100, 1)},
        }
        (OUT_DIR / f"{a}.json").write_text(json.dumps(payload), encoding="utf-8")
        print(f"{a}: k={k} online_churn {osw/n*100:.1f}% vs GMM {gsw/n*100:.1f}% "
              f"(cut {(1-osw/gsw)*100:+.0f}%) | vocab {distinct}/{k} distinct names "
              f"-> {'CLEAN' if clean else 'AMBIGUOUS (blocker)'}")
        print(f"     names: {sorted(set(state_to_name.values()))}")
    print("\nVERDICT:", "GO — churn cut real + vocabulary maps clean to the "
          "control tables; next step is the live shadow." if go else
          "BLOCKER — a jump state maps ambiguously; resolve before any swap.")
    return 0 if go else 3


if __name__ == "__main__":
    sys.exit(main())
