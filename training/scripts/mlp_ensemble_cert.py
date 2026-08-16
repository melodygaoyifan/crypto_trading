#!/usr/bin/env python3
"""[P285c] The 10-seed ensemble certification of BTC mlp_small — the
standard remedy for the P285b FRAGILE verdict, measured before anything
else is decided.

P285b measured the certified config's seed dispersion: median decisive
Sharpe −0.09 across seeds 1..10, with the certified seed 7 the second-best
draw — the pooled pass is substantially seed luck. The standard fix is an
ENSEMBLE: average the 10 seeds' predictions per refit (variance reduction
with NO selection — no seed is picked). This script runs the ensemble
through the EXACT pooled machinery (T.walk_forward_z, same purge/refit,
same fold windows, the P283b decisive cell: LIVE sign expression, carry
included, block-bootstrap CI).

PRE-COMMITTED DISPOSITION (written before running):
  * CI excludes zero -> the ensemble becomes THE candidate: re-export the
    shadow as the 10-seed ensemble (ledger discontinuity recorded; its
    14d/30d clocks restart at the swap) and the P285 early-seat criterion
    re-arms against the ensemble's ledger.
  * CI includes zero -> mlp_small is DEAD as a candidate (the family joins
    the supervised corpse pile honestly); the P285 early-seat criterion is
    WITHDRAWN; the shadow may keep running as a free ledger but nothing
    fires from it.
Window reads ledgered under mlp_ens:p285c (re-aggregation of already-spent
fold-val windows — and note honestly: these windows have now been read
MANY times; even a PASS carries the P260 discount, which is why the
forward ledger remains the exam).
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "training"))
sys.path.insert(0, str(REPO / "training" / "scripts"))

from splits import assert_clean_gmm, record_window_usage  # noqa: E402
import train_supervised_full as T  # noqa: E402
from regime_model_lab import _bar_carry_rate  # noqa: E402
from pooled_certification import live_expression, dsr, CANDIDATES  # noqa: E402

from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402

ASSET = "BTC"
SEEDS = tuple(range(1, 11))
N_TRIALS = 10   # the zoo's 8 + the ensemble as a distinct trial + deep-seq
                # spillover is charged in their own report; keep >= zoo's 8
OUT = REPO / "training" / "reports" / "mlp_ensemble_cert_p285c.json"


class EnsembleMLP(T.Candidate):
    """The certified mlp config, 10-seed prediction-averaged. Mirrors the
    zoo's mlp branch EXACTLY (scaler, width 32, alpha 1e-2, max_iter 200,
    early stopping) — only random_state varies, and the output is the
    mean; sig = std of the mean train prediction (the same normalization
    convention as every zoo family)."""

    def __init__(self):
        super().__init__("mlp_ens10", "mlp_ens", "ret", "top24", refit=42)

    def fit_predict(self, Xtr, ytr, Xpred, regime_tr=None, regime_pred=None):
        sc = StandardScaler().fit(Xtr)
        Xs, Xp = sc.transform(Xtr), sc.transform(Xpred)
        tr_preds, pr_preds = [], []
        for seed in SEEDS:
            m = MLPRegressor(hidden_layer_sizes=(32,), alpha=1e-2,
                             max_iter=200, early_stopping=True,
                             random_state=seed).fit(Xs, ytr)
            tr_preds.append(m.predict(Xs))
            pr_preds.append(m.predict(Xp))
        tr_avg = np.mean(tr_preds, axis=0)
        return np.mean(pr_preds, axis=0), float(np.std(tr_avg))


def main() -> int:
    assert_clean_gmm(ASSET)
    X, targets, close, gmm, feats = T.load_asset(ASSET)
    n = len(close)
    y = targets["ret"]
    cand = EnsembleMLP()
    pos_full = np.zeros(n)
    spans = []
    for fold, (train_end, val_start, val_end) in T.fold_splits(n).items():
        fsets = T.select_features(X, targets["ret"], train_end, feats)
        z = T.walk_forward_z(cand, X, y, None, fsets, val_start, val_end)
        pos_full[val_start:val_end] = T.positions_from_z(
            z[val_start:val_end], None)
        spans.append((val_start, val_end))
        print(f"  {fold} done", flush=True)
    lo = min(s for s, _ in spans); hi = max(e for _, e in spans)
    record_window_usage("mlp_ens:p285c", ASSET, lo, hi, "validation")
    mask = np.zeros(n, dtype=bool)
    for s, e in spans:
        mask[s:e] = True
    carry_rate = np.nan_to_num(
        np.asarray(_bar_carry_rate(ASSET, n), dtype=float))
    ret = np.zeros(n); ret[1:] = close[1:] / close[:-1] - 1.0
    pos = live_expression(pos_full)
    strat = np.zeros(n)
    strat[1:] = pos[:-1] * ret[1:] - pos[:-1] * carry_rate[1:]
    cost = np.zeros(n)
    cost[1:] = np.abs(np.diff(pos)) * (T.COST_BPS[ASSET] / 2.0) / 1e4
    seg = (strat - cost)[mask]
    sd = float(np.nanstd(seg))
    sh = float(np.nanmean(seg) / sd * math.sqrt(T.BARS_PER_YEAR)) \
        if sd > 0 else 0.0
    clo, chi = T.block_bootstrap_ci(seg)
    passes = clo is not None and clo > 0
    out = {"generated": datetime.now(timezone.utc).isoformat(),
           "asset": ASSET, "candidate": "mlp_ens10 (seeds 1..10 averaged)",
           "cell": "LIVE_sign_expression.with_carry",
           "pooled_bars": int(mask.sum()),
           "net_pct": round(float(np.nansum(seg)) * 100, 2),
           "sharpe": round(sh, 3),
           "ci": [None if clo is None else round(clo, 3),
                  None if chi is None else round(chi, 3)],
           "dsr": round(dsr(sh, int(mask.sum()), N_TRIALS, seg=seg), 3),
           "passes": passes,
           "disposition": ("PASS -> ensemble becomes the shadow candidate "
                           "(re-export, clocks restart); FAIL -> mlp_small "
                           "is dead as a candidate, P285 early seat "
                           "WITHDRAWN — pre-committed in the docstring")}
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\n[{ASSET}] mlp_ens10: net {out['net_pct']:+7.2f}%  "
          f"sh {out['sharpe']:+.2f}  CI[{out['ci'][0]},{out['ci'][1]}]  "
          f"DSR {out['dsr']:.2f}  {'PASS' if passes else 'FAIL'}")
    print(f"report: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
