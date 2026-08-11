"""[P263c] Deep debug of the banded-forecast family — verify the asserted
mechanisms and locate exactly what died where.

Three verifications + one discriminator:
  1. DOUBLE-NORMALIZATION (asserted in P263b): do the vol-scaled variant's
     positions actually migrate toward LOW-vol bars vs the raw variant?
  2. POOLING HETEROGENEITY (asserted in P263b): do per-asset ridge
     coefficients disagree in SIGN on major features?
  3. WHAT DECAYED (the P259b death): decompose the E-variant's increment by
     year, regime cell and direction — design vs validation.
  4. THE DISCRIMINATOR: forecast Spearman IC per era. If IC collapses at
     validation -> the SIGNAL is era-local (nothing to salvage by better
     expression). If IC holds but PnL dies -> expression/cost problem
     (potentially fixable). This is the question that decides whether any
     further engineering on this family can ever pay.

WINDOW ACCOUNTING: parts 3-4 read validation-era statistics for the SAME
candidate family whose validation read was already spent (P259b). Recorded
in the window ledger as a DIAGNOSIS of that spent read — deeper metrics of
a known verdict, not a new candidate look. Every future design that uses
these diagnostics inherits the contamination; that is what the ledger row
is for.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from scipy import stats  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402

from training.regime_model_lab import _ctx, DESIGN  # noqa: E402
from training.train_supervised_full import COST_BPS  # noqa: E402
from training.mechanism_lab import book_targets  # noqa: E402
from training.banded_forecast_lab import close_features  # noqa: E402
from training.pooled_banded_lab import banded_pos, _vol20, H, GAP, ALPHA  # noqa: E402
from training.splits import record_window_usage  # noqa: E402

DS, DE = DESIGN
PRE = (800, 3000)
HOME = ("BTC", "ETH", "SOL")
OUT = REPO / "training" / "reports" / "p263_debug2.json"


def wf_forecast(X, close, vol, vol_scaled, refit, end):
    n = len(close)
    y = np.full(n, np.nan)
    y[:-H] = close[H:] / close[:-H] - 1.0
    if vol_scaled:
        y = y / (vol * np.sqrt(H) + 1e-9)
    pred = np.full(n, np.nan)
    coefs = []
    for t0 in range(DS - 500, end, refit):
        hi_tr = min(t0 - GAP, n)
        ok = ~np.isnan(X[:hi_tr]).any(1) & ~np.isnan(y[:hi_tr]) & ~np.isnan(vol[:hi_tr])
        if ok.sum() < 800:
            continue
        mu, sd = X[:hi_tr][ok].mean(0), X[:hi_tr][ok].std(0) + 1e-12
        m = Ridge(alpha=ALPHA).fit((X[:hi_tr][ok] - mu) / sd, y[:hi_tr][ok])
        coefs.append(m.coef_.copy())
        hi = min(t0 + refit, end)
        seg = ~np.isnan(X[t0:hi]).any(1)
        p = np.full(hi - t0, np.nan)
        p[seg] = m.predict((X[t0:hi][seg] - mu) / sd)
        pred[t0:hi] = p
    return pred, np.mean(coefs, axis=0) if coefs else None


def main() -> int:
    report = {"generated": datetime.now(timezone.utc).isoformat()}
    data = {}
    for a in HOME:
        c = _ctx(a)
        close = c["close"]
        X, names = close_features(close, c["fz"])
        data[a] = dict(close=close, lab=c["lab"], fz=c["fz"], X=X,
                       vol=_vol20(close), names=names, n=len(close))

    # ---- 1. DOUBLE-NORMALIZATION: position vol-tercile occupancy ----
    occ = {}
    for a in HOME:
        d = data[a]
        p_raw, _ = wf_forecast(d["X"], d["close"], d["vol"], False, 500, DE)
        p_vs, _ = wf_forecast(d["X"], d["close"], d["vol"], True, 250, DE)
        import pandas as pd
        v = pd.Series(d["vol"][DS:DE])
        ter = v.rank(pct=True)
        def occupancy(pred, te, tx):
            pos = banded_pos(pred, d["lab"], te, tx)[DS:DE]
            act = np.abs(pos) > 0
            if act.sum() < 20:
                return None
            r = ter.values[act]
            return {"lo": round(float((r < 1/3).mean()), 3),
                    "mid": round(float(((r >= 1/3) & (r < 2/3)).mean()), 3),
                    "hi": round(float((r >= 2/3).mean()), 3),
                    "n_active_bars": int(act.sum())}
        occ[a] = {"raw": occupancy(p_raw, 1.0, 0.25),
                  "vol_scaled": occupancy(p_vs, 1.5, 0.25)}
        print(f"[1] {a} occupancy raw={occ[a]['raw']} vs={occ[a]['vol_scaled']}")
    report["1_vol_tercile_occupancy"] = occ

    # ---- 2. POOLING HETEROGENEITY: coefficient sign agreement ----
    cmat = {}
    for a in HOME:
        d = data[a]
        _, cf = wf_forecast(d["X"], d["close"], d["vol"], False, 500, DE)
        cmat[a] = cf
    names = data["BTC"]["names"]
    disagree = []
    for j, nm in enumerate(names):
        signs = [np.sign(cmat[a][j]) for a in HOME]
        mags = [abs(cmat[a][j]) for a in HOME]
        if len(set(signs)) > 1 and max(mags) > 0.0003:
            disagree.append({"feature": nm,
                             **{a: round(float(cmat[a][j]), 5) for a in HOME}})
    report["2_coef_sign_disagreements"] = disagree
    print(f"[2] sign-disagreeing major features: "
          f"{[d['feature'] for d in disagree]}")

    # ---- 3+4. E-variant era decomposition + IC discriminator ----
    for a in HOME:
        record_window_usage(
            "p263_debug2", a, 9100, data[a]["n"],
            "DIAGNOSIS of the spent P259b validation read — IC/decomposition "
            "of the same candidate family, no new candidate")
    dec = {}
    for a in HOME:
        d = data[a]
        n = d["n"]
        pred, _ = wf_forecast(d["X"], d["close"], d["vol"], False, 500, n)
        y = np.full(n, np.nan)
        y[:-H] = d["close"][H:] / d["close"][:-H] - 1.0
        eras = {"design": (DS, DE), "pre_design": PRE,
                "validation": (9100, n)}
        ics = {}
        for nm, (lo, hi) in eras.items():
            m = ~np.isnan(pred[lo:hi]) & ~np.isnan(y[lo:hi])
            ics[nm] = (round(float(stats.spearmanr(
                pred[lo:hi][m], y[lo:hi][m]).statistic), 4)
                if m.sum() > 200 else None)
        # increment decomposition (validation): by direction of the banded
        # entries in book-flat cells
        book = book_targets(a, d["lab"], d["fz"])
        posb = banded_pos(pred, d["lab"], 1.0, 0.25)
        active = (book == 0.0) & (np.abs(posb) > 0)
        r1 = np.zeros(n)
        r1[:-1] = d["close"][1:] / d["close"][:-1] - 1.0
        seg = {}
        for nm, (lo, hi) in eras.items():
            sl = slice(lo, hi - 1)
            longs = float(np.nansum(np.where(
                active[sl] & (posb[sl] > 0), r1[sl], 0.0)))
            shorts = float(np.nansum(np.where(
                active[sl] & (posb[sl] < 0), -r1[sl], 0.0)))
            seg[nm] = {"long_gross": round(longs, 4),
                       "short_gross": round(shorts, 4),
                       "active_bars": int(active[sl].sum())}
        dec[a] = {"forecast_ic_by_era": ics, "flat_cell_entries": seg}
        print(f"[3/4] {a} IC by era: {ics}")
        print(f"      flat-cell entries: {seg}")
    report["3_4_era_decomposition"] = dec

    OUT.write_text(json.dumps(report, indent=1, default=str),
                   encoding="utf-8")
    print(f"report: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
