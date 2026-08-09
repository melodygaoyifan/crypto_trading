"""[P249] Residual-driven feature pass — engineer features that explain the
surviving books' LOSING bars, judged by the ladder rule.

Method (design era only, purged CV, trial-counted):
  1. DIAGNOSE: per asset, compute the p247 book's per-bar PnL, isolate the
     bars where it LOSES while positioned, and profile them (regime, vol
     tercile, funding state, streak length) — losses must name their
     conditions before features are invented for them.
  2. ENGINEER: candidate features = per-regime crosses of the top-|IC|
     base features (feature x regime flag), vol-normalized returns, and
     loss-motivated conditions (drawdown-from-20d-high, vol-of-vol,
     funding x trend interaction). All causal by construction from causal
     inputs.
  3. JUDGE by the ladder rule: the SOL bear ridge (the roster's one
     trained model) is re-run with base vs base+engineered features on
     identical purged folds. Engineered features earn their place ONLY by
     beating the base set's CV realized gain — otherwise they are noise
     with a story.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from training.regime_model_lab import (  # noqa: E402
    _ctx, cell_series, assembled_series, REGIME_ID,
)
from training.splits import DESIGN_ERA, purged_folds, record_window_usage  # noqa: E402
from training.eval_report import seg_metrics  # noqa: E402
from training.provenance import provenance_stamp  # noqa: E402
from scipy import stats  # noqa: E402

P247_WINNERS = {
    "BTC": {"bull": {"kind": "hold", "params": {}},
            "bear": {"kind": "funding_short", "params": {"thr": 1.0}},
            "peace": {"kind": "funding_contrarian", "params": {"thr": 0.5}}},
    "ETH": {"bull": {"kind": "hold", "params": {}},   # trend-only
            "bear": {"kind": "flat", "params": {}},
            "peace": {"kind": "flat", "params": {}}},
    "SOL": {"bull": {"kind": "hold", "params": {}},
            "bear": {"kind": "ridge_defensive", "params": {"alpha": 30.0}},
            "peace": {"kind": "flat", "params": {}}},
}


def engineer(ctx):
    """Return (F, names): engineered feature matrix, all causal."""
    X, close, lab, fz, n = ctx["X"], ctx["close"], ctx["lab"], ctx["fz"], ctx["n"]
    feats = ctx["feats"]
    y = ctx["y"]
    s, e = DESIGN_ERA
    # top-6 base features by |design-era IC| (rank corr)
    m = (np.arange(n) >= s) & (np.arange(n) < e) & ~np.isnan(y)
    ics = []
    ry = np.argsort(np.argsort(y[m]))
    for i in range(X.shape[1]):
        col = X[m][:, i]
        if np.std(col) == 0 or np.isnan(col).any():
            ics.append(0.0); continue
        ics.append(abs(float(stats.spearmanr(col, ry).statistic)))
    top = list(np.argsort(ics)[::-1][:6])

    cols, names = [], []
    bull = (lab == 1).astype(float)
    bear = (lab == 2).astype(float)
    for i in top:
        cols.append(X[:, i] * bull); names.append(f"x_{feats[i]}_bull")
        cols.append(X[:, i] * bear); names.append(f"x_{feats[i]}_bear")
    # loss-motivated conditions
    c = pd.Series(close)
    dd20 = (close / c.rolling(120).max().to_numpy()) - 1.0
    cols.append(dd20); names.append("dd_from_20d_high")
    r1 = np.full(n, np.nan); r1[1:] = np.log(close[1:] / close[:-1])
    vol = pd.Series(r1).rolling(42).std()
    volvol = vol.rolling(42).std().to_numpy()
    cols.append(volvol); names.append("vol_of_vol_7d")
    trend = np.sign(close - c.rolling(200).mean().to_numpy())
    cols.append(fz * trend); names.append("funding_x_trend")
    F = np.column_stack([np.nan_to_num(np.asarray(col, dtype=float)) for col in cols])
    return F, names


def diagnose(ctx, asset):
    s, e = DESIGN_ERA
    seg = assembled_series(ctx, P247_WINNERS[asset], s, e, instrument="perp")
    lab_seg = ctx["lab"][s:e]
    loss = seg < -0.002       # >20bps loss bars
    print(f"  loss bars (<-20bps): {int(loss.sum())}/{len(seg)}; by regime: "
          + ", ".join(f"{r}={int((loss & (lab_seg == REGIME_ID[r])).sum())}"
                      for r in ("bull", "bear", "peace")), flush=True)
    r1 = np.diff(np.log(ctx["close"][s - 1:e]))
    vol = pd.Series(r1).rolling(42).std().to_numpy()
    terc = pd.qcut(pd.Series(vol), 3, labels=False, duplicates="drop").to_numpy()
    by_terc = [round(float(seg[terc == k].sum()) * 100, 1) for k in (0, 1, 2)]
    print(f"  pnl% by vol tercile (low->high): {by_terc}", flush=True)


def judge_sol_bear(ctx, F, names):
    """Ladder rule: base vs base+engineered on identical purged folds."""
    s, e = DESIGN_ERA
    results = {}
    for label, X_use in (("base", ctx["X"]),
                         ("base+engineered", np.column_stack([ctx["X"], F]))):
        ctx2 = dict(ctx); ctx2["X"] = X_use
        cv = []
        for tr, va in purged_folds(s, e):
            seg = cell_series("ridge_defensive", {"alpha": 30.0}, ctx2, "bear",
                              int(va[0]), int(va[-1] + 1), fit_lt=int(va[0]),
                              instrument="perp")
            cv.append(seg_metrics(seg)["pnl_pct"])
        results[label] = round(float(np.mean(cv)), 2)
        print(f"  SOL bear ridge [{label:<16}] CV pnl={results[label]:+.2f}%",
              flush=True)
    verdict = "EARNED" if results["base+engineered"] > results["base"] else "NOT EARNED"
    print(f"  ladder verdict: engineered features {verdict} "
          f"({results['base+engineered']:+.2f} vs {results['base']:+.2f})", flush=True)
    return results, verdict


def main():
    out = {"assets": {}, "provenance": provenance_stamp()}
    for asset in ("BTC", "ETH", "SOL"):
        ctx = _ctx(asset); ctx["asset"] = asset
        record_window_usage("residual_pass:p249", asset, *DESIGN_ERA, "design")
        print(f"\n########## {asset} residual diagnosis ##########", flush=True)
        diagnose(ctx, asset)
        F, names = engineer(ctx)
        out["assets"][asset] = {"engineered": names}
        if asset == "SOL":
            res, verdict = judge_sol_bear(ctx, F, names)
            out["assets"][asset]["sol_bear_ladder"] = {**res, "verdict": verdict}
    p = REPO / "training" / "reports" / "residual_pass_p249.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nDONE -> {p}", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
