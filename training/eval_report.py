"""[GP0] Standard evaluation report — train/CV/test side by side, overfit
gap as a first-class number, and the robustness battery as a stage.

The P243 composite falsification found a "winner" that collapsed outside
its design window — after the celebratory entry was written. The lesson,
made structural: no candidate's evaluation is complete without (a) its
train-vs-test gap printed next to the test number, and (b) the robustness
battery: era stability, parameter perturbation, cost perturbation. A
candidate whose edge only exists at one parameter setting, one era, or one
cost assumption is a fitted artifact wearing a strategy's name.
"""
from __future__ import annotations

import math

import numpy as np

BARS_PER_YEAR = 6 * 365


def seg_metrics(seg):
    seg = np.asarray(seg, dtype=float)
    seg = seg[~np.isnan(seg)]
    sd = float(seg.std())
    return {
        "pnl_pct": round(float(seg.sum()) * 100, 2),
        "sharpe": round(float(seg.mean() / sd * math.sqrt(BARS_PER_YEAR)), 3) if sd > 0 else 0.0,
        "n_bars": int(len(seg)),
    }


def standard_row(name, train_seg, cv_sharpe, test_seg=None):
    """One candidate's standard report row. overfit_gap = train - CV sharpe
    (both in-design); test columns appear only when a test window was
    legitimately spent."""
    tr = seg_metrics(train_seg)
    row = {"name": name,
           "train_pnl_pct": tr["pnl_pct"], "train_sharpe": tr["sharpe"],
           "cv_sharpe": round(float(cv_sharpe), 3),
           "overfit_gap": round(tr["sharpe"] - float(cv_sharpe), 3)}
    if test_seg is not None:
        te = seg_metrics(test_seg)
        row.update({"test_pnl_pct": te["pnl_pct"], "test_sharpe": te["sharpe"],
                    "train_test_gap": round(tr["sharpe"] - te["sharpe"], 3)})
    return row


def robustness_battery(run_fn, base_params, param_grid, windows, cost_mults=(0.5, 1.0, 2.0)):
    """Run `run_fn(params, window, cost_mult) -> {'pnl_pct','sharpe'}` over
    every perturbation. `param_grid` maps param -> list of alternates (base
    included implicitly). Verdict heuristics, printed with the numbers so
    they are criticizable:
      - ERA-FRAGILE  : sign of pnl flips between any two windows
      - PARAM-FRAGILE: any single-param perturbation flips the pnl sign
      - COST-FRAGILE : doubling costs flips the pnl sign
    """
    out = {"windows": {}, "params": {}, "costs": {}, "flags": []}
    signs = []
    for wname, w in windows.items():
        r = run_fn(base_params, w, 1.0)
        out["windows"][wname] = r
        signs.append(np.sign(r["pnl_pct"]))
    if len({s for s in signs if s != 0}) > 1:
        out["flags"].append("ERA-FRAGILE")

    wname0, w0 = next(iter(windows.items()))
    base_sign = np.sign(out["windows"][wname0]["pnl_pct"])
    for p, alts in param_grid.items():
        for v in alts:
            r = run_fn({**base_params, p: v}, w0, 1.0)
            out["params"][f"{p}={v}"] = r
            if base_sign != 0 and np.sign(r["pnl_pct"]) not in (0, base_sign):
                out["flags"].append(f"PARAM-FRAGILE({p}={v})")
    for cm in cost_mults:
        r = run_fn(base_params, w0, cm)
        out["costs"][f"x{cm}"] = r
        if cm > 1.0 and base_sign > 0 and r["pnl_pct"] < 0:
            out["flags"].append(f"COST-FRAGILE(x{cm})")
    out["flags"] = sorted(set(out["flags"]))
    return out
