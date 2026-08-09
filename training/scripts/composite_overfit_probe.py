"""[P243-probe] Is composite_bull_volscaled overfit? Four falsification tests.

The composite was DESIGNED after observing fold results on bars ~[7203,13095]
(model wins bears, holding wins bulls) — in-sample hypothesis formation is
the main overfitting risk, alongside best-of-8 pooled selection. This probe
answers with evidence:

  1. PRE-DESIGN WINDOW [4000,7203] (~Jun-2021 -> Nov-2023: top, full bear,
     recovery): never part of any window that motivated the design. If the
     composite only works where it was invented, this kills it.
  2. SWITCH ROBUSTNESS: SMA window {100,200,300}. Only-works-at-200 = fitted.
  3. CONTRACT ROBUSTNESS: deadband {0.15,0.25,0.35} at SMA200.
  4. BEAR-LEG ABLATION (paired): composite minus flag-only-long — identical
     bull legs, so the paired difference isolates exactly what the
     directional bear leg adds, with a block-bootstrap CI on the difference
     (far more powerful than absolute CIs).

The vol-scaled ridge z is computed ONCE per (asset, window) — the flag and
deadband only touch the position layer, so every variant shares the same
forecasts and differences are attributable to the switch alone.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from training.train_supervised_full import (   # noqa: E402
    load_asset, select_features, walk_forward_z, evaluate_segment,
    Candidate, COST_BPS, BARS_PER_YEAR, DI, SEED,
)

WINDOWS = {"design[7203:13095]": (7203, 13095),
           "PRE-design[4000:7203]": (4000, 7203)}


def positions(z_seg, flag_seg, deadband):
    pos = np.zeros(len(z_seg))
    last = 0.0
    for i in range(len(z_seg)):
        if i % DI == 0:
            if flag_seg is not None and flag_seg[i]:
                last = 1.0
            elif not np.isnan(z_seg[i]):
                last = 0.0 if abs(z_seg[i]) < deadband else float(np.clip(z_seg[i], -1, 1))
            elif flag_seg is not None:
                last = 0.0
        pos[i] = last
    return pos


def diff_ci(a, b, n_boot=1000, block=30):
    """Block-bootstrap CI on the TOTAL PnL difference (a - b), paired."""
    d = a - b
    d = d[~np.isnan(d)]
    rng = np.random.default_rng(SEED)
    nblocks = int(np.ceil(len(d) / block))
    totals = []
    for _ in range(n_boot):
        starts = rng.integers(0, max(1, len(d) - block), nblocks)
        boot = np.concatenate([d[s:s + block] for s in starts])[:len(d)]
        totals.append(boot.sum() * 100)
    return float(np.percentile(totals, 2.5)), float(np.percentile(totals, 97.5))


def main():
    for asset in ("BTC", "ETH"):
        X, targets, close, regime, feats = load_asset(asset)
        n = len(close)
        cost = COST_BPS[asset]
        print(f"\n########## {asset} ##########", flush=True)
        cand = Candidate("probe_volscaled", "composite", "volscaled", "pruned_all")
        for wname, (s, e) in WINDOWS.items():
            e = min(e, n)
            fsets = select_features(X, targets["ret"], s, feats)
            z = walk_forward_z(cand, X, targets["volscaled"], regime, fsets, s, e)
            zseg = z[s:e]
            closeseg = close  # evaluate_segment slices internally

            def run(flag_full, deadband, label):
                pos = np.zeros(n)
                pos[s:e] = positions(zseg, None if flag_full is None else flag_full[s:e], deadband)
                ev = evaluate_segment(closeseg, pos, cost, s, e)
                series = ev.pop("series")
                print(f"  {wname:<22} {label:<28} pnl={ev['pnl_pct']:+8.1f}% "
                      f"sharpe={ev['sharpe']:+.2f}", flush=True)
                return series, ev

            # buy & hold reference
            posb = np.zeros(n); posb[s:e] = 1.0
            evb = evaluate_segment(closeseg, posb, cost, s, e); evb.pop("series")
            print(f"  {wname:<22} {'buy_and_hold':<28} pnl={evb['pnl_pct']:+8.1f}% "
                  f"sharpe={evb['sharpe']:+.2f}", flush=True)

            # 2. switch robustness
            comp200 = None
            for w in (100, 200, 300):
                flag = close > pd.Series(close).rolling(w).mean().to_numpy()
                series, _ = run(flag, 0.25, f"composite_sma{w}_db0.25")
                if w == 200:
                    comp200 = series
                    flag200 = flag

            # 3. contract robustness at sma200
            for db in (0.15, 0.35):
                run(flag200, db, f"composite_sma200_db{db}")

            # 4. ablation: flag-only long (no bear leg) + pure model
            flag_only_series, _ = run(flag200, math.inf, "flag_only_long (no bear leg)")
            run(None, 0.25, "pure_volscaled (no bull leg)")
            lo, hi = diff_ci(comp200, flag_only_series)
            verdict = "SIGNIFICANT" if lo > 0 else ("negative!" if hi < 0 else "not significant")
            print(f"  {wname:<22} bear-leg increment (paired)  "
                  f"total={float(np.nansum(comp200 - flag_only_series)) * 100:+.1f}% "
                  f"CI[{lo:+.1f},{hi:+.1f}] -> {verdict}", flush=True)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
