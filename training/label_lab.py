"""[P249] Label lab — triple-barrier labels + meta-labeling (Lopez de Prado).

The one standard technique family this project never tried. Two stages:

  1. TRIPLE-BARRIER LABELS: for each bar, a long's outcome is decided by
     whichever comes first — profit barrier (+k*sigma), stop barrier
     (-k*sigma), or the vertical barrier (T bars). Vol-scaled, so a label
     means the same thing in calm and violent regimes.
  2. META-LABELING: the primary signal is the ERA-STABLE trend rule (hold
     in the bull regime — the one leg that survived every falsification).
     The meta-classifier does NOT predict direction; it predicts whether
     the primary's trade WORKS (barrier outcome > 0), and gates position
     to primary x P(success)>thr. Sizing an era-stable rule is exactly
     the shape of edge this data supports (methodology brief §1).

Protocol (P247 discipline): everything here runs on the DESIGN ERA with
purged CV. NO validation-era read is taken in this script — the validation
window is at 4 ledgered reads and a 5th is spent only if the design-era CV
margin justifies it (operator-visible decision, not a script side effect).
Trial count is printed for the DSR line.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

from training.regime_model_lab import _ctx, REGIME_ID  # noqa: E402
from training.splits import DESIGN_ERA, purged_folds, record_window_usage  # noqa: E402
from training.eval_report import seg_metrics  # noqa: E402
from training.provenance import provenance_stamp  # noqa: E402

# [P287] single-sourced from the supervised zoo (P172): these are
# ROUND-TRIP bps (P281 convention) and are halved per leg at the charge
# site — the restated local copy used to charge full RT per leg (2x).
from training.train_supervised_full import COST_BPS  # noqa: E402

VOL_WIN, K_BARRIER, T_VERTICAL = 42, 1.5, 24   # fixed a priori, not tuned
THRESHOLDS = (0.45, 0.55)                      # 2 gate thresholds (counted)
SEED = 7


def triple_barrier_long(close, k=K_BARRIER, T=T_VERTICAL, vol_win=VOL_WIN):
    """Outcome of a LONG opened at each bar: +1 profit barrier first,
    -1 stop first, sign of terminal return at the vertical barrier.
    Returns (labels, barrier_ret) — barrier_ret is the realized return of
    the episode (for PnL-weighted diagnostics)."""
    n = len(close)
    r1 = np.full(n, np.nan); r1[1:] = close[1:] / close[:-1] - 1.0
    import pandas as pd
    sigma = pd.Series(r1).rolling(vol_win).std().shift(1).to_numpy()
    labels = np.full(n, np.nan)
    ep_ret = np.full(n, np.nan)
    for t in range(vol_win + 1, n - 1):
        s = sigma[t]
        if not np.isfinite(s) or s <= 0:
            continue
        up, dn = close[t] * (1 + k * s), close[t] * (1 - k * s)
        end = min(t + T, n - 1)
        out, rr = None, None
        for j in range(t + 1, end + 1):
            if close[j] >= up:
                out, rr = 1.0, close[j] / close[t] - 1.0
                break
            if close[j] <= dn:
                out, rr = -1.0, close[j] / close[t] - 1.0
                break
        if out is None:
            rr = close[end] / close[t] - 1.0
            out = float(np.sign(rr))
        labels[t], ep_ret[t] = out, rr
    return labels, ep_ret


def run_asset(asset):
    ctx = _ctx(asset); ctx["asset"] = asset
    X, close, lab, n = ctx["X"], ctx["close"], ctx["lab"], ctx["n"]
    carry = ctx["carry_rate"]
    s, e = DESIGN_ERA
    record_window_usage(f"label_lab:p249", asset, s, e, "design")

    tb, _ = triple_barrier_long(close)
    bull = lab == REGIME_ID["bull"]
    meta_y = (tb > 0).astype(float)            # did the long WORK?

    print(f"\n########## {asset} label lab (design era) ##########", flush=True)
    m_all = bull & np.isfinite(tb) & (np.arange(n) >= s) & (np.arange(n) < e)
    base_rate = float(meta_y[m_all].mean()) if m_all.sum() else float("nan")
    print(f"  bull bars with labels: {int(m_all.sum())}, "
          f"P(long works) base rate: {base_rate:.3f}", flush=True)

    def bull_leg_pnl(idx_lo, idx_hi, gate=None):
        """After-cost+carry PnL of the bull-hold leg over [idx_lo, idx_hi),
        optionally gated bar-by-bar (gate: bool array over full n)."""
        pos = np.zeros(n)
        for i in range(idx_lo, idx_hi):
            on = bull[i] and (gate is None or gate[i])
            pos[i] = 1.0 if on else 0.0
        ret = np.zeros(n); ret[1:] = close[1:] / close[:-1] - 1.0
        strat = np.zeros(n); strat[1:] = pos[:-1] * ret[1:]
        cost = np.zeros(n)
        # [P287] COST_BPS is ROUND-TRIP; each |dpos| unit is one LEG.
        cost[1:] = np.abs(np.diff(pos)) * (COST_BPS[asset] / 2.0) / 1e4
        cy = np.zeros(n); cy[1:] = -pos[:-1] * carry[1:]
        return (strat - cost + cy)[idx_lo:idx_hi]

    trials = 0
    results = {"asset": asset, "base_rate": round(base_rate, 4), "cv": {}}
    for fam_name, make in (
        ("logistic", lambda: ("scale", LogisticRegression(max_iter=500, C=0.1))),
        ("lgbm", lambda: ("raw", lgb.LGBMClassifier(
            n_estimators=120, num_leaves=15, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=SEED,
            verbosity=-1))),
    ):
        for thr in THRESHOLDS:
            trials += 1
            cv_gated, cv_raw = [], []
            for tr, va in purged_folds(s, e):
                trm = tr[bull[tr] & np.isfinite(tb[tr])
                         & ~np.isnan(X[tr]).any(axis=1)]
                if len(trm) < 300:
                    continue
                mode, model = make()
                Xtr = X[trm]
                if mode == "scale":
                    sc = StandardScaler().fit(Xtr)
                    model.fit(sc.transform(Xtr), meta_y[trm])
                    p = model.predict_proba(
                        sc.transform(np.nan_to_num(X[va])))[:, 1]
                else:
                    model.fit(Xtr, meta_y[trm])
                    p = model.predict_proba(np.nan_to_num(X[va]))[:, 1]
                gate = np.zeros(n, dtype=bool)
                gate[va] = p >= thr
                g = seg_metrics(bull_leg_pnl(int(va[0]), int(va[-1] + 1), gate))
                r = seg_metrics(bull_leg_pnl(int(va[0]), int(va[-1] + 1), None))
                cv_gated.append(g["pnl_pct"]); cv_raw.append(r["pnl_pct"])
            if cv_gated:
                gm, rm = float(np.mean(cv_gated)), float(np.mean(cv_raw))
                results["cv"][f"{fam_name}_thr{thr}"] = {
                    "gated_pnl": round(gm, 2), "raw_pnl": round(rm, 2),
                    "delta": round(gm - rm, 2)}
                print(f"  {fam_name:<9} thr={thr}: gated={gm:+.2f}% "
                      f"vs raw bull-hold={rm:+.2f}%  delta={gm - rm:+.2f}%",
                      flush=True)
    results["trials"] = trials
    print(f"  trials evaluated: {trials} (DSR line — count them, P247)", flush=True)
    return results


def main():
    out = {"results": [run_asset(a) for a in ("BTC", "ETH", "SOL")],
           "config": {"k": K_BARRIER, "T": T_VERTICAL, "vol_win": VOL_WIN,
                      "thresholds": list(THRESHOLDS)},
           "provenance": provenance_stamp()}
    p = REPO / "training" / "reports" / "label_lab_p249.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nDONE -> {p}", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
