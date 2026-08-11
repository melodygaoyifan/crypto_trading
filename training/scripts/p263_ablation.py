"""[P263b] Ablation of the stage-1 death — which ingredient killed the
pooled design?

The P263 config (pooled, vol-scaled, refit 250, funding_z zeroed) died at
stage 1 (sum increment −1.18) while the P259 config (per-asset, raw target,
refit 500, funding_z real) PASSED stage 1 (+1.18) — four simultaneous
changes, so the kill is unattributed. One-factor-at-a-time from the dead
config toward the live one, measured as design-era overlay increment per
asset. DIAGNOSIS ONLY: every row here is another look at the design era,
so any "better" variant found this way has been selected by iteration and
faces the FULL P263 ladder (pre-design, virgin, cross-asset) before any
read — and the family's nearest neighbor already lost the validation era
(P259b), which is the prior any survivor must argue against.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from sklearn.linear_model import Ridge  # noqa: E402

from training.regime_model_lab import _ctx, DESIGN  # noqa: E402
from training.train_supervised_full import COST_BPS  # noqa: E402
from training.mechanism_lab import book_targets  # noqa: E402
from training.banded_forecast_lab import close_features  # noqa: E402
from training.pooled_banded_lab import (  # noqa: E402
    banded_pos, pnl, _vol20, H, GAP, ALPHA,
)

DS, DE = DESIGN
HOME = ("BTC", "ETH", "SOL")
OUT = REPO / "training" / "reports" / "p263_ablation.json"


def forecast(datasets, pooled: bool, vol_scaled: bool, refit: int):
    preds = {a: np.full(len(d["close"]), np.nan) for a, d in datasets.items()}
    max_n = max(len(d["close"]) for d in datasets.values())
    for t0 in range(DS - 500, DE, refit):
        fits = {}
        if pooled:
            Xtr, ytr = [], []
            for a, d in datasets.items():
                x, y = _training_rows(d, t0, vol_scaled)
                if x is not None:
                    Xtr.append(x)
                    ytr.append(y)
            if Xtr and sum(len(x) for x in Xtr) >= 1500:
                Xp, yp = np.vstack(Xtr), np.concatenate(ytr)
                mu, sd = Xp.mean(0), Xp.std(0) + 1e-12
                m = Ridge(alpha=ALPHA).fit((Xp - mu) / sd, yp)
                for a in datasets:
                    fits[a] = (m, mu, sd)
        else:
            for a, d in datasets.items():
                x, y = _training_rows(d, t0, vol_scaled)
                if x is not None and len(x) >= 800:
                    mu, sd = x.mean(0), x.std(0) + 1e-12
                    fits[a] = (Ridge(alpha=ALPHA).fit((x - mu) / sd, y),
                               mu, sd)
        for a, d in datasets.items():
            if a not in fits:
                continue
            m, mu, sd = fits[a]
            n = len(d["close"])
            lo, hi = t0, min(t0 + refit, n, DE)
            if lo >= hi:
                continue
            X = d["X"]
            seg = ~np.isnan(X[lo:hi]).any(1)
            p = np.full(hi - lo, np.nan)
            p[seg] = m.predict((X[lo:hi][seg] - mu) / sd)
            preds[a][lo:hi] = p
    return preds


def _training_rows(d, t0, vol_scaled):
    close, X, vol = d["close"], d["X"], d["vol"]
    n = len(close)
    y = np.full(n, np.nan)
    y[:-H] = close[H:] / close[:-H] - 1.0
    if vol_scaled:
        y = y / (vol * np.sqrt(H) + 1e-9)
    hi = min(t0 - GAP, n)
    ok = ~np.isnan(X[:hi]).any(1) & ~np.isnan(y[:hi]) & ~np.isnan(vol[:hi])
    if ok.sum() < 300:
        return None, None
    return X[:hi][ok], y[:hi][ok]


def run_config(datasets, pooled, vol_scaled, refit, label):
    preds = forecast(datasets, pooled, vol_scaled, refit)
    total = 0.0
    per = {}
    # band selected per config from the SAME small grid (fair comparison —
    # each config gets its best band, as stage 1 did)
    best = None
    for te in (1.0, 1.5, 2.0):
        for tx in (0.25, 0.5):
            tot = 0.0
            p2 = {}
            for a in HOME:
                d = datasets[a]
                posb = banded_pos(preds[a], d["lab"], te, tx)
                book = book_targets(a, d["lab"], d["fz"])
                ov = np.where(book != 0.0, book, posb)
                inc = (pnl(d["close"], ov, COST_BPS[a], DS, DE)
                       - pnl(d["close"], book, COST_BPS[a], DS, DE))
                p2[a] = round(inc, 4)
                tot += inc
            if best is None or tot > best[0]:
                best = (tot, (te, tx), p2)
    total, band, per = best
    print(f"{label:<44} sum={total:+.4f} band={band} per={per}")
    return {"sum_increment": round(total, 4), "band": band, "per": per}


def main() -> int:
    datasets = {}
    for a in HOME:
        c = _ctx(a)
        close = c["close"]
        Xz, _ = close_features(close, np.full(len(close), np.nan))
        Xf, _ = close_features(close, c["fz"])
        datasets[a] = {"close": close, "lab": c["lab"], "fz": c["fz"],
                       "vol": _vol20(close), "X": Xz, "Xf": Xf}

    report = {"generated": datetime.now(timezone.utc).isoformat(),
              "note": "design-era overlay increments; DIAGNOSIS ONLY — any "
                      "winner here is selection-by-iteration and faces the "
                      "full P263 ladder + the P259b validation prior"}
    # A: the dead P263 config
    report["A_dead_pooled_vs250_fz0"] = run_config(
        datasets, True, True, 250, "A dead: pooled+volscaled+250+fz0")
    # B: refit 250 -> 500
    report["B_pooled_vs500_fz0"] = run_config(
        datasets, True, True, 500, "B refit 500")
    # C: vol-scaled -> raw
    report["C_pooled_raw250_fz0"] = run_config(
        datasets, True, False, 250, "C raw target")
    # D: pooled -> per-asset
    report["D_perasset_vs250_fz0"] = run_config(
        datasets, False, True, 250, "D per-asset")
    # E: funding_z restored (per-asset features carry real fz)
    for a in HOME:
        datasets[a]["X"] = datasets[a]["Xf"]
    report["E_perasset_raw500_fzreal_p259ref"] = run_config(
        datasets, False, False, 500, "E p259 reference: perasset+raw+500+fz")
    report["F_pooled_raw500_fzreal"] = run_config(
        datasets, True, False, 500, "F pooled+raw+500+fz")
    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"report: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
