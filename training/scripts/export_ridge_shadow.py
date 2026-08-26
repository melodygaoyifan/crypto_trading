#!/usr/bin/env python3
"""[P409/P409b WITHDRAWN 2026-08-26] Export the held BTC ridge shadow.

WITHDRAWN: the +46 walk-forward edge (P409) rested on regime_proba_7, a feature
the training parquet carries but the RUNTIME does NOT emit (BTC's live GMM is
k=6, so only regime_proba_0..5 exist — the P215 train/serve GMM-vocabulary
skew). With SERVE-AVAILABLE features only (regime_proba_6/7 excluded), the same
top-8 recipe collapses to -18% vs buy-hold +9% (1/3 folds). So the held BTC
ridge does NOT beat buy-and-hold with features the live system can provide. The
deployed shadow was correctly recording FLAT(cov-1) every tick (refusing on the
missing feature) — caught by reading the first live artifact (P264/P390b). The
config was deleted; this script REFUSES unless --force so a weekly cron cannot
silently re-create a dead shadow. Re-open only if a serve-valid feature set
clears on its own merits (no feature-count fishing).

WHY THIS EXISTS (operator: "we are building a model that can adapt our venue,
not the other way ... if the profit can't cover the cost, place larger orders
or hold longer"). The flat CDE fee makes a per-bar forecaster untradeable
(P385: IC~0.04, break-even ~8-12bps RT, dead at the flat ~28bps). Applying the
operator's HOLD-LONGER lever (P386) and dropping the uniform-across-assets
assumption ("we don't have to trade what isn't tradeable"):

  * A HELD ridge (deadband on the trailing-z of the prediction, band 1.0,
    ~59 flips/yr) cuts BTC's cost from 287% -> 49% and flips it net-positive.
  * Selected WALK-FORWARD (deadband picked on data before each fold), the BTC
    held ridge CLEARS the flat fee: 4-fold sum +46% vs buy-hold +9%, and a
    SMALL 8-feature set is steadier than all 137 (mid-fold -15 vs -52).
  * ETH/SOL FAIL even held (-9 / -138) -> per the operator's principle we do
    NOT build them. BTC only.

Unlike the withdrawn mlpshadow (P285c, killed by SEED fragility), a ridge is
CLOSED-FORM/deterministic -- no seed, so it cannot die that way. Its remaining
risk is era-fragility (that one negative fold), which the FORWARD ledger is the
only honest exam for (this is a Rung-0.5 backtest on the multiply-read 6y
window, P260 discount; a pass here is a shadow candidate, never a live flip).

WHAT THIS EXPORTS (JSON, never pickle -- the P5 cross-script trap):
  feature_names(8), scaler_mean/scale, ridge coef+intercept, deadband=1.0,
  z_window/z_min (the trailing-z normalizer -- NOT a fixed sig; the recipe
  validated with a causal rolling z, and a fixed sig would be a different,
  unvalidated signal, the P164/P214 train/serve-skew class), target=ret16h,
  provenance. The live harness (defense/ridge_shadow.py) does the forward pass
  in stdlib math and REFUSES on any feature-coverage gap (P248 parity rule).

RECIPE: Ridge(alpha=10) + StandardScaler on the top-8 features by |corr| with
the 16h forward return, fit on ALL rows through the parquet end (the weekly
re-run IS the refit cadence, P248/P284 doctrine; the forward ledger is the OOS
exam). Selection + fit are on the SAME data on purpose -- the honesty comes
from the forward ledger, not from an in-sample holdout we'd only read once.

    python -X utf8 training/scripts/export_ridge_shadow.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "training"))

from splits import assert_clean_gmm  # noqa: E402

DRL_DIR = REPO / "training" / "training_data" / "drl_training"
OUT_DIR = REPO / "configs" / "ridgeshadow"
ASSET = "BTC"   # BTC is the sole asset whose held ridge clears the flat fee
                # walk-forward (P409); ETH/SOL fail and are NOT built.
N_FEATURES = 8
# [P409b] BTC runtime GMM is k=6 (P221/P267): regime_proba_6/7 exist in the
# training parquet but NOT at serve. Excluding them keeps selection serve-valid.
_SERVE_ABSENT = frozenset({"regime_proba_6", "regime_proba_7"})
Z_WINDOW = 500
Z_MIN = 100
DEADBAND = 1.0
ALPHA = 10.0
_NON_FEAT = {"timestamp", "open", "high", "low", "close", "volume", "vwap",
             "date", "asset", "symbol"}


def _load(asset: str):
    d = pd.read_parquet(DRL_DIR / f"{asset}_4H_full.parquet")
    feats = [c for c in d.columns
             if c not in _NON_FEAT and pd.api.types.is_numeric_dtype(d[c])
             and not c.startswith("fwd") and not c.startswith("target")
             and c not in _SERVE_ABSENT]
    close = d["close"].to_numpy(float)
    X = d[feats].to_numpy(float)
    return feats, X, close


def _select_top_features(X, feats, fwd, k):
    """Top-k features by |corr with the forward return|, over all valid rows.
    (Momentum/return/regime dominate on BTC -- P409: top-8 clears steadier
    than all 137, which partly overfit.)"""
    scored = []
    for j, name in enumerate(feats):
        x = X[:, j]
        m = ~np.isnan(x) & ~np.isnan(fwd)
        if m.sum() < 200 or np.std(x[m]) == 0:
            scored.append((0.0, j, name))
            continue
        c = abs(float(np.corrcoef(x[m], fwd[m])[0, 1]))
        scored.append((0.0 if np.isnan(c) else c, j, name))
    scored.sort(reverse=True)
    return [(j, name) for _, j, name in scored[:k]]


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="P409b: the candidate is WITHDRAWN (serve-valid set does not clear); required to re-export")
    if not ap.parse_args().force:
        print("REFUSED: P409 ridge shadow is WITHDRAWN (see the module docstring). "
              "Re-export only with --force after a serve-valid set clears on its merits.",
              file=sys.stderr)
        return 2
    assert_clean_gmm(ASSET)  # regime_proba_* is a GMM feature -- refuse a
                             # leaked (full-sample) GMM fit (P4/P164/P280)
    feats, X, close = _load(ASSET)
    n = len(close)
    fwd = np.full(n, np.nan)
    fwd[:n - 4] = close[4:] / close[:n - 4] - 1.0   # 16h forward return

    sel = _select_top_features(X, feats, fwd, N_FEATURES)
    idx = [j for j, _ in sel]
    names = [nm for _, nm in sel]
    Xf = X[:, idx]
    m = ~(np.isnan(Xf).any(axis=1) | np.isnan(fwd))
    if m.sum() < 500:
        print(f"[EXPORT] {ASSET}: only {int(m.sum())} valid rows -- refusing",
              file=sys.stderr)
        return 2

    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xf[m])
    mdl = Ridge(alpha=ALPHA).fit(sc.transform(Xf[m]), fwd[m])
    train_preds = mdl.predict(sc.transform(Xf[m]))
    pred_std = float(np.std(train_preds)) or 1e-9

    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True,
                             cwd=REPO, encoding="utf-8").stdout.strip()
    except Exception:
        sha = "unknown"

    payload = {
        "asset": ASSET,
        "candidate": "held ridge, top-8, band 1.0 (P409 hold-longer lever)",
        "feature_names": names,
        "scaler_mean": sc.mean_.tolist(),
        "scaler_scale": sc.scale_.tolist(),
        "coef": mdl.coef_.tolist(),
        "intercept": float(mdl.intercept_),
        "deadband": DEADBAND,
        "z_window": Z_WINDOW, "z_min": Z_MIN,
        "target": "ret16h", "alpha": ALPHA,
        "pred_std_train": pred_std,   # diagnostic only; serve uses trailing z
        "provenance": {"git": sha, "rows_fit": int(m.sum()), "n_total": n,
                       "fit_policy": "split_aware_verified",
                       "exported": datetime.now(timezone.utc).isoformat(),
                       "refit_doctrine": "re-run weekly; the refit job IS "
                                         "the model (P248/P284)"},
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{ASSET}.json"
    out.write_text(json.dumps(payload), encoding="utf-8")
    print(f"[EXPORT] {ASSET}: {N_FEATURES} features, {int(m.sum())} rows, "
          f"band={DEADBAND} -> {out}")
    print("  features:", ", ".join(names))
    return 0


if __name__ == "__main__":
    sys.exit(main())
