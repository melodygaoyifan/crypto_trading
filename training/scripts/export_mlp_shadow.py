#!/usr/bin/env python3
"""[P284] Export the pooled-certified BTC mlp_small for its Rung-3 live
shadow — JSON weights, never pickle (the P5 cross-script pickle trap).

WHAT THIS EXPORTS
-----------------
The EXACT certified configuration (P283b: mlp/ret/top24, StandardScaler +
MLPRegressor(hidden=(32,), alpha=1e-2, max_iter=200, early_stopping,
seed=7), weekly refit): fit on ALL rows through the parquet's end, with
the top24 features selected at export time by the zoo's own
select_features — the same machinery the certification walked forward.
**The weekly re-run of this script IS the candidate's refit=42-bar
cadence** (the P248 SOL-ridge doctrine: the refit job is the model).

Export payload: feature_names (24), scaler mean/scale, layer weights
(relu 24→32→1), sig (std of train predictions — walk_forward_z's
normalizer), DEADBAND=0.25, DI=4, provenance (git sha, rows, fit date,
clean-GMM verified). The live harness (defense/mlp_shadow.py) implements
the forward pass in ~5 lines of stdlib math and REFUSES on any feature
coverage gap (the parity discipline whose absence killed the SOL ridge).

CADENCE MECHANISM ([P287]): `make refresh-data` (training/Makefile) now
invokes this script as its LAST step, fail-soft — so the weekly refit
actually has a mechanism instead of depending on an operator remembering
(the P230 class). An export failure warns loudly but never blocks the
data refresh. Manual run remains:
    python -X utf8 training/scripts/export_mlp_shadow.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "training"))

from splits import assert_clean_gmm  # noqa: E402
import train_supervised_full as T  # noqa: E402

OUT_DIR = REPO / "configs" / "mlpshadow"
ASSET = "BTC"   # the sole pooled-certification pass (P283b); adding an
                # asset requires its own certification first


def main() -> int:
    assert_clean_gmm(ASSET)
    X, targets, close, gmm, feats = T.load_asset(ASSET)
    n = len(close)
    fsets = T.select_features(X, targets["ret"], n, feats)
    idx = fsets["top24"]
    names = [feats[i] for i in idx]
    y = targets["ret"]
    Xf = X[:, idx]
    m = ~(np.isnan(Xf).any(axis=1) | np.isnan(y))
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPRegressor
    sc = StandardScaler().fit(Xf[m])
    mdl = MLPRegressor(hidden_layer_sizes=(32,), alpha=1e-2, max_iter=200,
                       early_stopping=True, random_state=T.SEED).fit(
        sc.transform(Xf[m]), y[m])
    train_preds = mdl.predict(sc.transform(Xf[m]))
    sig = float(np.std(train_preds)) or 1e-9
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True,
                             cwd=REPO).stdout.strip()
    except Exception:
        sha = "unknown"
    payload = {
        "asset": ASSET, "candidate": "mlp_small (P283b pooled-certified)",
        "feature_names": names,
        "scaler_mean": sc.mean_.tolist(),
        "scaler_scale": sc.scale_.tolist(),
        "w1": mdl.coefs_[0].tolist(), "b1": mdl.intercepts_[0].tolist(),
        "w2": np.asarray(mdl.coefs_[1]).reshape(-1).tolist(),
        "b2": float(np.asarray(mdl.intercepts_[1]).reshape(-1)[0]),
        "hidden_activation": "relu", "sig": sig,
        "deadband": float(T.DEADBAND), "decision_interval": int(T.DI),
        "provenance": {"git": sha, "rows_fit": int(m.sum()),
                       "n_total": n, "fit_policy": "split_aware_verified",
                       "exported": datetime.now(timezone.utc).isoformat(),
                       "refit_doctrine": "re-run weekly; the refit job IS "
                                         "the model (P248/P284)"},
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{ASSET}.json"
    out.write_text(json.dumps(payload), encoding="utf-8")
    print(f"[EXPORT] {ASSET}: 24 features, {int(m.sum())} rows, sig={sig:.5f}"
          f" -> {out}")
    print("  features:", ", ".join(names))
    return 0


if __name__ == "__main__":
    sys.exit(main())
