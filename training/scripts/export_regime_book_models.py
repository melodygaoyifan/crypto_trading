"""[P248-GP2] Export the SOL bear-cell ridge for the runtime shadow harness.

The adaptive model's artifact is its CONFIG + current coefficients (Plan V3:
the refit job IS the model). This script fits ridge_defensive (alpha=30, the
p247_leakfix winner) on ALL SOL bear-regime rows to date — causal funding,
redundancy-pruned features — and writes feature names + scaler + coefficients
+ train sigma to configs/regimebook/SOL_bear_ridge.json (TRACKED, so it
rides git deploys and its history is its provenance).

Re-run weekly (or after any parquet refresh) and commit the diff. The
runtime harness activates the leg only when it can produce 100% of the
named features live — coverage is counted per tick, never assumed.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from training.regime_model_lab import _ctx, REGIME_ID  # noqa: E402
from training.train_supervised_full import DATA_DIR  # noqa: E402
from training.provenance import provenance_stamp  # noqa: E402

OUT = REPO / "configs" / "regimebook" / "SOL_bear_ridge.json"
ALPHA = 30.0


def main():
    ctx = _ctx("SOL"); ctx["asset"] = "SOL"
    X, y, lab, feats = ctx["X"], ctx["y"], ctx["lab"], ctx["feats"]
    rid = REGIME_ID["bear"]
    m = (lab == rid) & ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    # redundancy prune on the bear rows (the lab's rule, applied at n)
    Xb, yb = X[m], y[m]
    sd = Xb.std(axis=0)
    live = [i for i in range(X.shape[1]) if sd[i] > 0]
    corr = np.corrcoef(Xb[:, live].T)
    dropped = set(i for i in range(X.shape[1]) if sd[i] == 0)
    for a in range(len(live)):
        for b in range(a + 1, len(live)):
            if live[b] in dropped or live[a] in dropped:
                continue
            if abs(corr[a, b]) > 0.95:
                dropped.add(live[b])
    keep = [i for i in range(X.shape[1]) if i not in dropped]

    sc = StandardScaler().fit(Xb[:, keep])
    model = Ridge(alpha=ALPHA).fit(sc.transform(Xb[:, keep]), yb)
    train_sigma = float(np.std(model.predict(sc.transform(Xb[:, keep])))) or 1e-9

    payload = {
        "asset": "SOL", "cell": "bear", "family": "ridge_defensive",
        "alpha": ALPHA,
        "features": [feats[i] for i in keep],
        "mean": [float(v) for v in sc.mean_],
        "scale": [float(v) for v in sc.scale_],
        "coef": [float(v) for v in model.coef_],
        "intercept": float(model.intercept_),
        "train_sigma": train_sigma,
        "n_rows": int(m.sum()),
        "fitted_at": datetime.now(timezone.utc).isoformat(),
        "provenance": provenance_stamp(
            data_files=[DATA_DIR / "SOL_4H_full.parquet"]),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"exported {len(keep)} features, {m.sum()} bear rows, "
          f"sigma={train_sigma:.6f} -> {OUT}")


if __name__ == "__main__":
    sys.exit(main() or 0)
