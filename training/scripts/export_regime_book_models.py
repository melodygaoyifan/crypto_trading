"""[P248-GP2] Export the SOL bear-cell ridge for the runtime shadow harness.

╔════════════════════════════════════════════════════════════════════════╗
║ [P253] THIS EXPORT IS RETIRED ON EVIDENCE (P250-F1b). Do not re-run    ║
║ without --force-retired and a NEW P-entry recording why.               ║
╠════════════════════════════════════════════════════════════════════════╣
║ The "p247_leakfix winner" this docstring used to celebrate WAS the     ║
║ leak: the parquet's funding_rate_zscore column sat inside X itself, so ║
║ the SOL bear ridge trained on a 16h look-ahead. On clean X its CV      ║
║ collapsed +5.5% -> +0.3% and the SOL perp assembly's validation went   ║
║ +64.2% -> -22.9% (P250). The deployed artifact                         ║
║ (configs/regimebook/SOL_bear_ridge.json) was DELETED in commit 816ce56 ║
║ and the shadow harness deliberately degrades SOL to                    ║
║ hold-bull/flat (v1_degraded_no_bear_leg). Re-running this script       ║
║ would silently RESURRECT the retired leg on the next deploy.           ║
╚════════════════════════════════════════════════════════════════════════╝

Original mechanics (kept for the day a leg EARNS re-export): fits
ridge_defensive (alpha=30) on ALL SOL bear-regime rows to date — causal
funding, redundancy-pruned features — and writes feature names + scaler +
coefficients + train sigma to configs/regimebook/SOL_bear_ridge.json. The
runtime harness activates the leg only when it can produce 100% of the named
features live — coverage is counted per tick, never assumed.
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
    # [P253] Refusal gate — see the banner above. The cell was retired on
    # P250's clean-X measurement; an accidental re-run must not quietly put
    # the leg back on the deploy path.
    if "--force-retired" not in sys.argv:
        print(
            "REFUSING: the SOL bear ridge export was RETIRED on evidence "
            "(P250-F1b: the +64.2% was the funding leak; clean validation "
            "-22.9%). configs/regimebook/SOL_bear_ridge.json was deleted in "
            "816ce56 and the shadow harness runs SOL as hold-bull/flat by "
            "design. If a retrained candidate has EARNED re-export (P166 "
            "forward evidence + its own P-entry), re-run with "
            "--force-retired.")
        return 3
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
