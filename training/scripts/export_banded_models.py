"""[P259] Export the banded-forecast overlay models for the runtime harness.

Exports BTC and ETH ONLY — the two assets where the overlay beat the raw
book on BOTH the design and pre-design eras (banded_forecast_lab_p259.json).
SOL's book stood; exporting SOL here without new evidence would repeat the
P250 mistake the SOL bear-ridge refusal gate exists for.

The artifact is the P248 pattern: config + coefficients (the refit job IS
the model). Features are the P259 close+funding set, computable live from
the harness's own inputs — live parity by construction. The band runs
through defense/regime_book_shadow.banded_step, the same single source the
lab used, so the mechanism cannot drift.

ONE deliberate lab/live divergence, recorded: the lab normalizes the
forecast by a ROLLING 500-bar sigma; the runtime uses the CONSTANT sigma
exported here, refreshed on each re-export. Re-run weekly (with the parquet
refresh) and commit the diff.
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

from training.regime_model_lab import _ctx  # noqa: E402
from training.banded_forecast_lab import (  # noqa: E402
    close_features, RIDGE_ALPHA, H,
)
from training.provenance import provenance_stamp  # noqa: E402
from training.train_supervised_full import DATA_DIR  # noqa: E402

# The lab winners (banded_forecast_lab_p259.json grid_top3[0] per asset).
BAND_PARAMS = {
    "BTC": {"t_enter": 2.0, "t_exit": 0.5, "regime_gated": False},
    "ETH": {"t_enter": 1.0, "t_exit": 0.25, "regime_gated": False},
}


def main() -> int:
    # [P259b] REFUSAL GATE — the candidate was WITHDRAWN the same day it was
    # built. The operator-authorized single validation-era read (the one
    # window the selection never touched, ~1.8y recent) shows the banded
    # increment NEGATIVE on both assets: BTC -0.212, ETH -0.618 vs the raw
    # book. The overlay won BOTH older eras and survived the full same-day
    # kill battery (9/9 perturbations, cost x2, all sub-periods) — and still
    # failed the unread window. Era-fragility, the P243/P244 class. Re-run
    # only with new evidence + its own P-entry.
    if "--force-withdrawn" not in sys.argv:
        print(
            "REFUSING: the banded overlay was WITHDRAWN on its validation "
            "read (2026-08-10, ledgered: banded_overlay_p259) — increment "
            "-0.212 BTC / -0.618 ETH on the era the selection never "
            "touched. Re-exporting would resurrect a candidate that lost "
            "the only honest exam available same-day. If new lab evidence "
            "exists (with era-stability INCLUDING the recent era), re-run "
            "with --force-withdrawn and record a P-entry.")
        return 3
    for asset, band in BAND_PARAMS.items():
        c = _ctx(asset)
        close, fz = c["close"], c["fz"]
        X, names = close_features(close, fz)
        n = len(close)
        y = np.full(n, np.nan)
        y[:-H] = close[H:] / close[:-H] - 1.0
        ok = ~np.isnan(X).any(1) & ~np.isnan(y)
        mu, sd = X[ok].mean(0), X[ok].std(0) + 1e-12
        m = Ridge(alpha=RIDGE_ALPHA).fit((X[ok] - mu) / sd, y[ok])
        preds = m.predict((X[ok][-500:] - mu) / sd)
        sigma = float(np.std(preds)) or 1e-9
        payload = {
            "asset": asset, "family": "banded_ridge_overlay",
            "alpha": RIDGE_ALPHA, "horizon_bars": H,
            "features": names,
            "mean": [float(v) for v in mu],
            "scale": [float(v) for v in sd],
            "coef": [float(v) for v in m.coef_],
            "intercept": float(m.intercept_),
            "forecast_sigma": sigma,
            "band": band,
            "n_rows": int(ok.sum()),
            "fitted_at": datetime.now(timezone.utc).isoformat(),
            "note": "sigma is CONSTANT until the next export (lab uses "
                    "rolling 500) — the one recorded lab/live divergence",
            "provenance": provenance_stamp(
                data_files=[DATA_DIR / f"{asset}_4H_full.parquet"]),
        }
        out = REPO / "configs" / "regimebook" / f"{asset}_banded.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        print(f"{asset}: {ok.sum()} rows, sigma={sigma:.6f}, "
              f"band={band} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
