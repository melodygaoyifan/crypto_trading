#!/usr/bin/env python3
"""[P281] The TARGETED ridge_a30 re-certification — the P254 revival read.

ONE pre-registered config (P241's incumbent: adaptive weekly-refit ridge on
the 16h return target — lockbox Sharpe +1.28 BTC / +1.73 ETH under the OLD
regime: leaked-era parquets, dead futures columns, 2x-overcharged cost).
P254 authorized its revival strictly via "re-fit + re-certification on the
clean parquets". This script is that re-certification and nothing else:

- NO search: the config predates today, so this read is not contaminated by
  the p281 searches (whose 196-trial winners just collapsed at the lockbox,
  replicating P241's overfit lesson on fresh data).
- clean-GMM gate (P280), honest ROUND-TRIP cost (P281), the rebuilt
  parquets with real futures features, ledgered spend (P244).
- Judged on the SAME lockbox window as the incumbent numbers ([9100, n)),
  deployment-faithful position mapping (z, deadband 0.25, act every 4 bars)
  via train_supervised_full's own machinery — single-source, no re-
  implementation drift (P172).

PASS = beats B&H on the lockbox with Sharpe CI excluding zero (the P241
bar). A pass sends it to Rung-3 shadow (30d, P166) — nothing deploys from
here.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "training"))

from splits import assert_clean_gmm, record_window_usage  # noqa: E402
import train_supervised_full as T  # noqa: E402

LOCK_START = 9100
OUT = REPO / "training" / "reports" / "ridge_a30_recert_p281.json"


def bootstrap_ci(seg: np.ndarray, n_boot: int = 2000, seed: int = 7):
    rng = np.random.default_rng(seed)
    seg = seg[~np.isnan(seg)]
    sh = []
    for _ in range(n_boot):
        s = rng.choice(seg, size=len(seg), replace=True)
        sd = s.std()
        sh.append(s.mean() / sd * np.sqrt(T.BARS_PER_YEAR) if sd > 0 else 0.0)
    return float(np.percentile(sh, 2.5)), float(np.percentile(sh, 97.5))


def main() -> int:
    out = {"generated": datetime.now(timezone.utc).isoformat(),
           "candidate": "ridge_adaptive (the P241 ridge_a30 incumbent)",
           "conditions": "clean parquets + real futures cols + honest RT "
                         "cost (P281); pre-registered config, no search",
           "assets": {}}
    for asset in ("BTC", "ETH"):
        assert_clean_gmm(asset)
        X, targets, close, gmm, feats = T.load_asset(asset)
        n = len(close)
        record_window_usage("ridge_a30_recert:p281", asset,
                            LOCK_START, n, "validation")
        cand = T.Candidate("ridge_adaptive", "ridge", "ret", "pruned_all")
        # the EXACT machinery run() uses (single-source, P172): feature
        # sets selected on data BEFORE the lockbox, walk-forward z, the
        # deployed position mapping
        fsets = T.select_features(X, targets["ret"], LOCK_START, feats)
        y = targets[cand.target]
        z = T.walk_forward_z(cand, X, y, None, fsets, LOCK_START, n)
        pos = np.zeros(n)
        pos[LOCK_START:] = T.positions_from_z(z[LOCK_START:])
        ev = T.evaluate_segment(close, pos, T.COST_BPS[asset],
                                LOCK_START, n)
        ret = np.zeros(n); ret[1:] = close[1:] / close[:-1] - 1.0
        strat = np.zeros(n); strat[1:] = pos[:-1] * ret[1:]
        cost = np.zeros(n)
        cost[1:] = np.abs(np.diff(pos)) * (T.COST_BPS[asset] / 2.0) / 1e4
        seg = (strat - cost)[LOCK_START:n]
        lo, hi = bootstrap_ci(seg)
        bh = (close[n - 1] / close[LOCK_START] - 1) * 100
        passes = ev["pnl_pct"] > bh and lo > 0
        out["assets"][asset] = {
            "lockbox_pnl_pct": ev["pnl_pct"], "sharpe": ev["sharpe"],
            "sharpe_ci": [round(lo, 3), round(hi, 3)],
            "buy_and_hold_pct": round(bh, 2), "passes": passes,
            "incumbent_old_regime": {"BTC": 1.28, "ETH": 1.73}[asset],
        }
        print(f"[RECERT] {asset}: lockbox {ev['pnl_pct']:+.1f}% "
              f"sh {ev['sharpe']:+.2f} CI[{lo:+.2f},{hi:+.2f}] "
              f"vs B&H {bh:+.1f}% -> {'PASS' if passes else 'FAIL'}")
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"report: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
