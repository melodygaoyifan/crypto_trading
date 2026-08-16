#!/usr/bin/env python3
"""[P285b] Seed-robustness probe of the certified BTC mlp_small — the ONE
training spend justified now, answering the P283b caveat "single seed (7),
MLP seed-sensitivity unprobed".

WHAT THIS IS (and is not): the SAME fixed config (mlp/ret/top24/refit=42),
the SAME already-read pooled windows, the SAME decisive cell (LIVE sign
expression, carry included) — re-fit under 10 seeds to measure DISPERSION.
It is a FALSIFICATION probe, not selection: no seed is ever picked, and the
deployed export stays seed-7 whatever this reports. If the passing cell
collapses across seeds, the shadow candidate is fragile and we learn that
NOW instead of at day 30.

PRE-COMMITTED VERDICT (written before running):
  * ROBUST : every seed's decisive Sharpe > 0 AND the median >= 0.75
             (half the seed-7 certification value of +1.52);
  * FRAGILE: the median <= 0, OR >= 3/10 seeds flip the decisive cell's
             sign negative;
  * MIXED  : anything else — reported honestly, discounts the candidate
             without killing it (the forward ledger still decides).
Re-reads of the fold-val windows are ledgered under seed_probe:p285b
(re-aggregation of already-spent windows; no new window touched).
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "training"))

from splits import assert_clean_gmm, record_window_usage  # noqa: E402
import train_supervised_full as T  # noqa: E402
from regime_model_lab import _bar_carry_rate  # noqa: E402

# import the exam's own expression + config so the probe cannot drift from
# the certification it audits (P172 single-source)
sys.path.insert(0, str(REPO / "training" / "scripts"))
from pooled_certification import live_expression, CANDIDATES  # noqa: E402

ASSET = "BTC"
SEEDS = list(range(1, 11))          # includes 7 = reproduction check
CERT_SHARPE_SEED7 = 1.52            # the P283b decisive value
OUT = REPO / "training" / "reports" / "mlp_seed_probe_p285b.json"


def decisive_sharpe_for_seed(seed: int) -> dict:
    T.SEED = seed
    name, family, target, fset, refit = CANDIDATES[ASSET]
    X, targets, close, gmm, feats = T.load_asset(ASSET)
    n = len(close)
    y = targets[target]
    cand = T.Candidate(name, family, target, fset, refit=refit)
    pos_full = np.zeros(n)
    spans = []
    for fold, (train_end, val_start, val_end) in T.fold_splits(n).items():
        fsets = T.select_features(X, targets["ret"], train_end, feats)
        z = T.walk_forward_z(cand, X, y, None, fsets, val_start, val_end)
        seg = T.positions_from_z(z[val_start:val_end], None)
        pos_full[val_start:val_end] = seg
        spans.append((val_start, val_end))
    mask = np.zeros(n, dtype=bool)
    for s, e in spans:
        mask[s:e] = True
    carry_rate = np.nan_to_num(
        np.asarray(_bar_carry_rate(ASSET, n), dtype=float))
    ret = np.zeros(n); ret[1:] = close[1:] / close[:-1] - 1.0
    pos = live_expression(pos_full)
    strat = np.zeros(n); strat[1:] = pos[:-1] * ret[1:] \
        - pos[:-1] * carry_rate[1:]
    cost = np.zeros(n)
    cost[1:] = np.abs(np.diff(pos)) * (T.COST_BPS[ASSET] / 2.0) / 1e4
    seg = (strat - cost)[mask]
    sd = float(np.nanstd(seg))
    sh = float(np.nanmean(seg) / sd * math.sqrt(T.BARS_PER_YEAR)) \
        if sd > 0 else 0.0
    return {"seed": seed, "sharpe": round(sh, 3),
            "net_pct": round(float(np.nansum(seg)) * 100, 2),
            "spans": spans}


def main() -> int:
    assert_clean_gmm(ASSET)
    results = []
    for s in SEEDS:
        r = decisive_sharpe_for_seed(s)
        results.append(r)
        print(f"  seed {s:>2}: decisive sharpe {r['sharpe']:+.3f}  "
              f"net {r['net_pct']:+7.2f}%", flush=True)
    lo = min(s for sp in results[0]["spans"] for s in (sp[0],))
    hi = max(e for sp in results[0]["spans"] for e in (sp[1],))
    record_window_usage("seed_probe:p285b", ASSET, lo, hi, "validation")
    sharpes = [r["sharpe"] for r in results]
    med = float(np.median(sharpes))
    n_neg = sum(1 for s in sharpes if s <= 0)
    if med <= 0 or n_neg >= 3:
        verdict = "FRAGILE"
    elif all(s > 0 for s in sharpes) and med >= 0.75:
        verdict = "ROBUST"
    else:
        verdict = "MIXED"
    seed7 = next((r["sharpe"] for r in results if r["seed"] == 7), None)
    out = {"generated": datetime.now(timezone.utc).isoformat(),
           "asset": ASSET, "cell": "LIVE_sign_expression.with_carry",
           "cert_sharpe_seed7": CERT_SHARPE_SEED7,
           "reproduction_seed7": seed7,
           "results": [{k: v for k, v in r.items() if k != "spans"}
                       for r in results],
           "median": round(med, 3), "n_nonpositive": n_neg,
           "verdict": verdict,
           "note": "falsification probe, not selection — the deployed "
                   "export stays seed-7 regardless; the forward ledger "
                   "remains the exam"}
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\n  median {med:+.3f} | nonpositive {n_neg}/10 | "
          f"seed-7 reproduction {seed7} (cert {CERT_SHARPE_SEED7})")
    print(f"  VERDICT: {verdict}")
    print(f"  report: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
