#!/usr/bin/env python3
"""[P283-R1] Pooled/lockbox-length certification of the three p281
fold-winners — the operator-adopted criterion (2026-08-16), plus the
DERIVATIVES-FITNESS validation the fold numbers never had.

WHAT THIS IS
------------
The p281 honest-cost zoo produced three candidates that beat BOTH
baselines in their fold (BTC mlp_small, ETH composite_bull_ridge, SOL
hgb_small) but failed the per-fold CI criterion — which P242 proved
nothing (baselines included) can pass at ~1,900 bars. The operator
adopted pooled/lockbox-length certification; this script is that exam:
the SAME fixed candidates (no new selection — the p281 run chose them),
walk-forwarded over ALL THREE fold-val windows via train_supervised_full's
own machinery, pooled (~5,900 bars), block-bootstrap CI, deflated-Sharpe
reported at the zoo's trial count (N=8 per asset).

THE DERIVATIVES-FITNESS HALF (the operator's second question)
-------------------------------------------------------------
The zoo's numbers assume things the LIVE SLEEVE does not do:
  * continuous positions in [-1,1] — the sleeve quantizes at |dir|>=0.15
    to sized contracts (magnitude discarded). Each candidate is therefore
    ALSO evaluated under the LIVE expression (sign-quantized bang-bang) —
    the number that would actually be monetized.
  * price-only PnL — a perp position pays/receives funding. Each variant
    is evaluated +/- the carry term (P245 convention, Binance-proxy rate
    with the P218 wrong-venue caveat recorded).
PASS bar (recorded before running): pooled after-cost CI excludes zero
UNDER THE LIVE EXPRESSION, carry included. The trained-expression numbers
are reported as diagnostics. Reads ledgered (re-aggregation of
already-spent fold-val windows; no new window touched).
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

CANDIDATES = {
    "BTC": ("mlp_small", "mlp", "ret", "top24", 42),
    "ETH": ("composite_bull_ridge", "composite", "ret", "pruned_all", 42),
    "SOL": ("hgb_small", "hgb", "ret", "pruned_all", 180),
}
N_TRIALS = 8            # per-asset zoo size — the DSR denominator
LIVE_THRESHOLD = 0.15   # the sleeve's target_for_signal deadband
OUT = REPO / "training" / "reports" / "pooled_certification_p283.json"


def live_expression(pos: np.ndarray) -> np.ndarray:
    """The sleeve's actual mapping: sign at |dir|>=0.15, magnitude
    discarded (sized contracts are an equity-fraction constant)."""
    out = np.zeros_like(pos)
    out[pos >= LIVE_THRESHOLD] = 1.0
    out[pos <= -LIVE_THRESHOLD] = -1.0
    return out


def dsr(pooled_sharpe: float, n_obs: int, n_trials: int,
        seg: "np.ndarray | None" = None) -> float:
    """Deflated Sharpe (Bailey-LdP): probability the observed pooled Sharpe
    exceeds the expected max of n_trials zero-skill trials.

    [P287] The original implementation carried a dead ``sr0`` variable and a
    denominator ``sqrt(max(1e-12, 1 - 0*sr_bar))`` — always 1 — so the
    skew/kurtosis variance-inflation term of the Bailey-LdP statistic was
    dropped entirely. The P283b/P286 RECORDED DSRs (mlp_small 0.90, BTC
    tcn32 0.66) were computed under that pre-fix normal-form denominator and
    are therefore MILDLY OPTIMISTIC for fat-tailed/skewed return series; the
    pass/fail verdicts did not key on DSR (CI-excludes-zero decides), so no
    verdict changes. Pass ``seg`` (the per-bar net return series the Sharpe
    was computed from) to charge the real moments; omitted, the normal
    assumption (skew 0, kurtosis 3) reproduces the old arithmetic exactly.
    """
    if n_obs < 10:
        return 0.0
    z = math.sqrt(2 * math.log(n_trials)) if n_trials > 1 else 0.0
    exp_max = z - (math.log(math.log(n_trials)) + math.log(4 * math.pi)) \
        / (2 * z) if n_trials > 2 else 0.0
    # per-observation form: SR_ann = sr_bar * sqrt(BARS_PER_YEAR)
    sr_bar = pooled_sharpe / math.sqrt(T.BARS_PER_YEAR)
    sr0_bar = exp_max / math.sqrt(n_obs - 1)
    g3, g4 = 0.0, 3.0  # normal-form fallback
    if seg is not None:
        x = np.asarray(seg, dtype=float)
        x = x[np.isfinite(x)]
        if len(x) > 3:
            m, s = float(np.mean(x)), float(np.std(x))
            if s > 0:
                g3 = float(np.mean((x - m) ** 3) / s ** 3)
                g4 = float(np.mean((x - m) ** 4) / s ** 4)
    from math import erf
    denom = math.sqrt(max(1e-12,
                          1.0 - g3 * sr_bar + (g4 - 1.0) / 4.0 * sr_bar ** 2))
    stat = (sr_bar - sr0_bar) * math.sqrt(n_obs - 1) / denom
    return 0.5 * (1 + erf(stat / math.sqrt(2)))


def main() -> int:
    out = {"generated": datetime.now(timezone.utc).isoformat(),
           "criterion": "pooled/lockbox-length (operator-adopted "
                        "2026-08-16, the P242 disposition)",
           "pass_bar": "pooled after-cost CI excludes zero under the LIVE "
                       "expression, carry included",
           "assets": {}}
    for asset, (name, family, target, fset, refit) in CANDIDATES.items():
        assert_clean_gmm(asset)
        X, targets, close, gmm, feats = T.load_asset(asset)
        n = len(close)
        y = targets[target]
        bull = T.bull_flag(close)
        cand = T.Candidate(name, family, target, fset, refit=refit)
        pos_full = np.zeros(n)
        spans = []
        for fold, (train_end, val_start, val_end) in \
                T.fold_splits(n).items():
            fsets = T.select_features(X, targets["ret"], train_end, feats)
            z = T.walk_forward_z(cand, X, y, None, fsets,
                                 val_start, val_end)
            seg = T.positions_from_z(
                z[val_start:val_end],
                bull[val_start:val_end] if cand.is_composite else None)
            pos_full[val_start:val_end] = seg
            spans.append((val_start, val_end))
        lo_all = min(s for s, _ in spans)
        hi_all = max(e for _, e in spans)
        record_window_usage("pooled_cert:p283", asset, lo_all, hi_all,
                            "validation")
        mask = np.zeros(n, dtype=bool)
        for s, e in spans:
            mask[s:e] = True
        carry_rate = np.nan_to_num(
            np.asarray(_bar_carry_rate(asset, n), dtype=float))
        ret = np.zeros(n); ret[1:] = close[1:] / close[:-1] - 1.0
        res = {}
        for label, pos in (("trained_continuous", pos_full),
                           ("LIVE_sign_expression",
                            live_expression(pos_full))):
            for carry_label, use_carry in (("no_carry", False),
                                           ("with_carry", True)):
                strat = np.zeros(n); strat[1:] = pos[:-1] * ret[1:]
                if use_carry:
                    strat[1:] -= pos[:-1] * carry_rate[1:]
                cost = np.zeros(n)
                cost[1:] = np.abs(np.diff(pos)) * \
                    (T.COST_BPS[asset] / 2.0) / 1e4
                seg = (strat - cost)[mask]
                sd = float(np.nanstd(seg))
                sh = float(np.nanmean(seg) / sd *
                           math.sqrt(T.BARS_PER_YEAR)) if sd > 0 else 0.0
                lo, hi = T.block_bootstrap_ci(seg)
                res[f"{label}.{carry_label}"] = {
                    "pooled_bars": int(mask.sum()),
                    "net_pct": round(float(np.nansum(seg)) * 100, 2),
                    "sharpe": round(sh, 3),
                    "ci": [None if lo is None else round(lo, 3),
                           None if hi is None else round(hi, 3)],
                    "dsr": round(dsr(sh, int(mask.sum()), N_TRIALS,
                                     seg=seg), 3),
                }
        decisive = res["LIVE_sign_expression.with_carry"]
        passes = (decisive["ci"][0] is not None
                  and decisive["ci"][0] > 0)
        out["assets"][asset] = {"candidate": name, "windows": spans,
                                "results": res, "passes": passes}
        print(f"\n[{asset}] {name} — pooled {decisive['pooled_bars']} bars")
        for k, v in res.items():
            mark = "  <== DECISIVE" if k == \
                "LIVE_sign_expression.with_carry" else ""
            print(f"  {k:<34} net {v['net_pct']:+7.2f}%  "
                  f"sh {v['sharpe']:+.2f}  CI[{v['ci'][0]},{v['ci'][1]}]  "
                  f"DSR {v['dsr']:.2f}{mark}")
        print(f"  VERDICT: {'PASS' if passes else 'FAIL'}")
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nreport: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
