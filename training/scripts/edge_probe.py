"""[P200-followup] Supervised edge probe — the go/no-go gate BEFORE any DRL run.

The P200 measurement settled that TQC-on-these-features has no after-cost
edge (all three folds NOT PROMOTABLE, friction >= the loss). RL cannot
conjure edge from features that carry no predictive information — so before
any further training spend, this probe asks the cheaper question directly:

    Can ANY feature group predict forward returns, walk-forward, above the
    round-trip cost bar?

Method: expanding-window walk-forward (refit every REFIT bars, GAP-bar purge
between train end and prediction start), Ridge (linear) and
HistGradientBoosting (nonlinear) per feature group per horizon. Metrics:
OOS Spearman IC + t-stat, hit rate, and after-cost bps/trade under a simple
|prediction| >= quantile-threshold rule at COST_RT_BPS round trip.

The bar (from P166's arithmetic): required IC ~= cost_bps / (0.7979 *
1.253 * sigma_fwd_bps). Anything statistically indistinguishable from zero,
or below the bar, is a NO.

[P386 CAVEAT — READ THIS BEFORE TREATING A "NO_EDGE" AS FINAL] The bps/trade
and required-IC here charge cost PER BAR, as if every prediction is a round
trip. That is the WORST-CASE execution and overstates the fee barrier by
~20x (P386): a persistent signal should be HELD, paying the fee only on
direction flips, which is how the live SMA200 rule is net-positive. So a
"NO_EDGE" here is the worst-case bound, NOT the verdict. The verdict comes
from the HOLD-AWARE evaluator `training/scripts/signal_hold_backtest.py`
(position held across bars, fee on flips only, long/short + funding, deadband
selected walk-forward not by lookahead sweep). Use THIS probe as the cheap
first filter; use the hold-aware backtest as the actual Rung-0 gate before any
DRL retrain (see docs/research/PROFITABILITY_ROOT_CAUSE_PLAN_2026-08.md).

Verdict semantics:
  EDGE_CANDIDATE  — some group clears the cost bar with |t| >= 2 (justifies a
                    reformulated DRL/strategy build on that basis, SHADOW-first)
  NO_EDGE         — nothing clears; further training spend is unjustified.

Usage:  python -X utf8 training/scripts/edge_probe.py [--asset BTC]
Stdlib + numpy/pandas/sklearn only. ~1-2 min per asset on CPU.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_TRAINING_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = _TRAINING_DIR / "training_data" / "drl_training"
MANIFEST = _TRAINING_DIR.parent / "configs" / "feature_manifest.json"

GAP = 12            # purge bars between train end and prediction start
REFIT = 500         # walk-forward refit cadence (bars)
MIN_TRAIN = 3000    # first fit needs this much history
COST_RT_BPS = 6.0   # [P382] FLOOR only — the refuted 3bps/side percentage
                    # model (P315/P334). The bar actually used per asset is
                    # cost_rt_bps_for(asset): 2 x the measured CDE taker leg
                    # (BTC ~19.7 / ETH ~29.0 / SOL ~29.0 bps RT), so this probe
                    # prices the same venue the other two P166 gates do
                    # (compute_shadow_ic / agent_ic_review, P382). `--cost-rt-bps`
                    # overrides for a what-if (e.g. a percentage venue).
HORIZONS = (1, 4)   # 4h, 16h forward
COST_RT_OVERRIDE = None  # set by --cost-rt-bps


def cost_rt_bps_for(asset: str):
    """[P382] Round-trip cost bar for `asset`: max(floor, 2 x CDE taker leg).
    Returns (bps, source). Falls back to the floor — LOUDLY — if the
    calibration cannot be read, so a probe run on the refuted model says so."""
    if COST_RT_OVERRIDE is not None:
        return float(COST_RT_OVERRIDE), "override"
    try:
        import sys as _sys
        _root = str(_TRAINING_DIR.parent)
        if _root not in _sys.path:
            _sys.path.insert(0, _root)
        from core.cde_fees import CDE_FEE_BPS
        row = CDE_FEE_BPS.get(asset) or {}
        taker = float(row.get("taker"))
        return max(COST_RT_BPS, 2.0 * taker), "cde_fees"
    except Exception as e:
        print(f"WARNING: core.cde_fees unreadable ({type(e).__name__}: {e}) — "
              f"pricing {asset} at the REFUTED {COST_RT_BPS}bps RT floor",
              file=__import__("sys").stderr)
        return COST_RT_BPS, "fallback_refuted_model"

GROUPS = {
    "ALL": None,  # resolved from manifest (+ any fv2_* extras present)
    "ta_base": lambda c: not c.endswith("_denoised") and not c.startswith("regime_proba")
                          and c not in _EXTERNAL and c != "has_external_data"
                          and not c.startswith("fv2_"),
    "denoised": lambda c: c.endswith("_denoised"),
    "external": lambda c: c in _EXTERNAL,
    "regime": lambda c: c.startswith("regime_proba"),
    "momentum": lambda c: any(k in c for k in ("momentum", "macd", "ema", "sma", "adx", "ret_"))
                          and not c.startswith("fv2_"),
    # [P200-FEATURES] flow-v2: taker flow / activity / illiquidity from the
    # Binance kline flow columns + seasonality + cross-asset. Full-history,
    # free — the candidate replacement for the 180d-limited Coinglass group.
    "flow_v2": lambda c: c.startswith("fv2_"),
}
_EXTERNAL = {"funding_rate_zscore", "oi_change_5d", "liq_imbalance",
             "taker_ratio_zscore", "tradecount_zscore", "taker_vol_momentum"}


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 30:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def required_ic(cost_bps: float, sigma_fwd_bps: float) -> float:
    """P166 arithmetic: expected edge = E|z| * r_pearson * sigma,
    E|z|=0.7979, pearson ~= 2*sin(pi*rho/6) ~= 1.047*rho for small rho ->
    use the same 1.253 factor chain as compute_shadow_ic."""
    if sigma_fwd_bps <= 0:
        return float("inf")
    return cost_bps / (0.7979 * 1.047 * sigma_fwd_bps)


def walk_forward(X: np.ndarray, y: np.ndarray, model_name: str):
    """Yield (test_idx, predictions) per refit chunk. Purged by GAP bars."""
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler

    n = len(X)
    start = MIN_TRAIN
    while start + GAP < n:
        tr_end = start
        te_start = tr_end + GAP
        te_end = min(te_start + REFIT, n)
        if te_start >= n:
            break
        Xtr, ytr = X[:tr_end], y[:tr_end]
        mask = ~(np.isnan(Xtr).any(axis=1) | np.isnan(ytr))
        if mask.sum() < 500:
            start = te_end
            continue
        if model_name == "ridge":
            sc = StandardScaler().fit(Xtr[mask])
            m = Ridge(alpha=10.0).fit(sc.transform(Xtr[mask]), ytr[mask])
            Xte = sc.transform(np.nan_to_num(X[te_start:te_end]))
            pred = m.predict(Xte)
        else:
            m = HistGradientBoostingRegressor(
                max_iter=120, max_depth=4, learning_rate=0.06,
                l2_regularization=1.0, random_state=42)
            m.fit(Xtr[mask], ytr[mask])
            pred = m.predict(np.nan_to_num(X[te_start:te_end]))
        yield np.arange(te_start, te_end), pred
        start = te_end


def probe_asset(asset: str) -> dict:
    df = pd.read_parquet(DATA_DIR / f"{asset}_4H_full.parquet")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    feats = [c for c in manifest["all_features"] if c in df.columns]
    # fv2_* extras ride along when present (built by build_flow_features.py);
    # they are NOT manifest members, so obs_dim is unaffected.
    feats += [c for c in df.columns if c.startswith("fv2_")]
    close = df["close"].to_numpy(dtype=float)

    cost_bps, cost_src = cost_rt_bps_for(asset)   # [P382] measured CDE bar
    out = {"asset": asset, "n_bars": len(df), "results": [],
           "cost_rt_bps": cost_bps, "cost_source": cost_src}
    print(f"\n===== {asset}: {len(df)} bars, {len(feats)} features "
          f"(cost bar {cost_bps:.1f}bps RT, {cost_src}) =====")
    for h in HORIZONS:
        fwd = np.full(len(close), np.nan)
        fwd[:-h] = close[h:] / close[:-h] - 1.0
        sigma_bps = float(np.nanstd(fwd)) * 1e4
        req = required_ic(cost_bps, sigma_bps)
        print(f"\n-- horizon {h * 4}h: sigma_fwd={sigma_bps:.0f}bps, "
              f"required IC >= {req:.3f} (for {cost_bps:.1f}bps RT) --")
        print(f"{'group':<10}{'model':<7}{'n_oos':>7}{'IC':>8}{'t':>7}"
              f"{'hit':>7}{'bps/trade@q75':>15}{'verdict':>10}")
        for gname, sel in GROUPS.items():
            cols = feats if sel is None else [c for c in feats if sel(c)]
            if not cols:
                continue
            X = df[cols].to_numpy(dtype=float)
            y = fwd
            for model_name in ("ridge", "hgb"):
                preds = np.full(len(y), np.nan)
                for idx, p in walk_forward(X, y, model_name):
                    preds[idx] = p
                m = ~(np.isnan(preds) | np.isnan(y))
                n_oos = int(m.sum())
                ic = spearman(preds[m], y[m])
                t = ic * math.sqrt(max(n_oos - 1, 1)) if not math.isnan(ic) else float("nan")
                hit = float((np.sign(preds[m]) == np.sign(y[m])).mean()) if n_oos else float("nan")
                # threshold rule: trade only when |pred| >= its own 75th pct
                thr = np.nanquantile(np.abs(preds[m]), 0.75) if n_oos else np.nan
                tm = m & (np.abs(preds) >= thr)
                bps = (float(np.nanmean(np.sign(preds[tm]) * y[tm])) * 1e4 - cost_bps) \
                    if tm.sum() > 30 else float("nan")
                clears = (not math.isnan(ic)) and ic >= req and abs(t) >= 2.0 and bps > 0
                verdict = "CLEARS" if clears else ""
                out["results"].append(dict(
                    horizon_bars=h, group=gname, model=model_name, n_oos=n_oos,
                    ic=None if math.isnan(ic) else round(ic, 4),
                    t=None if math.isnan(t) else round(t, 2),
                    hit=None if math.isnan(hit) else round(hit, 4),
                    bps_after_cost_q75=None if math.isnan(bps) else round(bps, 2),
                    required_ic=round(req, 4), clears_bar=clears))
                print(f"{gname:<10}{model_name:<7}{n_oos:>7}"
                      f"{ic:>8.3f}{t:>7.2f}{hit:>7.1%}{bps:>15.2f}{verdict:>10}"
                      if not math.isnan(ic) else
                      f"{gname:<10}{model_name:<7}{n_oos:>7}{'nan':>8}")
    any_clears = any(r["clears_bar"] for r in out["results"])
    out["verdict"] = "EDGE_CANDIDATE" if any_clears else "NO_EDGE"
    print(f"\n{asset} VERDICT: {out['verdict']}")
    return out


def main() -> int:
    global MIN_TRAIN
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", choices=["BTC", "ETH", "SOL"], default=None)
    ap.add_argument("--out", default=None, help="write JSON report here")
    ap.add_argument("--min-train", type=int, default=MIN_TRAIN,
                    help="First fit uses this much history; predictions start "
                         "after it. Use ~7200 to keep ALL predictions past the "
                         "split-aware GMM fit boundary (regime features fully "
                         "out-of-fit) — the falsification setting.")
    ap.add_argument("--cost-rt-bps", type=float, default=None,
                    help="[P382] Override the round-trip cost bar (default: "
                         "per-asset 2 x measured CDE taker leg from "
                         "core.cde_fees, floored at 6.0). Use for what-ifs "
                         "(e.g. a percentage-fee venue), never for the "
                         "promotion read.")
    args = ap.parse_args()
    MIN_TRAIN = args.min_train
    global COST_RT_OVERRIDE
    COST_RT_OVERRIDE = args.cost_rt_bps
    assets = [args.asset] if args.asset else ["BTC", "ETH", "SOL"]
    reports = [probe_asset(a) for a in assets]
    overall = ("EDGE_CANDIDATE" if any(r["verdict"] == "EDGE_CANDIDATE" for r in reports)
               else "NO_EDGE")
    print(f"\n========== OVERALL: {overall} ==========")
    if args.out:
        Path(args.out).write_text(json.dumps(reports, indent=1), encoding="utf-8")
    # NO_EDGE exits 1 so a pipeline can gate on it; it is a finding, not an error.
    return 0 if overall == "EDGE_CANDIDATE" else 1


if __name__ == "__main__":
    sys.exit(main())
