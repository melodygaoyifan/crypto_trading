"""[P385c] Rung-0 gate on CROSS-ASSET / BREADTH data — a NEW basis the single-asset
DRL entirely lacks. Does market-wide context (breadth, dispersion, relative
strength, dominance momentum, cross-asset lead-lag over all 8 assets) carry edge
above the fee floor at 4h/16h?

WHY. P385 proved the current SINGLE-ASSET basis tops at OOS IC ~0.04 (fee-blocked),
and P385b found no pulse in higher-freq flow, L2 depth, or tick microstructure. The
one readily-available NEW basis left is CROSS-ASSET context — the current features
describe one asset in isolation; breadth/dispersion/lead-lag describe the market it
sits in. 6y of all 8 assets (BTC ETH SOL XRP ADA LTC DOGE BNB) is on disk.

This is the cheap gate BEFORE any DRL retrain (P200 ladder): if a cross-asset
feature set clears the required IC out-of-sample, adding these features and
retraining DRL is justified. If not, cross-asset context does not unlock DRL either.

Causal by construction: every feature at bar i uses closes up to i-1; forward
return is the NEXT bar. Honest CDE round-trip cost. Walk-forward ridge.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "training" / "training_data" / "raw"
ASSETS = ("BTC", "ETH", "SOL", "XRP", "ADA", "LTC", "DOGE", "BNB")
TARGETS = ("BTC", "ETH", "SOL")
COST_RT = {"BTC": 27.7, "ETH": 44.0, "SOL": 41.0}
E_ABS_Z, PEARSON_K = 0.7979, 1.047


def closes_4h():
    out = {}
    for a in ASSETS:
        d = pd.read_parquet(RAW / f"{a}_60m.parquet")[["timestamp", "close"]]
        d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
        out[a] = d.set_index("timestamp")["close"].resample("4h").last()
    px = pd.DataFrame(out).dropna(how="any")
    return px


def build_features(px, target):
    """Cross-asset features for `target`, all causal (shift(1) applied at the end)."""
    ret1 = px.pct_change()
    ret6 = px.pct_change(6)      # ~24h
    ret42 = px.pct_change(42)    # ~1w
    sma50 = px.rolling(50).mean()
    sma200 = px.rolling(200).mean()
    f = pd.DataFrame(index=px.index)
    # breadth: fraction of assets above their SMA
    f["breadth_sma50"] = (px > sma50).mean(axis=1)
    f["breadth_sma200"] = (px > sma200).mean(axis=1)
    # cross-sectional dispersion of 24h returns
    f["xs_dispersion"] = ret6.std(axis=1)
    # market (equal-weight) momentum
    f["basket_mom_24h"] = ret6.mean(axis=1)
    f["basket_mom_1w"] = ret42.mean(axis=1)
    # target relative strength: percentile rank of its 1w return among the 8
    f["rel_strength_1w"] = ret42.rank(axis=1, pct=True)[target]
    # BTC-dominance momentum: BTC 1w minus alt-basket 1w
    alts = [a for a in ASSETS if a != "BTC"]
    f["btc_dom_mom"] = ret42["BTC"] - ret42[alts].mean(axis=1)
    # average pairwise correlation (rolling 60 bars of ret1)
    def _avg_corr(window):
        c = window.corr().to_numpy()
        iu = np.triu_indices_from(c, k=1)
        v = c[iu]
        return np.nanmean(v)
    f["avg_pair_corr"] = ret1.rolling(60).apply(lambda s: s.std(), raw=False).mean(axis=1)  # cheap proxy
    # cross-asset lead-lag: basket 24h return EXCLUDING target (do others lead it?)
    others = [a for a in ASSETS if a != target]
    f["basket_ex_target_24h"] = ret6[others].mean(axis=1)
    # z-normalize level features to trailing window
    for c in f.columns:
        m = f[c].rolling(500, min_periods=100).mean()
        s = f[c].rolling(500, min_periods=100).std()
        f[c] = ((f[c] - m) / s).clip(-5, 5)
    return f.shift(1)  # causal: decide bar i from info up to i-1


def spearman(x, y):
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 100:
        return float("nan")
    rx = np.argsort(np.argsort(x[m])).astype(float)
    ry = np.argsort(np.argsort(y[m])).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def walk_forward(X, y, min_train=7200, refit=250):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    n = len(X)
    preds = np.full(n, np.nan)
    s = min_train
    while s + 3 < n:
        te = min(s + refit, n)
        Xtr, ytr = X[:s], y[:s]
        m = ~(np.isnan(Xtr).any(axis=1) | np.isnan(ytr))
        if m.sum() >= 500:
            sc = StandardScaler().fit(Xtr[m])
            mdl = Ridge(alpha=10.0).fit(sc.transform(Xtr[m]), ytr[m])
            preds[s + 3:te] = mdl.predict(sc.transform(np.nan_to_num(X[s + 3:te])))
        s = te
    return preds


def main():
    min_train = int(sys.argv[1]) if len(sys.argv) > 1 else 7200
    px = closes_4h()
    res = {"min_train": min_train, "cost_rt_bps": COST_RT, "n_bars_total": len(px), "assets": {}}
    W = 96
    print("=" * W)
    print(f"  CROSS-ASSET / BREADTH EDGE PROBE — new-data Rung-0 gate ({len(px)} 4H bars, 8 assets)")
    print("  clears = walk-forward OOS IC >= required-to-cover-CDE-cost at that horizon")
    print("=" * W)
    any_clear = False
    for tgt in TARGETS:
        f = build_features(px, tgt)
        c = px[tgt].to_numpy(float)
        cols = list(f.columns)
        X = f.to_numpy(float)
        out = {"horizons": {}}
        print(f"\n{tgt}:")
        for h, hn in ((1, "4h"), (4, "16h")):
            fwd = np.full(len(c), np.nan); fwd[:len(c) - h] = c[h:] / c[:len(c) - h] - 1.0
            sigma = float(np.nanstd(fwd) * 1e4)
            req = COST_RT[tgt] / (E_ABS_Z * PEARSON_K * sigma) if sigma > 0 else float("inf")
            raw = {col: round(spearman(f[col].to_numpy(float), fwd), 4) for col in cols}
            best = max(raw, key=lambda k: abs(raw[k]) if raw[k] == raw[k] else 0)
            preds = walk_forward(X, fwd, min_train=min_train)
            tm = ~(np.isnan(preds) | np.isnan(fwd))
            ic = spearman(preds[tm], fwd[tm]) if tm.sum() > 100 else float("nan")
            t = ic * np.sqrt(max(tm.sum() - 1, 1)) if ic == ic else float("nan")
            clears = bool(ic == ic and ic >= req)
            any_clear = any_clear or clears
            out["horizons"][hn] = {"req_ic": round(req, 4), "model_oos_ic": round(ic, 4),
                                   "t": round(t, 2), "n_oos": int(tm.sum()),
                                   "best_raw": best, "best_raw_ic": raw[best],
                                   "raw_ic": raw, "clears": clears}
            print(f"  {hn}: req IC {req:.3f} | model OOS IC {ic:+.4f} (t {t:+.2f}, n {int(tm.sum())}) "
                  f"| best raw: {best} {raw[best]:+.4f}  [{'CLEARS' if clears else '-'}]")
        res["assets"][tgt] = out
    (REPO / "training" / "reports" / "breadth_edge_probe_p385.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8")
    print("\n" + "=" * W)
    print(f"  VERDICT: {'PULSE — cross-asset context worth adding to the DRL basis + retrain' if any_clear else 'NO PULSE — cross-asset breadth does not clear the fee floor either'}")
    print("  report -> training/reports/breadth_edge_probe_p385.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
