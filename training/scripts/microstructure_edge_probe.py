"""[P385-followup] Does TICK-LEVEL microstructure carry edge the 60m kline basis
does not — enough to clear the fee floor and justify a DRL retrain on NEW data?

WHY. P385 proved the CURRENT feature basis tops out at OOS IC ~0.04, ~2x below the
CDE fee floor (required 0.07-0.11). "Retrain DRL with new data" only helps if the
new data raises IC ABOVE that floor. The kline-derived fv2 flow features are
ALREADY in the P385 probe (IC ~0.03, fee-blocked). What klines THROW AWAY is the
tick-level structure: trade-size distribution, large-trade concentration, VPIN-style
volume-imbalance, aggressor persistence. This probes those, from real aggTrades.

This is the Rung-0 gate (P200 ladder): cheap, BEFORE any GPU. If a microstructure
feature clears the required IC at 4h/16h out-of-sample, a DRL retrain on this basis
is justified. If not, tick microstructure does not unlock DRL either, and the honest
lever stays fee/scale (P385).

CAVEAT: the cached window is ~125 days (~750 4H bars), thin for a walk-forward IC.
A hint here warrants extending the download to 1y+; a flat zero here is decisive
against this basis at the horizon that matters.
"""
from __future__ import annotations
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "training"))

from whale_filter_reconstruction_lab import _read_full_day, SYMS, CACHE  # noqa: E402

RAW = REPO / "training" / "training_data" / "raw"
COST_RT = {"BTC": 27.7, "ETH": 44.0, "SOL": 41.0}  # bps, P315/P334
E_ABS_Z, PEARSON_K = 0.7979, 1.047


def micro_features_4h(asset, days):
    """Per-4H-bar tick microstructure features from aggTrades. LEFT-edge bins
    (match resample('4h')); each bar uses only trades WITHIN it (causal)."""
    sym = SYMS[asset]
    rows = []
    for d in days:
        x = _read_full_day(sym, d.isoformat())
        if x is None or not len(x):
            continue
        ts = x["ts"].to_numpy(float)
        notl = x["notional_usd"].to_numpy(float)
        buy = (x["side"].to_numpy() == "BUY")
        bar = (ts // (4 * 3600)).astype("int64")  # left-edge bin id
        df = pd.DataFrame({"bar": bar, "notl": notl, "buy": buy})
        for b, g in df.groupby("bar"):
            n = len(g)
            if n < 50:
                continue
            v = g["notl"].to_numpy()
            bv = g.loc[g["buy"], "notl"].sum()
            sv = g.loc[~g["buy"], "notl"].sum()
            tot = bv + sv
            thr = np.quantile(v, 0.99)
            rows.append({
                "bar_s": int(b) * 4 * 3600,
                "vpin": abs(bv - sv) / tot if tot > 0 else 0.0,        # |imbalance|
                "signed_imb": (bv - sv) / tot if tot > 0 else 0.0,     # directional
                "large_share": v[v >= thr].sum() / tot if tot > 0 else 0.0,  # top-1% conc
                "size_cv": float(v.std() / v.mean()) if v.mean() > 0 else 0.0,
                "aggr_cnt_imb": (g["buy"].sum() - (~g["buy"]).sum()) / n,   # by count
                "intensity": float(n),
            })
    if not rows:
        return None
    f = pd.DataFrame(rows).set_index("bar_s")
    f.index = pd.to_datetime(f.index, unit="s", utc=True)
    # normalize intensity/size_cv (level features) to trailing z
    for c in ("intensity", "size_cv", "large_share", "vpin"):
        m = f[c].rolling(180, min_periods=30).mean()
        s = f[c].rolling(180, min_periods=30).std()
        f[c + "_z"] = ((f[c] - m) / s).clip(-5, 5)
    return f


def load_close_4h(asset):
    d = pd.read_parquet(RAW / f"{asset}_60m.parquet")[["timestamp", "close"]]
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
    return d.set_index("timestamp")["close"].resample("4h").last().dropna()


def spearman(x, y):
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 50:
        return float("nan")
    rx = np.argsort(np.argsort(x[m])).astype(float)
    ry = np.argsort(np.argsort(y[m])).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def walk_forward_ic(X, y, min_train=350, refit=60):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    n = len(X)
    preds = np.full(n, np.nan)
    s = min_train
    while s + 3 < n:
        te = min(s + refit, n)
        Xtr, ytr = X[:s], y[:s]
        m = ~(np.isnan(Xtr).any(axis=1) | np.isnan(ytr))
        if m.sum() >= 150:
            sc = StandardScaler().fit(Xtr[m])
            mdl = Ridge(alpha=10.0).fit(sc.transform(Xtr[m]), ytr[m])
            preds[s + 3:te] = mdl.predict(sc.transform(np.nan_to_num(X[s + 3:te])))
        s = te
    return preds


def main():
    ndays = int(sys.argv[1]) if len(sys.argv) > 1 else 125
    assets = sys.argv[2].split(",") if len(sys.argv) > 2 else ["BTC"]
    end = date(2026, 7, 31)
    days = [end - timedelta(days=i) for i in range(ndays)][::-1]
    feat_cols = ["signed_imb", "aggr_cnt_imb", "vpin_z", "large_share_z",
                 "size_cv_z", "intensity_z"]
    res = {"ndays": ndays, "cost_rt_bps": COST_RT, "assets": {}}
    W = 96
    print("=" * W)
    print(f"  TICK-MICROSTRUCTURE EDGE PROBE — new-data Rung-0 gate, {ndays}d aggTrades")
    print("  clears = OOS IC >= required-to-cover-CDE-cost at that horizon")
    print("=" * W)
    for a in assets:
        f = micro_features_4h(a, days)
        if f is None:
            print(f"\n{a}: no aggTrades"); continue
        close = load_close_4h(a)
        j = f.join(close.rename("close"), how="inner").dropna(subset=["close"])
        if len(j) < 300:
            print(f"\n{a}: only {len(j)} bars — too thin"); continue
        c = j["close"].to_numpy(float)
        X = j[feat_cols].to_numpy(float)
        out = {"n_bars": len(j), "window": [str(j.index.min())[:10], str(j.index.max())[:10]],
               "horizons": {}}
        print(f"\n{a}  ({len(j)} 4H bars, {out['window'][0]} -> {out['window'][1]}):")
        for h, hn in ((1, "4h"), (4, "16h")):
            fwd = np.full(len(c), np.nan); fwd[:len(c) - h] = c[h:] / c[:len(c) - h] - 1.0
            sigma = float(np.nanstd(fwd) * 1e4)
            req = COST_RT[a] / (E_ABS_Z * PEARSON_K * sigma) if sigma > 0 else float("inf")
            # per-feature raw IC
            raw = {fc: round(spearman(j[fc].to_numpy(float), fwd), 4) for fc in feat_cols}
            best_fc = max(raw, key=lambda k: abs(raw[k]) if raw[k] == raw[k] else 0)
            # walk-forward model IC (all features)
            preds = walk_forward_ic(X, fwd)
            tm = ~(np.isnan(preds) | np.isnan(fwd))
            ic = spearman(preds[tm], fwd[tm]) if tm.sum() > 50 else float("nan")
            clears = bool(ic == ic and ic >= req)
            out["horizons"][hn] = {"req_ic": round(req, 4), "model_oos_ic": round(ic, 4),
                                   "n_oos": int(tm.sum()), "best_raw_feat": best_fc,
                                   "best_raw_ic": raw[best_fc], "raw_ic": raw,
                                   "clears": clears}
            print(f"  {hn}: req IC {req:.3f} | model OOS IC {ic:+.4f} (n={int(tm.sum())}) | "
                  f"best raw: {best_fc} {raw[best_fc]:+.4f}  [{'CLEARS' if clears else '-'}]")
        res["assets"][a] = out
    (REPO / "training" / "reports" / "microstructure_edge_probe_p385.json"
     ).write_text(json.dumps(res, indent=2), encoding="utf-8")
    any_clear = any(hh.get("clears") for r in res["assets"].values()
                    for hh in r.get("horizons", {}).values())
    print("\n" + "=" * W)
    print(f"  VERDICT: {'PULSE — tick microstructure worth a DRL retrain (extend window first)' if any_clear else 'NO PULSE at 4H/16H — tick microstructure does not clear the fee floor either'}")
    print("  report -> training/reports/microstructure_edge_probe_p385.json")
    print("  NOTE: ~125d is thin; a hint warrants extending to 1y+, a flat zero is decisive.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
