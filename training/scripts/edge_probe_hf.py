"""[P375 Stage-0] Higher-frequency order-flow edge probe — the GATE before any
venue/data/capital spend on the predictor bet (docs/research/GROWTH_PROGRAM_2026-08.md).

Question: does a HIGHER-FREQUENCY (1h/4h/12h) predictor using ORDER-FLOW have a
pulse on data ALREADY ON DISK, at a percentage-venue cost? The 60m archives carry
6y of hourly flow (taker_buy_base/quote, count, quote_volume) for all 8 assets —
the "order-flow > price features" basis the literature points to (Nguyen 2026).

Discipline (P200-LADDER Rung-0, P281): a cheap supervised edge probe BEFORE any
GPU/data/venue spend. Walk-forward, purged, Spearman IC + after-cost bps, scored
at BOTH cost models (percentage venue ~10bps RT; CDE flat per-asset). Required-IC
bar from the P166 arithmetic. Every feature causal + a P164 construction test.

This does NOT build a tradeable model. It tells you whether Stage 1 (procure
data / change venue) is worth starting. Kill: if even the flow probe on existing
data is dead at percentage cost, the predictor needs exotic (L2/tick) data.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "training" / "training_data" / "raw"
ASSETS = ("BTC", "ETH", "SOL")
HORIZONS = (1, 4, 12)                 # bars = hours forward
COST_PCT_RT = 10.0                    # percentage venue: ~4.5bps fee/leg + half-spread
COST_CDE_RT = {"BTC": 27.7, "ETH": 44.0, "SOL": 41.0}   # flat per-contract (P315/P334)
MIN_TRAIN = 8000                      # hourly bars (~11 months) before first fit
GAP = 24
REFIT = 2000
E_ABS_Z = 0.7979
PEARSON_K = 1.047


def spearman(x, y):
    if len(x) < 50:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def required_ic(cost_bps, sigma_fwd_bps):
    if sigma_fwd_bps <= 0:
        return float("inf")
    return cost_bps / (E_ABS_Z * PEARSON_K * sigma_fwd_bps)


def _roll(a, w, fn, n):
    out = np.full(n, np.nan)
    for i in range(w, n):
        out[i] = fn(a[i - w:i])       # STRICTLY past (excludes bar i)
    return out


def build_features(df):
    """All causal: every feature at bar i uses only bars <= i-1 (lagged)."""
    c = df["close"].to_numpy(float)
    vol = df["volume"].to_numpy(float)
    qv = df["quote_volume"].to_numpy(float)
    cnt = df["count"].to_numpy(float)
    tbb = df["taker_buy_base"].to_numpy(float)
    n = len(c)
    logret = np.zeros(n)
    logret[1:] = np.log(c[1:] / c[:-1])
    feats, names = {}, []
    for w in (4, 12, 24, 72):
        col = np.full(n, np.nan)
        col[w:] = c[w:] / c[:-w] - 1.0
        feats["mom_%d" % w] = col
        names.append("mom_%d" % w)
    feats["vol_24"] = _roll(logret, 24, np.std, n)
    names.append("vol_24")
    tbf = np.divide(tbb, vol, out=np.full(n, 0.5), where=vol > 0)
    tbf_lag = np.concatenate([[np.nan], tbf[:-1]])
    feats["tbf"] = tbf_lag
    names.append("tbf")
    m = _roll(tbf, 72, np.mean, n)
    s = _roll(tbf, 72, np.std, n)
    feats["tbf_z"] = (tbf_lag - m) / np.where(s > 0, s, np.nan)
    names.append("tbf_z")
    lc = np.log1p(cnt)
    lc_lag = np.concatenate([[np.nan], lc[:-1]])
    mc = _roll(lc, 72, np.mean, n)
    sc = _roll(lc, 72, np.std, n)
    feats["cnt_z"] = (lc_lag - mc) / np.where(sc > 0, sc, np.nan)
    names.append("cnt_z")
    amh = np.divide(np.abs(logret), qv, out=np.full(n, np.nan), where=qv > 0)
    feats["amihud_z"] = _roll(amh, 72, np.mean, n)
    names.append("amihud_z")
    X = np.column_stack([feats[k] for k in names])
    groups = {
        "price": [i for i, k in enumerate(names) if k.startswith(("mom_", "vol_"))],
        "flow": [i for i, k in enumerate(names) if k in ("tbf", "tbf_z", "cnt_z", "amihud_z")],
        "all": list(range(len(names))),
    }
    return X, names, groups


def walk_forward(X, y):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    n = len(X)
    start = MIN_TRAIN
    preds = np.full(n, np.nan)
    while start + GAP < n:
        te_s = start + GAP
        te_e = min(te_s + REFIT, n)
        if te_s >= n:
            break
        Xtr, ytr = X[:start], y[:start]
        mask = ~(np.isnan(Xtr).any(axis=1) | np.isnan(ytr))
        if mask.sum() >= 1000:
            sc = StandardScaler().fit(Xtr[mask])
            m = Ridge(alpha=10.0).fit(sc.transform(Xtr[mask]), ytr[mask])
            preds[te_s:te_e] = m.predict(sc.transform(np.nan_to_num(X[te_s:te_e])))
        start = te_e
    return preds


def causal_check(df):
    """P164: perturb the FUTURE violently; features at earlier bars must not move."""
    X0, _, _ = build_features(df)
    d2 = df.copy()
    d2.iloc[-50:, d2.columns.get_loc("close")] *= 3.0
    X1, _, _ = build_features(d2)
    k = len(df) - 60
    return bool(np.allclose(np.nan_to_num(X0[:k]), np.nan_to_num(X1[:k]), atol=1e-9))


def probe_asset(asset):
    df = pd.read_parquet(RAW / ("%s_60m.parquet" % asset)).sort_values("timestamp").reset_index(drop=True)
    if not causal_check(df):
        raise SystemExit("%s: FEATURE LEAK — probe invalid" % asset)
    X, names, groups = build_features(df)
    c = df["close"].to_numpy(float)
    n = len(c)
    out = {"asset": asset, "causal_ok": True, "horizons": {}}
    for h in HORIZONS:
        fwd = np.full(n, np.nan)
        fwd[:n - h] = c[h:] / c[:n - h] - 1.0
        sigma_bps = float(np.nanstd(fwd) * 1e4)
        rec = {"sigma_fwd_bps": round(sigma_bps, 1),
               "req_ic_pct": round(required_ic(COST_PCT_RT, sigma_bps), 4),
               "req_ic_cde": round(required_ic(COST_CDE_RT[asset], sigma_bps), 4),
               "groups": {}}
        for g, cols in groups.items():
            preds = walk_forward(X[:, cols], fwd)
            tm = ~(np.isnan(preds) | np.isnan(fwd))
            if tm.sum() < 500:
                continue
            ic = spearman(preds[tm], fwd[tm])
            gross = float(np.nanmean(np.sign(preds[tm]) * fwd[tm]) * 1e4)
            ic_ok = ic == ic
            rec["groups"][g] = {
                "ic": round(ic, 4), "n": int(tm.sum()),
                "gross_bps": round(gross, 2),
                "net_pct_venue": round(gross - COST_PCT_RT, 2),
                "net_cde": round(gross - COST_CDE_RT[asset], 2),
                "clears_pct": bool(ic_ok and ic >= rec["req_ic_pct"] and gross > COST_PCT_RT),
                "clears_cde": bool(ic_ok and ic >= rec["req_ic_cde"] and gross > COST_CDE_RT[asset])}
        out["horizons"][str(h)] = rec
    return out


def main():
    res = {"assets": {}, "cost_pct_rt": COST_PCT_RT, "cost_cde_rt": COST_CDE_RT}
    for a in ASSETS:
        res["assets"][a] = probe_asset(a)
    (REPO / "training" / "reports" / "edge_probe_hf_p375.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8")
    W = 92
    print("=" * W)
    print("  HIGHER-FREQUENCY ORDER-FLOW EDGE PROBE (60m data, on disk) — Stage-0 gate")
    print("  clears = IC>=required AND gross>cost. pct-venue=10bps RT | CDE=flat per-asset")
    print("=" * W)
    any_pulse = False
    for a in ASSETS:
        print("\n%s:" % a)
        for h, rec in res["assets"][a]["horizons"].items():
            print("  %dh fwd (sigma %sbps, req IC %.4f pct / %.4f CDE):"
                  % (int(h), rec["sigma_fwd_bps"], rec["req_ic_pct"], rec["req_ic_cde"]))
            for g, gr in rec["groups"].items():
                if gr["clears_pct"]:
                    flag = "CLEARS-pct-venue"
                elif gr["clears_cde"]:
                    flag = "clears-CDE"
                else:
                    flag = "-"
                if gr["clears_pct"] or gr["clears_cde"]:
                    any_pulse = True
                print("    %-6s IC %+.4f  gross %+7.2fbps  net(pctV %+.1f / CDE %+.1f)  [%s]"
                      % (g, gr["ic"], gr["gross_bps"], gr["net_pct_venue"], gr["net_cde"], flag))
    print("\n" + "=" * W)
    verdict = ("PULSE FOUND — predictor bet worth Stage 1" if any_pulse
               else "NO PULSE on existing data — predictor needs exotic (L2/tick) data")
    print("  STAGE-0 VERDICT: %s" % verdict)
    print("  report -> training/reports/edge_probe_hf_p375.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
