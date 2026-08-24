"""[P390] The REAL out-of-sample gate on the OI-positioning lead, now that the
Binance Vision metrics backfill (fetch_binance_metrics.py) gives ~5.5y of OI +
long/short + taker-ratio history spanning the training folds.

This resolves the two blockers on P388's lead:
  - the feature now has multi-year history (learnable), and
  - there is genuine OUT-OF-SAMPLE data to run the hold-aware gate on.

The 186d in-sample screen (oi_z raw IC -0.15, SOL held +81%) was IN-SAMPLE and
possibly regime-local (funding screened strong on 186d too but is ~0.04 over 6y).
This is the honest test: does the lead survive HELD, walk-forward, out-of-sample?

METHOD: build causal features from the metrics (oi_z = rolling-z of OI level;
toptrader/global/taker long-short z), predict fwd 24h return walk-forward (ridge),
hold-position with a PRE-COMMITTED deadband (1.0 — chosen before the run, not swept),
long/short + funding carry, honest CDE cost on flips only. Also the single-feature
oi_z CONTRARIAN held signal (the specific lead). Compared to buy-and-hold.

VERDICT (pre-committed): the positioning bundle EARNS iff its held OOS net-after-cost
is POSITIVE and > buy-and-hold on >= 2/3 assets at the pre-committed band. Otherwise
the 186d in-sample strength was regime-local/artifact and NO retrain is justified.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "training" / "training_data" / "raw"
CG = REPO / "training" / "training_data" / "coinglass_history"
ASSETS = ("BTC", "ETH", "SOL")
COST_RT = {"BTC": 27.7e-4, "ETH": 44.0e-4, "SOL": 41.0e-4}
PRECOMMIT_BAND = 1.0
H = 6  # 24h forward (the horizon where the raw screen was strongest)


def _z(s, w=180):
    z = (s - s.rolling(w, min_periods=30).mean()) / s.rolling(w, min_periods=30).std()
    return z.replace([np.inf, -np.inf], np.nan).clip(-5, 5)


def load(a):
    px = pd.read_parquet(RAW / f"{a}_60m.parquet")[["timestamp", "close"]]
    px["timestamp"] = pd.to_datetime(px["timestamp"], utc=True)
    c4 = px.set_index("timestamp")["close"].resample("4h").last()
    m = pd.read_parquet(CG / f"{a}_metrics_4h.parquet")
    m["timestamp"] = pd.to_datetime(m["timestamp"], utc=True)
    m = m.set_index("timestamp")
    d = pd.DataFrame({"close": c4})
    d["oi_z"] = _z(m["oi_close"])
    d["oi_chg"] = m["oi_close"].pct_change().replace([np.inf, -np.inf], np.nan)
    d["top_ls_z"] = _z(m["toptrader_ls_ratio"])
    d["glob_ls_z"] = _z(m["global_ls_ratio"])
    d["taker_ls_z"] = _z(m["taker_ls_ratio"])
    # funding (6y daily) for carry
    try:
        fu = pd.read_parquet(CG / f"{a}_funding_1d.parquet")
        fu["timestamp"] = pd.to_datetime(fu["timestamp"], utc=True)
        d["fund"] = fu.set_index("timestamp")["funding_close"].reindex(d.index, method="ffill") / 6.0
    except Exception:
        d["fund"] = 0.0
    return d.dropna(subset=["close"])


def stats(pnl):
    pnl = pnl[~np.isnan(pnl)]
    if len(pnl) < 50:
        return dict(net=0.0, sh=0.0, dd=0.0)
    eq = np.cumprod(1 + pnl); dd = ((np.maximum.accumulate(eq) - eq) / np.maximum.accumulate(eq)).max()
    sd = pnl.std()
    return dict(net=round(float(pnl.sum() * 100), 1),
                sh=round(float(pnl.mean() / sd * np.sqrt(6 * 365.25)), 2) if sd > 0 else 0.0,
                dd=round(float(dd * 100), 1))


def hold_sim(close, sig_z, fund, per_leg, band, contrarian):
    n = len(close); ret = np.zeros(n); ret[1:] = close[1:] / close[:-1] - 1.0
    pos = np.zeros(n); cur = 0.0
    s = 1.0 if not contrarian else -1.0
    for i in range(n):
        if not np.isfinite(sig_z[i]): pos[i] = cur; continue
        if sig_z[i] > band: cur = s
        elif sig_z[i] < -band: cur = -s
        pos[i] = cur
    dpos = np.abs(np.diff(pos, prepend=0.0))
    pnl = np.zeros(n); pnl[:-1] = pos[:-1] * ret[1:]
    pnl = pnl - dpos * (per_leg / 2.0) - pos * fund
    tr = float(dpos.sum() / (n / (6 * 365.25)))
    st = stats(pnl); st["trades_yr"] = round(tr, 1)
    return st


def walk_forward(X, y, min_train, refit=250):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    n = len(X); preds = np.full(n, np.nan); s = min_train
    while s + 3 < n:
        te = min(s + refit, n)
        mtr = np.isfinite(X[:s]).all(axis=1) & np.isfinite(y[:s])
        if mtr.sum() >= 400:
            sc = StandardScaler().fit(X[:s][mtr]); mdl = Ridge(alpha=10.0).fit(sc.transform(X[:s][mtr]), y[:s][mtr])
            preds[s + 3:te] = mdl.predict(sc.transform(np.nan_to_num(X[s + 3:te])))
        s = te
    return preds


def main():
    feat = ["oi_z", "oi_chg", "top_ls_z", "glob_ls_z", "taker_ls_z"]
    res = {"band": PRECOMMIT_BAND, "horizon_bars": H, "assets": {}}
    earns = 0; measured = 0
    W = 96
    print("=" * W)
    print(f"  OI-POSITIONING OOS HOLD-AWARE GATE (metrics backfill, walk-forward, band={PRECOMMIT_BAND})")
    print("  VERDICT: held OOS net POSITIVE and > buy&hold on >=2/3 assets (pre-committed band)")
    print("=" * W)
    for a in ASSETS:
        try:
            d = load(a)
        except Exception as e:
            print(f"\n{a}: load failed ({e})"); continue
        n = len(d); c = d["close"].to_numpy(float)
        if n < 3000:
            print(f"\n{a}: only {n} bars — backfill incomplete?"); continue
        min_train = n // 2  # first half train, validate on OOS second half + walk-forward
        fund = d["fund"].fillna(0).to_numpy()
        # buy&hold over OOS region
        bh = np.zeros(n); bh[min_train:-1] = c[min_train + 1:] / c[min_train:-1] - 1.0
        bh_st = stats(bh)
        # single-feature oi_z contrarian, held, OOS only (mask pre-OOS to flat)
        oiz = d["oi_z"].to_numpy(float).copy(); oiz[:min_train] = np.nan
        oi_st = hold_sim(c, oiz, fund, COST_RT[a], PRECOMMIT_BAND, contrarian=True)
        # bundle ridge -> predicted fwd return -> held (sign of prediction z)
        fwd = np.full(n, np.nan); fwd[:n - H] = c[H:] / c[:n - H] - 1.0
        X = d[feat].to_numpy(float)
        pr = walk_forward(X, fwd, min_train)
        prz = _z(pd.Series(pr)).to_numpy()
        bundle_st = hold_sim(c, prz, fund, COST_RT[a], PRECOMMIT_BAND, contrarian=False)
        best = max(oi_st, bundle_st, key=lambda s: s["net"])
        ok = best["net"] > 0 and best["net"] > bh_st["net"]
        measured += 1; earns += 1 if ok else 0
        res["assets"][a] = {"n_bars": n, "oos_from": str(d.index[min_train])[:10],
                            "buy_hold": bh_st, "oi_z_contrarian": oi_st,
                            "bundle_ridge": bundle_st, "earns": ok}
        print(f"\n{a} ({n} bars, OOS from {str(d.index[min_train])[:10]}):")
        print(f"  buy&hold        : net {bh_st['net']:+8.1f}%  Sh {bh_st['sh']:+.2f}  dd {bh_st['dd']}%")
        print(f"  oi_z contrarian : net {oi_st['net']:+8.1f}%  Sh {oi_st['sh']:+.2f}  dd {oi_st['dd']}%  tr/yr {oi_st['trades_yr']}")
        print(f"  bundle ridge    : net {bundle_st['net']:+8.1f}%  Sh {bundle_st['sh']:+.2f}  dd {bundle_st['dd']}%  tr/yr {bundle_st['trades_yr']}")
        print(f"  -> {'EARNS' if ok else 'no'}")
    res["earns_count"] = earns; res["measured"] = measured
    res["verdict"] = ("EARNS — OI positioning survives OOS; retrain JUSTIFIED"
                      if earns >= 2 else "NOT_EARNED — 186d in-sample was regime-local; NO retrain")
    (REPO / "training" / "reports" / "metrics_oos_probe_p390.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8")
    print("\n" + "=" * W)
    print(f"  VERDICT: {res['verdict']}  ({earns}/{measured} assets)")
    print("  report -> training/reports/metrics_oos_probe_p390.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
