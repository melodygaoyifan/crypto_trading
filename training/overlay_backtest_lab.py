"""[P377] The reframe: is the certified trend signal a viable DRAWDOWN-REDUCTION
OVERLAY on a hold, judged as a PRODUCT rather than as an alpha engine?

WHY. Every audit converged: the SMA200 trend/hold rule loses to buy-and-hold on
RETURN but wins on Sharpe/drawdown (P318/P321). As an ALPHA engine that is a
failure; as a "hold BTC/ETH/SOL with less pain" PRODUCT it may be exactly right.
This lab judges it on the product's terms, net of honest CDE cost, on 6y of
on-disk data.

THE OVERLAY. Long-or-flat only (never short — the product is "own the assets,
step aside in downtrends"): hold the asset when close > SMA200, flat otherwise.
This is close to what the live ETH/SOL books already do; the point here is to
MEASURE it as a product and give it a pass/fail on the right criterion.

THE BAR, pre-committed before the first number: the overlay is a VIABLE PRODUCT
iff, net of honest cost over the full 6y AND in >=2 of 3 eras, it cuts max
drawdown by >= 25% versus pure hold WHILE retaining >= 50% of hold's total
return. (A drawdown-reduction product is judged on risk kept vs return given up,
not on beating hold's return — which it cannot, by construction.) Reported for
each asset and the equal-weight 3-asset portfolio.

CONVENTIONS: 4H bars (live cadence) resampled from the 6y 60m archives; SMA over
200 4H bars (~33d); causal (position at bar i from close/SMA at i-1); honest CDE
round-trip cost charged on each position change (BTC 27.7 / ETH 44.0 / SOL 41.0
bps, P315/P334); additive per-bar return sums, the repo lab convention.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "training" / "training_data" / "raw"
ASSETS = ("BTC", "ETH", "SOL")
COST_RT = {"BTC": 27.7e-4, "ETH": 44.0e-4, "SOL": 41.0e-4}
SMA_BARS = 200
ERAS = {"2020-22": ("2020-01-01", "2023-01-01"),
        "2023-24": ("2023-01-01", "2025-01-01"),
        "2025-26": ("2025-01-01", "2027-01-01")}
DD_CUT_MIN = 0.25       # overlay must cut maxDD by >= 25%
RET_KEEP_MIN = 0.50     # while retaining >= 50% of hold's return


def load_4h(asset):
    d = pd.read_parquet(RAW / ("%s_60m.parquet" % asset))[["timestamp", "close"]]
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
    s = d.set_index("timestamp")["close"].resample("4h").last().dropna()
    return s


def _series_stats(pnl, ts):
    eq = np.cumprod(1 + pnl)
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / peak
    sd = pnl.std()
    return {"total_pct": round(float(pnl.sum() * 100), 1),
            "sharpe": round(float(pnl.mean() / sd * np.sqrt(6 * 365.25)), 2) if sd > 0 else 0.0,
            "maxdd_pct": round(float(dd.max() * 100), 1),
            "_pnl": pnl, "_ts": ts}


def _era_slice(pnl, ts, a, b):
    m = np.asarray((ts >= a) & (ts < b))
    return pnl[m], ts[m]


def run_asset(asset):
    px = load_4h(asset)
    ts = px.index
    c = px.to_numpy(float)
    n = len(c)
    ret = np.zeros(n)
    ret[1:] = c[1:] / c[:-1] - 1.0
    sma = px.rolling(SMA_BARS).mean().to_numpy(float)
    # causal long/flat: position for bar i decided from close/SMA at i-1
    long_flat = np.zeros(n)
    for i in range(SMA_BARS + 1, n):
        long_flat[i] = 1.0 if (c[i - 1] > sma[i - 1]) else 0.0
    hold = np.ones(n)
    hold[:SMA_BARS + 1] = 0.0  # align start with the overlay's first live bar

    def pnl_of(pos):
        p = pos * ret
        turn = np.abs(np.diff(pos, prepend=pos[0]))
        p = p - turn * COST_RT[asset]
        return p

    hold_pnl = pnl_of(hold)
    ovl_pnl = pnl_of(long_flat)
    hs = _series_stats(hold_pnl, ts)
    os_ = _series_stats(ovl_pnl, ts)
    out = {"asset": asset, "hold": {k: v for k, v in hs.items() if not k.startswith("_")},
           "overlay": {k: v for k, v in os_.items() if not k.startswith("_")},
           "exposure": round(float(long_flat.mean()), 3), "eras": {}}
    # full-window product metrics
    out["dd_cut"] = round(1 - os_["maxdd_pct"] / hs["maxdd_pct"], 3) if hs["maxdd_pct"] > 0 else 0.0
    out["ret_keep"] = round(os_["total_pct"] / hs["total_pct"], 3) if hs["total_pct"] > 0 else 0.0
    # per era
    era_pass = 0
    for name, (a, b) in ERAS.items():
        hp, _ = _era_slice(hold_pnl, ts, a, b)
        op, _ = _era_slice(ovl_pnl, ts, a, b)
        if len(hp) < 200:
            continue
        h_eq = np.cumprod(1 + hp); h_dd = float(((np.maximum.accumulate(h_eq) - h_eq) / np.maximum.accumulate(h_eq)).max())
        o_eq = np.cumprod(1 + op); o_dd = float(((np.maximum.accumulate(o_eq) - o_eq) / np.maximum.accumulate(o_eq)).max())
        h_ret = float(hp.sum()); o_ret = float(op.sum())
        dcut = (1 - o_dd / h_dd) if h_dd > 0 else 0.0
        rkeep = (o_ret / h_ret) if h_ret > 0 else (1.0 if o_ret >= 0 else 0.0)
        ok = (dcut >= DD_CUT_MIN and rkeep >= RET_KEEP_MIN)
        era_pass += 1 if ok else 0
        out["eras"][name] = {"hold_ret": round(h_ret * 100, 1), "ovl_ret": round(o_ret * 100, 1),
                             "hold_dd": round(h_dd * 100, 1), "ovl_dd": round(o_dd * 100, 1),
                             "dd_cut": round(dcut, 2), "ret_keep": round(rkeep, 2), "pass": ok}
    full_ok = out["dd_cut"] >= DD_CUT_MIN and out["ret_keep"] >= RET_KEEP_MIN
    out["verdict"] = "VIABLE PRODUCT" if (full_ok and era_pass >= 2) else "fails"
    out["_hold_pnl"] = hold_pnl
    out["_ovl_pnl"] = ovl_pnl
    out["_ts"] = ts
    return out


def main():
    res = {"assets": {}, "cost_rt": COST_RT, "bar": {"dd_cut_min": DD_CUT_MIN, "ret_keep_min": RET_KEEP_MIN}}
    per = {}
    for a in ASSETS:
        per[a] = run_asset(a)
        res["assets"][a] = {k: v for k, v in per[a].items() if not k.startswith("_")}
    # equal-weight 3-asset portfolio, aligned on the common index
    idx = per["BTC"]["_ts"]
    for a in ("ETH", "SOL"):
        idx = idx.intersection(per[a]["_ts"])
    def align(a, key):
        # [P378] `np.asarray` is load-bearing, not decoration. `pnl_of` returns
        # a pandas Series (it is `pos * ret`, and `ret` carries px's index), so
        # `pd.Series(<Series>, index=...)` makes pandas ALIGN on the source
        # index rather than REPLACE it. Today the index passed is that same
        # Series' own index, so the alignment is an identity and the values
        # are correct — which is exactly why it survived review. The moment the
        # two diverge it yields silent all-NaN, then `.fillna(0.0)` turns that
        # into a plausible-looking zero. P301 lost an entire six-year carry
        # column to this shape and left the guard that flags it.
        s = pd.Series(np.asarray(per[a][key]),
                      index=per[a]["_ts"]).reindex(idx).fillna(0.0)
        return s.to_numpy()
    hold_pf = np.mean([align(a, "_hold_pnl") for a in ASSETS], axis=0)
    ovl_pf = np.mean([align(a, "_ovl_pnl") for a in ASSETS], axis=0)
    hs = _series_stats(hold_pf, idx); os_ = _series_stats(ovl_pf, idx)
    dd_cut = round(1 - os_["maxdd_pct"] / hs["maxdd_pct"], 3) if hs["maxdd_pct"] > 0 else 0.0
    ret_keep = round(os_["total_pct"] / hs["total_pct"], 3) if hs["total_pct"] > 0 else 0.0
    res["portfolio"] = {"hold": {k: v for k, v in hs.items() if not k.startswith("_")},
                        "overlay": {k: v for k, v in os_.items() if not k.startswith("_")},
                        "dd_cut": dd_cut, "ret_keep": ret_keep,
                        "verdict": "VIABLE PRODUCT" if (dd_cut >= DD_CUT_MIN and ret_keep >= RET_KEEP_MIN) else "fails"}
    (REPO / "training" / "reports" / "overlay_backtest_p377.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8")

    W = 96
    print("=" * W)
    print("  DRAWDOWN-REDUCTION OVERLAY (long-or-flat SMA200 trend, net of honest CDE cost)")
    print("  PRODUCT BAR: cut maxDD >= %.0f%% while keeping >= %.0f%% of hold's return, full + >=2/3 eras"
          % (DD_CUT_MIN * 100, RET_KEEP_MIN * 100))
    print("=" * W)
    for a in ASSETS:
        r = res["assets"][a]
        print("\n%s (exposure %.0f%% of the time):" % (a, r["exposure"] * 100))
        print("  hold    : ret %+8.1f%%  Sharpe %+.2f  maxDD %.1f%%" % (r["hold"]["total_pct"], r["hold"]["sharpe"], r["hold"]["maxdd_pct"]))
        print("  overlay : ret %+8.1f%%  Sharpe %+.2f  maxDD %.1f%%   -> DD cut %.0f%%, ret kept %.0f%%   [%s]"
              % (r["overlay"]["total_pct"], r["overlay"]["sharpe"], r["overlay"]["maxdd_pct"],
                 r["dd_cut"] * 100, r["ret_keep"] * 100, r["verdict"]))
        for e, ev in r["eras"].items():
            print("    %-8s hold %+7.0f%% / dd %.0f%%  vs  overlay %+7.0f%% / dd %.0f%%  (cut %.0f%%, keep %.0f%%) %s"
                  % (e, ev["hold_ret"], ev["hold_dd"], ev["ovl_ret"], ev["ovl_dd"],
                     ev["dd_cut"] * 100, ev["ret_keep"] * 100, "OK" if ev["pass"] else "-"))
    p = res["portfolio"]
    print("\nEQUAL-WEIGHT 3-ASSET PORTFOLIO:")
    print("  hold    : ret %+8.1f%%  Sharpe %+.2f  maxDD %.1f%%" % (p["hold"]["total_pct"], p["hold"]["sharpe"], p["hold"]["maxdd_pct"]))
    print("  overlay : ret %+8.1f%%  Sharpe %+.2f  maxDD %.1f%%   -> DD cut %.0f%%, ret kept %.0f%%   [%s]"
          % (p["overlay"]["total_pct"], p["overlay"]["sharpe"], p["overlay"]["maxdd_pct"],
             p["dd_cut"] * 100, p["ret_keep"] * 100, p["verdict"]))
    print("\nreport -> training/reports/overlay_backtest_p377.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
