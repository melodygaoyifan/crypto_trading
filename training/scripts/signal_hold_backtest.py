"""[P386] The right way to monetize a weak-but-persistent signal: HOLD the position
while the signal persists, don't round-trip every 4h — and use the perp's revenue
levers (short the downs, collect/pay funding).

WHY (operator's insight, correcting P385's framing). The edge probe charged cost
PER BAR as if we buy this 4h and sell next 4h — the worst-case execution. A signal
with IC~0.04 should instead drive a POSITION that is HELD while the sign persists,
paying the fee only when direction FLIPS. That is exactly why the SMA200 rule is
net-positive (it trades ~6-13x/yr, not every bar). This backtest asks: does the ML
signal, expressed as a hold-position on a DERIVATIVE (long/short + funding), net
positive after honest cost given its ACTUAL turnover?

METHOD. Walk-forward ridge (same features/window as edge_probe) predicts the 16h
forward return. Position = deadband on the standardized prediction: long if z>+b,
short if z<-b, else HOLD the previous position. Cost = |Δposition| x (RT_bps/2)
charged only on changes. Funding carry: long pays when funding>0, short collects
(daily rate from coinglass, /6 per 4h bar, P245 convention). Sweep the deadband:
a wider band => fewer flips => less fee, at the cost of slower reaction. Compare
net/Sharpe/turnover/maxDD to buy_and_hold and to the sign-EVERY-bar baseline.

Honest scope: this is a backtest on ~6y (the multiply-read window, P260 discount);
a pass here is a candidate for a forward shadow, never a live flip. A FAIL settles
whether holding rescues the weak signal at all.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DRL = REPO / "training" / "training_data" / "drl_training"
CG = REPO / "training" / "training_data" / "coinglass_history"
ASSETS = ("BTC", "ETH", "SOL")
COST_RT = {"BTC": 27.7, "ETH": 44.0, "SOL": 41.0}   # bps
PER_LEG = {a: COST_RT[a] / 2.0 / 1e4 for a in ASSETS}
NON_FEAT = {"timestamp", "open", "high", "low", "close", "volume", "vwap",
            "date", "asset", "symbol"}


def load(asset):
    d = pd.read_parquet(DRL / f"{asset}_4H_full.parquet")
    if "timestamp" in d.columns:
        d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
        d = d.set_index("timestamp")
    feats = [c for c in d.columns
             if c not in NON_FEAT and pd.api.types.is_numeric_dtype(d[c])
             and not c.startswith("fwd") and not c.startswith("target")]
    close = d["close"].to_numpy(float)
    X = d[feats].to_numpy(float)
    # funding: daily -> per-4h, forward-filled onto the bar index
    fund = np.zeros(len(d))
    fp = CG / f"{asset}_funding_1d.parquet"
    if fp.exists():
        fd = pd.read_parquet(fp)
        tcol = "timestamp" if "timestamp" in fd.columns else fd.columns[0]
        rcol = next((c for c in fd.columns if "fund" in c.lower() and "rate" in c.lower()),
                    next((c for c in fd.columns if "fund" in c.lower()), None))
        if rcol is not None:
            fd[tcol] = pd.to_datetime(fd[tcol], utc=True)
            s = fd.set_index(tcol)[rcol].astype(float).reindex(d.index, method="ffill")
            fund = (s / 6.0).fillna(0.0).to_numpy()   # daily rate spread over 6 4h-bars
    return d.index, close, X, fund


def walk_forward(X, y, min_train=7200, refit=250):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    n = len(X); preds = np.full(n, np.nan); s = min_train
    while s + 3 < n:
        te = min(s + refit, n)
        m = ~(np.isnan(X[:s]).any(axis=1) | np.isnan(y[:s]))
        if m.sum() >= 500:
            sc = StandardScaler().fit(X[:s][m])
            mdl = Ridge(alpha=10.0).fit(sc.transform(X[:s][m]), y[:s][m])
            preds[s + 3:te] = mdl.predict(sc.transform(np.nan_to_num(X[s + 3:te])))
        s = te
    return preds


def _stats(pnl):
    pnl = pnl[~np.isnan(pnl)]
    if len(pnl) < 50:
        return {"net_pct": 0.0, "sharpe": 0.0, "maxdd_pct": 0.0}
    eq = np.cumprod(1 + pnl); peak = np.maximum.accumulate(eq)
    dd = ((peak - eq) / peak).max()
    sd = pnl.std()
    return {"net_pct": round(float(pnl.sum() * 100), 1),
            "sharpe": round(float(pnl.mean() / sd * np.sqrt(6 * 365.25)), 2) if sd > 0 else 0.0,
            "maxdd_pct": round(float(dd * 100), 1)}


def simulate(close, preds, fund, per_leg, deadband, allow_short=True):
    n = len(close)
    ret = np.zeros(n); ret[1:] = close[1:] / close[:-1] - 1.0
    # standardize prediction to a trailing z (causal)
    ps = pd.Series(preds)
    z = (ps - ps.rolling(500, min_periods=100).mean()) / ps.rolling(500, min_periods=100).std()
    z = z.to_numpy()
    pos = np.zeros(n); cur = 0.0
    for i in range(n):
        if np.isnan(z[i]):
            pos[i] = cur; continue
        if z[i] > deadband:
            cur = 1.0
        elif z[i] < -deadband:
            cur = -1.0 if allow_short else 0.0
        # else: HOLD cur (the whole point)
        pos[i] = cur
    dpos = np.abs(np.diff(pos, prepend=0.0))
    price_pnl = pos * ret                       # position earns next-bar return
    # shift: position decided at i acts over [i, i+1); align so pos[i]*ret[i+1]
    price_pnl = np.zeros(n); price_pnl[:-1] = pos[:-1] * ret[1:]
    cost = dpos * per_leg
    fund_pnl = -pos * fund                       # long pays positive funding
    total = price_pnl - cost + fund_pnl
    trades_per_yr = float(dpos.sum() / (n / (6 * 365.25)))
    st = _stats(total)
    st.update({"trades_per_yr": round(trades_per_yr, 1),
               "gross_price_pct": round(float(price_pnl.sum() * 100), 1),
               "cost_pct": round(float(cost.sum() * 100), 1),
               "funding_pct": round(float(fund_pnl.sum() * 100), 1),
               "exposure": round(float(np.abs(pos).mean()), 3)})
    return st


def main():
    res = {"cost_rt_bps": COST_RT, "assets": {}}
    W = 100
    print("=" * W)
    print("  SIGNAL HOLD-POSITION BACKTEST — hold while signal persists, long/short perp + funding")
    print("  (does the IC~0.04 signal net positive after cost when HELD, not round-tripped every 4h?)")
    print("=" * W)
    for a in ASSETS:
        idx, close, X, fund = load(a)
        n = len(close)
        fwd16 = np.full(n, np.nan); fwd16[:n - 4] = close[4:] / close[:n - 4] - 1.0
        preds = walk_forward(X, fwd16)
        # buy and hold over the OOS region (where preds exist)
        oos = ~np.isnan(preds)
        first = int(np.argmax(oos)) if oos.any() else 0
        bh = np.zeros(n); bh[first:-1] = close[first + 1:] / close[first:-1] - 1.0
        bh_st = _stats(bh)
        out = {"buy_and_hold": bh_st, "sweep": {}}
        print(f"\n{a}  (OOS from {str(idx[first])[:10]}, {int(oos.sum())} bars):")
        print(f"  buy_and_hold: net {bh_st['net_pct']:+.0f}%  Sharpe {bh_st['sharpe']:+.2f}  maxDD {bh_st['maxdd_pct']:.0f}%")
        for b in (0.0, 0.5, 1.0, 1.5, 2.0):
            st = simulate(close, preds, fund, PER_LEG[a], deadband=b, allow_short=True)
            out["sweep"][f"deadband_{b}"] = st
            print(f"  band {b:>3}: net {st['net_pct']:+7.0f}%  Sh {st['sharpe']:+.2f}  "
                  f"maxDD {st['maxdd_pct']:>3.0f}%  trades/yr {st['trades_per_yr']:>5.1f}  "
                  f"gross {st['gross_price_pct']:+.0f} cost {st['cost_pct']:.0f} fund {st['funding_pct']:+.0f}  "
                  f"expo {st['exposure']}")
        res["assets"][a] = out
    (REPO / "training" / "reports" / "signal_hold_backtest_p386.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8")
    print("\n  report -> training/reports/signal_hold_backtest_p386.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
