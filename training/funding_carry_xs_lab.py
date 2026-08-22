"""[P374] Cross-sectional funding carry across the 8 assets — the last un-run
offline candidate (found by the P373-era research audit).

WHY. Every other on-disk data source maps to a consumed signal or a run
backtest. The one exception the audit surfaced: 6y of daily funding for all 8
assets (training/training_data/coinglass_history/*_funding_1d.parquet) has been
used only single-asset-directionally (P297 funding legs) or as a carry-cost
accountant (P301 breadth) — never RANKED cross-sectionally to harvest the
funding spread. This closes it, the same way P373 closed xsmom.

THE PREMISE (measured read-only in the audit): rank assets by funding; low/
negative-funding assets get PAID to be held long and tend to outperform. Full
sample Spearman(-funding, next-ret) t=+3.76; but era-split shows +0.80/+1.22
Sharpe in 2020-23 and -0.29 (ann -12.9%) in 2024-26 — the P243/P244 carry
inversion. This lab prices it net of cost + carry to settle it.

THE BAR, pre-committed before the first number (identical to P373's, the bar
every candidate faced): the strategy EARNS iff its net-of-cost-AND-carry total
return beats EQUAL-WEIGHT buy-and-hold of the 8-asset universe, positive in
>=2 of 3 eras. Sharpe/drawdown reported, not deciding.

CONVENTIONS: causal (previous COMPLETED day's funding decides today's book,
the P247-F1 no-in-progress rule); dollar-neutral long-bottom-k/short-top-k plus
a long-only-bottom-k variant for the routable read; funding modelled at 3
intervals/day (standard 8h perp funding; a long collects -funding); honest CDE
round-trip cost per rebalance (measured BTC 27.7/ETH 44.0/SOL 41.0 bps, 45bps
assumed for the 5 unmeasured alts, P167); additive per-bar return sums.
CAVEAT: an 8-asset book is a SIGNAL test — the sleeve routes only BTC/ETH/SOL
(P292); the 5 alts have no live perp routing and thin volume.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "training" / "training_data" / "raw"
FUND = REPO / "training" / "training_data" / "coinglass_history"
UNIVERSE = ("BTC","ETH","SOL","BNB","XRP","ADA","DOGE","LTC")
COST_RT = {"BTC":27.7e-4,"ETH":44.0e-4,"SOL":41.0e-4,
           "BNB":45e-4,"XRP":45e-4,"ADA":45e-4,"DOGE":45e-4,"LTC":45e-4}
ERAS = {"2020-22":("2020-01-01","2023-01-01"),
        "2023-24":("2023-01-01","2025-01-01"),
        "2025-26":("2025-01-01","2027-01-01")}
FUND_INTERVALS_PER_DAY = 3   # standard 8h perp funding


def load_daily():
    """Daily close panel + daily funding panel, aligned on common UTC days."""
    px, fz = {}, {}
    for a in UNIVERSE:
        d = pd.read_parquet(RAW / f"{a}_60m.parquet")[["timestamp","close"]]
        d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
        d = d.set_index("timestamp")["close"].resample("1D").last()
        px[a] = d
        f = pd.read_parquet(FUND / f"{a}_funding_1d.parquet")[["timestamp","funding_close"]]
        f["timestamp"] = pd.to_datetime(f["timestamp"], utc=True)
        fz[a] = f.set_index("timestamp")["funding_close"]
    P = pd.DataFrame(px).sort_index().ffill()
    F = pd.DataFrame(fz).sort_index().ffill()
    idx = P.index.intersection(F.index)
    P, F = P.loc[idx], F.loc[idx]
    good = P.notna().all(axis=1) & F.notna().all(axis=1)
    return P.loc[good], F.loc[good]


def _stats(pnl, ts):
    eq = np.cumprod(1 + pnl)
    dd = (np.maximum.accumulate(eq) - eq) / np.maximum.accumulate(eq)
    sd = pnl.std()
    out = {"total_pct": round(float(pnl.sum()*100),1),
           "sharpe": round(float(pnl.mean()/sd*np.sqrt(365.25)),2) if sd>0 else 0.0,
           "maxdd_pct": round(float(dd.max()*100),1), "eras": {}}
    for name,(a,b) in ERAS.items():
        m = np.asarray((ts>=a)&(ts<b))
        if m.sum() > 30:
            out["eras"][name] = round(float(pnl[m].sum()*100),1)
    return out


def bnh(P):
    ret = np.vstack([np.zeros((1,P.shape[1])), P.to_numpy()[1:]/P.to_numpy()[:-1]-1.0])
    return _stats(ret @ np.full(P.shape[1],1.0/P.shape[1]), P.index)


def backtest(P, F, *, k, long_only):
    px = P.to_numpy(float); fund = F.to_numpy(float); ts = P.index
    n, m = px.shape
    ret = np.vstack([np.zeros((1,m)), px[1:]/px[:-1]-1.0])
    costs = np.array([COST_RT[a] for a in UNIVERSE])
    pos = np.zeros(m); pnl = np.zeros(n)
    for i in range(2, n):
        # price PnL of yesterday's book
        pnl[i] = float(np.dot(pos, ret[i]))
        # funding carry: a long collects -funding, 3 intervals/day
        pnl[i] += float(np.dot(pos, -fund[i] * FUND_INTERVALS_PER_DAY))
        # rebalance daily on the PREVIOUS completed day's funding (causal)
        rank = fund[i-1]
        order = np.argsort(rank)          # ascending: lowest funding first
        new = np.zeros(m)
        new[order[:k]] = 1.0/k            # long the lowest funders (collect carry)
        if not long_only:
            new[order[-k:]] = -1.0/k      # short the highest funders
        pnl[i] -= float(np.dot(np.abs(new-pos), costs))
        pos = new
    return _stats(pnl, ts)


def main():
    P, F = load_daily()
    yrs = round(len(P)/365.25, 2)
    bh = bnh(P)
    res = {"years": yrs, "buy_and_hold_eqw": bh, "variants": {}}
    for k in (2,3):
        for lo in (True, False):
            res["variants"][f"bottom{k}_{'long_only' if lo else 'dollar_neutral'}"] = backtest(P,F,k=k,long_only=lo)
    (REPO/"training"/"reports"/"funding_carry_xs_p374.json").write_text(json.dumps(res,indent=2),encoding="utf-8")
    W=84; print("="*W)
    print("  CROSS-SECTIONAL FUNDING CARRY (8 assets) — net of cost AND carry vs eqw B&H")
    print("  BAR (pre-committed): beat eqw B&H on TOTAL RETURN, positive in >=2 of 3 eras")
    print("="*W)
    print(f"\n8-asset ({yrs}y)   [signal test; only BTC/ETH/SOL routable, P292]")
    print(f"  {'variant':26s} {'net %':>9s} {'Sharpe':>7s} {'maxDD%':>8s}   eras")
    print(f"  {'buy_and_hold (eqw)':26s} {bh['total_pct']:+9.1f} {bh['sharpe']:>7.2f} {bh['maxdd_pct']:>8.1f}   "
          + " ".join(f"{v:+.0f}" for v in bh['eras'].values()))
    for tag,s in res["variants"].items():
        beats = s["total_pct"] > bh["total_pct"]
        ep = sum(1 for v in s["eras"].values() if v>0)
        verdict = "EARNS" if (beats and ep>=2) else "fails"
        print(f"  {tag:26s} {s['total_pct']:+9.1f} {s['sharpe']:>7.2f} {s['maxdd_pct']:>8.1f}   "
              + " ".join(f"{v:+.0f}" for v in s['eras'].values())
              + f"   [{verdict}: vs B&H {'+' if beats else '-'}, {ep}/3 eras+]")
    print("\nreport -> training/reports/funding_carry_xs_p374.json")
    return 0

if __name__ == "__main__":
    sys.exit(main())
