"""[P397] On-chain Rung-0 on FREE CoinMetrics community data — settles the on-chain
lead without buying Glassnode/CryptoQuant.

The CoinMetrics community repo (github.com/coinmetrics/data, CC BY-NC, CSV per coin)
carries ~17 years of MEASURED on-chain metrics free — including exchange flows
(FlowInExNtv/FlowOutExNtv) and valuation (CapMVRVCur), the two on-chain
direction leads (P388 tier 3). This is measured netflow, cleaner than the
CoinGlass balance-derived proxy (P396, weak 1/2).

VERDICT (2026-08-24, pre-committed signs): net exchange OUTFLOW = bullish
[z(FlowOut-FlowIn)]; MVRV contrarian = -z(MVRV). Hold-aware, band 1.0, OOS =
second half (BTC from 2018-11, ETH from 2020-12), honest CDE cost. Result
NOT_EARNED 0/4 — BTC netflow +216% but LOSES to hold +398% (Sh 0.32, 533 trades);
BTC MVRV -432%; ETH netflow +14% vs +265%; ETH MVRV -131%. Both on-chain leads
are dead on 5-8y OOS. So NO on-chain purchase is justified — a paid feed serves a
higher-resolution version of signals that already fail at this scale/fee floor.
Recorded so nobody buys on-chain data.

Usage: reads cached cm_{btc,eth}.csv from --cache-dir, or fetches free from the
CoinMetrics raw GitHub. Never raises to a caller.
"""
from __future__ import annotations
import csv
import sys
import urllib.request
from pathlib import Path

import numpy as np

COST_RT = {"BTC": 27.7e-4, "ETH": 44.0e-4}
BAND = 1.0
_URL = "https://raw.githubusercontent.com/coinmetrics/data/master/csv/{c}.csv"


def _load(asset, cache_dir):
    c = asset.lower()
    if cache_dir:
        p = Path(cache_dir) / f"cm_{c}.csv"
        if p.exists():
            with p.open(encoding="utf-8") as fh:
                return list(csv.DictReader(fh))
    try:
        raw = urllib.request.urlopen(_URL.format(c=c), timeout=60).read().decode("utf-8")
    except Exception as e:
        print(f"  {asset}: fetch failed ({e})")
        return None
    return list(csv.DictReader(raw.splitlines()))


def _col(rows, name):
    return np.array([float(x[name]) if x.get(name) not in (None, "", "NaN") else np.nan
                     for x in rows])


def _z(x, w=90):
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        win = x[max(0, i - w):i]
        win = win[np.isfinite(win)]
        if len(win) >= 30 and win.std() > 0:
            out[i] = np.clip((x[i] - win.mean()) / win.std(), -5, 5)
    return out


def _hold(px, sig, per_leg, band):
    n = len(px)
    ret = np.zeros(n)
    ret[1:] = px[1:] / np.where(px[:-1] == 0, np.nan, px[:-1]) - 1.0
    ret = np.nan_to_num(ret)
    pos = np.zeros(n)
    cur = 0.0
    for i in range(n):
        if not np.isfinite(sig[i]):
            pos[i] = cur
            continue
        if sig[i] > band:
            cur = 1.0
        elif sig[i] < -band:
            cur = -1.0
        pos[i] = cur
    dpos = np.abs(np.diff(pos, prepend=0.0))
    pnl = np.zeros(n)
    pnl[:-1] = pos[:-1] * ret[1:]
    pnl = pnl - dpos * (per_leg / 2.0)
    pnl = pnl[np.isfinite(pnl)]
    sd = pnl.std()
    return (round(float(pnl.sum() * 100), 1),
            round(float(pnl.mean() / sd * np.sqrt(365.25)), 2) if sd > 0 else 0.0,
            int(dpos.sum()))


def main() -> int:
    cache_dir = None
    if "--cache-dir" in sys.argv:
        cache_dir = sys.argv[sys.argv.index("--cache-dir") + 1]
    print("[P397] CoinMetrics on-chain hold-aware Rung-0 (free, daily, ~17y)")
    print("  pre-committed: net OUTFLOW bullish = z(FlowOut-FlowIn); MVRV contrarian = -z(MVRV); band 1.0; OOS 2nd half")
    earns = measured = 0
    for a in ("BTC", "ETH"):
        rows = _load(a, cache_dir)
        if not rows:
            print(f"  {a}: no data (--cache-dir or network)")
            continue
        px = _col(rows, "PriceUSD")
        fi = _col(rows, "FlowInExNtv")
        fo = _col(rows, "FlowOutExNtv")
        mv = _col(rows, "CapMVRVCur")
        mask = np.isfinite(px) & np.isfinite(fi) & np.isfinite(fo)
        idx = np.where(mask)[0]
        if len(idx) < 400:
            print(f"  {a}: insufficient flow coverage ({len(idx)})")
            continue
        s0 = idx[0]
        px, fi, fo, mv = px[s0:], fi[s0:], fo[s0:], mv[s0:]
        n = len(px)
        mid = n // 2
        bh = np.zeros(n)
        bh[mid:-1] = px[mid + 1:] / np.where(px[mid:-1] == 0, np.nan, px[mid:-1]) - 1.0
        bh_net = round(float(np.nan_to_num(bh).sum() * 100), 1)
        measured += 1
        for label, raw_sig, sign in (("netflow", fo - fi, +1), ("MVRV", mv, -1)):
            sig = sign * _z(raw_sig)
            sig[:mid] = np.nan
            net, sh, tr = _hold(px, sig, COST_RT[a], BAND)
            ok = net > 0 and net > bh_net
            earns += ok
            print(f"  {a} {label:8s}: net {net:+8.1f}% Sh {sh:+.2f} trades {tr} | buy&hold {bh_net:+8.1f}% -> {'EARNS' if ok else 'no'}")
    if measured == 0:
        print("  REFUSED — no data")
        return 2
    print(f"  VERDICT: {'SOME EARN' if earns else 'NOT_EARNED'} ({earns}/{measured * 2}) — "
          "on-chain via FREE CoinMetrics; no purchase justified if 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
