"""[P396] On-chain exchange-FLOW Rung-0 probe — answering "can our CURRENT API
fetch the on-chain data, before we buy deep history?"  ANSWER: YES for flow.

CoinGlass `/api/exchange/balance/chart` (our existing key) returns ~2 YEARS of
total-exchange-balance history WITH price, for BTC and ETH (SOL empty). So the
on-chain exchange-flow lead (P388 tier 3, thought to need paid Glassnode/
CryptoQuant) is probeable TODAY with no purchase and no 180-day wait — unlike
options put/call, whose history CoinGlass 404s (that one needs accumulation, P395/
P396 gated probe).

VERDICT (2026-08-24, pre-committed sign): net OUTFLOW (exchange balance falling =
coins leaving exchanges) = BULLISH; band 1.0; OOS = second half; honest CDE cost.
Result EARNS 1/2 — BTC +22.8% (Sh +0.40) vs hold -26.7%, but ETH -59.4% (loses,
109 trades). Weak/mixed, one ~1y OOS window, DAILY cadence (not the 4h book) —
consistent with the IC-0.04 ceiling, NOT a robust tradeable edge. So on-chain
flow costs nothing to fetch and does not clear; a PAID on-chain feed is unlikely
to change that at this scale (same fee floor). Recorded so nobody re-buys it.

Usage: reads cached bal_{ASSET}.json (from CoinGlass balance/chart) from
--cache-dir; or set COINGLASS_API_KEY to fetch live. Never raises to a caller.
"""
from __future__ import annotations
import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ASSETS = ("BTC", "ETH")  # CoinGlass balance/chart carries no SOL series
COST_RT = {"BTC": 27.7e-4, "ETH": 44.0e-4}
BAND = 1.0
_URL = "https://open-api-v4.coinglass.com/api/exchange/balance/chart?symbol={a}"


def _fetch(asset, cache_dir):
    cache = Path(cache_dir) / f"bal_{asset}.json" if cache_dir else None
    if cache and cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    key = os.environ.get("COINGLASS_API_KEY", "")
    if not key:
        return None
    req = urllib.request.Request(_URL.format(a=asset), headers={"CG-API-KEY": key})
    try:
        raw = urllib.request.urlopen(req, timeout=30).read()
    except Exception as e:
        print(f"  {asset}: fetch failed ({e})")
        return None
    d = json.loads(raw)
    if cache:
        cache.write_text(json.dumps(d), encoding="utf-8")
    return d


def _series(payload):
    d = (payload or {}).get("data", {})
    t = d.get("time_list", [])
    px = np.array(d.get("price_list", []), float)
    dm = d.get("data_map", {})
    n = len(t)
    tot = np.zeros(n)
    for _ex, s in dm.items():
        for i, v in enumerate(s or []):
            if v is not None and i < n:
                try:
                    tot[i] += float(v)
                except Exception:
                    pass
    return px, tot


def _z(x, w=30):
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        win = x[max(0, i - w):i]
        win = win[np.isfinite(win)]
        if len(win) >= 10 and win.std() > 0:
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
    return round(float(pnl.sum() * 100), 1), int(dpos.sum())


def main() -> int:
    cache_dir = None
    if "--cache-dir" in sys.argv:
        cache_dir = sys.argv[sys.argv.index("--cache-dir") + 1]
    print("[P396] EXCHANGE-FLOW Rung-0 (CoinGlass balance history, ~2y daily, BTC/ETH)")
    print("  pre-committed: net OUTFLOW = BULLISH; band 1.0; OOS 2nd half; honest cost")
    earns = measured = 0
    for a in ASSETS:
        p = _fetch(a, cache_dir)
        px, tot = _series(p) if p else (np.array([]), np.array([]))
        if len(px) < 200:
            print(f"  {a}: no/short data (need COINGLASS_API_KEY or --cache-dir)")
            continue
        measured += 1
        dbal = np.concatenate([[np.nan], np.diff(tot)])
        sig = -_z(dbal)                 # +sig = outflow = long
        mid = len(px) // 2
        sig[:mid] = np.nan             # OOS = second half
        net, tr = _hold(px, sig, COST_RT[a], BAND)
        bh = np.zeros(len(px))
        bh[mid:-1] = px[mid + 1:] / np.where(px[mid:-1] == 0, np.nan, px[mid:-1]) - 1.0
        bh_net = round(float(np.nan_to_num(bh).sum() * 100), 1)
        ok = net > 0 and net > bh_net
        earns += ok
        print(f"  {a}: held OOS net {net:+.1f}% trades {tr} | buy&hold {bh_net:+.1f}% -> {'EARNS' if ok else 'no'}")
    if measured == 0:
        print("  REFUSED — no data. Set COINGLASS_API_KEY or pass --cache-dir with bal_{ASSET}.json")
        return 2
    print(f"  VERDICT: {'EARNS' if earns >= 1 else 'NOT_EARNED'} ({earns}/{measured}) — "
          "on-chain flow via CURRENT API, no purchase needed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
