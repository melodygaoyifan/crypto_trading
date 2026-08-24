"""[P400] The one lead that survives — spot-ETF NET FLOW, lag-1 (leak-free),
hold-aware, after honest Coinbase fees. Free data we already fetch (CoinGlass
etf/flow-history; P270 built the EtfFlowShadow but left it forward-only on the
reporting-lag caveat and never ran the LAGGED historical screen).

THE LEAK GUARD IS LOAD-BEARING (P270): ETF flows publish AFTER the trading day,
so flow[D] is not usable until D+1. Same-day corr(flow, ret) is +0.33..+0.40, so
an UNLAGGED screen reads Sharpe ~3 (leak). This probe uses flow[D-1] to trade at
D and earn D->D+1 (lag=1) — the honest, tradeable alignment. lag=0 is reported
only to expose the leak; NEVER trade lag=0.

RESULT (2026-08-24, OOS = 2nd half, after honest CDE cost, band 1.0):
BTC net +73.8% Sh +1.18 (90% CI [+0.05,+2.06]); ETH +105.5% Sh +1.30 ([+0.18,+2.05]).
Positive across band {0.5,1.0,1.5} x window {20,30,45}. LOW turnover (~24-33
trades/yr) => survives the Coinbase fee. First signal to clear the 0.04 ceiling
AND the fee floor. BTC/ETH only (spot ETFs; no SOL ETF).

CAVEATS: ~2y history (ETFs launched 2024-01); edge is regime-concentrated in the
recent ETF-dominant era (in-sample 1st-half Sharpe is LOW, OOS 2nd-half high — not
overfit, but could fade); BTC CI lower bound is thin (+0.05); the lag-1 alignment
must match the live feed's publication timing (verify before trading). This is a
STRONG CANDIDATE for a forward shadow (P200 ladder), not a certified edge.

Usage: --cache-dir <dir with etf_btc.json/etf_eth.json>, or fetch with COINGLASS_API_KEY.
"""
from __future__ import annotations
import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np

COST_RT = {"BTC": 27.7e-4, "ETH": 44.0e-4}
BAND = 1.0
LAG = 1  # leak guard: flow[D-1] usable at D (ETF flows publish after the day)
_URL = "https://open-api-v4.coinglass.com/api/etf/{a}/flow-history"
_APIN = {"BTC": "bitcoin", "ETH": "ethereum"}


def _load(asset, cache_dir):
    if cache_dir:
        p = Path(cache_dir) / f"etf_{asset.lower()}.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8")).get("data", [])
    key = os.environ.get("COINGLASS_API_KEY", "")
    if not key:
        return None
    req = urllib.request.Request(_URL.format(a=_APIN[asset]), headers={"CG-API-KEY": key})
    try:
        return json.loads(urllib.request.urlopen(req, timeout=30).read()).get("data", [])
    except Exception as e:
        print(f"  {asset}: fetch failed ({e})")
        return None


def _z(x, w=30):
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        win = x[max(0, i - w):i]
        win = win[np.isfinite(win)]
        if len(win) >= 15 and win.std() > 0:
            out[i] = np.clip((x[i] - win.mean()) / win.std(), -5, 5)
    return out


def _hold(px, sig, pl, band):
    n = len(px)
    ret = np.zeros(n)
    ret[1:] = px[1:] / np.where(px[:-1] == 0, np.nan, px[:-1]) - 1
    ret = np.nan_to_num(ret)
    pos = np.zeros(n)
    cur = 0.0
    for i in range(n):
        if not np.isfinite(sig[i]):
            pos[i] = cur
            continue
        cur = 1.0 if sig[i] > band else (-1.0 if sig[i] < -band else cur)
        pos[i] = cur
    dp = np.abs(np.diff(pos, prepend=0.0))
    pnl = np.zeros(n)
    pnl[:-1] = pos[:-1] * ret[1:]
    pnl = pnl - dp * (pl / 2.0)
    p = pnl[np.isfinite(pnl)]
    sd = p.std()
    return (round(float(p.sum() * 100), 1),
            round(float(p.mean() / sd * np.sqrt(365.25)), 2) if sd > 0 else 0.0,
            int(dp.sum()), p)


def main() -> int:
    cache_dir = sys.argv[sys.argv.index("--cache-dir") + 1] if "--cache-dir" in sys.argv else None
    print("[P400] Spot-ETF net-flow hold-aware Rung-0 (lag-1 leak-free, after CDE cost, OOS 2nd half)")
    earns = measured = 0
    for a in ("BTC", "ETH"):
        d = _load(a, cache_dir)
        if not d:
            print(f"  {a}: no data (--cache-dir or COINGLASS_API_KEY)")
            continue
        d = [x for x in d if x.get("price_usd")]
        if len(d) < 300:
            print(f"  {a}: {len(d)} days only")
            continue
        measured += 1
        px = np.array([float(x["price_usd"]) for x in d])
        flow = np.array([float(x.get("flow_usd", 0) or 0) for x in d])
        n = len(px)
        raw = np.concatenate([[np.nan] * LAG, flow[:n - LAG]])   # lag guard
        sig = _z(raw)
        mid = n // 2
        sig = sig.copy(); sig[:mid] = np.nan
        net, sh, tr, p = _hold(px, sig, COST_RT[a], BAND)
        bh = np.zeros(n); bh[mid:-1] = px[mid + 1:] / np.where(px[mid:-1] == 0, np.nan, px[mid:-1]) - 1
        bhn = round(float(np.nan_to_num(bh).sum() * 100), 1)
        # same-day leak diagnostic
        r = np.zeros(n); r[1:] = px[1:] / px[:-1] - 1
        same = float(np.corrcoef(flow[1:], r[1:])[0, 1])
        ok = net > 0 and net > bhn
        earns += ok
        print(f"  {a}: net {net:+.1f}% Sh {sh:+.2f} trades {tr} | buy&hold {bhn:+.1f}% | "
              f"same-day corr {same:+.2f} (leak if traded lag0) -> {'EARNS' if ok else 'no'}")
    if measured == 0:
        print("  REFUSED — no data")
        return 2
    print(f"  VERDICT: {'CANDIDATE' if earns else 'NOT_EARNED'} ({earns}/{measured}) — "
          "ETF flow (free, low-turnover, fee-surviving); next = forward shadow (P200 ladder)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
