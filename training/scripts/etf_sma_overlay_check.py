"""[P404] Does ETF flow ADD to the certified SMA200 overlay, or overlap it?
2y history (not the thin forward window, so not the P388 leak trap). ETF = the
P402 lag-1 z-deadband signal; SMA200 = the certified long/flat overlay (P377).
Honest CDE per-leg cost (charged as dp*(pl/2.0)), OOS = 2nd half.

RESULT (2026-08-24): the two are COMPLEMENTARY, not overlapping (corr(SMA_pos,
ETF_pos) ~ -0.15 both assets). On BTC the de-risk stack (long only when SMA200
trends AND ETF is not signaling outflow) gives OOS Sharpe +1.49 / maxDD -9.7%
vs SMA200-alone +0.37 / -22.1% -- 4x Sharpe, half the drawdown. ETH: SMA200 is
the weak link, so ETF is better standalone there (asset-specific).
[P420] THOSE NUMBERS WERE LAG-2: `_stats` filled ret[:-1] and then used
ret[1:], so pos[t] earned the t+1 -> t+2 move. Fixed to lag-1 (ret[1:], the
etf_flow_probe convention); the corrected read is recorded in the P420 entry
and must be quoted beside the P404 figures, never instead of them.
CAVEATS: same recent-ETF-era window as P400 (regime-concentrated); n~300 OOS
(wide CI); the de-risk form was chosen after seeing 2 combos (the a-priori
sensible one). This strengthens the ARMING case; the live timing check
(etfflow_timing_check.py) + operator flip (P141) still gate a live seat.

Usage: python etf_sma_overlay_check.py [--cache-dir DIR with etf_btc.json/etf_eth.json]
"""
from __future__ import annotations
import json, os, sys, urllib.request
from pathlib import Path
import numpy as np

COST_RT = {"BTC": 27.7e-4, "ETH": 44.0e-4}
BAND = 1.0; LAG = 1
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
        print(f"  {asset}: fetch failed ({e})"); return None


def _z(x, w=30):
    o = np.full(len(x), np.nan)
    for i in range(len(x)):
        win = x[max(0, i - w):i]; win = win[np.isfinite(win)]
        if len(win) >= 15 and win.std() > 0:
            o[i] = np.clip((x[i] - win.mean()) / win.std(), -5, 5)
    return o


def _deadband(sig, band):
    pos = np.zeros(len(sig)); cur = 0.0
    for i in range(len(sig)):
        if not np.isfinite(sig[i]): pos[i] = cur; continue
        cur = 1.0 if sig[i] > band else (-1.0 if sig[i] < -band else cur); pos[i] = cur
    return pos


def _stats(px, pos, pl, mid):
    # [P420] ret[t] = the t-1 -> t return, so pos[t] * ret[t+1] earns the
    # t -> t+1 move: LAG-1, the alignment the docstring claims and the one
    # etf_flow_probe._hold uses. The pre-P420 line filled ret[:-1] (ret[t] =
    # the t -> t+1 move) and then multiplied by ret[t+1] -- pos[t] earned the
    # t+1 -> t+2 move, i.e. a lag-2 book: the P404 numbers were measured one
    # day late. Pinned by tests/test_p418_ops_labs_and_docs.py.
    ret = np.zeros(len(px)); ret[1:] = px[1:] / np.where(px[:-1] == 0, np.nan, px[:-1]) - 1
    ret = np.nan_to_num(ret); dp = np.abs(np.diff(pos, prepend=0.0))
    pnl = np.zeros(len(px)); pnl[:-1] = pos[:-1] * ret[1:]; pnl = pnl - dp * (pl / 2.0)
    p = pnl[mid:]; p = p[np.isfinite(p)]
    cum = np.cumsum(p); mdd = float((cum - np.maximum.accumulate(cum)).min()); sd = p.std()
    return (round(float(p.sum() * 100), 1),
            round(float(p.mean() / sd * np.sqrt(365.25)), 2) if sd > 0 else 0.0,
            round(mdd * 100, 1))


def main() -> int:
    cd = sys.argv[sys.argv.index("--cache-dir") + 1] if "--cache-dir" in sys.argv else None
    print("[P404] ETF flow vs / + certified SMA200 overlay (2y, OOS 2nd half, after CDE cost)")
    measured = 0
    for a in ("BTC", "ETH"):
        d = _load(a, cd)
        if not d:
            print(f"  {a}: no data (--cache-dir or COINGLASS_API_KEY)"); continue
        d = [x for x in d if x.get("price_usd")]
        if len(d) < 300:
            print(f"  {a}: {len(d)} days only"); continue
        measured += 1
        px = np.array([float(x["price_usd"]) for x in d])
        flow = np.array([float(x.get("flow_usd", 0) or 0) for x in d])
        n = len(px); mid = n // 2; pl = COST_RT[a]
        raw = np.concatenate([[np.nan] * LAG, flow[:n - LAG]])
        etf = _deadband(_z(raw), BAND)
        sma = np.array([np.mean(px[max(0, i - 200):i + 1]) for i in range(n)])
        smapos = (px > sma).astype(float)
        combo = np.where(etf < 0, 0.0, smapos)   # ETF outflow forces flat (de-risk)
        avg = 0.5 * (smapos + etf)
        corr = float(np.corrcoef(smapos[mid:], etf[mid:])[0, 1])
        print(f"=== {a} (OOS n={n - mid}) ===")
        for name, pos in (("SMA200 long/flat (certified)", smapos),
                          ("ETF lag1 z-deadband (P402)", etf),
                          ("SMA200 + ETF-outflow de-risk", combo),
                          ("0.5*(SMA+ETF)", avg)):
            net, sh, mdd = _stats(px, pos, pl, mid)
            print(f"  {name:32s}: net {net:+6.1f}%  Sharpe {sh:+.2f}  maxDD {mdd:+.1f}%")
        print(f"  corr(SMA_pos, ETF_pos) OOS = {corr:+.2f} (~0 complementary, ~1 overlaps)")
    if measured == 0:
        print("  REFUSED — no data"); return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
