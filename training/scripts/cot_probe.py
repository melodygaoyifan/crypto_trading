"""[P402] CFTC Commitment-of-Traders hold-aware Rung-0 — the last untested
'better source' candidate, screened after the ETF-flow signal was wired.

WHY: the winning profile is low-turnover, non-price, institutional-positioning
(ETF net flow: OOS Sh 1.18 BTC / 1.30 ETH after honest CDE cost, P400/P402).
CFTC COT is the same family and free, with DEEPER history than ETF flow's ~2yr
(CME BITCOIN ~8yr / ETHER CASH SETTLED ~5yr in the Traders-in-Financial-Futures
report, resource gpe5-46if) — which is exactly the OOS depth P400 wished for.

LEAK GUARD (load-bearing): COT is a Tuesday snapshot released Friday ~15:30 ET,
so each weekly reading is gated to report_date + PUB_LAG_DAYS (4 -> Sat) before
it can trade. Pre-committed: net = long - short per cohort, z-scored vs trailing
weeks, sign = +z (institutional net-long buildup -> bullish, same polarity as the
ETF finding); deadband band 1.0; OOS = 2nd half; honest CDE cost charged per-leg
as dp*(pl/2.0) so a flip pays one round-trip. Two cohorts: asset_mgr (asset
managers, institutional) and lev_money (leveraged funds, hedge funds).

RESULT (2026-08-24): NOT_EARNED. BTC asset_mgr Sh +0.16 / lev_money -0.26 (437
weeks); ETH asset_mgr -0.48 / lev_money -0.38 (281 weeks) — every cohort loses to
buy-and-hold and sits far below the wired ETF signal. Deep history makes this a
strong NO, not a thin-window artifact (P348). ETF net flow remains best; the
data-source search is exhausted (P397 on-chain / P398 options / P401 CoinAPI /
this COT).

Usage: python cot_probe.py [--cache-dir DIR with cm_btc.csv/cm_eth.csv]
       (price falls back to Binance daily klines, free/keyless).
"""
from __future__ import annotations
import urllib.request, urllib.parse, json, csv, sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
import numpy as np

COST_RT = {"BTC": 27.7e-4, "ETH": 44.0e-4}   # per-leg charged as dp * (pl / 2.0)
BAND = 1.0
PUB_LAG_DAYS = 4
RID = "gpe5-46if"
MKT = {"BTC": "BITCOIN - CHICAGO MERCANTILE EXCHANGE",
       "ETH": "ETHER CASH SETTLED - CHICAGO MERCANTILE EXCHANGE"}
_BN = {"BTC": "BTCUSDT", "ETH": "ETHUSDT"}


def _cot(asset):
    base = f"https://publicreporting.cftc.gov/resource/{RID}.json"
    q = {"$where": f"market_and_exchange_names = '{MKT[asset]}'", "$limit": "2000",
         "$order": "report_date_as_yyyy_mm_dd ASC"}
    return json.loads(urllib.request.urlopen(base + "?" + urllib.parse.urlencode(q), timeout=60).read())


def _price(asset, cache_dir):
    if cache_dir:
        f = Path(cache_dir) / ("cm_btc.csv" if asset == "BTC" else "cm_eth.csv")
        if f.exists():
            out = {}
            for x in csv.DictReader(f.open(encoding="utf-8")):
                v = x.get("PriceUSD")
                if v not in (None, "", "NaN"):
                    out[x["time"][:10]] = float(v)
            return out
    # Binance daily klines, keyless, paginated back to 2017
    out, start = {}, 1483228800000
    for _ in range(40):
        url = (f"https://api.binance.com/api/v3/klines?symbol={_BN[asset]}"
               f"&interval=1d&startTime={start}&limit=1000")
        try:
            d = json.loads(urllib.request.urlopen(url, timeout=40).read())
        except Exception:
            break
        if not d:
            break
        for k in d:
            out[datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc).date().isoformat()] = float(k[4])
        start = d[-1][0] + 86400000
        if len(d) < 1000:
            break
    return out


def _z(x, w=30):
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        win = x[max(0, i - w):i]; win = win[np.isfinite(win)]
        if len(win) >= 15 and win.std() > 0:
            out[i] = np.clip((x[i] - win.mean()) / win.std(), -5, 5)
    return out


def _hold(px, sig, pl, band=BAND):
    n = len(px); ret = np.zeros(n); ret[1:] = px[1:] / np.where(px[:-1] == 0, np.nan, px[:-1]) - 1
    ret = np.nan_to_num(ret); pos = np.zeros(n); cur = 0.0
    for i in range(n):
        if not np.isfinite(sig[i]):
            pos[i] = cur; continue
        cur = 1.0 if sig[i] > band else (-1.0 if sig[i] < -band else cur); pos[i] = cur
    dp = np.abs(np.diff(pos, prepend=0.0)); pnl = np.zeros(n); pnl[:-1] = pos[:-1] * ret[1:]
    pnl = (pnl - dp * (pl / 2.0)); p = pnl[np.isfinite(pnl)]; sd = p.std()
    return (round(float(p.sum() * 100), 1),
            round(float(p.mean() / sd * np.sqrt(365.25)), 2) if sd > 0 else 0.0, int(dp.sum()))


def main() -> int:
    cache = sys.argv[sys.argv.index("--cache-dir") + 1] if "--cache-dir" in sys.argv else None
    print("[COT] CFTC TFF hold-aware Rung-0 (leak-free, OOS 2nd half, after CDE cost)")
    beats_etf = measured = 0
    for a in ("BTC", "ETH"):
        try:
            d = _cot(a)
        except Exception as e:
            print(f"  {a}: COT fetch failed ({e})"); continue
        px = _price(a, cache)
        if not px:
            print(f"  {a}: no price"); continue
        weeks = []
        for r in d:
            rd = r.get("report_date_as_yyyy_mm_dd")
            if not rd:
                continue
            try:
                am = float(r.get("asset_mgr_positions_long") or 0) - float(r.get("asset_mgr_positions_short") or 0)
                lm = float(r.get("lev_money_positions_long") or 0) - float(r.get("lev_money_positions_short") or 0)
            except Exception:
                continue
            avail = (datetime.fromisoformat(rd.replace("Z", "")).replace(tzinfo=timezone.utc)
                     + timedelta(days=PUB_LAG_DAYS)).date().isoformat()
            weeks.append((avail, am, lm))
        if len(weeks) < 80:
            print(f"  {a}: only {len(weeks)} weeks"); continue
        measured += 1
        avails = [w[0] for w in weeks]
        for cohort, idx in (("asset_mgr", 1), ("lev_money", 2)):
            wz = _z(np.array([w[idx] for w in weeks], dtype=float))
            days = sorted(dd for dd in px if dd >= avails[0])
            sig = np.full(len(days), np.nan); wi = 0
            for j, dd in enumerate(days):
                while wi + 1 < len(avails) and avails[wi + 1] <= dd:
                    wi += 1
                sig[j] = wz[wi]
            pxa = np.array([px[dd] for dd in days]); mid = len(days) // 2
            s2 = sig.copy(); s2[:mid] = np.nan
            net, sh, tr = _hold(pxa, s2, COST_RT[a])
            bh = np.zeros(len(days)); bh[mid:-1] = pxa[mid + 1:] / np.where(pxa[mid:-1] == 0, np.nan, pxa[mid:-1]) - 1
            bhn = round(float(np.nan_to_num(bh).sum() * 100), 1)
            etf_sh = 1.18 if a == "BTC" else 1.30
            be = sh > etf_sh; beats_etf += be
            print(f"  {a} {cohort:9s}: net {net:+.1f}% Sh {sh:+.2f} trades {tr} | b&h {bhn:+.1f}% "
                  f"| vs ETF Sh {etf_sh} -> {'BEATS-ETF' if be else 'no'} (weeks={len(weeks)}, OOS {days[mid]})")
    if measured == 0:
        print("  REFUSED — no data"); return 2
    print(f"  VERDICT: {'A COT COHORT BEATS ETF' if beats_etf else 'ETF FLOW REMAINS BEST'} "
          f"({beats_etf} cohort-asset beat the wired ETF Sharpe) — data-source search exhausted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
