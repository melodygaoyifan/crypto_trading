"""[P261] The unread-era probe — attacking the P260 caveat with data no
selection has ever touched.

The caveat: every surviving backtest number rests on a validation window
read 7 times. This probe sits the surviving MECHANISMS on the two kinds of
data that are still genuinely unread:

  A. THE VIRGIN ERA — BTC/ETH 2017/2018 -> 2020-08-09 (the parquet start).
     Our parquets, and therefore every selection ever made in this repo,
     begin 2020-08-09. Everything before is untouched. Funding history
     (Binance UM archives) exists from late 2019, so the BTC book's funding
     legs run where funding exists and degrade to price legs before that
     (absence = flat for those legs, never fabricated — P2).
     HARD GUARD: no bar on/after 2020-08-09 is scored here.

  B. NEVER-FITTED ASSETS — BNB/XRP/LTC/DOGE/ADA over the same 2020-2026
     period. The WINDOW overlaps our sample but the ASSETS were never part
     of any selection, tuning, or read — cross-asset transfer is the
     orthogonal out-of-selection axis. Mechanism tested: trend_only
     (bull=SMA200 x 90d-momentum -> long, else flat), the exact ETH/SOL
     roster mechanism. Conservative 10bps RT cost.

  C. DEFLATED SHARPE — the BTC book's validation Sharpe corrected for the
     multi-family search (Bailey & Lopez de Prado DSR) at N=12 and N=50
     effective trials, computed from the book's actual bar returns.

Data source: data.binance.vision monthly archives (api.binance.com is
geo-blocked, HTTP 451). Downloads cached under
training/training_data/unread_probe/ (gitignored).
"""
from __future__ import annotations

import csv
import io
import json
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
if REPO.name == "training":
    REPO = REPO.parent
sys.path.insert(0, str(REPO))

from training.mechanism_lab import book_targets  # noqa: E402

CACHE = REPO / "training" / "training_data" / "unread_probe"
CACHE.mkdir(parents=True, exist_ok=True)
OUT = REPO / "training" / "reports" / "unread_era_probe_p261.json"
PARQUET_START_MS = int(datetime(2020, 8, 9, tzinfo=timezone.utc).timestamp() * 1000)
COST_RT = {"BTC": 6.0, "ETH": 8.0}
XASSET_COST_RT = 10.0


def _fetch(url: str, dest: Path):
    if dest.exists():
        return dest
    try:
        r = urllib.request.urlopen(
            urllib.request.Request(url, headers={"User-Agent": "hmats"}),
            timeout=30)
        dest.write_bytes(r.read())
        return dest
    except Exception:
        return None


def monthly_klines(symbol: str, y0: int, y1: int):
    """4h close series [(open_ms, close)] from vision monthly zips."""
    rows = []
    for y in range(y0, y1 + 1):
        for m in range(1, 13):
            z = _fetch(
                f"https://data.binance.vision/data/spot/monthly/klines/"
                f"{symbol}/4h/{symbol}-4h-{y}-{m:02d}.zip",
                CACHE / f"{symbol}-4h-{y}-{m:02d}.zip")
            if z is None:
                continue
            try:
                with zipfile.ZipFile(z) as zf:
                    with zf.open(zf.namelist()[0]) as fh:
                        for r in csv.reader(io.TextIOWrapper(fh, "utf-8")):
                            if not r or not r[0].isdigit():
                                continue
                            ts = int(r[0])
                            ts = ts // 1000 if ts > 10**14 else ts  # us->ms
                            rows.append((ts, float(r[4])))
            except Exception:
                continue
    rows.sort()
    # dedup on timestamp
    out, seen = [], set()
    for ts, c in rows:
        if ts not in seen:
            seen.add(ts)
            out.append((ts, c))
    return out


def monthly_funding(symbol: str, y0: int, y1: int):
    """Daily last funding rate {date_iso: rate} from UM monthly archives."""
    by_day = {}
    for y in range(y0, y1 + 1):
        for m in range(1, 13):
            z = _fetch(
                f"https://data.binance.vision/data/futures/um/monthly/"
                f"fundingRate/{symbol}/{symbol}-fundingRate-{y}-{m:02d}.zip",
                CACHE / f"{symbol}-fr-{y}-{m:02d}.zip")
            if z is None:
                continue
            try:
                with zipfile.ZipFile(z) as zf:
                    with zf.open(zf.namelist()[0]) as fh:
                        for r in csv.reader(io.TextIOWrapper(fh, "utf-8")):
                            if not r or not r[0].isdigit():
                                continue
                            ts = int(r[0])
                            ts = ts // 1000 if ts > 10**14 else ts
                            d = datetime.fromtimestamp(
                                ts / 1000, tz=timezone.utc).date().isoformat()
                            by_day[d] = float(r[-1] if len(r) < 4 else r[2])
            except Exception:
                continue
    return by_day


def causal_labels(close: np.ndarray) -> np.ndarray:
    """0=peace 1=bull 2=bear — same construction as regime_model_lab."""
    n = len(close)
    lab = np.zeros(n, dtype=int)
    import pandas as pd
    sma = pd.Series(close).rolling(200).mean().values
    for i in range(n):
        if i < 540 or np.isnan(sma[i]):
            continue
        mom = close[i] / close[i - 540] - 1.0
        above, up = close[i] > sma[i], mom > 0
        lab[i] = 1 if (above and up) else (2 if (not above and not up) else 0)
    return lab


def daily_causal_z(by_day: dict, dates):
    """Causal 30d trailing z of YESTERDAY's rate, per bar date list."""
    days = sorted(by_day)
    z_by_day = {}
    for i, d in enumerate(days):
        w = [by_day[x] for x in days[max(0, i - 29):i + 1]]
        if len(w) < 30:
            z_by_day[d] = None
            continue
        mu = sum(w) / len(w)
        sd = (sum((x - mu) ** 2 for x in w) / len(w)) ** 0.5
        z_by_day[d] = 0.0 if sd <= 0 else (w[-1] - mu) / sd
    # bar on date D reads day D-1's z (P247 discipline)
    out = []
    prev = {days[i]: days[i - 1] for i in range(1, len(days))}
    for d in dates:
        y = prev.get(d) if d in prev else None
        # nearest previous day
        if y is None:
            cands = [x for x in days if x < d]
            y = cands[-1] if cands else None
        out.append(z_by_day.get(y) if y else None)
    return out


def pnl(close, pos, cost_rt, lo, hi):
    r1 = np.zeros_like(close)
    r1[:-1] = close[1:] / close[:-1] - 1.0
    seg = slice(lo, hi - 1)
    gross = float(np.nansum(pos[seg] * r1[seg]))
    dpos = np.abs(np.diff(pos[lo:hi], prepend=pos[lo]))
    cost = float(np.nansum(dpos) * (cost_rt / 2.0) / 1e4)
    return round(gross - cost, 4)


def part_a():
    res = {}
    for a, sym in [("BTC", "BTCUSDT"), ("ETH", "ETHUSDT")]:
        rows = monthly_klines(sym, 2017, 2020)
        rows = [r for r in rows if r[0] < PARQUET_START_MS]   # HARD GUARD
        ts = np.array([r[0] for r in rows])
        close = np.array([r[1] for r in rows])
        assert ts.max() < PARQUET_START_MS
        lab = causal_labels(close)
        dates = [datetime.fromtimestamp(t / 1000, tz=timezone.utc
                                        ).date().isoformat() for t in ts]
        fr = monthly_funding(sym, 2019, 2020)
        zlist = daily_causal_z(fr, dates) if fr else [None] * len(ts)
        fz = np.array([np.nan if z is None else z for z in zlist])
        book = book_targets(a, lab, fz)
        trend = np.where(lab == 1, 1.0, 0.0)
        lo = 600                      # past SMA200+mom540 warmup
        hi = len(close)
        res[a] = {"bars_scored": hi - lo - 1,
                  "span": [dates[lo], dates[-1]],
                  "book_net": pnl(close, book, COST_RT[a], lo, hi),
                  "trend_only_net": pnl(close, trend, COST_RT[a], lo, hi),
                  "buy_hold": round(close[hi - 1] / close[lo] - 1.0, 4),
                  "funding_days": len(fr)}
        # per-year
        import pandas as pd
        yr = pd.Series([d[:4] for d in dates])
        res[a]["by_year"] = {}
        for y in sorted(yr.unique()):
            idx = np.where(yr.values == y)[0]
            l2, h2 = max(lo, idx.min()), idx.max() + 1
            if h2 - l2 < 100:
                continue
            res[a]["by_year"][y] = {
                "book": pnl(close, book, COST_RT[a], l2, h2),
                "trend": pnl(close, trend, COST_RT[a], l2, h2),
                "bh": round(close[h2 - 1] / close[l2] - 1.0, 4)}
        print(f"[A] {a}: {res[a]['span']} book={res[a]['book_net']:+.4f} "
              f"trend={res[a]['trend_only_net']:+.4f} "
              f"B&H={res[a]['buy_hold']:+.4f}")
    return res


def part_b():
    res = {}
    wins_vs_flat, wins_vs_bh = 0, 0
    for sym in ("BNBUSDT", "XRPUSDT", "LTCUSDT", "DOGEUSDT", "ADAUSDT"):
        rows = monthly_klines(sym, 2020, 2026)
        if len(rows) < 2000:
            res[sym] = {"error": f"only {len(rows)} bars"}
            continue
        close = np.array([r[1] for r in rows])
        lab = causal_labels(close)
        trend = np.where(lab == 1, 1.0, 0.0)
        lo, hi = 600, len(close)
        t = pnl(close, trend, XASSET_COST_RT, lo, hi)
        bh = round(close[hi - 1] / close[lo] - 1.0, 4)
        res[sym] = {"bars": hi - lo, "trend_net": t, "buy_hold": bh,
                    "beats_flat": t > 0, "beats_bh": t > bh}
        wins_vs_flat += t > 0
        wins_vs_bh += t > bh
        print(f"[B] {sym}: trend={t:+.4f} B&H={bh:+.4f} "
              f"{'beats both' if t > 0 and t > bh else ''}")
    res["_summary"] = {"beats_flat": f"{wins_vs_flat}/5",
                       "beats_buy_hold": f"{wins_vs_bh}/5"}
    return res


def part_c():
    """Deflated Sharpe of the BTC book on the validation era."""
    from scipy import stats
    from training.regime_model_lab import _ctx
    c = _ctx("BTC")
    close, lab, fz = c["close"], c["lab"], c["fz"]
    book = book_targets("BTC", lab, fz)
    r1 = np.zeros_like(close)
    r1[:-1] = close[1:] / close[:-1] - 1.0
    VS, VE = 9100, len(close)
    rets = (book * r1)[VS:VE - 1]
    dpos = np.abs(np.diff(book[VS:VE], prepend=book[VS]))[:VE - VS - 1]
    rets = rets - dpos * (6.0 / 2.0) / 1e4
    n = len(rets)
    sr = float(np.mean(rets) / (np.std(rets) + 1e-12))       # per-bar SR
    skew = float(stats.skew(rets))
    kurt = float(stats.kurtosis(rets, fisher=False))
    out = {"n_bars": n, "sharpe_ann": round(sr * np.sqrt(6 * 365), 3),
           "skew": round(skew, 3), "kurtosis": round(kurt, 2)}
    for N in (12, 50):
        e = np.exp(1.0)
        z1 = stats.norm.ppf(1 - 1.0 / N)
        z2 = stats.norm.ppf(1 - 1.0 / (N * e))
        gamma = 0.5772156649
        sr0 = (1.0 / np.sqrt(n - 1)) * ((1 - gamma) * z1 + gamma * z2)
        num = (sr - sr0) * np.sqrt(n - 1)
        den = np.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr ** 2))
        dsr = float(stats.norm.cdf(num / den))
        out[f"dsr_N{N}"] = round(dsr, 4)
        out[f"sr0_N{N}_ann"] = round(sr0 * np.sqrt(6 * 365), 3)
    print(f"[C] BTC book val: SR_ann={out['sharpe_ann']} "
          f"DSR(N=12)={out['dsr_N12']} DSR(N=50)={out['dsr_N50']}")
    return out


def main() -> int:
    report = {"generated": datetime.now(timezone.utc).isoformat(),
              "guard": "Part A scores NO bar on/after 2020-08-09 (the "
                       "parquet start — where read data begins)"}
    report["A_virgin_era"] = part_a()
    report["B_never_fitted_assets"] = part_b()
    report["C_deflated_sharpe"] = part_c()
    OUT.write_text(json.dumps(report, indent=1, default=str),
                   encoding="utf-8")
    print(f"report: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
