"""[P381] RECONSTRUCTION of the whale_filter — the honest resolution of "forward-only".

The whale_filter judges the whale AGENT's direction against the decider and vetoes
an entry on disagreement (P236/P356). Calling it "forward-only, cannot be
backtested by construction" was TOO BROAD: `WhaleDetector` is a DETERMINISTIC rule
($100K absolute notional, 10x average trade size, 5% of 1% depth), not a trained
model, so its output is RECOMPUTABLE over historical trades. This lab does exactly
that over real Binance futures aggTrades, replays the deterministic regimebook
decider (SMA200+funding), and buckets forward returns by whale agree/disagree at
the decider's directional entries — the P324/P337 disagreement methodology on a
TRUE detector replay over a fresh window rather than on stored attribution.

FIDELITY. The detector's whale rule is reproduced in numpy for speed (176M trades
over the window make a per-trade Python loop hopeless) and CROSS-CHECKED against
the real `WhaleDetector.detect()` on a sample (P172 — the reproduction must match
the source of truth or the run is void). Two of the three legs are reproduced:
ABSOLUTE (>$100K) and RELATIVE (>10x the rolling mean of the last 10k trades).
The DEPTH leg (5% of 1% orderbook depth) is genuinely absent from aggTrades and
is DOCUMENTED as omitted: live whales need >=2 of 3 legs, so with depth present a
few whales can qualify on ABSOLUTE+DEPTH; here they need ABSOLUTE+RELATIVE, a
slightly stricter set. The cross-check reports the size of that gap.

WHAT IT CAN AND CANNOT SETTLE. A real point estimate over the window. It CANNOT
reach significance — P348: the filter's effect size needs ~2.8 years for |t|>=2,
and no downloadable aggTrades window is that long. So NOT_EARNED is consistent
with P337 (neutral); a POSITIVE would still be a hypothesis, not a promotion.

VERDICT, pre-committed before the first number: the filter EARNS iff the
disagree-bucket mean 16h forward return (of the decider's own direction) is
negative AND lower than the agree-bucket, with >=30 disagreements. Otherwise
NOT_EARNED (it blocks entries that were not worse).
"""
from __future__ import annotations
import io
import json
import sys
import zipfile
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "training"))

from agents.whale_detector import WhaleDetector  # noqa: E402  (source of truth, P172)

RAW = REPO / "training" / "training_data" / "raw"
CACHE = Path(__file__).resolve().parent / "_whalecache"
CACHE.mkdir(exist_ok=True)
SYMS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"}
URL = ("https://data.binance.vision/data/futures/um/daily/aggTrades/"
       "{sym}/{sym}-aggTrades-{d}.zip")
ABS = WhaleDetector.ABSOLUTE_NOTIONAL_USD      # 100_000
REL = WhaleDetector.RELATIVE_TRADE_SIZE        # 10
AVGN = WhaleDetector.TRADE_HISTORY_SIZE        # 10_000
PWIN = WhaleDetector.PRESSURE_WINDOW_S         # 3600
MIN_WHALES = 2                                  # coinbase_whale_filter_min_whales (P349)


def whale_dir_from_pressure(net):
    """Mirror main.py:3001 whale_direction_from_pressure (P293d/P349)."""
    if net > 0.3:
        return 1.0
    if net < -0.3:
        return -1.0
    return 0.0


def _read_full_day(sym, d):
    """Download+parse one full day -> DataFrame(ts_s, notional, side). Cached raw."""
    f = CACHE / f"{sym}-{d}.parquet"
    if f.exists():
        df = pd.read_parquet(f)
        if "side" in df.columns:
            return df
    url = URL.format(sym=sym, d=d)
    try:
        raw = urllib.request.urlopen(url, timeout=180).read()
    except Exception:
        return None
    zf = zipfile.ZipFile(io.BytesIO(raw))
    df = pd.read_csv(zf.open(zf.namelist()[0]), header=None,
                     names=["agg_id", "price", "qty", "first_id", "last_id",
                            "ts", "is_buyer_maker"])
    df = df[pd.to_numeric(df["price"], errors="coerce").notna()].copy()
    df["notional_usd"] = df["price"].astype(float) * df["qty"].astype(float)
    ibm = df["is_buyer_maker"].astype(str).str.lower().isin(("true", "1"))
    df["side"] = np.where(ibm, "SELL", "BUY")  # aggressor side
    out = df[["ts", "notional_usd", "side"]].copy()
    out["ts"] = out["ts"].astype("int64") / 1000.0  # ms -> s
    out.to_parquet(f)
    return out


def whales_for_day(sym, d):
    """The detector's whale set for one day (ABSOLUTE + RELATIVE legs, vectorized).
    Cached small so a 120d run is fast on re-run."""
    wf = CACHE / f"{sym}-{d}-whales.parquet"
    if wf.exists():
        return pd.read_parquet(wf)
    day = _read_full_day(sym, d)
    if day is None or not len(day):
        return None
    n = day["notional_usd"].to_numpy(float)
    # rolling mean of last AVGN trade notionals (the detector's _avg_trade_size)
    avg = pd.Series(n).rolling(AVGN, min_periods=1).mean().to_numpy()
    is_whale = (n > ABS) & (avg > 0) & (n > REL * avg)  # 2 legs; depth absent
    w = day.loc[is_whale, ["ts", "notional_usd", "side"]].copy()
    w.to_parquet(wf)
    return w


def build_whale_dir_4h(asset, days):
    sym = SYMS[asset]
    frames = []
    for d in days:
        w = whales_for_day(sym, d.isoformat())
        if w is not None and len(w):
            frames.append(w)
    if not frames:
        return None
    w = pd.concat(frames).sort_values("ts").reset_index(drop=True)
    ts = w["ts"].to_numpy(float)
    signed = np.where(w["side"].to_numpy() == "BUY", 1.0, -1.0) * w["notional_usd"].to_numpy(float)
    lo, hi = ts.min(), ts.max()
    # LEFT-edge bins to match pandas resample("4h") (00,04,08,12,16,20 UTC).
    # The decider bar labeled T holds over [T,T+4h); the whale direction known
    # AT the decision time T is the net pressure over the trailing [T-1h, T].
    first_bar = (int(lo) // (4 * 3600)) * (4 * 3600)
    last_bar = (int(hi) // (4 * 3600)) * (4 * 3600)
    rows = []
    for T in range(first_bar, last_bar + 1, 4 * 3600):
        i0 = np.searchsorted(ts, T - PWIN, side="right")
        i1 = np.searchsorted(ts, T, side="right")
        if i1 <= i0:
            rows.append((T, 0.0, 0)); continue
        seg = signed[i0:i1]
        buy = seg[seg > 0].sum(); sell = -seg[seg < 0].sum()
        tot = buy + sell
        net = (buy - sell) / tot if tot > 0 else 0.0
        rows.append((T, whale_dir_from_pressure(net), int(i1 - i0)))
    df = pd.DataFrame(rows, columns=["bar_s", "whale_dir", "whale_cnt"])
    df["bar"] = pd.to_datetime(df["bar_s"], unit="s", utc=True)
    return df.set_index("bar")[["whale_dir", "whale_cnt"]]


def fidelity_check(asset, day):
    """Prove the vectorized whale set matches the REAL detector on a sample (P172)."""
    sym = SYMS[asset]
    full = _read_full_day(sym, day.isoformat())
    if full is None:
        return None
    s = full.head(200_000).reset_index(drop=True)
    # real detector, depth=0 (as in this reconstruction)
    det = WhaleDetector()
    real = np.zeros(len(s), bool)
    for i, (ts, notional, side) in enumerate(s[["ts", "notional_usd", "side"]].itertuples(index=False)):
        sig = det.detect(asset, float(notional), side, trade_ts=float(ts), trade_id=str(i))
        real[i] = bool(sig.is_whale)
    # vectorized, same 2-leg rule
    n = s["notional_usd"].to_numpy(float)
    avg = pd.Series(n).rolling(AVGN, min_periods=1).mean().to_numpy()
    vec = (n > ABS) & (avg > 0) & (n > REL * avg)
    agree = int((real == vec).sum())
    return {"sample": len(s), "real_whales": int(real.sum()),
            "vec_whales": int(vec.sum()), "match_rate": round(agree / len(s), 6),
            "abs_only_whales": int((n > ABS).sum())}


def decider_dir_4h(asset):
    from training.regime_model_lab import _ctx
    from training.mechanism_lab import book_targets
    c = _ctx(asset)
    pos = book_targets(asset, c["lab"], c["fz"])
    close = c["close"]
    d = pd.read_parquet(RAW / f"{asset}_60m.parquet")[["timestamp", "close"]]
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
    idx = d.set_index("timestamp")["close"].resample("4h").last().dropna().index
    n = min(len(pos), len(close), len(idx))
    return pd.DataFrame({"dec_dir": np.asarray(pos[:n], float),
                         "close": np.asarray(close[:n], float)}, index=idx[:n])


def analyze(asset, days):
    wd = build_whale_dir_4h(asset, days)
    if wd is None:
        return {"asset": asset, "error": "no whale data"}
    dd = decider_dir_4h(asset)
    j = dd.join(wd, how="inner").dropna(subset=["dec_dir", "whale_dir"])
    if len(j) < 100:
        return {"asset": asset, "error": f"only {len(j)} joined 4H bars"}
    c = j["close"].to_numpy(float)
    out = {"asset": asset, "n_bars": len(j),
           "window": [str(j.index.min()), str(j.index.max())], "horizons": {}}
    for h, hname in ((1, "4h"), (4, "16h")):
        fwd = np.full(len(c), np.nan); fwd[:len(c) - h] = c[h:] / c[:len(c) - h] - 1.0
        dec = j["dec_dir"].to_numpy(float); wdir = j["whale_dir"].to_numpy(float)
        wcnt = j["whale_cnt"].to_numpy(float)
        signed = dec * fwd
        entry = (dec != 0) & ~np.isnan(fwd)
        has_op = entry & (wdir != 0) & (wcnt >= MIN_WHALES)
        agree = has_op & (np.sign(wdir) == np.sign(dec))
        disag = has_op & (np.sign(wdir) != np.sign(dec))
        a = signed[agree] * 1e4; di = signed[disag] * 1e4
        out["horizons"][hname] = {
            "n_entry": int(entry.sum()), "n_agree": int(agree.sum()),
            "n_disagree": int(disag.sum()),
            "agree_bps": round(float(np.nanmean(a)), 2) if len(a) else None,
            "disagree_bps": round(float(np.nanmean(di)), 2) if len(di) else None,
            "contrast_bps": (round(float(np.nanmean(a) - np.nanmean(di)), 2)
                             if len(a) and len(di) else None)}
    h = out["horizons"]["16h"]
    earns = (h["disagree_bps"] is not None and h["n_disagree"] >= 30
             and h["disagree_bps"] < 0 and h["disagree_bps"] < (h["agree_bps"] or 0))
    out["verdict"] = ("EARNS (disagreements marked worse entries)" if earns
                      else "NOT_EARNED (disagreements not worse)")
    return out


def main():
    ndays = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    assets = sys.argv[2].split(",") if len(sys.argv) > 2 else list(SYMS)
    end = date(2026, 7, 22)  # within decider (_ctx) coverage
    days = [end - timedelta(days=i) for i in range(ndays)][::-1]
    res = {"ndays": ndays, "min_whales": MIN_WHALES, "assets": {},
           "note": "point estimate; not significance (P348: ~2.8y needed)"}
    W = 92
    print("=" * W)
    print(f"  WHALE_FILTER RECONSTRUCTION — real detector rule over {ndays}d futures aggTrades")
    print("  VERDICT: disagree 16h return negative AND < agree, >=30 disagreements")
    print("=" * W)
    fc = fidelity_check(assets[0], days[-1])
    if fc:
        res["fidelity"] = fc
        print(f"\nFIDELITY (real vs vectorized, {fc['sample']} trades of {assets[0]} "
              f"{days[-1].isoformat()}): match {fc['match_rate']*100:.4f}%  "
              f"(real {fc['real_whales']} / vec {fc['vec_whales']} / abs-only "
              f"{fc['abs_only_whales']})")
    for a in assets:
        r = analyze(a, days)
        res["assets"][a] = r
        if "error" in r:
            print(f"\n{a}: {r['error']}"); continue
        print(f"\n{a}  ({r['n_bars']} joined 4H bars, {r['window'][0][:10]} -> "
              f"{r['window'][1][:10]}):")
        for hn, g in r["horizons"].items():
            print(f"  {hn}: entries {g['n_entry']}  agree {g['n_agree']} "
                  f"({g['agree_bps']}bps)  disagree {g['n_disagree']} "
                  f"({g['disagree_bps']}bps)  contrast {g['contrast_bps']}")
        print(f"  -> {r['verdict']}")
    (REPO / "training" / "reports" / "whale_filter_reconstruction_p381.json"
     ).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("\nreport -> training/reports/whale_filter_reconstruction_p381.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
