"""[P378] "Provide the data TQC needs" — a NEW data basis: L2 order-book depth
imbalance from Binance futures bookDepth (backfillable, source we already use,
no new venue). Tests whether microstructure carries a 4H-horizon signal the
60m OHLCV+flow features do not (P375 higher-freq flow was NO PULSE).

WHY 4H: the sleeve trades on a 4H cadence at $11k on CDE's flat fee. A signal is
only monetizable at current scale if it lives at ~4h+. So depth imbalance is
aggregated to 4H and probed against 4h/12h forward returns, at honest CDE cost —
the same bar every other candidate faced. (Microstructure's natural edge is
short-horizon; if it only shows at 1h it is real information but NOT tradeable at
$11k/CDE — that distinction is reported.)

DATA: Binance Vision futures bookDepth daily CSVs (timestamp, percentage, depth,
notional). Depth imbalance per snapshot = (bid_notional - ask_notional) /
(bid+ask) over the +-percentage levels; aggregated to 4H (mean + last). Causal:
the 4H feature uses only snapshots within the completed bar; forward return is
the NEXT bar. Cost bar: CDE round-trip BTC 27.7 / ETH 44.0 / SOL 41.0 bps.
"""
from __future__ import annotations
import io
import sys
import zipfile
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "training" / "training_data" / "raw"
CACHE = Path(__file__).resolve().parent / "_microcache"
CACHE.mkdir(exist_ok=True)
SYMS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"}
COST_RT = {"BTC": 27.7e-4, "ETH": 44.0e-4, "SOL": 41.0e-4}
URL = "https://data.binance.vision/data/futures/um/daily/bookDepth/{sym}/{sym}-bookDepth-{d}.zip"
E_ABS_Z, PEARSON_K = 0.7979, 1.047


def fetch_day(sym, d):
    f = CACHE / f"{sym}-{d}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    url = URL.format(sym=sym, d=d)
    try:
        raw = urllib.request.urlopen(url, timeout=60).read()
    except Exception:
        return None
    zf = zipfile.ZipFile(io.BytesIO(raw))
    df = pd.read_csv(zf.open(zf.namelist()[0]))
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    # depth imbalance per snapshot: bid (percentage<0) vs ask (percentage>0) notional
    g = df.groupby("timestamp")
    bid = df[df["percentage"] < 0].groupby("timestamp")["notional"].sum()
    ask = df[df["percentage"] > 0].groupby("timestamp")["notional"].sum()
    imb = ((bid - ask) / (bid + ask)).rename("imb").to_frame()
    imb.to_parquet(f)
    return imb


def build_4h_feature(asset, days):
    sym = SYMS[asset]
    frames = []
    for d in days:
        x = fetch_day(sym, d.isoformat())
        if x is not None and len(x):
            frames.append(x)
    if not frames:
        return None
    imb = pd.concat(frames).sort_index()
    # aggregate to 4H: mean imbalance + last imbalance within each completed bar
    r = imb["imb"].resample("4h")
    feat = pd.DataFrame({"imb_mean": r.mean(), "imb_last": r.last()}).dropna()
    return feat


def load_4h_close(asset):
    d = pd.read_parquet(RAW / f"{asset}_60m.parquet")[["timestamp", "close"]]
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
    return d.set_index("timestamp")["close"].resample("4h").last().dropna()


def spearman(x, y):
    if len(x) < 50:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def required_ic(cost_bps, sigma_bps):
    return cost_bps / (E_ABS_Z * PEARSON_K * sigma_bps) if sigma_bps > 0 else float("inf")


def walk_forward(X, y, min_train=400, refit=100, gap=3):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    n = len(X)
    preds = np.full(n, np.nan)
    start = min_train
    while start + gap < n:
        te_s = start + gap
        te_e = min(te_s + refit, n)
        Xtr, ytr = X[:start], y[:start]
        m = ~(np.isnan(Xtr).any(axis=1) | np.isnan(ytr))
        if m.sum() >= 200:
            sc = StandardScaler().fit(Xtr[m])
            mdl = Ridge(alpha=10.0).fit(sc.transform(Xtr[m]), ytr[m])
            preds[te_s:te_e] = mdl.predict(sc.transform(np.nan_to_num(X[te_s:te_e])))
        start = te_e
    return preds


def probe(asset, days):
    feat = build_4h_feature(asset, days)
    if feat is None:
        print(f"{asset}: no bookDepth data fetched")
        return None
    close = load_4h_close(asset)
    df = feat.join(close.rename("close"), how="inner").dropna()
    if len(df) < 500:
        print(f"{asset}: only {len(df)} aligned 4H bars — insufficient")
        return None
    c = df["close"].to_numpy(float)
    X = df[["imb_mean", "imb_last"]].to_numpy(float)
    # causal: feature at bar i (completed) predicts forward return; lag X by 1
    Xl = np.vstack([np.full((1, X.shape[1]), np.nan), X[:-1]])
    out = {"asset": asset, "n_bars": len(df), "horizons": {}}
    for h in (1, 3):  # 4h, 12h
        fwd = np.full(len(c), np.nan)
        fwd[:len(c) - h] = c[h:] / c[:len(c) - h] - 1.0
        sigma = float(np.nanstd(fwd) * 1e4)
        preds = walk_forward(Xl, fwd)
        tm = ~(np.isnan(preds) | np.isnan(fwd))
        if tm.sum() < 100:
            continue
        ic = spearman(preds[tm], fwd[tm])
        gross = float(np.nanmean(np.sign(preds[tm]) * fwd[tm]) * 1e4)
        req = required_ic(COST_RT[asset] * 1e4, sigma)
        # raw single-feature IC too (is depth imbalance itself predictive?)
        raw_ic = spearman(np.nan_to_num(Xl[tm][:, 0]), fwd[tm])
        out["horizons"][h * 4] = {
            "n": int(tm.sum()), "sigma_bps": round(sigma, 1),
            "model_ic": round(ic, 4), "raw_imb_ic": round(raw_ic, 4),
            "req_ic": round(req, 4), "gross_bps": round(gross, 2),
            "net_bps": round(gross - COST_RT[asset] * 1e4, 2),
            "clears": bool(ic == ic and ic >= req and gross > COST_RT[asset] * 1e4)}
    return out


def main():
    ndays = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    assets = sys.argv[2].split(",") if len(sys.argv) > 2 else list(SYMS)
    end = date(2026, 7, 31)
    days = [end - timedelta(days=i) for i in range(ndays)][::-1]
    W = 90
    print("=" * W)
    print(f"  MICROSTRUCTURE PROBE — L2 depth imbalance (Binance bookDepth), {ndays}d, 4H horizon")
    print("  clears = IC>=required AND gross>CDE cost (monetizable at $11k/4H cadence)")
    print("=" * W)
    any_pulse = False
    for a in assets:
        r = probe(a, days)
        if not r:
            continue
        print(f"\n{a}  ({r['n_bars']} aligned 4H bars):")
        for hh, g in r["horizons"].items():
            if g["clears"]:
                any_pulse = True
            print(f"  {hh}h fwd (sigma {g['sigma_bps']}bps, req IC {g['req_ic']}): "
                  f"model IC {g['model_ic']:+.4f} | raw depth-imb IC {g['raw_imb_ic']:+.4f} | "
                  f"gross {g['gross_bps']:+.2f}bps net {g['net_bps']:+.2f}  "
                  f"[{'CLEARS' if g['clears'] else '-'}]")
    print("\n" + "=" * W)
    print(f"  VERDICT: {'PULSE — new-data basis worth pursuing' if any_pulse else 'NO PULSE at 4H — depth imbalance carries no tradeable 4H signal'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
