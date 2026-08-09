"""Per-asset data-pattern diagnostics that DRIVE model-family selection.

Model selection should follow measured data characteristics, not defaults.
For each asset this measures:

  1. Trend character      — variance ratios VR(4)/VR(8) and AC1 of 16h
                            returns: >1 momentum-supportive, <1 reverting.
  2. Signal decay         — fit a ridge at anchor points, track OOS rank-IC
                            by months-since-fit; half-life in bars decides
                            refit cadence (and whether frozen models can
                            work at all).
  3. Linearity gap        — purged-fold OOS rank-IC of HGB minus ridge on
                            identical data: positive = nonlinear structure
                            a tree/net can use; ~zero/negative = linear
                            family suffices.
  4. Regime dependence    — IC of a simple momentum signal per GMM regime
                            (argmax of regime_proba_*): high dispersion =
                            regime-conditional models / gating warranted.
  5. Sample geometry      — effective n, 16h-return kurtosis, vol
                            clustering (AC1 of |ret|): tail risk and how
                            much model capacity the data can support.

Verdict rules are printed with the numbers so the mapping from measurement
to family recommendation is explicit and criticizable.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent.parent
DATA = REPO / "training" / "training_data" / "drl_training"

from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

ASSETS = ("BTC", "ETH", "SOL")
H = 4  # 16h label on 4H bars
SEED = 7

manifest = json.loads((REPO / "configs" / "feature_manifest.json").read_text(encoding="utf-8"))


def rank_ic(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 30:
        return np.nan
    return float(stats.spearmanr(a[m], b[m]).statistic)


def load(asset):
    df = pd.read_parquet(DATA / f"{asset}_4H_full.parquet")
    feats = [c for c in manifest["all_features"] if c in df.columns]
    feats += sorted(c for c in df.columns if c.startswith("fv2_"))
    close = df["close"].to_numpy(dtype=float)
    y = np.full(len(close), np.nan)
    y[:-H] = close[H:] / close[:-H] - 1.0
    X = df[feats].to_numpy(dtype=float)
    regime = None
    rp = [c for c in df.columns if c.startswith("regime_proba_")]
    if rp:
        regime = df[sorted(rp)].to_numpy(dtype=float).argmax(axis=1)
    return X, y, close, regime, feats


def trend_character(close):
    r1 = np.diff(np.log(close))
    out = {}
    for q in (4, 8):
        rq = np.log(close[q:]) - np.log(close[:-q])
        out[f"VR{q}"] = float(np.var(rq) / (q * np.var(r1)))
    r16 = np.log(close[H:]) - np.log(close[:-H])
    nonoverlap = r16[::H]
    out["AC1_16h"] = float(np.corrcoef(nonoverlap[:-1], nonoverlap[1:])[0, 1])
    out["kurtosis_16h"] = float(stats.kurtosis(nonoverlap))
    out["vol_cluster_AC1"] = float(np.corrcoef(np.abs(nonoverlap[:-1]),
                                               np.abs(nonoverlap[1:]))[0, 1])
    return out


def signal_decay(X, y, n):
    """Fit ridge at anchors; measure OOS IC in 180-bar (30d) chunks after
    the fit. Averaged decay curve -> half-life in bars."""
    TRAIN, HORIZON, CHUNK = 4320, 1440, 180
    curves = []
    for t0 in range(TRAIN + 500, n - HORIZON, 720):
        tr = slice(t0 - TRAIN, t0 - H)
        m = ~(np.isnan(X[tr]).any(axis=1) | np.isnan(y[tr]))
        if m.sum() < 2000:
            continue
        sc = StandardScaler().fit(X[tr][m])
        mod = Ridge(alpha=30.0).fit(sc.transform(X[tr][m]), y[tr][m])
        chunk_ics = []
        for c0 in range(t0, t0 + HORIZON, CHUNK):
            seg = slice(c0, min(c0 + CHUNK, n))
            p = mod.predict(sc.transform(np.nan_to_num(X[seg])))
            chunk_ics.append(rank_ic(p, y[seg]))
        curves.append(chunk_ics)
    if not curves:
        return {}
    arr = np.nanmean(np.array(curves, dtype=float), axis=0)
    ic0 = arr[0]
    half_life = None
    if np.isfinite(ic0) and ic0 > 0:
        below = np.where(arr < ic0 / 2)[0]
        half_life = int(below[0] * 180) if len(below) else int(len(arr) * 180)
    return {"ic_by_month": [round(float(v), 4) for v in arr],
            "ic_month1": round(float(ic0), 4),
            "half_life_bars": half_life,
            "n_anchor_fits": len(curves)}


def linearity_gap(X, y, n):
    """Purged 5-fold: HGB rank-IC minus ridge rank-IC on identical folds."""
    start, embargo = 3000, 42
    span = (n - start) // 5
    gaps, ridge_ics, hgb_ics = [], [], []
    for k in range(5):
        v0, v1 = start + k * span, start + (k + 1) * span
        tr_idx = np.concatenate([np.arange(0, max(0, v0 - H - embargo)),
                                 np.arange(min(n, v1 + H + embargo), n)])
        m = ~(np.isnan(X[tr_idx]).any(axis=1) | np.isnan(y[tr_idx]))
        tr_idx = tr_idx[m]
        va = np.arange(v0, v1)
        sc = StandardScaler().fit(X[tr_idx])
        pr = Ridge(alpha=30.0).fit(sc.transform(X[tr_idx]), y[tr_idx]) \
            .predict(sc.transform(np.nan_to_num(X[va])))
        ph = HistGradientBoostingRegressor(max_iter=150, max_depth=3,
                                           random_state=SEED) \
            .fit(X[tr_idx], y[tr_idx]).predict(np.nan_to_num(X[va]))
        ir, ih = rank_ic(pr, y[va]), rank_ic(ph, y[va])
        ridge_ics.append(ir); hgb_ics.append(ih); gaps.append(ih - ir)
    return {"ridge_ic_mean": round(float(np.nanmean(ridge_ics)), 4),
            "hgb_ic_mean": round(float(np.nanmean(hgb_ics)), 4),
            "gap_mean": round(float(np.nanmean(gaps)), 4),
            "gap_per_fold": [round(float(g), 4) for g in gaps]}


def regime_dependence(close, y, regime):
    if regime is None:
        return {}
    mom = np.full(len(close), np.nan)
    mom[42:] = close[42:] / close[:-42] - 1.0  # 7d momentum signal
    out = {}
    for r in np.unique(regime):
        m = regime == r
        if m.sum() < 200:
            continue
        out[f"regime_{int(r)}"] = {"n": int(m.sum()),
                                   "mom_ic": round(rank_ic(mom[m], y[m]), 4)}
    ics = [v["mom_ic"] for v in out.values() if np.isfinite(v["mom_ic"])]
    out["dispersion"] = round(float(np.std(ics)), 4) if len(ics) > 1 else None
    return out


def verdict(t, d, l):
    fams = []
    vr = t.get("VR8", 1.0)
    if vr > 1.15:
        fams.append("trend/momentum-supportive (VR8>1.15)")
    elif vr < 0.85:
        fams.append("mean-reversion-supportive (VR8<0.85)")
    hl = d.get("half_life_bars")
    if hl is not None and hl <= 720:
        fams.append(f"ADAPTIVE REFIT MANDATORY (half-life {hl} bars <= 720); "
                    "frozen deep models will decay before deployment matures")
    gap = l.get("gap_mean", 0.0)
    if gap is not None and gap > 0.02:
        fams.append(f"NONLINEAR structure (HGB-ridge gap +{gap:.3f}): "
                    "trees/nets/RL warranted")
    elif gap is not None and gap < -0.01:
        fams.append(f"LINEAR family suffices (gap {gap:+.3f}): extra "
                    "capacity buys overfit, not signal")
    else:
        fams.append(f"linearity gap inconclusive ({gap:+.3f})")
    return fams


def main():
    report = {}
    for asset in ASSETS:
        X, y, close, regime, feats = load(asset)
        n = len(close)
        print(f"\n########## {asset} ({n} bars, {len(feats)} features) ##########", flush=True)
        t = trend_character(close)
        print(f"  trend: VR4={t['VR4']:.3f} VR8={t['VR8']:.3f} AC1_16h={t['AC1_16h']:+.3f} "
              f"kurt={t['kurtosis_16h']:.1f} vol_cluster={t['vol_cluster_AC1']:.2f}", flush=True)
        d = signal_decay(X, y, n)
        print(f"  decay: IC month-by-month {d.get('ic_by_month')} -> half-life "
              f"{d.get('half_life_bars')} bars ({d.get('n_anchor_fits')} anchors)", flush=True)
        l = linearity_gap(X, y, n)
        print(f"  linearity: ridge IC {l['ridge_ic_mean']:+.4f} vs HGB {l['hgb_ic_mean']:+.4f} "
              f"-> gap {l['gap_mean']:+.4f} per-fold {l['gap_per_fold']}", flush=True)
        r = regime_dependence(close, y, regime)
        print(f"  regime-dependence: dispersion {r.get('dispersion')} "
              f"({ {k: v['mom_ic'] for k, v in r.items() if k != 'dispersion'} })", flush=True)
        v = verdict(t, d, l)
        for line in v:
            print(f"  VERDICT: {line}", flush=True)
        report[asset] = {"trend": t, "decay": d, "linearity": l,
                         "regime": r, "verdict": v}
    out = REPO / "training" / "reports" / "data_pattern_diagnostics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nDONE -> {out}")


if __name__ == "__main__":
    sys.exit(main() or 0)
