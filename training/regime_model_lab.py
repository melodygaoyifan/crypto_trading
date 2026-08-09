"""[P244] Regime model lab — the full data-science lifecycle, per asset,
per regime (bull / bear / peace), with the overfitting protocol baked in.

Operator specification: EDA -> model selection -> hyperparameter tuning ->
train/test evaluation, per asset, with a model per regime (bull, bear,
peace/calm), families spanning time-series / ensemble / deep learning, and
derivatives (funding) granularity for the Coinbase-traded perp venue.

Anti-overfit protocol (the P243 composite falsification, baked in):
  * REGIME LABELS are causal and fixed A PRIORI (no tuning on outcomes):
      mom = close/close[540 bars ago] - 1     (90d momentum)
      bull  : close > SMA200 and mom > 0
      bear  : close < SMA200 and mom < 0
      peace : the two indicators disagree (ranging/transition)
  * ALL selection + tuning happens inside the DESIGN ERA [3000, 9100).
  * The assembled per-regime system is evaluated ONCE on the untouched
    VALIDATION ERA [9100, end) — and separately reported on the
    pre-design-era probe window for era stability.
  * TRAIN metrics are reported NEXT TO test metrics for every cell, so the
    overfit gap is a first-class artifact, not a post-hoc discovery.
  * The final arbiter is the 30d live forward shadow (P166) — every window
    in this dataset has by now been seen in aggregate; only forward data
    is unbiased.

Stages (run via --stage):
  eda      Stage 1: per-asset, per-regime profiles — durations, transition
           matrix, fwd-16h target stats, momentum/reversal IC per regime,
           top features per regime, funding-quartile conditioning (the
           derivatives cut), GMM-label agreement.
  select   Stage 2: per-regime model selection + tuning (design era only),
           families: flat/hold baselines, ridge, AR(p) time-series, LGBM,
           stacking ensemble, small GRU. Purged CV inside the design era.
  assemble Stage 3: assemble per-regime winners into the switched system;
           single-shot on the validation era + pre-design era report.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent
if REPO.name == "training":
    REPO = REPO.parent
sys.path.insert(0, str(REPO))

from scipy import stats

from training.train_supervised_full import (  # noqa: E402
    load_asset, COST_BPS, BARS_PER_YEAR, H,
)

DESIGN = (3000, 9100)
VALIDATION_START = 9100
SMA_W, MOM_W = 200, 540
SEED = 7


# ---------------------------------------------------------------- labels
def regime_labels(close):
    """Causal 3-state labels, fixed a priori. 0=peace 1=bull 2=bear."""
    sma = pd.Series(close).rolling(SMA_W).mean().to_numpy()
    mom = np.full(len(close), np.nan)
    mom[MOM_W:] = close[MOM_W:] / close[:-MOM_W] - 1.0
    lab = np.zeros(len(close), dtype=int)
    above, up = close > sma, mom > 0
    lab[above & up] = 1
    lab[~above & ~up & ~np.isnan(mom)] = 2
    lab[np.isnan(sma) | np.isnan(mom)] = 0
    return lab


NAMES = {0: "peace", 1: "bull", 2: "bear"}


def _rank_ic(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 30:
        return np.nan
    return float(stats.spearmanr(a[m], b[m]).statistic)


# ---------------------------------------------------------------- stage 1
def stage_eda(assets):
    report = {}
    for asset in assets:
        X, targets, close, gmm_regime, feats = load_asset(asset)
        n = len(close)
        lab = regime_labels(close)
        y = targets["ret"]
        s, e = DESIGN
        print(f"\n########## {asset} EDA (design era [{s},{e}), {e-s} bars) ##########", flush=True)
        rep = {"labels_pct": {}, "durations": {}, "transition": {},
               "target": {}, "signal_ic": {}, "top_features": {},
               "funding_cut": {}, "gmm_agreement": {}}

        seg = lab[s:e]
        # durations + transition matrix
        runs, cur, ln = [], seg[0], 1
        for v in seg[1:]:
            if v == cur:
                ln += 1
            else:
                runs.append((cur, ln)); cur, ln = v, 1
        runs.append((cur, ln))
        trans = np.zeros((3, 3))
        for a, b in zip(seg[:-1], seg[1:]):
            trans[a, b] += 1
        trans = trans / np.maximum(trans.sum(axis=1, keepdims=True), 1)

        mom7 = np.full(n, np.nan); mom7[42:] = close[42:] / close[:-42] - 1.0
        rev1 = np.full(n, np.nan); rev1[6:] = -(close[6:] / close[:-6] - 1.0)
        fund_i = feats.index("funding_rate_zscore") if "funding_rate_zscore" in feats else None

        for r in (0, 1, 2):
            m = (lab == r) & (np.arange(n) >= s) & (np.arange(n) < e)
            nseg = int(m.sum())
            rlens = [ln for v, ln in runs if v == r]
            yr = y[m]
            mu, sd = float(np.nanmean(yr)) * 1e4, float(np.nanstd(yr)) * 1e4
            tstat = mu / (sd / math.sqrt(max(1, (~np.isnan(yr)).sum())))
            rep["labels_pct"][NAMES[r]] = round(100 * nseg / (e - s), 1)
            rep["durations"][NAMES[r]] = {"mean_bars": round(float(np.mean(rlens)), 1) if rlens else 0,
                                          "median_bars": float(np.median(rlens)) if rlens else 0}
            rep["target"][NAMES[r]] = {
                "n": nseg, "fwd16h_mean_bps": round(mu, 1),
                "fwd16h_vol_bps": round(sd, 1), "t": round(tstat, 2),
                "skew": round(float(stats.skew(yr[~np.isnan(yr)])), 2),
                "kurt": round(float(stats.kurtosis(yr[~np.isnan(yr)])), 1)}
            rep["signal_ic"][NAMES[r]] = {
                "momentum_7d": round(_rank_ic(mom7[m], yr), 4),
                "reversal_24h": round(_rank_ic(rev1[m], yr), 4)}
            # top-5 features by |IC| inside this regime (design era only)
            ics = []
            for i, f in enumerate(feats):
                ic = _rank_ic(X[m][:, i], yr)
                if np.isfinite(ic):
                    ics.append((abs(ic), ic, f))
            ics.sort(reverse=True)
            rep["top_features"][NAMES[r]] = [(f, round(ic, 3)) for _, ic, f in ics[:5]]
            # derivatives cut: fwd return by funding-zscore quartile
            if fund_i is not None and nseg > 400:
                fz = X[m][:, fund_i]
                q = pd.qcut(pd.Series(fz), 4, labels=False, duplicates="drop").to_numpy()
                rep["funding_cut"][NAMES[r]] = {
                    f"q{int(k)+1}": round(float(np.nanmean(yr[q == k])) * 1e4, 1)
                    for k in np.unique(q[~np.isnan(q)])}
        rep["transition"] = {NAMES[a]: {NAMES[b]: round(float(trans[a, b]), 3)
                                        for b in range(3)} for a in range(3)}
        if gmm_regime is not None:
            ct = {}
            for r in (0, 1, 2):
                m = (lab == r) & (np.arange(n) >= s) & (np.arange(n) < e)
                vals, cnts = np.unique(gmm_regime[m], return_counts=True)
                top = vals[np.argmax(cnts)] if len(vals) else None
                ct[NAMES[r]] = {"dominant_gmm_cluster": int(top) if top is not None else None,
                                "share": round(float(cnts.max() / max(1, cnts.sum())), 2) if len(cnts) else None}
            rep["gmm_agreement"] = ct

        for r in ("peace", "bull", "bear"):
            t = rep["target"][r]
            print(f"  {r:<6} {rep['labels_pct'][r]:5.1f}% of bars | dur~{rep['durations'][r]['mean_bars']}b | "
                  f"fwd16h {t['fwd16h_mean_bps']:+.0f}bps (t={t['t']:+.2f}) vol={t['fwd16h_vol_bps']:.0f} "
                  f"skew={t['skew']} kurt={t['kurt']}", flush=True)
            print(f"         mom_ic={rep['signal_ic'][r]['momentum_7d']:+.3f} "
                  f"rev_ic={rep['signal_ic'][r]['reversal_24h']:+.3f} | "
                  f"top: {rep['top_features'][r][:3]}", flush=True)
            if r in rep["funding_cut"]:
                print(f"         funding-quartile fwd bps: {rep['funding_cut'][r]}", flush=True)
        report[asset] = rep

    out = REPO / "training" / "reports" / "regime_lab_eda.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nEDA -> {out}", flush=True)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["eda", "select", "assemble"], required=True)
    ap.add_argument("--assets", default="BTC,ETH,SOL")
    args = ap.parse_args()
    assets = [a.strip().upper() for a in args.assets.split(",")]
    if args.stage == "eda":
        stage_eda(assets)
    else:
        print(f"stage {args.stage}: built after EDA lands (P232: measure first)")
        return 1


if __name__ == "__main__":
    sys.exit(main() or 0)
