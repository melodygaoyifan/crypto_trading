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
def stage_eda(assets, engineered=False):
    report = {}
    for asset in assets:
        X, targets, close, gmm_regime, feats = load_asset(asset)
        if engineered:
            # [P250] EDA on the ENRICHED set: the per-regime feature
            # rankings must see the screened engineered features too, or
            # the zoo trains on features the EDA never profiled.
            _c = _ctx(asset); _c["asset"] = asset
            from training.feature_lab import build_screened
            _F, _names, _ = build_screened(_c)
            X = np.column_stack([X, _F])
            feats = list(feats) + _names
            print(f"  [feature-lab] EDA on enriched set (+{len(_names)})", flush=True)
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
        fzc = _causal_funding_z(asset, n)   # [P247-F1] causal, never the parquet feature

        for r in (0, 1, 2):
            m = (lab == r) & (np.arange(n) >= s) & (np.arange(n) < e)
            nseg = int(m.sum())
            rlens = [ln for v, ln in runs if v == r]
            yr = y[m]
            mu, sd = float(np.nanmean(yr)) * 1e4, float(np.nanstd(yr)) * 1e4
            # [P247-F4] 16h labels sampled every 4h overlap 4x -> n_eff = n/4.
            # The uncorrected t repeated the exact error P231 fixed in
            # agent_ic_review; this tool now corrects it at the source.
            n_eff = max(1.0, (~np.isnan(yr)).sum() / H)
            tstat = mu / (sd / math.sqrt(n_eff))
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
            if nseg > 400:
                fz = fzc[m]
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


# =====================================================================
# Stage 2 — per-regime model selection + tuning (design era ONLY)
# =====================================================================
from training.splits import (  # noqa: E402
    DESIGN_ERA, VALIDATION_ERA_START, purged_folds, record_window_usage,
)
from training.provenance import provenance_stamp  # noqa: E402
from training.eval_report import standard_row, robustness_battery, seg_metrics  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

DI, DEADBAND, REFIT = 4, 0.25, 42
from training.train_supervised_full import COST_BPS as _COST  # noqa: E402

# EDA-prescribed candidate grids per regime cell. Bear keeps BOTH the flat
# default (EDA: no significant short drift anywhere) and a conditional
# short so the data rules.
# [P246-E1] The 6-cell matrix: {bull,bear,peace} x {perp,spot}. Each
# instrument gets its TRUE economics and its own candidate lists — half the
# perp candidates are unexpressible on spot (no shorts, no carry), and
# spot's cost level (~20-26bps/side vs perp 3) makes low-turnover
# candidates the only sane entrants. flat is universal: every cell must be
# allowed to conclude "no model earns a position here".
# [P250] The zoo is the FULL ladder per cell — rules, then linear (ridge/
# elastic net), then trees (LightGBM), then nets (small MLP), each with a
# cell-appropriate position CLIP (long-only in bull/spot cells, short-only
# defensive in bears, full-range in peace). Capacity must earn its keep:
# the per-cell table carries every rung's train/CV/overfit-gap and trial
# counts are printed for the DSR line. ridge_long/ridge_defensive remain
# as aliases of mdl_long/mdl_short(family=ridge) for the export/probe.
CELL_CANDIDATES = {
    "perp": {
        "bull": [("flat", {}), ("hold", {}),
                 ("dip_buy", {"thr": [0.02, 0.04]}),
                 ("mdl_long", {"family": ["ridge"], "alpha": [30.0, 100.0]}),
                 ("mdl_long", {"family": ["enet"]}),
                 ("mdl_long", {"family": ["lgbm"]}),
                 ("mdl_long", {"family": ["mlp"]})],
        "bear": [("flat", {}), ("funding_short", {"thr": [0.5, 1.0]}),
                 ("mdl_short", {"family": ["ridge"], "alpha": [10.0, 30.0]}),
                 ("mdl_short", {"family": ["enet"]}),
                 ("mdl_short", {"family": ["lgbm"]}),
                 ("mdl_short", {"family": ["mlp"]})],
        "peace": [("flat", {}), ("funding_contrarian", {"thr": [0.5, 1.0]}),
                  ("meanrev", {"thr": [0.02, 0.04]}), ("ar_p", {"p": [3, 6]}),
                  ("mdl_full", {"family": ["ridge"], "alpha": [30.0]}),
                  ("mdl_full", {"family": ["lgbm"]}),
                  ("mdl_full", {"family": ["mlp"]})],
    },
    "spot": {
        "bull": [("flat", {}), ("hold", {}),
                 ("dip_buy", {"thr": [0.02, 0.04]}),
                 ("mdl_long", {"family": ["ridge"], "alpha": [100.0]}),
                 ("mdl_long", {"family": ["lgbm"]})],
        # long/flat in a bear: hold is the honest losing reference; models
        # are exit-timing (long only on a strongly positive forecast).
        "bear": [("flat", {}), ("hold", {}),
                 ("mdl_long", {"family": ["ridge"], "alpha": [100.0]})],
        "peace": [("flat", {}), ("funding_long", {"thr": [0.5, 1.0]}),
                  ("meanrev_long", {"thr": [0.02, 0.04]}),
                  ("mdl_long", {"family": ["lgbm"]})],
    },
}
REGIME_ID = {"peace": 0, "bull": 1, "bear": 2}

# Instrument economics. Spot = Kraken maker-first + slippage (per side,
# bps); no shorts; no funding carry. Perp = Coinbase CDE taker + slip,
# carry credited/charged. Break-even edge per round trip = 2x these.
INSTRUMENTS = {
    "perp": {"cost_bps": None, "long_only": False, "carry": True},   # per-asset _COST
    "spot": {"cost_bps": {"BTC": 20.0, "ETH": 22.0, "SOL": 26.0},
             "long_only": True, "carry": False},
}


def _grid(spec):
    keys = list(spec)
    if not keys:
        return [{}]
    out = [{}]
    for k in keys:
        out = [{**d, k: v} for d in out for v in spec[k]]
    return out


from training.train_supervised_full import DATA_DIR as _DATA_DIR  # noqa: E402
_FUND_DIR = REPO / "training" / "training_data" / "coinglass_history"


def _causal_funding_z(asset, n, window=30):
    """[P247-F1] CAUSAL funding z-score: bars on day D read day D-1's close
    rate, z-scored over a trailing 30d window. Replaces the parquet's
    funding_rate_zscore for SIGNAL use — that feature carries up to 16h of
    look-ahead (daily rows are stamped at day-OPEN while funding_close is
    the day's LAST 16:00 UTC event; merge_asof backward then hands bars at
    00:00-12:00 the future print). Found by the P247 fresh-eyes review;
    third instance of the P164/P221 timestamp-leak class."""
    p = _FUND_DIR / f"{asset}_funding_1d.parquet"
    if not p.exists():
        return np.zeros(n)
    px = pd.read_parquet(_DATA_DIR / f"{asset}_4H_full.parquet", columns=["timestamp"])
    f = pd.read_parquet(p)
    daily = f.assign(d=pd.to_datetime(f["timestamp"]).dt.date).set_index("d")["funding_close"]
    z = (daily - daily.rolling(window).mean()) / daily.rolling(window).std()
    z = z.shift(1)          # <- the fix: previous day's completed value only
    dates = pd.to_datetime(px["timestamp"]).dt.date
    out = dates.map(z).to_numpy(dtype=float)[:n]
    if len(out) < n:
        out = np.pad(out, (0, n - len(out)))
    return np.nan_to_num(out, nan=0.0)


def _bar_carry_rate(asset, n):
    """[P245] Per-4H-bar funding rate (FRACTION; positive -> longs PAY).

    On a perp, realized gain = price PnL + funding carry — every prior
    evaluation credited price PnL only, systematically understating
    strategies short during high funding and overstating longs held
    through it. Source: Binance Vision daily funding history (full 2020->
    now) as a PROXY for the CDE contract — P218 measured the two venues'
    signs can differ at a moment in time, so carry here is an estimate,
    not the venue's ledger. Daily close rate covers 3 events/day spread
    over 6 bars -> rate/2 per bar."""
    p = _FUND_DIR / f"{asset}_funding_1d.parquet"
    if not p.exists():
        return np.zeros(n)
    px = pd.read_parquet(_DATA_DIR / f"{asset}_4H_full.parquet", columns=["timestamp"])
    f = pd.read_parquet(p)
    daily = f.assign(d=pd.to_datetime(f["timestamp"]).dt.date).set_index("d")["funding_close"]
    # [P247-F1] previous-day rate as the accrual estimate — the same-day map
    # read the day's 16:00 print from its 00:00-12:00 bars (look-ahead).
    daily = daily.shift(1)
    dates = pd.to_datetime(px["timestamp"]).dt.date
    rate = dates.map(daily).to_numpy(dtype=float)[:n]
    if len(rate) < n:
        rate = np.pad(rate, (0, n - len(rate)))
    return np.nan_to_num(rate, nan=0.0) / 2.0


def _ctx(asset):
    X, targets, close, gmm, feats = load_asset(asset)
    n = len(close)
    lab = regime_labels(close)
    fz = _causal_funding_z(asset, n)   # [P247-F1] never the leaked parquet feature
    # [P250-F1b] The P247 leak's THIRD tentacle: the parquet's
    # funding_rate_zscore column sat in X itself, so every MODEL cell
    # (ridge/enet/lgbm/mlp — including the deployed SOL bear export) was
    # training on the 16h look-ahead, and the feature lab was deriving
    # crosses/transforms of it. Replace the column IN PLACE with the causal
    # series — same semantic, honest timing.
    if "funding_rate_zscore" in feats:
        X = X.copy()
        X[:, feats.index("funding_rate_zscore")] = fz
    ret6 = np.full(n, np.nan); ret6[6:] = close[6:] / close[:-6] - 1.0
    lr1 = np.full(n, np.nan); lr1[1:] = np.log(close[1:] / close[:-1])
    return dict(X=X, y=targets["ret"], close=close, lab=lab, fz=fz,
                ret6=ret6, lr1=lr1, n=n, feats=feats,
                carry_rate=_bar_carry_rate(asset, n))


def _make_model(family, params):
    """[P250] Family dispatch for the per-cell ladder. Returns
    (needs_scaling, model, refit_bars). Tree/net families refit monthly
    (180 bars) to bound walk-forward cost; linear refits weekly (42)."""
    if family == "ridge":
        return True, Ridge(alpha=params.get("alpha", 30.0)), REFIT
    if family == "enet":
        from sklearn.linear_model import ElasticNet
        return True, ElasticNet(alpha=1e-4, l1_ratio=0.5, max_iter=2000), REFIT
    if family == "lgbm":
        import lightgbm as lgb
        return False, lgb.LGBMRegressor(
            n_estimators=150, num_leaves=15, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=7,
            verbosity=-1), 180
    if family == "mlp":
        from sklearn.neural_network import MLPRegressor
        return True, MLPRegressor(hidden_layer_sizes=(32,), alpha=1e-2,
                                  max_iter=200, early_stopping=True,
                                  random_state=7), 180
    raise ValueError(family)


def _model_z(ctx, family, params, regime_id, s, e, fit_lt=None):
    """Generic walk-forward z over [s,e): refit every family-appropriate
    cadence on REGIME-MATCHED history (the per-cell training set)."""
    X, y, lab, n = ctx["X"], ctx["y"], ctx["lab"], ctx["n"]
    z = np.full(n, np.nan)
    needs_scale, _, refit = _make_model(family, params)
    for t0 in range(s, e, refit):
        lim = t0 - H if fit_lt is None else min(t0 - H, fit_lt)
        idx = np.where((lab[:lim] == regime_id)
                       & ~np.isnan(X[:lim]).any(axis=1) & ~np.isnan(y[:lim]))[0]
        if len(idx) < 400:
            continue
        _, model, _ = _make_model(family, params)
        if needs_scale:
            sc = StandardScaler().fit(X[idx])
            model.fit(sc.transform(X[idx]), y[idx])
            preds_tr = model.predict(sc.transform(X[idx]))
            t1 = min(t0 + refit, e)
            preds = model.predict(sc.transform(np.nan_to_num(X[t0:t1])))
        else:
            model.fit(X[idx], y[idx])
            preds_tr = model.predict(X[idx])
            t1 = min(t0 + refit, e)
            preds = model.predict(np.nan_to_num(X[t0:t1]))
        sig = float(np.std(preds_tr)) or 1e-9
        z[t0:t1] = preds / sig
    return z


def _ridge_z(ctx, regime_id, s, e, alpha, fit_lt=None):
    """Walk-forward ridge z over [s,e), trained on REGIME-MATCHED history.
    fit_lt caps training rows (Stage-2 in-design discipline)."""
    X, y, lab, n = ctx["X"], ctx["y"], ctx["lab"], ctx["n"]
    z = np.full(n, np.nan)
    for t0 in range(s, e, REFIT):
        lim = t0 - H if fit_lt is None else min(t0 - H, fit_lt)
        idx = np.where((lab[:lim] == regime_id)
                       & ~np.isnan(X[:lim]).any(axis=1) & ~np.isnan(y[:lim]))[0]
        if len(idx) < 400:
            continue
        sc = StandardScaler().fit(X[idx])
        m = Ridge(alpha=alpha).fit(sc.transform(X[idx]), y[idx])
        sig = float(np.std(m.predict(sc.transform(X[idx])))) or 1e-9
        t1 = min(t0 + REFIT, e)
        z[t0:t1] = m.predict(sc.transform(np.nan_to_num(X[t0:t1]))) / sig
    return z


def _ar_z(ctx, s, e, p):
    """Walk-forward AR(p) on 4h log returns; z of the next-bar forecast."""
    lr, n = ctx["lr1"], ctx["n"]
    z = np.full(n, np.nan)
    for t0 in range(s, e, REFIT):
        hist = lr[:t0]
        hist = hist[~np.isnan(hist)]
        if len(hist) < 500:
            continue
        Y = hist[p:]
        L = np.column_stack([hist[p - k - 1:len(hist) - k - 1] for k in range(p)])
        coef, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(Y)), L]), Y, rcond=None)
        sig = float(np.std(Y)) or 1e-9
        t1 = min(t0 + REFIT, e)
        for t in range(t0, t1):
            lags = lr[t - p:t][::-1]
            if np.isnan(lags).any():
                continue
            z[t] = (coef[0] + lags @ coef[1:]) / sig
    return z


def desired_positions(kind, params, ctx, regime_id, s, e, fit_lt=None):
    """Raw desired position per bar in [s,e) for one cell candidate,
    BEFORE regime masking and DI cadence."""
    n = ctx["n"]
    pos = np.zeros(n)
    if kind == "flat":
        pass
    elif kind == "hold":
        pos[s:e] = 1.0
    elif kind == "dip_buy":
        dip = ctx["ret6"] < -params["thr"]
        act = np.zeros(n, dtype=bool)
        for t in np.where(dip[s:e])[0] + s:
            act[t:min(t + 8, n)] = True
        pos[s:e] = act[s:e].astype(float)
    elif kind == "funding_short":
        pos[s:e] = np.where(ctx["fz"][s:e] > params["thr"], -1.0, 0.0)
    elif kind == "funding_contrarian":
        fz = ctx["fz"][s:e]
        pos[s:e] = np.where(fz < -params["thr"], 1.0,
                            np.where(fz > params["thr"], -1.0, 0.0))
    elif kind == "meanrev":
        r = ctx["ret6"][s:e]
        pos[s:e] = np.where(r < -params["thr"], 1.0,
                            np.where(r > params["thr"], -1.0, 0.0))
    elif kind == "funding_long":       # spot: long when funding deeply negative
        pos[s:e] = np.where(ctx["fz"][s:e] < -params["thr"], 1.0, 0.0)
    elif kind == "meanrev_long":       # spot: buy dips only, never short
        pos[s:e] = np.where(ctx["ret6"][s:e] < -params["thr"], 1.0, 0.0)
    elif kind in ("ridge_long", "ridge_defensive"):
        z = (_ridge_z(ctx, regime_id, s, e, params["alpha"], fit_lt))[s:e]
        raw = np.where(np.isnan(z) | (np.abs(z) < DEADBAND), 0.0, np.clip(z, -1, 1))
        pos[s:e] = np.clip(raw, 0.0, 1.0) if kind == "ridge_long" else np.clip(raw, -1.0, 0.0)
    elif kind in ("mdl_long", "mdl_short", "mdl_full"):
        # [P250] generic family ladder: same z contract, cell-appropriate clip
        z = (_model_z(ctx, params["family"], params, regime_id, s, e, fit_lt))[s:e]
        raw = np.where(np.isnan(z) | (np.abs(z) < DEADBAND), 0.0, np.clip(z, -1, 1))
        if kind == "mdl_long":
            pos[s:e] = np.clip(raw, 0.0, 1.0)
        elif kind == "mdl_short":
            pos[s:e] = np.clip(raw, -1.0, 0.0)
        else:
            pos[s:e] = raw
    elif kind == "ar_p":
        z = (_ar_z(ctx, s, e, params["p"]))[s:e]
        pos[s:e] = np.where(np.isnan(z) | (np.abs(z) < DEADBAND), 0.0, np.clip(z, -1, 1))
    else:
        raise ValueError(kind)
    return pos


def cell_series(kind, params, ctx, regime, s, e, cost_mult=1.0, fit_lt=None,
                lab_override=None, instrument="perp"):
    """After-cost per-bar PnL of one cell candidate active ONLY in its
    regime's bars, DI cadence applied, costs on every position change.
    [P246-E1] instrument-true economics: spot = long/flat clip, Kraken-level
    costs, NO carry; perp = +-1, CDE costs, funding carry."""
    inst = INSTRUMENTS[instrument]
    lab = ctx["lab"] if lab_override is None else lab_override
    rid = REGIME_ID[regime]
    want = desired_positions(kind, params, ctx, rid, s, e, fit_lt)
    if inst["long_only"]:
        want = np.clip(want, 0.0, 1.0)
    want = np.where(lab == rid, want, 0.0)
    pos = np.zeros(ctx["n"])
    last = 0.0
    for i in range(s, e):
        if (i - s) % DI == 0:
            last = want[i]
        pos[i] = last
    close = ctx["close"]
    ret = np.zeros(ctx["n"]); ret[1:] = close[1:] / close[:-1] - 1.0
    strat = np.zeros(ctx["n"]); strat[1:] = pos[:-1] * ret[1:]
    cost_bps = (_COST[ctx["asset"]] if inst["cost_bps"] is None
                else inst["cost_bps"][ctx["asset"]])
    cost = np.zeros(ctx["n"])
    cost[1:] = np.abs(np.diff(pos)) * cost_bps * cost_mult / 1e4
    # [P245] perp funding carry: shorts COLLECT when funding is positive.
    carry = np.zeros(ctx["n"])
    if inst["carry"]:
        carry[1:] = -pos[:-1] * ctx["carry_rate"][1:]
    return (strat - cost + carry)[s:e]


def _enrich(ctx):
    """[P250] Append the feature lab's screened engineered features —
    generated, causality-gated, screened (design era only) — so the model
    zoo trains on the enriched set. Deterministic and self-contained."""
    from training.feature_lab import build_screened
    F, names, _ = build_screened(ctx)
    ctx["X"] = np.column_stack([ctx["X"], F])
    ctx["feats"] = list(ctx["feats"]) + names
    print(f"  [feature-lab] +{len(names)} engineered features "
          f"(total {ctx['X'].shape[1]})", flush=True)
    return ctx


def stage_select(assets, tag, engineered=False):
    s, e = DESIGN_ERA
    all_out = {}
    for asset in assets:
        ctx = _ctx(asset); ctx["asset"] = asset
        if engineered:
            ctx = _enrich(ctx)
        record_window_usage(f"regime_lab:{tag}", asset, s, e, "design")
        print(f"\n########## {asset} Stage 2 (design era [{s},{e})) ##########", flush=True)
        out = {}
        for instrument, cells in CELL_CANDIDATES.items():
            out[instrument] = {}
            for regime, cands in cells.items():
                rows = []
                for kind, spec in cands:
                    for params in _grid(spec):
                        cv_pnl, cv_sh = [], []
                        for tr, va in purged_folds(s, e):
                            fit_lt = int(va[0])
                            seg = cell_series(kind, params, ctx, regime,
                                              int(va[0]), int(va[-1] + 1),
                                              fit_lt=fit_lt, instrument=instrument)
                            m = seg_metrics(seg)
                            cv_pnl.append(m["pnl_pct"]); cv_sh.append(m["sharpe"])
                        train_seg = cell_series(kind, params, ctx, regime, s, e,
                                                instrument=instrument)
                        row = standard_row(f"{kind}{params or ''}", train_seg,
                                           float(np.mean(cv_sh)) if cv_sh else 0.0)
                        # [P245] objective = REALIZED after-cost gain (incl.
                        # carry); risk stats reported, not deciding.
                        row["cv_pnl_pct"] = round(float(np.mean(cv_pnl)), 2) if cv_pnl else 0.0
                        row.update({"kind": kind, "params": params})
                        rows.append(row)
                rows.sort(key=lambda r: -r["cv_pnl_pct"])
                out[instrument][regime] = {"table": rows, "winner": rows[0],
                                           "trials": len(rows)}
                w = rows[0]
                print(f"  {instrument:<4} {regime:<6} winner={w['name']:<30} "
                      f"cv_pnl={w['cv_pnl_pct']:+.1f}% cv_sh={w['cv_sharpe']:+.2f} "
                      f"train={w['train_sharpe']:+.2f} gap={w['overfit_gap']:+.2f} "
                      f"| runner-up {rows[1]['name']} "
                      f"cv_pnl={rows[1]['cv_pnl_pct']:+.1f}% "
                      f"[{len(rows)} trials]", flush=True)
        all_out[asset] = out
    rpt = REPO / "training" / "reports" / f"regime_lab_select_{tag}.json"
    _data_files = [p for a in assets for p in
                   (_DATA_DIR / f"{a}_4H_full.parquet",
                    _FUND_DIR / f"{a}_funding_1d.parquet")]
    payload = {"results": all_out, "provenance": provenance_stamp(
        data_files=_data_files,
        config={"design_era": DESIGN_ERA, "candidates": {
            k: {r: [c[0] for c in v] for r, v in cells.items()}
            for k, cells in CELL_CANDIDATES.items()}})}
    rpt.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    print(f"\nStage 2 -> {rpt}", flush=True)
    return all_out


# =====================================================================
# Stage 3 — assembly, ONE validation shot, robustness battery
# =====================================================================
def assembled_series(ctx, winners, s, e, cost_mult=1.0, sma_w=SMA_W, mom_w=MOM_W,
                     instrument="perp"):
    close = ctx["close"]
    sma = pd.Series(close).rolling(sma_w).mean().to_numpy()
    mom = np.full(ctx["n"], np.nan)
    mom[mom_w:] = close[mom_w:] / close[:-mom_w] - 1.0
    lab = np.zeros(ctx["n"], dtype=int)
    above, up = close > sma, mom > 0
    lab[above & up] = 1
    lab[~above & ~up & ~np.isnan(mom)] = 2
    lab[np.isnan(sma) | np.isnan(mom)] = 0
    total = np.zeros(e - s)
    for regime in ("peace", "bull", "bear"):
        w = winners[regime]
        total += cell_series(w["kind"], w["params"], ctx, regime, s, e,
                             cost_mult=cost_mult, lab_override=lab,
                             instrument=instrument)
    return total


def stage_assemble(assets, tag, engineered=False):
    rpt_in = REPO / "training" / "reports" / f"regime_lab_select_{tag}.json"
    sel = json.loads(rpt_in.read_text(encoding="utf-8"))["results"]
    ds, de = DESIGN_ERA
    out = {}
    for asset in assets:
        ctx = _ctx(asset); ctx["asset"] = asset
        if engineered:
            ctx = _enrich(ctx)
        n = ctx["n"]
        out[asset] = {}
        prior = record_window_usage(f"regime_lab:{tag}", asset,
                                    VALIDATION_ERA_START, n, "validation")
        if prior:
            print(f"  !! VALIDATION SPEND WARNING ({asset}): window already read "
                  f"by {prior} prior experiment(s) — discount accordingly", flush=True)
        for instrument in CELL_CANDIDATES:
            winners = {r: sel[asset][instrument][r]["winner"]
                       for r in ("peace", "bull", "bear")}
            # Floor rule: a cell whose best candidate has NEGATIVE design-era
            # CV realized gain deploys nothing — flat.
            for r, w in winners.items():
                metric = w.get("cv_pnl_pct", w["cv_sharpe"])
                if metric < 0:
                    print(f"  floor rule: {instrument}/{r} winner {w['name']} "
                          f"cv={metric:+.2f} < 0 -> replaced by flat", flush=True)
                    winners[r] = {"name": "flat(floored)", "kind": "flat",
                                  "params": {}, "cv_sharpe": 0.0, "cv_pnl_pct": 0.0}
            print(f"\n########## {asset} [{instrument}] Stage 3 ##########", flush=True)
            print(f"  winners: " + ", ".join(f"{r}={winners[r]['name']}"
                                             for r in ("bull", "bear", "peace")), flush=True)
            design_seg = assembled_series(ctx, winners, ds, de, instrument=instrument)
            val_seg = assembled_series(ctx, winners, VALIDATION_ERA_START, n,
                                       instrument=instrument)
            row = standard_row("assembled", design_seg,
                               seg_metrics(design_seg)["sharpe"], val_seg)
            close = ctx["close"]
            bh = (close[1:] / close[:-1] - 1)[VALIDATION_ERA_START:n - 1]
            bh_pnl = round(float(np.nansum(bh)) * 100, 2)
            # [P247-F2] The decisive ablation the review demanded: the assembly
            # must beat the ALREADY-KNOWN era-stable baseline (trend filter =
            # hold-bull + flat elsewhere), not just B&H. If the excess comes
            # from the trend legs, the assembly adds nothing new.
            trend_only = {r: {"name": "trend_only", "kind": ("hold" if r == "bull" else "flat"),
                              "params": {}} for r in ("bull", "bear", "peace")}
            tf_val = seg_metrics(assembled_series(ctx, trend_only,
                                                  VALIDATION_ERA_START, n,
                                                  instrument=instrument))
            print(f"  design: pnl={row['train_pnl_pct']:+.1f}% "
                  f"sharpe={row['train_sharpe']:+.2f} | "
                  f"VALIDATION: pnl={row['test_pnl_pct']:+.1f}% "
                  f"sharpe={row['test_sharpe']:+.2f} (B&H {bh_pnl:+.1f}%, "
                  f"TREND-ONLY {tf_val['pnl_pct']:+.1f}%) "
                  f"train-test gap={row['train_test_gap']:+.2f}", flush=True)

            def run_fn(params, window, cost_mult, _w=winners, _i=instrument):
                seg = assembled_series(ctx, _w, window[0], window[1],
                                       cost_mult=cost_mult,
                                       sma_w=params.get("sma_w", SMA_W),
                                       mom_w=params.get("mom_w", MOM_W),
                                       instrument=_i)
                return seg_metrics(seg)

            # [P247-F3] design window FIRST: param/cost perturbations run on
            # w0, and mining the validation window through the battery was
            # exactly the unledgered re-reading the review caught.
            # [P247-F5] pre-design era [800,3000) added — the third era the
            # docstring promised and the code never scored.
            battery = robustness_battery(
                run_fn, {"sma_w": SMA_W, "mom_w": MOM_W},
                {"sma_w": [150, 250], "mom_w": [360, 720]},
                {"design": (ds, de),
                 "validation": (VALIDATION_ERA_START, n),
                 "pre_design": (800, 3000)})
            print(f"  robustness flags: {battery['flags'] or 'NONE'} "
                  f"(pre-design era: {battery['windows']['pre_design']['pnl_pct']:+.1f}%)",
                  flush=True)
            out[asset][instrument] = {"winners": winners, "report_row": row,
                                      "bh_validation_pnl_pct": bh_pnl,
                                      "trend_only_validation": tf_val,
                                      "battery": battery}
    rpt = REPO / "training" / "reports" / f"regime_lab_assemble_{tag}.json"
    _data_files = [p for a in assets for p in
                   (_DATA_DIR / f"{a}_4H_full.parquet",
                    _FUND_DIR / f"{a}_funding_1d.parquet")]
    rpt.write_text(json.dumps({"results": out,
                               "provenance": provenance_stamp(data_files=_data_files)},
                              indent=1, default=str), encoding="utf-8")
    print(f"\nStage 3 -> {rpt}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["eda", "select", "assemble"], required=True)
    ap.add_argument("--assets", default="BTC,ETH,SOL")
    ap.add_argument("--tag", default="p244")
    ap.add_argument("--engineered", action="store_true",
                    help="[P250] enrich the feature matrix with the feature "
                         "lab's screened engineered features (generated -> "
                         "causality-gated -> screened, design era only)")
    args = ap.parse_args()
    assets = [a.strip().upper() for a in args.assets.split(",")]
    if args.stage == "eda":
        stage_eda(assets, engineered=args.engineered)
    elif args.stage == "select":
        stage_select(assets, args.tag, engineered=args.engineered)
    elif args.stage == "assemble":
        stage_assemble(assets, args.tag, engineered=args.engineered)


if __name__ == "__main__":
    sys.exit(main() or 0)
