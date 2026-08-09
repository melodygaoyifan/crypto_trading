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
CELL_CANDIDATES = {
    "perp": {
        "bull": [("flat", {}), ("hold", {}),
                 ("ridge_long", {"alpha": [10.0, 30.0, 100.0]}),
                 ("dip_buy", {"thr": [0.02, 0.04]})],
        "bear": [("flat", {}), ("ridge_defensive", {"alpha": [10.0, 30.0, 100.0]}),
                 ("funding_short", {"thr": [0.5, 1.0]})],
        "peace": [("flat", {}), ("funding_contrarian", {"thr": [0.5, 1.0]}),
                  ("meanrev", {"thr": [0.02, 0.04]}), ("ar_p", {"p": [3, 6]})],
    },
    "spot": {
        "bull": [("flat", {}), ("hold", {}),
                 ("ridge_long", {"alpha": [30.0, 100.0]}),
                 ("dip_buy", {"thr": [0.02, 0.04]})],
        # long/flat in a bear: hold is the honest losing reference; ridge_long
        # is exit-timing (long only when the forecast is strongly positive).
        "bear": [("flat", {}), ("hold", {}),
                 ("ridge_long", {"alpha": [30.0, 100.0]})],
        "peace": [("flat", {}), ("funding_long", {"thr": [0.5, 1.0]}),
                  ("meanrev_long", {"thr": [0.02, 0.04]})],
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
    dates = pd.to_datetime(px["timestamp"]).dt.date
    rate = dates.map(daily).to_numpy(dtype=float)[:n]
    if len(rate) < n:
        rate = np.pad(rate, (0, n - len(rate)))
    return np.nan_to_num(rate, nan=0.0) / 2.0


def _ctx(asset):
    X, targets, close, gmm, feats = load_asset(asset)
    n = len(close)
    lab = regime_labels(close)
    fz = X[:, feats.index("funding_rate_zscore")] if "funding_rate_zscore" in feats else np.zeros(n)
    ret6 = np.full(n, np.nan); ret6[6:] = close[6:] / close[:-6] - 1.0
    lr1 = np.full(n, np.nan); lr1[1:] = np.log(close[1:] / close[:-1])
    return dict(X=X, y=targets["ret"], close=close, lab=lab, fz=fz,
                ret6=ret6, lr1=lr1, n=n, feats=feats,
                carry_rate=_bar_carry_rate(asset, n))


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


def stage_select(assets, tag):
    s, e = DESIGN_ERA
    all_out = {}
    for asset in assets:
        ctx = _ctx(asset); ctx["asset"] = asset
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
                out[instrument][regime] = {"table": rows, "winner": rows[0]}
                w = rows[0]
                print(f"  {instrument:<4} {regime:<6} winner={w['name']:<30} "
                      f"cv_pnl={w['cv_pnl_pct']:+.1f}% cv_sh={w['cv_sharpe']:+.2f} "
                      f"train={w['train_sharpe']:+.2f} gap={w['overfit_gap']:+.2f} "
                      f"| runner-up {rows[1]['name']} "
                      f"cv_pnl={rows[1]['cv_pnl_pct']:+.1f}%", flush=True)
        all_out[asset] = out
    rpt = REPO / "training" / "reports" / f"regime_lab_select_{tag}.json"
    payload = {"results": all_out, "provenance": provenance_stamp(
        config={"design_era": DESIGN_ERA, "candidates": {
            k: [c[0] for c in v] for k, v in CELL_CANDIDATES.items()}})}
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


def stage_assemble(assets, tag):
    rpt_in = REPO / "training" / "reports" / f"regime_lab_select_{tag}.json"
    sel = json.loads(rpt_in.read_text(encoding="utf-8"))["results"]
    ds, de = DESIGN_ERA
    out = {}
    for asset in assets:
        ctx = _ctx(asset); ctx["asset"] = asset
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
            print(f"  design: pnl={row['train_pnl_pct']:+.1f}% "
                  f"sharpe={row['train_sharpe']:+.2f} | "
                  f"VALIDATION: pnl={row['test_pnl_pct']:+.1f}% "
                  f"sharpe={row['test_sharpe']:+.2f} (B&H {bh_pnl:+.1f}%) "
                  f"train-test gap={row['train_test_gap']:+.2f}", flush=True)

            def run_fn(params, window, cost_mult, _w=winners, _i=instrument):
                seg = assembled_series(ctx, _w, window[0], window[1],
                                       cost_mult=cost_mult,
                                       sma_w=params.get("sma_w", SMA_W),
                                       mom_w=params.get("mom_w", MOM_W),
                                       instrument=_i)
                return seg_metrics(seg)

            battery = robustness_battery(
                run_fn, {"sma_w": SMA_W, "mom_w": MOM_W},
                {"sma_w": [150, 250], "mom_w": [360, 720]},
                {"validation": (VALIDATION_ERA_START, n), "design": (ds, de)})
            print(f"  robustness flags: {battery['flags'] or 'NONE'}", flush=True)
            out[asset][instrument] = {"winners": winners, "report_row": row,
                                      "bh_validation_pnl_pct": bh_pnl,
                                      "battery": battery}
    rpt = REPO / "training" / "reports" / f"regime_lab_assemble_{tag}.json"
    rpt.write_text(json.dumps({"results": out, "provenance": provenance_stamp()},
                              indent=1, default=str), encoding="utf-8")
    print(f"\nStage 3 -> {rpt}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["eda", "select", "assemble"], required=True)
    ap.add_argument("--assets", default="BTC,ETH,SOL")
    ap.add_argument("--tag", default="p244")
    args = ap.parse_args()
    assets = [a.strip().upper() for a in args.assets.split(",")]
    if args.stage == "eda":
        stage_eda(assets)
    elif args.stage == "select":
        stage_select(assets, args.tag)
    elif args.stage == "assemble":
        stage_assemble(assets, args.tag)


if __name__ == "__main__":
    sys.exit(main() or 0)
