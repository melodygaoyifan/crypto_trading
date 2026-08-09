"""End-to-end supervised training pipeline for BTC/ETH direction models.

The supervised sibling of train_drl_full.py, built so model selection follows
DATA PATTERN + PERFORMANCE (P242) rather than a default:

  Stage 1  Load 4H parquet (+ fv2 flow columns when present).
  Stage 2  Target engineering: raw 16h forward return AND vol-scaled return.
  Stage 3  Feature selection: redundancy prune (|rho|>0.95) + train-window
           |IC| ranking (recomputed per fold on that fold's train data only).
  Stage 4  Candidate zoo — CHOSEN BY the data-pattern diagnostics
           (training/reports/data_pattern_diagnostics.json, P242 B1):
             ridge_adaptive        incumbent: alpha=30, weekly refit
             ridge_volscaled      the dev-validated target hypothesis
             regime_ridge          per-GMM-regime ridge (the one measured
                                   structure: momentum IC flips sign by regime)
             hgb_small             capacity probe (diagnostics say it should
                                   lose — if it wins, the diagnostics were wrong)
             mlp_small             ditto, net flavor
             ens_ridge_hgb         z-mean ensemble
  Stage 5  Per-fold INNER walk-forward selection on the fold's train tail
           (last 20% of train), then the selected candidate runs walk-forward
           on the fold's val window. Selection never sees val.
  Stage 6  P182-style gate per fold: after-cost PnL must beat buy_and_hold
           AND sma_200bar on the val window, with a block-bootstrap Sharpe CI.
  Stage 7  Artifacts: models/supervised/{ASSET}/{tag}/results.json — same
           fold geometry as train_drl_full.py (val_ratio 0.15, gap 42,
           3 folds) so rows are DIRECTLY comparable with the TQC campaign.

Deployment note: the artifact of an adaptive model is its CONFIG (family,
hyperparams, refit cadence, feature recipe), not frozen weights — the weekly
refit IS the model. results.json records the config; a runtime consumer
refits from it.

Position contract (identical to the ridge_16h baseline in train_drl_full):
z-scored forecast, deadband 0.25, act every 4 bars, per-side cost =
venue fee + slippage.
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
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
DATA_DIR = REPO / "training" / "training_data" / "drl_training"
DIAG_PATH = REPO / "training" / "reports" / "data_pattern_diagnostics.json"

from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

H = 4                # 16h label on 4H bars
DI = 4               # act every 4 bars (deployment contract)
DEADBAND = 0.25
REFIT_EVERY = 42     # weekly
SEED = 7
# per-side cost bps = venue taker fee + slippage (coinbase nano sleeve)
COST_BPS = {"BTC": 6.0, "ETH": 8.0, "SOL": 13.0}
BARS_PER_YEAR = 6 * 365


# ---------------------------------------------------------------- stages 1-2
def load_asset(asset):
    df = pd.read_parquet(DATA_DIR / f"{asset}_4H_full.parquet")
    manifest = json.loads((REPO / "configs" / "feature_manifest.json").read_text(encoding="utf-8"))
    feats = [c for c in manifest["all_features"] if c in df.columns]
    feats += sorted(c for c in df.columns if c.startswith("fv2_") and c not in feats)
    close = df["close"].to_numpy(dtype=float)
    n = len(close)
    y_ret = np.full(n, np.nan)
    y_ret[:-H] = close[H:] / close[:-H] - 1.0
    vol = pd.Series(y_ret).rolling(180, min_periods=42).std().shift(1).to_numpy()
    y_vs = y_ret / np.where(vol > 0, vol, np.nan)
    regime = None
    rp = sorted(c for c in df.columns if c.startswith("regime_proba_"))
    if rp:
        regime = df[rp].to_numpy(dtype=float).argmax(axis=1)
    X = df[feats].to_numpy(dtype=float)
    return X, {"ret": y_ret, "volscaled": y_vs}, close, regime, feats


def fold_splits(n, n_folds=3, val_ratio=0.15, gap=42):
    """Same geometry as train_drl_full._get_fold_splits: sequential val
    windows walking back from the end; fold_1 = latest data."""
    val_size = int(n * val_ratio)
    out = {}
    for k in range(1, n_folds + 1):
        val_end = n - (k - 1) * val_size
        val_start = val_end - val_size
        train_end = val_start - gap
        if train_end < 3000:
            continue
        out[f"fold_{k}"] = (train_end, val_start, val_end)
    return out


# ---------------------------------------------------------------- stage 3
def select_features(X, y, train_end, feats):
    Xd, yd = X[:train_end], y[:train_end]
    m = ~(np.isnan(Xd).any(axis=1) | np.isnan(yd))
    Xd, yd = Xd[m], yd[m]
    sd = Xd.std(axis=0)
    live = [i for i in range(X.shape[1]) if sd[i] > 0]
    corr = np.corrcoef(Xd[:, live].T)
    dropped = set(i for i in range(X.shape[1]) if sd[i] == 0)
    for a in range(len(live)):
        for b in range(a + 1, len(live)):
            if live[b] in dropped or live[a] in dropped:
                continue
            if abs(corr[a, b]) > 0.95:
                dropped.add(live[b])
    keep = [i for i in range(X.shape[1]) if i not in dropped]
    ranks_y = np.argsort(np.argsort(yd))
    ics = []
    for i in keep:
        c = np.corrcoef(np.argsort(np.argsort(Xd[:, i])), ranks_y)[0, 1]
        ics.append(abs(c) if np.isfinite(c) else 0.0)
    order = [keep[i] for i in np.argsort(ics)[::-1]]
    return {"top24": order[:24], "pruned_all": keep}


# ---------------------------------------------------------------- stage 4 zoo
class Candidate:
    """A candidate is (family, target, feature set, refit cadence). fit()
    trains on history; predict() returns (raw_forecast, train_sigma)."""

    def __init__(self, name, family, target, fset, refit=REFIT_EVERY, params=None):
        self.name, self.family, self.target = name, family, target
        self.fset, self.refit, self.params = fset, refit, params or {}

    def fit_predict(self, Xtr, ytr, Xpred, regime_tr=None, regime_pred=None):
        if self.family == "ridge":
            sc = StandardScaler().fit(Xtr)
            m = Ridge(alpha=self.params.get("alpha", 30.0)).fit(sc.transform(Xtr), ytr)
            return m.predict(sc.transform(Xpred)), float(np.std(m.predict(sc.transform(Xtr))))
        if self.family == "regime_ridge":
            sc = StandardScaler().fit(Xtr)
            Xs, Xp = sc.transform(Xtr), sc.transform(Xpred)
            glob = Ridge(alpha=30.0).fit(Xs, ytr)
            sig = float(np.std(glob.predict(Xs))) or 1e-9
            per = {}
            for r in np.unique(regime_tr):
                mask = regime_tr == r
                if mask.sum() >= self.params.get("min_rows", 400):
                    per[r] = Ridge(alpha=30.0).fit(Xs[mask], ytr[mask])
            preds = glob.predict(Xp)
            for r, mod in per.items():
                mask = regime_pred == r
                if mask.any():
                    preds[mask] = mod.predict(Xp[mask])
            return preds, sig
        if self.family == "hgb":
            m = HistGradientBoostingRegressor(
                max_iter=self.params.get("iters", 100),
                max_depth=self.params.get("depth", 2),
                random_state=SEED).fit(Xtr, ytr)
            return m.predict(Xpred), float(np.std(m.predict(Xtr)))
        if self.family == "mlp":
            sc = StandardScaler().fit(Xtr)
            m = MLPRegressor(hidden_layer_sizes=(self.params.get("width", 32),),
                             alpha=self.params.get("alpha", 1e-2), max_iter=200,
                             early_stopping=True, random_state=SEED).fit(
                sc.transform(Xtr), ytr)
            return m.predict(sc.transform(Xpred)), float(np.std(m.predict(sc.transform(Xtr))))
        if self.family == "lgbm":
            import lightgbm as lgb
            m = lgb.LGBMRegressor(
                n_estimators=self.params.get("iters", 200),
                num_leaves=self.params.get("leaves", 15),
                max_depth=self.params.get("depth", 3),
                learning_rate=self.params.get("lr", 0.05),
                subsample=0.8, colsample_bytree=0.8,
                random_state=SEED, verbosity=-1).fit(Xtr, ytr)
            return m.predict(Xpred), float(np.std(m.predict(Xtr)))
        if self.family == "composite":
            # [P243] Regime-switched composite: the bear/chop leg is a ridge
            # forecast; the bull leg is applied at POSITION level (see
            # positions_from_z) — every fold showed nothing beats holding in
            # a bull and nothing beats the model in a bear, so the composite
            # uses a different model CLASS per regime instead of asking one
            # function to be both.
            sc = StandardScaler().fit(Xtr)
            m = Ridge(alpha=self.params.get("alpha", 30.0)).fit(sc.transform(Xtr), ytr)
            return m.predict(sc.transform(Xpred)), float(np.std(m.predict(sc.transform(Xtr))))
        raise ValueError(self.family)

    @property
    def is_composite(self):
        return self.family == "composite"


def build_zoo(diag, asset):
    """The zoo is derived from the diagnostics report — the P242 point.
    regime_ridge enters only when the measured regime dispersion supports it."""
    zoo = [
        Candidate("ridge_adaptive", "ridge", "ret", "pruned_all"),
        Candidate("ridge_volscaled", "ridge", "volscaled", "pruned_all"),
        Candidate("hgb_small", "hgb", "ret", "pruned_all", refit=180),
        Candidate("lgbm_small", "lgbm", "ret", "pruned_all", refit=180),
        Candidate("mlp_small", "mlp", "ret", "top24"),
        # [P243] Regime-switched composites (operator direction): bull regime
        # -> hold long, bear/chop regime -> the directional forecast. The
        # switch is close>SMA200 — runtime-available, no lookahead, and the
        # same signal the sma_200bar baseline uses, so the composite's edge
        # over that baseline is exactly the value of the bear-leg model.
        Candidate("composite_bull_ridge", "composite", "ret", "pruned_all"),
        Candidate("composite_bull_volscaled", "composite", "volscaled", "pruned_all"),
    ]
    disp = (diag.get(asset, {}).get("regime", {}) or {}).get("dispersion")
    if disp is not None and disp >= 0.04:
        zoo.append(Candidate("regime_ridge", "regime_ridge", "ret", "pruned_all"))
    return zoo


# ---------------------------------------------------------------- evaluation
def walk_forward_z(cand, X, y, regime, fsets, start, end):
    """Refit every cand.refit bars on ALL history before the prediction
    block (purged by H); emit z-scores for [start, end)."""
    fidx = fsets[cand.fset]
    Xf = X[:, fidx]
    z = np.full(len(y), np.nan)
    for t0 in range(start, end, cand.refit):
        pe = t0 - H
        m = ~(np.isnan(Xf[:pe]).any(axis=1) | np.isnan(y[:pe]))
        if m.sum() < 800:
            continue
        idx = np.where(m)[0]
        t1 = min(t0 + cand.refit, end)
        try:
            preds, sig = cand.fit_predict(
                Xf[idx], y[idx], np.nan_to_num(Xf[t0:t1]),
                regime_tr=None if regime is None else regime[idx],
                regime_pred=None if regime is None else regime[t0:t1])
        except Exception:
            continue
        sig = sig or 1e-9
        z[t0:t1] = preds / sig
    return z


def positions_from_z(z_seg, bull_seg=None):
    """bull_seg (composite only): boolean bull-regime flag aligned with
    z_seg. On decision bars a bull regime holds +1 regardless of the
    forecast; bear/chop uses the deadbanded z. Checked only on decision
    bars — the composite honors the same DI cadence as everything else."""
    pos = np.zeros(len(z_seg))
    last = 0.0
    for i in range(len(z_seg)):
        if i % DI == 0:
            if bull_seg is not None and bull_seg[i]:
                last = 1.0
            elif not np.isnan(z_seg[i]):
                last = 0.0 if abs(z_seg[i]) < DEADBAND else float(np.clip(z_seg[i], -1, 1))
            elif bull_seg is not None:
                last = 0.0
        pos[i] = last
    return pos


def bull_flag(close):
    sma = pd.Series(close).rolling(200).mean().to_numpy()
    return close > sma


def evaluate_segment(close, pos_full, cost_bps, s, e):
    ret = np.zeros(len(close)); ret[1:] = close[1:] / close[:-1] - 1.0
    strat = np.zeros(len(close)); strat[1:] = pos_full[:-1] * ret[1:]
    cost = np.zeros(len(close)); cost[1:] = np.abs(np.diff(pos_full)) * cost_bps / 1e4
    seg = (strat - cost)[s:e]
    sd = float(np.nanstd(seg))
    sharpe = float(np.nanmean(seg) / sd * math.sqrt(BARS_PER_YEAR)) if sd > 0 else 0.0
    trades = int((np.abs(np.diff(pos_full[s:e])) > 1e-9).sum())
    return {"pnl_pct": float(np.nansum(seg)) * 100, "sharpe": sharpe,
            "trades": trades,
            "cost_pct": float(np.nansum(cost[s:e])) * 100,
            "series": seg}


def block_bootstrap_ci(seg, n_boot=500, block=30):
    rng = np.random.default_rng(SEED)
    seg = seg[~np.isnan(seg)]
    if len(seg) < 2 * block:
        return None, None
    sharpes = []
    nblocks = int(np.ceil(len(seg) / block))
    for _ in range(n_boot):
        starts = rng.integers(0, len(seg) - block, nblocks)
        boot = np.concatenate([seg[s:s + block] for s in starts])[:len(seg)]
        sd = boot.std()
        if sd > 0:
            sharpes.append(boot.mean() / sd * math.sqrt(BARS_PER_YEAR))
    return (float(np.percentile(sharpes, 2.5)), float(np.percentile(sharpes, 97.5))) \
        if sharpes else (None, None)


def baseline_positions(name, close, s, e):
    pos = np.zeros(len(close))
    if name == "buy_and_hold":
        pos[s:e] = 1.0
    elif name == "sma_200bar":
        sma = pd.Series(close).rolling(200).mean().to_numpy()
        pos[s:e] = (close[s:e] > sma[s:e]).astype(float)
    return pos


# ---------------------------------------------------------------- main
def run_asset(asset, tag, diag):
    X, targets, close, regime, feats = load_asset(asset)
    n = len(close)
    print(f"\n########## {asset}: {n} bars, {len(feats)} features ##########", flush=True)
    zoo = build_zoo(diag, asset)
    print(f"  zoo ({len(zoo)}, diagnostics-chosen): {[c.name for c in zoo]}", flush=True)

    results = {"asset": asset, "tag": tag,
               "generated_at": datetime.now(timezone.utc).isoformat(),
               "position_contract": {"deadband": DEADBAND, "decision_interval": DI,
                                     "cost_bps_per_side": COST_BPS[asset],
                                     "refit_every_bars": REFIT_EVERY},
               "folds": {}}

    bull = bull_flag(close)
    fold_series = {}

    for fold, (train_end, val_start, val_end) in fold_splits(n).items():
        fsets = select_features(X, targets["ret"], train_end, feats)
        # ---- Stage 5a: inner selection on the train tail (never sees val)
        inner_start = train_end - int((train_end - 3000) * 0.2)
        table = {}
        for cand in zoo:
            y = targets[cand.target]
            z = walk_forward_z(cand, X, y, regime, fsets, inner_start, train_end)
            pos = np.zeros(n)
            pos[inner_start:train_end] = positions_from_z(
                z[inner_start:train_end],
                bull[inner_start:train_end] if cand.is_composite else None)
            ev = evaluate_segment(close, pos, COST_BPS[asset], inner_start, train_end)
            table[cand.name] = {"inner_pnl_pct": round(ev["pnl_pct"], 2),
                                "inner_sharpe": round(ev["sharpe"], 3)}
        sel_metric = {k: v["inner_sharpe"] for k, v in table.items()}
        winner_name = max(sel_metric, key=sel_metric.get)
        winner = next(c for c in zoo if c.name == winner_name)

        # ---- Stage 5b: winner walk-forward on the val window
        y = targets[winner.target]
        z = walk_forward_z(winner, X, y, regime, fsets, val_start, val_end)
        pos = np.zeros(n)
        pos[val_start:val_end] = positions_from_z(
            z[val_start:val_end],
            bull[val_start:val_end] if winner.is_composite else None)
        ev = evaluate_segment(close, pos, COST_BPS[asset], val_start, val_end)
        model_series = ev.pop("series")
        lo, hi = block_bootstrap_ci(model_series)

        # ---- Stage 6: gate vs baselines on the same window/costs
        base = {}
        base_series = {}
        for bname in ("buy_and_hold", "sma_200bar"):
            bpos = baseline_positions(bname, close, val_start, val_end)
            bev = evaluate_segment(close, bpos, COST_BPS[asset], val_start, val_end)
            base_series[bname] = bev.pop("series")
            base[bname] = {k: round(v, 3) if isinstance(v, float) else v
                           for k, v in bev.items()}
        fold_series[fold] = {"start": val_start, "model": model_series,
                             **{f"b_{k}": v for k, v in base_series.items()}}
        beats = [b for b in base if ev["pnl_pct"] > base[b]["pnl_pct"]]
        passes = len(beats) == len(base) and lo is not None and lo > 0

        results["folds"][fold] = {
            "train_end": train_end, "val": [val_start, val_end],
            "selection_table": table,
            "selected": winner_name,
            "selected_config": {"family": winner.family, "target": winner.target,
                                "features": winner.fset, "refit": winner.refit,
                                "params": winner.params,
                                "n_features": len(fsets[winner.fset])},
            "val": {k: round(v, 3) if isinstance(v, float) else v
                    for k, v in ev.items()},
            "sharpe_ci": [lo, hi],
            "baselines": base,
            "promotion": {"passes": bool(passes), "beaten": beats,
                          "lost_to": [b for b in base if b not in beats],
                          "ci_excludes_zero": bool(lo is not None and lo > 0)},
        }
        print(f"  {fold}: selected={winner_name} "
              f"(inner {sel_metric}) -> val pnl={ev['pnl_pct']:+.1f}% "
              f"sharpe={ev['sharpe']:+.2f} CI[{lo},{hi}] "
              f"vs B&H {base['buy_and_hold']['pnl_pct']:+.1f}% "
              f"SMA {base['sma_200bar']['pnl_pct']:+.1f}% "
              f"-> {'PASS' if passes else 'FAIL'}", flush=True)

    # ---- Stage 6b: POOLED certification view. Per-fold ~1,964-bar windows
    # cannot certify anything (P242: the CI needs Sharpe ~2.5-3 that no
    # baseline reaches either); pooling the fold vals chronologically gives
    # a full-cycle window comparable to the protocol lockbox, where the
    # same family DID certify. Reported alongside, never replacing, the
    # per-fold gate — which bar governs promotion is an operator decision.
    if fold_series:
        ordered = sorted(fold_series.values(), key=lambda d: d["start"])
        pooled = {}
        for key, label in (("model", "selected_per_fold"),
                           ("b_buy_and_hold", "buy_and_hold"),
                           ("b_sma_200bar", "sma_200bar")):
            seg = np.concatenate([d[key] for d in ordered])
            sd = float(np.nanstd(seg))
            lo, hi = block_bootstrap_ci(seg)
            pooled[label] = {
                "pnl_pct": round(float(np.nansum(seg)) * 100, 2),
                "sharpe": round(float(np.nanmean(seg) / sd * math.sqrt(BARS_PER_YEAR)), 3) if sd > 0 else 0.0,
                "sharpe_ci": [lo, hi],
            }
        mdl = pooled["selected_per_fold"]
        beats = [b for b in ("buy_and_hold", "sma_200bar")
                 if mdl["pnl_pct"] > pooled[b]["pnl_pct"]]
        mdl_lo = mdl["sharpe_ci"][0]
        pooled["verdict"] = {
            "beats": beats,
            "ci_excludes_zero": bool(mdl_lo is not None and mdl_lo > 0),
            "passes_pooled": bool(len(beats) == 2 and mdl_lo is not None and mdl_lo > 0),
        }
        results["pooled"] = pooled
        print(f"  POOLED ({len(ordered)} folds): model pnl={mdl['pnl_pct']:+.1f}% "
              f"sharpe={mdl['sharpe']:+.2f} CI{mdl['sharpe_ci']} vs "
              f"B&H {pooled['buy_and_hold']['pnl_pct']:+.1f}% "
              f"SMA {pooled['sma_200bar']['pnl_pct']:+.1f}% -> "
              f"{'POOLED-PASS' if pooled['verdict']['passes_pooled'] else 'POOLED-FAIL'}",
              flush=True)

    out = REPO / "models" / "supervised" / asset / tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"  -> {out / 'results.json'}", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", type=str, default="BTC,ETH")
    ap.add_argument("--tag", type=str, required=True,
                    help="fresh tag per run (P200: stale tags restore caches)")
    args = ap.parse_args()
    diag = json.loads(DIAG_PATH.read_text(encoding="utf-8")) if DIAG_PATH.exists() else {}
    if not diag:
        print("WARNING: no data-pattern diagnostics report — zoo falls back to "
              "the fixed candidate set (run data_pattern_diagnostics.py first)")
    for asset in args.assets.split(","):
        run_asset(asset.strip().upper(), args.tag, diag)
    print("\nDONE")


if __name__ == "__main__":
    sys.exit(main() or 0)
