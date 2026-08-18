#!/usr/bin/env python
"""Resolve the DRL half of docs/VALIDATION_EXPORT_SPEC_FOR_OPERATOR.md LOCALLY.

The spec said per-fold OOS realized-return SERIES were needed and "what's on the
box can't be validated". They CAN — every artifact exists:
  - val data:   training/training_data/drl_training/{ASSET}_4H_full.parquet
  - checkpoints: models/retrained/{ASSET}/fold_{k}/.../best_model.zip   (9/9 exist)
  - scalers:    models/retrained/{ASSET}/fold_{k}/.../feature_scaler.pkl (9/9 exist)
  - rollout:    training/drl/oracle_tqc_teacher.py (TQCTeacherOracle, n_stack=8)
  - harness:    analytics/validation/{sharpe_validation,cpcv}.py (PSR + CSCV-PBO)

This script rolls EACH fold's policy over the common val window, EXPORTS the
per-bar return series (the spec's deliverable), then runs:
  - per-fold annualized Sharpe + Probabilistic Sharpe Ratio (PSR)
  - CSCV Probability of Backtest Overfitting (PBO) across the 3 folds
  - best-fold CLEAN-OOS (post-2026-02-27, never trained on) Sharpe + PSR

Promotion rule (spec): trust a DRL version ONLY if PBO < 0.5 AND clean-OOS PSR > 0.95.

Usage:  python -X utf8 scripts/validate_drl_oos.py [--asset all|BTC|ETH|SOL]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

_proj = Path(__file__).resolve().parents[1]
if str(_proj) not in sys.path:
    sys.path.insert(0, str(_proj))

from analytics.validation.cpcv import cscv_pbo
from analytics.validation.sharpe_validation import (
    annualized_sharpe, probabilistic_sharpe_ratio,
)

# TQC checkpoints were trained 2026-02-25..03-02 (see file mtimes). The parquet
# ENDS 2026-03-31 and the LIVE period is Apr-Jun 2026 (NOT in the parquet at all).
# So the "val window" below (Oct'25-Mar'26) is a DIFFERENT REGIME than live and
# may be partly in-sample for these retrained checkpoints. It is NOT a substitute
# for forward-live OOS. The ONLY true OOS is the live result:
LIVE_SHARPE_ANCHOR = -2.62    # realized live Apr-Jun 2026 (P143 / live-performance memory)
TQC_CUTOFF = pd.Timestamp("2026-03-02", tz="UTC")  # latest ckpt train date
BARS_PER_YEAR = 2190          # 6 x 365, 24/7 crypto 4H bars
FRICTION_BPS = {"BTC": 3.0, "ETH": 5.0, "SOL": 10.0}  # matches training slippage
EXPORT_DIR = _proj / "data" / "validation_export"


def _model_path(asset: str, k: int) -> Path:
    return _proj / f"models/retrained/{asset}/fold_{k}/logs/LSTM_FILM_A/best_model/best_model.zip"


def _scaler_path(asset: str, k: int) -> Path:
    return _proj / f"models/retrained/{asset}/fold_{k}/logs/LSTM_FILM_A/feature_scaler.pkl"


def _scale(features_raw: np.ndarray, all_features: list, scaler_meta: dict) -> np.ndarray:
    """Replicate drl/runtime_obs_builder._scale_features: RobustScaler on scale_cols,
    clip [-10,10], pass_cols untouched."""
    feats = features_raw.copy()
    scaler = scaler_meta.get("scaler")
    scale_cols = scaler_meta.get("scale_cols", [])
    if scaler is None or not scale_cols:
        return feats
    idx = [all_features.index(c) for c in scale_cols if c in all_features]
    scaled = scaler.transform(feats[:, idx])
    feats[:, idx] = np.clip(scaled, -10.0, 10.0)
    return feats


def _roll_fold(asset: str, k: int, val: pd.DataFrame, all_features: list) -> dict | None:
    """Roll fold k's TQC over the val window -> per-bar net returns. Returns
    {pnl, ts, clean_mask} or None if the model/scaler can't load."""
    from training.drl.oracle_tqc_teacher import TQCTeacherOracle

    mp, sp = _model_path(asset, k), _scaler_path(asset, k)
    if not mp.exists() or not sp.exists():
        print(f"  [{asset} fold_{k}] SKIP — missing model/scaler")
        return None
    scaler_meta = joblib.load(sp)
    feats_raw = val[all_features].values.astype(np.float32)
    feats_scaled = _scale(feats_raw, all_features, scaler_meta)

    try:
        oracle = TQCTeacherOracle(asset=asset, tqc_model_path=str(mp))
        oracle.set_context(feats_scaled)
        oracle.reset_rollout()
    except Exception as e:
        print(f"  [{asset} fold_{k}] load failed: {type(e).__name__}: {e}")
        return None

    actions = np.zeros(len(val), dtype=np.float32)
    for i in range(len(val)):
        act, _ = oracle.get_action(None, i, "UNKNOWN")
        actions[i] = act

    close = val["close"].values.astype(np.float64)
    ret = np.zeros(len(val), dtype=np.float64)
    ret[:-1] = (close[1:] - close[:-1]) / close[:-1]

    pa, rt = actions[:-1], ret[:-1]
    ts = pd.to_datetime(pd.Series(val["timestamp"].values[:-1]), utc=True)
    turnover = np.abs(np.diff(np.concatenate([[0.0], pa])))
    pnl = pa * rt - turnover * (FRICTION_BPS[asset] / 1e4)
    clean_mask = (ts >= TQC_CUTOFF).values

    # export the per-bar return series (the spec deliverable)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = EXPORT_DIR / f"drl_{asset}_fold_{k}_val_returns.jsonl"
    with open(out, "w", encoding="utf-8") as fh:
        for t, r in zip(ts.values, pnl):
            fh.write(json.dumps({"bar_ts": int(pd.Timestamp(t).value // 1_000_000),
                                 "ret": float(r)}) + "\n")
    return {"pnl": pnl, "ts": ts, "clean_mask": clean_mask,
            "n_long": int((pa > 0.1).sum()), "n_short": int((pa < -0.1).sum())}


def run(asset: str) -> dict:
    parquet = _proj / f"training/training_data/drl_training/{asset}_4H_full.parquet"
    df = pd.read_parquet(parquet)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    val = df.iloc[int(len(df) * 0.85):].reset_index(drop=True)

    manifest = json.load(open(_proj / "configs/feature_manifest.json", encoding="utf-8"))
    all_features = manifest["all_features"]
    assert len(all_features) == 122

    print(f"\n{'='*72}\n  {asset}  DRL per-fold OOS validation  "
          f"(val {val['timestamp'].iloc[0].date()} -> {val['timestamp'].iloc[-1].date()}, "
          f"{len(val)} bars)\n{'='*72}")

    folds, per_fold = {}, {}
    for k in (1, 2, 3):
        r = _roll_fold(asset, k, val, all_features)
        if r is None:
            continue
        folds[k] = r["pnl"]
        full_sr = annualized_sharpe(r["pnl"], BARS_PER_YEAR)
        psr = probabilistic_sharpe_ratio(r["pnl"])
        clean = r["pnl"][r["clean_mask"]]
        clean_sr = annualized_sharpe(clean, BARS_PER_YEAR) if len(clean) > 5 else 0.0
        clean_psr = probabilistic_sharpe_ratio(clean) if len(clean) > 5 else {"psr": 0.0}
        per_fold[k] = {"full_sharpe": full_sr, "full_psr": psr.get("psr"),
                       "clean_bars": int(len(clean)), "clean_sharpe": clean_sr,
                       "clean_psr": clean_psr.get("psr"),
                       "n_long": r["n_long"], "n_short": r["n_short"]}
        print(f"  fold_{k}: full Sharpe={full_sr:+.2f} PSR={psr.get('psr',0)*100:4.1f}%  "
              f"| CLEAN-OOS({len(clean)}b) Sharpe={clean_sr:+.2f} "
              f"PSR={clean_psr.get('psr',0)*100:4.1f}%  "
              f"[L={r['n_long']} S={r['n_short']}]")

    pbo_res = {}
    if len(folds) >= 2:
        mat = [list(v) for v in folds.values()]
        pbo_res = cscv_pbo(mat)
        print(f"  CSCV-PBO across {len(folds)} folds: "
              f"PBO={pbo_res.get('pbo',float('nan'))*100:.1f}% -> {pbo_res.get('verdict')}")
    else:
        print("  CSCV-PBO: need >=2 folds")

    # best-fold = highest val-window Sharpe. The decisive test is NOT whether the
    # val window looks good (it's a historical/diff-regime window, possibly partly
    # in-sample) — it is whether the val-window metric PREDICTS live. It does not.
    best_k = max(per_fold, key=lambda k: per_fold[k]["full_sharpe"]) if per_fold else None
    verdict = "NO_DATA"
    if best_k is not None and per_fold:
        bf = per_fold[best_k]
        val_sr = bf["full_sharpe"]
        # The backtest is only trustworthy if it tracks live. A wildly positive
        # val Sharpe alongside a negative live Sharpe = non-predictive backtest.
        contradicts_live = (val_sr > 1.0) and (LIVE_SHARPE_ANCHOR < 0.0)
        if contradicts_live:
            verdict = "BACKTEST_CONTRADICTS_LIVE"  # overfit/non-stationary; DO NOT promote on backtest
        elif pbo_res.get("pbo", 1.0) < 0.5 and val_sr > 0:
            verdict = "INCONCLUSIVE_NEEDS_FORWARD_OOS"
        else:
            verdict = "FAIL_NO_EDGE"
        print(f"  >> best val fold = fold_{best_k}: val Sharpe={val_sr:+.2f}, "
              f"PSR={(bf.get('full_psr') or 0)*100:.0f}%, PBO={pbo_res.get('pbo',0)*100:.1f}%"
              f"  vs  LIVE Sharpe={LIVE_SHARPE_ANCHOR:+.2f}  =>  {verdict}")
        if contradicts_live:
            print(f"     ^ val window (Oct'25-Mar'26) looks great but LIVE (Apr-Jun'26) "
                  f"lost money — the backtest does NOT predict live. Textbook overfit/"
                  f"non-stationarity. Forward-live PSR is the ONLY valid arbiter.")
    return {"asset": asset, "per_fold": per_fold, "pbo": pbo_res,
            "best_fold": best_k, "val_sharpe": per_fold.get(best_k, {}).get("full_sharpe") if best_k else None,
            "live_sharpe_anchor": LIVE_SHARPE_ANCHOR, "verdict": verdict}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="all")
    args = ap.parse_args()
    assets = ["BTC", "ETH", "SOL"] if args.asset == "all" else [args.asset]
    report = {}
    for a in assets:
        try:
            report[a] = run(a)
        except Exception as exc:
            import traceback
            print(f"[{a}] ERROR: {exc}")
            traceback.print_exc()
            report[a] = {"asset": a, "error": str(exc)}
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(EXPORT_DIR / "drl_validation_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"\n{'='*72}\n  SUMMARY\n{'='*72}")
    for a, r in report.items():
        print(f"  {a}: {r.get('verdict','ERROR')}"
              + (f"  (best fold_{r.get('best_fold')}, "
                 f"PBO={ (r.get('pbo') or {}).get('pbo') }" if r.get('verdict') not in (None,'ERROR') else ""))
    print(f"\n  Per-bar return series + report exported to {EXPORT_DIR}")
    print("  Promotion rule: trust ONLY if PBO<0.5 AND clean-OOS PSR>0.95.")


if __name__ == "__main__":
    main()
