"""[P221-followup] Adaptive-refit ridge — the honest position backtest.

The ETH diagnosis found the campaign's frozen-per-fold evaluation handicaps
assets whose linear signal DECAYS: walk-forward ridge IC inside the fold
windows clears the cost bar on 5/9 windows (ETH fold1 +0.149!) while the
fit-once ridge baseline lost money on the same windows. This script measures
what that adaptivity is worth as a POSITION backtest, not an IC:

  refit cadences: fold  (fit once per ~15-month fold - the campaign design)
                  monthly (refit every 180 bars)
                  weekly  (refit every 42 bars)

Positions: z = pred/sigma_train, deadband, act every DECISION_INTERVAL bars,
clip to [-1,1]. Costs: 3bps per side on position CHANGE (coinbase taker).
Same features the campaign uses (122-manifest + fv2 extras).

This is a measurement script; promotion of any cadence goes through the same
ladder as everything else (shadow -> P166 forward gate).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_TRAINING_DIR = Path(__file__).resolve().parent.parent
REPO = _TRAINING_DIR.parent
DATA_DIR = _TRAINING_DIR / "training_data" / "drl_training"

ASSETS = ("BTC", "ETH", "SOL")
HORIZON = 4              # 16h forward target
DECISION_INTERVAL = 4    # act every 16h (campaign cadence)
DEADBAND = 0.25
# Per-side friction = coinbase taker fee + per-asset slippage (the campaign
# env's ASSET_SLIPPAGE_BPS). Charging fee alone understates friction 2-4x.
FEE_SIDE_BPS = {"BTC": 3.0 + 3.0, "ETH": 3.0 + 5.0, "SOL": 3.0 + 10.0}
MIN_TRAIN = 3000
CADENCES = {"fold": None, "monthly": 180, "weekly": 42}


def load(asset):
    df = pd.read_parquet(DATA_DIR / f"{asset}_4H_full.parquet")
    manifest = json.loads((REPO / "configs" / "feature_manifest.json").read_text(encoding="utf-8"))
    feats = [c for c in manifest["all_features"] if c in df.columns]
    feats += sorted(c for c in df.columns if c.startswith("fv2_"))
    return df, feats


def walkforward_positions(df, feats, refit_every, fold_bounds=None):
    """Return position series (one per bar). refit_every=None -> refit only at
    fold train_end boundaries (the campaign's frozen design)."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    X = df[feats].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    n = len(close)
    y = np.full(n, np.nan)
    y[:-HORIZON] = close[HORIZON:] / close[:-HORIZON] - 1.0

    refit_points = []
    if refit_every is None:
        refit_points = sorted(set(b for b in (fold_bounds or []) if b >= MIN_TRAIN))
        if not refit_points:
            refit_points = [MIN_TRAIN]
    else:
        refit_points = list(range(MIN_TRAIN, n, refit_every))

    pos = np.zeros(n)
    model = sc = None
    sigma = 1.0
    next_refit_idx = 0
    last_pos = 0.0
    for i in range(MIN_TRAIN, n):
        if next_refit_idx < len(refit_points) and i >= refit_points[next_refit_idx]:
            tr_end = refit_points[next_refit_idx]
            # PURGE: the last HORIZON labels use closes at/after tr_end —
            # training on them leaks the first evaluation bars' returns.
            pe = max(0, tr_end - HORIZON)
            m = ~(np.isnan(X[:pe]).any(axis=1) | np.isnan(y[:pe]))
            if m.sum() > 500:
                sc = StandardScaler().fit(X[:pe][m])
                model = Ridge(alpha=10.0).fit(sc.transform(X[:pe][m]), y[:pe][m])
                preds_tr = model.predict(sc.transform(X[:pe][m]))
                sigma = float(np.std(preds_tr)) or 1e-9
            next_refit_idx += 1
        if model is None:
            continue
        if (i - MIN_TRAIN) % DECISION_INTERVAL == 0:
            z = float(model.predict(sc.transform(np.nan_to_num(X[i:i + 1])))[0]) / sigma
            last_pos = 0.0 if abs(z) < DEADBAND or sigma < 1e-8 else float(np.clip(z, -1, 1))
        pos[i] = last_pos
    return pos


def evaluate(df, pos, label, windows, fee_side_bps):
    close = df["close"].to_numpy(dtype=float)
    ret = np.zeros(len(close))
    ret[1:] = close[1:] / close[:-1] - 1.0
    # pnl at bar i uses position held from bar i-1
    strat = np.zeros(len(close))
    strat[1:] = pos[:-1] * ret[1:]
    cost = np.zeros(len(close))
    cost[1:] = np.abs(np.diff(pos)) * fee_side_bps / 1e4
    net = strat - cost
    out = {"label": label}
    for wname, (s, e) in windows.items():
        seg = net[s:e]
        tot = float(np.nansum(seg)) * 100
        sd = float(np.nanstd(seg))
        sharpe = float(np.nanmean(seg) / sd * math.sqrt(6 * 365)) if sd > 0 else 0.0
        out[wname] = (tot, sharpe)
    turn = float(np.abs(np.diff(pos)).sum())
    out["turnover"] = turn
    return out


def main():
    sm = json.loads((REPO / "configs" / "split_manifest.json").read_text(encoding="utf-8"))
    for asset in ASSETS:
        df, feats = load(asset)
        folds = sm["assets"][asset]["folds"]
        fold_bounds = [f["train_end"] for f in folds]
        windows = {f"fold{f['fold']}": (f["val_start"], f["val_end"]) for f in folds}
        n = len(df)
        windows["ALL_OOS"] = (MIN_TRAIN, n)
        close = df["close"].to_numpy(dtype=float)

        print(f"\n===== {asset} ({n} bars, {len(feats)} features) =====")
        # passive references per window
        hdr = f"{'strategy':<12}" + "".join(f"{w:>20}" for w in windows) + f"{'turnover':>10}"
        print(hdr)
        bh_line = f"{'buy&hold':<12}"
        for wname, (s, e) in windows.items():
            tot = (close[e - 1] / close[s] - 1) * 100
            bh_line += f"{tot:>+14.0f}%     "
        print(bh_line)
        for label, cad in CADENCES.items():
            pos = walkforward_positions(df, feats, cad, fold_bounds)
            r = evaluate(df, pos, label, windows, FEE_SIDE_BPS[asset])
            line = f"{label:<12}"
            for wname in windows:
                tot, sh = r[wname]
                line += f"{tot:>+9.0f}% ({sh:>+4.1f})  "
            line += f"{r['turnover']:>8.0f}"
            print(line)


if __name__ == "__main__":
    sys.exit(main() or 0)
