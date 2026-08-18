"""TQC teacher direct backtest on the SAME val period as DT.

Measures TQC's own performance on DT's val window to answer:
  - Is TQC itself profitable on this period? (→ teacher is useful)
  - Or is TQC also a noise generator? (→ teacher is the root problem)

If TQC is profitable but DT imitation isn't, distillation broke.
If both are unprofitable, the TQC teacher hypothesis (KD-BeT) is invalid for
this dataset / regime.

Uses the same VecFrameStack(n_stack=8) contract as training.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_proj = Path(__file__).resolve().parents[2]
if str(_proj) not in sys.path:
    sys.path.insert(0, str(_proj))

TQC_CUTOFF = pd.Timestamp("2026-02-27", tz="UTC")


def run(asset: str) -> None:
    import json
    from training.drl.oracle_tqc_teacher import TQCTeacherOracle

    parquet = f"training/training_data/drl_training/{asset}_4H_full.parquet"
    df = pd.read_parquet(parquet)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    split = int(len(df) * 0.85)
    val = df.iloc[split:].reset_index(drop=True)

    manifest = json.load(open("configs/feature_manifest.json", encoding="utf-8"))
    feature_cols = manifest["all_features"]
    no_scale = set(manifest.get("no_scale_features", []))
    assert len(feature_cols) == 122

    # Fit scaler on val features (TQC expects scaled features; since we're
    # testing post-hoc, fit on val set itself — not ideal but consistent
    # across all experiments. Alternative is to load DT's training scaler.)
    import joblib
    scaler_path = f"models/decision_transformer/{asset}/dt_scaler.pkl"
    scaler_obj = joblib.load(scaler_path)

    feats_raw = val[feature_cols].values.astype(np.float32)
    feats_scaled = feats_raw.copy()
    scale_idx = [i for i, c in enumerate(feature_cols) if c not in no_scale]
    feats_scaled[:, scale_idx] = scaler_obj.transform(feats_raw[:, scale_idx])
    feats_scaled[:, scale_idx] = np.clip(feats_scaled[:, scale_idx], -10, 10)

    # Load TQC teacher (matches training BEST_FOLDS)
    oracle = TQCTeacherOracle(asset=asset)
    oracle.set_context(feats_scaled)
    oracle.reset_rollout()

    # Forward pass: get TQC action for each val bar
    actions = np.zeros(len(val), dtype=np.float32)
    rewards_oracle = np.zeros(len(val), dtype=np.float32)
    for i in range(len(val)):
        act, rew = oracle.get_action(None, i, "UNKNOWN")
        actions[i] = act
        rewards_oracle[i] = rew

    # Next-bar return
    close = val["close"].values.astype(np.float64)
    ret = np.zeros(len(val), dtype=np.float64)
    ret[:-1] = (close[1:] - close[:-1]) / close[:-1]

    # Trim last bar (no next-return)
    pa = actions[:-1]
    rt = ret[:-1]
    ts = val["timestamp"].values[:-1]

    active = np.abs(pa) > 0.1
    true_up = rt > 0
    pred_up = pa > 0
    overall_hit = (active & (pred_up == true_up)).sum() / max(active.sum(), 1)

    friction_bps = 5
    turnover = np.abs(np.diff(np.concatenate([[0.0], pa])))
    pnl = pa * rt - turnover * (friction_bps / 10000)
    cum = np.cumsum(pnl)

    def sharpe(x):
        if len(x) < 5 or x.std() == 0:
            return 0.0
        return float(x.mean() / x.std() * np.sqrt(1512))

    total_sharpe = sharpe(pnl)
    total_return = cum[-1]
    n_trades = int((turnover > 0.05).sum())

    ts_pd = pd.to_datetime(pd.Series(ts), utc=True).dt.tz_convert(None)
    cutoff = TQC_CUTOFF.tz_convert(None)
    leaked_mask = ts_pd < cutoff
    clean_mask = ~leaked_mask
    leaked_hit = (active[leaked_mask] & (pred_up[leaked_mask] == true_up[leaked_mask])).sum() / max(active[leaked_mask].sum(), 1)
    clean_hit = (active[clean_mask] & (pred_up[clean_mask] == true_up[clean_mask])).sum() / max(active[clean_mask].sum(), 1)
    leaked_sharpe = sharpe(pnl[leaked_mask])
    clean_sharpe = sharpe(pnl[clean_mask])

    # Action distribution diagnostic
    n_long = (pa > 0.1).sum(); n_short = (pa < -0.1).sum(); n_neut = len(pa) - n_long - n_short
    abs_mean = np.mean(np.abs(pa))

    print(f"\n{'=' * 70}")
    print(f"  {asset}  TQC teacher direct backtest (same val window)")
    print(f"{'=' * 70}")
    print(f"  val window: {ts[0]} -> {ts[-1]}  ({len(pa)} bars)")
    print(f"  action distrib: LONG={n_long}, SHORT={n_short}, NEUT={n_neut}  "
          f"abs_mean={abs_mean:.3f}")
    print(f"  active (|a|>0.1): {active.sum()}/{len(pa)}  trades: {n_trades}")
    print(f"")
    print(f"  OVERALL:")
    print(f"    true-direction hit rate: {overall_hit*100:.1f}%")
    print(f"    cumulative net return:   {total_return*100:+.2f}%")
    print(f"    annualized Sharpe:       {total_sharpe:+.2f}")
    print(f"")
    print(f"  LEAKED sub (pre-2026-02-27, TQC trained on this):")
    print(f"    bars: {leaked_mask.sum()}  hit: {leaked_hit*100:.1f}%  Sharpe: {leaked_sharpe:+.2f}")
    print(f"")
    print(f"  CLEAN sub (post-2026-02-27, TQC never saw):")
    print(f"    bars: {clean_mask.sum()}  hit: {clean_hit*100:.1f}%  Sharpe: {clean_sharpe:+.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="all")
    args = ap.parse_args()
    assets = ["BTC", "ETH", "SOL"] if args.asset == "all" else [args.asset]
    for a in assets:
        try:
            run(a)
        except Exception as exc:
            print(f"[{a}] error: {exc}")
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
