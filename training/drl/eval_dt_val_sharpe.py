"""DT val-period backtest: true direction accuracy + Sharpe.

Replaces the training-time "val acc" (= DT agreement with TQC teacher) with
the decision-relevant metrics:

  - directional hit rate vs next-bar price change
  - simulated PnL Sharpe with friction
  - leakage split: TQC-seen vs truly-unseen period

TQC best_model.zip for each asset was saved 2026-02-27; our DT val period
spans 2025-10-19 to 2026-03-31, so roughly 4.5 months of DT's val set overlap
with TQC's training data (leaked). Only 2026-02-28 → 2026-03-31 is clean.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Resolve project root on sys.path so training.drl.* imports work
_proj = Path(__file__).resolve().parents[2]
if str(_proj) not in sys.path:
    sys.path.insert(0, str(_proj))

# TQC training cutoff (from best_model.zip mtime inspection)
TQC_CUTOFF = pd.Timestamp("2026-02-27", tz="UTC")


def _load_dt(asset: str, weights_path: str, scaler_path: str):
    """Load DT model + scaler via training script internals."""
    # Models were saved from train_decision_transformer_v32.py run as __main__,
    # so the pickle embeds __main__.DTConfigV32 etc. Register the classes in
    # __main__ so unpickling resolves them.
    import __main__
    from training.drl import train_decision_transformer_v32 as _t
    for _name in (
        "DTConfigV32", "DecisionTransformerV32", "TrainerV32",
        "TrajectoryDatasetV32", "RegimeAwareExpert",
    ):
        if hasattr(_t, _name):
            setattr(__main__, _name, getattr(_t, _name))
    cfg = _t.DTConfigV32(asset=asset)
    model = _t.DecisionTransformerV32(cfg)
    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval()

    import joblib
    scaler = joblib.load(scaler_path)
    return model, scaler, cfg


def _infer_context(model, cfg, feats_block: np.ndarray, env_block: np.ndarray) -> float:
    """One forward pass on a (ctx, 126) window; returns scalar action for the last step."""
    states = np.concatenate([feats_block, env_block], axis=1).astype(np.float32)  # (ctx, 126)
    with torch.no_grad():
        s = torch.from_numpy(states).unsqueeze(0)            # (1, ctx, 126)
        a = torch.zeros(1, states.shape[0], 1, dtype=torch.float32)
        # RTG has 4 scales (config.rtg_scales=[1,4,24,168]); fill with benign target reward.
        n_scales = len(cfg.rtg_scales)
        rtg = torch.full((1, states.shape[0], n_scales), 10.0, dtype=torch.float32)
        t = torch.arange(states.shape[0]).unsqueeze(0)
        out = model(s, a, rtg, t)
        act = float(out["action_preds"][0, -1, 0].item())
    return max(-1.0, min(1.0, act))


def run(asset: str, days: int = 14) -> None:
    parquet = f"training/training_data/drl_training/{asset}_4H_full.parquet"
    df = pd.read_parquet(parquet)
    if "timestamp" not in df.columns:
        df = df.reset_index().rename(columns={"index": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Same split as training
    split = int(len(df) * 0.85)
    val = df.iloc[split:].reset_index(drop=True)

    # Feature manifest — must match training
    import json
    manifest = json.load(open("configs/feature_manifest.json"))
    feature_cols = manifest.get("all_features") or manifest.get("features")
    no_scale = set(manifest.get("no_scale_features", []))
    assert feature_cols is not None and len(feature_cols) == 122, (
        f"bad manifest: got {len(feature_cols) if feature_cols else 0} features, need 122"
    )

    model_root = Path(f"models/decision_transformer/{asset}")
    weights = model_root / "dt_v32_best.pt"
    scaler = model_root / "dt_scaler.pkl"
    if not weights.exists() or not scaler.exists():
        print(f"[{asset}] model/scaler missing at {model_root}")
        return
    model, scaler_obj, cfg = _load_dt(asset, str(weights), str(scaler))
    ctx = cfg.context_length  # 48

    # Scale features (honoring no_scale cols)
    feats_raw = val[feature_cols].values.astype(np.float32)
    feats_scaled = feats_raw.copy()
    scale_idx = [i for i, c in enumerate(feature_cols) if c not in no_scale]
    feats_scaled[:, scale_idx] = scaler_obj.transform(feats_raw[:, scale_idx])
    feats_scaled[:, scale_idx] = np.clip(feats_scaled[:, scale_idx], -10, 10)

    # Env-state block: use stateless proxy (position_dir evolves from prev action)
    pred_actions = np.zeros(len(val), dtype=np.float32)
    prev_action = 0.0
    for i in range(ctx, len(val) - 1):  # last bar has no next-return
        feat_window = feats_scaled[i - ctx:i]  # (ctx, 122)
        env_window = np.zeros((ctx, 4), dtype=np.float32)
        # Simulate rolling position state from previously predicted actions
        for j in range(ctx):
            env_window[j, 0] = abs(prev_action)
            env_window[j, 1] = np.sign(prev_action)
        act = _infer_context(model, cfg, feat_window, env_window)
        pred_actions[i] = act
        prev_action = act

    # Compute returns per next-bar close/open move
    close = val["close"].values.astype(np.float64)
    ret = np.zeros(len(val), dtype=np.float64)
    ret[:-1] = (close[1:] - close[:-1]) / close[:-1]

    # Evaluate ONLY where we produced a prediction
    mask = np.arange(len(val)) >= ctx
    mask[-1] = False  # last bar has no next-return
    pa = pred_actions[mask]
    rt = ret[mask]
    ts = val["timestamp"].values[mask]

    # Direction hit-rate vs ACTUAL next-bar return
    active = np.abs(pa) > 0.1  # only "directional" predictions
    true_up = rt > 0
    pred_up = pa > 0
    overall_hit = (active & (pred_up == true_up)).sum() / max(active.sum(), 1)

    # Simulated PnL: position = pa, next-bar return = rt
    friction_bps = 5  # per unit turnover
    turnover = np.abs(np.diff(np.concatenate([[0.0], pa])))
    pnl = pa * rt - turnover * (friction_bps / 10000)
    cum = np.cumsum(pnl)

    def sharpe(x):
        # 4H bars → 6 per day → ~252 trading days in crypto × 6 = 1512 bars/year
        # Using sqrt(annualization factor)
        if len(x) < 5 or x.std() == 0:
            return 0.0
        return float(x.mean() / x.std() * np.sqrt(1512))

    total_sharpe = sharpe(pnl)
    total_return = cum[-1]
    n_trades = int((turnover > 0.05).sum())

    # Split by TQC leakage (normalize tz: both to tz-naive UTC)
    ts_pd = pd.to_datetime(pd.Series(ts), utc=True).dt.tz_convert(None)
    cutoff_naive = TQC_CUTOFF.tz_convert(None)
    leaked_mask = ts_pd < cutoff_naive
    clean_mask = ~leaked_mask
    leaked_hit = (active[leaked_mask] & (pred_up[leaked_mask] == true_up[leaked_mask])).sum() / max(active[leaked_mask].sum(), 1)
    clean_hit = (active[clean_mask] & (pred_up[clean_mask] == true_up[clean_mask])).sum() / max(active[clean_mask].sum(), 1)
    leaked_sharpe = sharpe(pnl[leaked_mask])
    clean_sharpe = sharpe(pnl[clean_mask])

    print(f"\n{'=' * 70}")
    print(f"  {asset}  DT val-period backtest")
    print(f"{'=' * 70}")
    print(f"  val window: {ts[0]} -> {ts[-1]}  ({len(pa)} bars)")
    print(f"  active predictions (|action|>0.1): {active.sum()}/{len(pa)}  "
          f"({100*active.sum()/len(pa):.1f}%)")
    print(f"  trades (turnover>0.05): {n_trades}")
    print(f"")
    print(f"  OVERALL:")
    print(f"    true-direction hit rate: {overall_hit*100:.1f}%")
    print(f"    cumulative net return:   {total_return*100:+.2f}%")
    print(f"    annualized Sharpe:       {total_sharpe:+.2f}")
    print(f"")
    print(f"  LEAKED sub-period (TQC saw this, pre-2026-02-27):")
    print(f"    bars: {leaked_mask.sum()}")
    print(f"    hit rate: {leaked_hit*100:.1f}%")
    print(f"    Sharpe:   {leaked_sharpe:+.2f}")
    print(f"")
    print(f"  CLEAN sub-period (TQC did NOT see, post-2026-02-27):")
    print(f"    bars: {clean_mask.sum()}")
    print(f"    hit rate: {clean_hit*100:.1f}%")
    print(f"    Sharpe:   {clean_sharpe:+.2f}")
    print(f"")
    print(f"  interpretation: if hit-rate collapses on CLEAN sub-period,")
    print(f"  training val_acc was partially from TQC leakage.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="all", help="BTC/ETH/SOL or 'all'")
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
