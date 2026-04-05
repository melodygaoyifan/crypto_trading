#!/usr/bin/env python3
"""
Stage 7 Diagnostic Evaluation - Unified Metrics Across Reward Modes.

Loads all 5 Stage 7 best models and evaluates each on the SAME val data
(BTC fold_1) with a SINGLE deterministic episode, collecting 6 metrics:

  1. Return%          - (final_equity / initial - 1) * 100
  2. MaxDD%           - peak-to-trough drawdown
  3. MeanAbsExposure  - mean(|position|)
  4. Trades           - number of entries (flat -> non-flat)
  5. Turnover + FlipRate - sum(|Δpos|) and long↔short flips
  6. RewardBreakdown  - mean of pnl / dd_penalty / turnover_penalty / ratio_bonus

All models run with reward_mode="classic" in the env so the equity curve
is computed identically (same cost model, same data, same seed).

Usage:
    python -X utf8 scripts/stage7_diagnostic_eval.py
    python -X utf8 scripts/stage7_diagnostic_eval.py --seed 42
"""

import argparse
import json
import sys
from pathlib import Path
from collections import deque
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import torch

# ── Path setup ──────────────────────────────────────────────────────────
_TRAINING_DIR = Path(__file__).resolve().parent.parent   # training/
PROJECT_ROOT = _TRAINING_DIR.parent                      # project root
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(_TRAINING_DIR))

from sb3_contrib import TQC
from train_drl_full import (
    TradingEnvFull,
    REGIME_NAMES,
    POSITION_BIAS,
    BEAR_REGIMES,
    BULL_REGIMES,
    _load_regime_names,
)

# ── Constants ───────────────────────────────────────────────────────────
STAGE7_DIR = PROJECT_ROOT / "reports" / "stage7"
DATA_PATH = _TRAINING_DIR / "training_data" / "drl_training" / "BTC_4H_full.parquet"
ASSET = "BTC"
INITIAL_BALANCE = 100_000.0
TRANSACTION_COST = 0.001
WINDOW_SIZE = 10

EXPERIMENTS = {
    "classic":         "ULTIMATE",
    "sharpe":          "ULTIMATE_SHARPE",
    "sortino":         "ULTIMATE_SORTINO",
    "classic_aug":     "ULTIMATE_AUG",
    "sharpe_aug":      "ULTIMATE_SHARPE_AUG",
}


# ── Fold 1 split (same logic as train_drl_full.py) ─────────────────────
def get_fold1_val_data(data: pd.DataFrame):
    """Reproduce exact fold_1 val split."""
    n = len(data)
    val_size = int(n * 0.15)
    gap = 42
    # fold_idx=0 -> latest data
    val_end = n
    val_start = val_end - val_size
    train_end = val_start - gap
    train_df = data.iloc[:train_end].copy()
    val_df = data.iloc[val_start:val_end].copy()
    return train_df, val_df


def scale_with_saved_scaler(val_df: pd.DataFrame, scaler_path: Path,
                            feature_cols: List[str]) -> pd.DataFrame:
    """Apply the saved RobustScaler from training."""
    meta = joblib.load(scaler_path)
    scaler = meta["scaler"]
    scale_cols = meta["scale_cols"]

    val_scaled = val_df.copy()
    # Only scale columns that exist in both the scaler and the val data
    valid_scale = [c for c in scale_cols if c in val_scaled.columns]
    val_scaled[valid_scale] = scaler.transform(val_scaled[valid_scale])
    val_scaled[valid_scale] = val_scaled[valid_scale].replace(
        [np.inf, -np.inf], 0
    ).fillna(0)
    val_scaled[valid_scale] = val_scaled[valid_scale].clip(-10.0, 10.0)
    return val_scaled


# ── Reward breakdown tracker (shadow, doesn't affect the real env) ─────
class RewardBreakdownTracker:
    """Tracks what WOULD have been the reward components under sortino/sharpe."""

    def __init__(self, window: int = 48):
        self._returns: deque = deque(maxlen=window)
        self._sharpe_window = 20

    def reset(self):
        self._returns.clear()

    def compute_breakdown(
        self,
        pnl_pct: float,
        drawdown: float,
        position: float,
        position_change: float,
        regime: str,
        regime_weight: float,
    ) -> Dict[str, float]:
        self._returns.append(pnl_pct)

        # PNL component (classic scale)
        pnl_r = pnl_pct * 100.0 * regime_weight
        if pnl_pct < 0:
            pnl_r *= 2.0  # downside_weight

        # DD penalty
        dd_penalty = 0.0
        dd_threshold = 0.10
        dd_penalty_mult = 5.0  # enhanced uses dd_penalty_mult * 2.5 = 5.0
        if drawdown > dd_threshold:
            dd_penalty = (drawdown * dd_penalty_mult) ** 2

        # Turnover penalty
        turnover_pen = 0.0
        if abs(position_change) > 1.5:
            turnover_pen = 0.02

        # Rolling ratio bonus (sortino-like)
        ratio_bonus = 0.0
        if len(self._returns) >= 20:
            returns = np.array(self._returns)
            mean_ret = np.mean(returns)
            neg = returns[returns < 0]
            if len(neg) > 2:
                ds = np.std(neg)
                if ds > 1e-8:
                    sortino = mean_ret / ds
                    ratio_bonus = float(np.clip(sortino * 0.05, -0.2, 0.2))

        return {
            "pnl_component": pnl_r,
            "dd_penalty": -dd_penalty,
            "turnover_penalty": -turnover_pen,
            "ratio_bonus": ratio_bonus,
        }


# ── Run one episode ────────────────────────────────────────────────────
def run_episode(model, val_df: pd.DataFrame, feature_cols: List[str],
                seed: int = 42) -> Dict:
    """Run deterministic episode, collect all 6 diagnostic groups."""

    env = TradingEnvFull(
        df=val_df,
        feature_cols=feature_cols,
        initial_balance=INITIAL_BALANCE,
        transaction_cost=TRANSACTION_COST,
        window_size=WINDOW_SIZE,
        reward_mode="classic",          # unified cost model
        augment_enabled=False,
    )

    tracker = RewardBreakdownTracker()
    tracker.reset()

    obs, _ = env.reset(seed=seed)
    done = False

    # Accumulators
    positions = []
    position_changes = []
    equity_curve = [INITIAL_BALANCE]
    entries = 0
    flips = 0
    prev_position = 0.0
    total_turnover = 0.0
    breakdowns = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        done = done or truncated

        cur_pos = info.get("position", env.position)
        balance = info.get("balance", env.balance)
        pos_change = cur_pos - prev_position

        # Equity
        equity_curve.append(balance)

        # Position stats
        positions.append(abs(cur_pos))
        position_changes.append(abs(pos_change))
        total_turnover += abs(pos_change)

        # Entry count: flat -> non-flat
        if abs(prev_position) < 0.01 and abs(cur_pos) >= 0.01:
            entries += 1

        # Flip: long->short or short->long (exclude flat)
        if prev_position > 0.01 and cur_pos < -0.01:
            flips += 1
        elif prev_position < -0.01 and cur_pos > 0.01:
            flips += 1

        # Reward breakdown (shadow)
        regime = info.get("regime", "WEAK_CONSOLIDATION")
        pnl_pct = (balance - equity_curve[-2]) / INITIAL_BALANCE if len(equity_curve) >= 2 else 0
        drawdown = (max(equity_curve) - balance) / max(equity_curve) if max(equity_curve) > 0 else 0
        regime_weight = 1.0  # simplified; actual env uses computed weights
        bd = tracker.compute_breakdown(
            pnl_pct=pnl_pct,
            drawdown=drawdown,
            position=cur_pos,
            position_change=pos_change,
            regime=regime,
            regime_weight=regime_weight,
        )
        breakdowns.append(bd)

        prev_position = cur_pos

    # ── Compute final metrics ──
    equity = np.array(equity_curve)
    final_equity = equity[-1]
    return_pct = (final_equity / INITIAL_BALANCE - 1) * 100

    # MaxDD%
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / np.where(peak > 0, peak, 1)
    max_dd_pct = float(np.max(dd) * 100)

    # MeanAbsExposure
    mean_abs_exposure = float(np.mean(positions)) if positions else 0.0

    # Turnover
    turnover = total_turnover

    # Reward breakdown means
    bd_keys = ["pnl_component", "dd_penalty", "turnover_penalty", "ratio_bonus"]
    bd_means = {}
    if breakdowns:
        for k in bd_keys:
            bd_means[k] = float(np.mean([b[k] for b in breakdowns]))
    else:
        bd_means = {k: 0.0 for k in bd_keys}

    return {
        "return_pct": round(return_pct, 3),
        "max_dd_pct": round(max_dd_pct, 3),
        "mean_abs_exposure": round(mean_abs_exposure, 4),
        "trades": entries,
        "turnover": round(turnover, 2),
        "flip_rate": flips,
        "n_steps": len(positions),
        "final_equity": round(final_equity, 2),
        "reward_breakdown": {k: round(v, 6) for k, v in bd_means.items()},
    }


# ── Main ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Stage 7 Diagnostic Eval")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 80)
    print("  STAGE 7 - DIAGNOSTIC EVALUATION (Unified Metrics)")
    print("=" * 80)

    # Load data
    data = pd.read_parquet(DATA_PATH)
    print(f"  Data: {DATA_PATH.name} ({len(data)} rows)")

    # Load feature cols
    fc_path = STAGE7_DIR / ASSET / "feature_cols.json"
    with open(fc_path) as f:
        feature_cols = json.load(f)
    print(f"  Features: {len(feature_cols)} | Obs dim: {len(feature_cols) + 4}")

    # Load regime names
    global REGIME_NAMES
    REGIME_NAMES = _load_regime_names(ASSET)

    # Get fold_1 val data
    train_df, val_df_raw = get_fold1_val_data(data)
    print(f"  Val data: {len(val_df_raw)} rows (fold_1)")
    print(f"  Seed: {args.seed}")
    print(f"  Cost model: {TRANSACTION_COST * 100:.1f}% per trade")
    print()

    results = {}
    for exp_name, label in EXPERIMENTS.items():
        model_dir = STAGE7_DIR / ASSET / "fold_1" / "logs" / label
        model_path = model_dir / "best_model" / "best_model"
        scaler_path = model_dir / "feature_scaler.pkl"

        if not (model_path.parent / "best_model.zip").exists():
            print(f"  [{exp_name}] SKIP - model not found: {model_path}")
            continue
        if not scaler_path.exists():
            print(f"  [{exp_name}] SKIP - scaler not found: {scaler_path}")
            continue

        # Scale val data with this experiment's scaler
        val_df = scale_with_saved_scaler(val_df_raw, scaler_path, feature_cols)

        # Load model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = TQC.load(str(model_path), device=device)

        # Run episode
        r = run_episode(model, val_df, feature_cols, seed=args.seed)
        results[exp_name] = r

        print(f"  [{exp_name:12s}] Return={r['return_pct']:+8.2f}%  "
              f"MaxDD={r['max_dd_pct']:5.2f}%  "
              f"Exposure={r['mean_abs_exposure']:.3f}  "
              f"Trades={r['trades']:3d}  "
              f"Turnover={r['turnover']:7.1f}  "
              f"Flips={r['flip_rate']:3d}")

    # ── Summary table ──
    print()
    print("=" * 80)
    print("  SUMMARY TABLE")
    print("=" * 80)
    print()

    header = f"{'Experiment':15s} {'Return%':>9s} {'MaxDD%':>8s} {'Exposure':>9s} {'Trades':>7s} {'Turnover':>9s} {'Flips':>6s} {'Steps':>6s}"
    print(header)
    print("-" * len(header))
    for exp_name, r in results.items():
        print(f"{exp_name:15s} {r['return_pct']:+9.2f} {r['max_dd_pct']:8.2f} "
              f"{r['mean_abs_exposure']:9.4f} {r['trades']:7d} "
              f"{r['turnover']:9.1f} {r['flip_rate']:6d} {r['n_steps']:6d}")

    # ── Reward breakdown table ──
    print()
    print("  REWARD BREAKDOWN (per-step means, shadow classic-scale)")
    print()
    header2 = f"{'Experiment':15s} {'PnL':>10s} {'DD_Pen':>10s} {'Turn_Pen':>10s} {'Ratio':>10s}"
    print(header2)
    print("-" * len(header2))
    for exp_name, r in results.items():
        bd = r["reward_breakdown"]
        print(f"{exp_name:15s} {bd['pnl_component']:10.4f} {bd['dd_penalty']:10.4f} "
              f"{bd['turnover_penalty']:10.6f} {bd['ratio_bonus']:10.6f}")

    # ── Diagnosis ──
    print()
    print("=" * 80)
    print("  DIAGNOSIS")
    print("=" * 80)
    classic = results.get("classic", {})
    if not classic:
        print("  No classic baseline - cannot diagnose")
        return

    c_ret = classic["return_pct"]
    c_dd = classic["max_dd_pct"]
    c_exp = classic["mean_abs_exposure"]
    c_trades = classic["trades"]
    c_turnover = classic["turnover"]

    for exp_name, r in results.items():
        if exp_name == "classic":
            continue
        print(f"\n  --- {exp_name} vs classic ---")

        # Return comparison
        ret_ratio = r["return_pct"] / c_ret if c_ret != 0 else 0
        print(f"  Return ratio: {ret_ratio:.3f} ({r['return_pct']:+.2f}% vs {c_ret:+.2f}%)")

        # MaxDD
        dd_diff = r["max_dd_pct"] - c_dd
        print(f"  MaxDD diff: {dd_diff:+.2f}pp ({r['max_dd_pct']:.2f}% vs {c_dd:.2f}%)")

        # Exposure
        exp_ratio = r["mean_abs_exposure"] / c_exp if c_exp > 0 else 0
        print(f"  Exposure ratio: {exp_ratio:.3f} ({r['mean_abs_exposure']:.4f} vs {c_exp:.4f})")

        # Trades
        trade_ratio = r["trades"] / c_trades if c_trades > 0 else 0
        print(f"  Trade ratio: {trade_ratio:.3f} ({r['trades']} vs {c_trades})")

        # Turnover
        turn_ratio = r["turnover"] / c_turnover if c_turnover > 0 else 0
        print(f"  Turnover ratio: {turn_ratio:.3f} ({r['turnover']:.1f} vs {c_turnover:.1f})")

        # Category diagnosis
        causes = []
        if abs(ret_ratio - 1.0) < 0.15 and r["return_pct"] > 0:
            causes.append("(1) SCALE: returns similar, reward scale difference dominates")
        if exp_ratio < 0.7 or trade_ratio < 0.5:
            causes.append("(2) RISK_SHAPING: learned low-exposure/few-trades to avoid penalty")
        bd_c = classic["reward_breakdown"]
        bd_e = r["reward_breakdown"]
        if abs(bd_e.get("ratio_bonus", 0)) < abs(bd_c.get("pnl_component", 0)) * 0.01:
            causes.append("(3) ROLLING_RATIO: ratio bonus too small / noisy to be useful signal")
        if "aug" in exp_name and (exp_ratio < 0.8 or turn_ratio > 1.3):
            causes.append("(4) AUGMENTATION: noise/dropout degrading sample efficiency or causing instability")

        if not causes:
            causes.append("Multiple factors or unclear - need deeper analysis")

        print(f"  Likely causes:")
        for c in causes:
            print(f"    -> {c}")

    # ── Save ──
    out_path = STAGE7_DIR / "stage7_diagnostic.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {out_path}")


if __name__ == "__main__":
    main()
