#!/usr/bin/env python
"""Extract Max DD and Avg Hold from Stage 7 NAV retest best models."""
import sys, os, json
from pathlib import Path
import numpy as np

_TRAINING_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = _TRAINING_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(_TRAINING_DIR))

from models.feature_extractors import get_extractor_class, get_extractor_kwargs

import pandas as pd
import joblib
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import TQC

from train_drl_full import TradingEnvFull, REGIME_NAMES
from core.regime_smoother import RegimeSmoother

OUTPUT_DIR = PROJECT_ROOT / "reports" / "stage7_nav_retest"
DATA_PATH = _TRAINING_DIR / "training_data" / "drl_training" / "BTC_4H_full.parquet"
N_EPISODES = 10


def load_val_data():
    data = pd.read_parquet(DATA_PATH)
    if "regime" in data.columns:
        smoother = RegimeSmoother(min_persistence=2)
        data = smoother.smooth_column(data, "regime")

    manifest_path = PROJECT_ROOT / "configs" / "feature_manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    feature_cols = [c for c in manifest["all_features"] if c in data.columns]

    n = len(data)
    val_size = int(n * 0.15)
    gap = 42
    val_start = n - val_size
    val_df = data.iloc[val_start:].copy()
    return val_df, feature_cols


def evaluate_detailed(model, eval_env, n_episodes=10):
    initial = eval_env.get_attr("initial_balance")[0]
    all_max_dd = []
    all_hold_times = []
    all_balances = []

    for ep in range(n_episodes):
        obs = eval_env.reset()
        done = False
        peak_bal = initial
        max_dd = 0.0
        prev_dir = 0
        entry_step = 0
        step_count = 0
        hold_times_ep = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, infos = eval_env.step(action)
            done = dones[0]
            info = infos[0]

            bal = info.get("balance", initial)
            if bal > peak_bal:
                peak_bal = bal
            dd = (peak_bal - bal) / peak_bal if peak_bal > 0 else 0
            if dd > max_dd:
                max_dd = dd

            pos = info.get("position", 0.0)
            cur_dir = 1 if pos > 0.01 else (-1 if pos < -0.01 else 0)
            if prev_dir != 0 and cur_dir != prev_dir:
                hold_times_ep.append(step_count - entry_step)
                if cur_dir != 0:
                    entry_step = step_count
            elif prev_dir == 0 and cur_dir != 0:
                entry_step = step_count
            prev_dir = cur_dir
            step_count += 1

        all_max_dd.append(max_dd * 100)
        all_hold_times.extend(hold_times_ep)
        all_balances.append(info.get("balance", initial))

    nav_pct = (np.mean(all_balances) - initial) / initial * 100
    avg_hold = np.mean(all_hold_times) if all_hold_times else 0
    avg_hold_hours = avg_hold * 4  # 4H bars

    return {
        "nav_pct": nav_pct,
        "final_nav": np.mean(all_balances),
        "max_dd_pct": np.mean(all_max_dd),
        "avg_hold_bars": avg_hold,
        "avg_hold_hours": avg_hold_hours,
        "n_trades": len(all_hold_times) // n_episodes,
    }


def main():
    val_df, feature_cols = load_val_data()
    print(f"Val data: {len(val_df)} rows, {len(feature_cols)} features")

    experiments = ["classic", "sharpe", "sortino"]
    results = {}

    # Load results JSON for best reward info
    results_json = OUTPUT_DIR / "stage7_nav_retest_results.json"
    with open(results_json) as f:
        saved = json.load(f)
    saved_by_name = {r["experiment"]: r for r in saved["results"]}

    for name in experiments:
        exp_dir = OUTPUT_DIR / name
        best_path = exp_dir / "best_model" / "best_model.zip"
        scaler_path = exp_dir / "feature_scaler.pkl"

        if not best_path.exists():
            print(f"  SKIP {name}: no best_model")
            continue

        print(f"\n--- {name} ---")

        # Load scaler and scale val data
        scaler_meta = joblib.load(scaler_path)
        scaler = scaler_meta["scaler"]
        scale_cols = scaler_meta["scale_cols"]

        val_scaled = val_df.copy()
        val_scaled[scale_cols] = scaler.transform(val_df[scale_cols])
        val_scaled[scale_cols] = val_scaled[scale_cols].replace(
            [np.inf, -np.inf], 0
        ).fillna(0).clip(-10.0, 10.0)

        reward_mode = {"classic": "classic", "sharpe": "sharpe", "sortino": "sortino"}[name]

        def make_eval_env(df=val_scaled, rm=reward_mode):
            env = TradingEnvFull(
                df=df, feature_cols=feature_cols, reward_mode=rm,
                augment_enabled=False,
                bear_reward_multiplier=1.2, bull_reward_multiplier=1.1,
                sol_vol_multiplier=1.0,
            )
            return Monitor(env)

        eval_env = DummyVecEnv([make_eval_env])
        eval_env = VecFrameStack(eval_env, n_stack=8)

        model = TQC.load(str(best_path), env=eval_env, device="cuda")
        metrics = evaluate_detailed(model, eval_env, n_episodes=N_EPISODES)

        # Get best reward from evaluations.npz
        eval_npz = exp_dir / "eval" / "evaluations.npz"
        best_reward = 0
        if eval_npz.exists():
            evals = np.load(eval_npz)
            best_reward = float(np.max(evals["results"]))

        metrics["best_reward"] = best_reward
        results[name] = metrics

        del model
        eval_env.close()
        import torch, gc
        torch.cuda.empty_cache()
        gc.collect()

        print(f"  NAV%: {metrics['nav_pct']:+.2f}%")
        print(f"  Final NAV: ${metrics['final_nav']:,.0f}")
        print(f"  Max DD: {metrics['max_dd_pct']:.2f}%")
        print(f"  Avg Hold: {metrics['avg_hold_bars']:.1f} bars ({metrics['avg_hold_hours']:.0f}h)")
        print(f"  Best Reward: {best_reward:+.2f}")

    # Print final table
    print(f"\n{'=' * 90}")
    print(f"{'Mode':<12} {'Best Rew':>10} {'Final NAV':>14} {'NAV %':>10} "
          f"{'Max DD':>8} {'Trades':>8} {'Avg Hold':>10}")
    print(f"{'-' * 90}")
    for name in experiments:
        if name not in results:
            continue
        m = results[name]
        s = saved_by_name.get(name, {})
        print(f"{name:<12} {m['best_reward']:>+10.2f} "
              f"${m['final_nav']:>12,.0f} {m['nav_pct']:>+9.2f}% "
              f"{m['max_dd_pct']:>7.2f}% {s.get('trade_count', m['n_trades']):>7d} "
              f"{m['avg_hold_bars']:.0f}b/{m['avg_hold_hours']:.0f}h")


if __name__ == "__main__":
    main()