#!/usr/bin/env python3
"""
Stage 7: Reward Mode + State Augmentation - 200K step fair comparison.

Runs 5 experiments sequentially on BTC fold_1:
  1. classic          (baseline)
  2. sharpe
  3. sortino
  4. classic + augment
  5. sharpe + augment

Decision logic (applied after all runs complete):
  - sharpe >= classic * 0.95 -> sharpe goes into Optuna search
  - sharpe < classic * 0.80  -> sharpe confirmed ineffective
  - augment >= no_augment * 0.95 -> augment goes into Optuna
  - augment < no_augment * 0.90 -> augment too costly

Usage:
    python -X utf8 scripts/run_stage7_comparison.py
    python -X utf8 scripts/run_stage7_comparison.py --timesteps 100000   # quick test
"""

import subprocess
import sys
import json
import time
from pathlib import Path

_TRAINING_DIR = Path(__file__).resolve().parent.parent   # training/
PROJECT_ROOT = _TRAINING_DIR.parent                      # project root

TIMESTEPS = 200_000
OUTPUT_DIR = str(PROJECT_ROOT / "reports" / "stage7")
ASSET = "BTC"

EXPERIMENTS = [
    {"name": "classic",         "args": ["--reward-mode", "classic"]},
    {"name": "sharpe",          "args": ["--reward-mode", "sharpe"]},
    {"name": "sortino",         "args": ["--reward-mode", "sortino"]},
    {"name": "classic_augment", "args": ["--reward-mode", "classic", "--augment"]},
    {"name": "sharpe_augment",  "args": ["--reward-mode", "sharpe", "--augment"]},
]


def run_experiment(name: str, extra_args: list, timesteps: int):
    """Run a single training experiment."""
    cmd = [
        sys.executable, "-X", "utf8", "-u",
        str(_TRAINING_DIR / "train_drl_full.py"),
        "--asset", ASSET,
        "--folds", "1",
        "--timesteps", str(timesteps),
        "--output", OUTPUT_DIR,
        "--no-progress-bar",
    ] + extra_args

    print(f"\n{'=' * 70}")
    print(f"  EXPERIMENT: {name}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'=' * 70}\n")

    start = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - start

    print(f"\n  [{name}] Exit code: {result.returncode} | Time: {elapsed/60:.1f} min")
    return result.returncode, elapsed


def collect_results():
    """Collect eval results from all experiments."""
    import numpy as np

    results = {}
    label_map = {
        "classic":         "ULTIMATE",
        "sharpe":          "ULTIMATE_SHARPE",
        "sortino":         "ULTIMATE_SORTINO",
        "classic_augment": "ULTIMATE_AUG",
        "sharpe_augment":  "ULTIMATE_SHARPE_AUG",
    }

    for exp_name, label in label_map.items():
        eval_path = Path(OUTPUT_DIR) / ASSET / "fold_1" / "logs" / label / "eval" / "evaluations.npz"
        if eval_path.exists():
            data = np.load(eval_path)
            ep_rewards = data.get("results", [])
            if len(ep_rewards) > 0:
                # Best eval reward across all checkpoints
                mean_per_eval = [float(np.mean(r)) for r in ep_rewards]
                best = max(mean_per_eval)
                last = mean_per_eval[-1]
                results[exp_name] = {
                    "best_eval": best,
                    "last_eval": last,
                    "n_evals": len(mean_per_eval),
                }
            else:
                results[exp_name] = {"best_eval": 0, "last_eval": 0, "n_evals": 0}
        else:
            results[exp_name] = {"error": f"No eval file: {eval_path}"}

    return results


def analyze(results: dict):
    """Apply decision logic."""
    print(f"\n{'=' * 70}")
    print("  STAGE 7 - DECISION ANALYSIS")
    print(f"{'=' * 70}\n")

    for name, r in results.items():
        if "error" in r:
            print(f"  {name:20s}: ERROR - {r['error']}")
        else:
            print(f"  {name:20s}: best={r['best_eval']:+.2f}  last={r['last_eval']:+.2f}  evals={r['n_evals']}")

    classic = results.get("classic", {}).get("best_eval", 0)
    sharpe = results.get("sharpe", {}).get("best_eval", 0)
    sortino = results.get("sortino", {}).get("best_eval", 0)
    classic_aug = results.get("classic_augment", {}).get("best_eval", 0)
    sharpe_aug = results.get("sharpe_augment", {}).get("best_eval", 0)

    print(f"\n  --- Reward Mode Decisions ---")

    # Sharpe vs Classic
    if classic > 0:
        ratio = sharpe / classic
        print(f"  sharpe/classic = {ratio:.3f}")
        if ratio >= 0.95:
            print(f"  DECISION: sharpe PASSES (>= 0.95) -> include in Optuna search")
        elif ratio < 0.80:
            print(f"  DECISION: sharpe FAILS (< 0.80) -> exclude from Optuna")
        else:
            print(f"  DECISION: sharpe MARGINAL (0.80-0.95) -> needs judgment")

    # Sortino vs Classic
    if classic > 0:
        ratio = sortino / classic
        print(f"  sortino/classic = {ratio:.3f}")
        if ratio >= 0.95:
            print(f"  DECISION: sortino PASSES (>= 0.95) -> include in Optuna search")
        elif ratio < 0.80:
            print(f"  DECISION: sortino FAILS (< 0.80) -> exclude from Optuna")
        else:
            print(f"  DECISION: sortino MARGINAL (0.80-0.95) -> needs judgment")

    print(f"\n  --- Augmentation Decisions ---")

    # Classic augment vs Classic
    if classic > 0:
        ratio = classic_aug / classic
        print(f"  classic_aug/classic = {ratio:.3f}")
        if ratio >= 0.95:
            print(f"  DECISION: augment PASSES (>= 0.95) -> include in Optuna")
        elif ratio < 0.90:
            print(f"  DECISION: augment FAILS (< 0.90) -> too costly")
        else:
            print(f"  DECISION: augment MARGINAL (0.90-0.95) -> needs judgment")

    # Sharpe augment vs Sharpe
    if sharpe > 0:
        ratio = sharpe_aug / sharpe
        print(f"  sharpe_aug/sharpe = {ratio:.3f}")

    # Save results
    out_path = Path(OUTPUT_DIR) / "stage7_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {out_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=TIMESTEPS)
    parser.add_argument("--analyze-only", action="store_true",
                        help="Skip training, just analyze existing results")
    args = parser.parse_args()

    if not args.analyze_only:
        total_start = time.time()
        for exp in EXPERIMENTS:
            code, elapsed = run_experiment(exp["name"], exp["args"], args.timesteps)
            if code != 0:
                print(f"\n  WARNING: {exp['name']} exited with code {code}")
        total_time = time.time() - total_start
        print(f"\n  Total training time: {total_time/3600:.1f} hours")

    results = collect_results()
    analyze(results)


if __name__ == "__main__":
    main()
