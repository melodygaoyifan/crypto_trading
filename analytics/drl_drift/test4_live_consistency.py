"""TEST 4 — Policy consistency check (recent obs vs nearest-training-obs).

For each recent obs, find its k-nearest neighbors in the training-period obs
and compare actions. If the model's action on the recent obs differs sharply
from its action on the nearest training obs, the policy is unstable in this
region of obs space — it's either extrapolating to OOD or the local policy
landscape has high gradient.

Healthy: mean |action_recent - action_train_nn| < 0.10
Unstable: > 0.30 means the policy gives meaningfully different decisions on
          obs that are nearest-neighbors in input space.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

from analytics.drl_drift._common import (
    ASSETS,
    build_obs_matrix,
    load_drl_inference,
    predict_actions,
    stack_frames,
)

OUT = Path("/tmp/hmats_audit/drl_drift/test4_live_consistency.json")
OUT.parent.mkdir(parents=True, exist_ok=True)


def run():
    results = {}
    for asset in ASSETS:
        print(f"\n  === {asset} ===")
        try:
            model = load_drl_inference(asset)
            obs_recent, _ = build_obs_matrix(asset, days_back=180)
            obs_train, _ = build_obs_matrix(asset, days_start=540, days_end=180)
        except Exception as e:
            print(f"    [BUILD-FAIL] {e}")
            results[asset] = {"error": str(e)}
            continue

        if len(obs_recent) < 50 or len(obs_train) < 200:
            results[asset] = {"error": f"too few bars (recent={len(obs_recent)}, train={len(obs_train)})"}
            continue

        # Sample 200 recent obs for KNN lookup; predict on them and on their NNs
        n_sample = min(200, len(obs_recent))
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(len(obs_recent), size=n_sample, replace=False)
        obs_recent_sample = obs_recent[sample_idx]

        # KNN over the 122 feature dims (env-state is zero everywhere — would dominate)
        knn = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(obs_train[:, :122])
        distances, indices = knn.kneighbors(obs_recent_sample[:, :122])

        # Predict on recent sample (need stacked obs, so re-stack just these)
        # We re-stack each batch from raw 126; for KNN-paired training obs we do the same
        stk_recent = stack_frames(obs_recent)[sample_idx]
        a_recent = predict_actions(model, stk_recent)

        train_idx_flat = indices.flatten()
        # Stack the training sequence once and pull aligned rows
        stk_train_full = stack_frames(obs_train)
        stk_train_nn = stk_train_full[train_idx_flat]
        a_train_nn = predict_actions(model, stk_train_nn)

        action_diff = np.abs(a_recent - a_train_nn)
        avg_distance = float(distances.mean())
        max_distance = float(distances.max())
        median_distance = float(np.median(distances))
        mean_diff = float(action_diff.mean())
        median_diff = float(np.median(action_diff))
        max_diff = float(action_diff.max())
        n_disagree_strong = int((action_diff > 0.5).sum())  # Sign would have flipped if both were saturated

        if mean_diff < 0.10:
            v = "STABLE"
        elif mean_diff < 0.30:
            v = "MILD_INSTABILITY"
        else:
            v = "POLICY_UNSTABLE_IN_RECENT_REGION"

        print(f"    n_sample={n_sample}")
        print(f"    obs distance to train-NN:  mean={avg_distance:.3f}  median={median_distance:.3f}  max={max_distance:.3f}")
        print(f"    |action_recent - action_NN|: mean={mean_diff:.3f}  median={median_diff:.3f}  max={max_diff:.3f}")
        print(f"    strong disagreements (|diff|>0.5): {n_disagree_strong}/{n_sample} ({n_disagree_strong/n_sample:.0%})")
        print(f"    -> {v}")

        results[asset] = {
            "n_sample": int(n_sample),
            "avg_obs_distance_to_train_nn": avg_distance,
            "median_obs_distance_to_train_nn": median_distance,
            "max_obs_distance_to_train_nn": max_distance,
            "mean_action_diff": mean_diff,
            "median_action_diff": median_diff,
            "max_action_diff": max_diff,
            "n_strong_disagreements_gt_0_5": n_disagree_strong,
            "verdict": v,
        }

    OUT.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\n  saved: {OUT}")
    return results


if __name__ == "__main__":
    run()
