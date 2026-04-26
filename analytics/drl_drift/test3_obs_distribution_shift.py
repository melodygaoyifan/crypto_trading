"""TEST 3 — Obs distribution shift (training-period vs recent 180d).

Compares per-dim mean/std for the 126-dim scaled obs between:
  - "training-like" window: 540d-180d ago (~12 months)
  - "recent" window: last 180d

Reports drift in train-std units. Threshold ladder:
  |drift| > 1σ : MILD
  |drift| > 2σ : MODERATE
  |drift| > 3σ : SEVERE

Uses scaled obs (post-RobustScaler) — this is what the policy network actually sees.
Drift in scaled space tells us how different the live distribution is from what
the scaler was fit on.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

from analytics.drl_drift._common import (
    ASSETS,
    build_obs_matrix,
    load_manifest,
)

OUT = Path("/tmp/hmats_audit/drl_drift/test3_obs_distribution_shift.json")
OUT.parent.mkdir(parents=True, exist_ok=True)


def run():
    manifest = load_manifest()
    feature_names = manifest["all_features"] + ["env_position_ratio", "env_position_dir", "env_pnl_ratio", "env_drawdown"]

    results = {}
    for asset in ASSETS:
        print(f"\n  === {asset} ===")

        try:
            obs_recent, ts_recent = build_obs_matrix(asset, days_back=180)
            obs_train, ts_train = build_obs_matrix(asset, days_start=540, days_end=180)
        except Exception as e:
            print(f"    [BUILD-FAIL] {e}")
            results[asset] = {"error": str(e)}
            continue

        if len(obs_recent) < 50 or len(obs_train) < 50:
            results[asset] = {"error": f"too few bars (recent={len(obs_recent)}, train={len(obs_train)})"}
            continue

        train_means = obs_train.mean(axis=0)
        train_stds = obs_train.std(axis=0)
        recent_means = obs_recent.mean(axis=0)

        # Compute drift in std units (skip env-state dims since they're zero by design)
        drift_sigma = np.zeros(len(train_means))
        for i in range(len(train_means)):
            if train_stds[i] > 1e-6:
                drift_sigma[i] = (recent_means[i] - train_means[i]) / train_stds[i]
            # else: leave at 0

        # Exclude env-state from drift counts (they're always zero)
        drift_features_only = drift_sigma[:122]
        n_1s = int((np.abs(drift_features_only) > 1.0).sum())
        n_2s = int((np.abs(drift_features_only) > 2.0).sum())
        n_3s = int((np.abs(drift_features_only) > 3.0).sum())

        # Top 15 most-drifted feature dims
        top_idx = np.argsort(np.abs(drift_features_only))[::-1][:15]

        if n_3s >= 10:
            v = "SEVERE_OOD"
        elif n_3s >= 3 or n_2s >= 15:
            v = "MODERATE_OOD"
        elif n_2s >= 5 or n_1s >= 30:
            v = "MILD_OOD"
        else:
            v = "IN_DISTRIBUTION"

        print(f"    train window: {ts_train.iloc[0]} -> {ts_train.iloc[-1]} ({len(obs_train)} bars)")
        print(f"    recent window: {ts_recent.iloc[0]} -> {ts_recent.iloc[-1]} ({len(obs_recent)} bars)")
        print(f"    drift counts (122 features only):")
        print(f"      |drift| > 1σ : {n_1s:3d}/122  ({n_1s/122:.0%})")
        print(f"      |drift| > 2σ : {n_2s:3d}/122  ({n_2s/122:.0%})")
        print(f"      |drift| > 3σ : {n_3s:3d}/122  ({n_3s/122:.0%})")
        print(f"    Top 10 most-drifted features:")
        for idx in top_idx[:10]:
            name = feature_names[idx] if idx < len(feature_names) else f"dim{idx}"
            print(
                f"      {name:30s} drift={drift_features_only[idx]:+.2f}σ  "
                f"(train_mean={train_means[idx]:+.3f}, recent={recent_means[idx]:+.3f})"
            )
        print(f"    -> {v}")

        results[asset] = {
            "verdict": v,
            "n_train_bars": int(len(obs_train)),
            "n_recent_bars": int(len(obs_recent)),
            "train_window": [str(ts_train.iloc[0]), str(ts_train.iloc[-1])],
            "recent_window": [str(ts_recent.iloc[0]), str(ts_recent.iloc[-1])],
            "n_drift_1sigma": n_1s,
            "n_drift_2sigma": n_2s,
            "n_drift_3sigma": n_3s,
            "top15_drift": [
                {
                    "dim": int(idx),
                    "name": feature_names[idx] if idx < len(feature_names) else f"dim{idx}",
                    "drift_sigma": float(drift_features_only[idx]),
                    "train_mean": float(train_means[idx]),
                    "recent_mean": float(recent_means[idx]),
                }
                for idx in top_idx
            ],
        }

    OUT.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n  saved: {OUT}")
    return results


if __name__ == "__main__":
    run()
