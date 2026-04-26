"""TEST 2 — Q-value sanity on recent obs.

Calls TQC critic directly on (obs, action) batches.
Healthy: finite, bounded, non-collapsed Q values.
Drifted: NaN, inf, mean magnitude > 1000, std < 1e-3 (collapsed).

Also reports inter-critic disagreement (TQC has 2 critics × 25 quantiles each).
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch

from analytics.drl_drift._common import (
    ASSETS,
    build_obs_matrix,
    load_drl_inference,
    stack_frames,
)

OUT = Path("/tmp/hmats_audit/drl_drift/test2_qvalue_sanity.json")
OUT.parent.mkdir(parents=True, exist_ok=True)


def verdict(q_arr: np.ndarray) -> str:
    if np.isnan(q_arr).any():
        return f"NAN_FAIL ({int(np.isnan(q_arr).sum())} NaN)"
    if np.isinf(q_arr).any():
        return f"INF_FAIL ({int(np.isinf(q_arr).sum())} Inf)"
    q_mean = float(q_arr.mean())
    q_std = float(q_arr.std())
    if abs(q_mean) > 1000:
        return f"EXPLODED (mean={q_mean:+.1f})"
    if q_std < 1e-3:
        return f"COLLAPSED (std={q_std:.6f})"
    if (q_arr.max() - q_arr.min()) < 0.01:
        return f"FLAT (range {q_arr.max() - q_arr.min():.4f})"
    return "HEALTHY"


def run():
    results = {}
    for asset in ASSETS:
        print(f"\n  === {asset} ===")
        try:
            model = load_drl_inference(asset)
        except Exception as e:
            print(f"    [LOAD-FAIL] {e}")
            results[asset] = {"error": f"load: {e}"}
            continue

        obs, _ = build_obs_matrix(asset, days_back=180)
        if len(obs) < 100:
            results[asset] = {"error": f"insufficient bars ({len(obs)})"}
            continue

        stk = stack_frames(obs)
        sb3_model = model._tqc_model
        device = sb3_model.device

        # Predict actions first (we need them as input to critic)
        actions_np, _ = sb3_model.predict(stk, deterministic=True)
        actions_np = np.asarray(actions_np).reshape(-1, 1).astype(np.float32)

        obs_t = torch.tensor(stk, dtype=torch.float32, device=device)
        act_t = torch.tensor(actions_np, dtype=torch.float32, device=device)

        with torch.no_grad():
            try:
                q_out = sb3_model.critic(obs_t, act_t)
            except Exception as e:
                print(f"    [CRITIC-FAIL] {e}")
                results[asset] = {"error": f"critic: {e}"}
                continue

        # TQC critic returns tuple/list of N critics, each (batch, n_quantiles)
        if isinstance(q_out, (list, tuple)):
            q_per_critic = [q.cpu().numpy() for q in q_out]
            q_stacked = np.stack(q_per_critic, axis=0)  # (n_critics, batch, n_q)
        else:
            q_stacked = q_out.cpu().numpy()
            q_per_critic = [q_stacked]

        n_critics = q_stacked.shape[0]
        n_q = q_stacked.shape[-1] if q_stacked.ndim >= 2 else 1
        n_nan = int(np.isnan(q_stacked).sum())
        n_inf = int(np.isinf(q_stacked).sum())

        q_mean = float(q_stacked.mean())
        q_median = float(np.median(q_stacked))
        q_std = float(q_stacked.std())
        q_min = float(q_stacked.min())
        q_max = float(q_stacked.max())

        # Inter-critic disagreement: per-bar mean Q from each critic, std across critics
        if n_critics > 1:
            per_critic_means = q_stacked.mean(axis=-1)  # (n_critics, batch)
            inter_critic_std = float(per_critic_means.std(axis=0).mean())
        else:
            inter_critic_std = 0.0

        # Per-bar quantile spread (uncertainty proxy used by TQCInference)
        per_bar_iqr = np.subtract(
            np.quantile(q_stacked.reshape(n_critics * n_q, -1) if q_stacked.ndim == 3 else q_stacked, 0.75, axis=0),
            np.quantile(q_stacked.reshape(n_critics * n_q, -1) if q_stacked.ndim == 3 else q_stacked, 0.25, axis=0),
        )
        median_iqr = float(np.median(per_bar_iqr))

        v = verdict(q_stacked)

        print(f"    n_critics={n_critics}  n_quantiles={n_q}  n_obs={len(stk)}")
        print(f"    Q stats: mean={q_mean:+.3f}  median={q_median:+.3f}  std={q_std:.3f}  range=[{q_min:+.3f}, {q_max:+.3f}]")
        print(f"    NaN={n_nan}  Inf={n_inf}")
        print(f"    inter-critic mean-Q std: {inter_critic_std:.4f}")
        print(f"    median IQR per bar: {median_iqr:.4f}")
        print(f"    -> {v}")

        results[asset] = {
            "n_critics": n_critics,
            "n_quantiles": int(n_q),
            "n_obs": int(len(stk)),
            "q_mean": q_mean,
            "q_median": q_median,
            "q_std": q_std,
            "q_min": q_min,
            "q_max": q_max,
            "n_nan": n_nan,
            "n_inf": n_inf,
            "inter_critic_mean_std": inter_critic_std,
            "median_iqr_per_bar": median_iqr,
            "verdict": v,
        }

    OUT.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n  saved: {OUT}")
    return results


if __name__ == "__main__":
    run()
