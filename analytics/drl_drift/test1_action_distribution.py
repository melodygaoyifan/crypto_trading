"""TEST 1 — Action distribution on recent 180d.

Healthy: actions spread across [-1, 1] with reasonable variance.
Drifted: actions collapse to a single bin (e.g. 99% at 0.0 or 95% at +1.0).

We ALSO compare recent 180d distribution to an older 360-540d window
(roughly the training-time period) using KL divergence on the bin frequencies.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

from analytics.drl_drift._common import (
    ASSETS,
    build_obs_matrix,
    load_drl_inference,
    predict_actions,
    stack_frames,
)

OUT = Path("/tmp/hmats_audit/drl_drift/test1_action_distribution.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

# 8 bins covering [-1, 1]
BINS = np.linspace(-1.0, 1.0, 9)


def hist(actions: np.ndarray) -> np.ndarray:
    binned = np.digitize(actions, BINS)
    counts = np.bincount(binned, minlength=len(BINS) + 1)[1 : len(BINS)]
    return counts / max(1, counts.sum())


def kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()
    return float((p * np.log(p / q)).sum())


def verdict(freqs: np.ndarray, actions: np.ndarray) -> str:
    max_bin = float(freqs.max())
    active = float((np.abs(actions) > 0.1).mean())
    if max_bin > 0.80:
        return f"COLLAPSED (single bin {max_bin:.0%})"
    if active < 0.20:
        return f"INACTIVE ({1 - active:.0%} flat)"
    if active > 0.95:
        return f"OVER-ACTIVE ({active:.0%} non-flat)"
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

        obs_recent, ts_recent = build_obs_matrix(asset, days_back=180)
        obs_train, ts_train = build_obs_matrix(asset, days_start=540, days_end=180)

        if len(obs_recent) < 50 or len(obs_train) < 50:
            results[asset] = {"error": f"insufficient bars (recent={len(obs_recent)}, train={len(obs_train)})"}
            continue

        stk_recent = stack_frames(obs_recent)
        stk_train = stack_frames(obs_train)

        a_recent = predict_actions(model, stk_recent)
        a_train = predict_actions(model, stk_train)

        h_recent = hist(a_recent)
        h_train = hist(a_train)
        kl_div = kl(h_recent, h_train)
        kl_rev = kl(h_train, h_recent)
        js_div = 0.5 * (kl(h_recent, 0.5 * (h_recent + h_train)) + kl(h_train, 0.5 * (h_recent + h_train)))

        v = verdict(h_recent, a_recent)
        v_train = verdict(h_train, a_train)
        print(f"    n_recent={len(a_recent)}  n_train={len(a_train)}")
        print(f"    recent : mean={a_recent.mean():+.3f}  std={a_recent.std():.3f}  active={(np.abs(a_recent)>0.1).mean():.0%}")
        print(f"    train  : mean={a_train.mean():+.3f}   std={a_train.std():.3f}   active={(np.abs(a_train)>0.1).mean():.0%}")
        print(f"    KL(recent||train)={kl_div:.4f}   JS={js_div:.4f}")
        print(f"    recent verdict: {v}")
        print(f"    train  verdict: {v_train}")

        results[asset] = {
            "n_recent": int(len(a_recent)),
            "n_train": int(len(a_train)),
            "recent_window": [str(ts_recent.iloc[0]), str(ts_recent.iloc[-1])],
            "train_window": [str(ts_train.iloc[0]), str(ts_train.iloc[-1])],
            "recent_action_mean": float(a_recent.mean()),
            "recent_action_std": float(a_recent.std()),
            "recent_active_fraction": float((np.abs(a_recent) > 0.1).mean()),
            "train_action_mean": float(a_train.mean()),
            "train_action_std": float(a_train.std()),
            "train_active_fraction": float((np.abs(a_train) > 0.1).mean()),
            "recent_hist": h_recent.tolist(),
            "train_hist": h_train.tolist(),
            "kl_recent_vs_train": kl_div,
            "kl_train_vs_recent": kl_rev,
            "js_divergence": js_div,
            "recent_verdict": v,
            "train_verdict": v_train,
        }

    OUT.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n  saved: {OUT}")
    return results


if __name__ == "__main__":
    run()
