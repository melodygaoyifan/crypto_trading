"""
HMATS v6.5.2 - Phase 0: GMM Flip Rate Investigation

Analyzes the 6-regime GMM's flip behavior to determine whether the ~30% flip rate
is genuine market signal or artifact of poorly separated clusters.

Steps:
  1. Cluster Proximity Analysis (pairwise centroid distances)
  2. Temporal Flip Analysis (flip rate, ping-pong, transition matrix)
  3. RegimeSmoother experiment (persistence filter before/after)
  4. Recommendation (PROCEED / SMOOTH / MERGE)

Usage:
    python scripts/gmm_flip_analysis.py                      # Use cached data
    python scripts/gmm_flip_analysis.py --fetch               # Fetch fresh data
    python scripts/gmm_flip_analysis.py --min-persistence 3   # Override smoother

Output:
    models/regime_classifier/flip_analysis_report.json
"""

import argparse
import importlib.util
import json
import logging
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

# ── Project setup ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import feature pipeline from retrain_gmm (guarantees identical computation)
_spec = importlib.util.spec_from_file_location(
    "retrain_gmm", PROJECT_ROOT / "scripts" / "retrain_gmm.py"
)
_retrain_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_retrain_mod)

FEATURE_COLS = _retrain_mod.FEATURE_COLS
GMM_DIR = _retrain_mod.GMM_DIR
build_feature_matrix = _retrain_mod.build_feature_matrix
load_data_cache = _retrain_mod.load_data_cache
fetch_all_assets = _retrain_mod.fetch_all_assets
save_data_cache = _retrain_mod.save_data_cache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("gmm_flip_analysis")


# ══════════════════════════════════════════════════════════════════════
# STEP 1: CLUSTER PROXIMITY ANALYSIS
# ══════════════════════════════════════════════════════════════════════

def step1_cluster_proximity(config: dict) -> dict:
    """Compute pairwise distances between GMM cluster centroids in scaled space."""
    means = np.array(config["means"])  # shape: (6, 17)
    regime_names = config["regime_names"]
    feature_cols = config["feature_cols"]
    n_clusters = len(means)

    pairs = []
    for i, j in combinations(range(n_clusters), 2):
        diff = means[i] - means[j]
        dist = float(np.linalg.norm(diff))
        too_close = dist < 2.0

        # Top 3 differentiating features
        abs_diff = np.abs(diff)
        top_idx = np.argsort(abs_diff)[::-1][:3]
        top_feats = [
            {"feature": feature_cols[k], "delta": round(float(diff[k]), 3)}
            for k in top_idx
        ]

        pairs.append({
            "regime_a": regime_names[i],
            "regime_b": regime_names[j],
            "distance": round(dist, 3),
            "too_close": too_close,
            "top_differentiating_features": top_feats,
        })

    pairs.sort(key=lambda x: x["distance"])
    distances = [p["distance"] for p in pairs]
    close_count = sum(1 for p in pairs if p["too_close"])

    # Flag low-variance features (scaler_scale < 0.05)
    scaler_scale = config.get("scaler_scale", [])
    low_var_feats = []
    for idx, sc in enumerate(scaler_scale):
        if sc < 0.05 and idx < len(feature_cols):
            low_var_feats.append({"feature": feature_cols[idx], "scale": round(sc, 4)})

    return {
        "pairs": pairs,
        "min_distance": round(min(distances), 3),
        "max_distance": round(max(distances), 3),
        "mean_distance": round(float(np.mean(distances)), 3),
        "close_pair_count": close_count,
        "low_variance_features": low_var_feats,
    }


# ══════════════════════════════════════════════════════════════════════
# STEP 2: TEMPORAL FLIP ANALYSIS
# ══════════════════════════════════════════════════════════════════════

def step2_temporal_flip_analysis(
    model_path: Path,
    scaler_path: Path,
    config: dict,
    data: dict,
    bars_limit: Optional[int] = None,
) -> dict:
    """Predict regimes for all bars and analyze flip patterns."""
    gmm = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    regime_names = config["regime_names"]

    # Optionally trim data
    if bars_limit:
        trimmed = {}
        for name, df in data.items():
            trimmed[name] = df.tail(bars_limit).reset_index(drop=True)
        data = trimmed

    # Compute features (identical pipeline to retrain_gmm.py)
    features, timestamps, asset_labels = build_feature_matrix(data)

    # Scale + clip (same as main.py runtime)
    X_scaled = scaler.transform(features)
    X_scaled = np.clip(X_scaled, -4.0, 4.0)

    # Predict
    labels_raw = gmm.predict(X_scaled)
    probs = gmm.predict_proba(X_scaled)
    labels = [regime_names[l] for l in labels_raw]
    confidences = probs.max(axis=1)

    total_bars = len(labels)

    # Count flips (respecting asset boundaries)
    flips = []
    for i in range(1, total_bars):
        if asset_labels[i] != asset_labels[i - 1]:
            continue  # Asset boundary, NOT a flip
        if labels[i] != labels[i - 1]:
            flips.append({
                "index": i,
                "asset": asset_labels[i],
                "from": labels[i - 1],
                "to": labels[i],
            })

    # Countable bars (excluding first bar of each asset)
    asset_starts = set()
    prev_asset = None
    for i, a in enumerate(asset_labels):
        if a != prev_asset:
            asset_starts.add(i)
            prev_asset = a
    countable_bars = total_bars - len(asset_starts)

    flip_rate = len(flips) / max(countable_bars, 1)
    bars_per_flip = 1.0 / flip_rate if flip_rate > 0 else float("inf")

    # Ping-pong detection (A->B->A within 3 bars, same asset)
    pingpong_count = 0
    for i in range(2, total_bars):
        if asset_labels[i] != asset_labels[i - 1] or asset_labels[i - 1] != asset_labels[i - 2]:
            continue
        if labels[i] == labels[i - 2] and labels[i] != labels[i - 1]:
            pingpong_count += 1

    pingpong_rate = pingpong_count / max(len(flips), 1)

    # Transition matrix
    transitions = defaultdict(lambda: defaultdict(int))
    for f in flips:
        transitions[f["from"]][f["to"]] += 1

    # Top flip patterns
    flat_transitions = [
        {"from": fr, "to": to, "count": c}
        for fr, row in transitions.items()
        for to, c in row.items()
    ]
    flat_transitions.sort(key=lambda x: -x["count"])
    top_patterns = flat_transitions[:10]

    # Regime distribution
    regime_counts = Counter(labels)
    regime_dist = {
        r: round(regime_counts[r] / total_bars, 3)
        for r in regime_names
        if regime_counts.get(r, 0) > 0
    }

    # Per-asset flip rate
    per_asset_flips = defaultdict(lambda: {"flips": 0, "bars": 0})
    prev_label = {}
    for i, (label, asset) in enumerate(zip(labels, asset_labels)):
        if asset in prev_label:
            per_asset_flips[asset]["bars"] += 1
            if label != prev_label[asset]:
                per_asset_flips[asset]["flips"] += 1
        prev_label[asset] = label

    per_asset_flip_rate = {
        a: round(v["flips"] / max(v["bars"], 1), 3)
        for a, v in per_asset_flips.items()
    }

    # Regime durations (consecutive same-regime runs within asset)
    regime_durations = defaultdict(list)
    current_run_regime = None
    current_run_asset = None
    current_run_len = 0
    for i, (label, asset) in enumerate(zip(labels, asset_labels)):
        if asset != current_run_asset or label != current_run_regime:
            if current_run_regime and current_run_len > 0:
                regime_durations[current_run_regime].append(current_run_len)
            current_run_regime = label
            current_run_asset = asset
            current_run_len = 1
        else:
            current_run_len += 1
    if current_run_regime and current_run_len > 0:
        regime_durations[current_run_regime].append(current_run_len)

    duration_stats = {}
    for regime, durations in regime_durations.items():
        duration_stats[regime] = {
            "mean_bars": round(float(np.mean(durations)), 1),
            "median_bars": round(float(np.median(durations)), 1),
            "max_bars": int(max(durations)),
            "count_runs": len(durations),
        }

    # Confidence stats
    conf_stats = {
        "mean": round(float(confidences.mean()), 4),
        "std": round(float(confidences.std()), 4),
        "min": round(float(confidences.min()), 4),
    }

    return {
        "total_bars": total_bars,
        "countable_bars": countable_bars,
        "total_flips": len(flips),
        "flip_rate": round(flip_rate, 4),
        "bars_per_flip": round(bars_per_flip, 1),
        "ping_pong_count": pingpong_count,
        "ping_pong_rate": round(pingpong_rate, 4),
        "transition_matrix": {fr: dict(to_map) for fr, to_map in transitions.items()},
        "top_flip_patterns": top_patterns,
        "regime_distribution": regime_dist,
        "per_asset_flip_rate": per_asset_flip_rate,
        "regime_durations": duration_stats,
        "confidence_stats": conf_stats,
        # Pass through for Step 3
        "labels": labels,
        "asset_labels": asset_labels,
    }


# ══════════════════════════════════════════════════════════════════════
# STEP 3: REGIME SMOOTHER
# ══════════════════════════════════════════════════════════════════════

class RegimeSmoother:
    """
    Persistence filter: holds current regime until new one persists
    for min_persistence consecutive bars. Causal (no look-ahead).
    """

    def __init__(self, min_persistence: int = 2):
        self.min_persistence = min_persistence

    def smooth(self, labels: list, asset_labels: list) -> list:
        smoothed = []
        current_regime = labels[0]
        pending_regime = None
        pending_count = 0

        for i, (label, asset) in enumerate(zip(labels, asset_labels)):
            # Reset at asset boundary
            if i > 0 and asset_labels[i] != asset_labels[i - 1]:
                current_regime = label
                pending_regime = None
                pending_count = 0
                smoothed.append(label)
                continue

            if label == current_regime:
                pending_regime = None
                pending_count = 0
                smoothed.append(current_regime)
            elif label == pending_regime:
                pending_count += 1
                if pending_count >= self.min_persistence:
                    current_regime = pending_regime
                    pending_regime = None
                    pending_count = 0
                smoothed.append(current_regime)
            else:
                pending_regime = label
                pending_count = 1
                if self.min_persistence <= 1:
                    current_regime = label
                smoothed.append(current_regime)

        return smoothed


def _count_flips(labels: list, asset_labels: list) -> Tuple[int, int]:
    """Count flips and ping-pongs in a label sequence."""
    flips = 0
    pingpongs = 0
    for i in range(1, len(labels)):
        if asset_labels[i] != asset_labels[i - 1]:
            continue
        if labels[i] != labels[i - 1]:
            flips += 1
    for i in range(2, len(labels)):
        if asset_labels[i] != asset_labels[i - 1] or asset_labels[i - 1] != asset_labels[i - 2]:
            continue
        if labels[i] == labels[i - 2] and labels[i] != labels[i - 1]:
            pingpongs += 1
    return flips, pingpongs


def step3_regime_smoother(
    labels: list,
    asset_labels: list,
    min_persistence: int = 2,
) -> dict:
    """Apply RegimeSmoother and compare before/after."""
    # Before
    before_flips, before_pp = _count_flips(labels, asset_labels)
    before_dist = Counter(labels)

    # Compute countable bars
    prev_asset = None
    countable = 0
    for i, a in enumerate(asset_labels):
        if a == prev_asset:
            countable += 1
        prev_asset = a

    before_flip_rate = before_flips / max(countable, 1)

    # Smooth
    smoother = RegimeSmoother(min_persistence=min_persistence)
    smoothed = smoother.smooth(labels, asset_labels)

    # After
    after_flips, after_pp = _count_flips(smoothed, asset_labels)
    after_flip_rate = after_flips / max(countable, 1)
    after_dist = Counter(smoothed)

    flip_reduction = (1 - after_flip_rate / before_flip_rate) * 100 if before_flip_rate > 0 else 0
    pp_reduction = (1 - after_pp / max(before_pp, 1)) * 100 if before_pp > 0 else 0

    # Distribution shift
    all_regimes = sorted(set(list(before_dist.keys()) + list(after_dist.keys())))
    total = len(labels)
    dist_change = {}
    for r in all_regimes:
        before_pct = before_dist.get(r, 0) / total * 100
        after_pct = after_dist.get(r, 0) / total * 100
        dist_change[r] = {
            "before_pct": round(before_pct, 1),
            "after_pct": round(after_pct, 1),
            "delta_pct": round(after_pct - before_pct, 1),
        }

    return {
        "min_persistence": min_persistence,
        "before_flip_rate": round(before_flip_rate, 4),
        "after_flip_rate": round(after_flip_rate, 4),
        "flip_rate_reduction_pct": round(flip_reduction, 1),
        "before_flips": before_flips,
        "after_flips": after_flips,
        "before_ping_pong": before_pp,
        "after_ping_pong": after_pp,
        "ping_pong_reduction_pct": round(pp_reduction, 1),
        "regime_distribution_change": dist_change,
        "smoothed_labels": smoothed,
    }


# ══════════════════════════════════════════════════════════════════════
# STEP 4: RECOMMENDATION
# ══════════════════════════════════════════════════════════════════════

def step4_recommendation(
    proximity: dict,
    flips: dict,
    smoother: dict,
) -> dict:
    """Recommend PROCEED, SMOOTH, or MERGE based on analysis."""
    flip_rate = flips["flip_rate"]
    pp_rate = flips["ping_pong_rate"]
    close_pairs = proximity["close_pair_count"]
    after_flip_rate = smoother["after_flip_rate"]
    after_pp = smoother["after_ping_pong"]
    flip_reduction = smoother["flip_rate_reduction_pct"]
    total_flips = flips["total_flips"]

    reasons = []
    actions = []
    merge_candidates = []

    # Collect close pair names
    if close_pairs > 0:
        merge_candidates = [
            (p["regime_a"], p["regime_b"])
            for p in proximity["pairs"]
            if p["too_close"]
        ]

    # Decision tree
    if close_pairs > 2:
        rec = "MERGE"
        conf = "HIGH"
        reasons.append(f"{close_pairs} cluster pairs too close (< 2.0 in scaled space)")
        actions.append(f"Retrain GMM with n_components=5 or 4")
        actions.append(f"Merge candidates: {merge_candidates}")

    elif flip_rate > 0.40:
        rec = "MERGE"
        conf = "HIGH"
        reasons.append(f"Flip rate {flip_rate:.1%} exceeds 40% threshold")
        actions.append("Retrain GMM with fewer components")

    elif flip_rate < 0.15 and pp_rate < 0.05:
        rec = "PROCEED"
        conf = "HIGH"
        reasons.append(f"Flip rate {flip_rate:.1%} and ping-pong rate {pp_rate:.1%} both low")
        actions.append("No smoothing needed - flips are genuine transitions")

    elif flip_reduction >= 30 and after_flip_rate < 0.25:
        rec = "SMOOTH"
        conf = "HIGH"
        reasons.append(f"Smoother reduces flip rate from {flip_rate:.1%} to {after_flip_rate:.1%} ({flip_reduction:.0f}% reduction)")
        reasons.append(f"Ping-pong reduced from {flips['ping_pong_count']} to {after_pp}")
        actions.append(f"Add RegimeSmoother(min_persistence={smoother['min_persistence']}) to DRL pipeline")
        actions.append("Apply smoothing in both training data AND runtime inference")
        if close_pairs > 0:
            reasons.append(f"{close_pairs} close pair(s) exist but smoothing compensates")

    elif close_pairs == 0 and flip_rate < 0.35:
        rec = "SMOOTH"
        conf = "MEDIUM"
        reasons.append(f"Moderate flip rate {flip_rate:.1%} with well-separated clusters")
        reasons.append("Smoothing recommended as safety net")
        actions.append(f"Add RegimeSmoother(min_persistence={smoother['min_persistence']})")

    else:
        rec = "MERGE"
        conf = "MEDIUM"
        reasons.append(f"High flip rate {flip_rate:.1%} not adequately resolved by smoothing")
        reasons.append(f"After smoothing: {after_flip_rate:.1%} (target: < 25%)")
        actions.append("Retrain GMM with fewer components or increase regularization")

    return {
        "recommendation": rec,
        "confidence": conf,
        "reasons": reasons,
        "suggested_actions": actions,
        "merge_candidates": merge_candidates,
    }


# ══════════════════════════════════════════════════════════════════════
# REPORT PRINTING
# ══════════════════════════════════════════════════════════════════════

def print_report(proximity: dict, flips: dict, smoother: dict, rec: dict) -> None:
    W = 70
    sep = "=" * W
    thin = "-" * W

    def header(title):
        print(f"\n{sep}")
        print(f"  {title}")
        print(sep)

    header("GMM FLIP RATE INVESTIGATION - Phase 0 Report")
    print(f"  Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")

    # Step 1
    header("STEP 1: CLUSTER PROXIMITY ANALYSIS")
    print(f"\n  Pairwise Distances ({len(proximity['pairs'])} pairs):")
    print(f"    {'Pair':50s} {'Dist':>7s}  {'Status':>10s}")
    print(f"    {thin}")
    for p in proximity["pairs"]:
        status = "TOO CLOSE" if p["too_close"] else "OK"
        pair_name = f"{p['regime_a']} <-> {p['regime_b']}"
        print(f"    {pair_name:50s} {p['distance']:>7.2f}  {status:>10s}")

    print(f"\n  Close pairs (< 2.0): {proximity['close_pair_count']} of {len(proximity['pairs'])}")
    print(f"  Distance range: [{proximity['min_distance']}, {proximity['max_distance']}]")
    print(f"  Mean distance: {proximity['mean_distance']}")

    if proximity["close_pair_count"] > 0:
        print(f"\n  Close pair differentiating features:")
        for p in proximity["pairs"]:
            if p["too_close"]:
                print(f"    {p['regime_a']} <-> {p['regime_b']} (dist={p['distance']:.2f}):")
                for f in p["top_differentiating_features"]:
                    print(f"      {f['feature']:25s} delta={f['delta']:+.3f}")

    if proximity["low_variance_features"]:
        print(f"\n  Low-variance features (scale < 0.05):")
        for f in proximity["low_variance_features"]:
            print(f"    {f['feature']:25s} scale={f['scale']}")

    # Step 2
    header("STEP 2: TEMPORAL FLIP ANALYSIS")
    print(f"\n  Total bars analyzed:   {flips['total_bars']}")
    print(f"  Countable transitions: {flips['countable_bars']}")
    print(f"  Total flips:           {flips['total_flips']}")
    print(f"  Flip rate:             {flips['flip_rate']:.1%} (regime changes every ~{flips['bars_per_flip']:.1f} bars / {flips['bars_per_flip']*4:.0f}h)")
    print(f"  Ping-pong flips:       {flips['ping_pong_count']} ({flips['ping_pong_rate']:.1%} of all flips)")
    print(f"  Confidence:            mean={flips['confidence_stats']['mean']:.3f}, std={flips['confidence_stats']['std']:.3f}, min={flips['confidence_stats']['min']:.3f}")

    print(f"\n  Per-asset flip rates:")
    for asset, rate in sorted(flips["per_asset_flip_rate"].items()):
        print(f"    {asset}: {rate:.1%}")

    print(f"\n  Regime distribution:")
    for regime, pct in sorted(flips["regime_distribution"].items(), key=lambda x: -x[1]):
        print(f"    {regime:25s} {pct:.1%}")

    print(f"\n  Regime durations (consecutive bars):")
    for regime, stats in sorted(flips["regime_durations"].items()):
        print(f"    {regime:25s} mean={stats['mean_bars']:.1f}, median={stats['median_bars']:.1f}, max={stats['max_bars']}, runs={stats['count_runs']}")

    print(f"\n  Top 10 Flip Patterns:")
    for i, p in enumerate(flips["top_flip_patterns"][:10]):
        print(f"    {i+1:2d}. {p['from']:25s} -> {p['to']:25s}: {p['count']}x")

    # Step 3
    header(f"STEP 3: REGIME SMOOTHER (min_persistence={smoother['min_persistence']})")
    print(f"\n  Before: flip_rate={smoother['before_flip_rate']:.1%}, flips={smoother['before_flips']}, ping-pong={smoother['before_ping_pong']}")
    print(f"  After:  flip_rate={smoother['after_flip_rate']:.1%}, flips={smoother['after_flips']}, ping-pong={smoother['after_ping_pong']}")
    print(f"  Flip rate reduction:  {smoother['flip_rate_reduction_pct']:.1f}%")
    print(f"  Ping-pong reduction:  {smoother['ping_pong_reduction_pct']:.1f}%")

    print(f"\n  Regime distribution change:")
    for regime, change in sorted(smoother["regime_distribution_change"].items()):
        delta = change["delta_pct"]
        sign = "+" if delta > 0 else ""
        print(f"    {regime:25s} {change['before_pct']:5.1f}% -> {change['after_pct']:5.1f}% ({sign}{delta:.1f}%)")

    # Step 4
    header("STEP 4: RECOMMENDATION")
    rec_text = rec["recommendation"]
    print(f"\n  >>> {rec_text} <<< (confidence: {rec['confidence']})")

    print(f"\n  Reasons:")
    for r in rec["reasons"]:
        print(f"    - {r}")

    print(f"\n  Suggested Actions:")
    for i, a in enumerate(rec["suggested_actions"]):
        print(f"    {i+1}. {a}")

    if rec["merge_candidates"]:
        print(f"\n  Merge Candidates:")
        for a, b in rec["merge_candidates"]:
            print(f"    - {a} + {b}")

    print(f"\n{sep}")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="GMM Flip Rate Investigation (Phase 0)")
    parser.add_argument("--fetch", action="store_true", help="Fetch fresh data from Kraken")
    parser.add_argument("--bars", type=int, default=None, help="Limit bars per asset")
    parser.add_argument("--min-persistence", type=int, default=2, help="RegimeSmoother min_persistence")
    parser.add_argument("--json-only", action="store_true", help="Output JSON only, no report")
    args = parser.parse_args()

    # Load data
    if args.fetch:
        logger.info("Fetching fresh data from Kraken...")
        data = fetch_all_assets()
        save_data_cache(data)
    else:
        logger.info("Loading cached data...")
        data = load_data_cache()

    # Load model config
    config_path = GMM_DIR / "gmm_config.json"
    model_path = GMM_DIR / "gmm_model.pkl"
    scaler_path = GMM_DIR / "scaler.pkl"

    logger.info(f"Loading GMM config from {config_path}")
    with open(config_path) as f:
        config = json.load(f)

    # Step 1: Cluster Proximity
    logger.info("Step 1: Cluster proximity analysis...")
    proximity = step1_cluster_proximity(config)

    # Step 2: Temporal Flip Analysis
    logger.info("Step 2: Temporal flip analysis...")
    flip_results = step2_temporal_flip_analysis(
        model_path, scaler_path, config, data, args.bars,
    )

    # Step 3: RegimeSmoother
    logger.info(f"Step 3: RegimeSmoother (persistence={args.min_persistence})...")
    smoother_results = step3_regime_smoother(
        flip_results["labels"],
        flip_results["asset_labels"],
        min_persistence=args.min_persistence,
    )

    # Step 4: Recommendation
    logger.info("Step 4: Generating recommendation...")
    recommendation = step4_recommendation(proximity, flip_results, smoother_results)

    # Build results (exclude large lists for JSON)
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_config": config_path.name,
        "step1_cluster_proximity": proximity,
        "step2_temporal_flips": {
            k: v for k, v in flip_results.items()
            if k not in ("labels", "asset_labels")
        },
        "step3_regime_smoother": {
            k: v for k, v in smoother_results.items()
            if k != "smoothed_labels"
        },
        "step4_recommendation": recommendation,
    }

    # Output
    if not args.json_only:
        print_report(proximity, flip_results, smoother_results, recommendation)

    # Save JSON
    output_path = GMM_DIR / "flip_analysis_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {output_path}")

    return results


if __name__ == "__main__":
    main()
