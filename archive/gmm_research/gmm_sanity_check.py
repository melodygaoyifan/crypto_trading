#!/usr/bin/env python3
"""
GMM Sanity Check - Verify regime labels on known market periods.

Checks:
  1. COVID crash (2020-03): expect PANIC_SELLOFF + EXTREME_VOLATILITY
  2. LUNA/UST crash (2022-05): expect PANIC_SELLOFF + EXTREME_VOLATILITY
  3. FTX crash (2022-11): expect PANIC_SELLOFF + EXTREME_VOLATILITY
  4. Bull peak (2021-10/11): expect MOMENTUM_RALLY
  5. Sideways (2024-Q2): expect QUIET_ACCUMULATION / WEAK_CONSOLIDATION
  6. Per-regime confidence distribution
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load GMM config
config_path = PROJECT_ROOT / "data" / "gmm_models" / "gmm_config.json"
with open(config_path) as f:
    config = json.load(f)
regime_names = config["regime_names"]

print("Regime mapping:")
for i, name in enumerate(regime_names):
    print(f"  {i} -> {name}")

SEP = "=" * 60
PANIC_IDX = regime_names.index("PANIC_SELLOFF")
EXTREME_IDX = regime_names.index("EXTREME_VOLATILITY")
RALLY_IDX = regime_names.index("MOMENTUM_RALLY")
QUIET_IDX = regime_names.index("QUIET_ACCUMULATION")
WEAK_IDX = regime_names.index("WEAK_CONSOLIDATION")

issues = []


def check_period(df, ts, label, start, end, expect="bear"):
    """Check regime distribution during a specific period."""
    mask = (ts >= start) & (ts <= end)
    if mask.sum() == 0:
        print(f"\n  {label}: No data")
        return

    sub = df[mask]
    n = len(sub)
    regimes = sub["regime"].value_counts()
    close = sub["close"]

    print(f"\n  {label}: {n} bars")
    print(f"    Price: {close.iloc[0]:.0f} -> {close.min():.0f} (low) -> {close.iloc[-1]:.0f}")
    for idx, count in sorted(regimes.items()):
        nm = regime_names[idx] if idx < len(regime_names) else f"R{idx}"
        print(f"    {nm}: {count} ({count/n*100:.0f}%)")

    if expect == "bear":
        bear_count = regimes.get(PANIC_IDX, 0) + regimes.get(EXTREME_IDX, 0)
        bear_pct = bear_count / n * 100
        verdict = "OK" if bear_pct > 30 else "WARN"
        print(f"    -> PANIC+EXTREME: {bear_pct:.0f}% [{verdict}]")
        if verdict == "WARN":
            issues.append(f"{label}: PANIC+EXTREME only {bear_pct:.0f}% (expected >30%)")
    elif expect == "bull":
        rally_count = regimes.get(RALLY_IDX, 0)
        rally_pct = rally_count / n * 100
        verdict = "OK" if rally_pct > 20 else "WARN"
        print(f"    -> MOMENTUM_RALLY: {rally_pct:.0f}% [{verdict}]")
        if verdict == "WARN":
            issues.append(f"{label}: MOMENTUM_RALLY only {rally_pct:.0f}% (expected >20%)")
    elif expect == "sideways":
        calm_count = regimes.get(QUIET_IDX, 0) + regimes.get(WEAK_IDX, 0)
        calm_pct = calm_count / n * 100
        verdict = "OK" if calm_pct > 30 else "WARN"
        print(f"    -> QUIET+WEAK: {calm_pct:.0f}% [{verdict}]")


for asset in ["BTC", "ETH", "SOL"]:
    path = PROJECT_ROOT / "data" / "drl_training" / f"{asset}_4H_full.parquet"
    df = pd.read_parquet(path)
    ts = pd.to_datetime(df["timestamp"])

    print(f"\n{SEP}")
    print(f"{asset}: {len(df)} bars, {ts.iloc[0]} -> {ts.iloc[-1]}")
    print(SEP)

    # Overall distribution
    regime_dist = df["regime"].value_counts().sort_index()
    print("\nOverall regime distribution:")
    for idx, count in regime_dist.items():
        name = regime_names[idx] if idx < len(regime_names) else f"R{idx}"
        pct = count / len(df) * 100
        print(f"  {name:25s}: {count:5d} ({pct:5.1f}%)")

    # Crash periods
    check_period(df, ts, "COVID Crash (2020-03)", "2020-03-01", "2020-03-31", "bear")
    check_period(df, ts, "LUNA/UST Crash (2022-05)", "2022-05-01", "2022-05-31", "bear")
    check_period(df, ts, "FTX Crash (2022-11)", "2022-11-01", "2022-11-30", "bear")

    # Bull periods
    check_period(df, ts, "Bull Peak (2021-10/11)", "2021-10-15", "2021-11-15", "bull")
    check_period(df, ts, "2024 Rally (2024-02/03)", "2024-02-01", "2024-03-15", "bull")

    # Sideways
    check_period(df, ts, "Sideways (2024-Q2)", "2024-04-01", "2024-06-30", "sideways")

    # Per-regime confidence
    print("\n  Confidence by regime:")
    for r_idx in range(6):
        mask = df["regime"] == r_idx
        if mask.sum() > 0:
            conf = df.loc[mask, "regime_confidence"]
            nm = regime_names[r_idx]
            print(f"    {nm:25s}: mean={conf.mean():.3f}, std={conf.std():.3f}, "
                  f"min={conf.min():.3f}, n={mask.sum()}")

# Summary
print(f"\n{SEP}")
print("GMM SANITY CHECK SUMMARY")
print(SEP)
if issues:
    print(f"\n  {len(issues)} WARNINGS:")
    for issue in issues:
        print(f"    - {issue}")
else:
    print("\n  ALL CHECKS PASSED")
print(SEP)
