"""TEST 5 — Audit recent live DRL inference logs.

Greps the most recent .log files in logs/ for DRL action records.
This is GROUND TRUTH from production — not a re-simulation.

Patterns scanned (broad — production log format varies):
  - [AGENT-TRACE] ... drl=...
  - drl_direction=...
  - DRL] action=...
  - DRL_INFERENCE ...
  - [DRL_E2E] ...

If no DRL log lines are found, this test SKIPs gracefully — production
might not have run since logs were rotated.
"""
from __future__ import annotations
import glob
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[2]
OUT = Path("/tmp/hmats_audit/drl_drift/test5_live_action_log.json")
OUT.parent.mkdir(parents=True, exist_ok=True)


# Match a few common formats. First wins per line.
PATTERNS = [
    re.compile(r"drl_direction[=:\s]+([-+]?\d+\.?\d*)"),
    re.compile(r"drl_dir[=:\s]+([-+]?\d+\.?\d*)"),
    re.compile(r"\[DRL\][^\n]*?action[=:\s]+([-+]?\d+\.?\d*)"),
    re.compile(r"\[AGENT-TRACE\][^\n]*?drl[=:]([-+]?\d+\.?\d*)"),
    re.compile(r"DRL_INFERENCE[^\n]*?action[=:\s]+([-+]?\d+\.?\d*)"),
    re.compile(r"\bdrl_action[=:\s]+([-+]?\d+\.?\d*)"),
]

ASSET_PATTERN = re.compile(r"\b(BTC|ETH|SOL)\b")
BINS = np.linspace(-1.0, 1.0, 9)


def hist(actions: np.ndarray) -> np.ndarray:
    binned = np.digitize(actions, BINS)
    counts = np.bincount(binned, minlength=len(BINS) + 1)[1 : len(BINS)]
    return counts / max(1, counts.sum())


def run():
    log_paths = sorted(
        glob.glob(str(PROJ / "logs" / "*.log"))
        + glob.glob(str(PROJ / "data" / "**" / "*.jsonl"), recursive=True),
        key=lambda f: Path(f).stat().st_mtime,
        reverse=True,
    )[:8]  # Most recent 8

    print(f"  Scanning {len(log_paths)} log files...")
    actions_per_asset: dict = {"BTC": [], "ETH": [], "SOL": [], "_unknown": []}
    total_lines_scanned = 0

    for fpath in log_paths:
        size_mb = Path(fpath).stat().st_size / (1024 * 1024)
        print(f"    {fpath} ({size_mb:.1f} MB)")
        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    total_lines_scanned += 1
                    val = None
                    for pat in PATTERNS:
                        m = pat.search(line)
                        if m:
                            try:
                                val = float(m.group(1))
                            except ValueError:
                                continue
                            break
                    if val is None or not (-2.0 <= val <= 2.0):
                        continue
                    asset_match = ASSET_PATTERN.search(line)
                    asset = asset_match.group(1) if asset_match else "_unknown"
                    actions_per_asset[asset].append(val)
        except Exception as e:
            print(f"      [READ-FAIL] {e}")
            continue

    total_actions = sum(len(v) for v in actions_per_asset.values())
    print(f"\n  Scanned {total_lines_scanned:,} lines, found {total_actions} DRL action records")

    if total_actions == 0:
        print("  [SKIP] No DRL action log lines found in any scanned log.")
        OUT.write_text(json.dumps({"verdict": "NO_DATA_SKIP", "lines_scanned": total_lines_scanned}, indent=2), encoding="utf-8")
        return {"verdict": "NO_DATA_SKIP"}

    results = {"verdict_per_asset": {}, "lines_scanned": total_lines_scanned, "total_actions": total_actions}

    for asset, vals in actions_per_asset.items():
        if len(vals) < 5:
            continue
        arr = np.asarray(vals)
        h = hist(arr)
        zero_freq = float((np.abs(arr) < 0.1).mean())
        if zero_freq > 0.7:
            v = f"MOSTLY_FLAT ({zero_freq:.0%} flat)"
        elif h.max() > 0.6:
            v = f"COLLAPSED (single bin {h.max():.0%})"
        else:
            v = "HEALTHY"

        print(f"\n    {asset}: n={len(arr)}  mean={arr.mean():+.3f}  std={arr.std():.3f}")
        print(f"      distribution: {' '.join(f'{f:.2f}' for f in h)}")
        print(f"      -> {v}")

        results["verdict_per_asset"][asset] = {
            "n": int(len(arr)),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "distribution": h.tolist(),
            "verdict": v,
        }

    OUT.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\n  saved: {OUT}")
    return results


if __name__ == "__main__":
    run()
