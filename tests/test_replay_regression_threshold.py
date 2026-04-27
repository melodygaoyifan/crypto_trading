"""
test_replay_regression_threshold.py — gate for replay regression activation
=============================================================================

[P111 Tier3#7 2026-04-27] Placeholder for replay regression suite.

REPLAY regression replays historical signals + fills through the engine
in dry-run mode and asserts the same intents fire. It needs SUFFICIENT
DATA to be meaningful — with only ~10 closed trades, any test would
be noise. Threshold to activate: 100 closed trades minimum.

This test:
  1. Counts closed trades in trade_attribution.jsonl
  2. PASSES when count < 100 (not enough data yet)
  3. PASSES with WARNING when 100 ≤ count < 500 (borderline)
  4. FAILS when count ≥ 500 if no replay suite has been built

The fail-after-500 is the forcing function: as production accumulates
data, this test eventually demands we build the actual replay suite.

How to build the replay suite when count crosses ≥100:
  - Read trade_attribution.jsonl + signals_*.jsonl from same time window
  - For each closed trade, reconstruct agent_signals at entry_time
  - Feed into a dry-run instance of integration_v36.decide()
  - Assert resulting intent matches what was actually executed
    (direction, target_exposure within ±10% tolerance)
  - Any divergence = silent regression in fusion / gate / sizing logic
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CANDIDATES = [
    REPO / "data" / "trade_attribution.jsonl",
    Path("/opt/hmats/data/trade_attribution.jsonl"),
    Path("/var/lib/docker/volumes/hmats-data/_data/trade_attribution.jsonl"),
]

THRESHOLD_BUILD_NOW = 500
THRESHOLD_WARN = 100


def _count_closed_trades() -> int:
    for path in CANDIDATES:
        if path.exists():
            try:
                count = 0
                with open(path) as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if rec.get("is_closed") or rec.get("closed"):
                            count += 1
                return count
            except OSError:
                continue
    return 0


def test_replay_suite_data_volume_gate():
    """Trigger replay-suite build when closed-trade count exceeds threshold."""
    count = _count_closed_trades()

    if count == 0:
        pytest.skip(
            "No trade_attribution.jsonl found — running outside production "
            "or pre-data state. Skipping volume gate."
        )

    if count < THRESHOLD_WARN:
        # Below warn threshold — replay regression isn't useful yet.
        return  # PASS silently

    if count < THRESHOLD_BUILD_NOW:
        warnings.warn(
            f"Closed-trade count ({count}) crossed warn threshold "
            f"({THRESHOLD_WARN}). Replay regression suite would now provide "
            f"meaningful signal. Schedule build before count reaches "
            f"{THRESHOLD_BUILD_NOW} (this test will FAIL at that point).",
            UserWarning,
        )
        return  # PASS with warning

    # ≥ THRESHOLD_BUILD_NOW: forcing function — replay suite must exist.
    replay_suite_marker = REPO / "tests" / "test_replay_regression_active.py"
    assert replay_suite_marker.exists(), (
        f"Closed-trade count ({count}) >= {THRESHOLD_BUILD_NOW}. "
        f"Replay regression suite is now overdue. "
        f"Build tests/test_replay_regression_active.py per the design "
        f"docstring in test_replay_regression_threshold.py."
    )


if __name__ == "__main__":
    n = _count_closed_trades()
    print(f"Closed trades found: {n}")
    print(f"Build-now threshold: {THRESHOLD_BUILD_NOW}")
    print(f"Status: {'BUILD REQUIRED' if n >= THRESHOLD_BUILD_NOW else 'OK'}")
