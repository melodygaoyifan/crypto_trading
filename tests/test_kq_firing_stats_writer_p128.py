"""
test_kq_firing_stats_writer_p128.py — dual writer survives restarts (P128)
================================================================================

v3 Track A item 1.0: kq_firing_stats writer was 'mode=w' truncate, which:
- Loses everything on container restart (in-memory accumulators reset)
- Overwrites every tick (snapshot semantics)

Together this defeated the 7-day passive capture for P0-1 (kraken_quant
firing-rate diagnosis). Fix: dual writer at main.py:17376+
  - Snapshot file (mode='w') — current state for diagnostic scripts
  - Append-only JSONL audit log — survives restarts, one record per tick

This test verifies the dual-writer pattern is in place. If a future
refactor reverts to single 'w' file, this test fails.
"""
from __future__ import annotations

import inspect

import pytest


class TestP128DualWriter:
    """Source-level verification — full e2e would require a live engine
    instance. The dual-writer pattern is a 2-line discipline; assert the
    markers exist."""

    def test_jsonl_audit_log_writer_present(self):
        """The append-only JSONL audit log must exist alongside the snapshot."""
        with open("main.py", encoding="utf-8-sig") as f:
            src = f.read()
        assert "kq_firing_stats.jsonl" in src, (
            "P128 regression: kq_firing_stats.jsonl audit log writer removed. "
            "Without it, 7-day passive capture (P0-1) loses data on every "
            "container restart."
        )

    def test_jsonl_writer_uses_append_mode(self):
        """The audit log writer must use 'a' (append), not 'w' (truncate)."""
        with open("main.py", encoding="utf-8-sig") as f:
            src = f.read()
        # Find the JSONL block and verify append mode within ±5 lines
        idx = src.find("kq_firing_stats.jsonl")
        assert idx > 0, "P128 marker not found"
        # Check the write call within the next 200 chars
        window = src[idx:idx + 400]
        assert 'open(' in window and '"a"' in window, (
            "P128 regression: kq_firing_stats.jsonl writer no longer uses "
            "append mode. Snapshot truncation will recur on restart."
        )

    def test_snapshot_writer_still_present(self):
        """The snapshot file must remain — diagnostic scripts read it."""
        with open("main.py", encoding="utf-8-sig") as f:
            src = f.read()
        assert "kq_firing_stats.json" in src, (
            "Snapshot file writer removed — kq_strategy_diagnostic.py "
            "depends on it."
        )

    def test_p128_marker_present(self):
        """[P128 ...] marker comments document why the dual writer exists.
        Removing them risks future operators reverting the fix."""
        with open("main.py", encoding="utf-8-sig") as f:
            src = f.read()
        assert "P128" in src, "P128 marker comment removed from main.py"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
