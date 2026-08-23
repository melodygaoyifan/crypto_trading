"""[P384] main.py wiring for the integrity-shield REST feed.

The shield (P383: inert when unfed) is now FED the Kraken L2 REST snapshot
the pipeline already fetches, BEFORE the `_shield_fed` read; the P0 abort
(-> sleeve HOLD, P382) fires only on a SUSTAINED per-pair failure (stale
snapshot, or MAX_CONSECUTIVE_FAILURES bad decision-tick snapshots), an
isolated bad snapshot only WARNS; and the [WIRE-SHIELD] diag reads the
PRIMARY (fed) shield, not the never-fed secondary.
"""
from __future__ import annotations

import inspect

import main as m
from tests._guard_pins import assert_guard_live


def _p0_block():
    src = inspect.getsource(m.HMATSProductionRunner._process_4h_tick_inner)
    i = src.index("KRAKEN INTEGRITY SHIELD")
    j = src.index("TASK 3: ENHANCED REGIME NAVIGATOR", i)
    return src[i:j]


class TestShieldIsFedBeforeItIsRead:
    def test_feed_call_precedes_the_is_fed_read(self):
        blk = _p0_block()
        i_feed = blk.index("self.integrity_shield.feed_rest_snapshot(")
        i_read = blk.index("_shield_fed = False")
        assert i_feed < i_read, "feeding after the read leaves the tick reading last tick's state"

    def test_feed_reads_the_pipeline_snapshot_key_and_maps_the_pair(self):
        blk = _p0_block()
        assert 'market_data.get("orderbook_snapshot")' in blk
        assert "self._normalize_kraken_pair(asset)" in blk

    def test_feed_failure_cannot_abort_the_tick(self):
        blk = _p0_block()
        i_feed = blk.index("feed_rest_snapshot(")
        tail = blk[i_feed:i_feed + 1500]
        assert "except Exception as _feed_err:" in tail
        assert "continuing; the shield stays at its last state" in tail
        # the feed block must not set the abort flag
        head = blk[:blk.index("_shield_fed = False")]
        assert "p0_abort_tick = True" not in head


class TestSustainedVsIsolated:
    def test_abort_is_the_whole_condition_on_sustained(self):
        assert_guard_live(_p0_block(), "if orderbook is None or _sustained:",
                          why="P384: the P0 abort must fire only on a SUSTAINED "
                              "per-pair failure; a prefix would silently widen it",
                          near="_pair_reason.startswith(\"stale_snapshot\")")

    def test_sustained_is_per_pair_stale_or_consecutive(self):
        blk = _p0_block()
        assert '_pair_reason.startswith("stale_snapshot")' in blk
        assert "consecutive_failures" in blk and "MAX_CONSECUTIVE_FAILURES" in blk
        assert "health_reasons" in blk

    def test_isolated_failure_only_warns(self):
        blk = _p0_block()
        assert "isolated snapshot failure" in blk
        assert "aborting (P384)" in blk   # the isolated branch proceeds
        i = blk.index("isolated snapshot failure")
        # no abort write between the elif and the else
        seg = blk[blk.rindex("elif", 0, i):i]
        assert "p0_abort_tick = True" not in seg

    def test_the_old_blanket_unhealthy_abort_is_gone(self):
        blk = _p0_block()
        assert "if orderbook is None or not self.integrity_shield.is_healthy():" not in blk


class TestWireShieldDiagReadsThePrimary:
    def test_diag_reads_primary_first(self):
        src = inspect.getsource(m.HMATSProductionRunner._process_4h_tick_inner)
        i = src.index("[WIRE-SHIELD] Kraken Integrity Shield -orderbook health check")
        blk = src[i:i + 2500]
        assert '_wire_shield = getattr(self, "integrity_shield", None) or self._integrity_shield' in blk
        assert "_wire_shield.is_healthy()" in blk
        assert "self._integrity_shield.is_healthy()" not in blk
