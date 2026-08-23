"""[P383] The LIVE CORRELATION_COLLAPSE NO_TRADE trigger carries the two
conjuncts P253d believed it had.

Background: `defense/constitution.py::NoTradeTriggerChecker` (the live
checker, via integration_v36 -> compute_no_trade_triggers) fired
CORRELATION_COLLAPSE on `market_data['correlation_btc_eth_sol'] >= 0.92`
ALONE. P253d armed the real 20-bar correlation measure believing the trigger
"ALSO requires all-three-same-direction > 0.2 AND no validated edge" — those
conjuncts existed only in `signals/no_trade_triggers.py` (NOT the live path,
per its own P287 header). P382 measured the corr-alone form at >= 0.92 on
7.8% of bars (17% in the last year) and saw it fire live 2026-08-19 16:02 and
20:02 on all three assets; it classified the subtype as sleeve-HOLD as a
stopgap. P383 ports the conjuncts into the live checker.

Ported semantics (quoted from signals/no_trade_triggers.py):

    all_same = ((btc > 0.2 and eth > 0.2 and sol > 0.2) or
                (btc < -0.2 and eth < -0.2 and sol < -0.2))
    fire iff correlation >= thr and not has_validated_edge and all_same

Inputs (market_data, signal_data fallback):
    cross_asset_directions : {"BTC": d, "ETH": d, "SOL": d} floats
    has_validated_edge     : bool, ABSENT reads False (no exemption claimed)

FAIL DIRECTION pinned here:
  * direction map ABSENT / not a mapping / short of 3 assets / non-numeric
    -> the trigger does NOT fire (absence is not evidence of alignment, P2)
    and a WARNING is logged ONCE per process per reason;
  * the continuous `trigger_scores['correlation_collapse']` is UNCHANGED by
    the conjuncts — only the verdict (`active_conditions`) gained them;
  * every other trigger (DVOL, liquidity, stale data, ...) is untouched.
"""
from __future__ import annotations

import logging
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from defense.constitution import (  # noqa: E402
    NoTradeTriggerChecker,
    NoTradeTriggerType,
)

THR = NoTradeTriggerChecker.CORRELATION_COLLAPSE_THRESHOLD
KEY = NoTradeTriggerChecker.CORRELATION_DIRECTIONS_KEY
EDGE_KEY = NoTradeTriggerChecker.CORRELATION_VALIDATED_EDGE_KEY

ALIGNED_LONG = {"BTC": 0.6, "ETH": 0.5, "SOL": 0.7}
ALIGNED_SHORT = {"BTC": -0.6, "ETH": -0.5, "SOL": -0.7}
MIXED = {"BTC": 0.6, "ETH": -0.5, "SOL": 0.7}
WEAK = {"BTC": 0.6, "ETH": 0.1, "SOL": 0.7}  # ETH inside the +-0.2 band


def _fired(state) -> bool:
    return any(c.trigger_type == NoTradeTriggerType.CORRELATION_COLLAPSE
               for c in state.active_conditions)


def _md(corr: float, dirs=None, edge=None, **extra):
    md = {"current_price": 100_000.0, "correlation_btc_eth_sol": corr}
    if dirs is not None:
        md[KEY] = dirs
    if edge is not None:
        md[EDGE_KEY] = edge
    md.update(extra)
    return md


@pytest.fixture
def checker():
    return NoTradeTriggerChecker()


# ---------------------------------------------------------------------------
# Contract pins (the constants a future edit would silently move)
# ---------------------------------------------------------------------------

class TestContract:
    def test_threshold_and_constants_unchanged(self):
        # P383 ports conjuncts; it does NOT move the corr threshold.
        assert NoTradeTriggerChecker.CORRELATION_COLLAPSE_THRESHOLD == 0.92
        assert NoTradeTriggerChecker.CORRELATION_ALIGNMENT_DIRECTION_MIN == 0.2
        assert NoTradeTriggerChecker.CORRELATION_ALIGNMENT_ASSETS == (
            "BTC", "ETH", "SOL")
        assert KEY == "cross_asset_directions"
        assert EDGE_KEY == "has_validated_edge"

    def test_parallel_module_threshold_parity(self):
        # The two implementations share the numbers (P287 warns threshold
        # drift between the twins is silent). Direction band and corr
        # threshold must agree with the module the conjuncts were ported from.
        from signals import no_trade_triggers as twin
        assert twin.NoTradeTriggerChecker.CORRELATION_COLLAPSE_THRESHOLD == THR


# ---------------------------------------------------------------------------
# The 2x2x2 truth table: corr {above, below} x aligned {yes, no} x edge {no, yes}
# ---------------------------------------------------------------------------

class TestTruthTable:
    @pytest.mark.parametrize("corr", [THR, 0.95, 0.99])
    @pytest.mark.parametrize("dirs", [ALIGNED_LONG, ALIGNED_SHORT])
    def test_fires_only_on_all_three_conjuncts(self, checker, corr, dirs):
        state = checker.compute_triggers(_md(corr, dirs, edge=False), {})
        assert _fired(state), "corr >= thr AND aligned AND no edge must fire"
        cond = [c for c in state.active_conditions
                if c.trigger_type == NoTradeTriggerType.CORRELATION_COLLAPSE][0]
        assert "same direction" in cond.details
        assert "no validated edge" in cond.details

    @pytest.mark.parametrize("corr", [THR, 0.99])
    @pytest.mark.parametrize("dirs", [ALIGNED_LONG, ALIGNED_SHORT])
    def test_validated_edge_exempts(self, checker, corr, dirs):
        state = checker.compute_triggers(_md(corr, dirs, edge=True), {})
        assert not _fired(state)

    @pytest.mark.parametrize("corr", [THR, 0.99])
    @pytest.mark.parametrize("dirs", [MIXED, WEAK])
    @pytest.mark.parametrize("edge", [False, True])
    def test_not_aligned_never_fires(self, checker, corr, dirs, edge):
        state = checker.compute_triggers(_md(corr, dirs, edge=edge), {})
        assert not _fired(state)

    @pytest.mark.parametrize("corr", [0.0, 0.5, 0.919])
    @pytest.mark.parametrize("dirs", [ALIGNED_LONG, ALIGNED_SHORT, MIXED])
    @pytest.mark.parametrize("edge", [False, True])
    def test_below_threshold_never_fires(self, checker, corr, dirs, edge):
        state = checker.compute_triggers(_md(corr, dirs, edge=edge), {})
        assert not _fired(state)

    def test_edge_key_absent_reads_as_no_edge(self, checker):
        # Absence of an EXEMPTION is not a fabricated hazard — it declines to
        # exempt, exactly as the parallel module's `.get(..., False)`.
        state = checker.compute_triggers(_md(0.95, ALIGNED_LONG), {})
        assert _fired(state)

    def test_direction_band_is_strict(self, checker):
        # |d| == 0.2 is INSIDE the band (the port uses strict >), so a map
        # sitting exactly on the band is not aligned.
        on_band = {"BTC": 0.2, "ETH": 0.9, "SOL": 0.9}
        assert not _fired(checker.compute_triggers(_md(0.95, on_band), {}))
        just_out = {"BTC": 0.2000001, "ETH": 0.9, "SOL": 0.9}
        assert _fired(checker.compute_triggers(_md(0.95, just_out), {}))


# ---------------------------------------------------------------------------
# Fail direction: an unusable direction map never fires and says so once
# ---------------------------------------------------------------------------

class TestAbsenceIsNotAlignment:
    @pytest.mark.parametrize("corr", [THR, 0.99])
    def test_absent_map_does_not_fire(self, checker, corr):
        state = checker.compute_triggers(_md(corr), {})
        assert not _fired(state)
        # The continuous score is still produced — only the verdict is gated.
        assert state.trigger_scores["correlation_collapse"] >= 0.0

    def test_absent_map_logs_once_per_process(self, checker, caplog):
        with caplog.at_level(logging.WARNING, logger="defense.constitution"):
            checker.compute_triggers(_md(0.95), {})
            checker.compute_triggers(_md(0.95), {})
            checker.compute_triggers(_md(0.5), {})
        msgs = [r.getMessage() for r in caplog.records
                if "[P383][CORRELATION_COLLAPSE]" in r.getMessage()]
        assert len(msgs) == 1, msgs
        assert "directions_absent" in msgs[0]
        assert KEY in msgs[0]
        assert "INERT" in msgs[0]

    def test_absent_map_is_reported_on_first_tick_even_below_threshold(
            self, checker, caplog):
        # A missing producer must be visible at boot, not only on the first
        # >= 0.92 tick weeks later.
        with caplog.at_level(logging.WARNING, logger="defense.constitution"):
            checker.compute_triggers(_md(0.5), {})
        assert any("[P383][CORRELATION_COLLAPSE]" in r.getMessage()
                   for r in caplog.records)

    def test_each_distinct_reason_logs_once(self, checker, caplog):
        # A single bool latch would let the first reason consume the one shot
        # for all of them (P193/P202) — the latch is keyed per reason.
        with caplog.at_level(logging.WARNING, logger="defense.constitution"):
            checker.compute_triggers(_md(0.95), {})                          # absent
            checker.compute_triggers(_md(0.95, {"BTC": 0.9, "ETH": 0.9}), {})  # incomplete
            checker.compute_triggers(_md(0.95, {"BTC": 0.9, "ETH": 0.9}), {})  # again
            checker.compute_triggers(_md(0.95, "not-a-map"), {})              # not a mapping
        msgs = [r.getMessage() for r in caplog.records
                if "[P383][CORRELATION_COLLAPSE]" in r.getMessage()]
        assert len(msgs) == 3, msgs
        assert any("directions_absent" in m for m in msgs)
        assert any("directions_incomplete" in m for m in msgs)
        assert any("directions_not_a_mapping" in m for m in msgs)

    @pytest.mark.parametrize("dirs", [
        {"BTC": 0.9, "ETH": 0.9},                     # fewer than 3 assets
        {"BTC": 0.9},
        {},
        {"BTC": 0.9, "ETH": 0.9, "XRP": 0.9},         # 3 keys but not the roster
        {"BTC": 0.9, "ETH": 0.9, "SOL": None},        # None entry
        {"BTC": 0.9, "ETH": 0.9, "SOL": "up"},        # non-numeric
        {"BTC": 0.9, "ETH": 0.9, "SOL": math.nan},    # NaN
        "0.9,0.9,0.9",                                # not a mapping
        [0.9, 0.9, 0.9],
    ])
    def test_unusable_maps_never_fire(self, checker, dirs):
        assert not _fired(checker.compute_triggers(_md(0.99, dirs), {}))

    def test_usable_map_does_not_log(self, checker, caplog):
        with caplog.at_level(logging.WARNING, logger="defense.constitution"):
            checker.compute_triggers(_md(0.95, ALIGNED_LONG), {})
            checker.compute_triggers(_md(0.95, MIXED), {})
        assert not any("[P383][CORRELATION_COLLAPSE]" in r.getMessage()
                       for r in caplog.records)

    def test_extra_assets_in_map_are_ignored(self, checker):
        # The roster is BTC/ETH/SOL; a fourth asset does not widen or block.
        dirs = dict(ALIGNED_LONG, XRP=-0.9)
        assert _fired(checker.compute_triggers(_md(0.95, dirs), {}))

    def test_numeric_strings_are_accepted(self, checker):
        dirs = {"BTC": "0.6", "ETH": "0.5", "SOL": "0.7"}
        assert _fired(checker.compute_triggers(_md(0.95, dirs), {}))


# ---------------------------------------------------------------------------
# Input routing: market_data first, signal_data fallback
# ---------------------------------------------------------------------------

class TestInputRouting:
    def test_signal_data_fallback_for_directions(self, checker):
        state = checker.compute_triggers(_md(0.95), {KEY: ALIGNED_LONG})
        assert _fired(state)

    def test_signal_data_fallback_for_edge(self, checker):
        state = checker.compute_triggers(
            _md(0.95, ALIGNED_LONG), {EDGE_KEY: True})
        assert not _fired(state)

    def test_market_data_wins_over_signal_data(self, checker):
        state = checker.compute_triggers(
            _md(0.95, MIXED), {KEY: ALIGNED_LONG})
        assert not _fired(state)


# ---------------------------------------------------------------------------
# The continuous score and the other triggers are untouched
# ---------------------------------------------------------------------------

class TestOnlyTheVerdictChanged:
    @pytest.mark.parametrize("corr", [0.5, THR, 0.95, 0.99])
    def test_corr_score_is_independent_of_the_conjuncts(self, checker, corr):
        a = checker.compute_triggers(_md(corr, ALIGNED_LONG), {})
        b = checker.compute_triggers(_md(corr, MIXED), {})
        c = checker.compute_triggers(_md(corr), {})
        d = checker.compute_triggers(_md(corr, ALIGNED_LONG, edge=True), {})
        scores = {s.trigger_scores["correlation_collapse"] for s in (a, b, c, d)}
        assert len(scores) == 1, scores
        expected = 0.0 if corr < THR else min(
            1.0, (corr - THR) / (1.0 - THR + 1e-9))
        assert abs(scores.pop() - expected) < 1e-9

    def test_below_threshold_score_is_zero(self, checker):
        s = checker.compute_triggers(_md(0.7, ALIGNED_LONG), {})
        assert s.trigger_scores["correlation_collapse"] == 0.0

    def test_dvol_extreme_still_fires_without_a_direction_map(self, checker):
        s = checker.compute_triggers(_md(0.5, dvol_zscore=6.0), {})
        assert any(c.trigger_type == NoTradeTriggerType.EXTREME_DVOL
                   for c in s.active_conditions)

    def test_liquidity_critical_still_fires_without_a_direction_map(
            self, checker):
        s = checker.compute_triggers(
            _md(0.5, orderbook_depth_1pct_usd=50_000.0), {})
        assert any(c.trigger_type == NoTradeTriggerType.LIQUIDITY_CRITICAL
                   for c in s.active_conditions)

    def test_stale_data_still_fires_without_a_direction_map(self, checker):
        s = checker.compute_triggers(_md(0.5, data_age_seconds=120.0), {})
        assert any(c.trigger_type == NoTradeTriggerType.STALE_DATA
                   for c in s.active_conditions)

    def test_an_unusable_map_does_not_suppress_other_triggers(self, checker):
        # The one-shot warning path must not raise or short-circuit the rest
        # of compute_triggers.
        s = checker.compute_triggers(
            _md(0.99, "garbage", dvol_zscore=6.0), {})
        assert not _fired(s)
        assert any(c.trigger_type == NoTradeTriggerType.EXTREME_DVOL
                   for c in s.active_conditions)
        assert s.should_no_trade

    def test_no_correlation_key_at_all_is_unchanged(self, checker):
        # Pre-P383 the whole block was skipped when no correlation was present
        # (the `if correlation > 0` guard); that stays, and nothing logs.
        s = checker.compute_triggers({"current_price": 100_000.0}, {})
        assert "correlation_collapse" not in s.trigger_scores
        assert not _fired(s)


# ---------------------------------------------------------------------------
# Wiring pin: the verdict consults the helper and the constant
# ---------------------------------------------------------------------------

class TestWiring:
    def test_verdict_branch_reads_both_conjuncts(self):
        import inspect
        src = inspect.getsource(NoTradeTriggerChecker.compute_triggers)
        assert "self._correlation_conjuncts(" in src
        # [P311/P330] a substring pin of a CONDITION survives
        # `if False and ...`; assert_guard_live requires it to be the WHOLE
        # condition of its `if`.
        from tests._guard_pins import assert_guard_live
        assert_guard_live(
            src, "if _all_same is True and not _has_edge:",
            why=("the CORRELATION_COLLAPSE verdict must require BOTH conjuncts "
                 "(all-three-same-direction AND no validated edge) — P383"),
            near="self._correlation_conjuncts(")
