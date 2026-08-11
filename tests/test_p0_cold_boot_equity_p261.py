"""[P261] The phantom kill switch of 2026-08-11 00:09.

A restart's first tick has no sleeve object yet (run_live builds it AFTER
the first tick), but the SOTA controller's peak and the P253 daily anchor
are PERSISTED at combined (Kraken + sleeve) denomination. The P227 fold-in
only applied its held-combined fallback when the sleeve object existed, and
its closing comment certified the no-object case as safe — true before
P253 made the anchors persistent, false after. Result: Kraken-only
$7,088.69 against the combined anchor $10,865.13 -> "daily loss" of exactly
-$3,776.44 (the sleeve's entire equity) -> sticky kill switch, on a book
that was flat and healthy.

Contracts pinned:
  - the feed decision is pure (combined_p0_equity) and behaviorally tested
    with the incident's exact numbers;
  - a partial book is NEVER fed while a held combined value exists;
  - first-ever boot (no held value anywhere) feeds Kraken-only — the only
    case where no combined-denominated anchor can exist;
  - the held value is persisted AND restored (either half alone re-opens
    the hole across exactly the restart that creates it).
"""

import re
from pathlib import Path

import pytest

from main import combined_p0_equity

REPO = Path(__file__).resolve().parents[1]
MAIN = (REPO / "main.py").read_text(encoding="utf-8-sig", errors="replace")


class TestPureFeedDecision:
    def test_the_incident_exactly(self):
        """2026-08-11 00:09:19 — with the fix, the first tick feeds the held
        combined value and the phantom -$3,776.44 cannot be computed."""
        equity, held, src = combined_p0_equity(
            kraken_equity=7088.69, sleeve_equity=0.0,
            held_combined=10865.13)
        assert equity == pytest.approx(10865.13)
        assert src == "held"
        anchor = 10865.13
        assert equity - anchor == pytest.approx(0.0, abs=0.01), (
            "the phantom daily loss is back"
        )

    def test_known_sleeve_folds_and_updates_held(self):
        equity, held, src = combined_p0_equity(7088.69, 3776.44, 10000.0)
        assert equity == pytest.approx(10865.13)
        assert held == pytest.approx(10865.13)
        assert src == "folded"

    def test_first_ever_boot_is_kraken_only(self):
        """No held value anywhere = genuinely first boot = no persisted
        combined anchor can exist, so Kraken-only is honest and safe."""
        equity, held, src = combined_p0_equity(7088.69, 0.0, None)
        assert equity == pytest.approx(7088.69)
        assert held is None
        assert src == "kraken_only_first_boot"

    def test_unreadable_sleeve_mid_process_still_holds(self):
        """The original P227 case (sleeve exists, API down) is unchanged."""
        equity, _, src = combined_p0_equity(7000.0, 0.0, 10800.0)
        assert equity == pytest.approx(10800.0)
        assert src == "held"


class TestWiringAndPersistence:
    def test_the_feed_goes_through_the_pure_function(self):
        """The P251 lesson: the load-bearing path must BE the pure call."""
        i = MAIN.find("equity, _p0_new_held, _p0_src = combined_p0_equity(")
        assert i > 0, "the P0 feed bypasses combined_p0_equity"
        j = MAIN.find("self.p0_integrator.pre_tick_update(", i)
        assert 0 < i < j, "the pure decision must precede pre_tick_update"

    def test_daily_anchor_is_set_after_the_combined_resolution(self):
        """The P253 anchor must anchor on the RESOLVED equity — anchoring on
        the partial value would flip the bug's sign (phantom daily GAIN
        masking real losses)."""
        i = MAIN.find("equity, _p0_new_held, _p0_src = combined_p0_equity(")
        j = MAIN.find("self._daily_pnl_anchor = float(equity)", i)
        assert 0 < i < j

    def test_restore_function_behavioral(self):
        """The restore decision is pure (the P251 rule — a textual pin on
        the inline version was defeated by a `False and` probe, third catch
        of that shape in two days)."""
        from main import restore_p0_combined_equity as rest
        assert rest({"p0_last_combined_equity": 10865.13}, None) == \
            pytest.approx(10865.13)
        assert rest({}, None) is None                       # first boot
        assert rest({}, 9000.0) == 9000.0                   # keeps current
        assert rest({"p0_last_combined_equity": "bad"}, 9000.0) == 9000.0
        assert rest({"p0_last_combined_equity": -5}, 9000.0) == 9000.0
        assert rest(None, 9000.0) == 9000.0

    def test_held_value_is_persisted_and_restore_is_load_bearing(self):
        """Both halves, or the hole re-opens across exactly the restart that
        creates it. The restore ASSIGNMENT flows through the pure function —
        bypassing it means changing this line, which fails here."""
        assert '"p0_last_combined_equity": getattr(' in MAIN, "save half gone"
        assert ("self._p0_last_combined_equity = restore_p0_combined_equity("
                in MAIN), "restore no longer flows through the pure function"

    def test_restore_runs_in_the_governor_section(self):
        """run_live restores governors only (P211, restore_positions=False) —
        the restore must sit in that section or LIVE never sees it."""
        i = MAIN.find(
            "self._p0_last_combined_equity = restore_p0_combined_equity(")
        sota = MAIN.find('data.get("sota_risk_state"', i)
        assert 0 < i < sota, (
            "the P261 restore must precede the SOTA restore in the same "
            "governor block"
        )
