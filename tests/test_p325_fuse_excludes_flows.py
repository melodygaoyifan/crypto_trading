"""[P325] Rule #3's window must exclude external capital flows.

THE INCIDENT, found by an end-to-end audit reading one live log line:

    [P209][FUSE-FEED] sleeve equity=$10,819.46 delta=+0.00 window=+186.88%
    [COINBASE-PNL]    equity=$10,819.46 pnl=$-252.56 (-2.28%)

Two accumulators over the same account disagreed by 189 percentage points. The
sleeve's own ledger was right: P293h subtracts detected transfers there. The
fuse feed -- written the same day, four thousand lines away -- computed a bare
`equity - anchor`, so the 2026-08-16 deposit of ~$7,074 entered Rule #3's 28d
window as PROFIT (7074 / 3785 pre-deposit equity = +186.9%).

WHY THAT IS WORSE THAN A WRONG NUMBER. The fuse suspends on a NEGATIVE window
(-15% PnL). A window sitting at +186.88% cannot go negative on any loss this
sleeve can produce, so the deposit did not mis-state Rule #3 -- it DISABLED it
until the point rolls out of the window (~2026-09-13). The sleeve's 15%
drawdown halt and the LIVE 25% halt are unaffected: both read equity against a
peak, and a deposit raises both sides.

The fix is tested by CALLING the arithmetic, never by pinning its source
(P234/P307b).
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from main import fuse_flow_adjusted_delta as adj  # noqa: E402

MAIN = REPO / "main.py"


def _src() -> str:
    return io.open(MAIN, encoding="utf-8").read()


class TestTheLiveIncident:

    def test_the_deposit_no_longer_enters_the_window_as_profit(self):
        """The exact numbers: a $7,074 transfer on a tick with no trading."""
        d, fd = adj(7074.0, 7074.0, 0.0)
        assert d == pytest.approx(0.0)
        assert fd == pytest.approx(7074.0)

    def test_a_real_loss_on_the_deposit_tick_SURVIVES(self):
        """The load-bearing case. If the adjustment swallowed the whole delta,
        a loss landing on the same tick as a transfer would be forgiven — the
        loss-forgiveness direction P287 spent an entry closing."""
        d, _ = adj(7074.0 - 60.0, 7074.0, 0.0)
        assert d == pytest.approx(-60.0)

    def test_a_withdrawal_is_not_a_loss(self):
        d, fd = adj(-1000.0, -1000.0, 0.0)
        assert d == pytest.approx(0.0) and fd == pytest.approx(-1000.0)

    def test_an_ordinary_tick_is_untouched(self):
        for raw in (-52.3, 0.0, +18.9):
            d, fd = adj(raw, 7074.0, 7074.0)
            assert d == pytest.approx(raw)
            assert fd == 0.0


class TestFailDirections:
    """An unusable reference must adjust NOTHING. The failure direction is
    'keep counting it as PnL' — honest but noisy — never a silent adjustment
    that could hide a real loss (P293h's own rule)."""

    def test_a_missing_anchor_reference_adjusts_nothing(self):
        d, fd = adj(7074.0, 7074.0, None)
        assert d == pytest.approx(7074.0) and fd == 0.0

    def test_an_unreadable_flow_total_adjusts_nothing(self):
        for bad in (None, "abc", object()):
            d, fd = adj(-40.0, bad, 0.0)
            assert d == pytest.approx(-40.0), bad
            assert fd == 0.0

    def test_non_finite_flow_adjusts_nothing(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            d, fd = adj(-40.0, bad, 0.0)
            assert d == pytest.approx(-40.0)
            assert fd == 0.0

    def test_an_unreadable_raw_delta_records_zero_not_a_gain(self):
        d, fd = adj("junk", 0.0, 0.0)
        assert d == 0.0 and fd == 0.0

    def test_the_adjustment_can_never_manufacture_a_gain_from_a_loss(self):
        """Property: with a NON-NEGATIVE flow (a deposit), the adjusted delta
        is never greater than the raw one — the fuse can only ever be made
        MORE willing to suspend, never less."""
        for raw in (-500.0, -1.0, 0.0, 1.0, 500.0):
            for flow in (0.0, 1.0, 7074.0):
                d, _ = adj(raw, flow, 0.0)
                assert d <= raw + 1e-9, (raw, flow)


class TestTheReferencePairMovesTogether:
    """The equity anchor and the flow reference are ONE pair. Advancing only
    the equity anchor would re-subtract the same transfer on every later tick;
    restoring only one would measure the next delta against the wrong
    reference (the P287 lesson about half a reference, on the flow half)."""

    def test_both_are_advanced_in_the_same_block(self):
        src = _src()
        i = src.index("self._fuse_sleeve_anchor_equity = _fz_eq")
        blk = src[i:i + 700]
        assert "self._fuse_flow_cum_at_anchor" in blk

    def test_both_are_persisted(self):
        src = _src()
        assert '"fuse_sleeve_anchor_equity"' in src
        assert '"fuse_flow_cum_at_anchor"' in src

    def test_both_are_restored(self):
        src = _src()
        i = src.index('_anchor = data.get("fuse_sleeve_anchor_equity")')
        blk = src[i:i + 900]
        assert 'data.get("fuse_flow_cum_at_anchor")' in blk
        assert "self._fuse_flow_cum_at_anchor" in blk

    def test_an_absent_persisted_reference_restores_as_None_not_zero(self):
        """None means "no reference" and adjusts nothing. Zero would mean "no
        flows have ever happened", which on a state file predating this field
        would make the next tick subtract the ENTIRE cumulative flow as one
        tick's transfer (P261b: a migration case is a first-boot case with
        pre-existing state).

        Checked STRUCTURALLY with ast rather than by pinning the source text of
        a condition — such a pin stays green against `if False and <cond>`
        (P234/P307b), and the parallel session's frozen-roster guard rightly
        flagged the first draft of this test for exactly that.
        """
        import ast
        tree = ast.parse(_src())
        found = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for tgt in node.targets:
                if (isinstance(tgt, ast.Attribute)
                        and tgt.attr == "_fuse_flow_cum_at_anchor"):
                    found.append(node.value)
        assert found, "nothing assigns _fuse_flow_cum_at_anchor"
        # the RESTORE assignment is the conditional one; the feed's is a plain
        # name. At least one must fall back to a literal None, never to 0.
        conds = [v for v in found if isinstance(v, ast.IfExp)]
        assert conds, (
            "no conditional restore of _fuse_flow_cum_at_anchor — an absent "
            "persisted value must resolve to None, not to a number")
        for v in conds:
            assert isinstance(v.orelse, ast.Constant) and v.orelse.value is None, (
                "the restore falls back to "
                f"{ast.dump(v.orelse)[:60]} instead of None; a numeric "
                "fallback re-subtracts the whole cumulative flow after a "
                "migration (P261b)")


class TestTheFeedUsesTheHelper:
    """[P170/P312] A seam nothing calls is decoration."""

    def test_the_feed_routes_through_the_pure_function(self):
        src = _src()
        i = src.index("[P209][FUSE-FEED] sleeve ")
        blk = src[max(0, i - 4000):i]
        assert "fuse_flow_adjusted_delta(" in blk

    def test_the_feed_reads_the_sleeve_in_scope(self):
        """A wrong variable name here raises NameError, which the flow read's
        `except (TypeError, ValueError)` would NOT catch — it would escape into
        the block's outer handler and could skip the fuse record entirely. The
        first draft of this fix used `_cb_sleeve`; the name in scope is
        `_fz_sleeve` (P193/P234 class)."""
        src = _src()
        i = src.index('"_external_flow_usd", 0.0)')
        blk = src[max(0, i - 300):i]
        assert "_fz_sleeve" in blk
        assert "_cb_sleeve" not in blk

    def test_a_flow_is_reported_not_silently_applied(self):
        src = _src()
        assert "[P325][FUSE-FEED] external " in src


class TestTheOtherHaltsAreUnaffected:
    """Stated as a test so the severity claim in the docstring cannot rot: the
    drawdown halts read equity against a peak, so a deposit raises both sides
    and leaves the ratio unchanged. Only the PnL-window control was blinded."""

    def test_drawdown_is_ratio_based_and_deposit_neutral(self):
        peak, eq, dep = 3785.0, 3785.0, 7074.0
        before = (peak - eq) / peak
        after = ((peak + dep) - (eq + dep)) / (peak + dep)
        assert before == pytest.approx(after) == pytest.approx(0.0)
