"""[P326] Guards for the producer of core.seat_alpha.REGIMEBOOK_ALPHA_BY_ERA.

The constants gate live trading, and until now their derivation existed only as
one docstring sentence. Re-deriving from that sentence gave BTC
240.3 / 58.4 / 44.0 against the shipped 2.3 / 68.5 / 24.1 — the convention has
three clauses and the sentence stated one.

These tests pin the ARITHMETIC, and run without the parquets so they are not
silently skipped in CI (P194). The end-to-end reproduction is the tool's own
`--verify` mode, which is operator-local by necessity (P213).
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

pd = pytest.importorskip("pandas")

from training.seat_alpha_calibration import (  # noqa: E402
    TOLERANCE_BPS,
    round_trip_edge_bps,
)

MOD = REPO / "training" / "seat_alpha_calibration.py"


def _src() -> str:
    return io.open(MOD, encoding="utf-8").read()


def S(vals):
    return pd.Series([float(v) for v in vals])


class TestClauseOne_TwoTimesGrossOverTurnover:

    def test_one_full_round_trip_reports_its_whole_gross(self):
        """flat -> long -> flat is turnover 2, so 2*g/2 = g: a round trip's
        edge IS its total gross, in bps. Any other factor silently re-scales
        every number the gate compares against friction."""
        pos = S([0, 1, 1, 0])
        gross = S([0, 0.001, 0.002, 0])       # 30bps total
        assert round_trip_edge_bps(gross, pos) == pytest.approx(30.0)

    def test_two_round_trips_report_the_AVERAGE_not_the_sum(self):
        pos = S([0, 1, 0, 1, 0])
        gross = S([0, 0.003, 0, 0.003, 0])    # 30bps each
        assert round_trip_edge_bps(gross, pos) == pytest.approx(30.0)

    def test_a_flip_counts_as_two_units_of_turnover(self):
        """long -> short is |dpos| = 2, i.e. one round trip's worth."""
        pos = S([1, -1])
        gross = S([0.001, 0.001])
        assert round_trip_edge_bps(gross, pos) == pytest.approx(20.0)

    def test_it_is_gross_only(self):
        """The caller must pass the GROSS column. Charging cost here would
        double-count: the gate computes its own friction and compares."""
        src = _src()
        assert 'df["gross"]' in src
        assert 'df["net"]' not in src


class TestClauseThree_AnOpeningPositionIsNotTurnover:
    """A position already standing when a window opens was not entered inside
    it. This is the clause that made ETH's pre_design read 245.9 instead of
    251.7 — ETH's book is +1 at that boundary and flat at the other two, which
    is exactly why only that cell disagreed."""

    def test_a_window_opening_flat_is_unaffected_by_the_clause(self):
        pos = S([0, 1, 0])
        gross = S([0, 0.002, 0])
        assert round_trip_edge_bps(gross, pos) == pytest.approx(20.0)

    def test_a_window_opening_mid_position_excludes_that_entry(self):
        """Only the exit is turnover here, so the same gross divides by 1 unit
        rather than 2 — a factor of two, not a rounding difference."""
        pos = S([1, 1, 0])
        gross = S([0.001, 0.001, 0])
        assert round_trip_edge_bps(gross, pos) == pytest.approx(40.0)

    def test_the_clause_is_a_diff_with_no_backfill_of_the_first_bar(self):
        src = _src()
        assert "pos.diff().abs().fillna(0.0)" in src
        assert "fillna(pos.abs())" not in src, (
            "back-filling the first bar with |pos| re-counts an inherited "
            "position as turnover — the ETH pre_design discrepancy")


class TestClauseTwo_ErasIndexThePositionsFrame:
    """Correcting for the MIN_BARS warmup that build_positions drops moves BTC
    pre_design from 2.3 to 243.2 — a hundredfold, on the number that decides
    whether BTC trades."""

    def test_eras_are_applied_to_the_positions_frame_directly(self):
        src = _src()
        assert "pos_df[series].iloc[lo:hi]" in src

    def test_no_warmup_offset_is_subtracted(self):
        src = _src()
        for bad in ("lo - off", "hi - off", "get_loc(pos"):
            assert bad not in src, bad


class TestFailDirections:

    def test_no_turnover_reports_None_never_zero(self):
        """"The position never moved" is not an edge of zero — dividing by it
        would fabricate one (P2), and a fabricated 0 would drag any median
        toward not trading for a reason nobody measured."""
        assert round_trip_edge_bps(S([0.001, 0.001]), S([1, 1])) is None
        assert round_trip_edge_bps(S([0.0, 0.0]), S([0, 0])) is None

    def test_a_negative_edge_passes_through_unclamped(self):
        """BTC trend-only measures -3.6 as its era median. Clamping would
        silently upgrade "this series loses money per round trip" to
        merely-unprofitable."""
        pos = S([0, 1, 0])
        gross = S([0, -0.002, 0])
        assert round_trip_edge_bps(gross, pos) == pytest.approx(-20.0)

    def test_verify_refuses_to_compare_a_non_book_series(self):
        """Comparing the trend series against the BOOK's table would report
        drift that is really a different question."""
        from training.seat_alpha_calibration import main
        rc = main(["--series", "trend", "--verify"])
        assert rc == 2

    def test_drift_is_reported_not_silently_absorbed(self):
        src = _src()
        assert "DRIFT" in src
        assert "return 3" in src
        assert TOLERANCE_BPS <= 0.5, (
            "a loose tolerance turns the control into a rubber stamp")


class TestItIsAProducerNotADuplicate:
    """[P172/P310] The point is that the shipped constants become reproducible,
    not that a second copy of them appears."""

    def test_the_shipped_table_is_imported_never_restated(self):
        src = _src()
        assert "from core.seat_alpha import REGIMEBOOK_ALPHA_BY_ERA" in src
        for literal in ("68.5", "251.7", "221.7", "427.6"):
            code = src.split('"""', 2)[-1]      # past the module docstring
            assert literal not in code, (
                f"{literal} is restated in code; import the table instead")

    def test_the_lab_machinery_is_imported_never_reimplemented(self):
        src = _src()
        assert "import training.funding_legs_lab as lab" in src
        for own in ("def load_closes", "def build_positions", "def pnl"):
            assert own not in src, own
