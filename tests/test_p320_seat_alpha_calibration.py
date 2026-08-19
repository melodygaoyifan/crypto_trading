"""
[P320] The asserted alpha, calibrated to the seat's own holding horizon.

P318 established the gate compares a per-TICK alpha constant against a
per-ROUND-TRIP friction. This replaces the regimebook seat's flat
`30.0 * |dir|` with its MEASURED gross bps per round trip in its worst era.

The measurement is a verdict, not a knob:

    asset   pre_design   design   validation   MIN (asserted)
    BTC            2.3     68.5         24.1              2.3
    ETH          251.7     88.1         52.1             52.1
    SOL          427.6    221.7        -20.8            -20.8

SOL asserts a NEGATIVE edge and so can never clear friction; BTC's worst era
is far below cost (profitable in one era of three); ETH clears in every era
and is the certified, un-fitted P247 config.

THE INTERLOCK IS THE POINT. Calibrated alpha alone raises ETH 22.5 -> 52.1
while fees stay ~3x understated — a pure loosening. Honest fees alone reject
trades earning 2.5x-27x cost (P318). They are halves of one correction.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from core.seat_alpha import (  # noqa: E402
    REGIMEBOOK_ALPHA_BY_ERA,
    REGIMEBOOK_ALPHA_BPS_PER_ROUND_TRIP,
    calibrated_seat_alpha,
    regimebook_alpha_bps,
)


class TestTheCalibrationIsTheMinimum:

    @pytest.mark.parametrize("asset", ["BTC", "ETH", "SOL"])
    def test_asserted_value_is_the_worst_era(self, asset):
        """P167: a safety control must assume the worst era repeats. Using the
        mean instead would be a deliberate loosening (BTC 2.3 -> 31.6,
        SOL -20.8 -> +209.5) and needs its own P-entry."""
        eras = REGIMEBOOK_ALPHA_BY_ERA[asset]
        assert REGIMEBOOK_ALPHA_BPS_PER_ROUND_TRIP[asset] == \
            pytest.approx(min(eras.values()))

    def test_sol_asserts_a_negative_edge_and_is_not_clamped(self):
        """'This seat loses money per round trip' is information the gate
        should act on. Clamping to 0 would silently upgrade it to merely
        unprofitable, and SOL would then look like every other flat seat."""
        v, prov = regimebook_alpha_bps("SOL")
        assert v < 0
        assert "validation" in prov

    def test_btc_cannot_clear_its_own_friction(self):
        """The finding, pinned: BTC's worst era (2.3bps) is an order of
        magnitude under ~28bps of honest round-trip friction. If this ever
        passes, BTC's calibration was raised — check the window, not the test."""
        from core.cde_fees import cde_fee_bps
        friction = 2 * (cde_fee_bps("BTC", 64435.0, is_maker=False)[0] + 2.0 + 2.0)
        assert regimebook_alpha_bps("BTC")[0] < friction

    def test_eth_is_the_only_seat_that_clears_in_every_era(self):
        for era, v in REGIMEBOOK_ALPHA_BY_ERA["ETH"].items():
            assert v > 0, era
        assert min(REGIMEBOOK_ALPHA_BY_ERA["BTC"].values()) < 10
        assert min(REGIMEBOOK_ALPHA_BY_ERA["SOL"].values()) < 0


class TestFailDirections:

    def test_unknown_asset_asserts_zero(self):
        """A seat with no measurement must not be able to trade on one."""
        v, prov = regimebook_alpha_bps("DOGE")
        assert v == 0.0 and "no_calibration" in prov

    def test_an_uncalibrated_seat_keeps_its_own_constant(self):
        """Calibrating one seat must not silently re-price another whose
        horizon was never measured (the P315 lesson inverted)."""
        v, prov = calibrated_seat_alpha("BTC", "trend", fallback_bps=30.0)
        assert v == 30.0 and "uncalibrated_seat" in prov

    def test_regimebook_dispatches_to_the_calibration(self):
        assert calibrated_seat_alpha("ETH", "regimebook", 30.0)[0] == \
            pytest.approx(52.1)


class TestInterlock:
    """Either flag alone moves the gate the WRONG way."""

    def _src(self):
        return io.open(REPO / "main.py", encoding="utf-8").read()

    def test_the_interlock_is_enforced_behaviourally(self):
        """Pinned by CALLING the resolver, not by reading source: a source pin
        proves the code was written, not that it runs (P234/P307b — and a
        substring pin of an earlier draft of this very test stayed GREEN
        against the defect it guarded)."""
        from core.seat_alpha import resolve_seat_edge
        assert resolve_seat_edge("ETH", "regimebook", 1.0, 30.0,
                                 True, True) == pytest.approx(52.1)
        for cal, fee in ((True, False), (False, True), (False, False)):
            assert resolve_seat_edge("ETH", "regimebook", 1.0, 30.0,
                                     cal, fee) == pytest.approx(30.0), (
                f"calibrated={cal} honest_fees={fee} changed the edge; either "
                f"half alone moves the gate the wrong way (P318)")

    def test_a_flat_book_asserts_nothing(self):
        """A calibrated ROUND-TRIP value must never be asserted for a position
        the seat is not taking."""
        from core.seat_alpha import resolve_seat_edge
        assert resolve_seat_edge("ETH", "regimebook", 0.0, 30.0,
                                 True, True) == 0.0

    def test_the_calibrated_value_is_not_scaled_by_direction(self):
        """It is already a whole-round-trip expectation; scaling by |dir|
        would restore the per-tick shape this replaces."""
        from core.seat_alpha import resolve_seat_edge
        a = resolve_seat_edge("ETH", "regimebook", 1.0, 30.0, True, True)
        b = resolve_seat_edge("ETH", "regimebook", 0.5, 30.0, True, True)
        assert a == pytest.approx(b) == pytest.approx(52.1), (
            "the calibrated round-trip edge is being rescaled by |direction|")

    def test_the_seat_passes_both_flags_to_the_resolver(self):
        src = self._src()
        i = src.index("resolve_seat_edge")
        blk = src[max(0, i - 600):i + 600]
        assert '"seat_alpha_calibrated"' in blk
        assert '"coinbase_per_contract_fees"' in blk

    def test_config_trio_and_both_default_off(self):
        import dataclasses
        import main
        names = {f.name for f in dataclasses.fields(main.ProductionConfig)}
        assert {"seat_alpha_calibrated", "coinbase_per_contract_fees"} <= names
        cfg = main.ProductionConfig()
        assert cfg.seat_alpha_calibrated is False
        assert cfg.coinbase_per_contract_fees is False
        assert 'data.get("seat_alpha_calibrated", False)' in self._src()

    def test_neither_is_armed_in_the_live_profile(self):
        """Arming BOTH takes the book to ~flat: BTC -32.9, ETH -3.9, SOL -80.3
        against their thresholds. That is the correct answer on this evidence
        and it is an operator decision (P141) — it stops trading."""
        live = json.loads(
            (REPO / "configs" / "live_high_risk.json").read_text(encoding="utf-8"))
        assert "seat_alpha_calibrated" not in live
        assert "coinbase_per_contract_fees" not in live


class TestTheUnitsFixIsReal:

    def test_the_default_path_still_asserts_thirty_bps(self):
        """ANTI-ROT: if the seat stops asserting 30.0 by default, this whole
        module's premise changed and the interlock must be re-derived."""
        from core.seat_alpha import resolve_seat_edge
        assert resolve_seat_edge("BTC", "regimebook", 1.0, 30.0,
                                 False, False) == pytest.approx(30.0)

    def test_calibrated_alpha_is_a_round_trip_quantity(self):
        """Sanity: the asserted numbers are round-trip sized, i.e. materially
        larger than a per-tick constant for the seat that actually earns."""
        assert regimebook_alpha_bps("ETH")[0] > 30.0
