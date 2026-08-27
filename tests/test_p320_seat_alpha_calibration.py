"""
[P320] The asserted alpha, calibrated to the seat's own holding horizon.

P318 established the gate compares a per-TICK alpha constant against a
per-ROUND-TRIP friction. This replaces the regimebook seat's flat
`30.0 * |dir|` with its MEASURED gross bps per round trip.

    asset   pre_design   design   validation    MIN   MEDIAN (asserted)
    BTC            2.3     68.5         24.1    2.3               24.1
    ETH  [P419] donchian book  674.4  291.6  375.5     median  375.5
    SOL          427.6    221.7        -20.8  -20.8              221.7

[P321] The asserted statistic is the era-MEDIAN, changed from the MIN by
explicit operator decision once the goal was stated as profit rather than
pure capital preservation. The MIN stops every asset; the MEAN can be carried
by one dominant era (the P243/P244 era-fragility). The median is the robust
middle, and the STATISTIC is pinned so a silent switch to either extreme
fails. On it, BTC stops trading (24.1 vs a ~35bps threshold) while ETH (375.5, donchian book [P419])
and SOL (221.7) clear.

THE INTERLOCK IS THE POINT. Calibrated alpha alone raises ETH's assertion
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


class TestTheCalibrationIsTheEraMedian:

    # [P420] parametrized over the SHIPPED table, not a hand list: XRP/BNB
    # joined the table in P412 and this pin silently did not cover them (an
    # asset can enter the live gate without ever meeting the statistic pin).
    @pytest.mark.parametrize("asset", sorted(REGIMEBOOK_ALPHA_BPS_PER_ROUND_TRIP))
    def test_asserted_value_is_the_era_median(self, asset):
        """[P321] The STATISTIC is part of the decision. The MIN stops every
        asset (BTC 2.3, SOL -20.8); the MEAN can be carried by one dominant
        era (BTC 80.3, SOL 354.7) — the era-fragility P243/P244 disqualify on.
        The median is the robust middle, and pinning it means a silent switch
        to either extreme fails here."""
        import statistics
        eras = REGIMEBOOK_ALPHA_BY_ERA[asset]
        assert REGIMEBOOK_ALPHA_BPS_PER_ROUND_TRIP[asset] == \
            pytest.approx(statistics.median(eras.values()))

    def test_every_gate_asset_has_an_era_table(self):
        """[P420] the per-RT table and the per-era table must carry the same
        assets — a per-RT entry with no eras cannot be verified by the
        producer at all."""
        assert set(REGIMEBOOK_ALPHA_BPS_PER_ROUND_TRIP) == set(REGIMEBOOK_ALPHA_BY_ERA)

    def test_a_negative_calibration_would_pass_through_unclamped(self):
        """[P321] No asset's MEDIAN is negative now (SOL's validation era is,
        its median is not), so this is a property of the FUNCTION rather than
        of the current table — and it must stay, because a re-derivation can
        produce one. Clamping to 0 would silently upgrade "this seat loses
        money per round trip" to merely-unprofitable."""
        import core.seat_alpha as sa
        saved = dict(sa.REGIMEBOOK_ALPHA_BPS_PER_ROUND_TRIP)
        try:
            sa.REGIMEBOOK_ALPHA_BPS_PER_ROUND_TRIP["SOL"] = -20.8
            assert sa.regimebook_alpha_bps("SOL")[0] == pytest.approx(-20.8)
        finally:
            sa.REGIMEBOOK_ALPHA_BPS_PER_ROUND_TRIP.clear()
            sa.REGIMEBOOK_ALPHA_BPS_PER_ROUND_TRIP.update(saved)
        # SOL's own worst era is still negative and still visible in the table
        assert REGIMEBOOK_ALPHA_BY_ERA["SOL"]["validation"] < 0

    def test_btc_cannot_clear_its_own_friction(self):
        """The finding, pinned: BTC's worst era (2.3bps) is an order of
        magnitude under ~28bps of honest round-trip friction. If this ever
        passes, BTC's calibration was raised — check the window, not the test."""
        from core.cde_fees import cde_fee_bps
        friction = 2 * (cde_fee_bps("BTC", 64435.0, is_maker=False)[0] + 2.0 + 2.0)
        assert regimebook_alpha_bps("BTC")[0] < friction

    def test_eth_is_the_only_seat_positive_in_every_era(self):
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
            pytest.approx(375.5)  # [P419]


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
                                 True, True, 2252.0) == pytest.approx(375.5)  # [P419]
        for cal, fee in ((True, False), (False, True), (False, False)):
            assert resolve_seat_edge("ETH", "regimebook", 1.0, 30.0,
                                     cal, fee, 2252.0) == pytest.approx(30.0), (
                f"calibrated={cal} honest_fees={fee} changed the edge; either "
                f"half alone moves the gate the wrong way (P318)")

    def test_interlock_gates_on_EFFECT_not_on_the_flag(self):
        """[P321b] THE LIVE INCIDENT. Both flags were on and the honest fee
        still did not apply — the gate block read market_data["price"], a key
        no producer writes (it is "current_price"), so it fell back to the
        modelled 3bps while the calibrated alpha DID apply. That is exactly
        the half P318 warned about (calibrated alpha + cheap fees = pure
        loosening) and it reached production for one tick.

        A flag being true is not evidence the correction took effect, so the
        alpha half must refuse whenever the fee half cannot price THIS asset.
        """
        from core.seat_alpha import resolve_seat_edge
        assert resolve_seat_edge("ETH", "regimebook", 1.0, 30.0, True, True,
                                 price=1916.5) == pytest.approx(375.5)  # [P419]
        for bad in (0.0, -1.0, float("nan")):
            assert resolve_seat_edge("ETH", "regimebook", 1.0, 30.0, True,
                                     True, price=bad) == pytest.approx(30.0), (
                f"px={bad!r}: the calibrated alpha applied while the honest "
                f"fee could not — the two must move together")

    def test_the_seat_passes_the_price_so_the_effect_check_can_run(self):
        """Anti-vacuity: the effect-interlock is inert unless the call site
        actually supplies a price."""
        src = self._src()
        i = src.index("resolve_seat_edge")
        blk = src[max(0, i - 600):i + 800]
        assert 'price=market_data.get("current_price")' in blk

    def test_a_flat_book_asserts_nothing(self):
        """A calibrated ROUND-TRIP value must never be asserted for a position
        the seat is not taking."""
        from core.seat_alpha import resolve_seat_edge
        assert resolve_seat_edge("ETH", "regimebook", 0.0, 30.0,
                                 True, True, 2252.0) == 0.0

    def test_the_calibrated_value_is_not_scaled_by_direction(self):
        """It is already a whole-round-trip expectation; scaling by |dir|
        would restore the per-tick shape this replaces."""
        from core.seat_alpha import resolve_seat_edge
        a = resolve_seat_edge("ETH", "regimebook", 1.0, 30.0, True, True, 2252.0)
        b = resolve_seat_edge("ETH", "regimebook", 0.5, 30.0, True, True, 2252.0)
        assert a == pytest.approx(b) == pytest.approx(375.5), (  # [P419]
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

    def test_both_are_armed_at_their_decided_values(self):
        """[P321] ARMED by explicit operator instruction. Pinned at the DECIDED
        value rather than at OFF, so a silent revert fails too — either
        direction is a live-money change (the P237/P270 pattern).

        Live effect of the pair: BTC stops entering (24.1 vs a ~35bps
        threshold); ETH (375.5, donchian [P419]) and SOL (221.7) continue on honest economics."""
        live = json.loads(
            (REPO / "configs" / "live_high_risk.json").read_text(encoding="utf-8"))
        assert live.get("coinbase_per_contract_fees") is True
        assert live.get("seat_alpha_calibrated") is True
        assert "_p321_honest_economics_note" in live, (
            "the note carrying the decision, the expected effect and the "
            "revert must travel with the flags — a bare `true` loses why it "
            "was armed")


class TestTheUnitsFixIsReal:

    def test_the_default_path_still_asserts_thirty_bps(self):
        """ANTI-ROT: if the seat stops asserting 30.0 by default, this whole
        module's premise changed and the interlock must be re-derived."""
        from core.seat_alpha import resolve_seat_edge
        assert resolve_seat_edge("BTC", "regimebook", 1.0, 30.0,
                                 False, False, 69280.0) == pytest.approx(30.0)

    def test_calibrated_alpha_is_a_round_trip_quantity(self):
        """Sanity: the asserted numbers are round-trip sized, i.e. materially
        larger than a per-tick constant for the seat that actually earns."""
        assert regimebook_alpha_bps("ETH")[0] > 30.0
