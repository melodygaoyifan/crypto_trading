"""[P167] The alpha gate charges a ROUND TRIP, not one leg.

The defect: `check_alpha_gate` computed

    friction = fee + slippage + latency + margin

and compared a ROUND-TRIP alpha estimate against `friction * multiplier`.
With NORMAL_MULTIPLIER = 1.10 that is a demand for 1.10 legs of friction from
a position that pays 2 — entry and exit. The gate was structurally incapable
of rejecting a trade whose only problem was that it had to be closed.

Measured at Coinbase taker (3bps) with the live pipeline's own calibration
(`market_data_pipeline.py`: avg |quant_dir| ~= 0.3 -> ~14.6bps of alpha after
the ALPHA-FEEDBACK haircut), the old gate passed BTC and ETH and the new one
does not:

    asset   old friction / thr / pass      new friction / thr / pass
    BTC      8.0 /  8.8 / True             16.0 / 17.6 / False
    ETH     10.0 / 11.0 / True             20.0 / 22.0 / False
    SOL     15.0 / 16.5 / False            30.0 / 33.0 / False

This module pins the contract so it cannot silently revert:

  1. per-ORDER costs (fee, spread, latency) are charged twice
  2. per-HOLD costs (margin opening + rollover) are charged ONCE — they are
     already integrated over `expected_hold_periods_4h`; doubling them would
     be a second, unrelated bug wearing this fix's clothes
  3. `friction_legs` is reported on the result, so a reader can tell which
     arithmetic produced a decision instead of inferring it
  4. the change only ever TIGHTENS — no input that the old gate rejected is
     accepted by the new one
"""

import os

import pytest

from defense.constitution import (
    AlphaThresholdCalculator,
    FrictionComponents,
)


# --------------------------------------------------------------------------
# 1. FrictionComponents: per-leg vs round-trip
# --------------------------------------------------------------------------

def _fc(**kw) -> FrictionComponents:
    fc = FrictionComponents(
        taker_fee_bps=kw.pop("taker_fee_bps", 3.0),
        maker_fee_bps=kw.pop("maker_fee_bps", 1.0),
        slippage_bps=kw.pop("slippage_bps", 3.0),
        latency_cost_bps=kw.pop("latency_cost_bps", 2.0),
        **kw,
    )
    return fc


class TestPerLegVsRoundTrip:
    def test_per_leg_is_fee_plus_spread_plus_latency(self):
        fc = _fc()
        assert fc.per_leg_bps() == pytest.approx(3.0 + 3.0 + 2.0)
        assert fc.per_leg_bps(is_maker=True) == pytest.approx(1.0 + 3.0 + 2.0)

    def test_round_trip_is_exactly_two_legs(self):
        fc = _fc()
        assert fc.round_trip_bps() == pytest.approx(2 * fc.per_leg_bps())
        assert fc.round_trip_bps(is_maker=True) == pytest.approx(
            2 * fc.per_leg_bps(is_maker=True))

    def test_margin_is_charged_once_not_twice(self):
        """Per-HOLD cost, not per-ORDER.

        `_margin_cost_bps` is opening_fee + rollover * expected_hold_periods_4h
        — it already spans the whole hold. Doubling it would charge the funding
        for a 24h position as if it were held 48h.
        """
        flat = _fc()
        lev = _fc(margin_opening_fee_bps=1.0,
                  margin_rollover_bps_per_4h=0.5,
                  expected_hold_periods_4h=6.0)
        margin = lev._margin_cost_bps
        assert margin == pytest.approx(1.0 + 0.5 * 6.0)
        # Legs doubled, margin did not.
        assert lev.round_trip_bps() == pytest.approx(flat.round_trip_bps() + margin)
        assert lev.per_leg_bps() == pytest.approx(flat.per_leg_bps())

    def test_legs_parameter_scales_only_the_per_order_part(self):
        fc = _fc(margin_opening_fee_bps=4.0)
        one = fc.round_trip_bps(legs=1.0)
        two = fc.round_trip_bps(legs=2.0)
        assert two - one == pytest.approx(fc.per_leg_bps())

    def test_totals_are_still_one_leg(self):
        """`total_taker`/`total_maker` keep their old one-leg meaning.

        They are read elsewhere; this fix renames nothing. Note they fold the
        per-HOLD margin term into a per-ORDER number, which is why they are not
        the right thing to gate on — `round_trip_bps` separates the two.
        """
        fc = _fc()
        assert fc._margin_cost_bps == 0.0
        assert fc.total_taker == pytest.approx(fc.per_leg_bps())
        assert fc.total_maker == pytest.approx(fc.per_leg_bps(is_maker=True))

        lev = _fc(margin_opening_fee_bps=1.0, margin_rollover_bps_per_4h=0.5)
        assert lev.total_taker == pytest.approx(
            lev.per_leg_bps() + lev._margin_cost_bps)


# --------------------------------------------------------------------------
# 2. The gate itself
# --------------------------------------------------------------------------

def _calc(taker_fee_bps: float = 3.0) -> AlphaThresholdCalculator:
    """A calculator on the Coinbase fee schedule (3bps taker, 0 maker).

    Fees are set last: `update_for_volume` writes the Kraken schedule and
    `update_fee_bps` is the override the live path uses (main.py:8553).
    """
    c = AlphaThresholdCalculator()
    c.FRICTION.update_for_asset("")
    c.FRICTION.update_for_volume(0.0)
    c.FRICTION.update_fee_bps(taker_fee_bps=taker_fee_bps, maker_fee_bps=0.0)
    return c


def _gate(calc, alpha_bps, asset="BTC", mode="NORMAL", min_alpha_bps=0.0):
    return calc.check_alpha_gate(
        signal_strength=0.5,
        regime_confidence=0.8,
        mode=mode,
        min_alpha_bps=min_alpha_bps,
        asset=asset,
        estimated_alpha_override=alpha_bps,
    )


class TestGateChargesTwoLegs:
    def test_result_reports_two_legs(self):
        r = _gate(_calc(), 100.0)
        assert r.friction_legs == 2.0

    @pytest.mark.parametrize("asset,per_leg", [
        ("BTC", 3.0 + 3.0 + 2.0),    # 3bps taker + 3bps spread + 2bps latency
        ("ETH", 3.0 + 5.0 + 2.0),
        ("SOL", 3.0 + 10.0 + 2.0),
    ])
    def test_friction_is_twice_the_per_leg_cost(self, asset, per_leg):
        r = _gate(_calc(), 100.0, asset=asset)
        # threshold = friction * NORMAL_MULTIPLIER
        friction = r.threshold_bps / AlphaThresholdCalculator.NORMAL_MULTIPLIER
        assert friction == pytest.approx(2 * per_leg, abs=0.01)

    def test_reason_names_the_arithmetic(self):
        """A rejection has to be legible: how many legs, at what cost each."""
        r = _gate(_calc(), 1.0, asset="BTC")
        assert r.gate_decision == "REJECT_EV"
        assert "2x8.0bps/leg" in r.reason

    def test_typical_live_signal_is_now_rejected(self):
        """The regression this fix exists for.

        market_data_pipeline sets signal_edge_bps = |quant_dir| * 65 and its own
        comment puts avg |quant_dir| at ~0.3 -> 19.5bps raw, 14.6 after the
        0.75 ALPHA-FEEDBACK haircut. That cleared BTC's old 8.8bps bar while
        being a guaranteed loser against a 16bps round trip.
        """
        calc = _calc()
        for asset in ("BTC", "ETH", "SOL"):
            r = _gate(calc, 0.3 * 65, asset=asset)
            assert r.passes_threshold is False, asset
            assert r.gate_decision == "REJECT_EV", asset

    def test_a_genuinely_large_edge_still_passes(self):
        """Tightening a gate is only useful if it is not a wall."""
        r = _gate(_calc(), 200.0, asset="BTC")
        assert r.passes_threshold is True
        assert r.gate_decision == "ALLOW"


class TestOnlyTightens:
    """No input the old gate rejected may be accepted by the new one.

    Friction is non-negative, so 2 legs >= 1 leg pointwise and the threshold
    can only rise. Verified by construction across the live parameter grid
    rather than asserted in prose.
    """

    @pytest.mark.parametrize("asset", ["BTC", "ETH", "SOL"])
    @pytest.mark.parametrize("mode", ["NORMAL", "OPPORTUNITY"])
    @pytest.mark.parametrize("alpha", [0.0, 5.0, 14.6, 50.0, 200.0])
    @pytest.mark.parametrize("fee", [0.0, 3.0, 26.0])   # free tier / Coinbase / Kraken
    def test_threshold_never_falls(self, asset, mode, alpha, fee):
        os.environ["HMATS_ROUND_TRIP_FRICTION"] = "0"
        try:
            old = _gate(_calc(fee), alpha, asset=asset, mode=mode)
        finally:
            os.environ.pop("HMATS_ROUND_TRIP_FRICTION", None)
        new = _gate(_calc(fee), alpha, asset=asset, mode=mode)

        assert old.friction_legs == 1.0
        assert new.friction_legs == 2.0
        assert new.threshold_bps >= old.threshold_bps - 1e-9
        # The implication that matters: pass under the new gate => pass under old.
        if new.passes_threshold:
            assert old.passes_threshold


class TestKillSwitch:
    def test_env_var_restores_one_leg_arithmetic(self):
        """`HMATS_ROUND_TRIP_FRICTION=0` is an operator escape hatch.

        Default is ON because the safe direction is to charge more, not less.
        The flag is read at construction, so flipping it mid-process does not
        retroactively change a live calculator — assert that too, so nobody
        mistakes it for a hot toggle.
        """
        os.environ["HMATS_ROUND_TRIP_FRICTION"] = "0"
        try:
            calc = _calc()
        finally:
            os.environ.pop("HMATS_ROUND_TRIP_FRICTION", None)

        r = _gate(calc, 100.0, asset="BTC")
        assert r.friction_legs == 1.0
        friction = r.threshold_bps / AlphaThresholdCalculator.NORMAL_MULTIPLIER
        assert friction == pytest.approx(8.0, abs=0.01)

        # Already constructed: unaffected by the env var being gone.
        assert _gate(calc, 100.0, asset="BTC").friction_legs == 1.0

    def test_default_is_on(self):
        assert "HMATS_ROUND_TRIP_FRICTION" not in os.environ
        assert _gate(_calc(), 100.0).friction_legs == 2.0

    @pytest.mark.parametrize("value", ["1", "true", "yes", ""])
    def test_only_the_literal_zero_disables_it(self, value):
        """Fail-safe parsing: anything that is not "0" leaves the gate tight."""
        os.environ["HMATS_ROUND_TRIP_FRICTION"] = value
        try:
            calc = _calc()
        finally:
            os.environ.pop("HMATS_ROUND_TRIP_FRICTION", None)
        assert _gate(calc, 100.0).friction_legs == 2.0
