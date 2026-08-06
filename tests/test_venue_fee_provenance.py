"""[P169] Venue fee provenance: measured beats modelled, and the two are never
conflated.

Before this fix the venue *told us* what it charged — `OrderResult.fee` is parsed
straight out of the ccxt order response at `execution/execution_manager.py:1462` —
and the number was dropped on the floor. Every fee in `data/trade_attribution.jsonl`
is therefore modelled, none measured. The fingerprint of that is visible in the
data: over the 52 closed trades, 32/52 entry legs recorded *exactly* 16.0bps and
59/90 exit legs recorded *exactly* 16.0bps. Real fills do not land on a round
number two thirds of the time; a constant does.

16.0bps is Kraken's maker rate, and it was applied because:
  1. `fee_std` was hardcoded to Kraken's 0.0016/0.0026 regardless of venue, and
  2. `is_maker` was `order_type == "LIMIT"` — the ORDER type, not the FILL type.

(2) is the same conflation P166 found in `review_aggregator.maker_fee_ratio`
(`n_limit / n_classified`). A limit order that crosses the spread is filled as a
taker and charged as one. We cannot observe which happened from the order type,
so the derived flag is now carried as `is_maker_assumed=True` rather than passed
off as fact.

These tests pin the resolution rules. The load-bearing one is the *negative*
direction: a modelled number must never be labelled `fee_source="venue"`. If that
ever regresses, the fee column silently stops being an observation while still
looking like one — which is precisely the failure this entry exists to end.
"""

import math

import pytest

from core.paper_fee_service import (
    USD_EQUIVALENT_CURRENCIES,
    VENUE_FEE_STD,
    build_execution_fee_result,
    resolve_trade_fee_usd,
    venue_fee_std,
)

NOTIONAL = 10_000.0


def _resolve(venue_fee_usd, currency="USD", modelled=16.0, notional=NOTIONAL):
    return resolve_trade_fee_usd(
        executed_notional_usd=notional,
        venue_fee_usd=venue_fee_usd,
        venue_fee_currency=currency,
        modelled_fee_usd=modelled,
    )


def _build(**kw):
    params = dict(
        asset="BTC",
        executed_notional_usd=NOTIONAL,
        order_type="MARKET",
        execution_direction=1.0,
        regime_leverage=1.0,
        existing_position=None,
        fee_blending_enabled=True,
        default_margin_opening_fee_bps=1.0,
        margin_fee_map={},
        fee_record_fn=None,
    )
    params.update(kw)
    return build_execution_fee_result(**params)


class TestVenueScheduleSelection:
    """`fee_std` used to be hardcoded Kraken. It must follow the venue."""

    def test_kraken_rates(self):
        assert venue_fee_std("kraken", is_maker=True) == 0.0016
        assert venue_fee_std("kraken", is_maker=False) == 0.0026

    def test_coinbase_rates(self):
        assert venue_fee_std("coinbase", is_maker=True) == 0.0000
        assert venue_fee_std("coinbase", is_maker=False) == 0.0003

    def test_coinbase_taker_is_far_cheaper_than_kraken(self):
        # The whole reason the hardcode mattered: post-cutover the model was
        # pricing the wrong exchange by nearly an order of magnitude.
        assert venue_fee_std("kraken", is_maker=False) > (
            8 * venue_fee_std("coinbase", is_maker=False)
        )

    @pytest.mark.parametrize("venue", ["KRAKEN", "Kraken", "  kraken  ", "kRaKeN"])
    def test_venue_match_is_case_and_whitespace_insensitive(self, venue):
        assert venue_fee_std(venue, is_maker=False) == 0.0026

    @pytest.mark.parametrize("venue", ["binance", "", None, "coinbse", "unknown"])
    def test_unknown_venue_falls_back_to_the_expensive_one(self, venue):
        # Over-charging fails toward not trading. Under-charging fails toward
        # trading a signal that does not clear its real costs.
        assert venue_fee_std(venue, is_maker=False) == VENUE_FEE_STD["kraken"]["taker"]
        assert venue_fee_std(venue, is_maker=True) == VENUE_FEE_STD["kraken"]["maker"]

    def test_fallback_venue_is_never_the_cheap_one(self):
        for venue in ("binance", "", None, "typo"):
            assert venue_fee_std(venue, is_maker=False) >= max(
                s["taker"] for s in VENUE_FEE_STD.values()
            )


class TestVenueFeeWins:
    """When the venue reports a usable fee, it is the answer."""

    def test_venue_fee_overrides_model(self):
        out = _resolve(3.10, modelled=16.0)
        assert out["trade_fee_usd"] == 3.10
        assert out["fee_source"] == "venue"
        assert out["venue_fee_usd"] == 3.10

    def test_venue_fee_wins_even_when_larger_than_model(self):
        # Not a "pick the cheaper one" rule — it is a "prefer the observation" rule.
        out = _resolve(40.0, modelled=16.0)
        assert out["trade_fee_usd"] == 40.0
        assert out["fee_source"] == "venue"

    @pytest.mark.parametrize("currency", sorted(USD_EQUIVALENT_CURRENCIES))
    def test_usd_equivalents_are_accepted(self, currency):
        out = _resolve(3.10, currency=currency)
        assert out["fee_source"] == "venue"

    @pytest.mark.parametrize("currency", ["usd", " Usd ", "usdc"])
    def test_currency_match_is_case_and_whitespace_insensitive(self, currency):
        assert _resolve(3.10, currency=currency)["fee_source"] == "venue"

    def test_no_reason_is_recorded_when_the_venue_number_is_used(self):
        assert _resolve(3.10)["fee_source_reason"] == ""

    def test_venue_fee_flows_through_build(self):
        r = _build(venue="coinbase", venue_fee_usd=3.10, venue_fee_currency="USD")
        assert r["fee_source"] == "venue"
        assert r["trade_fee_usd"] == 3.10
        assert r["total_fee_usd"] == 3.10
        assert r["fee_usd"] == 3.10

    def test_fee_effective_reflects_what_was_actually_paid(self):
        # Not the schedule rate — the realised one.
        r = _build(venue="coinbase", venue_fee_usd=3.10, venue_fee_currency="USD")
        assert r["fee_effective"] == pytest.approx(3.10 / NOTIONAL)

    def test_modelled_fee_is_retained_alongside_the_measured_one(self):
        # Keeping both is what makes model error measurable after the fact.
        r = _build(venue="coinbase", venue_fee_usd=3.10, venue_fee_currency="USD")
        assert r["modelled_fee_usd"] == pytest.approx(NOTIONAL * 0.0003)
        assert r["trade_fee_usd"] == 3.10
        assert r["modelled_fee_usd"] != r["trade_fee_usd"]


class TestFallsBackToModel:
    """Every rejection path must keep the model AND say why."""

    @pytest.mark.parametrize(
        "bad,fragment",
        [
            (None, "no fee"),
            ("abc", "not numeric"),
            (float("nan"), "not finite"),
            (float("inf"), "not finite"),
            (float("-inf"), "not finite"),
            (-1.0, "negative"),
        ],
    )
    def test_unusable_values_fall_back(self, bad, fragment):
        out = _resolve(bad, modelled=16.0)
        assert out["fee_source"] == "model"
        assert out["trade_fee_usd"] == 16.0
        assert fragment in out["fee_source_reason"]

    def test_every_fallback_states_a_reason(self):
        for bad in (None, "abc", float("nan"), -1.0, 0.0):
            out = _resolve(bad)
            assert out["fee_source"] == "model"
            assert out["fee_source_reason"], f"silent fallback for {bad!r}"

    def test_rejected_value_is_not_smuggled_into_venue_fee_usd(self):
        # If it was not good enough to use, it is not good enough to record as
        # the venue's number either.
        for bad in ("abc", float("nan"), -1.0, 0.0):
            assert _resolve(bad)["venue_fee_usd"] is None

    def test_nan_never_reaches_the_output(self):
        out = _resolve(float("nan"))
        assert not math.isnan(out["trade_fee_usd"])


class TestNonUsdCurrency:
    """0.0031 in ETH is not 0.0031 in USD."""

    @pytest.mark.parametrize("currency", ["ETH", "BTC", "SOL", "EUR", "XBT", "JPY"])
    def test_non_usd_falls_back(self, currency):
        out = _resolve(0.0031, currency=currency)
        assert out["fee_source"] == "model"
        assert currency in out["fee_source_reason"]

    def test_no_conversion_is_attempted(self):
        # A wrong FX/spot rate is worse than a modelled fee, because the result
        # carries the authority of an observation while being fiction.
        out = _resolve(0.0031, currency="ETH", modelled=16.0)
        assert out["trade_fee_usd"] == 16.0
        assert out["venue_fee_usd"] is None

    def test_currency_is_reported_even_when_rejected(self):
        # The operator needs to see *what* denomination showed up.
        assert _resolve(0.0031, currency="ETH")["venue_fee_currency"] == "ETH"


class TestZeroIsAmbiguous:
    """0.0 is indistinguishable from absent, so it cannot be trusted."""

    def test_zero_falls_back_to_model(self):
        out = _resolve(0.0, modelled=16.0)
        assert out["fee_source"] == "model"
        assert out["trade_fee_usd"] == 16.0

    def test_reason_names_the_ambiguity(self):
        assert "indistinguishable" in _resolve(0.0)["fee_source_reason"]

    def test_a_free_trade_is_never_booked_from_a_zero(self):
        # `OrderResult.fee` is
        #   float((order_status.get('fee') or {}).get('cost', 0) or 0)
        # which collapses "missing" and "genuinely zero" into the same 0.0.
        # Booking a free round trip off that is how a losing strategy looks
        # profitable.
        r = _build(venue="kraken", venue_fee_usd=0.0, venue_fee_currency="USD")
        assert r["trade_fee_usd"] > 0.0
        assert r["fee_source"] == "model"


class TestSourceIsNeverMislabelled:
    """The load-bearing invariant."""

    @pytest.mark.parametrize(
        "venue_fee,currency",
        [
            (None, ""),
            (0.0, "USD"),
            (-5.0, "USD"),
            (float("nan"), "USD"),
            (0.0031, "ETH"),
            ("abc", "USD"),
        ],
    )
    def test_model_is_never_labelled_venue(self, venue_fee, currency):
        out = _resolve(venue_fee, currency=currency)
        assert out["fee_source"] != "venue"

    def test_source_is_always_one_of_the_known_values(self):
        for venue_fee in (None, 0.0, -5.0, 3.10, "abc", float("inf")):
            assert _resolve(venue_fee)["fee_source"] in ("venue", "model")

    def test_build_source_is_always_one_of_the_known_values(self):
        for enabled in (True, False):
            for venue_fee in (None, 0.0, 3.10):
                r = _build(fee_blending_enabled=enabled, venue_fee_usd=venue_fee,
                           venue_fee_currency="USD")
                assert r["fee_source"] in ("venue", "model", "disabled")

    def test_venue_source_implies_a_recorded_venue_number(self):
        for venue_fee in (0.01, 3.10, 999.0):
            out = _resolve(venue_fee)
            if out["fee_source"] == "venue":
                assert out["venue_fee_usd"] == out["trade_fee_usd"]


class TestDisabledIsNotModelled:
    """$0.00-because-disabled must be readable apart from $0.00-because-free."""

    def test_disabled_has_its_own_source(self):
        r = _build(fee_blending_enabled=False)
        assert r["fee_source"] == "disabled"
        assert r["fee_source"] != "model"
        assert r["trade_fee_usd"] == 0.0

    def test_disabled_states_a_reason(self):
        assert _build(fee_blending_enabled=False)["fee_source_reason"]

    def test_disabled_ignores_a_reported_venue_fee(self):
        # Fee modelling off means fees are off; a stray venue number must not
        # sneak a charge back in through the side door.
        r = _build(fee_blending_enabled=False, venue_fee_usd=3.10,
                   venue_fee_currency="USD")
        assert r["trade_fee_usd"] == 0.0
        assert r["fee_source"] == "disabled"

    def test_disabled_still_reports_the_venue_and_schedule(self):
        r = _build(fee_blending_enabled=False, venue="coinbase")
        assert r["venue"] == "coinbase"
        assert r["fee_std"] == 0.0003  # MARKET -> taker


class TestMakerIsAnAssumption:
    """`order_type == "LIMIT"` is not an observation of a maker fill."""

    @pytest.mark.parametrize("enabled", [True, False])
    @pytest.mark.parametrize("order_type", ["LIMIT", "MARKET", "", "limit"])
    def test_is_maker_assumed_is_always_true(self, enabled, order_type):
        # There is no path on which the fill type is actually observed, so the
        # flag is unconditionally an assumption. If a real maker/taker signal is
        # ever wired in, this test is the thing that should fail.
        r = _build(fee_blending_enabled=enabled, order_type=order_type)
        assert r["is_maker_assumed"] is True

    def test_limit_order_still_drives_the_maker_rate(self):
        # The assumption is retained as the model's best guess; it is only
        # relabelled, not silently changed.
        assert _build(order_type="LIMIT", venue="kraken")["fee_std"] == 0.0016
        assert _build(order_type="MARKET", venue="kraken")["fee_std"] == 0.0026

    def test_order_type_matching_is_case_insensitive(self):
        assert _build(order_type="limit", venue="kraken")["is_maker"] is True

    def test_a_measured_fee_can_contradict_the_maker_assumption(self):
        # A LIMIT order that crossed and paid taker: the venue number must win
        # rather than being overruled by the order type.
        r = _build(order_type="LIMIT", venue="coinbase", venue_fee_usd=3.00,
                   venue_fee_currency="USD")
        assert r["is_maker"] is True          # what we assumed
        assert r["fee_std"] == 0.0            # what the assumption implied
        assert r["trade_fee_usd"] == 3.00     # what was actually charged
        assert r["fee_source"] == "venue"


class TestModelledFeeUsesTheRightVenue:
    """The regression that produced the 16.0bps fingerprint."""

    def test_coinbase_market_is_not_priced_at_kraken(self):
        r = _build(venue="coinbase", order_type="MARKET")
        assert r["trade_fee_usd"] == pytest.approx(NOTIONAL * 0.0003)
        assert r["trade_fee_usd"] != pytest.approx(NOTIONAL * 0.0026)

    def test_kraken_path_is_unchanged(self):
        # The legacy sleeve must price exactly as it did before P169.
        assert _build(venue="kraken", order_type="LIMIT")["trade_fee_usd"] == (
            pytest.approx(NOTIONAL * 0.0016)
        )
        assert _build(venue="kraken", order_type="MARKET")["trade_fee_usd"] == (
            pytest.approx(NOTIONAL * 0.0026)
        )

    def test_default_venue_is_kraken(self):
        # Callers that have not been threaded yet keep the old behaviour.
        assert _build(order_type="LIMIT")["trade_fee_usd"] == (
            _build(venue="kraken", order_type="LIMIT")["trade_fee_usd"]
        )

    def test_venue_is_recorded_on_the_result(self):
        assert _build(venue="Coinbase")["venue"] == "coinbase"

    def test_no_venue_prices_above_or_equal_to_the_kraken_model(self):
        # Nothing may become *more* expensive than the pre-P169 model, so this
        # fix cannot manufacture losses that were not already booked.
        for order_type in ("LIMIT", "MARKET"):
            base = _build(venue="kraken", order_type=order_type)["trade_fee_usd"]
            for venue in ("kraken", "coinbase", "unknown"):
                assert _build(venue=venue, order_type=order_type)["trade_fee_usd"] <= base


class TestNotionalEdges:
    def test_zero_notional_keeps_the_model_and_charges_nothing(self):
        r = _build(executed_notional_usd=0.0)
        assert r["trade_fee_usd"] == 0.0
        assert r["fee_effective"] in (0.0, 0.0003, 0.0026)

    def test_zero_venue_fee_on_zero_notional_is_not_ambiguous(self):
        # With no notional there is nothing to charge, so 0.0 is not suspicious.
        out = _resolve(0.0, notional=0.0, modelled=0.0)
        assert out["trade_fee_usd"] == 0.0

    def test_negative_notional_is_clamped(self):
        assert _build(executed_notional_usd=-500.0)["executed_notional_usd"] == 0.0


class TestOrderResultCarriesCurrency:
    """A fee without its denomination is not a fee."""

    def test_to_dict_emits_fee_currency(self):
        from execution.execution_manager import OrderResult

        d = OrderResult(success=True, fee=0.0031, fee_currency="ETH").to_dict()
        assert d["fee_currency"] == "ETH"
        assert d["fee"] == 0.0031

    def test_currency_survives_into_the_resolver(self):
        from execution.execution_manager import OrderResult

        d = OrderResult(success=True, fee=0.0031, fee_currency="ETH").to_dict()
        out = _resolve(d["fee"], currency=d["fee_currency"])
        assert out["fee_source"] == "model"  # rejected, not misread as $0.0031
