"""[P291] Venue-true HOLD cost for Coinbase-routed assets.

THE FINDING (verified at source before any code was written):
the alpha gate charged routed CDE assets a KRAKEN SPOT-MARGIN BORROW
schedule the venue does not levy.

  * `update_for_leverage` applies MARGIN_FEE_BPS {BTC:1, ETH:2, SOL:2} as
    BOTH an opening fee and a per-4h rollover whenever leverage > 1.
  * The live profile's `regime_leverage` overlay sets the two dominant
    regimes (WEAK_CONSOLIDATION / QUIET_ACCUMULATION, ~93% of live ticks
    per P198/P267) to 2.0 — so the leverage>1 branch fires essentially
    always.
  * `_margin_cost_bps` = opening + rollover x expected_hold_periods_4h(6)
    => BTC 1 + 1x6 = 7.0 ; ETH/SOL 2 + 2x6 = 14.0 — an EXACT match to the
    live gate lines ("margin=7", "14.0bps hold") on all three assets.
  * CDE levies neither: margin there is posted COLLATERAL (P275 probed
    INTRADAY_MARGIN_SETTING_STANDARD), and the venue's real carry is
    FUNDING, which P277 already wires in.
  * NOT additive double-counting — `update_funding_rate` takes a max(),
    not a sum. It is worse in a quieter way: the max() picks the Kraken
    rollover (1.0-2.0) over live funding (~0.04-1.9 bps/4h) every time,
    so P277's venue funding was STRUCTURALLY INERT on the hold path.

Pinned below in both directions, and the fail directions are the point:
every way this can go wrong falls back to the Kraken (more expensive)
charge, never to a free hold.
"""
import pytest

from defense.constitution import FrictionComponents


# Live funding readings (P218 measured, Coinbase side, per-8h ratio).
LIVE_FUNDING_8H = {"BTC": 0.000040, "ETH": 0.000008, "SOL": 0.000168}

# The gate's real composition (defense/constitution.py check_alpha_gate):
#   friction  = ROUND_TRIP_LEGS(2) x per_leg_bps + _margin_cost_bps
#   threshold = friction x NORMAL_MULTIPLIER(1.1) x smart-beta gate mult
EV_MULT = 1.1
SMART_BETA_MULT = 1.1435          # live observed (P230)
TAKER_FEE_BPS = 3.0               # CDE taker (P165 venue-aware fees)
ALPHA_CEILING_BPS = 40.0 * 1.0 * 0.75   # trend base_edge x max|sig| x feedback


def _threshold(f: FrictionComponents) -> float:
    return (2.0 * f.per_leg_bps(False) + f._margin_cost_bps) * EV_MULT * SMART_BETA_MULT


def _kraken_hold_fc(asset: str) -> FrictionComponents:
    """Today's live shape: CDE spreads (P289) + Kraken margin hold."""
    f = FrictionComponents()
    f.taker_fee_bps = TAKER_FEE_BPS
    f.set_spread_venue(asset, "coinbase")
    f.update_for_asset(asset)
    f.update_for_leverage(asset, 2.0)          # the live regime overlay
    f.update_funding_rate(LIVE_FUNDING_8H[asset])   # untagged => today's path
    return f


def _venue_true_fc(asset: str) -> FrictionComponents:
    """Venue-true: same, with the P291 gate ON and funding tagged."""
    f = FrictionComponents()
    f.venue_true_hold_enabled = True
    f.taker_fee_bps = TAKER_FEE_BPS
    f.set_spread_venue(asset, "coinbase")
    f.update_for_leverage(asset, 2.0)
    f.update_funding_rate(LIVE_FUNDING_8H[asset], asset=asset)
    f.update_for_asset(asset)
    return f


class TestTheFindingItself:
    """Pin the measured composition of the live hold cost, so that if
    someone later 'simplifies' MARGIN_FEE_BPS or the hold periods, the
    entry's evidence trail fails loudly instead of rotting."""

    @pytest.mark.parametrize("asset,expected", [("BTC", 7.0), ("ETH", 14.0), ("SOL", 14.0)])
    def test_live_hold_is_the_kraken_margin_schedule(self, asset, expected):
        f = _kraken_hold_fc(asset)
        assert f._margin_cost_bps == pytest.approx(expected), (
            "the live 'margin=7'/'14.0bps hold' gate lines are reproduced by "
            "the Kraken margin schedule, not by funding")

    def test_funding_is_structurally_inert_on_the_kraken_path(self):
        # The max() suppression: live funding never exceeds the Kraken
        # rollover, so P277's venue funding changed the hold by exactly 0.
        for asset in ("BTC", "ETH", "SOL"):
            with_funding = _kraken_hold_fc(asset)._margin_cost_bps
            f = FrictionComponents()
            f.taker_fee_bps = TAKER_FEE_BPS
            f.set_spread_venue(asset, "coinbase")
            f.update_for_asset(asset)
            f.update_for_leverage(asset, 2.0)   # no funding call at all
            assert with_funding == pytest.approx(f._margin_cost_bps), asset


class TestVenueTrueHold:
    def test_venue_true_charges_funding_only(self):
        for asset in ("BTC", "ETH", "SOL"):
            f = _venue_true_fc(asset)
            expected = LIVE_FUNDING_8H[asset] * 10000.0 * 0.5 * f.expected_hold_periods_4h
            assert f._margin_cost_bps == pytest.approx(expected), asset
            # and it is strictly cheaper than the Kraken charge it replaces
            assert f._margin_cost_bps < _kraken_hold_fc(asset)._margin_cost_bps

    def test_non_routed_asset_is_byte_identical_to_today(self):
        # No venue memory => Kraken schedule, funding tag irrelevant.
        f = FrictionComponents()
        f.taker_fee_bps = TAKER_FEE_BPS
        f.update_for_asset("ETH")
        f.update_for_leverage("ETH", 2.0)
        f.update_funding_rate(LIVE_FUNDING_8H["ETH"], asset="ETH")
        assert f._margin_cost_bps == pytest.approx(14.0)


class TestIndependentGate:
    """The hold correction must NOT ride P289's spread flag. Spreads are
    already live; this one OPENS two assets (see the arithmetic class
    below), so it ships default-OFF and is enabled by an explicit operator
    decision (P141) with the P237 tripwire question attached."""

    def test_default_is_off(self):
        assert FrictionComponents().venue_true_hold_enabled is False

    def test_off_reproduces_today_even_when_everything_else_is_armed(self):
        # Venue memory set, funding tagged, leverage applied — the ONLY
        # thing missing is the flag, and the charge must be the Kraken one.
        for asset, expected in (("BTC", 7.0), ("ETH", 14.0), ("SOL", 14.0)):
            f = FrictionComponents()
            f.taker_fee_bps = TAKER_FEE_BPS
            f.set_spread_venue(asset, "coinbase")
            f.update_for_leverage(asset, 2.0)
            f.update_funding_rate(LIVE_FUNDING_8H[asset], asset=asset)
            f.update_for_asset(asset)
            assert f.venue_true_hold_enabled is False
            assert f._margin_cost_bps == pytest.approx(expected), asset

    def test_spread_pricing_is_unaffected_by_the_hold_flag(self):
        # P289 spreads must keep working with the hold flag off — the two
        # corrections share a venue memory but not a decision.
        f = FrictionComponents()
        f.set_spread_venue("SOL", "coinbase")
        f.update_for_asset("SOL")
        assert f.venue_true_hold_enabled is False
        assert f.slippage_bps == 4.0          # CDE table, not Kraken's 10.0


class TestFailDirections:
    """Every failure mode must OVERCHARGE (fall back to Kraken), never
    produce a free hold. These are the tests that make the feature safe
    to ship default-off and safer still to ship on."""

    def test_unknown_venue_keeps_the_kraken_charge(self):
        f = FrictionComponents()
        f.venue_true_hold_enabled = True
        f.taker_fee_bps = TAKER_FEE_BPS
        f.set_spread_venue("SOL", "kraken")       # not 'coinbase'
        f.update_for_asset("SOL")
        f.update_for_leverage("SOL", 2.0)
        f.update_funding_rate(LIVE_FUNDING_8H["SOL"], asset="SOL")
        assert f._margin_cost_bps == pytest.approx(14.0)

    def test_funding_never_read_falls_back_not_free(self):
        # main.py only calls update_funding_rate when |funding| > 1e-8, so
        # "no funding this tick" is a REAL and common state. It must not
        # read as a zero hold.
        f = FrictionComponents()
        f.venue_true_hold_enabled = True
        f.taker_fee_bps = TAKER_FEE_BPS
        f.set_spread_venue("SOL", "coinbase")
        f.update_for_asset("SOL")
        f.update_for_leverage("SOL", 2.0)
        assert f._funding_bps_per_4h is None      # the never-read sentinel
        assert f._margin_cost_bps == pytest.approx(14.0)

    def test_untagged_funding_falls_back_so_unwired_callers_overcharge(self):
        # If main.py is never wired to pass asset=, the feature is INERT
        # rather than wrong — forgetting costs money, it does not risk it.
        f = FrictionComponents()
        f.venue_true_hold_enabled = True
        f.taker_fee_bps = TAKER_FEE_BPS
        f.set_spread_venue("SOL", "coinbase")
        f.update_for_asset("SOL")
        f.update_for_leverage("SOL", 2.0)
        f.update_funding_rate(LIVE_FUNDING_8H["SOL"])   # no asset kwarg
        assert f._margin_cost_bps == pytest.approx(14.0)

    def test_cross_asset_funding_leak_is_refused(self):
        # FRICTION is ONE shared object walked per asset each tick (P225
        # class). A funding reading taken for BTC must never price SOL.
        f = FrictionComponents()
        f.venue_true_hold_enabled = True
        f.taker_fee_bps = TAKER_FEE_BPS
        f.set_spread_venue("SOL", "coinbase")
        f.update_for_leverage("SOL", 2.0)
        f.update_funding_rate(LIVE_FUNDING_8H["BTC"], asset="BTC")
        f.update_for_asset("SOL")
        assert f._margin_cost_bps == pytest.approx(14.0)

    def test_negative_funding_is_never_a_subsidy(self):
        f = FrictionComponents()
        f.venue_true_hold_enabled = True
        f.taker_fee_bps = TAKER_FEE_BPS
        f.set_spread_venue("SOL", "coinbase")
        f.update_for_leverage("SOL", 2.0)
        f.update_funding_rate(-0.000500, asset="SOL")   # book would RECEIVE
        f.update_for_asset("SOL")
        assert f._margin_cost_bps >= 0.0
        # charged as a cost (|rate|), not credited
        assert f._margin_cost_bps == pytest.approx(0.000500 * 10000 * 0.5 * 6)

    def test_sentinel_check_is_reached_when_the_asset_guard_cannot_help(self):
        # WHITE-BOX, deliberately. In today's call graph the asset guard
        # short-circuits first (_funding_asset is only ever set alongside
        # _funding_bps_per_4h), so the `is not None` clause is
        # defence-in-depth for a FUTURE caller that sets one without the
        # other. Simulating that state is the only way to pin it — a
        # falsification probe on the clause alone stays green otherwise.
        f = FrictionComponents()
        f.venue_true_hold_enabled = True
        f.taker_fee_bps = TAKER_FEE_BPS
        f.set_spread_venue("SOL", "coinbase")
        f.update_for_leverage("SOL", 2.0)
        f.update_for_asset("SOL")
        f._funding_asset = "SOL"          # matches, but no reading exists
        assert f._funding_bps_per_4h is None
        assert f._margin_cost_bps == pytest.approx(14.0)   # not a TypeError, not free

    def test_clamp_is_reached_when_abs_is_bypassed(self):
        # WHITE-BOX for the same reason: update_funding_rate's abs() means
        # the stored value is never negative today, so the max(0.0, ...)
        # floor is defence-in-depth against a future change that credits
        # negative funding. Pin it by writing the field directly.
        f = FrictionComponents()
        f.venue_true_hold_enabled = True
        f.taker_fee_bps = TAKER_FEE_BPS
        f.set_spread_venue("SOL", "coinbase")
        f.update_for_leverage("SOL", 2.0)
        f.update_funding_rate(0.000100, asset="SOL")
        f.update_for_asset("SOL")
        f._funding_bps_per_4h = -5.0      # a subsidy, if unclamped
        assert f._margin_cost_bps == 0.0

    def test_genuine_zero_funding_is_distinguishable_from_never_read(self):
        f = FrictionComponents()
        f.venue_true_hold_enabled = True
        f.taker_fee_bps = TAKER_FEE_BPS
        f.set_spread_venue("SOL", "coinbase")
        f.update_for_leverage("SOL", 2.0)
        f.update_funding_rate(0.0, asset="SOL")     # a real zero reading
        f.update_for_asset("SOL")
        assert f._funding_bps_per_4h == 0.0
        assert f._margin_cost_bps == 0.0


class TestThresholdArithmeticBothDirections:
    """THE LOAD-BEARING PIN.

    Venue-true hold + the P289 CDE spreads move ETH and SOL BELOW the
    30bps asserted-alpha ceiling — i.e. this change WOULD OPEN both assets
    at full conviction, where today they are arithmetically locked out at
    any signal strength.

    That is precisely why the parent ships this flag OFF in the live
    profile pending an explicit operator decision: it is a venue-truth
    correction whose EFFECT is to widen trading of a signal whose measured
    alpha slope is ~0 (the P237 tripwire question, dated 2026-09-01). If
    someone later reads this change as cosmetic, this test says otherwise.
    """

    def test_today_eth_and_sol_are_locked_out(self):
        for asset in ("ETH", "SOL"):
            assert _threshold(_kraken_hold_fc(asset)) > ALPHA_CEILING_BPS, asset

    def test_venue_true_would_open_eth_and_sol(self):
        for asset in ("ETH", "SOL"):
            t = _threshold(_venue_true_fc(asset))
            assert t < ALPHA_CEILING_BPS, (
                f"{asset}: venue-true threshold {t:.2f} — if this ever rises "
                f"back above {ALPHA_CEILING_BPS:.1f} the entry's headline "
                f"claim is stale, fix the entry not the test")

    def test_the_measured_before_after_table(self):
        # Recorded numbers (bps), so a later constant edit that changes the
        # story fails here instead of silently rewriting P291's evidence.
        expected = {
            "BTC": (26.41, 19.12),
            "ETH": (44.02, 26.72),
            "SOL": (40.25, 28.98),
        }
        for asset, (now, vt) in expected.items():
            assert _threshold(_kraken_hold_fc(asset)) == pytest.approx(now, abs=0.05), asset
            assert _threshold(_venue_true_fc(asset)) == pytest.approx(vt, abs=0.05), asset

    def test_sol_is_the_marginal_case_and_funding_sensitive(self):
        # SOL clears the ceiling by ~1bps at the measured funding; a funding
        # spike re-closes it. Recorded so nobody reads "SOL opens" as
        # unconditional.
        f = FrictionComponents()
        f.venue_true_hold_enabled = True
        f.taker_fee_bps = TAKER_FEE_BPS
        f.set_spread_venue("SOL", "coinbase")
        f.update_for_leverage("SOL", 2.0)
        f.update_funding_rate(0.000400, asset="SOL")   # elevated funding
        f.update_for_asset("SOL")
        assert _threshold(f) > ALPHA_CEILING_BPS
